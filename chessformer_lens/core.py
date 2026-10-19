"""The small set of types and helpers a chess
transformer has to provide so that ChessformerEngine can run on it.

Everything model-specific for ChessformerEngine lives in a `ModelAdapter` 
subclass in adapters/ (e.g., maia3.py, lc0.py, searchless.py).

  ModelAdapter       the abstract base every model family subclasses. Supplies
                     tokens / forward, the vocab, the token layout, the named
                     hook modules, the policy head as an unembedding, each
                     layer's attention taken apart, each MLP's output projection
  Conditioning       the non-board inputs: a rating pair, the clock, anything
                     else. BT4 ignores the ratings, Maia-3 reads them
  HeadOutputs        one batched forward pass: policy logits, raw value, aux heads
  ValueEstimate      the value head read as an expected score, plus W/D/L
                     on models that have it
  Capabilities       what an architecture has (rating input? wdl or scalar
                     value? pre or post norm? which attention score terms? a
                     softmax policy or win probabilities per move?) so the
                     engine and the app can skip what isn't there
  ArchSpec           n_layers, n_heads, d_model, d_mlp, the names
  Token, TokenLayout what each token of the residual stream is (a square, a
                     move, a control token). `is_board` means 64 square tokens
                     in canonical order, so activations can be drawn on a board
  AttnDecomp         one layer's attention as q, k, v, the additive score terms,
                     and per-head output weights; pattern() and head_writes()
                     recompute what the model does from those
  PolicyVocab        the policy head's move list and index <-> chess.Move on a
                     real board. Lc0 subclasses it (castling as king-takes-rook)

  register_adapter   class decorator that makes an adapter resolvable by alias;
                     third-party packages can do the same through the
                     `chessformer_lens.adapters` entry-point group
  resolve_adapter    alias -> (adapter class, display name), no weights loaded
  available_models   every family's models, what the app's model picker lists
  format_catalog     the same as text, for error messages and --list

  canon_square       python-chess square -> index in the side-to-move frame
  real_square        and back
  mirror_uci         a uci move flipped vertically, for Black
  uci_vocabulary     the 64x64 + promotions move list in from*64 + to order
  sample_from_logits argmax / temperature / nucleus sampling over masked logits
  pick_device        explicit > $CHESSFORMER_DEVICE > cuda > mps > cpu

Frames: Maia-3 and lc0 both see the board from the side to move. Squares are
indexed rank*8 + file with the board flipped vertically when Black is to move,
so a Black move is looked up mirrored (`mirror_uci`) and comes back
un-mirrored. That is the "stm_mirror" frame. A model in the "absolute" frame
uses python-chess squares as they are.

engine.py imports from here, the adapters subclass ModelAdapter and PolicyVocab,
and bridge.py calls the to_json methods to send layouts and capabilities to
the app's page.
"""
from __future__ import annotations

import dataclasses
import math
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping, Sequence

import chess
import torch

__all__ = ["Conditioning", "ValueEstimate", "Token", "TokenLayout", "Capabilities",
           "ArchSpec", "AttnDecomp", "HeadOutputs", "PolicyVocab", "ModelAdapter",
           "register_adapter", "registered_adapters", "resolve_adapter", "format_catalog",
           "available_models",
           "canon_square", "real_square", "mirror_uci", "sample_from_logits", "pick_device",
           "uci_vocabulary"]


# ----- device -------------------------------------------------------------------

def pick_device(explicit: str | None = None) -> str:
    """Resolve a torch device: explicit argument > $CHESSFORMER_DEVICE > cuda >
    mps > cpu. $MAIA3_DEVICE, the pre-1.1 name, still works.

    Apple-silicon MPS agrees with CPU to ~1e-5 on both engines (policy, head
    ablation grid, carrier neurons). It nearly halves BT4's 495-pass head
    sweep and is a wash on the 5M model, whose single forward is ~6 ms on
    CPU. Set CHESSFORMER_DEVICE=cpu to opt out."""
    if explicit:
        return explicit
    env = os.environ.get("CHESSFORMER_DEVICE") or os.environ.get("MAIA3_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----- frames ---------------------------------------------------------------------

def canon_square(square: int, turn: bool) -> int:
    """python-chess square -> canonical index in the side-to-move frame
    (rank*8 + file): identity for White, vertically mirrored for Black."""
    rank, file = chess.square_rank(square), chess.square_file(square)
    return (rank if turn == chess.WHITE else 7 - rank) * 8 + file


def real_square(canon: int, turn: bool) -> int:
    """Canonical index -> python-chess square. Inverse of `canon_square`."""
    rank, file = divmod(int(canon), 8)
    return chess.square(file, rank if turn == chess.WHITE else 7 - rank)


def mirror_uci(uci: str) -> str:
    """A uci move flipped vertically (rank r -> 9 - r), promotion suffix kept:
    the side-to-move frame's move for Black."""
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


# ----- sampling -------------------------------------------------------------------

def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """Sample an index from 1-D logits already masked to legal moves (-inf
    elsewhere). Temperature 0 is argmax; `top_p` < 1 is nucleus sampling, the
    most likely entry always kept."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        ranked, order = torch.sort(probs, descending=True)
        keep = torch.cumsum(ranked, dim=-1) <= top_p
        keep[0] = True
        nucleus = ranked[keep]
        pick = torch.multinomial(nucleus / nucleus.sum(), num_samples=1).item()
        return int(order[keep][pick].item())
    return int(torch.multinomial(probs, num_samples=1).item())


# ----- what goes in -----------------------------------------------------------------

@dataclass(frozen=True)
class Conditioning:
    """Everything besides the board that a model may be conditioned on.

      self_elo, oppo_elo   the ratings of the side to move and its opponent;
                           `oppo_elo` None means "same as self_elo"
      clock                (base, increment, clock left) in seconds, for
                           models with time-control inputs
      extras               anything model-specific, by name

    A model ignores what it has no input for (Lc0 BT4 ignores the ratings);
    `Capabilities.conditioning` says what it reads. Every engine method takes
    either one of these or the legacy `(self_elo, oppo_elo)` pair."""
    self_elo: int | None = None
    oppo_elo: int | None = None
    clock: tuple[float, float, float] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def opponent_elo(self) -> int | None:
        return self.self_elo if self.oppo_elo is None else self.oppo_elo

    @classmethod
    def coerce(cls, cond=None, oppo_elo=None) -> "Conditioning":
        """A Conditioning from what a caller passed as `self_elo` (an int, a
        Conditioning, a mapping of fields, or None) plus an optional `oppo_elo`,
        which overrides the opponent rating either way."""
        if isinstance(cond, Conditioning):
            out = cond
        elif isinstance(cond, Mapping):
            out = cls(**cond)
        else:
            out = cls(self_elo=None if cond is None else int(cond))
        if oppo_elo is not None:
            out = dataclasses.replace(out, oppo_elo=int(oppo_elo))
        return out

    def label(self) -> str:
        """Short human-readable form, e.g. '1500' or '1500 vs 1300'."""
        if self.self_elo is None:
            return "-"
        if self.oppo_elo is None or self.oppo_elo == self.self_elo:
            return str(self.self_elo)
        return f"{self.self_elo} vs {self.oppo_elo}"


# ----- what comes out -----------------------------------------------------------------

@dataclass(frozen=True)
class ValueEstimate:
    """A value head's verdict for the side to move, whatever its native form.

      expected   expected score in [0, 1] (win = 1, draw = 0.5): every value
                 head has one
      wdl        (win, draw, loss) probabilities, on heads that predict them"""
    expected: float
    wdl: tuple[float, float, float] | None = None

    @classmethod
    def from_wdl(cls, win: float, draw: float, loss: float) -> "ValueEstimate":
        return cls(expected=win + 0.5 * draw, wdl=(win, draw, loss))

    def legacy_wdl(self) -> dict | None:
        """The `evaluate()["wdl"]` dict, or None on a scalar-value model."""
        if self.wdl is None:
            return None
        win, draw, loss = self.wdl
        return {"win": win, "draw": draw, "loss": loss}

    def to_json(self) -> dict:
        return {"expected": self.expected, "wdl": self.legacy_wdl()}


@dataclass
class HeadOutputs:
    """One batched forward pass: raw policy logits (B, V) over the vocabulary,
    the raw value-head output (B, ...), and any auxiliary heads by name, each
    (B,) — e.g. "mlh" (moves left) or "ponder" (predicted think time)."""
    policy: torch.Tensor
    value: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


# ----- tokens -------------------------------------------------------------------------

TOKEN_KINDS = ("square", "move", "fen_char", "channel", "control", "spatial")


@dataclass(frozen=True)
class Token:
    """What one position of the residual stream stands for.

      kind     one of TOKEN_KINDS: "square" (one board square), "move" (one move
               of the game so far), "fen_char", "channel" (a whole-board feature
               map), "control" (rating, side to move, CLS, …) or "spatial"
      label    short display text
      square   for kind "square": the python-chess square on the real board
      move     for kind "move": the uci move on the real board"""
    kind: str
    label: str
    square: int | None = None
    move: str | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind, "label": self.label, "square": self.square, "move": self.move}


@dataclass(frozen=True)
class TokenLayout:
    """The token axis of every activation, for one position.

      tokens    one Token per stream position
      frame     "stm_mirror" (canonical side-to-move squares) or "absolute"
      readout   which token(s) the heads read: "all" or a token index

    `is_board` is the switch every frontend keys on: 64 square tokens in
    canonical order, so activations can be drawn on the board."""
    tokens: tuple[Token, ...]
    frame: str = "stm_mirror"
    readout: Any = "all"
    is_board: bool = False

    @classmethod
    def squares(cls, board: chess.Board, frame: str = "stm_mirror") -> "TokenLayout":
        """The square-token layout: token i is canonical square i."""
        turn = board.turn if frame == "stm_mirror" else chess.WHITE
        toks = []
        for i in range(64):
            sq = real_square(i, turn)
            toks.append(Token("square", chess.square_name(sq), square=sq))
        return cls(tuple(toks), frame=frame, readout="all", is_board=True)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def labels(self) -> list[str]:
        return [t.label for t in self.tokens]

    def to_json(self) -> dict:
        return {"frame": self.frame, "readout": self.readout, "is_board": self.is_board,
                "n_tokens": len(self.tokens), "tokens": [t.to_json() for t in self.tokens]}


# ----- architecture -------------------------------------------------------------------

@dataclass(frozen=True)
class Capabilities:
    """What an architecture has, so the engine and every frontend can degrade
    instead of failing.

      conditioning        the Conditioning fields it reads, e.g. {"elo"}
      value               "wdl" (win/draw/loss head) or "scalar"
      norm                "post" (x = norm(x + f(x))) or "pre" (x = x + f(norm(x)))
      attn_terms          the additive pre-softmax score terms its attention
                          decomposes into, "content" (scaled QKᵀ) first
      causal              attention is masked to earlier tokens
      needs_game_history  the input is the game's moves, not just the position
      aux_heads           names of HeadOutputs.aux entries
      policy              "softmax": the policy logits are softmaxed over the legal moves;
                          "win_prob": each legal move's logit is its win probability as
                          the model states it (action-value models), listed as it is"""
    conditioning: frozenset = frozenset()
    value: str = "wdl"
    norm: str = "post"
    attn_terms: tuple = ("content",)
    causal: bool = False
    needs_game_history: bool = False
    aux_heads: frozenset = frozenset()
    policy: str = "softmax"

    def to_json(self) -> dict:
        return {"conditioning": sorted(self.conditioning), "value": self.value,
                "norm": self.norm, "attn_terms": list(self.attn_terms), "causal": self.causal,
                "needs_game_history": self.needs_game_history,
                "aux_heads": sorted(self.aux_heads), "policy": self.policy}


@dataclass(frozen=True)
class ArchSpec:
    """Sizes and names. `d_mlp` is the MLP hidden width (the neuron count)."""
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    name: str
    display_name: str
    checkpoint: str | None = None

    def legacy_cfg(self, **extra):
        """The args-namespace older code reads as `engine.cfg`."""
        import types
        return types.SimpleNamespace(
            num_blocks=self.n_layers, num_heads=self.n_heads, dim_vit=self.d_model,
            mlp_ratio=self.d_mlp / self.d_model, gab_gen_size=None,
            checkpoint_path=self.checkpoint, trust_checkpoint=False, **extra)

    def legacy_spec(self, aliases=()):
        """The namespace older code reads as `engine.spec`."""
        import types
        return types.SimpleNamespace(name=self.name, display_name=self.display_name,
                                     repo_id=None, aliases=tuple(aliases),
                                     checkpoint_filename=None, config={})


@dataclass
class AttnDecomp:
    """One layer's attention, taken apart from its input.

      q, k, v   (B, H, T, dh)
      terms     the additive pre-softmax score terms by name, each (B, H, T, T),
                in `Capabilities.attn_terms` order — "content" is q kᵀ/√dh
      mask      bool, broadcastable to (B, H, T, T), True = may not attend; or None
      W_O       (H, dh, d): head h's block of the output projection
      b_O       (d,) output-projection bias, or None

    pattern() is the softmax the model runs; head_writes() each head's exact
    write to the residual stream, whose sum plus b_O is the attention output."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    terms: dict
    W_O: torch.Tensor
    b_O: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def scores(self, terms: Sequence[str] | None = None) -> torch.Tensor:
        """Sum of the named terms (all by default), masked with -inf."""
        names = list(self.terms) if terms is None else list(terms)
        s = self.terms[names[0]]
        for n in names[1:]:
            s = s + self.terms[n]
        if self.mask is not None:
            s = s.masked_fill(self.mask, float("-inf"))
        return s

    def pattern(self, terms: Sequence[str] | None = None) -> torch.Tensor:
        """softmax over keys of `scores(terms)`: (B, H, T, T)."""
        return torch.softmax(self.scores(terms), dim=-1)

    def head_writes(self) -> torch.Tensor:
        """(B, H, T, d): (A_h V_h) W_O^(h) for every head."""
        return (self.pattern() @ self.v) @ self.W_O


def split_heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, d) -> (B, H, T, d/H)."""
    B, T, d = t.shape
    return t.view(B, T, n_heads, d // n_heads).transpose(1, 2)


def content_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product logits q kᵀ / √dh: (B, H, T, T)."""
    return (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))


def out_proj_heads(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """An (d, d) nn.Linear output-projection weight as (H, dh, d) per-head blocks."""
    d = weight.size(0)
    return weight.view(d, n_heads, weight.size(1) // n_heads).permute(1, 2, 0)


# ----- moves --------------------------------------------------------------------------

def uci_vocabulary(knight_promotions: bool = True) -> list[str]:
    """The square-pair move vocabulary in the side-to-move frame: all 64x64
    (from, to) pairs in from*64 + to order, then every rank-7 -> rank-8
    promotion as from_file, to_file, piece (q r b [n])."""
    moves = [chess.square_name(f) + chess.square_name(t) for f in range(64) for t in range(64)]
    pieces = "qrbn" if knight_promotions else "qrb"
    moves += [f"{ff}7{tf}8{p}" for ff in "abcdefgh" for tf in "abcdefgh" for p in pieces]
    return moves


class PolicyVocab:
    """A policy head's move list and the two-way map between its indices and
    chess.Move on a real board. `moves` are uci strings in the model's frame;
    in "stm_mirror" a Black move is looked up mirrored.

    Subclass to encode moves differently (see adapters.lc0.Lc0Vocab: castling
    as king-takes-rook, knight promotions on the plain square pair)."""

    def __init__(self, moves: Sequence[str], frame: str = "stm_mirror"):
        self.moves = list(moves)
        self.index_of = {m: i for i, m in enumerate(self.moves)}
        self.frame = frame
        sq = [(chess.parse_square(m[:2]), chess.parse_square(m[2:4])) for m in self.moves]
        self._squares = sq

    def __len__(self) -> int:
        return len(self.moves)

    def _to_frame(self, board: chess.Board, uci: str) -> str:
        return mirror_uci(uci) if self.frame == "stm_mirror" and board.turn == chess.BLACK else uci

    def index(self, board: chess.Board, move: chess.Move) -> int:
        """Policy index of a move. KeyError if the vocabulary has no such move."""
        return self.index_of[self._to_frame(board, move.uci())]

    def decode(self, board: chess.Board, idx: int) -> chess.Move:
        """Policy index -> chess.Move on the real board, no legality check.
        ValueError on an undecodable uci."""
        return chess.Move.from_uci(self._to_frame(board, self.moves[int(idx)]))

    def squares(self, idx: int) -> tuple[int, int]:
        """(from, to) of an index in the model's frame (canonical squares)."""
        return self._squares[int(idx)]

    def legal_mask(self, board: chess.Board) -> torch.Tensor:
        """Boolean (n_moves,) mask of the legal moves; moves outside the
        vocabulary are skipped."""
        mask = torch.zeros(len(self.moves), dtype=torch.bool)
        for mv in board.legal_moves:
            try:
                mask[self.index(board, mv)] = True
            except KeyError:
                pass
        return mask


# ----- the adapter ----------------------------------------------------------------------

class ModelAdapter(ABC):
    """Everything the engine needs from one chess transformer. Subclass it,
    decorate with `@register_adapter`, and `load_engine(alias)` builds a
    `ChessformerEngine` on it; every method of the engine and every frontend
    then works, within what `caps` says the model has.

    Attributes a loaded adapter sets:
      name, caps, arch, vocab, model (an eval-mode nn.Module), device,
      cfg / spec (namespaces older code reads; `arch.legacy_cfg()` builds them)

    Hook names (keys of `hook_modules`), as the cache and every plot use them:
      embed_in      the stream entering layer 0
      attn_NN       layer NN's attention write (the module's output is the
                    vector added to the stream)
      postattn_NN   the running stream after the attention sub-layer
      mlp_NN        layer NN's MLP write (the output of `mlp_out(NN)`)
      block_NN      the running stream after the whole layer
      encoder_out   after the final norm the heads read through (optional)
      attn_in_NN    the input of layer NN's attention, where that is not
                    already one of the above (Pre-LN: the ln1 output)
    """

    name: str = ""
    family: str = ""
    caps: Capabilities = Capabilities()

    # --- registry (class level: must not load weights) ---
    @classmethod
    def resolve(cls, alias: str) -> str | None:
        """Display name if this adapter serves `alias`, else None."""
        return None

    @classmethod
    def models(cls) -> list[dict]:
        """The named models this family offers, for engine lists and the app's
        model picker: [{alias, display_name, aliases, available, note}].
        `available` is False when loading would fail as things stand (no
        weights file, a missing package); `note` says why, or what loading
        involves (a download)."""
        return []

    @classmethod
    def catalog(cls) -> list[str]:
        """Lines for `format_engine_list`: the family, then one per model."""
        models = cls.models()
        if not models:
            return []
        lines = [f"  {cls.family or cls.name}:"]
        for m in models:
            names = ", ".join([m["alias"], *m.get("aliases", [])])
            note = f"  ({m['note']})" if m.get("note") else ""
            lines.append(f"    {m['display_name']:<22}{names}{note}")
        return lines

    @classmethod
    @abstractmethod
    def load(cls, alias: str, *, device: str | None = None, **kwargs) -> "ModelAdapter":
        """Build the adapter with its weights loaded, in eval mode."""

    # --- inputs and outputs ---
    @abstractmethod
    def tokens(self, boards: Sequence[chess.Board]) -> torch.Tensor:
        """The positions as the model's board input, batched, on the device."""

    @abstractmethod
    def forward(self, boards: Sequence[chess.Board], cond: Conditioning) -> HeadOutputs:
        """One batched forward pass. No @no_grad: attribution backpropagates."""

    @abstractmethod
    def value(self, raw: torch.Tensor) -> ValueEstimate:
        """One position's raw value-head output -> ValueEstimate."""

    def layout(self, board: chess.Board) -> TokenLayout:
        """The tokens of `board`'s forward pass. Square-token models keep the default."""
        return TokenLayout.squares(board, self.vocab.frame)

    # --- internals ---
    @abstractmethod
    def hook_modules(self) -> dict:
        """{hook name: nn.Module}, see the class docstring."""

    @abstractmethod
    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        """The policy head on one position's residual (T, d) -> (n_moves,) logits."""

    @abstractmethod
    def decompose_attention(self, layer: int, x: torch.Tensor, cond: Conditioning) -> AttnDecomp:
        """Layer `layer`'s attention from its input `x` (B, T, d)."""

    @abstractmethod
    def mlp_out(self, layer: int) -> torch.nn.Linear:
        """The MLP's output projection: its input is the hidden (neuron)
        activation, its output the MLP write."""

    def attn_input_key(self, layer: int) -> str:
        """The hook whose output is layer `layer`'s attention input."""
        if self.caps.norm == "post":
            return "embed_in" if layer == 0 else f"block_{layer - 1:02d}"
        return f"attn_in_{layer:02d}"

    def extensions(self) -> dict:
        """Architecture-specific accessors by name, e.g. {"gab": ...}."""
        return {}


# ----- registry ---------------------------------------------------------------------------

_USER_ADAPTERS: list = []      # register_adapter / entry points, most recent first
_BUILTIN_ADAPTERS: list = []   # chessformer_lens.adapters, in priority order
_LOADED = False


def register_adapter(cls):
    """Class decorator: make `cls` resolvable by alias. Adapters registered this
    way are tried before the built-in ones, most recent first."""
    if cls in _USER_ADAPTERS:
        _USER_ADAPTERS.remove(cls)
    _USER_ADAPTERS.insert(0, cls)
    return cls


def _register_builtin(cls):
    if cls not in _BUILTIN_ADAPTERS:
        _BUILTIN_ADAPTERS.append(cls)
    return cls


def registered_adapters() -> list:
    """Every adapter class, in resolution order. Imports the built-ins and the
    `chessformer_lens.adapters` entry points on first use."""
    global _LOADED
    if not _LOADED:
        _LOADED = True
        import_module("chessformer_lens.adapters")
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="chessformer_lens.adapters")
        except Exception:
            eps = ()
        for ep in eps:
            try:
                register_adapter(ep.load())
            except Exception as exc:
                warnings.warn(f"chessformer_lens: adapter entry point {ep.name!r} failed "
                              f"to load: {exc}")
    return [*_USER_ADAPTERS, *_BUILTIN_ADAPTERS]


def resolve_adapter(alias: str):
    """alias -> (adapter class, display name), without loading weights.
    ValueError if no adapter serves it."""
    for cls in registered_adapters():
        name = cls.resolve(str(alias).strip())
        if name is not None:
            return cls, name
    raise ValueError(f"Unknown model {alias!r}.\n\n{format_catalog()}")


def available_models() -> list[dict]:
    """Every registered family's models (`ModelAdapter.models`), each tagged
    with its "family" — what an engine picker lists. Families alphabetically."""
    out = []
    for cls in sorted(registered_adapters(), key=lambda c: (c.family or c.name).lower()):
        out += [{**m, "family": cls.family or cls.name} for m in cls.models()]
    return out


def format_catalog() -> str:
    """Every registered family's aliases as text, families alphabetically."""
    lines = ["Chessformer engines, by alias:"]
    for cls in sorted(registered_adapters(), key=lambda c: (c.family or c.name).lower()):
        lines += cls.catalog()
    return "\n".join(lines)
