"""DeepMind's searchless chess transformers (Ruoss et al. 2024, "Amortized Planning with
Large-Scale Transformers: A Case Study on Chess", arXiv 2402.04494) — the action-value
models 9M, 136M and 270M, distilled from Stockfish 16 with no search. The 270M model plays
at 2895 Lichess blitz Elo against humans.

Reads DeepMind's own Orbax checkpoints (needs the `tensorstore` package, once: the weights
are then kept as a .pt), rebuilds the net in PyTorch (`SearchlessModel`), and serves it to
the engine through `SearchlessAdapter`:

  input     one sequence per (position, candidate move): a 0 token, the 77 characters of
            the FEN (side, 64 squares in FEN order, castling, en passant, the two
            counters), then the move. 79 tokens, absolute frame (no mirroring for Black)
  output    128 buckets over the win probability of that move, read at the last token;
            Q = the expected bucket value. The model plays argmax Q
  policy    the 1968-move action list of the paper; `caps.policy = "win_prob"`: evaluate()
            lists each legal move's Q (they do not sum to 1), select_move is argmax unless a
            temperature is given (softmax(Q / temperature), DeepMind's own sampling)
  value     the best move's Q (scalar; no win/draw/loss split)
  rating    none: Conditioning is ignored, except `extras["move"]`, below

One forward pass of a position runs every legal move's sequence as one batch, so it costs
N_legal sequences of 79 tokens (the original engine works the same way): seconds per
position for 270M on a CPU, well under a second on MPS or CUDA. Row 0 of that batch is the
"focal" move — `Conditioning(extras={"move": "e2e4"})`, or the model's best move — and is
what every read path (residual stream, attention, neurons) describes; the engine's
move-centric methods (move_logit_lens, ablate_grid, carrier_neurons) put the move they
analyze there. Ablations apply to every row, so they are exact for every move.

The logit lens at an intermediate depth reads the focal move's Q estimate so far against
the other moves' final estimates, so `move_logit_lens`' `logit` curve (Q through depth) is
the one to read; its `rank` only ever switches between the focal move and the runner-up.
DeepMind's play() also scores a move that allows a draw by repetition as 0.5 before the
argmax; this adapter does not.

Weights (CC-BY 4.0) download from storage.googleapis.com/searchless_chess on first use:
the zip holds params and params_ema; only `params` (what DeepMind's engines load) is kept,
as `~/.cache/chessformer_lens/searchless-<size>.pt`.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import shutil
import ssl
import types
import urllib.request
import zipfile
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import (ArchSpec, AttnDecomp, Capabilities, HeadOutputs, ModelAdapter, PolicyVocab,
                    Token, TokenLayout, ValueEstimate, _register_builtin, content_scores,
                    out_proj_heads, pick_device, split_heads)
from .lc0 import _weights_dirs, lc0_cache_dir

# ----- tokens and moves -----------------------------------------------------------
# Ported from searchless_chess/src/tokenizer.py and src/utils.py (Apache-2.0, DeepMind 2025).

_CHARACTERS = list("0123456789abcdefghpnrkqPBNRQKw.")
_CHAR_INDEX = {c: i for i, c in enumerate(_CHARACTERS)}
N_BUCKETS = 128
SEQ_LEN = 79            # a 0 token, 77 FEN characters, the move


def _fen_tokens(fen: str) -> list[int]:
    """The 77 token ids of a FEN: side, the 64 squares in FEN order ('.' for empty),
    castling padded to 4, en passant as 2 characters, 3 for each counter."""
    board, side, castling, ep, halfmove, fullmove = fen.split(" ")
    out = [_CHAR_INDEX[side]]
    for c in board.replace("/", ""):
        if c in "12345678":
            out += [_CHAR_INDEX["."]] * int(c)
        else:
            out.append(_CHAR_INDEX[c])
    if castling == "-":
        castling = ""
    out += [_CHAR_INDEX[c] for c in castling] + [_CHAR_INDEX["."]] * (4 - len(castling))
    out += [_CHAR_INDEX[c] for c in ep.replace("-", "..")]
    out += [_CHAR_INDEX[c] for c in halfmove.ljust(3, ".")]
    out += [_CHAR_INDEX[c] for c in fullmove.ljust(3, ".")]
    assert len(out) == 77, fen
    return out


def _all_actions() -> list[str]:
    """DeepMind's 1968-move action list: from every square, the squares a queen or a knight
    there attacks (castling is the king's two-square move), then the promotions on the last
    rank of either side, 4 pieces each, in that order."""
    moves = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        targets = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        targets += list(board.attacks(square))
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        targets += list(board.attacks(square))
        board.remove_piece_at(square)
        for t in targets:
            moves.append(chess.square_name(square) + chess.square_name(t))
    files = "abcdefgh"
    for rank, next_rank in (("2", "1"), ("7", "8")):
        for i, f in enumerate(files):
            moves += [f"{f}{rank}{f}{next_rank}{p}" for p in "qrbn"]
            if i > 0:
                moves += [f"{f}{rank}{files[i - 1]}{next_rank}{p}" for p in "qrbn"]
            if i < 7:
                moves += [f"{f}{rank}{files[i + 1]}{next_rank}{p}" for p in "qrbn"]
    return moves


# ----- the model ---------------------------------------------------------------------
# A LLaMA-style decoder with the causal mask off: Pre-LN, SwiGLU, no biases but the head's,
# learned positions, the token embedding scaled by sqrt(d). Laid out like the toy adapter
# (Identity modules as hook points for streams no module outputs).

class _Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def scores(self, x):
        q, k, v = (split_heads(m(x), self.n_heads) for m in (self.q, self.k, self.v))
        return q, k, v, content_scores(q, k)

    def forward(self, x):
        B, T, d = x.shape
        _, _, v, s = self.scores(x)
        z = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(z)


class _Block(nn.Module):
    def __init__(self, d, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = _Attention(d, n_heads)
        self.resid_mid = nn.Identity()
        self.ln2 = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d_mlp, bias=False)
        self.up = nn.Linear(d, d_mlp, bias=False)
        self.down = nn.Linear(d_mlp, d, bias=False)

    def forward(self, x):
        x = self.resid_mid(x + self.attn(self.ln1(x)))
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class SearchlessModel(nn.Module):
    """forward(tokens (B, 79) long) -> (B, 128) log-probabilities over the return buckets
    of the move in the last token, read at that token."""

    def __init__(self, n_layers=8, d=256, n_heads=8, n_actions=1968, seq_len=SEQ_LEN):
        super().__init__()
        self.d = d
        self.tok_emb = nn.Embedding(n_actions, d)
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, d))
        self.embed_out = nn.Identity()
        self.layers = nn.ModuleList(_Block(d, n_heads, 4 * d) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_BUCKETS)
        self.register_buffer("bucket_values", (torch.arange(N_BUCKETS) + 0.5) / N_BUCKETS,
                             persistent=False)

    def readout(self, x):
        """(…, d) at the last token -> (…, 128) log-probs."""
        return torch.log_softmax(self.head(self.ln_f(x)), dim=-1)

    def forward(self, tokens):
        x = self.embed_out(self.tok_emb(tokens) * math.sqrt(self.d) + self.pos_emb[:tokens.size(1)])
        for blk in self.layers:
            x = blk(x)
        return self.readout(x[:, -1])


# ----- weights --------------------------------------------------------------------------
# The official checkpoints. A zip holds <size>/6400000/{params,params_ema}; the net is kept
# as a .pt once read. sha256 of the zips: 9M checked; the other two are checked by size
# until a download has recorded them.
_NETS = {
    "searchless-9m": dict(display_name="Searchless 9M", zip="9M.zip", n_layers=8, d=256, n_heads=8,
                          size=66375867,
                          sha256="e7a62f07c14819ec77433f108d3490599cdf097a49afe64f1be3973d19c34eed"),
    "searchless-136m": dict(display_name="Searchless 136M", zip="136M.zip", n_layers=8, d=1024,
                            n_heads=8, size=1012357690, sha256=None),
    "searchless-270m": dict(display_name="Searchless 270M", zip="270M.zip", n_layers=16, d=1024,
                            n_heads=8, size=2011006753, sha256=None),
}
_URL = "https://storage.googleapis.com/searchless_chess/checkpoints/"
_CITATION = ("Ruoss, Delétang, Medapati, Grau-Moya, Wenliang, Catt, Reid, Lewis, Veness, Genewein "
             "(2024). Amortized Planning with Large-Scale Transformers: A Case Study on Chess. "
             "arXiv:2402.04494. Weights CC-BY 4.0, https://github.com/google-deepmind/searchless_chess")


def _has_tensorstore() -> bool:
    return importlib.util.find_spec("tensorstore") is not None


def _find_pt(name) -> Path | None:
    for d in _weights_dirs():
        if (d / f"{name}.pt").exists():
            return d / f"{name}.pt"
    return None


def _params_dir(path) -> Path:
    """The `params` directory of an Orbax checkpoint, given it, its step directory, or the
    checkpoint root (the largest step is taken)."""
    path = Path(path)
    if (path / "manifest.ocdbt").exists():
        return path
    if (path / "params" / "manifest.ocdbt").exists():
        return path / "params"
    subdirs = [p for p in path.iterdir() if p.is_dir()] if path.is_dir() else []
    steps = [p for p in subdirs if p.name.isdigit()]
    if steps:
        return _params_dir(max(steps, key=lambda p: int(p.name)) / "params")
    if len(subdirs) == 1:                    # the zip's top directory (9M/), or a clone's checkpoints/
        return _params_dir(subdirs[0])
    raise FileNotFoundError(f"no Orbax checkpoint (manifest.ocdbt) under {path}")


def _orbax_leaves(params_dir) -> dict:
    """{leaf path: float32 tensor} of an Orbax OCDBT checkpoint, read with tensorstore."""
    try:
        import tensorstore as ts
    except ImportError as exc:
        raise ImportError("Reading DeepMind's checkpoints needs the `tensorstore` package:\n"
                          "    pip install tensorstore") from exc
    params_dir = Path(params_dir).resolve()
    sharding = params_dir / "_sharding"
    if sharding.exists():
        leaves = sorted(base64.b64decode(k).decode() for k in json.loads(sharding.read_text()))
    else:
        store = ts.KvStore.open({"driver": "ocdbt", "base": f"file://{params_dir}/"}).result()
        leaves = sorted(k.decode()[:-len("/.zarray")] for k in store.list().result()
                        if k.endswith(b"/.zarray"))
    W = {}
    for leaf in leaves:
        spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{params_dir}/",
                                               "path": leaf}}
        W[leaf] = torch.from_numpy(np.asarray(ts.open(spec).result().read().result(), dtype=np.float32))
    return W


def _state_dict(W: dict) -> dict:
    """Haiku leaves (call order: per layer an LN, the attention's q k v o, an LN, the MLP's
    gate up down; then the final LN and the head) -> SearchlessModel state dict. Haiku
    Linear weights are (in, out)."""
    n_layers = sum(1 for k in W if k.startswith("multi_head_dot_product_attention") and k.endswith("linear.w"))

    def suffix(i):
        return "" if i == 0 else f"_{i}"

    def lin(name):
        return W[name].t().contiguous()

    sd = {"tok_emb.weight": W["embed.embeddings"], "pos_emb": W["embed_1.embeddings"]}
    for i in range(n_layers):
        p, a = f"layers.{i}.", f"multi_head_dot_product_attention{suffix(i)}/"
        sd[p + "ln1.weight"], sd[p + "ln1.bias"] = W[f"layer_norm{suffix(2 * i)}.scale"], W[f"layer_norm{suffix(2 * i)}.offset"]
        for c, j in (("q", 0), ("k", 1), ("v", 2), ("o", 3)):
            sd[p + f"attn.{c}.weight"] = lin(a + f"linear{suffix(j)}.w")
        sd[p + "ln2.weight"], sd[p + "ln2.bias"] = W[f"layer_norm{suffix(2 * i + 1)}.scale"], W[f"layer_norm{suffix(2 * i + 1)}.offset"]
        sd[p + "gate.weight"] = lin(f"linear{suffix(3 * i)}.w")
        sd[p + "up.weight"] = lin(f"linear{suffix(3 * i + 1)}.w")
        sd[p + "down.weight"] = lin(f"linear{suffix(3 * i + 2)}.w")
    sd["ln_f.weight"], sd["ln_f.bias"] = W[f"layer_norm_{2 * n_layers}.scale"], W[f"layer_norm_{2 * n_layers}.offset"]
    sd["head.weight"], sd["head.bias"] = lin(f"linear_{3 * n_layers}.w"), W[f"linear_{3 * n_layers}.b"]
    return sd


def download_searchless_net(name, dest_dir=None) -> Path:
    """Download official net `name` from storage.googleapis.com into `dest_dir` (default
    `lc0_cache_dir()`), keep only its `params`, convert them to `<name>.pt` and return that
    path. The zip (66 MB / 1.0 GB / 2.0 GB) is deleted once extracted."""
    net = _NETS[name]
    dest_dir = Path(dest_dir or lc0_cache_dir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / net["zip"]
    part = zip_path.with_name(zip_path.name + ".part")
    print(f"chessformer_lens: {net['display_name']} weights are not on this machine; downloading "
          f"them now ({net['size'] / 1e6:.0f} MB, one time; peak disk use about "
          f"{2.6 * net['size'] / 1e9:.1f} GB while converting).\n  from:  {_URL + net['zip']}\n"
          f"  to:    {dest_dir / (name + '.pt')}", flush=True)
    try:        # python.org's macOS builds ship without system CAs; certifi has them
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(_URL + net["zip"], headers={"User-Agent": "chessformer_lens"})
    digest, done, shown = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if done * 10 // net["size"] > shown:
                shown = done * 10 // net["size"]
                print(f"  {min(100, shown * 10)}%", end="\n" if shown >= 10 else "", flush=True)
    bad = digest.hexdigest() != net["sha256"] if net["sha256"] else done != net["size"]
    if bad:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{net['zip']}: checksum mismatch after download; try again")
    part.replace(zip_path)
    return convert_searchless_zip(zip_path, name, dest_dir, remove_zip=True)


def convert_searchless_zip(zip_path, name, dest_dir=None, remove_zip=False) -> Path:
    """Extract only `params` from a checkpoint zip, read it and save `<name>.pt`."""
    dest_dir = Path(dest_dir or lc0_cache_dir())
    extract = dest_dir / f"{name}.extract"
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if "/params/" in m and "params_ema" not in m]
        zf.extractall(extract, members)
    try:
        out = convert_searchless_checkpoint(extract, name, dest_dir)
    finally:
        shutil.rmtree(extract, ignore_errors=True)
    if remove_zip:                           # only once the .pt exists
        Path(zip_path).unlink()
    return out


def convert_searchless_checkpoint(checkpoint_dir, name, dest_dir=None) -> Path:
    """An Orbax checkpoint directory (from DeepMind's zip or a searchless_chess clone) ->
    `<dest_dir>/<name>.pt`, a plain float32 state dict plus a `_meta` entry naming the source."""
    dest_dir = Path(dest_dir or lc0_cache_dir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    sd = _state_dict(_orbax_leaves(_params_dir(checkpoint_dir)))
    sd["_meta"] = {"source": _URL + _NETS[name]["zip"], "license": "CC-BY 4.0", "citation": _CITATION}
    out = dest_dir / f"{name}.pt"
    torch.save(sd, out)
    return out


# ----- the adapter --------------------------------------------------------------------

@_register_builtin
class SearchlessAdapter(ModelAdapter):
    """A searchless_chess action-value model (9M, 136M or 270M)."""

    name = "searchless"
    family = "Searchless chess (DeepMind)"
    caps = Capabilities(conditioning=frozenset(), value="scalar", norm="pre",
                        attn_terms=("content",), causal=False, policy="win_prob")

    @classmethod
    def resolve(cls, alias):
        net = _NETS.get(str(alias).strip().lower())
        return net["display_name"] if net else None

    @classmethod
    def models(cls):
        out = []
        for name, net in _NETS.items():
            found = _find_pt(name)
            if found:
                note = str(found)
            else:
                note = f"downloads {net['size'] / 1e6:.0f} MB from storage.googleapis.com on first use"
                if not _has_tensorstore():
                    note += "; needs `pip install tensorstore`"
            out.append({"alias": name, "display_name": net["display_name"], "aliases": [],
                        "available": bool(found) or _has_tensorstore(), "note": note})
        return out

    @classmethod
    def catalog(cls):
        return super().catalog() + ["    (one evaluate runs every legal move's sequence: slow on a CPU for 270M)"]

    @classmethod
    def load(cls, alias="searchless-9m", *, device=None, checkpoint_path=None, **_):
        """`alias` names a net; `checkpoint_path` is a converted .pt or an Orbax checkpoint
        directory (the zip's `<size>/6400000` or a clone's `checkpoints/<size>`). Without
        it the .pt is looked for in $CHESSFORMER_WEIGHTS, ./weights and the cache, and the
        zip downloaded and converted on first use (CHESSFORMER_NO_DOWNLOAD=1 refuses)."""
        key = str(alias).strip().lower()
        if key not in _NETS:
            from ..core import format_catalog
            raise ValueError(f"Unknown searchless alias {alias!r}.\n\n{format_catalog()}")
        net = _NETS[key]
        if checkpoint_path is None:
            checkpoint_path = _find_pt(key)
            if checkpoint_path is None:
                if os.environ.get("CHESSFORMER_NO_DOWNLOAD"):
                    raise FileNotFoundError(
                        f"{net['display_name']}: {key}.pt is not in {', '.join(map(str, _weights_dirs()))} "
                        f"and CHESSFORMER_NO_DOWNLOAD is set. Download {_URL + net['zip']} and pass "
                        f"its unzipped directory as checkpoint_path")
                checkpoint_path = download_searchless_net(key)
        checkpoint_path = Path(checkpoint_path).expanduser()
        if checkpoint_path.is_dir():
            sd = _state_dict(_orbax_leaves(_params_dir(checkpoint_path)))
        else:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            sd.pop("_meta", None)
        print(f"chessformer_lens: loading {net['display_name']} from {checkpoint_path}", flush=True)

        self = cls()
        self.device = pick_device(device)
        self.model = SearchlessModel(net["n_layers"], net["d"], net["n_heads"])
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        self.arch = ArchSpec(n_layers=net["n_layers"], n_heads=net["n_heads"], d_model=net["d"],
                             d_mlp=4 * net["d"], name=key, display_name=net["display_name"],
                             checkpoint=str(checkpoint_path))
        self.cfg = self.arch.legacy_cfg(device=self.device, history=1)
        self.spec = self.arch.legacy_spec(aliases=())
        self.vocab = PolicyVocab(_all_actions(), frame="absolute")
        self._best = {}            # fen -> the model's best move, uci (the default focal move)
        self._last_q = None        # the last single-position forward's Q vector, for unembed
        self._last_focal_idx = None
        return self

    # --- inputs and outputs ---
    def _ordered_moves(self, board, focal):
        """The legal moves in action order with the focal move (a uci, or None) first."""
        moves = sorted(board.legal_moves, key=lambda m: self.vocab.index_of[m.uci()])
        if focal is None:
            focal = self._best.get(board.fen())
        if focal is not None:
            mv = chess.Move.from_uci(focal)
            if mv in moves:
                moves.remove(mv)
                moves.insert(0, mv)
        return moves

    def _sequences(self, boards, focals):
        """(rows (R, 79) long on the device, [(start, moves)] per board)."""
        rows, spans = [], []
        for board, focal in zip(boards, focals):
            fen = _fen_tokens(board.fen())
            moves = self._ordered_moves(board, focal)
            actions = [self.vocab.index_of[m.uci()] for m in moves] or [0]   # a dummy row for a game over
            spans.append((len(rows), moves))
            for a in actions:
                rows.append([0] + fen + [a])
        return torch.tensor(rows, dtype=torch.long, device=self.device), spans

    def tokens(self, boards):
        """(B, 79): each position with its focal move (the cached best move, else action 0)."""
        toks, spans = self._sequences(boards, [None] * len(boards))
        return torch.stack([toks[start] for start, _ in spans])

    def forward(self, boards, cond):
        want = cond.extras.get("move")
        focals = list(want) if isinstance(want, (list, tuple)) else [want] * len(boards)
        toks, spans = self._sequences(boards, focals)
        logp = self.model(toks)                                      # (R, 128)
        q = (logp.exp() * self.model.bucket_values).sum(-1)          # (R,)
        # a position whose focal move is unknown is run again with its best move in row 0
        rerun = False
        for board, focal, (start, moves) in zip(boards, focals, spans):
            if focal is None and moves and board.fen() not in self._best:
                best = int(q[start:start + len(moves)].argmax())
                if len(self._best) >= 4096:
                    del self._best[next(iter(self._best))]
                self._best[board.fen()] = moves[best].uci()
                rerun = True
        if rerun:
            toks, spans = self._sequences(boards, focals)
            logp = self.model(toks)
            q = (logp.exp() * self.model.bucket_values).sum(-1)

        policy = q.new_full((len(boards), len(self.vocab)), float("-inf"))
        value = []
        for b, (start, moves) in enumerate(spans):
            if moves:
                idx = torch.tensor([self.vocab.index_of[m.uci()] for m in moves], device=q.device)
                policy[b, idx] = q[start:start + len(moves)]
                value.append(logp[start + int(q[start:start + len(moves)].argmax())])
            else:
                value.append(logp[start])
        if len(boards) == 1:
            self._last_q = policy[0].detach()
            self._last_focal_idx = int(toks[0, -1]) if spans[0][1] else None
        return HeadOutputs(policy=policy, value=torch.stack(value))

    def value(self, raw):
        """128 bucket log-probs -> the expected win probability."""
        probs = torch.softmax(raw.float(), dim=-1)
        return ValueEstimate(expected=float((probs * self.model.bucket_values.to(probs.device).float()).sum()))

    def layout(self, board):
        fen = board.fen().split(" ")
        toks = [Token("control", "[bos]"), Token("fen_char", fen[1])]
        for rank in range(7, -1, -1):
            for file in range(8):
                sq = chess.square(file, rank)
                toks.append(Token("square", chess.square_name(sq), square=sq))
        castling = "" if fen[2] == "-" else fen[2]
        toks += [Token("fen_char", c) for c in castling.ljust(4, ".")]
        toks += [Token("fen_char", c) for c in fen[3].replace("-", "..")]
        toks += [Token("fen_char", c) for c in fen[4].ljust(3, ".")]
        toks += [Token("fen_char", c) for c in fen[5].ljust(3, ".")]
        focal = self._best.get(board.fen())
        toks.append(Token("move", focal or "[move]", move=focal))
        return TokenLayout(tuple(toks), frame="absolute", readout=SEQ_LEN - 1, is_board=False)

    # --- internals ---
    def hook_modules(self):
        hooks = {"embed_in": self.model.embed_out}
        for i, blk in enumerate(self.model.layers):
            hooks[f"attn_in_{i:02d}"] = blk.ln1
            hooks[f"attn_{i:02d}"] = blk.attn
            hooks[f"postattn_{i:02d}"] = blk.resid_mid
            hooks[f"mlp_{i:02d}"] = blk.down
            hooks[f"block_{i:02d}"] = blk
        return hooks

    def unembed(self, x):
        """(79, d) residual of the focal move's sequence -> (1968,): the focal move's Q read
        at the last token through the final norm and the head, in place of its entry in the
        last forward's Q vector (so at the final block this is exactly the policy)."""
        probs = torch.softmax(self.model.head(self.model.ln_f(x[-1])), dim=-1)
        q_focal = (probs * self.model.bucket_values).sum()
        if self._last_q is None or self._last_focal_idx is None:
            return q_focal.new_full((len(self.vocab),), float("-inf"))
        out = self._last_q.to(x.device).clone()
        out[self._last_focal_idx] = q_focal
        return out

    def decompose_attention(self, layer, x, cond):
        attn = self.model.layers[int(layer)].attn
        q, k, v, s = attn.scores(x)
        return AttnDecomp(q, k, v, {"content": s}, W_O=out_proj_heads(attn.o.weight, attn.n_heads),
                          b_O=None, mask=None)

    def mlp_out(self, layer):
        return self.model.layers[int(layer)].down
