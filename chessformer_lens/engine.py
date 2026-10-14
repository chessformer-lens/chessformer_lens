"""Mechanistic Interpretability core for chess transformers.

`ChessformerEngine` class contains every read and intervention path below and works on
any model a `ModelAdapter` describes (see core.py): the adapter supplies the
model and its idiosyncraties in adapters/. 

  MaiaEngine    a Maia-3 checkpoint (5M / 23M / 79M / 3M-ablation), conditioned
                on a rating pair
  LeelaEngine   Leela Chess Zero BT4 (185M params, 15 layers x 32 heads x 1024d),
                no rating input; `bt4` / `bt4-332` download on first use, extremely
                high performance chess engine, trained with self-play


  evaluate              one forward pass yielding the full normalized policy over
                        legal moves (in descending order), the W/D/L for the side to
                        move (`value` for any value head), and auxiliary heads such
                        as the moves-left estimate
  evaluate_batch        the same over many positions

  select_move           pick a move at a rating (temperature 0 = argmax)
  tokens                the position as the model's input tokens

  layout                what each token of the residual stream stands for


  run_with_cache        forward pass returning (out, cache): the whole
                        residual stream, transformer_lens style
  run_with_hooks        forward pass with intervention hooks: activation
                        patching, ablation, steering
  logit_lens            decode any residual activation at a readout point
                        by pushing it through the policy head

  attention             one (layer, head)'s attention, by score term: semantic
                        QKᵀ, plus the geometric GAB on smolgen models, and the
                        softmax the model actually runs
  attention_terms       every score term of one layer, all heads, in one pass
  qk_scores             one layer's scaled QKᵀ logits
  gab_bias              one layer's generated bias (smolgen models)
  gab_coeffs            the coefficients head h applies to the GAB bank
  gab_templates         the static square-pair template bank every layer shares
  head_writes           exact residual writes of one layer's attention
  
  
    
  ablate_head           forward pass with one head's write removed, exactly
  steer_head            the same write multiplied by a factor (activation steering:
                        2 doubles it, -1 flips it); steer_neuron for one MLP unit
  ablate_grid           the carrier heatmap of one move: every head ablated in
                        turn, Δlogit = ablated − clean (negative = the head
                        carried it)
  ablate_grid_batch     ablate_grid called with batches of many positions at once;
                        minimizes unnecessary syncing to cpu to take advantage of gpus
  carrier_neurons       the carrier table of one move at neuron grain: every MLP
                        unit scored at once by attribution patching, with its
                        per-token footprint; optionally the top ones re-measured
                        by exact zero-ablation
  neuron_activation     one MLP unit's activation on every token (and its write's size)
  neuron_overview       every layer's most active MLP units on this position, for
                        flipping through the network layer by layer
  ablate_neuron         forward pass with one MLP unit's write removed exactly, on
                        every token
  top_neurons           the MLP units that fire most over many positions
  top_positions         the positions that make one MLP unit fire most
  neuron_effects        what one MLP unit does on each position: the moves it pushes
                        up and down, and whether removing it changes the top move
  search                (LeelaEngine) lc0's search on the same net: visits and Q per
                        move beside the raw policy, the PV, the best move
  residual_stream       per-token views of how the stream is built up, one row
                        per readout point
  compare_residual      the same position at two ratings, differenced, where
                        skill diverges inside the stream, not just in the
                        output
  depth_points          the readout points as [{label, kind}]—the x axis in the plots below
  move_logit_lens       one move's depth curve: logit, probability and rank at
                        every readout point
  logit_per_depth       that curve's raw logit alone, unmasked
  policy_per_depth      that curve after the softmax over legal moves
  rank_per_depth        that curve as a rank, 1 being the top move at that depth

  move_info             One move in every representation at once in dictionary.
                        THE method to use when converting between any of the
                        five move forms
  to_move               any move form -> chess.Move (used frequently in other files)

  save_activations      persist the most recent forward's snapshot to disk
  no_capture            context manager: forward passes leave the snapshot alone
  remove_hooks          detach the capture hooks, for a bare forward with no
                        CPU copies

Module level: `load_engine(alias)` builds the engine an alias names from the
adapter registry (`resolve_engine` validates an alias without loading weights,
`format_engine_list` prints every family's aliases), `register_adapter` adds
a model family, and `pick_device` resolves the torch device (explicit >
$CHESSFORMER_DEVICE > cuda > mps > cpu). `load_positions` reads FENs, boards
or a PGN file into boards, `positions_from_pgn` samples one position per game,
and `ordinary_positions` is a bundled pool of 1,000 from real games.

Every tensor a read path returns is on CPU. Depth reads the same as in the figures: 
`emb`, then `aN`/`mN` for layer N's attention and MLP sub-layers, then `enc`. 
`depth_points` hands you that axis directly (18 points on
every Maia-3 size, 31 on BT4).

engine.py imports cleanly into a notebook and is called by interp_plot.py
(static figures),  interp_widget.py (interactive panels), and the standalone
app in bridge.py/app.py/ui.py (and web.py, in a browser).
"""
from __future__ import annotations

import itertools
import os
import random
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn
import torch

from .adapters.lc0 import BT4Model, Lc0Adapter
from .adapters.smolgen import check_recon
from .core import (Conditioning, ModelAdapter, TokenLayout, canon_square, format_catalog,
                   pick_device, real_square, register_adapter, resolve_adapter,
                   sample_from_logits)

__all__ = ["ChessformerEngine", "MaiaEngine", "LeelaEngine", "BT4Model",
           "load_engine", "resolve_engine", "format_engine_list",
           "build_cfg", "pick_device", "register_adapter", "Conditioning",
           "load_positions", "positions_from_pgn", "ordinary_positions",
           "NeuronReport", "NeuronScan"]


def _elo_out(x):
    """How a rating argument is echoed back in a result dict."""
    return x.self_elo if isinstance(x, Conditioning) else int(x)


class ChessformerEngine:
    """Hook-based mechanistic interpretability engine over any chess transformer a
    `ModelAdapter` describes.

    Provides read paths (run_with_cache, logit_lens, residual_stream, attention,
    gab_*) and intervention paths (run_with_hooks, ablate_head, ablate_grid,
    ablate_grid_batch). Every tensor a read path returns is on CPU.

    What the model has, for code that degrades gracefully:
      caps                the adapter's Capabilities (conditioning, value kind,
                          norm placement, attention score terms, aux heads)
      has_conditioning    the rating pair is a real input (Maia-3) or ignored (BT4)
      has_generated_bias  attention carries a generated square-pair bias
                          (GAB / smolgen), so the gab_* methods work
      has_mlh             evaluate() carries a moves-left estimate"""

    def __init__(self, adapter: ModelAdapter | str | None = None, activation_dir="activations",
                 *, alias: str | None = None, **load_kwargs):
        """Wrap a loaded adapter, or resolve and load one from `alias`
        (`load_kwargs`, e.g. device=, go to its `load`). Installs forward hooks
        that copy every sub-layer's output to CPU on each forward — what makes
        the read paths work, at the cost of a handful of device->host transfers
        per forward. `activation_dir` is noted for `save_activations`."""
        if isinstance(adapter, str):
            alias, adapter = adapter, None
        if adapter is None:
            if alias is None:
                raise TypeError("ChessformerEngine needs a model: pass an adapter or an "
                                "alias, e.g. load_engine('bt4').\n\n" + format_catalog())
            cls, _ = resolve_adapter(alias)
            adapter = cls.load(alias, **load_kwargs)
        self.adapter = adapter
        self.model = adapter.model
        self.device = adapter.device
        self.cfg = adapter.cfg
        self.spec = adapter.spec
        self.arch = adapter.arch
        self.caps = adapter.caps
        self.vocab = adapter.vocab
        self.all_moves = self.vocab.moves
        self.all_moves_dict = self.vocab.index_of
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        # Not created here: `save_activations` makes it on first write, so
        # importing the engine in a notebook leaves no directory behind.
        self.activation_dir = Path(activation_dir)

        self._activations: dict[str, torch.Tensor] = {}
        self._aux: dict = {}              # extra head outputs of the last captured forward
        self._last_aux: dict = {}         # ... of the last forward, captured or not
        self._capture_on_device = False   # see _register_hooks; ablate_grid_batch flips it
        self._capture_paused = False      # see no_capture
        self._hooks: list = []
        self.hook_points: dict[str, torch.nn.Module] = {}   # name -> module (read or patch)
        self._register_hooks()

    # ----- what the model has -------------------------------------------------
    @property
    def has_conditioning(self) -> bool:
        return "elo" in self.caps.conditioning

    @property
    def has_generated_bias(self) -> bool:
        return "gab" in self.adapter.extensions()

    @property
    def has_mlh(self) -> bool:
        return "mlh" in self.caps.aux_heads

    def layout(self, board: chess.Board) -> TokenLayout:
        """What each token of this position's residual stream stands for."""
        return self.adapter.layout(board)

    # ----- activation hooks -------------------------------------------------
    def _register_hooks(self):
        """Install the permanent capture hooks on every module the adapter
        names (`ModelAdapter.hook_modules`), overwritten on every forward, so
        the snapshot is always the most recent position:

          embed_in      the stream entering block 0
          attn_NN       block NN's attention write
          postattn_NN   the running stream after attention
          mlp_NN        block NN's MLP write
          block_NN      the running stream after the whole block
          encoder_out   after the final encoder norm, where the heads read
                        through one (Maia-3; BT4's heads read block_14
                        directly, so it has no such point)
          attn_in_NN    the attention input, on Pre-LN models

        attn_NN/mlp_NN are the vectors *added* to the stream (dropout is
        identity in eval), while the others are the stream itself. Everything
        is copied to CPU on the way out. `hook_points` records name -> module
        for all of these, so `run_with_hooks` can patch anywhere you can read.

        `_capture_on_device` suspends that CPU copy. On unified memory it costs
        nothing either way, but on a discrete GPU it is 34 pageable transfers
        per forward pass — enough to dominate a sweep that runs hundreds of
        them. Callers that set it must restore it, and must not read the
        snapshot from a path that assumes CPU tensors (`save_activations`)."""
        def make_hook(name):
            def hook(_module, _inp, out): #classic PyTorch
                if self._capture_paused:
                    return
                t = out[0] if isinstance(out, tuple) else out #get first element of tuple
                self._activations[name] = t.detach() if self._capture_on_device \
                    else t.detach().to("cpu") #remove tensor from autograd and move it to cpu or not
            return hook

        for name, module in self.adapter.hook_modules().items():
            self.hook_points[name] = module
            self._hooks.append(module.register_forward_hook(make_hook(name)))

    def remove_hooks(self):
        """Detach the capture hooks, for a bare forward with no CPU copies.

        `hook_points` stays populated so `run_with_hooks` keeps working, but
        the read paths will KeyError once `_activations` stops being refreshed.
        There is no re-register; build a new engine."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @contextmanager
    def no_capture(self):
        """Forward passes inside leave the activation snapshot (and `_aux`) as
        they were — for running the model on other positions without
        disturbing what a panel is showing. The numbers are unchanged."""
        prev = self._capture_paused
        self._capture_paused = True
        try:
            yield self
        finally:
            self._capture_paused = prev

    def _snap(self, name):
        """A snapshot activation; inside no_capture it would be the previous position's."""
        if self._capture_paused:
            raise RuntimeError("activation reads inside no_capture() would answer for the previous position")
        return self._activations[name]

    # ----- the forward pass ---------------------------------------------------
    def tokens(self, board: chess.Board) -> torch.Tensor:
        """The position as the model's input tensor, batch of one, on the device."""
        return self.adapter.tokens([board])

    def _forward(self, board: chess.Board, self_elo=None, oppo_elo: int | None = None):
        """One raw forward pass. Resets and repopulates `self._activations` via the
        hooks, stashes any extra head output in `self._aux`, and returns
        (logits_move (n_moves,), raw value) as floats.
        No @no_grad here: carrier_neurons backpropagates through it."""
        cond = Conditioning.coerce(self_elo, oppo_elo)
        if not self._capture_paused:
            self._activations = {}
            self._aux = {}
        out = self.adapter.forward([board], cond)
        aux = {k: float(v[0].detach()) for k, v in out.aux.items()}
        self._last_aux = aux
        if not self._capture_paused:
            self._aux = aux
        return out.policy[0].float(), out.value[0].float()

    def _forward_batch(self, boards, self_elo=None, oppo_elo=None):
        """`_forward` over a list of boards in one pass: (B, n_moves) move logits.

        The capture hooks fire as usual, so afterwards `_activations` holds this
        batch's (B, tokens, dim) tensors rather than one position's. The read
        paths index [0] and so would only ever see the first board — this is
        for the intervention paths, which stay batch-aware throughout."""
        cond = Conditioning.coerce(self_elo, oppo_elo)
        if not self._capture_paused:
            self._activations = {}
            self._aux = {}
        out = self.adapter.forward(list(boards), cond)
        aux = {k: v.detach().tolist() for k, v in out.aux.items()}
        self._last_aux = aux
        if not self._capture_paused:
            self._aux = aux
        return out.policy.float()

    # ----- per-model members (compat) -----------------------------------------
    # The old subclass seam, now answered by the adapter; kept because notebooks
    # reach for them.
    def _move_logits(self, x):
        """Full (n_moves,) move logits from one position's residual x (tokens, dim)."""
        return self.adapter.unembed(x)

    def _move_squares(self, idx):
        """Canonical (from, to) squares for a policy-move index."""
        return self.vocab.squares(idx)

    def _wdl(self, logits_value) -> dict | None:
        """The value head's raw output -> {"win", "draw", "loss"} for the side to move."""
        return self.adapter.value(logits_value).legacy_wdl()

    # ----- policy index <-> move ---------------------------------------------
    def _decode_idx(self, board: chess.Board, idx: int) -> chess.Move:
        """Policy index -> chess.Move on the real board (un-mirroring for Black),
        with no legality check. Raises ValueError on an undecodable uci."""
        return self.vocab.decode(board, idx)

    def _idx_to_move(self, board: chess.Board, idx: int):
        """Decode a policy index to a legal chess.Move, un-mirroring for Black.
        Returns None if the index doesn't decode to a move that's legal here.
        (Distinct from the `idx_to_move` attribute, the raw index -> uci table.)"""
        try:
            mv = self._decode_idx(board, idx)
        except ValueError:
            return None
        return mv if mv in board.legal_moves else None

    def _legal_mask(self, board: chess.Board) -> torch.Tensor:
        """Boolean (n_moves,) mask of this position's legal moves in the policy
        index space (Black mirrored). Moves outside the vocabulary are skipped."""
        return self.vocab.legal_mask(board)

    def _policy(self, board, logits):
        """(legal-masked logits, [(uci, prob)] over legal moves, descending)."""
        legal_mask = self._legal_mask(board).to(logits.device)
        policy = []
        if bool(legal_mask.any()):                  # may be empty for hand-edited positions
            logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)   # normalized over legal moves
            for idx in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist():
                mv = self._idx_to_move(board, idx)
                if mv is not None:
                    policy.append((mv.uci(), float(probs[idx])))
            policy.sort(key=lambda x: x[1], reverse=True)
        return logits, policy

    # ----- forward / policy -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, board: chess.Board, self_elo=None, oppo_elo: int | None = None):
        """One forward pass. Returns the full normalized policy over legal moves
        (descending), the WDL for the side to move (None on a scalar value head;
        `value` has the expected score either way), the moves-left estimate
        (None without one), every auxiliary head, and stashes activations."""
        logits, logits_value = self._forward(board, self_elo, oppo_elo)
        logits, policy = self._policy(board, logits)
        value = self.adapter.value(logits_value)
        return {
            "policy": policy,                                   # [(uci, prob)] desc
            "wdl": value.legacy_wdl(),                          # side-to-move perspective
            "mlh": self._last_aux.get("mlh"),                   # expected plies to game end, or None
            "_logits": logits,                                  # masked, for sampling
            "value": value,                                     # ValueEstimate
            "aux": dict(self._last_aux),                        # every auxiliary head
        }

    @torch.no_grad()
    def evaluate_batch(self, boards, self_elo=None, oppo_elo=None):
        """`evaluate` over many positions in one forward pass.
        Returns one {policy, wdl, value, aux} dict per board."""
        boards = list(boards)
        if not boards:
            return []
        with self.no_capture():
            out = self.adapter.forward(boards, Conditioning.coerce(self_elo, oppo_elo))
        res = []
        for i, b in enumerate(boards):
            _, policy = self._policy(b, out.policy[i].float())
            value = self.adapter.value(out.value[i].float())
            res.append({"policy": policy, "wdl": value.legacy_wdl(), "value": value,
                        "aux": {k: float(v[i].detach()) for k, v in out.aux.items()}})
        return res

    def select_move(self, board: chess.Board, self_elo=None, oppo_elo: int | None = None,
                    temperature: float = 1.0, top_p: float = 1.0):
        """Pick a move at the given rating (temperature 0 = argmax, `top_p` < 1
        = nucleus sampling). Activations correspond to this position."""
        res = self.evaluate(board, self_elo, oppo_elo)
        idx = sample_from_logits(res["_logits"], temperature, top_p)
        return self._idx_to_move(board, idx), res

    # ----- transformer_lens-style ergonomics --------------------------------
    def run_with_cache(self, board: chess.Board, self_elo=None, oppo_elo: int | None = None):
        """Forward pass returning (out, cache).

          out   = the evaluate() dict (policy / wdl / _logits / value / aux)
          cache = {name: Tensor(tokens, d_model)} snapshot of the residual
                  stream, cloned so it survives the next forward. Keys are the
                  `hook_points` names.

        Cached tensors are always on CPU, whatever `self.device` is. Contrast
        `run_with_hooks`, whose hook functions see activations on the live
        device — move tensors yourself when feeding a cached value into an
        intervention. On a CPU-only setup (or with CHESSFORMER_DEVICE=cpu) 
        both are on the CPU and none of this matters."""
        out = self.evaluate(board, self_elo, oppo_elo)
        cache = {k: self._snap(k).squeeze(0).clone() for k in self._activations}
        return out, cache

    @staticmethod
    def _patch_hook(fn):
        """Wrap `fn(activation) -> activation | None` as a forward hook, leaving
        the module's other outputs alone (self_attn returns (out, None))."""
        def hook(_module, _inp, out):
            is_tuple = isinstance(out, tuple)
            t = out[0] if is_tuple else out
            new = fn(t)
            if new is None:
                return out
            return (new, *out[1:]) if is_tuple else new
        return hook

    def run_with_hooks(self, board: chess.Board, self_elo=None, oppo_elo: int | None = None,
                       *, fwd_hooks=(), return_type: str = "policy"):
        """Forward pass with temporary intervention hooks — activation patching,
        ablation, steering. Each `fwd_hooks` entry is (name, fn) where `name`
        is a key of `hook_points` and `fn(activation) -> activation | None`
        (return None to leave it unchanged). `activation` is that module's
        output on the live device: the residual *write* for attn_NN/mlp_NN,
        the running stream for everything else.

        return_type='policy' -> the evaluate() dict; 'logits' -> the raw
        (logits_move, value) tensors, unmasked. Hooks are always removed
        afterward, even on error.        
        Slicing attn_NN channels does not isolate a head
        use `ablate_head()` / `head_writes()` for that.
        Take your clean cache before you patch, not after.

        Example — halve layer 5's whole attention write:
            self.run_with_hooks(board, 1500, fwd_hooks=[("attn_05", lambda a: a * 0.5)])
        """
        handles = []
        try:
            for name, fn in fwd_hooks:
                handles.append(self.hook_points[name].register_forward_hook(self._patch_hook(fn)))
            if return_type == "logits":
                with torch.no_grad():
                    return self._forward(board, self_elo, oppo_elo)
            return self.evaluate(board, self_elo, oppo_elo)
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def logit_lens(self, activation: torch.Tensor, board: chess.Board | None = None,
                   *, legal_only: bool = True):
        """Decode a residual-stream activation through the policy head (logit lens).

        `activation` is (tokens, dim), any value from
        a cache. Returns the top move as a dict {idx, uci, san, from, to, piece,
        logit}, masked to legal moves when `legal_only` (`uci`/`san`/`piece` are
        None if the argmax index doesn't decode to a legal move).

        Caveat: on Post-LN models whose heads read through a final norm
        (Maia-3), this applies the trained head directly to an intermediate
        residual, skipping that norm — only `encoder_out` is read in
        distribution; earlier points are a lens, not a prediction. Compare
        readout points against each other rather than against the real policy.
        """
        x = activation.to(self.device)
        if x.dim() == 3:
            x = x[0]
        logits = self.adapter.unembed(x)                    # (n_moves,)
        if board is None:
            return logits
        if legal_only:
            legal = self._legal_mask(board).to(self.device)
            if bool(legal.any()):
                logits = logits.masked_fill(~legal, float("-inf"))
        idx = int(torch.argmax(logits))
        frm, to = self.vocab.squares(idx)
        mv = self._idx_to_move(board, idx)
        pc = board.piece_at(mv.from_square) if mv is not None else None
        return {
            "idx": idx, "logit": float(logits[idx]),
            "uci": mv.uci() if mv else None,
            "san": board.san(mv) if mv is not None else None,
            "from": frm, "to": to,
            "piece": pc.symbol() if pc is not None else None,
        }

    # ----- live attention (per layer / head) --------------------------------
    @torch.no_grad()
    def _block_input(self, board, self_elo, oppo_elo, layer):
        """One forward pass, then the input of layer `layer`'s attention:
        (1, tokens, dim) on the live device — exactly the tensor its query, key
        and value (and, on smolgen models, its GAB generator) are made from."""
        self.evaluate(board, self_elo, oppo_elo)
        return self._snap(self.adapter.attn_input_key(int(layer))).to(self.device)

    def _decompose(self, board, self_elo, oppo_elo, layer):
        """(attention input x, Conditioning, AttnDecomp) of one layer, batch of one."""
        cond = Conditioning.coerce(self_elo, oppo_elo)
        x = self._block_input(board, cond, None, layer)
        return x, cond, self.adapter.decompose_attention(int(layer), x, cond)

    @torch.no_grad()
    def attention(self, board, self_elo=None, oppo_elo=None, layer=0, head=0):
        """The attention of one (layer, head) for the current position, taken
        apart by score term (Chessformer Fig. 1 on the smolgen models):

          qk           = semantic dot-product logits (scaled QKᵀ)
          gab          = geometric attention bias (smolgen models; None elsewhere)
          attn         = softmax of every term — the head's actual attention
          attn_content = softmax(qk), attention without the other terms;
                         compare with `attn` to see what GAB adds
          attn_layer   = mean over all heads of `attn`
          coeffs       = the smolgen mixing coefficients that generated this
                         head's gab (see gab_coeffs())
          term_names   every additive score term, "content" (= qk) first
          terms        the selected head's matrix of each term other than
                         content and gab
          mask         the (tokens, tokens) attention mask, True = blocked, or None
          layout       what each row / column stands for (TokenLayout.to_json)

        Computed directly from the stream entering the layer, using the layer's
        own projections and bias generator — no re-implementation of the model.
        On square-token models the matrices are 64x64 in the side-to-move
        frame; square = rank*8 + file."""
        L, h = int(layer), int(head)
        x, cond, dec = self._decompose(board, self_elo, oppo_elo, L)
        attn = dec.pattern()
        attn_content = dec.pattern(["content"])
        gab = self.adapter.extensions().get("gab")
        has_gab = gab is not None and gab.has(L)
        coeffs = gab.coeffs(L, x) if has_gab else None
        return {
            "layer": L, "head": h, "num_heads": int(dec.q.size(1)),
            "gen_size": gab.gen_size(L) if has_gab else None,
            "qk": dec.terms["content"][0, h].cpu().tolist(),          # selected head
            "gab": dec.terms["gab"][0, h].cpu().tolist() if "gab" in dec.terms else None,
            "attn": attn[0, h].cpu().tolist(),                       # selected head
            "attn_content": attn_content[0, h].cpu().tolist(),       # selected head, content only
            "attn_layer": attn[0].mean(0).cpu().tolist(),            # whole layer: mean over heads
            "coeffs": coeffs[0, h].cpu().tolist() if coeffs is not None else None,
            "term_names": list(dec.terms),
            "terms": {n: t[0, h].cpu().tolist() for n, t in dec.terms.items()
                      if n not in ("content", "gab")},
            "mask": (dec.mask.expand_as(attn)[0, h].cpu().tolist()
                     if dec.mask is not None else None),
            "layout": self.layout(board).to_json(),
        }

    @torch.no_grad()
    def attention_terms(self, board, self_elo=None, oppo_elo=None, layer=0):
        """Every additive score term of one layer's attention, all heads, from
        one forward pass: {"terms": {name: (H, tokens, tokens)}, "mask":
        (tokens, tokens) bool or None (True = blocked)}. The names are
        `caps.attn_terms` — "content" (scaled QKᵀ) first, then e.g. "gab"; the
        attention the model runs is softmax(sum of the terms, masked)."""
        _, _, dec = self._decompose(board, self_elo, oppo_elo, layer)
        mask = None
        if dec.mask is not None:
            T = dec.q.size(2)
            mask = dec.mask.expand(1, 1, T, T)[0, 0].cpu()
        return {"terms": {n: t[0].cpu() for n, t in dec.terms.items()}, "mask": mask}

    # ----- GAB / smolgen decomposition --------------------------------------
    # GAB is generated, not stored: a tiny MLP ("smolgen") reads the board state
    # and emits, per head, `gen_size` mixing coefficients over a bank of static
    # 64x64 square-pair templates (`gab_shared_weight`, shared by EVERY layer and
    # head). The bias is exactly  gab[h] = sum_i coeffs[h,i] * template_i.
    # These methods expose the pieces of that factorization, on the models that
    # have it (`has_generated_bias`).

    def _gab(self):
        gab = self.adapter.extensions().get("gab")
        if gab is None:
            raise RuntimeError(f"{self.arch.display_name} has no generated attention bias "
                               "(GAB / smolgen)")
        return gab

    @staticmethod
    def _check_recon(recon, target, what):
        """The decomposition sanity checks compare a re-derived tensor with the
        model's own, relative to the target's norm (see adapters.smolgen)."""
        check_recon(recon, target, what)

    @torch.no_grad()
    def gab_templates(self):
        """The static square-pair template bank behind every GAB: (gen_size, 64, 64).

        Template i is gab_shared_weight[:, i] reshaped so that template[i][q][k]
        is its contribution to query square q attending to key square k (canonical
        side-to-move frame, square = rank*8 + file). Position-independent and
        shared across all layers and heads — this is the model's entire geometric
        vocabulary. Computed once and cached; the same tensor is returned each
        call, so `.clone()` before mutating."""
        return self._gab().templates()

    @torch.no_grad()
    def gab_coeffs(self, board, self_elo=None, oppo_elo=None, layer=0):
        """The smolgen mixing coefficients of one layer for this position:
        (H, gen_size). Row h holds the weights with which head h mixes the
        static `gab_templates()` into its 64x64 bias:
            gab_bias(layer, h) == (coeffs[h, :, None, None] * gab_templates()).sum(0)
        (verified internally). This is the model choosing its geometry live."""
        gab = self._gab()
        x = self._block_input(board, self_elo, oppo_elo, layer)
        return gab.coeffs(int(layer), x)[0].cpu()

    @torch.no_grad()
    def gab_bias(self, board, self_elo=None, oppo_elo=None, layer=0, head=None):
        """The generated geometric attention bias of one layer for this position:
        (H, 64, 64), or (64, 64) for a single `head`. bias[h][q][k] is added to
        the scaled QK^T logit of query q, key k before the softmax."""
        gab = self._gab()
        x = self._block_input(board, self_elo, oppo_elo, layer)
        g = gab.bias(int(layer), x)[0].cpu()
        return g if head is None else g[int(head)]

    @torch.no_grad()
    def qk_scores(self, board, self_elo=None, oppo_elo=None, layer=0, head=None):
        """The content term of attention — scaled QK^T logits — of one layer
        for this position: (H, tokens, tokens), or one `head`'s matrix.
        softmax of it plus the other terms (gab_bias on smolgen models) is the
        attention the model actually runs."""
        _, _, dec = self._decompose(board, self_elo, oppo_elo, layer)
        qk = dec.terms["content"][0].cpu()
        return qk if head is None else qk[int(head)]

    # ----- per-head attention writes (for true head ablation) ---------------
    @torch.no_grad()
    def head_writes(self, board, self_elo=None, oppo_elo=None, layer=0):
        """Exact per-head residual-stream writes of one layer's attention:
        (H, tokens, dim).

        Head h's write is (A_h V_h) W_O^{(h)} — its attention-weighted values
        pushed through its own dh-column block of the output projection. This
        is the tensor a true head ablation must subtract: the hooked attn_NN
        activation is post-projection, where the heads are already mixed across
        every channel. Recomputed from the layer's own weights (same approach
        as `attention()`), and asserted on the spot to sum back — with the
        output bias — to this forward's attn_NN activation."""
        cond = Conditioning.coerce(self_elo, oppo_elo)
        x = self._block_input(board, cond, None, layer)
        return self._head_writes_from_x(x, layer, cond)[0].cpu()

    @torch.no_grad()
    def _head_writes_from_x(self, x, layer, cond=None):
        """`head_writes`' algebra over a whole batch: `x` is the (B, tokens, dim)
        attention input of `layer`, the result is (B, H, tokens, dim) on the
        live device. Batch of one is the single-position case, so `head_writes`
        is this plus the forward pass that produces `x`."""
        L = int(layer)
        dec = self.adapter.decompose_attention(L, x, cond if cond is not None else Conditioning())
        writes = dec.head_writes()                             # (B, H, T, dim)
        recon = writes.sum(1)
        if dec.b_O is not None:
            recon = recon + dec.b_O
        target = self._snap(f"attn_{L:02d}").to(self.device)
        check_recon(recon, target, f"head_writes reconstruction failed for layer {L}")
        return writes

    def ablate_head(self, board, self_elo, layer, head, oppo_elo=None,
                    return_type: str = "policy"):
        """Forward pass with one attention head's write removed, exactly.

        Upstream of `layer` is untouched by the ablation, so the write computed
        from a clean pass is exactly the write the ablated pass would have
        produced; we subtract it from attn_NN via run_with_hooks, and
        downstream layers react to the head's absence normally.

        `return_type` is passed straight to run_with_hooks. Watch the argument
        order: `layer`/`head` come before `oppo_elo` here, unlike the other
        methods in this file."""
        w = self.head_writes(board, self_elo, oppo_elo, layer)[int(head)]

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"attn_{int(layer):02d}", sub)],
                                   return_type=return_type)

    def steer_head(self, board, self_elo, layer, head, scale, oppo_elo=None,
                   return_type: str = "policy"):
        """Forward pass with one attention head's write multiplied by `scale`
        (activation steering): 1 is the clean model, 0 is `ablate_head`, 2
        doubles the head's say, -1 flips it. The (1 - scale) share of the
        head's exact write is subtracted from attn_NN, as in `ablate_head`."""
        w = self.head_writes(board, self_elo, oppo_elo, layer)[int(head)] * (1.0 - float(scale))

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"attn_{int(layer):02d}", sub)],
                                   return_type=return_type)

    # ----- residual-stream evolution across depth ---------------------------
    @torch.no_grad()
    def residual_stream(self, board, self_elo=None, oppo_elo=None):
        """Two per-token views of how the residual stream is built up (on the
        square-token models: per square, side-to-move frame, square =
        rank*8 + file):

          delta = the per-token magnitude of the vector each structure writes
                  into the stream, in execution order: the input embedding
                  (`emb`), then for every layer the self-attention write
                  (`aN` = ||sa_out||) and the feed-forward write
                  (`mN` = ||ff_out||). Each entry is tagged with its `kind`
                  ('emb'/'attn'/'mlp') so the UI can mark what is writing at
                  each point. 1 + 2·num_blocks entries.

          moves = logit lens on the running residual stream at every readout
                  point (see `_lens_steps`): emb, then per layer the
                  post-attention and post-MLP points, then a final `enc`
                  readout where there is one. Decode each through the policy
                  head, take the top legal move, and watch the prediction form
                  sub-layer by sub-layer. The `logit_lens` caveat applies.

        `delta` is a list of {label, kind, norm:[tokens]}; `moves` is a list of
        {label, kind, from, to, uci, san, piece} (from/to canonical squares,
        uci/san real-board, piece the moving piece's symbol); `layout` names
        the tokens."""
        self.evaluate(board, self_elo, oppo_elo)         # populates activations + logits
        nb = self.arch.n_layers

        # ---- delta: the vector each structure adds to the stream, in order ----
        def per_sq_norm(name):
            return self._snap(name)[0].norm(dim=-1).tolist()   # (tokens,)

        delta = [{"label": "emb", "kind": "emb", "norm": per_sq_norm("embed_in")}]
        for i in range(nb):
            delta.append({"label": f"a{i}", "kind": "attn",
                          "norm": per_sq_norm(f"attn_{i:02d}")})
            delta.append({"label": f"m{i}", "kind": "mlp",
                          "norm": per_sq_norm(f"mlp_{i:02d}")})

        # ---- moves: per-sub-layer logit lens on the running residual stream ----
        # Same resolution as delta: emb, then (post-attn, post-mlp) per layer, enc.
        legal = self._legal_mask(board).to(self.device)
        moves = [{"label": lab, "kind": kind,
                  **self._lens_move(self._snap(name)[0], board, legal)}
                 for name, lab, kind in self._lens_steps()]

        return {"delta": delta, "moves": moves, "layout": self.layout(board).to_json()}

    # ----- skill diff on internals ------------------------------------------
    def _lens_steps(self):
        """The readout points of the running residual stream, in order: emb, then
        per layer the post-attention and post-MLP points, then — when the model
        has a final norm the heads read through — enc. So 2·num_blocks + 2 of
        them on Maia-3 (18 on every size), 2·num_blocks + 1 on BT4 (31). The
        labels are the app-wide depth names: `aN`/`mN` for layer N's attention
        and MLP sub-layers (see interp_plot._depth_label)."""
        steps = [("embed_in", "emb", "emb")]
        for i in range(self.arch.n_layers):
            steps.append((f"postattn_{i:02d}", f"a{i}", "attn"))
            steps.append((f"block_{i:02d}",    f"m{i}", "mlp"))
        if "encoder_out" in self.hook_points:
            steps.append(("encoder_out", "enc", "enc"))
        return steps

    def _lens_move(self, activation, board, legal_mask):
        """Top legal move of one residual snapshot through the policy head."""
        logits = self.adapter.unembed(activation.to(self.device))
        if bool(legal_mask.any()):
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        idx = int(torch.argmax(logits))
        frm, to = self.vocab.squares(idx)
        mv = self._idx_to_move(board, idx)
        pc = board.piece_at(mv.from_square) if mv is not None else None
        return {"from": frm, "to": to, "uci": mv.uci() if mv else None,
                "san": board.san(mv) if mv is not None else None,
                "piece": pc.symbol() if pc is not None else None}

    @torch.no_grad()
    def compare_residual(self, board, elo_a, elo_b):
        """Skill diff on internals, not just outputs: run the same position at
        two ratings (ints, or Conditionings) and, at each readout point of the
        running residual stream (see `_lens_steps`), report

          norm     = per-token ||x_A − x_B|| — where on the board, and at
                     what depth, the two skill levels diverge
          move_a/b = logit-lens top legal move of each run at that point
          same     = whether the two lenses agree

        On Maia-3, Elo enters as an embedding concatenated to every square token
        before token_projection, so at `emb` the diff is one constant "skill
        vector" repeated on all 64 squares; the interesting structure is how
        depth localizes it. Both runs use oppo_elo == self_elo, so both ratings'
        embeddings move — read the result as the diff between two whole skill
        settings, not one player's. The `logit_lens` caveat applies to
        move_a/move_b. On a model without a rating input (LeelaEngine) both
        runs are identical and every norm is zero."""
        _, cache_a = self.run_with_cache(board, Conditioning.coerce(elo_a))
        _, cache_b = self.run_with_cache(board, Conditioning.coerce(elo_b))
        legal = self._legal_mask(board).to(self.device)
        steps = []
        for name, lab, kind in self._lens_steps():
            xa, xb = cache_a[name], cache_b[name]
            ma = self._lens_move(xa, board, legal)
            mb = self._lens_move(xb, board, legal)
            steps.append({
                "label": lab, "kind": kind,
                "norm": (xa - xb).norm(dim=-1).tolist(),
                "move_a": ma, "move_b": mb,
                "same": ma["uci"] == mb["uci"],
            })
        return {"elo_a": _elo_out(elo_a), "elo_b": _elo_out(elo_b), "steps": steps}

    # ----- move notation ----------------------------------------------------
    # Five representations of the same move circulate in this file, and they are
    # easy to mix up:
    #
    #   Move         chess.Move             real board (python-chess)
    #   uci          "e2e4", "e7e8q"        real board
    #   san          "Nf3", "exd5"          real board, only readable with one
    #   idx          0 .. n_moves-1         policy index — in the model's frame
    #                                       (side-to-move: Black mirrored)
    #   (from, to)   0 .. 63 each           canonical squares, rank*8 + file, in
    #                                       that same frame
    #
    # The first three are what a human types, the last two are what the model
    # thinks in, and the mirror sits between them. Every `from`/`to` this file
    # returns is canonical; every `uci`/`san` is real-board. `to_move()` reads
    # any of the five, `move_info()` returns all five at once — go through those
    # instead of hand-rolling the mirror.

    @staticmethod
    def _canon_square(square: int, turn: bool) -> int:
        """python-chess square -> canonical index (side-to-move frame,
        rank*8 + file): identity for White, vertically mirrored for Black.
        Mirrored in ui.py's realToCanon() for the JS side."""
        return canon_square(square, turn)

    @staticmethod
    def _real_square(canon: int, turn: bool) -> int:
        """Canonical index -> python-chess square. Inverse of `_canon_square`."""
        return real_square(canon, turn)

    def _frame_square(self, canon: int, board: chess.Board) -> int:
        """A square in the model's frame -> the python-chess square on `board`."""
        return (real_square(canon, board.turn) if self.vocab.frame == "stm_mirror"
                else int(canon))

    def to_move(self, board: chess.Board, move) -> chess.Move:
        """Any of the five forms above -> chess.Move on `board`: a chess.Move, a
        uci string, a SAN string, a policy index, or a (from, to) pair of
        canonical squares. Raises ValueError on anything unreadable.

        A (from, to) pair carries no promotion piece — the policy layout folds
        every rank7->rank8 promotion onto the same square pair — so it resolves
        to whichever legal move matches those squares.

        Castling written as king-takes-rook (e1h1, lc0's own encoding) comes
        back as the standard e1g1: python-chess accepts both as legal but they
        are different Move objects, and every uci this package returns is the
        standard one."""
        if isinstance(move, chess.Move):
            return self._standard_castling(board, move)
        if isinstance(move, str):
            try:
                return self._standard_castling(board, chess.Move.from_uci(move))
            except ValueError:
                pass
            try:
                return board.parse_san(move)
            except ValueError as e:
                raise ValueError(f"neither uci nor SAN on this board: {move!r}") from e
        if isinstance(move, (tuple, list)):
            frm, to = (self._frame_square(s, board) for s in move)
            mv = chess.Move(frm, to)
            if mv in board.legal_moves:
                return self._standard_castling(board, mv)
            promo = next((m for m in board.legal_moves
                          if m.from_square == frm and m.to_square == to), None)
            if promo is None:
                raise ValueError("no legal move "
                                 f"{chess.square_name(frm)}{chess.square_name(to)}")
            return promo
        idx = int(move)
        if idx not in self.idx_to_move:
            raise ValueError(f"policy index out of range: {idx}")
        return self._decode_idx(board, idx)

    @staticmethod
    def _standard_castling(board: chess.Board, mv: chess.Move) -> chess.Move:
        """A legal king-takes-rook castling move -> the standard king-two-squares
        form (e1h1 -> e1g1); any other move unchanged."""
        if mv.promotion is None and board.is_castling(mv) and mv in board.legal_moves:
            return board.parse_uci(mv.uci())
        return mv

    def _move_index(self, board: chess.Board, move) -> int:
        """Policy index of a move on this board — this is where the Black mirror
        is applied. Takes any form `to_move` does. Raises KeyError if the move
        isn't in the policy vocabulary (a null move, an under-promotion the
        head doesn't encode)."""
        return self.vocab.index(board, self.to_move(board, move))

    def move_info(self, board: chess.Board, move) -> dict:
        """One move in every representation at once — the table above as a dict,
        and the single call to reach for when converting:

          move         chess.Move
          uci, san     real-board strings (`san` falls back to uci if illegal)
          idx          policy index, the model's frame
          from, to     canonical squares of that index — the frame the lens
                       dicts, heatmaps and `residual_stream` are indexed by
          from_sq, to_sq   the same two squares as python-chess squares
          piece        symbol of the moving piece, None on an empty from-square
          legal        whether the move is legal in this position

        Reads any form `to_move` does, so it converts in every direction:
        `eng.move_info(board, "Nf3")["idx"]`, `eng.move_info(board, 1234)["san"]`."""
        mv = self.to_move(board, move)
        idx = self._move_index(board, mv)
        frm, to = self.vocab.squares(idx)
        pc = board.piece_at(mv.from_square)
        legal = mv in board.legal_moves
        return {"move": mv, "uci": mv.uci(),
                "san": board.san(mv) if legal else mv.uci(),
                "idx": idx, "from": frm, "to": to,
                "from_sq": mv.from_square, "to_sq": mv.to_square,
                "piece": pc.symbol() if pc is not None else None,
                "legal": legal}

    # ----- move-centric lenses ----------------------------------------------

    def depth_points(self):
        """The readout points as [{label, kind}] in depth order — the x axis
        every *_per_depth curve below is indexed by, without spending a forward
        pass to get it. Same list, same order, as the `steps` of
        `move_logit_lens` and `compare_residual`."""
        return [{"label": lab, "kind": kind} for _, lab, kind in self._lens_steps()]

    @torch.no_grad()
    def move_logit_lens(self, board, self_elo, uci, oppo_elo=None):
        """One move's depth curve: the logit lens applied to a single chosen
        move at every readout point of the residual stream. This is where a
        move "snaps" into the plan — watch its logit, its probability over
        legal moves, and its rank (1 = currently the top move) across depth.

        `uci` is any form `to_move` reads (uci, SAN, chess.Move, policy index,
        canonical (from, to)); the returned `uci` is always the real-board one.

        Returns {uci, san, steps: [{label, kind, logit, prob, rank}], n_legal}.
        One forward pass for all three curves — the `*_per_depth` helpers below
        are views on this. The `logit_lens` caveat applies."""
        self.evaluate(board, self_elo, oppo_elo)
        info = self.move_info(board, uci)
        idx = info["idx"]
        legal = self._legal_mask(board).to(self.device)
        n_legal = int(legal.sum())
        steps = []
        for name, lab, kind in self._lens_steps():
            logits = self.adapter.unembed(self._snap(name)[0].to(self.device))
            lg = float(logits[idx])
            masked = logits.masked_fill(~legal, float("-inf"))
            prob = float(torch.softmax(masked, dim=-1)[idx]) if n_legal else None
            rank = int((masked > masked[idx]).sum()) + 1 if n_legal else None
            steps.append({"label": lab, "kind": kind,
                          "logit": lg, "prob": prob, "rank": rank})
        return {"uci": info["uci"], "san": info["san"],
                "steps": steps, "n_legal": n_legal}

    # Bare-list views of that curve, for plotting and for arithmetic on depth.
    # Each is one forward pass, so take `move_logit_lens` directly when you want
    # more than one of them. Indexed by `depth_points()`.
    def logit_per_depth(self, board, self_elo, move, oppo_elo=None) -> list[float]:
        """This move's raw policy logit at every readout point (unmasked, so it
        is comparable across positions in a way the probability is not)."""
        return [s["logit"] for s in
                self.move_logit_lens(board, self_elo, move, oppo_elo)["steps"]]

    def policy_per_depth(self, board, self_elo, move, oppo_elo=None) -> list[float]:
        """This move's probability at every readout point — softmax over the
        legal moves only, i.e. the policy the app would show if the stream
        stopped there. All None in a position with no legal moves."""
        return [s["prob"] for s in
                self.move_logit_lens(board, self_elo, move, oppo_elo)["steps"]]

    def rank_per_depth(self, board, self_elo, move, oppo_elo=None) -> list[int]:
        """This move's rank among the legal moves at every readout point,
        1 = the top move. The step where it reaches 1 and stays is the snap."""
        return [s["rank"] for s in
                self.move_logit_lens(board, self_elo, move, oppo_elo)["steps"]]

    @torch.no_grad()
    def ablate_grid(self, board, self_elo, uci, oppo_elo=None):
        """The carrier heatmap of one move: ablate every attention head
        (exactly, via head_writes) and record what that did to the move's
        logit. Sign convention throughout the app: delta = ablated − clean,
        so negative means the head was supporting the move (removing it
        hurts) and positive means it suppresses it.

        Returns {uci, san, base_logit, deltas: (num_blocks, num_heads) nested
        list}. Costs ~num_blocks·(num_heads+1) forward passes — seconds, not
        milliseconds. For many positions, `ablate_grid_batch` runs that same
        count per batch rather than per position."""
        return self.ablate_grid_batch([(board, uci)], self_elo, oppo_elo)[0]

    @torch.no_grad()
    def ablate_grid_batch(self, items, self_elo=None, oppo_elo=None, batch_size=32):
        """`ablate_grid` over many positions: `items` is [(board, move)] and the
        result is one ablate_grid dict per item, in order.

        The arithmetic is the single-position one, position by position — the
        positions are independent, so ablating a head is the same hook over a
        stacked forward. What changes is the bookkeeping: a batch costs
        ~num_blocks·(num_heads+1) forward passes in total instead of that many
        each, and a batched forward is several times cheaper per position than a
        batch of one. `batch_size` trades that against memory — the per-layer
        head writes are (batch_size, H, tokens, dim)."""
        cond = Conditioning.coerce(self_elo, oppo_elo)
        nb, nh = self.arch.n_layers, self.arch.n_heads
        out = []
        # This method runs nb·(nh+1)+1 forward passes per chunk and reads only two
        # of the captured tensors, both of which it sends straight back to the
        # device. Leaving the capture on-device skips that round trip; on a
        # discrete GPU it is the difference between a sweep that is transfer-bound
        # and one that is compute-bound.
        self._capture_on_device = True
        try:
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                boards = [b for b, _ in chunk]
                infos = [self.move_info(b, m) for b, m in chunk]
                rows = torch.arange(len(chunk), device=self.device)
                idx = torch.tensor([i["idx"] for i in infos], device=self.device)

                base = self._forward_batch(boards, cond)[rows, idx]   # (B,)
                # accumulated on-device and synced once below, so consecutive head
                # passes pipeline instead of draining the launch queue each time
                deltas = torch.zeros(len(chunk), nb, nh, device=self.device)
                for L in range(nb):
                    # the clean pass this layer's writes are read off; also what the
                    # ablated passes below are compared against, so it is re-run per
                    # layer rather than cached (an ablated forward overwrites it)
                    self._forward_batch(boards, cond)
                    x = self._snap(self.adapter.attn_input_key(L)).to(self.device)
                    writes = self._head_writes_from_x(x, L, cond)
                    for h in range(nh):
                        w = writes[:, h]
                        handle = self.hook_points[f"attn_{L:02d}"].register_forward_hook(
                            self._patch_hook(lambda a, w=w: a - w.to(a.device, a.dtype)))
                        try:
                            abl = self._forward_batch(boards, cond)[rows, idx]
                        finally:
                            handle.remove()
                        deltas[:, L, h] = abl - base      # ablated − clean
                deltas = deltas.cpu()

                out += [{"uci": info["uci"], "san": info["san"],
                         "base_logit": float(base[r]), "deltas": deltas[r].tolist()}
                        for r, info in enumerate(infos)]
        finally:
            self._capture_on_device = False
        return out


    # ----- neurons: activation, exact ablation, carrier table ---------------
    def _mlp_hidden(self, board, self_elo=None, oppo_elo=None, layers=None):
        """One forward pass; returns {L: (tokens, ff)} — the MLP hidden
        activation (the input of the MLP's output projection, post-activation)
        of each requested layer, on the live device, detached. `hook_points`
        holds that projection as mlp_NN, so the hidden is its input."""
        want = range(self.arch.n_layers) if layers is None else list(layers)
        hidden = {}

        def keep(L):
            def hook(_module, inp, _out):
                hidden[L] = inp[0].detach()[0]
            return hook

        handles = [self.adapter.mlp_out(L).register_forward_hook(keep(L)) for L in want]
        try:
            with torch.no_grad():
                self._forward(board, self_elo, oppo_elo)
        finally:
            for h in handles:
                h.remove()
        return hidden

    def _neuron_write(self, layer, neuron, hidden):
        """The (tokens, dim) vector one MLP unit adds to the stream: h[s, n] · W2[:, n]."""
        w2 = self.adapter.mlp_out(int(layer)).weight[:, int(neuron)]
        return hidden[:, int(neuron), None] * w2[None, :]

    @torch.no_grad()
    def neuron_activation(self, board, self_elo, layer, neuron, oppo_elo=None):
        """One MLP unit on this position: {layer, neuron, n_neurons, act: [tokens]
        (its post-activation value on every token — every square, canonical
        frame, on the square-token models), write_norm: ‖h[:, n]‖·‖W2[:, n]‖,
        the size of what it adds to the stream}. Argument order as
        `ablate_head`: layer/neuron before oppo_elo."""
        L, n = int(layer), int(neuron)
        h = self._mlp_hidden(board, self_elo, oppo_elo, layers=[L])[L]
        w2 = self.adapter.mlp_out(L).weight[:, n]
        return {"layer": L, "neuron": n, "n_neurons": int(h.size(1)),
                "act": h[:, n].float().cpu().tolist(),
                "write_norm": float(h[:, n].norm() * w2.norm())}

    @torch.no_grad()
    def neuron_overview(self, board, self_elo=None, oppo_elo=None, k=12):
        """Every layer's `k` most active MLP units on this position, by their
        peak activation over the tokens (the score `top_neurons` uses):
        {n_layers, n_neurons, layers: [[{neuron, peak}, …] per layer, strongest
        first]}. One forward pass; the app's network diagram is drawn from this."""
        hidden = self._mlp_hidden(board, self_elo, oppo_elo)
        layers = []
        for L in sorted(hidden):
            peaks = hidden[L].float().amax(dim=0)                  # (ff,)
            top = peaks.topk(min(int(k), peaks.numel()))
            layers.append([{"neuron": int(n), "peak": float(v)}
                           for v, n in zip(top.values.tolist(), top.indices.tolist())])
        return {"n_layers": len(layers), "n_neurons": int(hidden[0].size(1)), "layers": layers}

    def ablate_neuron(self, board, self_elo, layer, neuron, oppo_elo=None,
                      return_type: str = "policy"):
        """Forward pass with one MLP unit's write removed exactly, on every
        token — the neuron-grain twin of `ablate_head`: the unit's write
        h[s,n]·W2[:,n] is subtracted from mlp_NN via run_with_hooks and the
        layers downstream react normally. Same argument order as `ablate_head`."""
        L = int(layer)
        w = self._neuron_write(L, neuron, self._mlp_hidden(board, self_elo, oppo_elo, layers=[L])[L])

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"mlp_{L:02d}", sub)], return_type=return_type)

    def steer_neuron(self, board, self_elo, layer, neuron, scale, oppo_elo=None,
                     return_type: str = "policy"):
        """Forward pass with one MLP unit's write multiplied by `scale` on every
        token (activation steering): 1 is the clean model, 0 is
        `ablate_neuron`, 2 doubles it, -1 flips it."""
        L = int(layer)
        w = self._neuron_write(L, neuron, self._mlp_hidden(board, self_elo, oppo_elo, layers=[L])[L]) \
            * (1.0 - float(scale))

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"mlp_{L:02d}", sub)], return_type=return_type)

    def carrier_neurons(self, board, self_elo, move, oppo_elo=None, *,
                        top_k=12, skip_last_layer=True, verify=False):
        """The carrier table of one move at neuron grain: which MLP units, on
        which tokens, the move's logit rests on. The neuron-grain counterpart
        of `ablate_grid`, using the same causal method as `head_writes`/
        `ablate_head` — a real intervention, not a correlational read of
        activations.

        Every unit is scored at once by attribution patching: one backward pass giving,
        for every hidden unit h of every MLP (layer L, token s, neuron n), the first-order estimate
        of the logit change from zeroing it, Δ ≈ −(∂logit/∂h)·h. """
        info = self.move_info(board, move)
        idx = info["idx"]
        nb = self.arch.n_layers

        # capture every MLP's hidden activation (the input of its output projection)
        hidden: dict[int, torch.Tensor] = {}

        def keep(L):
            def hook(_module, inp, _out):
                hidden[L] = inp[0]
            return hook

        handles = [self.adapter.mlp_out(L).register_forward_hook(keep(L)) for L in range(nb)]
        try:
            with torch.enable_grad():
                logits, _ = self._forward(board, self_elo, oppo_elo)
                grads = torch.autograd.grad(logits[idx], [hidden[L] for L in range(nb)])
        finally:
            for h in handles:
                h.remove()
        base = float(logits[idx].detach())
        # Δ ≈ −grad·h for zeroing one unit on one token: (nb, tokens, ff)
        est = torch.stack([-(g[0] * hidden[L][0]) for L, g in enumerate(grads)]).detach().float().cpu()
        hidden = {L: h[0].detach() for L, h in hidden.items()}

        per_neuron = est.sum(1)                                   # (nb, ff)
        ff = per_neuron.size(1)
        rank = per_neuron.abs().clone()
        if skip_last_layer and nb > 1:
            rank[nb - 1] = -1.0
        top = []
        pool = min((2 * top_k) if verify else top_k, rank.numel())
        for flat in rank.flatten().topk(pool).indices.tolist():
            L, n = divmod(flat, ff)
            exact = None
            if verify:
                write = self._neuron_write(L, n, hidden[L])

                def sub(act, write=write):
                    return act - write.to(act.device, act.dtype)

                lm, _ = self.run_with_hooks(board, self_elo, oppo_elo,
                                            fwd_hooks=[(f"mlp_{L:02d}", sub)], return_type="logits")
                exact = float(lm[idx]) - base
            sq = est[L, :, n]
            top.append({"layer": L, "neuron": n, "est": float(per_neuron[L, n]), "exact": exact,
                        "squares": sq.tolist(), "peak": int(sq.abs().argmax())})
        if verify:
            top.sort(key=lambda t: -abs(t["exact"]))
            top = top[:top_k]
        return {"uci": info["uci"], "san": info["san"], "base_logit": base,
                "n_layers": nb, "n_neurons": ff,
                "layer_abs": per_neuron.abs().sum(1).tolist(), "top": top}

    # ----- many positions (the helpers live below the class) ---------------
    def _layer_ids(self, layers):
        """Layer indices, checked: all of them if None."""
        count = self.arch.n_layers
        layers = list(range(count)) if layers is None else [int(L) for L in layers]
        if not layers:
            raise ValueError("no layers")
        bad = [L for L in layers if not 0 <= L < count]
        if bad:
            raise ValueError(f"no layer(s) {bad}: this model has layers 0-{count - 1}")
        return layers

    def _mlp_hidden_batches(self, boards, layers, elo, oppo_elo, batch_size):
        """Yield (start, {layer: (B, 64, ff) post-activation MLP hidden}) per batch."""
        for start in range(0, len(boards), batch_size):
            got, handles = {}, []

            def keep(L):
                def hook(_module, inp, _out):
                    got[L] = inp[0].detach().float()
                return hook

            try:
                for L in layers:
                    handles.append(self.adapter.mlp_out(L).register_forward_hook(keep(L)))
                with torch.no_grad(), self.no_capture():
                    self.adapter.forward(boards[start:start + batch_size], Conditioning.coerce(elo, oppo_elo))
            finally:
                for h in handles:
                    h.remove()
            yield start, got

    def _ratings(self, elo, oppo_elo):
        """(elo, oppo_elo) as used, or (None, None) for a model with no rating input."""
        if not self.has_conditioning:
            return None, None
        return int(elo), int(elo if oppo_elo is None else oppo_elo)

    def top_neurons(self, positions, k=10, *, layers=None, elo=1500, oppo_elo=None,
                    batch_size=32, n_best=5) -> NeuronReport:
        """The MLP neurons that fire most on `positions` (FENs, boards, or a PGN
        path, as `load_positions` reads them): for every layer, the `k` neurons
        whose peak activation over the 64 squares, averaged over the positions, is
        highest. Raw activations, no baseline. `elo` is the rating Maia-3 is
        conditioned on (both sides unless `oppo_elo`); Leela ignores both. Memory:
        one float per position, layer and neuron (BT4: ~92 KB a position).
        `NeuronScan(eng, ...)` does the same a batch at a time."""
        return NeuronScan(self, positions, layers=layers, elo=elo, oppo_elo=oppo_elo,
                          batch_size=batch_size).run().report(k, n_best)

    @torch.no_grad()
    def top_positions(self, layer, neuron, positions=None, k=8, *, elo=1500, oppo_elo=None,
                      batch_size=32) -> list:
        """The `k` positions (all of them if k is None) that make one MLP neuron
        fire most: from `positions` (as `load_positions` reads them), or the bundled
        `ordinary_positions()`. Strongest first, each {index, fen, board, peak,
        square (the real square it peaks on), peak_canon (the same, a model-frame
        index), act (its 64 values, model-frame order)}. `elo`/`oppo_elo` as in
        `top_neurons`."""
        boards = ordinary_positions() if positions is None else load_positions(positions)
        if not boards:
            raise ValueError("no positions")
        L = self._layer_ids([layer])[0]
        u = int(neuron)
        ff = self.arch.d_mlp
        if not 0 <= u < ff:
            raise ValueError(f"no neuron {u}: layer {L} has neurons 0-{ff - 1}")
        if k is not None and int(k) < 1:
            raise ValueError("k must be at least 1, or None for every position")
        elo, oppo_elo = self._ratings(elo, oppo_elo)
        acts = torch.zeros(len(boards), 64)
        for start, got in self._mlp_hidden_batches(boards, [L], elo or 0, oppo_elo or 0, batch_size):
            h = got[L][:, :, u].cpu()
            acts[start:start + h.size(0)] = h
        peak, where = acts.max(dim=1)
        out = []
        for i in peak.argsort(descending=True)[:None if k is None else int(k)].tolist():
            b, canon = boards[i], int(where[i])
            out.append({"index": i, "fen": b.fen(), "board": b, "peak": float(peak[i]),
                        "square": chess.square_name(self._frame_square(canon, b)),
                        "peak_canon": canon, "act": acts[i].tolist()})
        return out

    @torch.no_grad()
    def neuron_effects(self, layer, neuron, positions, *, k=3, elo=1500, oppo_elo=None,
                       batch_size=32) -> list:
        """What one MLP neuron does on each of `positions`, exactly: the policy with
        its write removed on every square (as `ablate_neuron` removes it, the later
        layers reacting normally) against the clean policy. One dict per position:
        {fen, top (the clean top move), p_top, top_ablated, p_top_ablated, changed
        (the top move changes), supports: [(uci, Δp)] the `k` moves it raises
        most, opposes: the `k` it lowers most}, Δp = p(clean) − p(without it).
        The activation snapshot is left alone."""
        boards = load_positions(positions)
        if not boards:
            raise ValueError("no positions")
        L = self._layer_ids([layer])[0]
        u = int(neuron)
        lin2 = self.adapter.mlp_out(L)
        if not 0 <= u < lin2.in_features:
            raise ValueError(f"no neuron {u}: layer {L} has neurons 0-{lin2.in_features - 1}")
        elo, oppo_elo = self._ratings(elo, oppo_elo)
        elos = (elo or 0, oppo_elo or 0)
        w2 = lin2.weight[:, u].float()
        out = []
        for start, got in self._mlp_hidden_batches(boards, [L], *elos, batch_size):
            chunk = boards[start:start + batch_size]
            write = got[L][:, :, u, None] * w2[None, None, :]          # (B, 64, dim)
            clean = self.evaluate_batch(chunk, *elos)
            handle = self.hook_points[f"mlp_{L:02d}"].register_forward_hook(
                self._patch_hook(lambda a: a - write.to(a.dtype)))
            try:
                ablated = self.evaluate_batch(chunk, *elos)
            finally:
                handle.remove()
            for b, c, a in zip(chunk, clean, ablated):
                pc, pa = dict(c["policy"]), dict(a["policy"])
                delta = sorted(((m, pc[m] - pa.get(m, 0.0)) for m in pc), key=lambda t: -t[1])
                top, top_a = c["policy"][0][0], a["policy"][0][0]
                out.append({"fen": b.fen(), "top": top, "p_top": pc[top], "top_ablated": top_a,
                            "p_top_ablated": pa[top_a], "changed": top != top_a,
                            "supports": [d for d in delta[:k] if d[1] > 0],
                            "opposes": [d for d in delta[::-1][:k] if d[1] < 0]})
        return out

    # ----- activation dump --------------------------------------------------
    def save_activations(self, filename: str, meta: dict | None = None) -> str:
        """Persist the most recent forward's residual-stream snapshot, plus a
        `meta` entry. Each tensor is (tokens, d_model), on CPU. Keys are every
        name in `hook_points` — see `_register_hooks` for what each one is.

        `activation_dir` is created here, on the first save, rather than in
        __init__."""
        snap = {k: self._snap(k).squeeze(0).clone() for k in self._activations}
        snap["meta"] = meta or {}
        self.activation_dir.mkdir(parents=True, exist_ok=True)
        path = self.activation_dir / filename
        torch.save(snap, path)
        return str(path)


# ----- many positions -------------------------------------------------------
# Positions in, neurons out: `eng.top_neurons`, `eng.top_positions`, `eng.neuron_effects`.
# A position's score for a neuron is its peak activation over the 64 squares;
# averages over many positions are in the model's frame (side to move at the bottom).

def _board(item) -> chess.Board:
    if isinstance(item, chess.Board):
        return item.copy()
    if isinstance(item, str):
        return chess.Board(item.strip())
    raise TypeError(f"a position is a FEN string or a chess.Board, not {type(item).__name__}")


def load_positions(source, n=None, *, seed=0, **pgn_options) -> list:
    """A list of boards from `source`: FEN strings and/or chess.Boards (copied,
    history kept; any iterable), or the path to a PGN file, from which
    `positions_from_pgn` samples one position per game (`n` games, 1,000 if
    None; it takes `pgn_options`). For FENs and boards, `n` keeps the first n."""
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".pgn"):
        return positions_from_pgn(source, n if n is not None else 1000, seed=seed, **pgn_options)
    if isinstance(source, Path):
        raise ValueError(f"{source}: only .pgn files are read; or pass FEN strings or boards")
    if isinstance(source, (str, chess.Board)):
        source = [source]
    return [_board(x) for x in (source if n is None else itertools.islice(source, n))]


def positions_from_pgn(path, n=1000, *, seed=0, min_ply=12, max_ply=100, max_games=None):
    """One position per game, from `n` games sampled with `seed` among the first
    `max_games` standard games of the PGN file at `path` (all of them if None).
    In each game the position is a random ply in [min_ply, max_ply] with a move
    still to come; a position where the game is over anyway (fivefold
    repetition, the 75-move rule) is skipped. Boards keep the whole game as
    history."""
    rng = random.Random(seed)
    offsets = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        while max_games is None or len(offsets) < max_games:
            at = fh.tell()
            headers = chess.pgn.read_headers(fh)
            if headers is None:
                break
            if headers.get("Variant", "Standard").lower() in ("standard", "chess") \
                    and "FEN" not in headers:
                offsets.append(at)
        boards = []
        for at in rng.sample(offsets, len(offsets)):     # a seeded order over the games
            if len(boards) == n:
                break
            fh.seek(at)
            game = chess.pgn.read_game(fh)
            moves = list(game.mainline_moves()) if game is not None else []
            top = min(max_ply, len(moves) - 1)
            if top < min_ply:
                continue
            board = game.board()
            for mv in moves[:rng.randint(min_ply, top)]:
                board.push(mv)
            if not board.is_game_over():
                boards.append(board)
    return boards


def ordinary_positions(n=None) -> list:
    """The bundled pool of ordinary positions (ordinary_positions.txt says
    where they come from), each with the 8 plies before it as history. The
    first `n`, or all."""
    text = (Path(__file__).parent / "ordinary_positions.txt").read_text()
    boards = []
    for line in [l for l in text.splitlines() if l and l[0] != "#"][:n]:
        fen, _, moves = line.partition("|")
        b = chess.Board(fen.strip())
        for u in moves.split():
            b.push_uci(u)
        boards.append(b)
    return boards


@dataclass
class NeuronReport:
    """`top_neurons`' answer. `rows` holds each layer's top neurons, layer by
    layer, strongest first: {layer, neuron, rank (1 = the layer's strongest),
    mean (its peak activation averaged over the positions), std, max,
    squares (64 per-square means, the model's frame), peak_canon (where that
    map peaks, a model-frame index), best (indices into `positions` where it
    peaks highest, strongest first)}. `peaks[i, j, n]` is position i's peak for
    neuron n of `layers[j]`; `peaks_of(layer, neuron)` reads one neuron's."""
    positions: list = field(repr=False)
    layers: list
    rows: list = field(repr=False)
    peaks: torch.Tensor = field(repr=False)
    elo: int | None = None
    oppo_elo: int | None = None

    def top(self, layer=None, n=None) -> list:
        """The rows of one layer (or all), the first `n`."""
        rows = [r for r in self.rows if layer is None or r["layer"] == layer]
        return rows[:n]

    def peaks_of(self, layer, neuron) -> list:
        """One neuron's peak on every position, in the order of `positions`."""
        return self.peaks[:, self.layers.index(int(layer)), int(neuron)].tolist()

    def table(self, n=3) -> str:
        """A short text table: each layer's `n` strongest neurons."""
        rating = "" if self.elo is None else f" · rating {self.elo}/{self.oppo_elo}"
        out = [f"{len(self.positions)} positions · peak activation per position, "
               f"averaged{rating}"]
        for L in self.layers:
            cells = [f"n{r['neuron']:<5} {r['mean']:6.2f}" for r in self.top(L, n)]
            out.append(f"L{L:<3}" + "   ".join(cells))
        return "\n".join(out)


class NeuronScan:
    """`top_neurons` a batch at a time, for a progress bar or stopping early:

        scan = NeuronScan(eng, positions)
        while not scan.finished:
            scan.step()                    # one batch
            print(scan.done, "/", scan.total)
        rep = scan.report(k=10)            # over the positions scanned so far

    Arguments as `top_neurons`."""

    def __init__(self, eng, positions, *, layers=None, elo=1500, oppo_elo=None,
                 batch_size=32):
        self.eng = eng
        self.positions = load_positions(positions)
        if not self.positions:
            raise ValueError("no positions")
        self.layers = eng._layer_ids(layers)
        self.elo, self.oppo_elo = eng._ratings(elo, oppo_elo)
        self.batch_size = int(batch_size)
        ff = eng.arch.d_mlp
        self.peaks = torch.zeros(len(self.positions), len(self.layers), ff)
        self._square_sum = torch.zeros(len(self.layers), 64, ff)
        self.done = 0

    @property
    def total(self):
        return len(self.positions)

    @property
    def finished(self):
        return self.done >= self.total

    @torch.no_grad()
    def step(self):
        """Run the next batch; returns how many positions it took."""
        if self.finished:
            return 0
        chunk = self.positions[self.done:self.done + self.batch_size]
        for _, got in self.eng._mlp_hidden_batches(chunk, self.layers, self.elo or 0, self.oppo_elo or 0,
                              len(chunk)):
            for j, L in enumerate(self.layers):
                h = got[L]                                 # (B, 64, ff), on the device
                self.peaks[self.done:self.done + h.size(0), j] = h.amax(dim=1).cpu()
                self._square_sum[j] += h.sum(dim=0).cpu()
        self.done += len(chunk)
        return len(chunk)

    def run(self):
        while not self.finished:
            self.step()
        return self

    def report(self, k=10, n_best=5) -> NeuronReport:
        """Each layer's `k` strongest neurons over the positions scanned so far."""
        if not self.done:
            raise ValueError("nothing scanned yet")
        peaks = self.peaks[:self.done]
        mean, std, mx = peaks.mean(0), peaks.std(0, unbiased=False), peaks.amax(0)
        squares = self._square_sum / self.done
        rows = []
        for j, L in enumerate(self.layers):
            top = mean[j].topk(min(int(k), mean.size(1)))
            for rank, n in enumerate(top.indices.tolist(), 1):
                sq = squares[j, :, n]
                order = peaks[:, j, n].argsort(descending=True)[:n_best]
                rows.append({"layer": L, "neuron": n, "rank": rank, "mean": float(mean[j, n]),
                             "std": float(std[j, n]), "max": float(mx[j, n]),
                             "squares": sq.tolist(), "peak_canon": int(sq.argmax()),
                             "best": order.tolist()})
        return NeuronReport(positions=self.positions[:self.done], layers=self.layers, rows=rows,
                            peaks=peaks, elo=self.elo, oppo_elo=self.oppo_elo)


# ----- the two named families ---------------------------------------------------
# Thin shims kept for the constructors existing code calls. Each defaults to its
# own family's model; nothing in the package defaults to a family.

class MaiaEngine(ChessformerEngine):
    """ChessformerEngine on a Maia-3 checkpoint (downloaded from Hugging Face on
    first use; needs the maia3 package). Conditioned on a rating pair;
    4352-move policy."""

    def __init__(self, alias="maia3-5m", device=None, checkpoint_path=None,
                 activation_dir="activations", trust_checkpoint=False):
        """`device`: an explicit string wins; otherwise see `pick_device`.
        `trust_checkpoint=True` loads with `weights_only=False`, i.e. it can
        execute pickled code from the checkpoint file — only use it for
        checkpoints you produced yourself."""
        from .adapters.maia3 import Maia3Adapter
        super().__init__(Maia3Adapter.load(alias, device=device, checkpoint_path=checkpoint_path,
                                           trust_checkpoint=trust_checkpoint), activation_dir)


class LeelaEngine(ChessformerEngine):
    """ChessformerEngine on Leela Chess Zero BT4, from lc0's .pb.gz (or an ONNX
    export).

    BT4 has no rating input: self_elo / oppo_elo are accepted and ignored, so
    the same calls work on every engine. No final norm, so the depth points
    end at m14 (no enc). evaluate()["mlh"] is the moves-left estimate in plies."""

    def __init__(self, alias="bt4", device=None, checkpoint_path=None,
                 activation_dir="activations"):
        """`alias` names an official net, `bt4` (the base BT4) or `bt4-332`
        (policytune-332), or is a path to an lc0 .pb.gz (or an .onnx export).
        Without `checkpoint_path` the net is looked for in $CHESSFORMER_WEIGHTS,
        ./weights and the cache, and downloaded from lczero.org on first use
        (CHESSFORMER_NO_DOWNLOAD=1 refuses instead)."""
        super().__init__(Lc0Adapter.load(alias, device=device, checkpoint_path=checkpoint_path),
                         activation_dir)

    def search(self, board, nodes=400, *, lc0=None, timeout=600.0, **options) -> dict:
        """What Leela's search makes of this position: the lc0 program run on the
        same net this engine loaded (lc0 from `lc0`, $CHESSFORMER_LC0 or PATH),
        `nodes` visits at the root. Returns {bestmove, pv (UCI), nodes, seconds,
        root: {q, wl, d, m} for the side to move, rows: [{uci, san, n (visits),
        share, p (lc0's prior, after its policy temperature), q}] most visited
        first}. `options` are lc0 UCI options, e.g. SmartPruningFactor=0.
        Raises RuntimeError without lc0."""
        import chess.engine
        exe = lc0 or find_lc0()
        if not exe or not (Path(exe).exists() or shutil.which(exe)):
            raise RuntimeError(f"lc0 not found ({exe or 'not on PATH'}): install it, e.g. brew install lc0, "
                               "or set $CHESSFORMER_LC0")
        weights = self.adapter.cfg.checkpoint_path
        if not str(weights).endswith(".pb.gz"):
            raise RuntimeError(f"lc0 needs the .pb.gz net, not {weights}")
        lines, pv, t0 = [], [], time.monotonic()
        with chess.engine.SimpleEngine.popen_uci(exe, timeout=timeout) as prog:
            prog.configure({"WeightsFile": str(weights), "VerboseMoveStats": True, **options})
            with prog.analysis(board, chess.engine.Limit(nodes=int(nodes))) as run:
                for info in run:
                    if "string" in info:
                        lines.append(info["string"])
                    elif "pv" in info:
                        pv = [m.uci() for m in info["pv"]]
                best = run.wait().move
        root, rows = self._parse_search(lines, board)
        return {"bestmove": best.uci() if best else None, "pv": pv, "nodes": int(nodes),
                "seconds": time.monotonic() - t0, "root": root, "rows": rows}

    @staticmethod
    def _parse_search(lines, board):
        """lc0's VerboseMoveStats lines (after "info string") -> (root, rows): one
        line per move, then the root as "node"; the last complete block counts."""
        stat = re.compile(r"^(\S+)\s+\(\s*\d*\s*\)\s+N:\s+(\d+)")
        field = re.compile(r"\((P|WL|D|M|Q):\s*([^)%]*)%?\)")
        block, rows, root = [], [], {}
        for line in lines:
            m = stat.match(line)
            if not m:
                continue
            f = {k: (float(v) if v.strip().lstrip("-").replace(".", "", 1).isdigit() else None)
                 for k, v in field.findall(line)}
            if m.group(1) == "node":
                root = {"q": f.get("Q"), "wl": f.get("WL"), "d": f.get("D"), "m": f.get("M")}
                block, rows = rows, []
            else:
                mv = board.parse_uci(m.group(1))           # castling comes as king-takes-rook
                n = int(m.group(2))
                rows.append({"uci": mv.uci(), "san": board.san(mv), "n": n,
                             "p": None if f.get("P") is None else f["P"] / 100,
                             "q": f.get("Q") if n else None})
        total = sum(r["n"] for r in block) or 1
        for r in block:
            r["share"] = r["n"] / total
        block.sort(key=lambda r: (-r["n"], -(r["p"] or 0)))
        return root, block


_ENGINE_CLASSES = {"maia3": MaiaEngine, "lc0": LeelaEngine}


def find_lc0():
    """The lc0 program for `LeelaEngine.search`: $CHESSFORMER_LC0, PATH, or the
    usual install places (an app started from the Finder has no shell PATH)."""
    env = os.environ.get("CHESSFORMER_LC0")
    if env and Path(env).exists():
        return env
    return shutil.which("lc0") or next(
        (p for p in ("/opt/homebrew/bin/lc0", "/usr/local/bin/lc0") if Path(p).exists()), None)


# ----- aliases ----------------------------------------------------------------

def resolve_engine(alias: str):
    """alias -> (engine class, display name), without loading weights. Takes any
    registered alias (Maia-3 registry names, Leela aliases or a .pb.gz / .onnx
    path, …); ValueError otherwise."""
    if alias is None or not str(alias).strip():
        raise ValueError("No model named.\n\n" + format_catalog())
    adapter_cls, display = resolve_adapter(alias)
    return _ENGINE_CLASSES.get(adapter_cls.name, ChessformerEngine), display


def load_engine(alias: str, **kwargs):
    """Build the engine an alias names, e.g. load_engine("bt4"),
    load_engine("maia3-23m"). Keyword arguments go to its constructor."""
    cls, _ = resolve_engine(alias)
    return cls(alias=alias, **kwargs)


def format_engine_list() -> str:
    return format_catalog()


def build_cfg(alias="maia3-5m", device=None, checkpoint_path=None, trust_checkpoint=False):
    """The args-namespace a Maia-3 model expects (see adapters.maia3.build_cfg)."""
    from .adapters.maia3 import build_cfg as _build_cfg
    return _build_cfg(alias, device, checkpoint_path, trust_checkpoint)
