"""
engine.py — the Maia-3 interpretability core (no UI, safe to import anywhere).

`MaiaEngine` is a thin, hook-instrumented wrapper around a loaded MAIA3Model. It
loads the checkpoint, runs forward passes, and captures the residual stream at
every sub-layer. Everything the app's visuals show is computed here; nothing in
this file imports pywebview or the UI, so it imports cleanly into a notebook.

Quick start (Colab / Jupyter / REPL):

    import sys; sys.path.append("interp_app")     # if not already on the path
    import chess
    from engine import MaiaEngine

    eng   = MaiaEngine()                           # downloads 5M weights on first run
    board = chess.Board()

    # 1. policy + WDL at a chosen rating
    out = eng.evaluate(board, self_elo=1500)
    out["policy"][:3]                              # [(uci, prob), ...] descending

    # 2. transformer_lens-style cache of the whole residual stream
    out, cache = eng.run_with_cache(board, self_elo=1500)
    cache["block_03"].shape                        # (64, dim_vit)  per-square residual
    eng.hook_points.keys()                         # every name you can read / patch

    # 3. logit lens: decode any (64, dim) residual through the policy head
    eng.logit_lens(cache["postattn_04"], board)    # {'uci','san',...} top legal move

    # 4. activation patching / ablation: inject, don't just observe
    def zero_attn(act):  return act * 0            # kill block 5's attention write
    eng.run_with_hooks(board, 1500, fwd_hooks=[("attn_05", zero_attn)])

See `attention()` and `residual_stream()` for the structured views the app draws.
"""
import math
import os
import types
from collections import deque
from pathlib import Path

import chess
import torch

from maia3.models import MAIA3Model  # noqa: F401
from maia3.uci import load_model, sample_from_logits
from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
from maia3.utils import get_all_possible_moves, mirror_move
from maia3.model_registry import resolve_model_spec, apply_model_config, resolve_checkpoint_path

__all__ = ["MaiaEngine", "build_cfg", "pick_device"]


def pick_device(explicit: str | None = None) -> str:
    """cuda if present, else honor MAIA3_DEVICE, else cpu.

    Default is CPU on purpose: the 5M model on a single position is instant on
    CPU and avoids the occasional MPS op gap. Set MAIA3_DEVICE=mps to override.
    """
    if explicit:
        return explicit
    env = os.environ.get("MAIA3_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_cfg(alias="maia3-5m", device=None, checkpoint_path=None,
              trust_checkpoint=False):
    """Build the args-namespace the model + load_model expect, using the repo's
    own model spec so dim_vit / num_heads / gab_* / history all match the weights."""
    cfg = types.SimpleNamespace()
    spec = resolve_model_spec(alias)
    apply_model_config(cfg, spec)          # copies the architecture preset onto cfg
    cfg.model_spec = spec
    cfg.device = pick_device(device)
    cfg.trust_checkpoint = trust_checkpoint
    cfg.checkpoint_path = checkpoint_path   # None -> resolved from HF cache below
    return cfg, spec


class MaiaEngine:
    """Thin, hook-instrumented wrapper around a loaded MAIA3Model."""

    def __init__(self, alias="maia3-5m", device=None, checkpoint_path=None,
                 activation_dir="activations", trust_checkpoint=False):
        self.cfg, self.spec = build_cfg(alias, device, checkpoint_path, trust_checkpoint)

        if self.cfg.checkpoint_path is None:
            # Use the checkpoint from the local HF cache if present, otherwise
            # download it from Hugging Face — so the app runs on a fresh machine.
            self.cfg.checkpoint_path = resolve_checkpoint_path(
                self.spec, local_files_only=False
            )

        self.device = self.cfg.device
        self.model = load_model(self.cfg)   # builds MAIA3Model(cfg), loads weights, .eval()
        self.model.to(self.device)          # ensure placement (cuda/mps/cpu) for GPU runs

        # exact index <-> UCI mapping used by the released engine
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        self.activation_dir = Path(activation_dir)
        self.activation_dir.mkdir(parents=True, exist_ok=True)

        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self.hook_points: dict[str, torch.nn.Module] = {}   # name -> module (read or patch)
        self._register_hooks()

    # ----- activation hooks -------------------------------------------------
    def _register_hooks(self):
        """Capture the residual stream entering block 0, after every block
        (post-LN), and after the final encoder norm. Overwritten each forward,
        so the snapshot always corresponds to the most recent position.

        Also records `hook_points`: a name -> module map covering exactly the
        same points, so `run_with_hooks` can patch/ablate wherever you can read."""
        def make_hook(name):
            def hook(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                self._activations[name] = t.detach().to("cpu")
            return hook

        def add(name, module):
            self.hook_points[name] = module
            self._hooks.append(module.register_forward_hook(make_hook(name)))

        add("embed_in", self.model.token_projection)
        for i, blk in enumerate(self.model.transformer.layers):
            add(f"block_{i:02d}", blk)
            # Sub-layer writes: the actual vectors each structure ADDS to the
            # residual stream inside a (Post-LN) block. self_attn returns
            # (sa_out, weights) -> sa_out is the attention add; linear2's output
            # is ff_out, the MLP add. (dropout is identity in eval, so these are
            # exactly the vectors summed onto x before each norm.)
            add(f"attn_{i:02d}", blk.self_attn)
            add(f"mlp_{i:02d}", blk.linear2)
            # Running residual stream AFTER the attention sub-layer (norm1's
            # output = norm1(x + sa_out)), so the logit lens can be read at the
            # mid-block point, not just post-block. (post-MLP point = block_NN.)
            add(f"postattn_{i:02d}", blk.norm1)
        add("encoder_out", self.model.transformer.norm)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ----- tokenization -----------------------------------------------------
    def tokens(self, board: chess.Board) -> torch.Tensor:
        """Single current position, padded to fill `history` (matches the
        default `--use-uci-history` OFF behavior of the released engine)."""
        hist = deque([tokenize_board(board)], maxlen=self.cfg.history)
        toks = get_historical_tokens(
            hist, self.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0
        )
        return toks.unsqueeze(0).to(self.device)

    def _idx_to_move(self, board: chess.Board, idx: int):
        """Decode a policy index to a legal chess.Move, un-mirroring for Black."""
        uci = self.idx_to_move[int(idx)]
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return None
        return mv if mv in board.legal_moves else None

    # ----- forward / policy -------------------------------------------------
    @torch.no_grad()
    def _forward(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One raw forward pass. Resets and repopulates `self._activations` via the
        hooks, and returns (logits_move (4352,), logits_value (3,)) as floats."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self._activations = {}

        tokens = self.tokens(board)
        self_elos = torch.tensor([int(self_elo)], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([int(oppo_elo)], dtype=torch.long, device=self.device)

        logits_move, logits_value, _ = self.model(tokens, self_elos, oppo_elos)
        return logits_move[0].float(), logits_value[0].float()

    @torch.no_grad()
    def evaluate(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One forward pass. Returns the full normalized policy over legal moves
        (descending), the WDL for the side to move, and stashes activations."""
        logits, logits_value = self._forward(board, self_elo, oppo_elo)

        legal_mask = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
        policy = []
        if bool(legal_mask.any()):                  # may be empty for hand-edited positions
            logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)   # normalized over legal moves
            for idx in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist():
                mv = self._idx_to_move(board, idx)
                if mv is not None:
                    policy.append((mv.uci(), float(probs[idx])))
            policy.sort(key=lambda x: x[1], reverse=True)

        loss, draw, win = torch.softmax(logits_value.float(), dim=-1).tolist()
        return {
            "policy": policy,                                   # [(uci, prob)] desc
            "wdl": {"win": win, "draw": draw, "loss": loss},    # side-to-move perspective
            "_logits": logits,                                  # masked, for sampling
        }

    def select_move(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None,
                    temperature: float = 1.0, top_p: float = 1.0):
        """Pick a move at the given rating (temperature 0 = argmax). Reuses the
        released engine's sampler. Activations correspond to this position."""
        res = self.evaluate(board, self_elo, oppo_elo)
        idx = sample_from_logits(res["_logits"], temperature, top_p)
        return self._idx_to_move(board, idx), res

    # ----- transformer_lens-style ergonomics --------------------------------
    def run_with_cache(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """Forward pass returning (out, cache).

          out   = the evaluate() dict (policy / wdl / _logits)
          cache = {name: Tensor(64, dim_vit)} snapshot of the residual stream,
                  copied so it survives the next forward. Keys are `hook_points`:
                  embed_in, then per block attn_NN / postattn_NN / mlp_NN /
                  block_NN, then encoder_out.
        """
        out = self.evaluate(board, self_elo, oppo_elo)
        cache = {k: v.squeeze(0).clone() for k, v in self._activations.items()}
        return out, cache

    def run_with_hooks(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None,
                       *, fwd_hooks=(), return_type: str = "policy"):
        """Forward pass with temporary intervention hooks — activation patching,
        ablation, steering. Each `fwd_hooks` entry is (name, fn) where `name` is a
        key of `hook_points` and `fn(activation) -> activation | None` (return
        None to leave it unchanged). `activation` is that module's output on the
        live device: the residual *write* for attn_NN/mlp_NN, the running stream
        for embed_in / postattn_NN / block_NN / encoder_out.

        return_type='policy' -> the evaluate() dict; 'logits' -> the raw
        (logits_move (4352,), logits_value (3,)) tensors (unmasked). The hooks are
        always removed afterward, even on error.

        NOTE on heads: attn_NN is the write AFTER out_proj, which mixes every
        head into every channel — slicing attn_NN channels does NOT isolate a
        head. For a true per-head ablation use `ablate_head()` / `head_writes()`.
        Example — halve block 5's whole attention write:
            self.run_with_hooks(board, 1500, fwd_hooks=[("attn_05", lambda a: a * 0.5)])
        """
        def make(fn):
            def hook(_module, _inp, out):
                is_tuple = isinstance(out, tuple)
                t = out[0] if is_tuple else out
                new = fn(t)
                if new is None:
                    return out
                return (new, *out[1:]) if is_tuple else new
            return hook

        handles = []
        try:
            for name, fn in fwd_hooks:
                handles.append(self.hook_points[name].register_forward_hook(make(fn)))
            if return_type == "logits":
                return self._forward(board, self_elo, oppo_elo)
            return self.evaluate(board, self_elo, oppo_elo)
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def logit_lens(self, activation: torch.Tensor, board: chess.Board | None = None,
                   *, legal_only: bool = True):
        """Decode a residual-stream activation through the policy head (logit lens).

        `activation` is (64, dim) or (1, 64, dim) — e.g. any value from a cache.
        With no `board`, returns the raw (4352,) move logits. With a `board`,
        returns the top move as a dict {idx, uci, san, from, to, piece, logit},
        masked to legal moves when `legal_only` (None if no legal move decodes).
        """
        x = activation.to(self.device)
        if x.dim() == 3:
            x = x[0]
        logits = self._move_logits(x)                       # (4352,)
        if board is None:
            return logits
        if legal_only:
            legal = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
            if bool(legal.any()):
                logits = logits.masked_fill(~legal, float("-inf"))
        idx = int(torch.argmax(logits))
        frm, to = self.move_squares(idx)
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
    def attention(self, board, self_elo, oppo_elo=None, layer=0, head=0):
        """Return the 64x64 attention components of one (layer, head) for the
        current position, reproducing Chessformer Fig. 1:
          qk         = semantic dot-product logits (scaled QK^T)  -- selected head
          gab        = geometric attention bias (learned positional bias) -- selected head
          attn       = softmax(qk + gab)  -- the selected head's attention
          attn_layer = mean over ALL heads of softmax(qk + gab) -- the whole layer's
                       aggregate attention pattern (head-independent)
        Matrices are in the side-to-move (mirrored) frame; square = rank*8 + file.
        Computed directly from the residual stream entering the block, using that
        block's own projections and GAB generator -- no model re-implementation."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        L = int(layer)

        # one forward populates the residual-stream snapshot via the hooks
        self.evaluate(board, self_elo, oppo_elo)
        key = "embed_in" if L == 0 else f"block_{L-1:02d}"
        x = self._activations[key].to(self.device)            # (1, 64, dim) -> input to block L
        blk = self.model.transformer.layers[L].self_attn
        H = blk.num_heads
        d = x.size(-1)
        dh = d // H

        gab = blk._sq_bias(x)                                  # (1, H, 64, 64)

        W = blk.mha.in_proj_weight                             # (3d, d), order [q; k; v]
        q = x @ W[:d].t()
        k = x @ W[d:2 * d].t()
        b = blk.mha.in_proj_bias
        if b is not None:
            q = q + b[:d]
            k = k + b[d:2 * d]
        q = q.view(1, 64, H, dh).transpose(1, 2)              # (1, H, 64, dh)
        k = k.view(1, 64, H, dh).transpose(1, 2)
        qk = (q @ k.transpose(-2, -1)) / math.sqrt(dh)        # (1, H, 64, 64)
        attn = torch.softmax(qk + gab, dim=-1)

        h = int(head)
        return {
            "layer": L, "head": h, "num_heads": H,
            "qk": qk[0, h].cpu().tolist(),                    # selected head
            "gab": gab[0, h].cpu().tolist(),                  # selected head
            "attn": attn[0, h].cpu().tolist(),                # selected head
            "attn_layer": attn[0].mean(0).cpu().tolist(),     # whole layer: mean softmax over heads
        }

    # ----- per-head attention writes (for true head ablation) ---------------
    @torch.no_grad()
    def head_writes(self, board, self_elo, oppo_elo=None, layer=0):
        """Exact per-head residual-stream writes of one block's attention: (H, 64, dim).

        Head h's write is (A_h V_h) W_O^{(h)} — its attention-weighted values pushed
        through its own dh-column block of the OUTPUT projection. This is the tensor a
        true head ablation must subtract: the hooked attn_NN activation is post-out_proj,
        where the heads are already mixed across every channel, so slicing attn_NN
        channels does not correspond to heads. Recomputed from the block's own weights
        (same approach as `attention()`); verified on the spot — the writes must sum
        (plus the out_proj bias) back to the attn_NN activation of this forward."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        L = int(layer)

        self.evaluate(board, self_elo, oppo_elo)
        key = "embed_in" if L == 0 else f"block_{L-1:02d}"
        x = self._activations[key].to(self.device)             # (1, 64, dim) into block L
        blk = self.model.transformer.layers[L].self_attn
        H = blk.num_heads
        d = x.size(-1)
        dh = d // H

        gab = blk._sq_bias(x)                                  # (1, H, 64, 64)
        W = blk.mha.in_proj_weight                             # (3d, d), order [q; k; v]
        b = blk.mha.in_proj_bias
        q, k, v = (x @ W[i * d:(i + 1) * d].t() +
                   (b[i * d:(i + 1) * d] if b is not None else 0) for i in range(3))
        q, k, v = (t.view(1, 64, H, dh).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(dh) + gab, dim=-1)

        Wo = blk.mha.out_proj.weight                           # (dim, dim)
        per_head_out = Wo.view(d, H, dh).permute(1, 2, 0)      # (H, dh, dim)
        writes = (attn @ v)[0] @ per_head_out                  # (H, 64, dim)

        recon = writes.sum(0)
        if blk.mha.out_proj.bias is not None:
            recon = recon + blk.mha.out_proj.bias
        target = self._activations[f"attn_{L:02d}"][0].to(self.device)
        assert torch.allclose(recon, target, atol=1e-4, rtol=1e-4), \
            f"head_writes reconstruction failed for block {L} — do not trust the ablation"
        return writes.cpu()

    def ablate_head(self, board, self_elo, layer, head, oppo_elo=None,
                    return_type: str = "policy"):
        """Forward pass with ONE attention head's write removed, exactly.

        Upstream of `layer` is untouched by the ablation, so the write computed from a
        clean pass is exactly the write the ablated pass would have produced — we
        subtract it from attn_NN via run_with_hooks. Downstream layers then react to
        the head's absence normally."""
        w = self.head_writes(board, self_elo, oppo_elo, layer)[int(head)]

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"attn_{layer:02d}", sub)],
                                   return_type=return_type)

    # ----- residual-stream evolution across depth ---------------------------
    @staticmethod
    def move_squares(idx):
        """Canonical (from, to) squares for a policy-move index (handles promotions).
        Mirrors MAIA3Model.forward's move layout: first 64*64 are from*64+to, then
        256 promotions ordered from_file*32 + to_file*4 + piece (rank7 -> rank8)."""
        if idx < 64 * 64:
            return idx // 64, idx % 64
        idx -= 64 * 64
        from_file, to_file = idx // 32, (idx % 32) // 4
        return 48 + from_file, 56 + to_file          # rank-7 -> rank-8, canonical

    def _move_logits(self, x):
        """Full (4352,) move logits from one position's residual x (64, dim),
        replicating MAIA3Model.forward's policy head (64*64 moves + 256 promotions)."""
        hid = self.cfg.head_hid_dim
        sq_from = self.model.proj_sq_from(x)                  # (64, hid)
        sq_to = self.model.proj_sq_to(x)                      # (64, hid)
        scores = (sq_from @ sq_to.t()) / math.sqrt(hid)      # (64, 64)
        promo_bias = self.model.promo_bias_proj(sq_to[56:64]) * math.sqrt(hid)  # (8 files, 4 pieces)
        promo = [scores[48 + ff, 56 + tf] + promo_bias[tf, pc]
                 for ff in range(8) for tf in range(8) for pc in range(4)]      # (256,)
        return torch.cat([scores.reshape(-1), torch.stack(promo)])             # (4352,)

    @torch.no_grad()
    def residual_stream(self, board, self_elo, oppo_elo=None):
        """Two per-square views of how the residual stream is built up, in the
        side-to-move frame (square = rank*8 + file):

          delta = the per-square magnitude of the vector each STRUCTURE writes
                  into the stream, in execution order. The blocks are Post-LN
                  (x = norm(x + sublayer(x))), so the things that actually add a
                  vector are: the input embedding (`emb`), then, for every block,
                  the self-attention sub-layer (`bN attn` = ||sa_out||) and the
                  feed-forward sub-layer (`bN mlp` = ||ff_out||). Each entry is
                  tagged with its `kind` ('emb'/'attn'/'mlp') so the UI can mark
                  *what* is writing at each point. This is the residual-stream
                  evolution decomposed by contributing module, not just norms.

          moves = per-SUB-LAYER logit lens on the running residual stream, same
                  resolution as `delta`: emb, then for every block the post-
                  attention point (norm1 out) and the post-MLP point (block out),
                  then enc. Decode each through the policy head, take the top
                  *legal* move. Watch the prediction form sub-layer by sub-layer.

        `delta` is a list of {label, kind, norm:[64]}; `moves` is a list of
        {label, kind, from, to, uci, san, piece} (from/to canonical squares;
        uci/san real-board; piece = symbol of the moving piece, e.g. 'N'/'n')."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self.evaluate(board, self_elo, oppo_elo)         # populates activations + logits
        nb = self.cfg.num_blocks

        # ---- delta: the vector each structure adds to the stream, in order ----
        def per_sq_norm(name):
            return self._activations[name][0].norm(dim=-1).tolist()   # (64,)

        delta = [{"label": "emb", "kind": "emb", "norm": per_sq_norm("embed_in")}]
        for i in range(nb):
            delta.append({"label": f"b{i} attn", "kind": "attn",
                          "norm": per_sq_norm(f"attn_{i:02d}")})
            delta.append({"label": f"b{i} mlp", "kind": "mlp",
                          "norm": per_sq_norm(f"mlp_{i:02d}")})

        # ---- moves: per-sub-layer logit lens on the running residual stream ----
        # Same resolution as delta: emb, then (post-attn, post-mlp) per block, enc.
        steps = [("embed_in", "emb", "emb")]
        for i in range(nb):
            steps.append((f"postattn_{i:02d}", f"b{i} attn", "attn"))
            steps.append((f"block_{i:02d}",    f"b{i} mlp",  "mlp"))
        steps.append(("encoder_out", "enc", "enc"))

        legal = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
        moves = []
        for name, lab, kind in steps:
            a = self._activations[name][0]
            logits = self._move_logits(a.to(self.device))    # (4352,), honors device
            if bool(legal.any()):
                logits = logits.masked_fill(~legal, float("-inf"))
            idx = int(torch.argmax(logits))
            frm, to = self.move_squares(idx)
            mv = self._idx_to_move(board, idx)
            san = board.san(mv) if mv is not None else None
            # the piece doing the move (real board), so the UI can draw it on the
            # from-square instead of a generic highlight.
            pc = board.piece_at(mv.from_square) if mv is not None else None
            moves.append({"label": lab, "kind": kind, "from": frm, "to": to,
                          "uci": mv.uci() if mv else None, "san": san,
                          "piece": pc.symbol() if pc is not None else None})

        return {"delta": delta, "moves": moves}

    # ----- activation dump --------------------------------------------------
    def save_activations(self, filename: str, meta: dict | None = None) -> str:
        """Persist the most recent forward's residual-stream snapshot.
        Each tensor is (64, dim_vit). Keys: embed_in, block_00..block_07,
        encoder_out (post-block residual at each depth), plus attn_NN / mlp_NN
        (the raw vector each sub-layer writes into the stream inside block NN)."""
        snap = {k: v.squeeze(0).clone() for k, v in self._activations.items()}
        snap["meta"] = meta or {}
        path = self.activation_dir / filename
        torch.save(snap, path)
        return str(path)
