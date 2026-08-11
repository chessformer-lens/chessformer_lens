"""Interpretability core for chessformers that treat the 64 squares as tokens.
`MaiaEngine` loads a Maia-3 checkpoint, runs forward passes, and captures the
residual stream at every layer.

  evaluate              one forward pass yielding the full normalized policy over
                        legal moves (in descending order) and the W/D/L for the side to
                        move
  select_move           pick a move at a rating (temperature 0 = argmax),
                        through the released engine's own sampler
  tokens                the position as the model's input tokens, padded to
                        fill `history`
  run_with_cache        forward pass returning (out, cache): the whole
                        residual stream, transformer_lens style
  run_with_hooks        forward pass with intervention hooks: activation
                        patching, ablation, steering
  logit_lens            decode any residual activation at a readout point
                        by pushing it through the policy head
                        


  attention             the 64×64 components of one (layer, head): semantic
                        QKᵀ, geometric GAB, and the softmax the model actually
                        runs
  qk_scores             one block's scaled QKᵀ logits
  gab_bias              one block's generated bias
  gab_coeffs            the coefficients head h applies to the GAB bank
  gab_templates         the static square-pair template bank every layer shares
  head_writes           exact residual writes of one block's attention

  
  ablate_head           forward pass with one head's write removed, exactly
  ablate_grid           the carrier heatmap of one move: every head ablated in
                        turn, Δlogit = ablated − clean (negative = the head
                        carried it)
  residual_stream       per-square views of how the stream is built up, one row
                        per readout point
  compare_residual      the same position at two ratings, differenced — where
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
  remove_hooks          detach the capture hooks, for a bare forward with no
                        CPU copies

Module level, beside the class: `build_cfg` builds the args-namespace the model
expects from a registry alias, and `pick_device` resolves the torch device
(explicit > $MAIA3_DEVICE > cuda > cpu).

Every tensor a read path returns is on CPU, in the model's canonical
side-to-move frame (square = rank*8 + file). Depth reads the same as in the
figures: `emb`, then `aN`/`mN` for block N's attention and MLP sub-layers, then
`enc` — `depth_points` hands you that axis directly.

It imports cleanly into a notebook; plotting lives next door in interp_plot.py
(static figures) and interp_widget.py (interactive panels), and the standalone
app in bridge.py/app.py/ui.py.
"""
import math
import os
import types
from collections import deque
from pathlib import Path

import chess
import torch

# maia3 is the one dependency pip cannot fetch for us: it is not on PyPI, and
# PyPI rejects direct git URLs in dependency metadata. Every entry point into
# this package (bridge.py, app.py, the lazy names in __init__.py) reaches
# maia3 through this module, so this is the single boundary where a missing
# install should be turned into an instruction rather than a traceback.
try:
    from maia3.models import MAIA3Model  # noqa: F401
    from maia3.uci import load_model, sample_from_logits
    from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
    from maia3.utils import get_all_possible_moves, mirror_move
    from maia3.model_registry import resolve_model_spec, apply_model_config, resolve_checkpoint_path
except ModuleNotFoundError as exc:
    if exc.name != "maia3" and not str(exc.name or "").startswith("maia3."):
        raise
    raise ModuleNotFoundError(
        "chessformer_lens needs the Maia-3 model code, which is not on PyPI "
        "and so is not installed by `pip install chessformer_lens`.\n\n"
        "    pip install git+https://github.com/CSSLab/maia3\n"
    ) from exc
except ImportError as exc:
    # maia3 imports torch.nn.RMSNorm, which only exists from torch 2.4.
    if "RMSNorm" not in str(exc):
        raise
    raise ImportError(
        f"Maia-3 needs torch >= 2.4; this environment has {torch.__version__}.\n"
        "    pip install --upgrade 'torch>=2.4'"
    ) from exc

__all__ = ["MaiaEngine", "build_cfg", "pick_device"]


def pick_device(explicit: str | None = None) -> str:
    """Resolve a torch device: explicit argument > $MAIA3_DEVICE > cuda > cpu.

    MPS is never picked automatically — the 5M model on one position is instant
    on CPU anyway, and this avoids the occasional MPS op gap. Set
    MAIA3_DEVICE=mps to opt in."""
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
    own model spec so dim_vit / num_heads / gab_* / history all match the weights.

    `trust_checkpoint=True` becomes torch.load(weights_only=False) downstream,
    i.e. it will execute pickled code from the checkpoint — see MaiaEngine."""
    cfg = types.SimpleNamespace()
    spec = resolve_model_spec(alias)
    apply_model_config(cfg, spec)          # copies the architecture preset onto cfg
    cfg.model_spec = spec
    cfg.device = pick_device(device)
    cfg.trust_checkpoint = trust_checkpoint
    cfg.checkpoint_path = checkpoint_path   # None -> resolved from HF cache below
    return cfg, spec


class MaiaEngine:
    """Hook-based interpretability engine for a Maia-3 checkpoint.

    Exposes read paths (run_with_cache, logit_lens, residual_stream, attention,
    gab_*) and intervention paths (run_with_hooks, ablate_head, ablate_grid).
    Every tensor a read path returns is on CPU, in the model's canonical
    side-to-move frame (square = rank*8 + file)."""

    def __init__(self, alias="maia3-5m", device=None, checkpoint_path=None,
                 activation_dir="activations", trust_checkpoint=False):
        """Build the model and install the permanent capture hooks.

        Resolves the checkpoint (downloading it from Hugging Face on first
        use), notes `activation_dir` for later, and registers forward hooks that copy
        every sub-layer's output to CPU on each forward. That copy is what
        makes the read paths work; it also costs a handful of device->host
        transfers per forward.

        `device`: an explicit string wins; otherwise see `pick_device`.
        `trust_checkpoint=True` loads with `weights_only=False`, i.e. it can
        execute pickled code from the checkpoint file — only use it for
        checkpoints you produced yourself."""
        self.cfg, self.spec = build_cfg(alias, device, checkpoint_path, trust_checkpoint)

        if self.cfg.checkpoint_path is None:
            # Use the checkpoint from the local HF cache if present, otherwise
            # download it from Hugging Face — so the app runs on a fresh machine.
            # Say so before the download starts: it is hundreds of MB
            try:
                self.cfg.checkpoint_path = resolve_checkpoint_path(
                    self.spec, local_files_only=True
                )
            except Exception:
                print(f"chessformer_lens: {alias} weights are not in the local "
                      f"Hugging Face cache; downloading them now (this is a "
                      f"one-time, several-hundred-MB fetch for the larger "
                      f"models).\n  from:  https://huggingface.co/UofTCSSLab\n"
                      f"  cache: {os.environ.get('HF_HOME') or '~/.cache/huggingface'}",
                      flush=True)
                self.cfg.checkpoint_path = resolve_checkpoint_path(
                    self.spec, local_files_only=False
                )

        self.device = self.cfg.device
        self.model = load_model(self.cfg)   # builds MAIA3Model(cfg), loads weights, .eval()
        self.model.to(self.device)          # no-op if load_model already placed it; cheap insurance

        # exact index <-> UCI mapping used by the released engine
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        # Not created here: `save_activations` makes it on first write, so
        # importing the engine in a notebook leaves no directory behind.
        self.activation_dir = Path(activation_dir)

        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self.hook_points: dict[str, torch.nn.Module] = {}   # name -> module (read or patch)
        self._gab_templates: torch.Tensor | None = None     # lazy (gen, 64, 64) cache
        self._register_hooks()

    # ----- activation hooks -------------------------------------------------
    def _register_hooks(self):
        """Install the permanent capture hooks: 4 per block plus 2, overwritten
        on every forward, so the snapshot is always the most recent position.

          embed_in      the stream entering block 0 (token_projection out)
          attn_NN       block NN's attention write (self_attn out, post out_proj)
          postattn_NN   the running stream after attention (norm1 out)
          mlp_NN        block NN's MLP write (linear2 out)
          block_NN      the running stream after the whole block (post-LN)
          encoder_out   after the final encoder norm

        This is a Post-LN model, x = norm(x + sublayer(x)), so attn_NN/mlp_NN
        are the vectors *added* to the stream (dropout is identity in eval),
        while the other four are the stream itself. Everything is copied to
        CPU on the way out. `hook_points` records name -> module for all of
        these, so `run_with_hooks` can patch anywhere you can read."""
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
            # Sub-layer writes (see the docstring): self_attn returns
            # (sa_out, None) — MHA is called with need_weights=False — and
            # sa_out is the attention add; linear2's output is the MLP add.
            add(f"attn_{i:02d}", blk.self_attn)
            add(f"mlp_{i:02d}", blk.linear2)
            # Running residual stream AFTER the attention sub-layer (norm1's
            # output = norm1(x + sa_out)), so the logit lens can be read at the
            # mid-block point, not just post-block. (post-MLP point = block_NN.)
            add(f"postattn_{i:02d}", blk.norm1)
        add("encoder_out", self.model.transformer.norm)

    def remove_hooks(self):
        """Detach the capture hooks, for a bare forward with no CPU copies.

        `hook_points` stays populated so `run_with_hooks` keeps working, but
        the read paths will KeyError once `_activations` stops being refreshed.
        There is no re-register; build a new MaiaEngine."""
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
        """Decode a policy index to a legal chess.Move, un-mirroring for Black.
        Returns None if the index doesn't decode to a move that's legal here.
        (Distinct from the `idx_to_move` attribute, the raw index -> uci table.)"""
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
                  cloned so it survives the next forward. Keys are the
                  `hook_points` names.

        Cached tensors are always on CPU, whatever `self.device` is. Contrast
        `run_with_hooks`, whose hook functions see activations on the live
        device — move tensors yourself when feeding a cached value into an
        intervention."""
        out = self.evaluate(board, self_elo, oppo_elo)
        cache = {k: v.squeeze(0).clone() for k, v in self._activations.items()}
        return out, cache

    def run_with_hooks(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None,
                       *, fwd_hooks=(), return_type: str = "policy"):
        """Forward pass with temporary intervention hooks — activation patching,
        ablation, steering. Each `fwd_hooks` entry is (name, fn) where `name`
        is a key of `hook_points` and `fn(activation) -> activation | None`
        (return None to leave it unchanged). `activation` is that module's
        output on the live device: the residual *write* for attn_NN/mlp_NN,
        the running stream for everything else.

        return_type='policy' -> the evaluate() dict; 'logits' -> the raw
        (logits_move, logits_value) tensors, unmasked. Hooks are always
        removed afterward, even on error.

        Two gotchas. attn_NN is the write after out_proj, which mixes every
        head into every channel, so slicing attn_NN channels does not isolate
        a head — use `ablate_head()` / `head_writes()` for that. And the
        capture hooks from __init__ fire before your hook on the same module,
        so after a patched run `_activations[name]` (and any cache taken from
        it) holds the clean output of the patched module, while downstream
        entries do reflect the patch. Take your clean cache before you patch,
        not after.

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
        masked to legal moves when `legal_only` (`uci`/`san`/`piece` are None
        if the argmax index doesn't decode to a legal move).

        Caveat: this applies the trained head's proj_sq_from/to directly to an
        intermediate residual, skipping `transformer.norm` — the final
        LayerNorm the head was trained behind. Only `encoder_out` is read in
        distribution; earlier points are a lens, not a prediction. Compare
        readout points against each other rather than against the real policy.
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
        frm, to = self._move_squares(idx)
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
        """One forward pass, then the residual stream entering block `layer`:
        (1, 64, dim) on the live device — exactly the tensor the block's
        attention (and its GAB generator) sees as query/key/value."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self.evaluate(board, self_elo, oppo_elo)
        key = "embed_in" if layer == 0 else f"block_{layer-1:02d}"
        return self._activations[key].to(self.device)

    @staticmethod
    def _qk_from_x(blk, x):
        """Scaled QK^T content logits of one MHA block from its input x:
        (1, H, 64, 64), using the block's own in-projection."""
        H = blk.num_heads
        d = x.size(-1)
        dh = d // H
        W = blk.mha.in_proj_weight                             # (3d, d), order [q; k; v]
        q = x @ W[:d].t()
        k = x @ W[d:2 * d].t()
        b = blk.mha.in_proj_bias
        if b is not None:
            q = q + b[:d]
            k = k + b[d:2 * d]
        q = q.view(1, 64, H, dh).transpose(1, 2)              # (1, H, 64, dh)
        k = k.view(1, 64, H, dh).transpose(1, 2)
        return (q @ k.transpose(-2, -1)) / math.sqrt(dh)      # (1, H, 64, 64)

    @torch.no_grad()
    def attention(self, board, self_elo, oppo_elo=None, layer=0, head=0):
        """The 64x64 attention components of one (layer, head) for the current
        position, reproducing Chessformer Fig. 1:

          qk           = semantic dot-product logits (scaled QK^T)
          gab          = geometric attention bias (learned positional bias)
          attn         = softmax(qk + gab), the head's actual attention
          attn_content = softmax(qk), attention without the geometry; compare
                         with `attn` to see what GAB adds
          attn_layer   = mean over all heads of softmax(qk + gab)
          coeffs       = the smolgen mixing coefficients that generated this
                         head's gab (see gab_coeffs())

        Computed directly from the residual stream entering the block, using
        the block's own projections and GAB generator — no re-implementation
        of the model. Requires a GAB block: with use_gab=False there are no
        smolgen submodules and the _sq_bias call raises AttributeError.
        Matrices are in the side-to-move frame; square = rank*8 + file."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)   # (1, 64, dim) -> input to block L
        blk = self.model.transformer.layers[L].self_attn

        gab = blk._sq_bias(x)                                  # (1, H, 64, 64)
        qk = self._qk_from_x(blk, x)                           # (1, H, 64, 64)
        attn = torch.softmax(qk + gab, dim=-1)
        attn_content = torch.softmax(qk, dim=-1)
        coeffs = self._smolgen_coeffs(blk, x) if blk.use_gab else None

        h = int(head)
        return {
            "layer": L, "head": h, "num_heads": blk.num_heads,
            "gen_size": blk.gen_size if blk.use_gab else None,
            "qk": qk[0, h].cpu().tolist(),                    # selected head
            "gab": gab[0, h].cpu().tolist(),                  # selected head
            "attn": attn[0, h].cpu().tolist(),                # selected head
            "attn_content": attn_content[0, h].cpu().tolist(),  # selected head, no GAB
            "attn_layer": attn[0].mean(0).cpu().tolist(),     # whole layer: mean softmax over heads
            "coeffs": coeffs[0, h].cpu().tolist() if coeffs is not None else None,
        }

    # ----- GAB / smolgen decomposition --------------------------------------
    # GAB is generated, not stored: a tiny MLP ("smolgen") reads the board state
    # and emits, per head, `gen_size` mixing coefficients over a bank of static
    # 64x64 square-pair templates (`gab_shared_weight`, shared by EVERY layer and
    # head). The bias is exactly  gab[h] = sum_i coeffs[h,i] * template_i.
    # These methods expose the pieces of that factorization.

    @staticmethod
    def _smolgen_coeffs(blk, x):
        """Replicate one block's smolgen generator up to the mixing coefficients:
        (1, H, gen_size). Sanity-checked on the spot: mixing the shared
        templates with these coefficients must reproduce the block's own
        _sq_bias() (asserted, so the check vanishes under python -O)."""
        B = x.size(0)
        if blk.sm1 is not None:                                # per-square path
            y = blk.sm1(x).reshape(B, -1)                      # (B, 64*p)
        else:                                                  # mean-pooled path
            y = torch.mean(x, dim=1)                           # (B, d_model)
        y = blk.sm_act(blk.sm2(y))
        y = blk.ln1(y)
        y = blk.sm_act(blk.sm3(y))
        y = blk.ln2(y).view(B, blk.num_heads, blk.gen_size)    # (B, H, gen)

        recon = torch.einsum("bhi,oi->bho", y, blk.gab_weight).view(B, blk.num_heads, 64, 64)
        target = blk._sq_bias(x)
        assert torch.allclose(recon, target, atol=1e-4, rtol=1e-4), \
            "smolgen coefficient reconstruction failed — do not trust the decomposition"
        return y

    @torch.no_grad()
    def gab_templates(self):
        """The static square-pair template bank behind every GAB: (gen_size, 64, 64).

        Template i is gab_shared_weight[:, i] reshaped so that template[i][q][k]
        is its contribution to query square q attending to key square k (canonical
        side-to-move frame, square = rank*8 + file). Position-independent and
        shared across all layers and heads — this is the model's entire geometric
        vocabulary. Computed once and cached; the same tensor is returned each
        call, so `.clone()` before mutating."""
        if self.model.gab_shared_weight is None:
            raise RuntimeError("this model was built without GAB (use_gab=False)")
        if self._gab_templates is None:
            w = self.model.gab_shared_weight.detach()          # (64*64, gen)
            self._gab_templates = w.t().reshape(-1, 64, 64).cpu().clone()
        return self._gab_templates

    @torch.no_grad()
    def gab_coeffs(self, board, self_elo, oppo_elo=None, layer=0):
        """The smolgen mixing coefficients of one block for this position:
        (H, gen_size). Row h holds the weights with which head h mixes the
        static `gab_templates()` into its 64x64 bias:
            gab_bias(layer, h) == (coeffs[h, :, None, None] * gab_templates()).sum(0)
        (verified internally, see `_smolgen_coeffs`). This is the model
        choosing its geometry live."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)
        blk = self.model.transformer.layers[L].self_attn
        if not blk.use_gab:
            raise RuntimeError(f"block {L} has no GAB (use_gab=False)")
        return self._smolgen_coeffs(blk, x)[0].cpu()

    @torch.no_grad()
    def gab_bias(self, board, self_elo, oppo_elo=None, layer=0, head=None):
        """The generated geometric attention bias of one block for this position:
        (H, 64, 64), or (64, 64) for a single `head`. bias[h][q][k] is added to
        the scaled QK^T logit of query q, key k before the softmax."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)
        blk = self.model.transformer.layers[L].self_attn
        if not blk.use_gab:
            raise RuntimeError(f"block {L} has no GAB (use_gab=False)")
        gab = blk._sq_bias(x)[0].cpu()
        return gab if head is None else gab[int(head)]

    @torch.no_grad()
    def qk_scores(self, board, self_elo, oppo_elo=None, layer=0, head=None):
        """The raw content half of attention — scaled QK^T logits — of one block
        for this position: (H, 64, 64), or (64, 64) for a single `head`.
        softmax(qk_scores + gab_bias) is the attention the model actually runs."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)
        blk = self.model.transformer.layers[L].self_attn
        qk = self._qk_from_x(blk, x)[0].cpu()
        return qk if head is None else qk[int(head)]

    # ----- per-head attention writes (for true head ablation) ---------------
    @torch.no_grad()
    def head_writes(self, board, self_elo, oppo_elo=None, layer=0):
        """Exact per-head residual-stream writes of one block's attention:
        (H, 64, dim).

        Head h's write is (A_h V_h) W_O^{(h)} — its attention-weighted values
        pushed through its own dh-column block of the output projection. This
        is the tensor a true head ablation must subtract: the hooked attn_NN
        activation is post-out_proj, where the heads are already mixed across
        every channel. Recomputed from the block's own weights (same approach
        as `attention()`), and asserted on the spot to sum back — with the
        out_proj bias — to this forward's attn_NN activation."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)    # (1, 64, dim) into block L
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
                                   fwd_hooks=[(f"attn_{layer:02d}", sub)],
                                   return_type=return_type)

    # ----- residual-stream evolution across depth ---------------------------
    def _move_logits(self, x):
        """Full (4352,) move logits from one position's residual x (64, dim),
        replicating MAIA3Model.forward's policy head (64*64 moves + 256 promotions).

        `x` must be exactly (64, dim) with no batch dim — a (1, 64, dim) input
        produces garbage shapes silently. Callers that accept both strip it
        first (see `logit_lens`)."""
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

          delta = the per-square magnitude of the vector each structure writes
                  into the stream, in execution order: the input embedding
                  (`emb`), then for every block the self-attention write
                  (`bN attn` = ||sa_out||) and the feed-forward write
                  (`bN mlp` = ||ff_out||). Each entry is tagged with its `kind`
                  ('emb'/'attn'/'mlp') so the UI can mark what is writing at
                  each point. 1 + 2·num_blocks entries.

          moves = logit lens on the running residual stream at every readout
                  point (see `_lens_steps`): emb, then per block the
                  post-attention and post-MLP points, then a final `enc`
                  readout. Decode each through the policy head, take the top
                  legal move, and watch the prediction form sub-layer by
                  sub-layer. The `logit_lens` caveat applies everywhere but
                  `enc`.

        `delta` is a list of {label, kind, norm:[64]}; `moves` is a list of
        {label, kind, from, to, uci, san, piece} (from/to canonical squares,
        uci/san real-board, piece the moving piece's symbol)."""
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
        legal = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
        moves = [{"label": lab, "kind": kind,
                  **self._lens_move(self._activations[name][0], board, legal)}
                 for name, lab, kind in self._lens_steps()]

        return {"delta": delta, "moves": moves}

    # ----- skill diff on internals ------------------------------------------
    def _lens_steps(self):
        """The readout points of the running residual stream, in order: emb, then
        per block the post-attention and post-MLP points, then enc — so
        2·num_blocks + 2 of them (18 on every current Maia-3 size)."""
        steps = [("embed_in", "emb", "emb")]
        for i in range(self.cfg.num_blocks):
            steps.append((f"postattn_{i:02d}", f"b{i} attn", "attn"))
            steps.append((f"block_{i:02d}",    f"b{i} mlp",  "mlp"))
        steps.append(("encoder_out", "enc", "enc"))
        return steps

    def _lens_move(self, activation, board, legal_mask):
        """Top legal move of one residual snapshot through the policy head."""
        logits = self._move_logits(activation.to(self.device))
        if bool(legal_mask.any()):
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        idx = int(torch.argmax(logits))
        frm, to = self._move_squares(idx)
        mv = self._idx_to_move(board, idx)
        pc = board.piece_at(mv.from_square) if mv is not None else None
        return {"from": frm, "to": to, "uci": mv.uci() if mv else None,
                "san": board.san(mv) if mv is not None else None,
                "piece": pc.symbol() if pc is not None else None}

    @torch.no_grad()
    def compare_residual(self, board, elo_a, elo_b):
        """Skill diff on internals, not just outputs: run the same position at
        two ratings and, at each readout point of the running residual stream
        (see `_lens_steps`), report

          norm     = per-square ||x_A − x_B|| — where on the board, and at
                     what depth, the two skill levels diverge
          move_a/b = logit-lens top legal move of each run at that point
          same     = whether the two lenses agree

        Elo enters as an embedding concatenated to every square token before
        token_projection, so at `emb` the diff is one constant "skill vector"
        repeated on all 64 squares; the interesting structure is how depth
        localizes it. Both runs use oppo_elo == self_elo, so both ratings'
        embeddings move — read the result as the diff between two whole skill
        settings, not one player's. Side-to-move frame; the `logit_lens`
        caveat applies to move_a/move_b everywhere but `enc`."""
        _, cache_a = self.run_with_cache(board, int(elo_a))
        _, cache_b = self.run_with_cache(board, int(elo_b))
        legal = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
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
        return {"elo_a": int(elo_a), "elo_b": int(elo_b), "steps": steps}

    # ----- move notation ----------------------------------------------------
    # Five representations of the same move circulate in this file, and they are
    # easy to mix up:
    #
    #   Move         chess.Move             real board (python-chess)
    #   uci          "e2e4", "e7e8q"        real board
    #   san          "Nf3", "exd5"          real board, only readable with one
    #   idx          0 .. 4351              policy index — in the model's
    #                                       side-to-move frame (Black mirrored)
    #   (from, to)   0 .. 63 each           canonical squares, rank*8 + file, in
    #                                       that same side-to-move frame
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
        Duplicated as interp_plot._canon and ui.py's realToCanon() so each side
        stands alone — keep the three in sync."""
        rank, file = chess.square_rank(square), chess.square_file(square)
        return (rank if turn == chess.WHITE else 7 - rank) * 8 + file

    @staticmethod
    def _real_square(canon: int, turn: bool) -> int:
        """Canonical index -> python-chess square. Inverse of `_canon_square`."""
        rank, file = divmod(int(canon), 8)
        return chess.square(file, rank if turn == chess.WHITE else 7 - rank)

    @staticmethod
    def _move_squares(idx):
        """Canonical (from, to) squares for a policy-move index (handles promotions).
        Mirrors MAIA3Model.forward's move layout: first 64*64 are from*64+to, then
        256 promotions ordered from_file*32 + to_file*4 + piece (rank7 -> rank8)."""
        if idx < 64 * 64:
            return idx // 64, idx % 64
        idx -= 64 * 64
        from_file, to_file = idx // 32, (idx % 32) // 4
        return 48 + from_file, 56 + to_file          # rank-7 -> rank-8, canonical

    def to_move(self, board: chess.Board, move) -> chess.Move:
        """Any of the five forms above -> chess.Move on `board`: a chess.Move, a
        uci string, a SAN string, a policy index, or a (from, to) pair of
        canonical squares. Raises ValueError on anything unreadable.

        A (from, to) pair carries no promotion piece — the policy layout folds
        every rank7->rank8 promotion onto the same square pair — so it resolves
        to whichever legal move matches those squares."""
        if isinstance(move, chess.Move):
            return move
        if isinstance(move, str):
            try:
                return chess.Move.from_uci(move)
            except ValueError:
                pass
            try:
                return board.parse_san(move)
            except ValueError as e:
                raise ValueError(f"neither uci nor SAN on this board: {move!r}") from e
        if isinstance(move, (tuple, list)):
            frm, to = (self._real_square(s, board.turn) for s in move)
            mv = chess.Move(frm, to)
            if mv in board.legal_moves:
                return mv
            promo = next((m for m in board.legal_moves
                          if m.from_square == frm and m.to_square == to), None)
            if promo is None:
                raise ValueError("no legal move "
                                 f"{chess.square_name(frm)}{chess.square_name(to)}")
            return promo
        idx = int(move)
        if idx not in self.idx_to_move:
            raise ValueError(f"policy index out of range: {idx}")
        uci = self.idx_to_move[idx]
        return chess.Move.from_uci(mirror_move(uci) if board.turn == chess.BLACK else uci)

    def _move_index(self, board: chess.Board, move) -> int:
        """Policy index of a move on this board — this is where the Black mirror
        is applied. Takes any form `to_move` does. Raises KeyError if the move
        isn't in the 4352-move vocabulary (a null move, an under-promotion the
        head doesn't encode)."""
        uci = self.to_move(board, move).uci()
        return self.all_moves_dict[mirror_move(uci) if board.turn == chess.BLACK else uci]

    def move_info(self, board: chess.Board, move) -> dict:
        """One move in every representation at once — the table above as a dict,
        and the single call to reach for when converting:

          move         chess.Move
          uci, san     real-board strings (`san` falls back to uci if illegal)
          idx          policy index, side-to-move frame
          from, to     canonical squares of that index — the frame the lens
                       dicts, heatmaps and `residual_stream` are indexed by
          from_sq, to_sq   the same two squares as python-chess squares
          piece        symbol of the moving piece, None on an empty from-square
          legal        whether the move is legal in this position

        Reads any form `to_move` does, so it converts in every direction:
        `eng.move_info(board, "Nf3")["idx"]`, `eng.move_info(board, 1234)["san"]`."""
        mv = self.to_move(board, move)
        idx = self._move_index(board, mv)
        frm, to = self._move_squares(idx)
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
        are views on this. The `logit_lens` caveat applies everywhere but `enc`."""
        self.evaluate(board, self_elo, oppo_elo)
        info = self.move_info(board, uci)
        idx = info["idx"]
        legal = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
        n_legal = int(legal.sum())
        steps = []
        for name, lab, kind in self._lens_steps():
            logits = self._move_logits(self._activations[name][0].to(self.device))
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
        milliseconds."""
        info = self.move_info(board, uci)
        idx = info["idx"]
        base_logits, _ = self._forward(board, self_elo, oppo_elo)
        base = float(base_logits[idx])
        nb, nh = self.cfg.num_blocks, self.cfg.num_heads
        deltas = []
        for L in range(nb):
            writes = self.head_writes(board, self_elo, oppo_elo, layer=L)  # (H, 64, dim)
            row = []
            for h in range(nh):
                w = writes[h]
                abl_logits, _ = self.run_with_hooks(
                    board, self_elo, oppo_elo,
                    fwd_hooks=[(f"attn_{L:02d}", lambda a, w=w: a - w.to(a.device, a.dtype))],
                    return_type="logits",
                )
                row.append(float(abl_logits[idx]) - base)     # ablated − clean
            deltas.append(row)
        return {"uci": info["uci"], "san": info["san"],
                "base_logit": base, "deltas": deltas}

    # ----- activation dump --------------------------------------------------
    def save_activations(self, filename: str, meta: dict | None = None) -> str:
        """Persist the most recent forward's residual-stream snapshot, plus a
        `meta` entry. Each tensor is (64, dim_vit), on CPU. Keys are every name
        in `hook_points` — see `_register_hooks` for what each one is.

        `activation_dir` is created here, on the first save, rather than in
        __init__."""
        snap = {k: v.squeeze(0).clone() for k, v in self._activations.items()}
        snap["meta"] = meta or {}
        self.activation_dir.mkdir(parents=True, exist_ok=True)
        path = self.activation_dir / filename
        torch.save(snap, path)
        return str(path)
