"""Interpretability core for chessformers that treat the 64 squares as tokens.

`ChessformerEngine` class contains every read and
intervention path below; the subclasses supply the model, its input
tokens, and its move vocabulary:

  MaiaEngine    a Maia-3 checkpoint (5M / 23M / 79M / 3M-ablation), conditioned
                on a rating pair — `self_elo` / `oppo_elo` are real inputs
  LeelaEngine   Leela Chess Zero BT4 (185M params, 15 layers x 32 heads x 1024d),
                

  evaluate              one forward pass yielding the full normalized policy over
                        legal moves (in descending order), the W/D/L for the side to
                        move, and the moves-left estimate when the model has one
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
  qk_scores             one layer's scaled QKᵀ logits
  gab_bias              one layer's generated bias
  gab_coeffs            the coefficients head h applies to the GAB bank
  gab_templates         the static square-pair template bank every layer shares
  head_writes           exact residual writes of one layer's attention

  
  ablate_head           forward pass with one head's write removed, exactly
  ablate_grid           the carrier heatmap of one move: every head ablated in
                        turn, Δlogit = ablated − clean (negative = the head
                        carried it)
  ablate_grid_batch     ablate_grid called with batches of many positions at once;
                        minimizes unnecessary syncing to cpu to take advantage of gpus
  carrier_neurons       the carrier table of one move at neuron grain: every MLP
                        unit scored at once by attribution patching, with its
                        per-square footprint; optionally the top ones re-measured
                        by exact zero-ablation
  neuron_activation     one MLP unit's activation on every square (and its write's size)
  neuron_overview       every layer's most active MLP units on this position, for
                        flipping through the network layer by layer
  ablate_neuron         forward pass with one MLP unit's write removed exactly, on
                        every square
  residual_stream       per-square views of how the stream is built up, one row
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
  remove_hooks          detach the capture hooks, for a bare forward with no
                        CPU copies

Module level, beside the classes: `load_engine(alias)` builds either engine
from one alias table (`resolve_engine` validates an alias without loading
weights, `format_engine_list` prints the table; neither family is a default),
`build_cfg` builds the args-namespace a Maia-3 model expects, and
`pick_device` resolves the torch device (explicit > $CHESSFORMER_DEVICE >
cuda > mps > cpu).

Every tensor a read path returns is on CPU, in the model's canonical
side-to-move frame (square = rank*8 + file) — the same frame for both models,
Black mirrored vertically. Depth reads the same as in the figures: `emb`, then
`aN`/`mN` for layer N's attention and MLP sub-layers, then — on Maia-3, whose
heads read through a final LayerNorm — `enc`. `depth_points` hands you that
axis directly (18 points on every Maia-3 size, 31 on BT4).

engine.py imports cleanly into a notebook and is called by interp_plot.py
(static figures),  interp_widget.py (interactive panels), and the standalone
app in bridge.py/app.py/ui.py.
"""
import math
import os
import types
import gzip
import struct
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# maia3 is not on PyPI, yet both engines depend on it: MaiaEngine for the model
# and checkpoint work, LeelaEngine for the encoder block class it shares.

try:
    from maia3.models import MAIA3Model, EncoderOnlyBlock  # noqa: F401
    from maia3.uci import load_model, sample_from_logits
    from maia3.dataset import tokenize_board, get_historical_tokens
    from maia3.utils import get_all_possible_moves, mirror_move
    from maia3.model_registry import (MODEL_SPECS, ModelResolutionError, resolve_model_spec,
                                      apply_model_config, resolve_checkpoint_path)
except ModuleNotFoundError as exc:
    if exc.name != "maia3" and not str(exc.name or "").startswith("maia3."):
        raise
    raise ModuleNotFoundError(
        "chessformer_lens needs the Maia-3 model code, which is not on PyPI "
        "and so is not installed by `pip install chessformer_lens`.\n\n"
        "    pip install https://github.com/CSSLab/maia3/archive/1e13597c42d4858b7cfd7cfdae01e297263364b2.zip\n"
    ) from exc
except ImportError as exc:
    # maia3 imports torch.nn.RMSNorm, which only exists from torch 2.4.
    if "RMSNorm" not in str(exc):
        raise
    raise ImportError(
        f"Maia-3 needs torch >= 2.4; this environment has {torch.__version__}.\n"
        "    pip install --upgrade 'torch>=2.4'"
    ) from exc

__all__ = ["ChessformerEngine", "MaiaEngine", "LeelaEngine", "BT4Model",
           "load_engine", "resolve_engine", "format_engine_list",
           "build_cfg", "pick_device"]


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


# ----- Leela Chess Zero BT4 ---------------------------------------------------
# BT4 (1024 dim, 15 layers, 32 heads) is Post-LN like Maia-3, and its smolgen bias
# is the same generator as Maia-3's GAB. So each block is maia3's EncoderOnlyBlock
# with four patches (see _bt4_block), and the engine reads both models through the
# same attribute names. BT4 also scales every sublayer by alpha (0.427) before the
# residual add; that is folded into the output projections at load time, so each
# sublayer's output is exactly its write to the stream. Weights are read from the
# ONNX file `lc0 leela2onnx` produces.

_BT4_LN_EPS = 1e-3


class _BT4Embedding(nn.Module):
    """planes (B, 112, 8, 8) -> (B, 64, d), the stream entering block 0.

    The current position's 12 piece planes go through one linear layer over the
    whole board, so every square token sees every piece. That is concatenated
    to each square's 112 inputs, projected, gated per square, and passed
    through one FFN sublayer."""

    def __init__(self, d: int, ff: int, pre: int, planes: int = 112):
        super().__init__()
        self.preproc = nn.Linear(64 * 12, 64 * pre)
        self.proj = nn.Linear(planes + pre, d)
        self.norm = nn.LayerNorm(d, eps=_BT4_LN_EPS)
        self.mul_gate = nn.Parameter(torch.ones(64, d))
        self.add_gate = nn.Parameter(torch.zeros(64, d))
        self.linear1 = nn.Linear(d, ff)
        self.linear2 = nn.Linear(ff, d)              # alpha folded in at load time
        self.norm2 = nn.LayerNorm(d, eps=_BT4_LN_EPS)

    def forward(self, planes):
        B = planes.size(0)
        x = planes.reshape(B, planes.size(1), 64).transpose(1, 2)         # (B, 64, 112), sq = rank*8 + file
        pos = self.preproc(x[:, :, :12].reshape(B, -1)).view(B, 64, -1)   # (B, 64, pre)
        x = self.norm(F.mish(self.proj(torch.cat([x, pos], dim=-1))))
        x = x * self.mul_gate + self.add_gate
        return self.norm2(x + self.linear2(F.mish(self.linear1(x))))


class _BT4Encoder(nn.Module):
    """The block stack, like maia3's CustomTransformerEncoder but with no final
    norm (norm = None): BT4's heads read the last block directly."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = None

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


def _bt4_block(d, heads, ff, gen, per_square, intermediate, gab_weight):
    """maia3's EncoderOnlyBlock, patched to BT4: mish FFN, swish smolgen,
    LayerNorm eps 1e-3, and no bias on the smolgen compress layer."""
    cfg = types.SimpleNamespace(
        use_gab=True, use_relative_bias=False, gab_gen_size=gen,
        gab_per_square_dim=per_square, gab_intermediate_dim=intermediate,
        omit_qkv_biases=False, use_rms_norm=False, activation="gelu")
    blk = EncoderOnlyBlock(cfg, d_model=d, nhead=heads, dim_feedforward=ff,
                           dropout=0.0, gab_weight=gab_weight)
    blk.activation = F.mish                      # FFN activation
    sa = blk.self_attn
    sa.sm_act = nn.SiLU()                        # smolgen activation (swish)
    sa.sm1.bias = None                           # the compress layer has no bias
    for ln in (blk.norm1, blk.norm2, sa.ln1, sa.ln2):
        ln.eps = _BT4_LN_EPS
    return blk


class BT4Model(nn.Module):
    """Lc0 BT4 in PyTorch, with MAIA3Model's attribute names where the engine
    reads them (transformer.layers[i], gab_shared_weight (64*64, gen)).

    forward(planes (B, 112, 8, 8)) -> logits_move (B, 4288), logits_value (B, 3),
    moves_left (B,).
      logits_move    64*64 square pairs (from*64 + to), then 192 promotions
                     (4096 + from_file*24 + to_file*3 + piece, piece in q r b).
                     A knight promotion uses its square pair's logit.
      logits_value   win, draw, loss for the side to move
      moves_left     expected plies to the end of the game
    `policy_map` (1858,) is lc0's gather to its own move list, kept for testing."""

    def __init__(self, num_blocks=15, dim=1024, num_heads=32, ff=1536, gen=256,
                 per_square=32, intermediate=256, pre=512, policy_dim=1024,
                 value_dim=128, mlh_dim=32, policy_scale=1 / 32):
        super().__init__()
        self.cfg = types.SimpleNamespace(
            num_blocks=num_blocks, dim_vit=dim, num_heads=num_heads,
            mlp_ratio=ff / dim, gab_gen_size=gen, gab_per_square_dim=per_square,
            gab_intermediate_dim=intermediate, head_hid_dim=policy_dim, use_gab=True)
        self.embedding = _BT4Embedding(dim, ff, pre)
        self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, gen))
        self.transformer = _BT4Encoder(
            [_bt4_block(dim, num_heads, ff, gen, per_square, intermediate, self.gab_shared_weight)
             for _ in range(num_blocks)])
        # attention policy head
        self.policy_dense = nn.Linear(dim, policy_dim)
        self.policy_q = nn.Linear(policy_dim, policy_dim)
        self.policy_k = nn.Linear(policy_dim, policy_dim)
        self.policy_promo = nn.Parameter(torch.empty(policy_dim, 4))   # q, r, b, n offsets
        self.policy_scale = policy_scale
        # value (WDL) and moves-left heads
        self.value_embed = nn.Linear(dim, value_dim)
        self.value_dense1 = nn.Linear(64 * value_dim, 128)
        self.value_dense2 = nn.Linear(128, 3)
        self.mlh_embed = nn.Linear(dim, mlh_dim)
        self.mlh_dense1 = nn.Linear(64 * mlh_dim, 128)
        self.mlh_dense2 = nn.Linear(128, 1)
        self.register_buffer("policy_map", torch.zeros(1858, dtype=torch.long), persistent=False)

    def policy_logits(self, x):
        """(B, 64, dim) encoder output -> (B, 4288) move logits."""
        B = x.size(0)
        h = F.mish(self.policy_dense(x))
        q, k = self.policy_q(h), self.policy_k(h)
        scores = (q @ k.transpose(-2, -1)) * self.policy_scale              # (B, 64, 64)
        offs = k[:, 56:64] @ self.policy_promo                             # (B, 8 to-files, 4)
        offs = (offs[:, :, :3] + offs[:, :, 3:4]).reshape(B, 1, 24)         # q, r, b, each + the n offset
        base = scores[:, 48:56, 56:64].reshape(B, 64, 1).expand(-1, -1, 3).reshape(B, 8, 24)
        return torch.cat([scores.reshape(B, 4096), (base + offs).reshape(B, 192)], dim=1)

    def value_logits(self, x):
        v = F.mish(self.value_embed(x)).reshape(x.size(0), -1)
        return self.value_dense2(F.mish(self.value_dense1(v)))             # (B, 3): W, D, L

    def moves_left(self, x):
        m = F.mish(self.mlh_embed(x)).reshape(x.size(0), -1)
        return F.relu(self.mlh_dense2(F.mish(self.mlh_dense1(m)))).squeeze(-1)

    def forward(self, planes):
        x = self.transformer(self.embedding(planes))
        return self.policy_logits(x), self.value_logits(x), self.moves_left(x)


def _load_bt4(path, device) -> BT4Model:
    """Load BT4 from lc0's own weights file (.pb.gz) or an `lc0 leela2onnx`
    export (.onnx)."""
    name = str(path).lower()
    W = _onnx_arrays(path) if name.endswith(".onnx") else _pb_arrays(path)
    return _bt4_from_arrays(W, device)


def _onnx_arrays(path) -> dict:
    """{initializer name: tensor} from an `lc0 leela2onnx` export."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError("Reading an lc0 .onnx export needs the `onnx` package "
                          "(or pass the .pb.gz instead):\n    pip install onnx") from exc
    graph = onnx.load(str(path), load_external_data=True).graph
    return {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}


# ----- lc0's .pb.gz weights ----------------------------------------------------
# Converting the .pb.gz weights to .pt by hand instead of with ONNX

def _varint(buf, i):
    out, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _pb_fields(buf) -> dict:
    """One protobuf message -> {field number: [values]}. A value is an int
    (varint), or a memoryview (length-delimited, fixed32, fixed64)."""
    out, i, end = {}, 0, len(buf)
    while i < end:
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            v, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            v, i = buf[i:i + n], i + n
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.setdefault(key >> 3, []).append(v)
    return out


def _pb_msg(msg, field) -> dict:
    return _pb_fields(msg[field][0]) if field in msg else {}


def _pb_int(msg, field, default=0) -> int:
    return msg[field][0] if field in msg else default


def _pb_layer(msg, field) -> torch.Tensor:
    """A stored Layer as a flat float32 tensor. The multiply-add is rounded once
    (lc0's build fuses it), which reproduces `lc0 leela2onnx` bit for bit."""
    f = _pb_msg(msg, field)
    if 3 not in f:
        return torch.zeros(0)
    lo = np.float32(struct.unpack("<f", f[1][0])[0]) if 1 in f else np.float32(0)
    hi = np.float32(struct.unpack("<f", f[2][0])[0]) if 2 in f else np.float32(0)
    x = np.frombuffer(f[3][0], dtype="<u2").astype(np.float32) / np.float32(0xFFFF)
    return torch.from_numpy((x.astype(np.float64) * np.float64(hi - lo) + np.float64(lo)).astype(np.float32))


def _pb_arrays(path) -> dict:
    """lc0 .pb.gz -> {ONNX initializer name: tensor}, in the ONNX layouts.
    Refuses nets this engine does not implement, rather than misreading them."""
    try:
        with gzip.open(path, "rb") as fh:
            net = _pb_fields(memoryview(fh.read()))
        magic = net[1][0]
        ok = isinstance(magic, memoryview) and struct.unpack("<I", magic)[0] == 0x1C0
    except (OSError, ValueError, IndexError, KeyError, struct.error):
        ok = False
    if not ok:
        raise ValueError(f"{path} is not an lc0 weights file")
    fmt = _pb_msg(net, 4)
    nf = _pb_msg(fmt, 2)
    w = _pb_msg(net, 10)

    problems = []
    if _pb_int(fmt, 1) != 1:
        problems.append("weights are not LINEAR16-encoded")
    if _pb_int(nf, 1) != 1:
        problems.append(f"input format {_pb_int(nf, 1)}: only the classical 112-plane "
                        "input (format 1) is implemented; this net uses a canonical one")
    if _pb_int(nf, 3) not in (6, 7, 134) or 27 not in w or 2 in w:
        problems.append("not an attention-body network")
    if _pb_int(nf, 4) != 3 or _pb_int(nf, 5) != 2 or _pb_int(nf, 6) != 1:
        problems.append("needs an attention policy, a WDL value head and a moves-left head")
    if _pb_int(nf, 7) != 1 or _pb_int(nf, 9) not in (0, 1) or _pb_int(nf, 8) != 7:
        problems.append("needs mish body and FFN activations and swish smolgen")
    if 37 not in w or 35 not in w:
        problems.append("needs the dense input embedding and smolgen")
    if problems:
        raise NotImplementedError(f"{path}: this lc0 net is not supported: " + "; ".join(problems)
                                  + ". Tested on BT4-1024x15x32h.")

    def dense(msg, field, n_in, n_out):            # stored (out, in) -> ONNX (in, out)
        return _pb_layer(msg, field).reshape(n_out, n_in).t().contiguous()

    W = {}
    emb = _pb_layer(w, 26).numel()
    pre = _pb_layer(w, 38).numel() // 64
    W["/attn_body/embedding/preprocess/matmul/w"] = dense(w, 37, 64 * 12, 64 * pre)
    W["/attn_body/embedding/preprocess/add/w"] = _pb_layer(w, 38)
    W["/attn_body/matmul/w"] = dense(w, 25, 112 + pre, emb)
    W["/attn_body/add/w"] = _pb_layer(w, 26)
    W["/attn_body/ln/w/scale"], W["/attn_body/ln/w/bias"] = _pb_layer(w, 39), _pb_layer(w, 40)
    W["/ip_mul_gate/w"] = dense(w, 33, 64, emb)
    W["/ip_add_gate/w"] = dense(w, 34, 64, emb)

    encoders = [_pb_fields(e) for e in w[27]]
    nb = len(encoders)
    alpha = torch.tensor([np.power(np.float32(2 * nb), np.float32(-0.25))], dtype=torch.float32)

    def ffn(prefix, f):
        dff = _pb_layer(f, 2).numel()
        W[prefix + "ffn/dense1/w/w"] = dense(f, 1, emb, dff)
        W[prefix + "ffn/dense1/b/w"] = _pb_layer(f, 2)
        W[prefix + "ffn/dense2/w/w"] = dense(f, 3, dff, emb)
        W[prefix + "ffn/dense2/b/w"] = _pb_layer(f, 4)
        W[prefix + "ffn/alpha/w"] = alpha

    ffn("/attn_body/", _pb_msg(w, 41))
    W["/attn_body/ln2/w/scale"], W["/attn_body/ln2/w/bias"] = _pb_layer(w, 42), _pb_layer(w, 43)

    heads = _pb_int(w, 28)
    gen = _pb_layer(w, 35).numel() // 4096
    W["/const/smolgen_w"] = dense(w, 35, gen, 4096)
    for i, e in enumerate(encoders):
        o, mha = f"/encoder{i}/", _pb_msg(e, 1)
        d_model = _pb_layer(mha, 2).numel()
        for c, fw, fb in (("Q", 1, 2), ("K", 3, 4), ("V", 5, 6)):
            W[o + f"mha/{c}/w/w"] = dense(mha, fw, emb, d_model)
            W[o + f"mha/{c}/b/w"] = _pb_layer(mha, fb)
        W[o + "mha/QK/scale/w"] = torch.tensor([np.float32(1) / np.sqrt(np.float32(d_model // heads))])
        W[o + "mha/out/dense/w/w"] = dense(mha, 7, d_model, emb)
        W[o + "mha/out/dense/b/w"] = _pb_layer(mha, 8)
        W[o + "alpha*input/w"] = alpha
        sm = _pb_msg(mha, 9)
        hidden_ch = _pb_layer(sm, 1).numel() // emb
        hs, gh = _pb_layer(sm, 3).numel(), _pb_layer(sm, 7).numel()
        W[o + "smolgen/compress/w"] = dense(sm, 1, emb, hidden_ch)
        W[o + "smolgen/dense1/w/w"] = dense(sm, 2, 64 * hidden_ch, hs)
        W[o + "smolgen/dense1/b/w"] = _pb_layer(sm, 3)
        W[o + "smolgen/ln1/w/scale"], W[o + "smolgen/ln1/w/bias"] = _pb_layer(sm, 4), _pb_layer(sm, 5)
        W[o + "smolgen/dense2/w/w"] = dense(sm, 6, hs, gh)
        W[o + "smolgen/dense2/b/w"] = _pb_layer(sm, 7)
        W[o + "smolgen/ln2/w/scale"], W[o + "smolgen/ln2/w/bias"] = _pb_layer(sm, 8), _pb_layer(sm, 9)
        W[o + "ln1/w/scale"], W[o + "ln1/w/bias"] = _pb_layer(e, 2), _pb_layer(e, 3)
        ffn(o, _pb_msg(e, 4))
        W[o + "ln2/w/scale"], W[o + "ln2/w/bias"] = _pb_layer(e, 5), _pb_layer(e, 6)
    W["/const/encoder0/mha/shape"] = torch.tensor([-1, 64, heads, d_model // heads])

    # policy: the "vanilla" head; its embedding may be shared across heads
    heads_msg = _pb_msg(w, 45)
    pol = _pb_msg(heads_msg, 3)
    if 8 in pol:
        raise NotImplementedError(f"{path}: policy-head encoder layers are not supported")
    pw = pol if 1 in pol else heads_msg
    pemb, pd = _pb_layer(pw, 2).numel(), _pb_layer(pol, 4).numel()
    W["/policy/dense1/matmul/w"] = dense(pw, 1, emb, pemb)
    W["/policy/dense1/add/w"] = _pb_layer(pw, 2)
    W["/policy/Q/matmul/w"], W["/policy/Q/add/w"] = dense(pol, 3, pemb, pd), _pb_layer(pol, 4)
    W["/policy/K/matmul/w"], W["/policy/K/add/w"] = dense(pol, 5, pemb, pd), _pb_layer(pol, 6)
    W["/policy/scale/w"] = torch.tensor([np.float32(1) / np.sqrt(np.float32(pd))])
    W["/policy/promotion/matmul/w"] = dense(pol, 7, pd, 4)

    # value: the "winner" head; moves-left
    val = _pb_msg(_pb_msg(w, 44), 1)
    vc = _pb_layer(val, 2).numel()
    W["/value/embed/matmul/w"], W["/value/embed/add/w"] = dense(val, 1, emb, vc), _pb_layer(val, 2)
    W["/value/dense1/matmul/w"], W["/value/dense1/add/w"] = dense(val, 3, 64 * vc, 128), _pb_layer(val, 4)
    W["/value/dense2/matmul/w"], W["/value/dense2/add/w"] = dense(val, 5, 128, 3), _pb_layer(val, 6)
    mc, m1 = _pb_layer(w, 32).numel(), _pb_layer(w, 14).numel()
    W["/mlh/embed/matmul/w"], W["/mlh/embed/add/w"] = dense(w, 31, emb, mc), _pb_layer(w, 32)
    W["/mlh/dense1/matmul/w"], W["/mlh/dense1/add/w"] = dense(w, 13, 64 * mc, m1), _pb_layer(w, 14)
    W["/mlh/dense2/matmul/w"], W["/mlh/dense2/add/w"] = dense(w, 15, m1, 1), _pb_layer(w, 16)
    return W


def _bt4_from_arrays(W, device) -> BT4Model:
    """Build a BT4Model from ONNX-named weights. Sizes come from the shapes.
    ONNX MatMul weights are (in, out), so they are transposed; Q, K, V are
    packed into in_proj; alpha is folded into each sublayer's output projection."""
    def lin(name):                       # MatMul weight -> nn.Linear weight
        return W[name].t().contiguous()

    nb = 0
    while f"/encoder{nb}/ln1/w/scale" in W:
        nb += 1
    d = W["/attn_body/matmul/w"].shape[1]
    heads, dh = (int(v) for v in W["/const/encoder0/mha/shape"][2:])
    gen = W["/const/smolgen_w"].shape[0]
    per_sq = W["/encoder0/smolgen/compress/w"].shape[1]
    inter = W["/encoder0/smolgen/dense1/w/w"].shape[1]
    ff = W["/encoder0/ffn/dense1/w/w"].shape[1]
    pre = W["/attn_body/embedding/preprocess/matmul/w"].shape[1] // 64
    model = BT4Model(nb, d, heads, ff, gen, per_sq, inter, pre,
                     policy_dim=W["/policy/Q/matmul/w"].shape[1],
                     value_dim=W["/value/embed/matmul/w"].shape[1],
                     mlh_dim=W["/mlh/embed/matmul/w"].shape[1],
                     policy_scale=float(W["/policy/scale/w"]))
    for i in range(nb):     # nn.MultiheadAttention assumes a 1/sqrt(d_head) scale
        s = float(W[f"/encoder{i}/mha/QK/scale/w"])
        assert abs(s - dh ** -0.5) < 1e-6, f"encoder{i}: QK scale {s} is not 1/sqrt({dh})"

    sd = {}

    def put(key, weight, bias=None, alpha=1.0):
        sd[key + ".weight"] = weight * alpha
        if bias is not None:
            sd[key + ".bias"] = bias * alpha

    put("embedding.preproc", lin("/attn_body/embedding/preprocess/matmul/w"),
        W["/attn_body/embedding/preprocess/add/w"])
    put("embedding.proj", lin("/attn_body/matmul/w"), W["/attn_body/add/w"])
    put("embedding.norm", W["/attn_body/ln/w/scale"], W["/attn_body/ln/w/bias"])
    sd["embedding.mul_gate"] = W["/ip_mul_gate/w"]
    sd["embedding.add_gate"] = W["/ip_add_gate/w"]
    put("embedding.linear1", lin("/attn_body/ffn/dense1/w/w"), W["/attn_body/ffn/dense1/b/w"])
    put("embedding.linear2", lin("/attn_body/ffn/dense2/w/w"), W["/attn_body/ffn/dense2/b/w"],
        alpha=float(W["/attn_body/ffn/alpha/w"]))
    put("embedding.norm2", W["/attn_body/ln2/w/scale"], W["/attn_body/ln2/w/bias"])
    shared = W["/const/smolgen_w"].t().contiguous()
    sd["gab_shared_weight"] = shared
    for i in range(nb):
        p, o = f"transformer.layers.{i}.", f"/encoder{i}/"
        sd[p + "self_attn.mha.in_proj_weight"] = torch.cat([lin(o + f"mha/{c}/w/w") for c in "QKV"])
        sd[p + "self_attn.mha.in_proj_bias"] = torch.cat([W[o + f"mha/{c}/b/w"] for c in "QKV"])
        put(p + "self_attn.mha.out_proj", lin(o + "mha/out/dense/w/w"), W[o + "mha/out/dense/b/w"],
            alpha=float(W[o + "alpha*input/w"]))
        put(p + "self_attn.sm1", lin(o + "smolgen/compress/w"))
        put(p + "self_attn.sm2", lin(o + "smolgen/dense1/w/w"), W[o + "smolgen/dense1/b/w"])
        put(p + "self_attn.ln1", W[o + "smolgen/ln1/w/scale"], W[o + "smolgen/ln1/w/bias"])
        put(p + "self_attn.sm3", lin(o + "smolgen/dense2/w/w"), W[o + "smolgen/dense2/b/w"])
        put(p + "self_attn.ln2", W[o + "smolgen/ln2/w/scale"], W[o + "smolgen/ln2/w/bias"])
        sd[p + "self_attn.gab_weight"] = shared
        put(p + "norm1", W[o + "ln1/w/scale"], W[o + "ln1/w/bias"])
        put(p + "linear1", lin(o + "ffn/dense1/w/w"), W[o + "ffn/dense1/b/w"])
        put(p + "linear2", lin(o + "ffn/dense2/w/w"), W[o + "ffn/dense2/b/w"],
            alpha=float(W[o + "ffn/alpha/w"]))
        put(p + "norm2", W[o + "ln2/w/scale"], W[o + "ln2/w/bias"])
    put("policy_dense", lin("/policy/dense1/matmul/w"), W["/policy/dense1/add/w"])
    put("policy_q", lin("/policy/Q/matmul/w"), W["/policy/Q/add/w"])
    put("policy_k", lin("/policy/K/matmul/w"), W["/policy/K/add/w"])
    sd["policy_promo"] = W["/policy/promotion/matmul/w"]
    for head in ("value", "mlh"):
        put(f"{head}_embed", lin(f"/{head}/embed/matmul/w"), W[f"/{head}/embed/add/w"])
        put(f"{head}_dense1", lin(f"/{head}/dense1/matmul/w"), W[f"/{head}/dense1/add/w"])
        put(f"{head}_dense2", lin(f"/{head}/dense2/matmul/w"), W[f"/{head}/dense2/add/w"])
    model.load_state_dict(sd, strict=True)
    if "/const/mapping_table" in W:                        # only in ONNX exports
        model.policy_map.copy_(W["/const/mapping_table"].long())
    return model.to(device).eval()


def _leela_all_moves() -> list[str]:
    """BT4's 4288 moves as uci, side-to-move frame, in BT4Model's policy order.
    No knight promotions (they use the square pair)."""
    moves = [chess.square_name(f) + chess.square_name(t) for f in range(64) for t in range(64)]
    moves += [f"{ff}7{tf}8{p}" for ff in "abcdefgh" for tf in "abcdefgh" for p in "qrb"]
    return moves


_STARTPOS_FIELDS = chess.STARTING_FEN.split(" ")[:4]


def _bits(mask: int) -> torch.Tensor:
    """python-chess bitboard -> (64,) float plane, square = rank*8 + file."""
    return torch.tensor([(mask >> s) & 1 for s in range(64)], dtype=torch.float32)


def _leela_planes(board: chess.Board) -> torch.Tensor:
    """lc0's 112 input planes (INPUT_CLASSICAL_112_PLANE) for `board`: (112, 8, 8).
    Ported from lc0's encoder.cc with history-fill=fen_only. For Black every
    position is mirrored, so "ours" is always White and squares match the
    engine's canonical frame.

      0-103     8 history slots x 13 planes: our P N B R Q K, their P N B R Q K,
                repetition. Slot k is k plies ago (from board.move_stack).
                Missing slots repeat the earliest position, with any pending
                en passant push undone, or stay zero after the start position.
      104-107   castling: our O-O-O, our O-O, their O-O-O, their O-O
      108       ones if Black to move
      109       halfmove clock
      110, 111  zeros, ones"""
    planes = torch.zeros(112, 64)
    game = [board.root()]                                  # oldest first
    for mv in board.move_stack:
        nxt = game[-1].copy(stack=False)
        nxt.push(mv)
        game.append(nxt)
    keys = [(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
             b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK],
             b.turn, b.clean_castling_rights(), b.ep_square) for b in game]
    reps = [keys[:i].count(k) for i, k in enumerate(keys)]
    we_black = board.turn == chess.BLACK

    n = len(game)
    for k in range(8):
        i = n - 1 - k
        filled = i < 0
        if filled:
            if game[0].fen().split(" ")[:4] == _STARTPOS_FIELDS:
                break                                      # no history before move 1
            i = 0
        pos = game[i]
        view = pos.mirror() if we_black else pos           # side to move = White
        base = 13 * k
        for colour, off in ((chess.WHITE, 0), (chess.BLACK, 6)):
            for j, piece in enumerate(chess.PIECE_TYPES):   # P N B R Q K
                planes[base + off + j] = _bits(view.pieces_mask(piece, colour))
        if reps[i] >= 1:
            planes[base + 12] = 1.0
        if filled and pos.ep_square is not None:           # undo the double push
            f = chess.square_file(pos.ep_square)
            if view.turn == chess.WHITE:                   # theirs: rank 5 -> 7
                planes[base + 6, 32 + f], planes[base + 6, 48 + f] = 0.0, 1.0
            else:                                          # ours: rank 4 -> 2
                planes[base + 0, 24 + f], planes[base + 0, 8 + f] = 0.0, 1.0

    view = board.mirror() if we_black else board
    planes[104] = float(view.has_queenside_castling_rights(chess.WHITE))
    planes[105] = float(view.has_kingside_castling_rights(chess.WHITE))
    planes[106] = float(view.has_queenside_castling_rights(chess.BLACK))
    planes[107] = float(view.has_kingside_castling_rights(chess.BLACK))
    planes[108] = float(we_black)
    planes[109] = float(board.halfmove_clock)
    planes[111] = 1.0
    return planes.view(112, 8, 8)


class ChessformerEngine:
    """Hook-based interpretability engine over a square-token chess transformer.

    Provides read paths (run_with_cache, logit_lens, residual_stream, attention,
    gab_*) and intervention paths (run_with_hooks, ablate_head, ablate_grid,
    ablate_grid_batch). Every tensor a read path returns is on CPU.

    Subclass differences:
      has_conditioning    the rating pair is a real input (Maia-3) or ignored (BT4)
      has_generated_bias  attention carries a generated square-pair bias
                          (GAB / smolgen), so the gab_* methods work
      has_mlh             evaluate() carries a moves-left estimate"""

    has_conditioning = True
    has_generated_bias = True
    has_mlh = False

    def __init__(self, activation_dir="activations"):
        """Finish construction once the subclass has set `cfg`, `spec`, `device`,
        `model` and `all_moves`: build the index <-> uci tables, note
        `activation_dir` for later, and register forward hooks that copy every
        sub-layer's output to CPU on each forward. That copy is what makes the
        read paths work; it also costs a handful of device->host transfers per
        forward."""
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        # Not created here: `save_activations` makes it on first write, so
        # importing the engine in a notebook leaves no directory behind.
        self.activation_dir = Path(activation_dir)

        self._activations: dict[str, torch.Tensor] = {}
        self._aux: dict = {}              # extra head outputs of the last forward (e.g. "mlh")
        self._capture_on_device = False   # see _register_hooks; ablate_grid_batch flips it
        self._hooks: list = []
        self.hook_points: dict[str, torch.nn.Module] = {}   # name -> module (read or patch)
        self._gab_templates: torch.Tensor | None = None     # lazy (gen, 64, 64) cache
        self._register_hooks()

    # ----- activation hooks -------------------------------------------------
    def _register_hooks(self):
        """Install the permanent capture hooks: 4 per block plus the input and
        (on Maia-3) the output norm, overwritten on every forward, so the
        snapshot is always the most recent position.

          embed_in      the stream entering block 0 (token_projection out)
          attn_NN       block NN's attention write (self_attn out, post out_proj)
          postattn_NN   the running stream after attention (norm1 out)
          mlp_NN        block NN's MLP write (linear2 out)
          block_NN      the running stream after the whole block (post-LN)
          encoder_out   after the final encoder norm (Maia-3; BT4's heads read
                        block_14 directly, so it has no such point)

        Both models are Post-LN, x = norm(x + sublayer(x)), so attn_NN/mlp_NN
        are the vectors *added* to the stream (dropout is identity in eval),
        while the other four are the stream itself. Everything is copied to
        CPU on the way out. `hook_points` records name -> module for all of
        these, so `run_with_hooks` can patch anywhere you can read.

        `_capture_on_device` suspends that CPU copy. On unified memory it costs
        nothing either way, but on a discrete GPU it is 34 pageable transfers
        per forward pass — enough to dominate a sweep that runs hundreds of
        them. Callers that set it must restore it, and must not read the
        snapshot from a path that assumes CPU tensors (`save_activations`)."""
        def make_hook(name):
            def hook(_module, _inp, out): #classic PyTorch
                t = out[0] if isinstance(out, tuple) else out #get first element of tuple
                self._activations[name] = t.detach() if self._capture_on_device \
                    else t.detach().to("cpu") #remove tensor from autograd and move it to cpu or not
            return hook

        def add(name, module):
            self.hook_points[name] = module
            self._hooks.append(module.register_forward_hook(make_hook(name)))

        add("embed_in", self._embed_module())
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
        if self._final_norm() is not None:
            add("encoder_out", self._final_norm())

    def remove_hooks(self):
        """Detach the capture hooks, for a bare forward with no CPU copies.

        `hook_points` stays populated so `run_with_hooks` keeps working, but
        the read paths will KeyError once `_activations` stops being refreshed.
        There is no re-register; build a new engine."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ----- per-model members ------------------------------------------------
    # The whole seam between the two models. Each subclass defines these (and
    # `LeelaEngine` two of the vocabulary helpers further down); every public
    # method is written against them.
    def _embed_module(self) -> torch.nn.Module:
        """The module whose output is the stream entering block 0 (`embed_in`)."""
        raise NotImplementedError

    def _final_norm(self):
        """The final encoder norm the heads read through (`encoder_out` / `enc`),
        or None when the heads read the last block directly."""
        raise NotImplementedError

    def tokens(self, board: chess.Board) -> torch.Tensor:
        """The position as the model's input tensor, batch of one, on the device."""
        raise NotImplementedError

    def _forward(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One raw forward pass. Resets and repopulates `self._activations` via the
        hooks, stashes any extra head output in `self._aux`, and returns
        (logits_move (n_moves,), logits_value (3,)) as floats."""
        raise NotImplementedError

    def _forward_batch(self, boards, self_elo, oppo_elo=None):
        """`_forward` over a list of boards in one pass: (B, n_moves) move logits.

        The capture hooks fire as usual, so afterwards `_activations` holds this
        batch's (B, 64, dim) tensors rather than one position's (1, 64, dim).
        The read paths index [0] and so would only ever see the first board —
        this is for the intervention paths, which stay batch-aware throughout."""
        raise NotImplementedError

    def _move_logits(self, x):
        """Full (n_moves,) move logits from one position's residual x (64, dim),
        replicating the model's policy head. `x` must be exactly (64, dim)."""
        raise NotImplementedError

    @staticmethod
    def _move_squares(idx):
        """Canonical (from, to) squares for a policy-move index (handles promotions)."""
        raise NotImplementedError

    @staticmethod
    def _wdl(logits_value) -> dict:
        """The value head's three logits -> {"win", "draw", "loss"} for the side to move."""
        raise NotImplementedError

    # ----- policy index <-> move ---------------------------------------------
    def _decode_idx(self, board: chess.Board, idx: int) -> chess.Move:
        """Policy index -> chess.Move on the real board (un-mirroring for Black),
        with no legality check. Raises ValueError on an undecodable uci."""
        uci = self.idx_to_move[int(idx)]
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        return chess.Move.from_uci(uci)

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
        mask = torch.zeros(len(self.all_moves), dtype=torch.bool)
        for mv in board.legal_moves:
            try:
                mask[self._move_index(board, mv)] = True
            except KeyError:
                pass
        return mask

    # ----- forward / policy -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One forward pass. Returns the full normalized policy over legal moves
        (descending), the WDL for the side to move, the moves-left estimate
        (None on Maia-3), and stashes activations."""
        logits, logits_value = self._forward(board, self_elo, oppo_elo)

        legal_mask = self._legal_mask(board).to(self.device)
        policy = []
        if bool(legal_mask.any()):                  # may be empty for hand-edited positions
            logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)   # normalized over legal moves
            for idx in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist():
                mv = self._idx_to_move(board, idx)
                if mv is not None:
                    policy.append((mv.uci(), float(probs[idx])))
            policy.sort(key=lambda x: x[1], reverse=True)

        return {
            "policy": policy,                                   # [(uci, prob)] desc
            "wdl": self._wdl(logits_value),                     # side-to-move perspective
            "mlh": self._aux.get("mlh"),                        # expected plies to game end, or None
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
            legal = self._legal_mask(board).to(self.device)
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
        """Scaled QK^T content logits of one MHA layer from its input x:
        (1, H, 64, 64), using the layer's own in-projection."""
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

        Computed directly from the residual stream entering the layer, using
        the layer's own projections and GAB generator — no re-implementation
        of the model. Requires a GAB layer: with use_gab=False there are no
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
    def _check_recon(recon, target, what):
        """The decomposition sanity checks compare a re-derived tensor with the
        model's own. Relative to the target's norm, because absolute float32
        error scales with the activations — BT4's last layer runs at |x| ≈ 25,
        where a fixed 1e-4 trips on accumulation-order noise of 1e-5 relative."""
        err = (recon - target).norm() / (target.norm() + 1e-12)
        assert float(err) < 1e-3, f"{what} — relative error {float(err):.2e}; do not trust the decomposition"

    @staticmethod
    def _smolgen_coeffs(blk, x):
        """Replicate one layer's smolgen generator up to the mixing coefficients:
        (1, H, gen_size). Sanity-checked on the spot: mixing the shared
        templates with these coefficients must reproduce the layer's own
        _sq_bias() (asserted, so the check vanishes under python -O; see _check_recon)."""
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
        ChessformerEngine._check_recon(recon, blk._sq_bias(x), "smolgen coefficient reconstruction failed")
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
        """The smolgen mixing coefficients of one layer for this position:
        (H, gen_size). Row h holds the weights with which head h mixes the
        static `gab_templates()` into its 64x64 bias:
            gab_bias(layer, h) == (coeffs[h, :, None, None] * gab_templates()).sum(0)
        (verified internally, see `_smolgen_coeffs`). This is the model
        choosing its geometry live."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)
        blk = self.model.transformer.layers[L].self_attn
        if not blk.use_gab:
            raise RuntimeError(f"layer {L} has no GAB (use_gab=False)")
        return self._smolgen_coeffs(blk, x)[0].cpu()

    @torch.no_grad()
    def gab_bias(self, board, self_elo, oppo_elo=None, layer=0, head=None):
        """The generated geometric attention bias of one layer for this position:
        (H, 64, 64), or (64, 64) for a single `head`. bias[h][q][k] is added to
        the scaled QK^T logit of query q, key k before the softmax."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)
        blk = self.model.transformer.layers[L].self_attn
        if not blk.use_gab:
            raise RuntimeError(f"layer {L} has no GAB (use_gab=False)")
        gab = blk._sq_bias(x)[0].cpu()
        return gab if head is None else gab[int(head)]

    @torch.no_grad()
    def qk_scores(self, board, self_elo, oppo_elo=None, layer=0, head=None):
        """The raw content half of attention — scaled QK^T logits — of one layer
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
        """Exact per-head residual-stream writes of one layer's attention:
        (H, 64, dim).

        Head h's write is (A_h V_h) W_O^{(h)} — its attention-weighted values
        pushed through its own dh-column block of the output projection. This
        is the tensor a true head ablation must subtract: the hooked attn_NN
        activation is post-out_proj, where the heads are already mixed across
        every channel. Recomputed from the layer's own weights (same approach
        as `attention()`), and asserted on the spot to sum back — with the
        out_proj bias — to this forward's attn_NN activation."""
        L = int(layer)
        x = self._block_input(board, self_elo, oppo_elo, L)    # (1, 64, dim) into block L
        return self._head_writes_from_x(x, L)[0].cpu()

    @torch.no_grad()
    def _head_writes_from_x(self, x, layer):
        """`head_writes`' algebra over a whole batch: `x` is the (B, 64, dim)
        stream entering block `layer`, the result is (B, H, 64, dim) on the live
        device. Batch of one is the single-position case, so `head_writes` is
        this plus the forward pass that produces `x`."""
        L = int(layer)
        blk = self.model.transformer.layers[L].self_attn
        H = blk.num_heads
        B, _, d = x.shape
        dh = d // H

        gab = blk._sq_bias(x)                                  # (B, H, 64, 64)
        W = blk.mha.in_proj_weight                             # (3d, d), order [q; k; v]
        b = blk.mha.in_proj_bias
        q, k, v = (x @ W[i * d:(i + 1) * d].t() +
                   (b[i * d:(i + 1) * d] if b is not None else 0) for i in range(3))
        q, k, v = (t.view(B, 64, H, dh).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(dh) + gab, dim=-1)

        Wo = blk.mha.out_proj.weight                           # (dim, dim)
        per_head_out = Wo.view(d, H, dh).permute(1, 2, 0)      # (H, dh, dim)
        writes = (attn @ v) @ per_head_out                     # (B, H, 64, dim)

        recon = writes.sum(1)
        if blk.mha.out_proj.bias is not None:
            recon = recon + blk.mha.out_proj.bias
        target = self._activations[f"attn_{L:02d}"].to(self.device)
        self._check_recon(recon, target, f"head_writes reconstruction failed for layer {L}")
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
                                   fwd_hooks=[(f"attn_{layer:02d}", sub)],
                                   return_type=return_type)

    # ----- residual-stream evolution across depth ---------------------------
    @torch.no_grad()
    def residual_stream(self, board, self_elo, oppo_elo=None):
        """Two per-square views of how the residual stream is built up, in the
        side-to-move frame (square = rank*8 + file):

          delta = the per-square magnitude of the vector each structure writes
                  into the stream, in execution order: the input embedding
                  (`emb`), then for every layer the self-attention write
                  (`aN` = ||sa_out||) and the feed-forward write
                  (`mN` = ||ff_out||). Each entry is tagged with its `kind`
                  ('emb'/'attn'/'mlp') so the UI can mark what is writing at
                  each point. 1 + 2·num_blocks entries.

          moves = logit lens on the running residual stream at every readout
                  point (see `_lens_steps`): emb, then per layer the
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
            delta.append({"label": f"a{i}", "kind": "attn",
                          "norm": per_sq_norm(f"attn_{i:02d}")})
            delta.append({"label": f"m{i}", "kind": "mlp",
                          "norm": per_sq_norm(f"mlp_{i:02d}")})

        # ---- moves: per-sub-layer logit lens on the running residual stream ----
        # Same resolution as delta: emb, then (post-attn, post-mlp) per layer, enc.
        legal = self._legal_mask(board).to(self.device)
        moves = [{"label": lab, "kind": kind,
                  **self._lens_move(self._activations[name][0], board, legal)}
                 for name, lab, kind in self._lens_steps()]

        return {"delta": delta, "moves": moves}

    # ----- skill diff on internals ------------------------------------------
    def _lens_steps(self):
        """The readout points of the running residual stream, in order: emb, then
        per layer the post-attention and post-MLP points, then — when the model
        has a final norm the heads read through — enc. So 2·num_blocks + 2 of
        them on Maia-3 (18 on every size), 2·num_blocks + 1 on BT4 (31). The
        labels are the app-wide depth names: `aN`/`mN` for layer N's attention
        and MLP sub-layers (see interp_plot._depth_label)."""
        steps = [("embed_in", "emb", "emb")]
        for i in range(self.cfg.num_blocks):
            steps.append((f"postattn_{i:02d}", f"a{i}", "attn"))
            steps.append((f"block_{i:02d}",    f"m{i}", "mlp"))
        if "encoder_out" in self.hook_points:
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
        caveat applies to move_a/move_b everywhere but `enc`. On a model
        without a rating input (LeelaEngine) both runs are identical and every
        norm is zero."""
        _, cache_a = self.run_with_cache(board, int(elo_a))
        _, cache_b = self.run_with_cache(board, int(elo_b))
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
        return self._decode_idx(board, idx)

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
        legal = self._legal_mask(board).to(self.device)
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
        milliseconds. For many positions, `ablate_grid_batch` runs that same
        count per batch rather than per position."""
        return self.ablate_grid_batch([(board, uci)], self_elo, oppo_elo)[0]

    @torch.no_grad()
    def ablate_grid_batch(self, items, self_elo, oppo_elo=None, batch_size=32):
        """`ablate_grid` over many positions: `items` is [(board, move)] and the
        result is one ablate_grid dict per item, in order.

        The arithmetic is the single-position one, position by position — the
        positions are independent, so ablating a head is the same hook over a
        stacked forward. What changes is the bookkeeping: a batch costs
        ~num_blocks·(num_heads+1) forward passes in total instead of that many
        each, and a batched forward is several times cheaper per position than a
        batch of one. `batch_size` trades that against memory — the per-layer
        head writes are (batch_size, H, 64, dim)."""
        nb, nh = self.cfg.num_blocks, self.cfg.num_heads
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

                base = self._forward_batch(boards, self_elo, oppo_elo)[rows, idx]   # (B,)
                # accumulated on-device and synced once below, so consecutive head
                # passes pipeline instead of draining the launch queue each time
                deltas = torch.zeros(len(chunk), nb, nh, device=self.device)
                for L in range(nb):
                    # the clean pass this layer's writes are read off; also what the
                    # ablated passes below are compared against, so it is re-run per
                    # layer rather than cached (an ablated forward overwrites it)
                    self._forward_batch(boards, self_elo, oppo_elo)
                    key = "embed_in" if L == 0 else f"block_{L - 1:02d}"
                    writes = self._head_writes_from_x(self._activations[key].to(self.device), L)
                    for h in range(nh):
                        w = writes[:, h]
                        handle = self.hook_points[f"attn_{L:02d}"].register_forward_hook(
                            self._patch_hook(lambda a, w=w: a - w.to(a.device, a.dtype)))
                        try:
                            abl = self._forward_batch(boards, self_elo, oppo_elo)[rows, idx]
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
    def _mlp_hidden(self, board, self_elo, oppo_elo=None, layers=None):
        """One forward pass; returns {L: (64, ff)} — the MLP hidden activation
        (the input of linear2, post-activation) of each requested layer, on the
        live device, detached. `hook_points` holds linear2 as mlp_NN, so the
        hidden is its input."""
        blocks = self.model.transformer.layers
        want = range(len(blocks)) if layers is None else list(layers)
        hidden = {}

        def keep(L):
            def hook(_module, inp, _out):
                hidden[L] = inp[0].detach()[0]
            return hook

        handles = [blocks[L].linear2.register_forward_hook(keep(L)) for L in want]
        try:
            with torch.no_grad():
                self._forward(board, self_elo, oppo_elo)
        finally:
            for h in handles:
                h.remove()
        return hidden

    def _neuron_write(self, layer, neuron, hidden):
        """The (64, dim) vector one MLP unit adds to the stream: h[s, n] · W2[:, n]."""
        w2 = self.model.transformer.layers[int(layer)].linear2.weight[:, int(neuron)]
        return hidden[:, int(neuron), None] * w2[None, :]

    @torch.no_grad()
    def neuron_activation(self, board, self_elo, layer, neuron, oppo_elo=None):
        """One MLP unit on this position: {layer, neuron, n_neurons, act: [64]
        (its post-activation value on every square, canonical frame),
        write_norm: ‖h[:, n]‖·‖W2[:, n]‖, the size of what it adds to the
        stream}. Argument order as `ablate_head`: layer/neuron before oppo_elo."""
        L, n = int(layer), int(neuron)
        h = self._mlp_hidden(board, self_elo, oppo_elo, layers=[L])[L]
        w2 = self.model.transformer.layers[L].linear2.weight[:, n]
        return {"layer": L, "neuron": n, "n_neurons": int(h.size(1)),
                "act": h[:, n].float().cpu().tolist(),
                "write_norm": float(h[:, n].norm() * w2.norm())}

    @torch.no_grad()
    def neuron_overview(self, board, self_elo, oppo_elo=None, k=12):
        """Every layer's `k` most active MLP units on this position, by the
        norm of their activation over the 64 squares: {n_layers, n_neurons,
        layers: [[{neuron, norm}, …] per layer, strongest first]}. One forward
        pass; the app's network diagram is drawn from this."""
        hidden = self._mlp_hidden(board, self_elo, oppo_elo)
        layers = []
        for L in sorted(hidden):
            norms = hidden[L].float().norm(dim=0)                  # (ff,)
            top = norms.topk(min(int(k), norms.numel()))
            layers.append([{"neuron": int(n), "norm": float(v)}
                           for v, n in zip(top.values.tolist(), top.indices.tolist())])
        return {"n_layers": len(layers), "n_neurons": int(hidden[0].size(1)), "layers": layers}

    def ablate_neuron(self, board, self_elo, layer, neuron, oppo_elo=None,
                      return_type: str = "policy"):
        """Forward pass with one MLP unit's write removed exactly, on every
        square — the neuron-grain twin of `ablate_head`: the unit's write
        h[s,n]·W2[:,n] is subtracted from mlp_NN via run_with_hooks and the
        layers downstream react normally. Same argument order as `ablate_head`."""
        L = int(layer)
        w = self._neuron_write(L, neuron, self._mlp_hidden(board, self_elo, oppo_elo, layers=[L])[L])

        def sub(act):
            return act - w.to(act.device, act.dtype)

        return self.run_with_hooks(board, self_elo, oppo_elo,
                                   fwd_hooks=[(f"mlp_{L:02d}", sub)], return_type=return_type)

    def carrier_neurons(self, board, self_elo, move, oppo_elo=None, *,
                        top_k=12, skip_last_layer=True, verify=False):
        """The carrier table of one move at neuron grain: which MLP units, on
        which squares, the move's logit rests on. The neuron-grain counterpart
        of `ablate_grid`, using the same causal method as `head_writes`/
        `ablate_head` — a real intervention, not a correlational read of
        activations.

        Heads are cheap to ablate one by one; neurons are not (BT4 has 23k of
        them, each firing on 64 squares), so every unit is scored at once by
        attribution patching: one backward pass giving, for every hidden unit
        h of every MLP (layer L, square s, neuron n), the first-order estimate
        of the logit change from zeroing it, Δ ≈ −(∂logit/∂h)·h. Same sign
        convention as `ablate_grid`: negative = removing the unit would lower
        the move's logit (a carrier), positive = it was suppressing the move.
        Summed over squares that ranks the units; kept per square it says
        where on the board each does its work. The last layer is skipped by
        default, as the head grid does (it writes straight into the logits).

        The estimate is a linearization — Post-LN and the bilinear policy head
        can bend it — so with verify=True the 2·top_k strongest candidates are
        re-measured by exact zero-ablation (`ablate_neuron`) and the table is
        the top_k by |exact|; otherwise `exact` is None and the table is the
        top_k by |est|.

        `move` is any form `to_move` reads. Returns {uci, san, base_logit,
        n_layers, n_neurons, layer_abs: [nb] (Σ|Δ| over each layer's units),
        top: [{layer, neuron, est, exact, squares: [64] canonical, peak}]}.
        Costs one forward + one backward pass, plus 2·top_k with verify."""
        info = self.move_info(board, move)
        idx = info["idx"]
        layers = self.model.transformer.layers
        nb = len(layers)

        # capture every MLP's hidden activation (the input of its second linear)
        hidden: dict[int, torch.Tensor] = {}

        def keep(L):
            def hook(_module, inp, _out):
                hidden[L] = inp[0]
            return hook

        handles = [blk.linear2.register_forward_hook(keep(L)) for L, blk in enumerate(layers)]
        try:
            with torch.enable_grad():
                logits, _ = self._forward(board, self_elo, oppo_elo)
                grads = torch.autograd.grad(logits[idx], [hidden[L] for L in range(nb)])
        finally:
            for h in handles:
                h.remove()
        base = float(logits[idx].detach())
        # Δ ≈ −grad·h for zeroing one unit on one square: (nb, 64, ff)
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


# ----- the two engines --------------------------------------------------------

class MaiaEngine(ChessformerEngine):
    """ChessformerEngine on a Maia-3 checkpoint (downloaded from Hugging Face on
    first use). Conditioned on a rating pair; 4352-move policy."""

    has_conditioning = True
    has_generated_bias = True
    has_mlh = False

    def __init__(self, alias="maia3-5m", device=None, checkpoint_path=None,
                 activation_dir="activations", trust_checkpoint=False):
        """Build the model and install the permanent capture hooks.

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
        super().__init__(activation_dir)

    # ----- per-model members ------------------------------------------------
    def _embed_module(self):
        return self.model.token_projection

    def _final_norm(self):
        return self.model.transformer.norm

    def tokens(self, board: chess.Board) -> torch.Tensor:
        """Single current position, padded to fill `history` (matches the
        default `--use-uci-history` OFF behavior of the released engine)."""
        hist = deque([tokenize_board(board)], maxlen=self.cfg.history)
        toks = get_historical_tokens(
            hist, self.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0
        )
        return toks.unsqueeze(0).to(self.device)

    def _forward(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One raw forward pass. Resets and repopulates `self._activations` via the
        hooks, and returns (logits_move (4352,), logits_value (3,)) as floats.
        No @no_grad here: carrier_neurons backpropagates through it."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self._activations = {}
        self._aux = {}

        tokens = self.tokens(board)
        self_elos = torch.tensor([int(self_elo)], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([int(oppo_elo)], dtype=torch.long, device=self.device)

        logits_move, logits_value, _ = self.model(tokens, self_elos, oppo_elos)
        return logits_move[0].float(), logits_value[0].float()

    def _forward_batch(self, boards, self_elo, oppo_elo=None):
        """`_forward` over a list of boards in one pass: (B, 4352) move logits."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self._activations = {}
        self._aux = {}

        tokens = torch.cat([self.tokens(b) for b in boards], dim=0)
        elos = torch.full((len(boards),), 0, dtype=torch.long, device=self.device)
        self_elos, oppo_elos = elos + int(self_elo), elos + int(oppo_elo)

        logits_move, _, _ = self.model(tokens, self_elos, oppo_elos)
        return logits_move.float()

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

    @staticmethod
    def _wdl(logits_value) -> dict:
        """Maia-3's value logits are (loss, draw, win)."""
        loss, draw, win = torch.softmax(logits_value.float(), dim=-1).tolist()
        return {"win": win, "draw": draw, "loss": loss}


# alias -> (display name, file stem looked for in $CHESSFORMER_WEIGHTS or ./weights)
_LEELA_SPECS = {"bt4": ("Leela BT4", "Leela_BT4_large_model")}
_LEELA_SUFFIXES = (".pb.gz", ".pb", ".onnx")     # lc0's own weights file, or an ONNX export
_LEELA_ALIASES = {"bt4": "bt4", "leela-bt4": "bt4", "lc0-bt4": "bt4",
                  "leela": "bt4", "lc0": "bt4", "bt4-1024x15x32h": "bt4"}


class LeelaEngine(ChessformerEngine):
    """ChessformerEngine on Leela Chess Zero BT4, from an lc0 ONNX export.

    BT4 has no rating input: self_elo / oppo_elo are accepted and ignored, so
    the same calls work on both engines. No final norm, so the depth points
    end at m14 (no enc). evaluate()["mlh"] is the moves-left estimate in plies."""

    has_conditioning = False
    has_generated_bias = True
    has_mlh = True

    def __init__(self, alias="bt4", device=None, checkpoint_path=None,
                 activation_dir="activations"):
        """`alias` is a Leela alias or a path to lc0's .pb.gz (or an .onnx
        export). Without `checkpoint_path`, <stem>.pb.gz then <stem>.onnx are
        looked for in $CHESSFORMER_WEIGHTS, then ./weights. Nothing is downloaded."""
        key = str(alias).strip()
        if key.lower().endswith(_LEELA_SUFFIXES):
            name, checkpoint_path = "bt4", checkpoint_path or key
        else:
            name = _LEELA_ALIASES.get(key.lower())
            if name is None:
                raise ValueError(f"Unknown Leela alias {alias!r}.\n\n{format_engine_list()}")
        display_name, stem = _LEELA_SPECS[name]
        if checkpoint_path is None:
            weights_dir = Path(os.environ.get("CHESSFORMER_WEIGHTS", "weights")).expanduser()
            found = [weights_dir / (stem + sfx) for sfx in _LEELA_SUFFIXES
                     if (weights_dir / (stem + sfx)).exists()]
            checkpoint_path = found[0] if found else weights_dir / (stem + ".pb.gz")
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{display_name}: no weights at {checkpoint_path}. Download the network "
                f"(.pb.gz) from https://lczero.org and pass its path.")
        filename = checkpoint_path.name

        self.device = pick_device(device)
        self.model = _load_bt4(checkpoint_path, self.device)
        self.cfg = types.SimpleNamespace(**vars(self.model.cfg), checkpoint_path=str(checkpoint_path),
                                         device=self.device, history=8, trust_checkpoint=False)
        self.spec = types.SimpleNamespace(name=f"lc0-{name}", display_name=display_name,
                                          repo_id=None, checkpoint_filename=filename,
                                          aliases=tuple(a for a, n in _LEELA_ALIASES.items() if n == name),
                                          config=vars(self.model.cfg))
        self.all_moves = _leela_all_moves()
        super().__init__(activation_dir)

    # ----- per-model members ------------------------------------------------
    def _embed_module(self):
        return self.model.embedding

    def _final_norm(self):
        return None

    def tokens(self, board: chess.Board) -> torch.Tensor:
        """lc0's input planes, (1, 112, 8, 8); history from board.move_stack."""
        return _leela_planes(board).unsqueeze(0).to(self.device)

    def _forward(self, board: chess.Board, self_elo: int = 0, oppo_elo: int | None = None):
        """One raw forward pass (elo ignored): (logits_move (4288,), logits_value
        (3,)). The moves-left estimate goes to self._aux["mlh"]."""
        self._activations = {}
        logits_move, logits_value, mlh = self.model(self.tokens(board))
        self._aux = {"mlh": float(mlh[0].detach())}
        return logits_move[0].float(), logits_value[0].float()

    def _forward_batch(self, boards, self_elo=0, oppo_elo=None):
        """`_forward` over a list of boards in one pass: (B, 4288) move logits."""
        self._activations = {}
        planes = torch.stack([_leela_planes(b) for b in boards]).to(self.device)
        logits_move, _, mlh = self.model(planes)
        self._aux = {"mlh": mlh.detach().tolist()}
        return logits_move.float()

    def _move_logits(self, x):
        """(64, dim) residual -> (4288,) move logits, through BT4's policy head."""
        return self.model.policy_logits(x.unsqueeze(0))[0]

    @staticmethod
    def _move_squares(idx):
        """Canonical (from, to) squares for a policy index; promotions are
        4096 + from_file*24 + to_file*3 + piece."""
        if idx < 64 * 64:
            return idx // 64, idx % 64
        idx -= 64 * 64
        return 48 + idx // 24, 56 + (idx % 24) // 3

    @staticmethod
    def _wdl(logits_value) -> dict:
        """lc0's value logits are (win, draw, loss)."""
        win, draw, loss = torch.softmax(logits_value.float(), dim=-1).tolist()
        return {"win": win, "draw": draw, "loss": loss}

    # ----- policy index <-> move ---------------------------------------------
    # lc0 has no knight-promotion index (the bare square pair is used) and
    # stores castling as king takes rook (e1h1, not e1g1). So for castling,
    # move_info()["to"] is the rook's square; its uci stays e1g1.
    def _decode_idx(self, board: chess.Board, idx: int) -> chess.Move:
        """As the base, plus: pawn to last rank -> knight promotion; king onto
        its own rook -> castling."""
        mv = super()._decode_idx(board, idx)
        piece = board.piece_type_at(mv.from_square)
        if (piece == chess.PAWN and mv.promotion is None
                and chess.square_rank(mv.to_square) in (0, 7)):
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.KNIGHT)
        elif (piece == chess.KING
              and board.piece_at(mv.to_square) == chess.Piece(chess.ROOK, board.turn)):
            kingside = chess.square_file(mv.to_square) > chess.square_file(mv.from_square)
            mv = chess.Move(mv.from_square,
                            chess.square(6 if kingside else 2, chess.square_rank(mv.from_square)))
        return mv

    def _move_index(self, board: chess.Board, move) -> int:
        """As the base, plus: knight promotion -> its square pair; castling ->
        king takes rook."""
        mv = self.to_move(board, move)
        if mv.promotion == chess.KNIGHT:
            mv = chess.Move(mv.from_square, mv.to_square)
        elif board.is_castling(mv):
            kingside = chess.square_file(mv.to_square) > chess.square_file(mv.from_square)
            mv = chess.Move(mv.from_square,
                            chess.square(7 if kingside else 0, chess.square_rank(mv.from_square)))
        uci = mv.uci()
        return self.all_moves_dict[mirror_move(uci) if board.turn == chess.BLACK else uci]


# ----- aliases ----------------------------------------------------------------

def resolve_engine(alias: str):
    """alias -> (engine class, display name), without loading weights. Takes
    Maia-3 registry names, Leela aliases, or a .pb.gz / .onnx path; ValueError otherwise."""
    key = str(alias).strip().lower()
    if key in _LEELA_ALIASES or key.endswith(_LEELA_SUFFIXES):
        return LeelaEngine, _LEELA_SPECS[_LEELA_ALIASES.get(key, "bt4")][0]
    try:
        return MaiaEngine, resolve_model_spec(alias).display_name
    except ModelResolutionError as exc:
        raise ValueError(f"Unknown model {alias!r}.\n\n{format_engine_list()}") from exc


def load_engine(alias: str, **kwargs):
    """Build the engine an alias names, e.g. load_engine("23m"), load_engine("bt4").
    Keyword arguments go to its constructor."""
    cls, _ = resolve_engine(alias)
    return cls(alias=alias, **kwargs)


def format_engine_list() -> str:
    lines = ["Chessformer engines, by alias:", "  Maia-3 (MaiaEngine):"]
    for spec in MODEL_SPECS:
        lines.append(f"    {spec.display_name:<18}{', '.join((spec.name, *spec.aliases))}")
    lines.append("  Leela Chess Zero (LeelaEngine):")
    for name, (display_name, stem) in _LEELA_SPECS.items():
        aliases = ", ".join(a for a, n in _LEELA_ALIASES.items() if n == name)
        lines.append(f"    {display_name:<18}{aliases}, or a path to lc0's .pb.gz "
                     f"(default weights/{stem}.pb.gz)")
    return "\n".join(lines)
