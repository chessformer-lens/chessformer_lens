"""Leela Chess Zero (lc0) attention-body networks — BT4, T82 and nets built like them.

Reads lc0's own weights file (.pb.gz, decoded here with no protobuf
dependency) or an `lc0 leela2onnx` export (.onnx, needs the `onnx` package),
rebuilds the net in PyTorch (`BT4Model`), and serves it to the engine through
`Lc0Adapter`:

  input     lc0's 112 classical planes, history from board.move_stack
  tokens    the 64 squares, side-to-move frame (Black mirrored)
  policy    lc0's attention policy, 64x64 square pairs + 192 promotions
            (4288; knight promotions share their square pair's logit)
  value     win / draw / loss, plus a moves-left head ("mlh")
  rating    none: Conditioning is ignored

T82 (768x15x24h, the generation before BT4) differs from BT4 only in how the
planes become the stream entering block 0 (`_T82Embedding` vs `_BT4Embedding`)
and in where its .pb.gz keeps the head weights; block 0 onward is read by the
same code and has the same hook names, so every engine method works on both.
"""
from __future__ import annotations

import gzip
import os
import struct
import types
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import (Capabilities, ArchSpec, HeadOutputs, ModelAdapter, PolicyVocab,
                    ValueEstimate, _register_builtin, pick_device)
from .smolgen import GabAccess, PostLNBlock, SmolgenAttention, decompose

# ----- the model ---------------------------------------------------------------
# BT4 (1024 dim, 15 layers, 32 heads) is Post-LN like Maia-3, and its smolgen bias
# is the same generator as Maia-3's GAB, so both read through the same attribute
# names (see smolgen.py). BT4 also scales every sublayer by alpha (0.427) before
# the residual add; that is folded into the output projections at load time, so
# each sublayer's output is exactly its write to the stream. Weights are read from
# lc0's own .pb.gz, or the ONNX file `lc0 leela2onnx` produces.

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


def _pos_encoding() -> torch.Tensor:
    """lc0's kPosEncoding, the fixed (64, 64) table T82-era nets concatenate
    to each square's input planes: row s is -1 on s itself, 1 on every square
    a queen or a knight standing on s could reach, 0 elsewhere. Reproduced by
    that rule, which matches `lc0 leela2onnx`'s /const/pos_encoding exactly."""
    table = torch.zeros(64, 64)
    for a in range(64):
        for b in range(64):
            df = abs(a % 8 - b % 8)
            dr = abs(a // 8 - b // 8)
            if a == b:
                table[a, b] = -1.0
            elif df == 0 or dr == 0 or df == dr or {df, dr} == {1, 2}:
                table[a, b] = 1.0
    return table


class _T82Embedding(nn.Module):
    """planes (B, 112, 8, 8) -> (B, 64, d) the T82 way (lc0's PE_MAP embedding):
    each square's 112 inputs plus its row of the fixed position table, one
    projection, mish, and the per-square gates. No norm, no FFN sublayer."""

    def __init__(self, d: int, planes: int = 112):
        super().__init__()
        self.register_buffer("pos_encoding", _pos_encoding(), persistent=False)
        self.proj = nn.Linear(planes + 64, d)
        self.mul_gate = nn.Parameter(torch.ones(64, d))
        self.add_gate = nn.Parameter(torch.zeros(64, d))

    def forward(self, planes):
        B = planes.size(0)
        x = planes.reshape(B, planes.size(1), 64).transpose(1, 2)         # (B, 64, 112), sq = rank*8 + file
        pos = self.pos_encoding.expand(B, -1, -1)                        # (B, 64, 64)
        x = F.mish(self.proj(torch.cat([x, pos], dim=-1)))
        return x * self.mul_gate + self.add_gate


class _BT4Encoder(nn.Module):
    """The block stack, with no final norm (norm = None): BT4's heads read the
    last block directly."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = None

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


def _bt4_block(d, heads, ff, gen, per_square, intermediate, gab_weight):
    """One BT4 encoder layer: Post-LN, mish FFN, swish smolgen with no bias on
    its compress layer, LayerNorm eps 1e-3."""
    attn = SmolgenAttention(d, heads, gen, per_square, intermediate, gab_weight,
                            act=nn.SiLU(), ln_eps=_BT4_LN_EPS, compress_bias=False)
    return PostLNBlock(attn, d, ff, F.mish, ln_eps=_BT4_LN_EPS)


class BT4Model(nn.Module):
    """Lc0 BT4 in PyTorch, laid out like Maia-3 where the adapters read it
    (transformer.layers[i], gab_shared_weight (64*64, gen)).

    forward(planes (B, 112, 8, 8)) -> logits_move (B, 4288), logits_value (B, 3),
    moves_left (B,).
      logits_move    64*64 square pairs (from*64 + to), then 192 promotions
                     (4096 + from_file*24 + to_file*3 + piece, piece in q r b).
                     A knight promotion uses its square pair's logit.
      logits_value   win, draw, loss for the side to move
      moves_left     expected plies to the end of the game
    `policy_map` (1858,) is lc0's gather to its own move list, kept for testing.
    `embedding` picks the input embedding: "dense" (BT4, `pre` channels of
    board-wide preprocessing) or "pe_map" (T82, the fixed position table)."""

    def __init__(self, num_blocks=15, dim=1024, num_heads=32, ff=1536, gen=256,
                 per_square=32, intermediate=256, pre=512, policy_dim=1024,
                 value_dim=128, mlh_dim=32, policy_scale=1 / 32, embedding="dense"):
        super().__init__()
        self.cfg = types.SimpleNamespace(
            num_blocks=num_blocks, dim_vit=dim, num_heads=num_heads,
            mlp_ratio=ff / dim, gab_gen_size=gen, gab_per_square_dim=per_square,
            gab_intermediate_dim=intermediate, head_hid_dim=policy_dim, use_gab=True,
            embedding=embedding)
        if embedding == "dense":
            self.embedding = _BT4Embedding(dim, ff, pre)
        elif embedding == "pe_map":
            self.embedding = _T82Embedding(dim)
        else:
            raise ValueError(f"embedding must be 'dense' or 'pe_map', not {embedding!r}")
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
        return self.policy_from_hidden(F.mish(self.policy_dense(x)))

    def policy_from_hidden(self, h):
        """The head after its activation: (B, 64, policy_dim) -> (B, 4288)."""
        B = h.size(0)
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
    if 35 not in w:
        problems.append("needs smolgen")
    if 23 in w:
        problems.append("policy-head encoder layers are not supported")
    if problems:
        raise NotImplementedError(f"{path}: this lc0 net is not supported: " + "; ".join(problems)
                                  + ". Tested on BT4-1024x15x32h and T82-768x15x24h.")

    def dense(msg, field, n_in, n_out):            # stored (out, in) -> ONNX (in, out)
        return _pb_layer(msg, field).reshape(n_out, n_in).t().contiguous()

    W = {}
    emb = _pb_layer(w, 26).numel()
    dense_embedding = 37 in w                      # BT4's preprocess layer; T82 has the position table instead
    if dense_embedding:
        pre = _pb_layer(w, 38).numel() // 64
        W["/attn_body/embedding/preprocess/matmul/w"] = dense(w, 37, 64 * 12, 64 * pre)
        W["/attn_body/embedding/preprocess/add/w"] = _pb_layer(w, 38)
        W["/attn_body/matmul/w"] = dense(w, 25, 112 + pre, emb)
        W["/attn_body/ln/w/scale"], W["/attn_body/ln/w/bias"] = _pb_layer(w, 39), _pb_layer(w, 40)
    else:
        W["/const/pos_encoding"] = _pos_encoding().reshape(1, 64, 64)
        W["/attn_body/matmul/w"] = dense(w, 25, 112 + 64, emb)
    W["/attn_body/add/w"] = _pb_layer(w, 26)
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

    if dense_embedding:
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

    # policy: the "vanilla" head. BT4 keeps it in the policy_heads message (45), with
    # an embedding that may be shared across heads; T82 keeps it in the older
    # top-level fields. Same layers either way.
    if 45 in w:
        heads_msg = _pb_msg(w, 45)
        pol = _pb_msg(heads_msg, 3)
        if 8 in pol:
            raise NotImplementedError(f"{path}: policy-head encoder layers are not supported")
        pw = pol if 1 in pol else heads_msg
        dense1, q, k, promo = (pw, 1, 2), (pol, 3, 4), (pol, 5, 6), (pol, 7)
    else:
        dense1, q, k, promo = (w, 4, 5), (w, 17, 18), (w, 19, 20), (w, 22)
    pemb, pd = _pb_layer(dense1[0], dense1[2]).numel(), _pb_layer(q[0], q[2]).numel()
    W["/policy/dense1/matmul/w"] = dense(dense1[0], dense1[1], emb, pemb)
    W["/policy/dense1/add/w"] = _pb_layer(dense1[0], dense1[2])
    W["/policy/Q/matmul/w"], W["/policy/Q/add/w"] = dense(q[0], q[1], pemb, pd), _pb_layer(q[0], q[2])
    W["/policy/K/matmul/w"], W["/policy/K/add/w"] = dense(k[0], k[1], pemb, pd), _pb_layer(k[0], k[2])
    W["/policy/scale/w"] = torch.tensor([np.float32(1) / np.sqrt(np.float32(pd))])
    W["/policy/promotion/matmul/w"] = dense(promo[0], promo[1], pd, 4)

    # value: the "winner" head, in the value_heads message (44) on BT4 and in the
    # older top-level fields on T82; moves-left
    if 44 in w:
        val = _pb_msg(_pb_msg(w, 44), 1)
        embed_w, embed_b, d1_w, d1_b, d2_w, d2_b = 1, 2, 3, 4, 5, 6
    else:
        val = w
        embed_w, embed_b, d1_w, d1_b, d2_w, d2_b = 29, 30, 7, 8, 9, 10
    vc = _pb_layer(val, embed_b).numel()
    W["/value/embed/matmul/w"], W["/value/embed/add/w"] = dense(val, embed_w, emb, vc), _pb_layer(val, embed_b)
    W["/value/dense1/matmul/w"], W["/value/dense1/add/w"] = dense(val, d1_w, 64 * vc, 128), _pb_layer(val, d1_b)
    W["/value/dense2/matmul/w"], W["/value/dense2/add/w"] = dense(val, d2_w, 128, 3), _pb_layer(val, d2_b)
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
    dense_embedding = "/attn_body/embedding/preprocess/matmul/w" in W     # BT4; T82 has the position table
    pre = W["/attn_body/embedding/preprocess/matmul/w"].shape[1] // 64 if dense_embedding else 0
    model = BT4Model(nb, d, heads, ff, gen, per_sq, inter, pre,
                     policy_dim=W["/policy/Q/matmul/w"].shape[1],
                     value_dim=W["/value/embed/matmul/w"].shape[1],
                     mlh_dim=W["/mlh/embed/matmul/w"].shape[1],
                     policy_scale=float(W["/policy/scale/w"]),
                     embedding="dense" if dense_embedding else "pe_map")
    for i in range(nb):     # nn.MultiheadAttention assumes a 1/sqrt(d_head) scale
        s = float(W[f"/encoder{i}/mha/QK/scale/w"])
        assert abs(s - dh ** -0.5) < 1e-6, f"encoder{i}: QK scale {s} is not 1/sqrt({dh})"

    sd = {}

    def put(key, weight, bias=None, alpha=1.0):
        sd[key + ".weight"] = weight * alpha
        if bias is not None:
            sd[key + ".bias"] = bias * alpha

    put("embedding.proj", lin("/attn_body/matmul/w"), W["/attn_body/add/w"])
    sd["embedding.mul_gate"] = W["/ip_mul_gate/w"]
    sd["embedding.add_gate"] = W["/ip_add_gate/w"]
    if dense_embedding:
        put("embedding.preproc", lin("/attn_body/embedding/preprocess/matmul/w"),
            W["/attn_body/embedding/preprocess/add/w"])
        put("embedding.norm", W["/attn_body/ln/w/scale"], W["/attn_body/ln/w/bias"])
        put("embedding.linear1", lin("/attn_body/ffn/dense1/w/w"), W["/attn_body/ffn/dense1/b/w"])
        put("embedding.linear2", lin("/attn_body/ffn/dense2/w/w"), W["/attn_body/ffn/dense2/b/w"],
            alpha=float(W["/attn_body/ffn/alpha/w"]))
        put("embedding.norm2", W["/attn_body/ln2/w/scale"], W["/attn_body/ln2/w/bias"])
    else:       # the table is built into the model; an ONNX export carries lc0's own copy to check against
        table = W["/const/pos_encoding"].reshape(64, 64)
        assert torch.equal(table, model.embedding.pos_encoding), "position table differs from lc0's"
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
        for color, off in ((chess.WHITE, 0), (chess.BLACK, 6)):
            for j, piece in enumerate(chess.PIECE_TYPES):   # P N B R Q K
                planes[base + off + j] = _bits(view.pieces_mask(piece, color))
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


# ----- moves --------------------------------------------------------------------

class Lc0Vocab(PolicyVocab):
    """BT4's move list, plus lc0's two encoding quirks: there is no
    knight-promotion index (the bare square pair is used) and castling is
    stored as king takes rook (e1h1, not e1g1). So for castling the index's
    `to` square is the rook's; the decoded move is the real e1g1."""

    def index(self, board: chess.Board, move: chess.Move) -> int:
        mv = move
        if mv.promotion == chess.KNIGHT:
            mv = chess.Move(mv.from_square, mv.to_square)
        elif board.is_castling(mv):
            kingside = chess.square_file(mv.to_square) > chess.square_file(mv.from_square)
            mv = chess.Move(mv.from_square,
                            chess.square(7 if kingside else 0, chess.square_rank(mv.from_square)))
        return super().index(board, mv)

    def decode(self, board: chess.Board, idx: int) -> chess.Move:
        mv = super().decode(board, idx)
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


# ----- the adapter ----------------------------------------------------------------

# The official lc0 nets (https://lczero.org/play/networks/bestnets/): BT4,
# policytune-332 (BT4 fine-tuned for a better policy), and T82, the smaller net of
# the generation before BT4. A file is recognised by its exact size, so a renamed
# copy still says which net it is.
_LC0_NETS = {
    "bt4": dict(
        display_name="Leela BT4", file="BT4-1024x15x32h-swa-6147500.pb.gz",
        url="https://storage.lczero.org/files/networks-contrib/big-transformers/"
            "BT4-1024x15x32h-swa-6147500.pb.gz",
        size=335897546, sha256="66e6c1b5308c34230801dc171df580d7a66ec916dc2211246a485ef8e545c67e"),
    "bt4-332": dict(
        display_name="Leela BT4 policytune-332",
        file="BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz",
        url="https://storage.lczero.org/files/networks-contrib/"
            "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz",
        size=382645315,
        sha256="e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618"),
    "t82": dict(
        display_name="Leela T82", file="768x15x24h-t82-swa-7464000.pb.gz",
        url="https://storage.lczero.org/files/768x15x24h-t82-swa-7464000.pb.gz",
        size=170339855, sha256="98c8e077647092b7645c683bc60ca0c888b4904d9981a63a70ac47b0c4501b88"),
}
_LEELA_SUFFIXES = (".pb.gz", ".pb", ".onnx")     # lc0's own weights file, or an ONNX export
_LEELA_ALIASES = {"bt4": "bt4", "leela-bt4": "bt4", "lc0-bt4": "bt4", "leela": "bt4",
                  "lc0": "bt4", "bt4-1024x15x32h": "bt4", "bt4-base": "bt4",
                  "bt4-332": "bt4-332", "bt4-it332": "bt4-332", "bt4-policytune": "bt4-332",
                  "leela-bt4-332": "bt4-332",
                  "t82": "t82", "leela-t82": "t82", "lc0-t82": "t82", "t82-768x15x24h": "t82"}
_LEELA_LEGACY_FILE = "Leela_BT4_large_model.pb.gz"   # the old fixed name; used only if its size matches


def lc0_cache_dir() -> Path:
    """Where downloaded lc0 nets go: $CHESSFORMER_CACHE, else ~/.cache/chessformer_lens."""
    return Path(os.environ.get("CHESSFORMER_CACHE")
                or Path.home() / ".cache" / "chessformer_lens").expanduser()


def _weights_dirs() -> list:
    dirs = [Path(os.environ["CHESSFORMER_WEIGHTS"]).expanduser()] \
        if os.environ.get("CHESSFORMER_WEIGHTS") else []
    return dirs + [Path("weights"), lc0_cache_dir()]


def identify_lc0_net(path) -> str | None:
    """Which official net (`_LC0_NETS` key) a weights file is, by its exact size."""
    try:
        size = Path(path).stat().st_size
    except OSError:
        return None
    return next((k for k, net in _LC0_NETS.items() if net["size"] == size), None)


def _find_lc0_net(name):
    """A local copy of net `name`: its official file name in $CHESSFORMER_WEIGHTS,
    ./weights or the cache, or the old fixed name if its size says it is this net."""
    for d in _weights_dirs():
        if (d / _LC0_NETS[name]["file"]).exists():
            return d / _LC0_NETS[name]["file"]
        if identify_lc0_net(d / _LEELA_LEGACY_FILE) == name:
            return d / _LEELA_LEGACY_FILE
    return None


def download_lc0_net(name, dest_dir=None):
    """Download official net `name` from lczero.org into `dest_dir` (default:
    `lc0_cache_dir()`), checking its sha256; returns the file's path. Written to
    a .part file first, so an interrupted download never looks complete."""
    import hashlib
    import ssl
    import urllib.request

    net = _LC0_NETS[name]
    dest_dir = Path(dest_dir or lc0_cache_dir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / net["file"]
    part = dest.with_name(dest.name + ".part")
    print(f"chessformer_lens: {net['display_name']} weights are not on this machine; "
          f"downloading them now ({net['size'] / 1e6:.0f} MB, one time).\n"
          f"  from:  {net['url']}\n  to:    {dest}", flush=True)
    try:        # python.org's macOS builds ship without system CAs; certifi has them
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(net["url"], headers={"User-Agent": "chessformer_lens"})
    digest, done, shown = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if done * 10 // net["size"] > shown:
                shown = done * 10 // net["size"]
                print(f"  {min(100, shown * 10)}%", end="\n" if shown >= 10 else "", flush=True)
    if digest.hexdigest() != net["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{net['file']}: checksum mismatch after download; try again")
    part.replace(dest)
    return dest


@_register_builtin
class Lc0Adapter(ModelAdapter):
    """An lc0 attention-body net (tested on BT4-1024x15x32h and T82-768x15x24h)."""

    name = "lc0"
    family = "Leela Chess Zero"
    caps = Capabilities(conditioning=frozenset(), value="wdl", norm="post",
                        attn_terms=("content", "gab"), aux_heads=frozenset({"mlh"}))

    @classmethod
    def resolve(cls, alias):
        key = str(alias).strip().lower()
        if key in _LEELA_ALIASES:
            return _LC0_NETS[_LEELA_ALIASES[key]]["display_name"]
        if key.endswith(_LEELA_SUFFIXES):
            net = identify_lc0_net(Path(str(alias).strip()).expanduser())
            return _LC0_NETS[net]["display_name"] if net else "Leela (lc0 net)"
        return None

    @classmethod
    def models(cls):
        out = []
        for name, net in _LC0_NETS.items():
            found = _find_lc0_net(name)
            out.append({
                "alias": name, "display_name": net["display_name"],
                "aliases": [a for a, n in _LEELA_ALIASES.items() if n == name and a != name],
                "available": True,
                "note": str(found) if found else
                f"downloads {net['size'] / 1e6:.0f} MB from lczero.org on first use",
            })
        return out

    @classmethod
    def catalog(cls):
        return super().catalog() + ["    or a path to any lc0 attention-body net (.pb.gz / .onnx)"]

    @classmethod
    def load(cls, alias="bt4", *, device=None, checkpoint_path=None, **_):
        """`alias` names an official net, `bt4` (the base BT4), `bt4-332`
        (policytune-332) or `t82`, or is a path to an lc0 .pb.gz (or an .onnx export).
        Without `checkpoint_path` the net is looked for in $CHESSFORMER_WEIGHTS,
        ./weights and the download cache (`lc0_cache_dir()`), and downloaded from
        lczero.org on first use (set CHESSFORMER_NO_DOWNLOAD=1 to refuse instead).
        Which net was loaded is printed, recognised by the file's size."""
        key = str(alias).strip()
        if key.lower().endswith(_LEELA_SUFFIXES):
            name, checkpoint_path = None, checkpoint_path or key
        else:
            name = _LEELA_ALIASES.get(key.lower())
            if name is None:
                from ..core import format_catalog
                raise ValueError(f"Unknown Leela alias {alias!r}.\n\n{format_catalog()}")
        if checkpoint_path is None:
            checkpoint_path = _find_lc0_net(name)
            if checkpoint_path is None:
                if os.environ.get("CHESSFORMER_NO_DOWNLOAD"):
                    raise FileNotFoundError(
                        f"{_LC0_NETS[name]['display_name']}: {_LC0_NETS[name]['file']} is not in "
                        f"{', '.join(map(str, _weights_dirs()))} and CHESSFORMER_NO_DOWNLOAD is set. "
                        f"Download it from {_LC0_NETS[name]['url']}")
                checkpoint_path = download_lc0_net(name)
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"no lc0 weights at {checkpoint_path}. Download a network (.pb.gz) from "
                f"https://lczero.org and pass its path.")
        net = identify_lc0_net(checkpoint_path)
        if net is not None:
            display_name = _LC0_NETS[net]["display_name"]
            note = (f"  (you asked for {_LC0_NETS[name]['display_name']})"
                    if name is not None and name != net else "")
        else:
            display_name = f"Leela ({checkpoint_path.name})"
            note = "  (not one of the official nets this package knows)"
        print(f"chessformer_lens: loading {display_name} from {checkpoint_path}{note}", flush=True)
        name = net or name or "bt4"

        self = cls()
        self.device = pick_device(device)
        self.model = _load_bt4(checkpoint_path, self.device)
        mc = self.model.cfg
        self.cfg = types.SimpleNamespace(**vars(mc), checkpoint_path=str(checkpoint_path),
                                         device=self.device, history=8, trust_checkpoint=False)
        aliases = tuple(a for a, n in _LEELA_ALIASES.items() if n == name)
        self.spec = types.SimpleNamespace(name=f"lc0-{name}", display_name=display_name,
                                          repo_id=None, checkpoint_filename=checkpoint_path.name,
                                          aliases=aliases, config=vars(mc))
        self.arch = ArchSpec(n_layers=mc.num_blocks, n_heads=mc.num_heads, d_model=mc.dim_vit,
                             d_mlp=int(round(mc.dim_vit * mc.mlp_ratio)), name=f"lc0-{name}",
                             display_name=display_name, checkpoint=str(checkpoint_path))
        self.vocab = Lc0Vocab(_leela_all_moves())
        layers = self.model.transformer.layers
        self._gab = GabAccess([blk.self_attn for blk in layers], self.model.gab_shared_weight)
        return self

    # --- inputs and outputs ---
    def tokens(self, boards):
        """lc0's input planes, (B, 112, 8, 8); history from board.move_stack."""
        return torch.stack([_leela_planes(b) for b in boards]).to(self.device)

    def forward(self, boards, cond):
        logits_move, logits_value, mlh = self.model(self.tokens(boards))
        return HeadOutputs(policy=logits_move, value=logits_value, aux={"mlh": mlh})

    def value(self, raw):
        """lc0's value logits are (win, draw, loss)."""
        win, draw, loss = torch.softmax(raw.float(), dim=-1).tolist()
        return ValueEstimate.from_wdl(win, draw, loss)

    # --- internals ---
    def hook_modules(self):
        hooks = {"embed_in": self.model.embedding}
        for i, blk in enumerate(self.model.transformer.layers):
            hooks[f"block_{i:02d}"] = blk
            hooks[f"attn_{i:02d}"] = blk.self_attn
            hooks[f"mlp_{i:02d}"] = blk.linear2
            hooks[f"postattn_{i:02d}"] = blk.norm1
        return hooks

    def unembed(self, x):
        """(64, dim) residual -> (4288,) move logits, through BT4's policy head."""
        return self.model.policy_logits(x.unsqueeze(0))[0]

    def unembed_frozen(self, x, x0):
        """The head with its mish replaced by its tangent at the activation `x0`
        produced, so the logits are bilinear in `x`."""
        m = self.model
        z0 = m.policy_dense(x0).detach()
        with torch.enable_grad():
            z = z0.clone().requires_grad_(True)
            slope = torch.autograd.grad(F.mish(z).sum(), z)[0]
        h = F.mish(z0) + slope * (m.policy_dense(x) - z0)
        return m.policy_from_hidden(h.unsqueeze(0))[0]

    def stream_norms(self):
        out = []
        for i, blk in enumerate(self.model.transformer.layers):
            out += [(f"attn_{i:02d}", blk.norm1), (f"mlp_{i:02d}", blk.norm2)]
        return out

    def decompose_attention(self, layer, x, cond):
        return decompose(self.model.transformer.layers[int(layer)].self_attn, x)

    def mlp_out(self, layer):
        return self.model.transformer.layers[int(layer)].linear2

    def extensions(self):
        return {"gab": self._gab}
