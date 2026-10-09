"""Smolgen attention — multi-head self-attention over the 64 square tokens plus
a square-pair bias generated from the position — shared by Lc0 BT4 and
Maia-3 (where it is called GAB, the geometric attention bias).

The bias is not stored, it is generated: a small MLP ("smolgen") reads the
whole board and emits, per head, `gen_size` mixing coefficients over a bank
of static 64x64 templates shared by every layer and head (`gab_weight`,
stored (64*64, gen)). So for every head

    bias[h] = sum_i coeffs[h, i] * template_i

`SmolgenAttention` / `PostLNBlock` implement BT4's layers from lc0's
description. `decompose` and `GabAccess` read any module with the same
attribute surface — these classes, or Maia-3's attention module — and are
what the engine's attention and GAB methods go through.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core import AttnDecomp, content_scores, out_proj_heads, split_heads


class SmolgenAttention(nn.Module):
    """Self-attention with a smolgen bias, packed QKV.

      mha        nn.MultiheadAttention (batch_first); the bias goes in as its
                 additive float attn_mask, (B*H, 64, 64)
      sm1        per-square compression (d -> per_square), or None to
                 mean-pool the squares instead
      sm2, ln1   board summary -> hidden
      sm3, ln2   hidden -> H * gen coefficients
      gab_weight the shared (64*64, gen) template bank
    Returns (output, None), the nn.MultiheadAttention convention."""

    def __init__(self, d_model: int, n_heads: int, gen_size: int, per_square: int,
                 hidden: int, gab_weight: nn.Parameter, *, act: nn.Module | None = None,
                 ln_eps: float = 1e-5, compress_bias: bool = True, qkv_bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True,
                                         bias=qkv_bias)
        self.num_heads = n_heads
        self.gen_size = gen_size
        self.use_gab = True
        self.gab_weight = gab_weight
        if per_square:
            self.sm1 = nn.Linear(d_model, per_square, bias=compress_bias)
            self.sm2 = nn.Linear(64 * per_square, hidden)
        else:
            self.sm1 = None
            self.sm2 = nn.Linear(d_model, hidden)
        self.ln1 = nn.LayerNorm(hidden, eps=ln_eps)
        self.sm3 = nn.Linear(hidden, n_heads * gen_size)
        self.ln2 = nn.LayerNorm(n_heads * gen_size, eps=ln_eps)
        self.sm_act = act if act is not None else nn.GELU()

    def _sq_bias(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 64, d) -> the generated bias (B, H, 64, 64)."""
        B = x.size(0)
        y = coefficients(self, x)
        return torch.einsum("bhi,oi->bho", y, self.gab_weight).view(B, self.num_heads, 64, 64)

    def forward(self, x: torch.Tensor):
        bias = self._sq_bias(x).reshape(-1, 64, 64)
        return self.mha(x, x, x, need_weights=False, attn_mask=bias)


class PostLNBlock(nn.Module):
    """x = norm1(x + attn(x)); x = norm2(x + linear2(act(linear1(x))))."""

    def __init__(self, attn: nn.Module, d_model: int, d_ff: int, act, *, ln_eps: float = 1e-5):
        super().__init__()
        self.self_attn = attn
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.activation = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(x)
        x = self.norm1(x + sa)
        return self.norm2(x + self.linear2(self.activation(self.linear1(x))))


# ----- reading smolgen modules ---------------------------------------------------------

def coefficients(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The smolgen generator of one attention module up to its mixing
    coefficients: (B, 64, d) -> (B, H, gen)."""
    B = x.size(0)
    if attn.sm1 is not None:                                   # per-square path
        y = attn.sm1(x).reshape(B, -1)                         # (B, 64*p)
    else:                                                      # mean-pooled path
        y = torch.mean(x, dim=1)                               # (B, d)
    y = attn.sm_act(attn.sm2(y))
    y = attn.ln1(y)
    y = attn.sm_act(attn.sm3(y))
    return attn.ln2(y).view(B, attn.num_heads, attn.gen_size)


def decompose(attn: nn.Module, x: torch.Tensor) -> AttnDecomp:
    """One packed-QKV attention module, from its input x (B, T, d), as
    q, k, v, the score terms ("content" = scaled QKᵀ, then "gab" — or
    "relbias" for a learned static bias — when the module has one) and the
    per-head output projection."""
    mha = attn.mha
    H = attn.num_heads
    d = x.size(-1)
    W, b = mha.in_proj_weight, mha.in_proj_bias                # (3d, d), order [q; k; v]
    q, k, v = (x @ W[i * d:(i + 1) * d].t() + (b[i * d:(i + 1) * d] if b is not None else 0)
               for i in range(3))
    q, k, v = (split_heads(t, H) for t in (q, k, v))
    terms = {"content": content_scores(q, k)}
    if getattr(attn, "use_gab", False):
        terms["gab"] = attn._sq_bias(x)
    elif getattr(attn, "use_relative_bias", False):
        terms["relbias"] = attn.relative_bias().unsqueeze(0).expand(x.size(0), -1, -1, -1)
    return AttnDecomp(q, k, v, terms, W_O=out_proj_heads(mha.out_proj.weight, H),
                      b_O=mha.out_proj.bias)


def check_recon(recon: torch.Tensor, target: torch.Tensor, what: str):
    """Assert a re-derived tensor matches the model's own. Relative to the
    target's norm, because absolute float32 error scales with the activations —
    BT4's last layer runs at |x| ≈ 25, where a fixed 1e-4 trips on
    accumulation-order noise of 1e-5 relative."""
    err = (recon - target).norm() / (target.norm() + 1e-12)
    assert float(err) < 1e-3, f"{what} — relative error {float(err):.2e}; do not trust the decomposition"


class GabAccess:
    """The generated attention bias of a smolgen model, layer by layer: the
    engine's `gab_templates` / `gab_coeffs` / `gab_bias` go through this.

    `attn_modules[L]` is layer L's attention module; `shared_weight` the
    (64*64, gen) template bank, or None on a model built without GAB."""

    def __init__(self, attn_modules, shared_weight):
        self.attn_modules = list(attn_modules)
        self.shared_weight = shared_weight
        self._templates = None

    def has(self, layer: int) -> bool:
        return bool(getattr(self.attn_modules[int(layer)], "use_gab", False))

    def gen_size(self, layer: int) -> int | None:
        attn = self.attn_modules[int(layer)]
        return attn.gen_size if self.has(layer) else None

    def _require(self, layer: int):
        if not self.has(layer):
            raise RuntimeError(f"layer {int(layer)} has no GAB (use_gab=False)")
        return self.attn_modules[int(layer)]

    def templates(self) -> torch.Tensor:
        """(gen, 64, 64) on CPU, cached: template i is shared_weight[:, i]."""
        if self.shared_weight is None:
            raise RuntimeError("this model was built without GAB (use_gab=False)")
        if self._templates is None:
            w = self.shared_weight.detach()                    # (64*64, gen)
            self._templates = w.t().reshape(-1, 64, 64).cpu().clone()
        return self._templates

    def coeffs(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """(B, H, gen) for the layer input x, checked on the spot: mixing the
        templates with them must reproduce the layer's own bias."""
        attn = self._require(layer)
        y = coefficients(attn, x)
        B = x.size(0)
        recon = torch.einsum("bhi,oi->bho", y, attn.gab_weight).view(B, attn.num_heads, 64, 64)
        check_recon(recon, attn._sq_bias(x), "smolgen coefficient reconstruction failed")
        return y

    def bias(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """(B, H, 64, 64), the bias added to the scaled QKᵀ logits."""
        return self._require(layer)._sq_bias(x)
