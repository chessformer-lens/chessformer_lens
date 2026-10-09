"""A tiny random-weight chess transformer whose tokens are NOT squares — the
test model for every code path a square-token model never reaches. It plays
nonsense: its weights are random (seeded), not trained. Test-only: importing
this module registers it (tests/conftest.py does), and nothing in the package
knows about it.

    load_engine("toy")          bidirectional attention
    load_engine("toy-causal")   each token attends only to itself and earlier ones

  tokens   8: [cls] (castling rights, rule-50 counter), one "channel" token per
           piece type (that type's signed occupancy — +1 ours, -1 theirs — over
           the 64 squares, side-to-move frame) and [elo] (the rating pair)
  blocks   2, Pre-LN, 4 heads, d_model 32, MLP width 64 — plain attention
           (one "content" score term), so no GAB
  policy   read through the final norm at the [cls] token (bidirectional) or
           the last token (causal, which is the only one that sees them all),
           over the 4352-move square-pair vocabulary
  value    one scalar in [-1, 1] (tanh), no win/draw/loss split
"""
from __future__ import annotations

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer_lens.core import (ArchSpec, AttnDecomp, Capabilities, HeadOutputs, ModelAdapter,
                                   PolicyVocab, Token, TokenLayout, ValueEstimate, canon_square,
                                   content_scores, out_proj_heads, pick_device, register_adapter,
                                   split_heads, uci_vocabulary)

_PIECES = [(chess.PAWN, "pawns"), (chess.KNIGHT, "knights"), (chess.BISHOP, "bishops"),
           (chess.ROOK, "rooks"), (chess.QUEEN, "queens"), (chess.KING, "kings")]
N_TOKENS = 2 + len(_PIECES)
N_FEAT = 64 + 6


def toy_features(board: chess.Board, self_elo, oppo_elo) -> torch.Tensor:
    """(N_TOKENS, N_FEAT): 64 board features then 6 scalar features per token."""
    f = torch.zeros(N_TOKENS, N_FEAT)
    us = board.turn
    f[0, 64:68] = torch.tensor([
        float(board.has_kingside_castling_rights(us)), float(board.has_queenside_castling_rights(us)),
        float(board.has_kingside_castling_rights(not us)),
        float(board.has_queenside_castling_rights(not us))])
    f[0, 68] = board.halfmove_clock / 100.0
    for j, (ptype, _) in enumerate(_PIECES, start=1):
        for sq in board.pieces(ptype, us):
            f[j, canon_square(sq, us)] = 1.0
        for sq in board.pieces(ptype, not us):
            f[j, canon_square(sq, us)] = -1.0
    if self_elo is not None:
        f[-1, 64] = self_elo / 3000.0
        f[-1, 65] = (self_elo if oppo_elo is None else oppo_elo) / 3000.0
    return f


class ToyAttention(nn.Module):
    def __init__(self, d, n_heads, causal):
        super().__init__()
        self.n_heads = n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        mask = torch.triu(torch.ones(N_TOKENS, N_TOKENS, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask if causal else None, persistent=False)

    def scores(self, x):
        q, k, v = (split_heads(m(x), self.n_heads) for m in (self.q, self.k, self.v))
        s = content_scores(q, k)
        if self.mask is not None:
            s = s.masked_fill(self.mask, float("-inf"))
        return q, k, v, s

    def forward(self, x):
        B, T, d = x.shape
        _, _, v, s = self.scores(x)
        z = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(z)


class ToyBlock(nn.Module):
    """Pre-LN: x = x + attn(ln1(x)); x = x + linear2(gelu(linear1(ln2(x)))).
    The Identity modules are hook points for streams no module outputs."""

    def __init__(self, d, n_heads, d_mlp, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn_in = nn.Identity()
        self.attn = ToyAttention(d, n_heads, causal)
        self.resid_mid = nn.Identity()
        self.ln2 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d_mlp)
        self.linear2 = nn.Linear(d_mlp, d)

    def forward(self, x):
        x = self.resid_mid(x + self.attn(self.attn_in(self.ln1(x))))
        return x + self.linear2(F.gelu(self.linear1(self.ln2(x))))


class ToyNet(nn.Module):
    def __init__(self, n_layers=2, d=32, n_heads=4, d_mlp=64, n_moves=4352, causal=False):
        super().__init__()
        self.readout = N_TOKENS - 1 if causal else 0
        self.embed = nn.Linear(N_FEAT, d)
        self.type_emb = nn.Parameter(torch.randn(N_TOKENS, d) * 0.5)
        self.embed_out = nn.Identity()
        self.layers = nn.ModuleList(ToyBlock(d, n_heads, d_mlp, causal) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.policy = nn.Linear(d, n_moves)
        self.value_head = nn.Linear(d, 1)

    def heads(self, x):
        """Policy logits and value from the stream (B, T, d), read at the readout token."""
        h = self.ln_f(x)[:, self.readout]
        return self.policy(h), torch.tanh(self.value_head(h)).squeeze(-1)

    def forward(self, feats):
        x = self.embed_out(self.embed(feats) + self.type_emb)
        for blk in self.layers:
            x = blk(x)
        return self.heads(x)


_TOY = {"toy": ("Toy (random)", False), "toy-causal": ("Toy causal (random)", True)}


@register_adapter
class ToyAdapter(ModelAdapter):
    """The toy model. Not a chess player: a fixture for the non-board paths."""

    name = "toy"
    family = "Toy (random weights, tests only)"

    @classmethod
    def resolve(cls, alias):
        hit = _TOY.get(str(alias).strip().lower())
        return hit[0] if hit else None

    @classmethod
    def models(cls):
        return [{"alias": a, "display_name": name, "aliases": [], "available": True,
                 "note": "random weights"} for a, (name, _) in _TOY.items()]

    @classmethod
    def load(cls, alias="toy", *, device=None, seed=0, **_):
        name, causal = _TOY[str(alias).strip().lower()]
        self = cls()
        self.caps = Capabilities(conditioning=frozenset({"elo"}), value="scalar", norm="pre",
                                 attn_terms=("content",), causal=causal)
        self.device = pick_device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = ToyNet(causal=causal).to(self.device).eval()
        a = self.arch = ArchSpec(n_layers=len(self.model.layers), n_heads=4, d_model=32, d_mlp=64,
                                 name=str(alias).strip().lower(), display_name=name)
        self.cfg = a.legacy_cfg(device=self.device, history=1)
        self.spec = a.legacy_spec(aliases=())
        self.vocab = PolicyVocab(uci_vocabulary())
        return self

    def tokens(self, boards):
        return torch.stack([toy_features(b, None, None) for b in boards]).to(self.device)

    def forward(self, boards, cond):
        feats = torch.stack([toy_features(b, cond.self_elo, cond.oppo_elo) for b in boards])
        policy, value = self.model(feats.to(self.device))
        return HeadOutputs(policy=policy, value=value)

    def value(self, raw):
        v = float(raw.reshape(-1)[0])
        return ValueEstimate(expected=(v + 1.0) / 2.0)

    def layout(self, board):
        toks = [Token("control", "[cls]")]
        toks += [Token("channel", label) for _, label in _PIECES]
        toks.append(Token("control", "[elo]"))
        return TokenLayout(tuple(toks), frame="stm_mirror", readout=self.model.readout,
                           is_board=False)

    def hook_modules(self):
        hooks = {"embed_in": self.model.embed_out}
        for i, blk in enumerate(self.model.layers):
            hooks[f"attn_in_{i:02d}"] = blk.attn_in
            hooks[f"attn_{i:02d}"] = blk.attn
            hooks[f"postattn_{i:02d}"] = blk.resid_mid
            hooks[f"mlp_{i:02d}"] = blk.linear2
            hooks[f"block_{i:02d}"] = blk
        return hooks

    def unembed(self, x):
        """Policy logits from one residual (T, d): through the final norm, at the readout token."""
        return self.model.heads(x.unsqueeze(0))[0][0]

    def decompose_attention(self, layer, x, cond):
        attn = self.model.layers[int(layer)].attn
        q, k, v, _ = attn.scores(x)
        return AttnDecomp(q, k, v, {"content": content_scores(q, k)},
                          W_O=out_proj_heads(attn.o.weight, attn.n_heads), b_O=attn.o.bias,
                          mask=attn.mask)

    def mlp_out(self, layer):
        return self.model.layers[int(layer)].linear2


__all__ = ["ToyAdapter", "ToyNet", "toy_features"]
