"""Maia-3 / Chessformer checkpoints (3M ablation, 5M, 23M, 79M), through the
`maia3` package — which is not on PyPI, so it is imported only when a Maia-3
model is actually requested:

    pip install https://github.com/CSSLab/maia3/archive/1e13597c42d4858b7cfd7cfdae01e297263364b2.zip

  tokens    the 64 squares, side-to-move frame, the current position's pieces
            padded to fill `history`
  policy    64x64 square pairs + 256 promotions (4352)
  value     loss / draw / win, plus a think-time head ("ponder")
  rating    self_elo / oppo_elo are real inputs (an embedding on every token)
  clock     (base, increment, clock left) in seconds, on checkpoints trained
            with time information; zeros otherwise
"""
from __future__ import annotations

import math
import os
import re
import types
from collections import deque

import torch

from ..core import (ArchSpec, Capabilities, HeadOutputs, ModelAdapter, PolicyVocab,
                    ValueEstimate, _register_builtin, pick_device)
from .smolgen import GabAccess, decompose

MAIA3_INSTALL = ("pip install https://github.com/CSSLab/maia3/archive/"
                 "1e13597c42d4858b7cfd7cfdae01e297263364b2.zip")

# Maia-3 aliases (maia3-5m, 23m, maia3-3m-ablation, ...), recognisable without
# importing maia3, so a missing install is reported as such rather than as an
# unknown model.
_MAIA_LIKE = re.compile(r"(maia3?[-_ ]?)?\d+m([-_ ]?ablation)?", re.I)


def _maia3():
    """The maia3 modules this adapter uses, imported on first need."""
    try:
        from maia3 import model_registry, utils, uci, dataset
    except ModuleNotFoundError as exc:
        if exc.name != "maia3" and not str(exc.name or "").startswith("maia3."):
            raise
        raise ModuleNotFoundError(
            "Maia-3 models need the Maia-3 model code, which is not on PyPI and so "
            "is not installed by `pip install chessformer_lens`.\n\n"
            f"    {MAIA3_INSTALL}\n\n"
            "(Leela BT4 and the other engines work without it.)"
        ) from exc
    except ImportError as exc:
        # maia3 imports torch.nn.RMSNorm, which only exists from torch 2.4.
        if "RMSNorm" not in str(exc):
            raise
        raise ImportError(
            f"Maia-3 needs torch >= 2.4; this environment has {torch.__version__}.\n"
            "    pip install --upgrade 'torch>=2.4'"
        ) from exc
    return types.SimpleNamespace(registry=model_registry, utils=utils, uci=uci, dataset=dataset)


def build_cfg(alias="maia3-5m", device=None, checkpoint_path=None, trust_checkpoint=False):
    """Build the args-namespace the model + load_model expect, using the repo's
    own model spec so dim_vit / num_heads / gab_* / history all match the weights.

    `trust_checkpoint=True` becomes torch.load(weights_only=False) downstream,
    i.e. it will execute pickled code from the checkpoint — see MaiaEngine."""
    m = _maia3()
    cfg = types.SimpleNamespace()
    spec = m.registry.resolve_model_spec(alias)
    m.registry.apply_model_config(cfg, spec)   # copies the architecture preset onto cfg
    cfg.model_spec = spec
    cfg.device = pick_device(device)
    cfg.trust_checkpoint = trust_checkpoint
    cfg.checkpoint_path = checkpoint_path      # None -> resolved from HF cache in load()
    return cfg, spec


@_register_builtin
class Maia3Adapter(ModelAdapter):
    """A Maia-3 checkpoint, downloaded from Hugging Face on first use."""

    name = "maia3"
    family = "Maia-3"
    caps = Capabilities(conditioning=frozenset({"elo"}), value="wdl", norm="post",
                        attn_terms=("content", "gab"), aux_heads=frozenset({"ponder"}))

    @classmethod
    def resolve(cls, alias):
        try:
            m = _maia3()
        except ImportError:
            if _MAIA_LIKE.fullmatch(str(alias).strip()):
                raise
            return None
        try:
            return m.registry.resolve_model_spec(alias).display_name
        except m.registry.ModelResolutionError:
            return None

    @classmethod
    def models(cls):
        try:
            m = _maia3()
        except ImportError:
            return [{"alias": "maia3-5m", "display_name": "Maia-3", "aliases": [],
                     "available": False, "note": f"needs the maia3 package: {MAIA3_INSTALL}"}]
        out = []
        for spec in m.registry.MODEL_SPECS:
            try:
                m.registry.resolve_checkpoint_path(spec, local_files_only=True)
                note = "downloaded"
            except Exception:
                note = "downloads from Hugging Face on first use"
            out.append({"alias": spec.name, "display_name": spec.display_name,
                        "aliases": list(spec.aliases), "available": True, "note": note})
        return out

    @classmethod
    def load(cls, alias="maia3-5m", *, device=None, checkpoint_path=None,
             trust_checkpoint=False, **_):
        """`device`: an explicit string wins; otherwise see `pick_device`.
        `trust_checkpoint=True` loads with `weights_only=False`, i.e. it can
        execute pickled code from the checkpoint file — only use it for
        checkpoints you produced yourself."""
        m = _maia3()
        self = cls()
        self.cfg, self.spec = build_cfg(alias, device, checkpoint_path, trust_checkpoint)
        if self.cfg.checkpoint_path is None:
            # Use the checkpoint from the local HF cache if present, otherwise
            # download it from Hugging Face — so the app runs on a fresh machine.
            # Say so before the download starts: it is hundreds of MB
            try:
                self.cfg.checkpoint_path = m.registry.resolve_checkpoint_path(
                    self.spec, local_files_only=True)
            except Exception:
                if os.environ.get("CHESSFORMER_NO_DOWNLOAD"):
                    raise FileNotFoundError(
                        f"{alias}: weights are not in the local Hugging Face cache and "
                        f"CHESSFORMER_NO_DOWNLOAD is set. Unset it to download them from "
                        f"https://huggingface.co/UofTCSSLab, or pass checkpoint_path") from None
                print(f"chessformer_lens: {alias} weights are not in the local "
                      f"Hugging Face cache; downloading them now (this is a "
                      f"one-time, several-hundred-MB fetch for the larger "
                      f"models).\n  from:  https://huggingface.co/UofTCSSLab\n"
                      f"  cache: {os.environ.get('HF_HOME') or '~/.cache/huggingface'}",
                      flush=True)
                self.cfg.checkpoint_path = m.registry.resolve_checkpoint_path(
                    self.spec, local_files_only=False)

        self.device = self.cfg.device
        self.model = m.uci.load_model(self.cfg)   # builds MAIA3Model(cfg), loads weights, .eval()
        self.model.to(self.device)                # no-op if load_model already placed it
        self._m = m
        cfg = self.cfg
        if getattr(cfg, "include_time_info", False):           # a checkpoint with clock inputs
            self.caps = Capabilities(**{**vars(cls.caps), "conditioning": frozenset({"elo", "clock"})})
        self.arch = ArchSpec(n_layers=cfg.num_blocks, n_heads=cfg.num_heads, d_model=cfg.dim_vit,
                             d_mlp=int(cfg.dim_vit * cfg.mlp_ratio), name=self.spec.name,
                             display_name=self.spec.display_name,
                             checkpoint=str(cfg.checkpoint_path))
        # exact index <-> UCI mapping used by the released engine
        self.vocab = PolicyVocab(m.utils.get_all_possible_moves())
        layers = self.model.transformer.layers
        self._gab = (GabAccess([blk.self_attn for blk in layers], self.model.gab_shared_weight)
                     if getattr(self.model, "gab_shared_weight", None) is not None else None)
        return self

    # --- inputs and outputs ---
    def _board_tokens(self, board, clock=None):
        """Single current position, padded to fill `history` (matches the
        default `--use-uci-history` OFF behavior of the released engine)."""
        base, inc, left = clock if clock is not None else (0.0, 0.0, 0.0)
        hist = deque([self._m.dataset.tokenize_board(board)], maxlen=self.cfg.history)
        toks = self._m.dataset.get_historical_tokens(
            hist, self.cfg, base=base, inc=inc, clk_left_before=left, clk_ponder=0.0)
        return toks.unsqueeze(0).to(self.device)

    def tokens(self, boards, clock=None):
        return torch.cat([self._board_tokens(b, clock) for b in boards], dim=0)

    def forward(self, boards, cond):
        if cond.self_elo is None:
            raise ValueError("Maia-3 is conditioned on a rating: pass self_elo "
                             "(e.g. eng.evaluate(board, 1500))")
        toks = self.tokens(boards, cond.clock)
        B = len(boards)
        self_elos = torch.full((B,), int(cond.self_elo), dtype=torch.long, device=self.device)
        oppo_elos = torch.full((B,), int(cond.opponent_elo), dtype=torch.long, device=self.device)
        logits_move, logits_value, logits_ponder = self.model(toks, self_elos, oppo_elos)
        return HeadOutputs(policy=logits_move, value=logits_value, aux={"ponder": logits_ponder})

    def value(self, raw):
        """Maia-3's value logits are (loss, draw, win)."""
        loss, draw, win = torch.softmax(raw.float(), dim=-1).tolist()
        return ValueEstimate.from_wdl(win, draw, loss)

    # --- internals ---
    def hook_modules(self):
        m = self.model
        hooks = {"embed_in": m.abs_pe if hasattr(m, "abs_pe") else m.token_projection}
        for i, blk in enumerate(m.transformer.layers):
            hooks[f"block_{i:02d}"] = blk
            hooks[f"attn_{i:02d}"] = blk.self_attn
            hooks[f"mlp_{i:02d}"] = blk.linear2
            hooks[f"postattn_{i:02d}"] = blk.norm1
        hooks["encoder_out"] = m.transformer.norm
        return hooks

    def unembed(self, x):
        """Full (4352,) move logits from one position's residual x (64, dim),
        replicating MAIA3Model.forward's policy head (64*64 moves + 256
        promotions). `x` must be exactly (64, dim) with no batch dim."""
        hid = self.cfg.head_hid_dim
        sq_from = self.model.proj_sq_from(x)                  # (64, hid)
        sq_to = self.model.proj_sq_to(x)                      # (64, hid)
        scores = (sq_from @ sq_to.t()) / math.sqrt(hid)      # (64, 64)
        promo_bias = self.model.promo_bias_proj(sq_to[56:64]) * math.sqrt(hid)  # (8 files, 4 pieces)
        promo = [scores[48 + ff, 56 + tf] + promo_bias[tf, pc]
                 for ff in range(8) for tf in range(8) for pc in range(4)]      # (256,)
        return torch.cat([scores.reshape(-1), torch.stack(promo)])             # (4352,)

    def unembed_frozen(self, x, x0):
        """The final norm as the affine map it was on `x0`, then the bilinear head."""
        ln = self.model.transformer.norm
        mu = x0.mean(-1, keepdim=True)
        s = torch.sqrt(x0.var(-1, unbiased=False, keepdim=True) + ln.eps)
        return self.unembed((x - mu) / s * ln.weight + ln.bias)

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
        return {"gab": self._gab} if self._gab is not None else {}
