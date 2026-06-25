"""
Maia-3 · play & probe — desktop app.

Play a transformer based chess bot (Maia-3) trained to mimic human play and watch its move policy, 
its attention (regular self attention vs unique geometric GAB), 
and how its residual stream evolves with depth. 

Drag the ELO slider to re-evaluate a position at
different skill levels (e.g. eval at very low ELO for King and Queen vs King comes out to
roughly 75% chance draw).

Everything is in this one file:
  1. MODEL ENGINE   — loads the checkpoint, runs forward passes, captures hooks
  2. GAME + BRIDGE  — game state and the methods the UI calls (window.pywebview.api)
  3. UI             — the whole interface (HTML + CSS + JS) as one string
  4. LAUNCH         — opens the native window

Run:  python app.py        (see README.md)
"""
import math
import os
import sys
import threading
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

HERE = Path(__file__).resolve().parent
ACT_DIR = HERE / "activations"


# ============================================================================
# 1. MODEL ENGINE
# ============================================================================
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

        # exact index <-> UCI mapping used by the released engine
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        self.activation_dir = Path(activation_dir)
        self.activation_dir.mkdir(parents=True, exist_ok=True)

        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    # ----- activation hooks -------------------------------------------------
    def _register_hooks(self):
        """Capture the residual stream entering block 0, after every block
        (post-LN), and after the final encoder norm. Overwritten each forward,
        so the snapshot always corresponds to the most recent position."""
        def make_hook(name):
            def hook(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                self._activations[name] = t.detach().to("cpu")
            return hook

        self._hooks.append(
            self.model.token_projection.register_forward_hook(make_hook("embed_in"))
        )
        for i, blk in enumerate(self.model.transformer.layers):
            self._hooks.append(
                blk.register_forward_hook(make_hook(f"block_{i:02d}"))
            )
        self._hooks.append(
            self.model.transformer.norm.register_forward_hook(make_hook("encoder_out"))
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ----- tokenization -----------------------------------------------------
    def _tokens(self, board: chess.Board) -> torch.Tensor:
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
    def evaluate(self, board: chess.Board, self_elo: int, oppo_elo: int | None = None):
        """One forward pass. Returns the full normalized policy over legal moves
        (descending), the WDL for the side to move, and stashes activations."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self._activations = {}

        tokens = self._tokens(board)
        self_elos = torch.tensor([int(self_elo)], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([int(oppo_elo)], dtype=torch.long, device=self.device)

        logits_move, logits_value, _ = self.model(tokens, self_elos, oppo_elos)

        logits = logits_move[0].float()
        legal_mask = get_legal_moves_mask(board, self.all_moves_dict).to(self.device)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)   # normalized over legal moves

        policy = []
        for idx in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist():
            mv = self._idx_to_move(board, idx)
            if mv is not None:
                policy.append((mv.uci(), float(probs[idx])))
        policy.sort(key=lambda x: x[1], reverse=True)

        loss, draw, win = torch.softmax(logits_value[0].float(), dim=-1).tolist()
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

    # ----- live attention (per layer / head) --------------------------------
    @torch.no_grad()
    def attention(self, board, self_elo, oppo_elo=None, layer=0, head=0):
        """Return the 64x64 attention components of one (layer, head) for the
        current position, reproducing Chessformer Fig. 1:
          qk   = semantic dot-product logits (scaled QK^T)
          gab  = geometric attention bias (the learned positional bias)
          attn = softmax(qk + gab)   -- the attention the block actually uses
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
            "qk": qk[0, h].cpu().tolist(),
            "gab": gab[0, h].cpu().tolist(),
            "attn": attn[0, h].cpu().tolist(),
        }

    # ----- residual-stream evolution across depth ---------------------------
    @torch.no_grad()
    def residual_stream(self, board, self_elo, oppo_elo=None):
        """Per-square summaries of the residual stream at every captured depth
        (embed_in, block_00..block_07, encoder_out) for the current position:
          norm      = ||x_L|| per square
          delta     = ||x_L - x_(L-1)|| per square  (where each block edits the stream)
          cos_final = cosine(x_L, x_final) per square (how early a square settles)
        Each is (num_layers, 64), side-to-move frame, square = rank*8 + file."""
        oppo_elo = self_elo if oppo_elo is None else oppo_elo
        self.evaluate(board, self_elo, oppo_elo)
        names = (["embed_in"]
                 + [f"block_{i:02d}" for i in range(self.cfg.num_blocks)]
                 + ["encoder_out"])
        acts = [self._activations[n][0] for n in names]      # each (64, dim)
        final = acts[-1]
        norm, delta, cos_final = [], [], []
        for i, a in enumerate(acts):
            norm.append(a.norm(dim=-1).tolist())
            delta.append([0.0] * 64 if i == 0
                         else (a - acts[i - 1]).norm(dim=-1).tolist())
            cos_final.append(torch.nn.functional.cosine_similarity(a, final, dim=-1).tolist())
        labels = ["emb"] + [f"b{i}" for i in range(self.cfg.num_blocks)] + ["enc"]
        return {"labels": labels, "norm": norm, "delta": delta, "cos_final": cos_final}

    # ----- activation dump --------------------------------------------------
    def save_activations(self, filename: str, meta: dict | None = None) -> str:
        """Persist the most recent forward's residual-stream snapshot.
        Each tensor is (64, dim_vit); keys: embed_in, block_00..block_07, encoder_out."""
        snap = {k: v.squeeze(0).clone() for k, v in self._activations.items()}
        snap["meta"] = meta or {}
        path = self.activation_dir / filename
        torch.save(snap, path)
        return str(path)


# ============================================================================
# 2. GAME STATE + UI BRIDGE
# ============================================================================
def side_name(turn):
    return "white" if turn == chess.WHITE else "black"


class MaiaApi:
    """Everything JS can call. Methods return plain JSON-able dicts."""

    def __init__(self):
        self.engine = None
        self.ready = False
        self.error = None
        self.board = chess.Board()
        self.human = chess.WHITE
        self.human_both = False
        self.san_history = []
        self._lock = threading.Lock()
        # load the (small) model off the UI thread so the window opens instantly
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        import traceback
        try:
            print("[maia-app] loading model…", file=sys.stderr, flush=True)
            self.engine = MaiaEngine(
                alias=os.environ.get("MAIA3_ALIAS", "maia3-5m"),
                activation_dir=str(ACT_DIR),
            )
            self.ready = True
            print(f"[maia-app] model ready on {self.engine.device}: "
                  f"{self.engine.cfg.checkpoint_path}", file=sys.stderr, flush=True)
        except Exception as exc:  # surface load errors to the UI instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            print("[maia-app] MODEL LOAD FAILED:\n" + traceback.format_exc(),
                  file=sys.stderr, flush=True)

    # ----- introspection ----------------------------------------------------
    def info(self):
        return {
            "ready": self.ready,
            "error": self.error,
            "alias": self.engine.spec.display_name if self.ready else None,
            "device": self.engine.device if self.ready else None,
            "checkpoint": self.engine.cfg.checkpoint_path if self.ready else None,
            "num_blocks": self.engine.cfg.num_blocks if self.ready else None,
            "num_heads": self.engine.cfg.num_heads if self.ready else None,
            "dim_vit": self.engine.cfg.dim_vit if self.ready else None,
            "activation_dir": str(ACT_DIR),
        }

    # ----- state ------------------------------------------------------------
    def _base(self):
        b = self.board
        over = b.is_game_over()
        return {
            "fen": b.fen(),
            "turn": side_name(b.turn),
            "human_color": "both" if self.human_both else side_name(self.human),
            "human_to_move": (not over) and (self.human_both or b.turn == self.human),
            "maia_to_move": (not over) and (not self.human_both) and (b.turn != self.human),
            "legal_moves": [m.uci() for m in b.legal_moves],
            "move_number": b.fullmove_number,
            "ply": len(b.move_stack),
            "last_move": b.move_stack[-1].uci() if b.move_stack else None,
            "in_check": b.is_check(),
            "game_over": over,
            "result": b.result() if over else None,
            "termination": (str(b.outcome().termination).split(".")[-1]
                            if over and b.outcome() else None),
            "san_history": list(self.san_history),
        }

    def new_game(self, human_color="white"):
        self.board = chess.Board()
        self.human_both = (human_color == "both")
        self.human = chess.BLACK if human_color == "black" else chess.WHITE
        self.san_history = []
        return self._base()

    # ----- model-backed -----------------------------------------------------
    def _policy_for_current(self, elo, save=True):
        b = self.board
        with self._lock:
            res = self.engine.evaluate(b, self_elo=int(elo))
            act_file = None
            if save and not b.is_game_over():
                fname = f"ply{len(b.move_stack):03d}_{side_name(b.turn)}_elo{int(elo)}.pt"
                act_file = self.engine.save_activations(fname, meta={
                    "fen": b.fen(),
                    "self_elo": int(elo),
                    "side_to_move": side_name(b.turn),
                    "ply": len(b.move_stack),
                })
        pol = []
        for uci, p in res["policy"]:
            try:
                san = b.san(chess.Move.from_uci(uci))
            except Exception:
                san = uci
            pol.append({"uci": uci, "san": san, "p": p})
        return pol, res["wdl"], act_file

    def policy(self, elo=1500, save=True):
        """Re-evaluate the CURRENT position at a given ELO (no move made).
        This is the slider probe + per-position activation dump."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        st = self._base()
        if st["game_over"]:
            return {**st, "policy": [], "wdl": None, "activation_file": None}
        pol, wdl, act = self._policy_for_current(elo, save=save)
        return {**st, "policy": pol, "wdl": wdl, "activation_file": act}

    def human_move(self, uci):
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if not self.human_both and b.turn != self.human:
            return {**self._base(), "error": "not your turn"}
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            return {**self._base(), "error": f"bad uci: {uci}"}
        if mv not in b.legal_moves:
            return {**self._base(), "error": f"illegal move: {uci}"}
        self.san_history.append(b.san(mv))
        b.push(mv)
        return self._base()

    def maia_move(self, elo=1500, temperature=1.0):
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over() or self.human_both:
            return self._base()
        if b.turn == self.human:
            return {**self._base(), "error": "not Maia's turn"}

        # policy + activations for the position Maia is about to move in
        pol, wdl, act = self._policy_for_current(elo, save=True)
        with self._lock:
            mv, _ = self.engine.select_move(
                b, self_elo=int(elo), temperature=float(temperature)
            )
        if mv is None and pol:
            mv = chess.Move.from_uci(pol[0]["uci"])
        maia = None
        if mv is not None:
            maia = {"uci": mv.uci(), "san": b.san(mv)}
            self.san_history.append(maia["san"])
            b.push(mv)
        return {**self._base(), "maia_move": maia, "maia_policy": pol,
                "maia_wdl": wdl, "activation_file": act}

    def undo(self):
        """Step back to the previous position where it is the human's move
        (pops Maia's reply and your move). Pops at least one ply."""
        b = self.board
        if not b.move_stack:
            return self._base()
        b.pop()
        if self.san_history:
            self.san_history.pop()
        while b.move_stack and not self.human_both and b.turn != self.human:
            b.pop()
            if self.san_history:
                self.san_history.pop()
        return self._base()

    def attention(self, elo=1500, layer=0, head=0):
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        with self._lock:
            return self.engine.attention(b, self_elo=int(elo),
                                         layer=int(layer), head=int(head))

    def residual(self, elo=1500):
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        with self._lock:
            return self.engine.residual_stream(b, self_elo=int(elo))


# ============================================================================
# 3. UI  (HTML + CSS + JS, one string — edit the interface here)
# ============================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maia-3 · play & probe</title>
<style>
  :root{
    --bg:#0e1014; --panel:#161a21; --panel2:#1b2029; --line:#262c37;
    --text:#e7eaf0; --muted:#8b93a3; --accent:#6ea8fe; --accent2:#7bd88f;
    --sq-light:#c9d1dc; --sq-dark:#6b7686;
    --hl:rgba(245,213,107,.50); --sel:#7bd88f; --win:#5fb878; --draw:#6b7480; --loss:#d9606a;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;}
  .wrap{display:flex;gap:22px;padding:20px;height:100%;align-items:flex-start}
  .left{display:flex;flex-direction:column;gap:12px}
  .right{flex:1;display:flex;flex-direction:column;gap:14px;min-width:300px;max-width:360px;height:100%}
  .arch{flex:0 0 384px;display:flex;flex-direction:column;height:100%}

  h1{font-size:15px;font-weight:600;letter-spacing:.3px;margin:0}
  .sub{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:3px}

  /* board */
  #board{width:576px;height:576px;display:grid;grid-template-columns:repeat(8,1fr);
    grid-template-rows:repeat(8,1fr);border-radius:8px;overflow:hidden;
    box-shadow:0 10px 40px rgba(0,0,0,.45);user-select:none}
  .sq{position:relative;display:flex;align-items:center;justify-content:center;
    font-size:60px;line-height:1;cursor:default}
  .sq.light{background:var(--sq-light)} .sq.dark{background:var(--sq-dark)}
  .sq.lastmove::after{content:"";position:absolute;inset:0;background:var(--hl)}
  .sq.sel{box-shadow:inset 0 0 0 4px var(--sel)}
  .sq .pc{position:relative;z-index:2;text-shadow:0 1px 2px rgba(0,0,0,.35)}
  .pc.white{color:#fbfdff} .pc.black{color:#15181e}
  .sq .dot{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:30%;height:30%;border-radius:50%;background:rgba(40,50,40,.42);z-index:1;pointer-events:none}
  .sq.cap .dot{width:86%;height:86%;background:transparent;
    box-shadow:inset 0 0 0 4px rgba(40,50,40,.40)}
  .sq.playable{cursor:pointer}
  .sq.attq::before{content:"";position:absolute;inset:2px;border:2px solid rgba(110,168,254,.75);border-radius:4px;z-index:1;pointer-events:none}
  .coord{position:absolute;font-size:9px;font-family:var(--mono);color:rgba(20,24,30,.55);z-index:3}
  .coord.f{right:3px;bottom:2px} .coord.r{left:3px;top:2px}

  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button,select{background:var(--panel2);color:var(--text);border:1px solid var(--line);
    border-radius:7px;padding:7px 11px;font-size:12px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:600}
  label.lbl{font-size:11px;color:var(--muted)}

  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
  .card h2{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.6px}

  /* elo slider */
  .elorow{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px}
  .eloval{font-family:var(--mono);font-size:26px;font-weight:600;color:var(--accent)}
  input[type=range]{-webkit-appearance:none;width:100%;height:5px;border-radius:4px;
    background:linear-gradient(90deg,var(--accent2),var(--accent));outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;
    border-radius:50%;background:#fff;border:3px solid var(--accent);cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.4)}
  .ticks{display:flex;justify-content:space-between;font-size:9px;color:var(--muted);
    font-family:var(--mono);margin-top:5px}
  .temprow{display:flex;align-items:center;gap:8px;margin-top:12px}
  .temprow input{width:64px}

  /* wdl */
  .wdl{display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:10px;
    font-family:var(--mono);color:#0d130d}
  .wdl div{display:flex;align-items:center;justify-content:center;min-width:0}
  .wdl .w{background:var(--win)} .wdl .d{background:#6b7480;color:#10141a} .wdl .l{background:var(--loss)}

  /* policy list */
  #policy{flex:1;overflow-y:auto;min-height:0}
  .prow{display:grid;grid-template-columns:52px 1fr 52px;align-items:center;gap:10px;
    padding:4px 0;font-size:12px}
  .prow .san{font-family:var(--mono);color:var(--text)}
  .prow .barwrap{height:18px;background:#0f131a;border:1px solid var(--line);
    border-radius:10px;overflow:hidden}
  .prow .barwrap{display:block}
  .prow .bar{display:block;height:100%;min-width:3px;border-radius:10px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .18s ease}
  .prow .pct{font-family:var(--mono);text-align:right;color:var(--muted)}
  .prow.top .san{color:var(--accent2);font-weight:700}
  .prow.top .pct{color:var(--accent2)}
  .prow.played .barwrap{box-shadow:0 0 0 2px var(--hl)}

  /* architecture diagram */
  .diagram{padding:10px 12px}
  .diagram svg{display:block;width:100%;height:auto}

  /* live attention panel */
  .attctrls{display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
  .chiprow{display:flex;align-items:flex-start;gap:8px}
  .chiprow .lbl{flex:0 0 38px;padding-top:4px}
  .chips{display:flex;flex:1;min-width:0;flex-wrap:wrap;gap:4px}
  .chip{padding:3px 8px;font-size:11px;border:1px solid var(--line);border-radius:6px;
    background:var(--panel2);cursor:pointer;font-family:var(--mono);color:var(--text)}
  .chip:hover{border-color:var(--accent)}
  .chip.active{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:700}
  .attcap{font-size:10px;color:var(--muted);margin-bottom:10px;line-height:1.45}
  .attset{display:flex;flex-direction:column;gap:12px}
  .attlabel{font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:4px}
  .attboard{width:100%;max-width:202px;aspect-ratio:1/1;margin:0 auto;display:grid;
    grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);
    border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#10141b}
  .attcell{cursor:pointer}
  .attcell.q{box-shadow:inset 0 0 0 2px #ff5d6c}

  /* residual-stream filmstrip */
  .resid{padding:12px 14px}
  .residctrls{display:flex;gap:6px;margin-bottom:10px}
  .rbtn{padding:4px 9px;font-size:11px}
  .rbtn.active{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:600}
  .film{display:flex;gap:6px}
  .filmcol{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0}
  .miniboard{width:100%;aspect-ratio:1/1;display:grid;grid-template-columns:repeat(8,1fr);
    grid-template-rows:repeat(8,1fr);border:1px solid var(--line);border-radius:3px;overflow:hidden;background:#10141b}
  .filmlbl{font-size:8px;color:var(--muted);font-family:var(--mono)}
  .resid.hidden{display:none}

  .status{font-size:12px;color:var(--muted);min-height:16px}
  .status b{color:var(--text)}
  .act{font-size:10px;color:var(--muted);font-family:var(--mono);word-break:break-all}
  .moves{font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.6;
    max-height:70px;overflow-y:auto}

  /* overlays */
  #promo,#loading{position:fixed;inset:0;background:rgba(10,12,16,.72);display:none;
    align-items:center;justify-content:center;z-index:50}
  #promo .box,#loading .box{background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:20px;text-align:center}
  #promo .glyphs{display:flex;gap:6px;margin-top:10px}
  #promo .glyphs button{font-size:34px;padding:6px 12px;line-height:1}
  .spinner{width:26px;height:26px;border:3px solid var(--line);border-top-color:var(--accent);
    border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 12px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div>
      <h1>Maia-3 · play &amp; probe</h1>
      <div class="sub" id="modelinfo">loading model…</div>
    </div>
    <div id="board"></div>
    <div class="controls">
      <button class="primary" id="newbtn">New game</button>
      <button id="undobtn">← Back</button>
      <label class="lbl">You play</label>
      <select id="color"><option value="white">White</option><option value="black">Black</option><option value="both">Both (analyze)</option></select>
      <label class="lbl"><input type="checkbox" id="showresid" checked> residual</label>
      <span class="status" id="status" style="margin-left:auto"></span>
    </div>

    <div class="card resid">
      <h2>Residual stream across depth · this position</h2>
      <div class="residctrls">
        <button class="rbtn active" data-m="delta">Δ from prev</button>
        <button class="rbtn" data-m="cos_final">cos → final</button>
        <button class="rbtn" data-m="norm">‖·‖ norm</button>
      </div>
      <div class="film" id="film"></div>
      <div class="act" id="residinfo" style="margin-top:6px"></div>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <div class="elorow">
        <h2 style="margin:0">Maia rating (self_elo)</h2>
        <span class="eloval" id="eloval">1500</span>
      </div>
      <input type="range" id="elo" min="600" max="2800" step="25" value="1500">
      <div class="ticks"><span>600</span><span>1100</span><span>1600</span><span>2100</span><span>2800</span></div>
    </div>

    <div class="card">
      <h2>Win / Draw / Loss · side to move</h2>
      <div class="wdl" id="wdl"><div class="w" style="width:33%">—</div><div class="d" style="width:34%"></div><div class="l" style="width:33%"></div></div>
    </div>

    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2 id="poltitle">Policy over legal moves</h2>
      <div id="policy"></div>
      <div class="act" id="actfile" style="margin-top:8px"></div>
    </div>

    <div class="card">
      <h2>Moves</h2>
      <div class="moves" id="moves">—</div>
    </div>
  </div>

  <div class="arch">
    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0;overflow:auto">
      <h2>Live attention · this position</h2>
      <div class="attctrls">
        <div class="chiprow"><span class="lbl">Layer</span><div class="chips" id="layerChips"></div></div>
        <div class="chiprow"><span class="lbl">Head</span><div class="chips" id="headChips"></div></div>
      </div>
      <div class="attcap">QKᵀ ⊕ GAB → softmax, computed for the position on the board.
        Click any square to set the query — each board shows attention <b>from</b> that square.</div>
      <div class="attset">
        <div><div class="attlabel">QKᵀ · semantic (dot-product)</div><div class="attboard" id="att_qk"></div></div>
        <div><div class="attlabel">GAB · geometric bias</div><div class="attboard" id="att_gab"></div></div>
        <div><div class="attlabel">attention · softmax(QKᵀ + GAB)</div><div class="attboard" id="att_attn"></div></div>
      </div>
      <div class="act" id="attinfo" style="margin-top:8px"></div>
    </div>
  </div>
</div>

<div id="loading"><div class="box"><div class="spinner"></div><div id="loadtext">Loading Maia3-5M…</div></div></div>
<div id="promo"><div class="box"><div>Promote to</div><div class="glyphs" id="promoglyphs"></div></div></div>

<script>
const FILES=['a','b','c','d','e','f','g','h'];
const GLYPH={k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const $=id=>document.getElementById(id);

let API=null, cur=null, orient='white', sel=null, busy=false;
let elo=1500, temp=0, pendingPromo=null, MODEL_INFO=null;
const MAXPOL=14;

const sleep=ms=>new Promise(r=>setTimeout(r,ms));

/* ---- wait for the python bridge, then boot (poll; don't rely on the event) ---- */
let booted=false, booting=false, waitN=0;
window.addEventListener('pywebviewready', tryBoot);
function showLoading(msg){
  $('loadtext').textContent=msg;
  $('loading').style.display='flex';
}
function tryBoot(){
  if(booted || booting) return;
  if(window.pywebview && window.pywebview.api){ boot(); return; }
  if(++waitN===1) showLoading('Connecting to Python bridge…');
  if(waitN>120){ showLoading('Bridge not connecting — check the terminal for errors.'); return; }
  setTimeout(tryBoot,100);
}
showLoading('Loading Maia3-5M…');
tryBoot();

async function boot(){
  if(booted || booting) return;
  booting=true;
  showLoading('Loading Maia3-5M…');
  try{
    API = window.pywebview.api;
    let info = await API.info();
    let n=0;
    while(!info.ready && !info.error){ await sleep(400); info = await API.info(); if(++n>300) break; }
    if(info.error){ showLoading('Model failed to load:\n'+info.error); return; }
    if(!info.ready){ showLoading('Model load timed out — check the terminal.'); return; }
    MODEL_INFO = info;
    setModelInfo();
    initAttUi(info);
    console.log('[maia] bridge ready', info);
    $('loading').style.display='none';
    booted=true;
    await newGame();
  }catch(e){
    showLoading('Bridge error: '+(e && e.message ? e.message : e));
    setTimeout(tryBoot, 500);
  }finally{
    booting=false;
  }
}

function setModelInfo(){
  if(!MODEL_INFO) return;
  const i=MODEL_INFO;
  $('modelinfo').textContent =
    `${i.alias||'Maia3-5M'} · ${i.device||'cpu'} · ${i.num_blocks||8} blocks × ${i.dim_vit||256}d`;
}

/* ---- controls ---- */
$('newbtn').onclick = ()=>{ if(!busy) newGame(); };
$('undobtn').onclick = ()=>{ if(!busy) doUndo(); };
$('color').addEventListener('change', ()=>{ if(!busy) newGame(); });
$('elo').addEventListener('input', e=>{
  elo = +e.target.value; $('eloval').textContent = elo;
  scheduleProbe();
});

let probeTimer=null;
function scheduleProbe(){
  clearTimeout(probeTimer);
  probeTimer=setTimeout(probe, 180);   // debounce slider -> re-evaluate same position
}
async function probe(){
  if(!API || busy || !cur || cur.game_over || !cur.human_to_move) return;
  const d = await API.policy(elo, true);
  if(d.error) return;
  cur = d; renderPolicy(d.policy, d.wdl, d.activation_file, null); renderBoard();
  updateAttention(); updateResidual();
}

async function newGame(){
  sel=null; busy=true;
  const hc = $('color').value;
  cur = await API.new_game(hc);
  orient = (hc==='black') ? 'black' : 'white';
  renderBoard(); renderMoves(); setModelInfo();
  busy=false;
  await advance();
}

async function doUndo(){
  if(busy || !API || !cur || !cur.ply) return;
  sel=null;
  cur = await API.undo();
  renderBoard(); renderMoves();
  await advance();
}

/* ---- main loop ---- */
async function advance(){
  // show policy for whoever is to move; if Maia, let it reply
  if(cur.game_over){ finishUI(); return; }
  busy=true;
  let d = await API.policy(elo, true);
  cur = d; renderBoard(); renderPolicy(d.policy, d.wdl, d.activation_file, null);
  setStatus();
  if(d.maia_to_move){
    setStatus(`Maia (${cur.turn}) is thinking…`);
    await sleep(1500);
    const r = await API.maia_move(elo, temp);
    cur = r; renderBoard(); renderMoves();
    renderPolicy(r.maia_policy, r.maia_wdl, r.activation_file, r.maia_move && r.maia_move.uci);
    if(!r.game_over){
      const h = await API.policy(elo, true);
      cur = h; renderBoard(); renderPolicy(h.policy, h.wdl, h.activation_file, null);
    }
  }
  busy=false;
  setStatus();
  updateAttention(); updateResidual();
  if(cur.game_over) finishUI();
}

/* ---- board rendering ---- */
function parseFen(fen){
  const map={}; const rows=fen.split(' ')[0].split('/');
  for(let r=0;r<8;r++){ let file=0; for(const ch of rows[r]){
    if(/\d/.test(ch)) file+=+ch;
    else { map[FILES[file]+(8-r)]=ch; file++; }
  }}
  return map;
}
function sqName(row,col){
  return orient==='white' ? FILES[col]+(8-row) : FILES[7-col]+(row+1);
}
function legalTargets(from){
  if(!cur||!cur.legal_moves) return {};
  const t={};
  for(const m of cur.legal_moves) if(m.slice(0,2)===from){ t[m.slice(2,4)]=true; }
  return t;
}
function renderBoard(){
  const board=$('board'); board.innerHTML='';
  const pieces = cur ? parseFen(cur.fen) : {};
  const last = cur && cur.last_move ? [cur.last_move.slice(0,2),cur.last_move.slice(2,4)] : [];
  const targets = sel ? legalTargets(sel) : {};
  const hlSq = attQueryReal || null;
  for(let row=0;row<8;row++) for(let col=0;col<8;col++){
    const name=sqName(row,col);
    const fileIdx=FILES.indexOf(name[0]), rankNum=+name[1];
    const isLight=(fileIdx+rankNum)%2===0;
    const d=document.createElement('div');
    d.className='sq '+(isLight?'light':'dark');
    if(last.includes(name)) d.classList.add('lastmove');
    if(sel===name) d.classList.add('sel');
    if(name===hlSq) d.classList.add('attq');
    if(cur && cur.human_to_move) d.classList.add('playable');
    const pc=pieces[name];
    if(pc){
      const span=document.createElement('span');
      span.className='pc '+(pc===pc.toUpperCase()?'white':'black');
      span.textContent=GLYPH[pc.toLowerCase()];
      d.appendChild(span);
    }
    if(targets[name]){
      const dot=document.createElement('span'); dot.className='dot'; d.appendChild(dot);
      if(pc) d.classList.add('cap');
    }
    // edge coordinates
    if(col===0){const c=document.createElement('span');c.className='coord r';c.textContent=name[1];d.appendChild(c);}
    if(row===7){const c=document.createElement('span');c.className='coord f';c.textContent=name[0];d.appendChild(c);}
    d.onclick=()=>onSquare(name);
    board.appendChild(d);
  }
}

/* ---- click-to-move ---- */
function pieceColorAt(name){
  if(!cur) return null; const pc=parseFen(cur.fen)[name];
  if(!pc) return null; return pc===pc.toUpperCase()?'white':'black';
}
function canMove(color){
  if(!color || !cur) return false;
  return cur.human_color==='both' ? color===cur.turn : color===cur.human_color;
}
function realToCanon(name, turn){
  const file=FILES.indexOf(name[0]); let rank0=(+name[1])-1;
  if(turn==='black') rank0=7-rank0;   // apply the side-to-move board mirror
  return rank0*8+file;
}
async function onSquare(name){
  if(busy || !cur || !cur.human_to_move) return;
  if(sel===null){
    if(canMove(pieceColorAt(name)) && legalMovesFrom(name).length){ sel=name; renderBoard(); }
    return;
  }
  if(name===sel){ sel=null; renderBoard(); return; }
  if(canMove(pieceColorAt(name)) && legalMovesFrom(name).length){ sel=name; renderBoard(); return; }
  // attempt sel -> name
  const base=sel+name;
  const promos=cur.legal_moves.filter(m=>m.length>4 && m.slice(0,4)===base);
  if(promos.length){ askPromo(base, promos); return; }
  if(cur.legal_moves.includes(base)){ await doHuman(base); }
  else { sel=null; renderBoard(); }
}
function legalMovesFrom(from){ return cur.legal_moves.filter(m=>m.slice(0,2)===from); }

async function doHuman(uci){
  busy=true; const prev=sel; sel=null;
  const r = await API.human_move(uci);
  if(r.error){ busy=false; setStatus('⚠ '+r.error); sel=prev; renderBoard(); return; }
  cur=r; renderBoard(); renderMoves();
  busy=false;
  await advance();
}

/* promotion picker */
function askPromo(base, promos){
  pendingPromo=base;
  const box=$('promoglyphs'); box.innerHTML='';
  const order=['q','r','b','n'].filter(p=>promos.includes(base+p));
  const white=cur.turn==='white';
  for(const p of order){
    const b=document.createElement('button');
    b.className='pc '+(white?'white':'black');
    b.textContent=GLYPH[p];
    b.onclick=async()=>{ $('promo').style.display='none'; const u=pendingPromo+p; pendingPromo=null; await doHuman(u); };
    box.appendChild(b);
  }
  $('promo').style.display='flex';
}

/* ---- panels ---- */
function renderPolicy(pol, wdl, actfile, playedUci){
  const box=$('policy'); box.innerHTML='';
  $('poltitle').textContent = pol ? `Policy over ${pol.length} legal moves` : 'Policy over legal moves';
  if(pol && pol.length){
    pol.slice(0,MAXPOL).forEach((m,i)=>{
      const row=document.createElement('div');
      row.className='prow'+(i===0?' top':'')+(playedUci&&m.uci===playedUci?' played':'');
      // bar width = the move's actual probability mass (0–100%), so the track reads as a true slider
      row.innerHTML=`<span class="san">${m.san}</span>`+
        `<span class="barwrap"><span class="bar" style="width:${Math.max(1.5,(m.p*100)).toFixed(1)}%"></span></span>`+
        `<span class="pct">${(m.p*100).toFixed(1)}%</span>`;
      box.appendChild(row);
    });
    if(pol.length>MAXPOL){ box.insertAdjacentHTML("beforeend",
      `<div style="font-size:10px;color:var(--muted);margin-top:5px">+${pol.length-MAXPOL} more legal moves</div>`); }
  }
  if(wdl){
    const w=Math.round(wdl.win*100), d=Math.round(wdl.draw*100), l=Math.max(0,100-w-d);
    $('wdl').innerHTML=`<div class="w" style="width:${w}%">${w>8?w+'%':''}</div>`+
      `<div class="d" style="width:${d}%">${d>8?d+'%':''}</div>`+
      `<div class="l" style="width:${l}%">${l>8?l+'%':''}</div>`;
  }
  $('actfile').textContent = actfile ? '↳ saved '+actfile.split('/').slice(-1)[0] : '';
}
function renderMoves(){
  const h=cur && cur.san_history ? cur.san_history : [];
  let out=''; for(let i=0;i<h.length;i+=2){ out+=`${i/2+1}. ${h[i]||''} ${h[i+1]||''}  `; }
  $('moves').textContent = out.trim() || '—';
}
function setStatus(msg){
  if(msg){ $('status').innerHTML=msg; return; }
  if(!cur){ $('status').textContent=''; return; }
  if(cur.game_over){ $('status').innerHTML=`<b>Game over</b> · ${cur.result} (${cur.termination||''})`; return; }
  const who = cur.human_to_move ? 'Your move' : 'Maia to move';
  $('status').innerHTML = `<b>${who}</b>`+(cur.in_check?' · check':'')+` · move ${cur.move_number}`;
}
function finishUI(){ sel=null; renderBoard(); setStatus(); }

/* ---- live attention panel (real QKᵀ / GAB / softmax for the current board) ---- */
let attLayer=0, attHead=0, attQueryReal='d4', lastAtt=null;

function viridis(t){
  t=Math.max(0,Math.min(1,t));
  const s=[[68,1,84],[59,82,139],[33,144,141],[93,200,99],[253,231,37]];
  const x=t*(s.length-1), i=Math.min(Math.floor(x),s.length-2), f=x-i, a=s[i], b=s[i+1];
  const c=k=>Math.round(a[k]+(b[k]-a[k])*f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}
function sqName2(sq){ return FILES[sq%8]+(Math.floor(sq/8)+1); }

function buildAttBoards(){
  ['att_qk','att_gab','att_attn'].forEach(id=>{
    const el=$(id); if(!el || el.children.length) return;
    for(let idx=0;idx<64;idx++){
      const d=document.createElement('div'); d.className='attcell'; d.dataset.idx=idx;
      d.onclick=()=>{ const r=Math.floor(idx/8), c=idx%8; attQueryReal=sqName(r,c); renderAttention(); renderBoard(); };
      el.appendChild(d);
    }
  });
}
function paintAtt(id, mat){
  const el=$(id); if(!el || !mat || !cur) return;
  const row=mat[realToCanon(attQueryReal, cur.turn)]; if(!row) return;
  let lo=Infinity, hi=-Infinity;
  for(const v of row){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const span=(hi-lo)||1;
  for(const cell of el.children){
    const idx=+cell.dataset.idx, r=Math.floor(idx/8), c=idx%8;
    const name=sqName(r,c), canon=realToCanon(name, cur.turn);
    cell.style.background=viridis((row[canon]-lo)/span);
    cell.classList.toggle('q', name===attQueryReal);
  }
}
function renderAttention(){
  if(!lastAtt) return;
  paintAtt('att_qk', lastAtt.qk);
  paintAtt('att_gab', lastAtt.gab);
  paintAtt('att_attn', lastAtt.attn);
  $('attinfo').textContent =
    `block ${lastAtt.layer} · head ${lastAtt.head} · query ${attQueryReal} · elo ${elo}`;
}
async function updateAttention(){
  if(!API || !cur || cur.game_over) return;
  ensureAttUi(MODEL_INFO);
  try{
    const d = await API.attention(elo, attLayer, attHead);
    if(d && !d.error){
      if(d.num_heads && $('headChips') && $('headChips').children.length !== d.num_heads){
        attHead = Math.min(attHead, d.num_heads - 1);
        buildChips('headChips', d.num_heads, attHead, i=>{ attHead=i; updateAttention(); });
      }
      lastAtt=d; renderAttention();
    }
  }catch(e){ /* ignore transient bridge errors */ }
}
function buildChips(id, n, current, onpick){
  const el=$(id); if(!el) return;
  const count = Math.max(0, Number(n) || 0);
  if(!count) return;
  el.innerHTML='';
  for(let i=0;i<count;i++){
    const b=document.createElement('div');
    b.className='chip'+(i===current?' active':''); b.textContent=i; b.dataset.i=i;
    b.onclick=()=>{ el.querySelectorAll('.chip').forEach(c=>c.classList.toggle('active',+c.dataset.i===i)); onpick(i); };
    el.appendChild(b);
  }
}
function populateAttSelects(info){
  if(!info) return;
  buildChips('layerChips', info.num_blocks||8, attLayer, i=>{ attLayer=i; updateAttention(); });
  buildChips('headChips', info.num_heads||8, attHead, i=>{ attHead=i; updateAttention(); });
}
function ensureAttUi(info){
  if(!info) return;
  buildAttBoards();
  const lc=$('layerChips'), hc=$('headChips');
  if(lc && !lc.children.length) buildChips('layerChips', info.num_blocks||8, attLayer, i=>{ attLayer=i; updateAttention(); });
  if(hc && !hc.children.length) buildChips('headChips', info.num_heads||8, attHead, i=>{ attHead=i; updateAttention(); });
}
function initAttUi(info){
  populateAttSelects(info);
  buildAttBoards();
}

/* ---- residual-stream filmstrip (live, per position + ELO) ---- */
let residMetric='delta', lastRes=null;
function buildFilm(labels){
  const f=$('film'); f.innerHTML='';
  labels.forEach((lab,li)=>{
    const col=document.createElement('div'); col.className='filmcol';
    const mb=document.createElement('div'); mb.className='miniboard'; mb.dataset.li=li;
    for(let r=0;r<8;r++) for(let c=0;c<8;c++){ const d=document.createElement('div'); d.dataset.sq=(7-r)*8+c; mb.appendChild(d); }
    const t=document.createElement('div'); t.className='filmlbl'; t.textContent=lab;
    col.appendChild(mb); col.appendChild(t); f.appendChild(col);
  });
}
function renderResidual(){
  if(!lastRes) return;
  const mats=lastRes[residMetric];
  let lo=Infinity,hi=-Infinity;
  for(const row of mats) for(const v of row){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const span=(hi-lo)||1;
  const cols=$('film').children;
  for(let li=0; li<cols.length; li++){
    const cells=cols[li].querySelector('.miniboard').children, row=mats[li];
    for(const cell of cells){ const sq=+cell.dataset.sq; cell.style.background=viridis((row[sq]-lo)/span); }
  }
  const desc={delta:'‖x_L − x_(L-1)‖ per square — where each block edits the stream',
              cos_final:'cosine(x_L, x_final) per square — how early each square settles',
              norm:'‖x_L‖ per square (RMS-normed blocks look ~flat — expected)'};
  $('residinfo').textContent = desc[residMetric]+' · side-to-move frame · elo '+elo;
}
async function updateResidual(){
  if(!API || !cur || cur.game_over) return;
  if(!$('showresid').checked) return;
  try{
    const d=await API.residual(elo);
    if(d && !d.error){ if(!lastRes || !$('film').children.length) buildFilm(d.labels); lastRes=d; renderResidual(); }
  }catch(e){ /* ignore */ }
}
document.querySelectorAll('.rbtn').forEach(b=>{ b.onclick=()=>{
  document.querySelectorAll('.rbtn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); residMetric=b.dataset.m; renderResidual();
};});
const _rc=$('showresid');
if(_rc) _rc.onchange=e=>{ document.querySelector('.resid').classList.toggle('hidden', !e.target.checked); if(e.target.checked) updateResidual(); };
</script>
</body>
</html>
"""


# ============================================================================
# 4. LAUNCH
# ============================================================================
def main():
    try:
        import webview  # pywebview
    except ImportError:
        sys.exit("pywebview is not installed.  Run:  pip install -r requirements.txt")
    api = MaiaApi()
    webview.create_window(
        "Maia-3 · play & probe",
        html=INDEX_HTML,
        js_api=api,
        width=1400, height=920, min_size=(1240, 840),
        background_color="#0e1014",
    )
    webview.start()


if __name__ == "__main__":
    main()
