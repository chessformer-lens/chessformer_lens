"""
bridge.py — the pywebview JS API (game state + the JSON methods it exposes).

`MaiaApi` is what gets exposed to the browser as `window.pywebview.api`: every
method returns a plain JSON-able dict. It owns the chess.Board, drives new
game / move / undo / analyze, and forwards the interp queries (policy, attention,
residual) to a `MaiaEngine`. The model loads on a background thread so the window
opens instantly. This file is the app's glue — for notebook work use
interp_plot.py / interp_widget.py, or import `MaiaEngine` from engine.py
directly; none of them go through here.
"""
import os
import sys
import threading
from pathlib import Path

import chess

from engine import MaiaEngine

HERE = Path(__file__).resolve().parent
ACT_DIR = HERE / "activations"


def side_name(turn):
    return "white" if turn == chess.WHITE else "black"


class MaiaApi:
    """The JS-callable surface. Methods return plain JSON-able dicts.

    `ui.py` drives most of these. `gab_coeffs`, `gab_bias`, `qk_scores` and
    `compare_residual` are not wired to any control — they're kept as a stable
    surface for poking at the model from the webview console."""

    def __init__(self, alias=None):
        self.engine = None
        self.ready = False
        self.error = None
        # which model to load: explicit arg > env var > 5M default
        self.alias = alias or os.environ.get("MAIA3_ALIAS", "maia3-5m")
        self.target_name = self._display_name(self.alias)   # for the loading screen
        self.board = chess.Board()
        self.human = chess.WHITE
        self.human_both = False
        self.san_history = []
        self._lock = threading.Lock()
        # load the model off the UI thread so the window opens instantly. Bigger
        # models (23M/79M) just take longer here; the UI polls info() until ready.
        threading.Thread(target=self._load, daemon=True).start()

    @staticmethod
    def _display_name(alias):
        """Friendly name for `alias` without loading weights (for the loading
        screen). Falls back to the raw alias if it can't be resolved."""
        try:
            from maia3.model_registry import resolve_model_spec
            return resolve_model_spec(alias).display_name
        except Exception:
            return str(alias)

    def _load(self):
        import traceback
        try:
            print(f"[maia-app] loading {self.target_name} ({self.alias})…",
                  file=sys.stderr, flush=True)
            self.engine = MaiaEngine(
                alias=self.alias,
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
            # `target` is known immediately (before load) so the loading screen
            # can name the model; `alias` is the confirmed name once ready.
            "target": self.target_name,
            "alias": self.engine.spec.display_name if self.ready else None,
            "device": self.engine.device if self.ready else None,
            "checkpoint": self.engine.cfg.checkpoint_path if self.ready else None,
            "num_blocks": self.engine.cfg.num_blocks if self.ready else None,
            "num_heads": self.engine.cfg.num_heads if self.ready else None,
            "dim_vit": self.engine.cfg.dim_vit if self.ready else None,
            "gen_size": self.engine.cfg.gab_gen_size if self.ready else None,
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
        """Re-evaluate the current position at a given Elo (no move made).
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

    def analyze(self):
        """Switch to set-up / analyze mode (you move both sides; Maia never
        auto-moves) without resetting the board."""
        self.human_both = True
        return self._base()

    def resume(self, human_color="white"):
        """Leave set-up / analyze mode and play on from the CURRENT position:
        assign the side you play without touching the board. `new_game` is the
        one that resets, so switching the dropdown back to White/Black keeps
        the position you just set up."""
        self.human_both = False
        self.human = chess.BLACK if human_color == "black" else chess.WHITE
        return self._base()

    def set_fen(self, fen):
        """Load an arbitrary position from a FEN (enters analyze mode)."""
        try:
            board = chess.Board(fen)
        except Exception:
            return {**self._base(), "error": "invalid FEN"}
        self.board = board
        self.human_both = True
        self.san_history = []
        return self._base()

    def edit_square(self, frm, to=None):
        """Free position editing: move the piece on `frm` to `to` ignoring
        legality, or delete it if `to` is None. Stays in analyze mode."""
        try:
            f = chess.parse_square(frm)
        except Exception:
            return self._base()
        piece = self.board.piece_at(f)
        self.board.remove_piece_at(f)
        if to and piece is not None:
            try:
                self.board.set_piece_at(chess.parse_square(to), piece)
            except Exception:
                pass
        self.human_both = True
        self.san_history = []
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

    # ----- GAB / smolgen decomposition (see engine.py) ----------------------
    def gab_templates(self):
        """The static square-pair stencils shared by every layer & head — the
        model's whole geometric vocabulary. `gen_size` of them (64 on 3m/5m, 128
        on 23m/79m), read off the tensor. Position-independent, so the UI fetches
        this exactly once. Rounded to 4 decimals to keep the one-time payload
        small."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        try:
            t = self.engine.gab_templates()
        except RuntimeError as exc:
            return {"error": str(exc)}
        return {
            "gen_size": int(t.shape[0]),
            "templates": [[[round(float(v), 4) for v in row] for row in tmpl]
                          for tmpl in t.tolist()],
        }

    def gab_coeffs(self, elo=1500, layer=0):
        """The smolgen mixing coefficients (all heads) generated for the current
        position: coeffs[h][i] is how much of template i head h mixes into its
        GAB bias. (attention() also returns the selected head's row, so the UI
        normally doesn't need this extra call.)"""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                c = self.engine.gab_coeffs(b, self_elo=int(elo), layer=int(layer))
        except (RuntimeError, AssertionError) as exc:
            return {"error": str(exc)}
        return {"layer": int(layer), "num_heads": int(c.shape[0]),
                "gen_size": int(c.shape[1]), "coeffs": c.tolist()}

    def gab_bias(self, elo=1500, layer=0, head=0):
        """One head's generated 64×64 geometric attention bias for the current
        position (the matrix added to QKᵀ before the softmax)."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                g = self.engine.gab_bias(b, self_elo=int(elo),
                                         layer=int(layer), head=int(head))
        except RuntimeError as exc:
            return {"error": str(exc)}
        return {"layer": int(layer), "head": int(head), "gab": g.tolist()}

    def qk_scores(self, elo=1500, layer=0, head=0):
        """One head's raw content logits (scaled QKᵀ) for the current position —
        the semantic half of attention, before GAB is added."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        with self._lock:
            q = self.engine.qk_scores(b, self_elo=int(elo),
                                      layer=int(layer), head=int(head))
        return {"layer": int(layer), "head": int(head), "qk": q.tolist()}

    def residual(self, elo=1500):
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        with self._lock:
            return self.engine.residual_stream(b, self_elo=int(elo))

    def compare_residual(self, elo_a=1500, elo_b=1100):
        """Skill diff on internals: per-square ||x_A − x_B|| of the running
        residual stream at every readout point, plus each run's logit-lens
        move, for the current position (see engine.compare_residual). Not
        consumed by the UI — its skill comparison uses `compare_policy`."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        with self._lock:
            return self.engine.compare_residual(b, elo_a=int(elo_a), elo_b=int(elo_b))

    def move_lens(self, elo=1500, uci=None):
        """One move's depth curve (logit / prob / rank at every readout point)
        for the current position: the "snap" view."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        if not self._is_legal(uci):
            return {"error": f"not a legal move here: {uci}"}
        with self._lock:
            return self.engine.move_logit_lens(b, self_elo=int(elo), uci=uci)

    def ablate_grid(self, elo=1500, uci=None):
        """The carrier heatmap of one move: every head ablated in turn,
        delta = ablated − clean logit (the app-wide sign: negative = the head
        supports the move). Slow-ish: ~num_blocks·(num_heads+1) forward
        passes."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        if not self._is_legal(uci):
            return {"error": f"not a legal move here: {uci}"}
        try:
            with self._lock:
                return self.engine.ablate_grid(b, self_elo=int(elo), uci=uci)
        except Exception as exc:  # e.g. the head-write reconstruction assert
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _san(self, uci):
        try:
            return self.board.san(chess.Move.from_uci(uci))
        except Exception:
            return uci

    def _is_legal(self, uci):
        try:
            return uci and chess.Move.from_uci(uci) in self.board.legal_moves
        except ValueError:
            return False

    def compare_policy(self, elo_a=1500, elo_b=1100):
        """Evaluate the current position at two ratings (no move, no activation
        dump) for the skill-comparison view. Rows sorted by max probability."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        st = self._base()
        out = {**st, "elo_a": int(elo_a), "elo_b": int(elo_b)}
        if st["game_over"]:
            return {**out, "rows": [], "wdl_a": None, "wdl_b": None}
        b = self.board
        with self._lock:
            ra = self.engine.evaluate(b, self_elo=int(elo_a))
            rb = self.engine.evaluate(b, self_elo=int(elo_b))
        pa, pb = dict(ra["policy"]), dict(rb["policy"])
        rows = [{"uci": u, "san": self._san(u),
                 "p_a": pa.get(u, 0.0), "p_b": pb.get(u, 0.0)}
                for u in pa.keys() | pb.keys()]
        rows.sort(key=lambda r: max(r["p_a"], r["p_b"]), reverse=True)
        return {**out, "rows": rows, "wdl_a": ra["wdl"], "wdl_b": rb["wdl"]}

    def ablate(self, elo=1500, layer=0, head=0):
        """Remove one attention head's exact residual write (engine.ablate_head)
        and report how the policy + WDL move vs a clean forward pass. Rows are
        sorted by |Δp| so the moves the head matters for come first."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                base = self.engine.evaluate(b, self_elo=int(elo))
                abl = self.engine.ablate_head(b, self_elo=int(elo),
                                              layer=int(layer), head=int(head))
        except Exception as exc:  # e.g. the head-write reconstruction assert
            return {"error": f"{type(exc).__name__}: {exc}"}
        pa, pb = dict(base["policy"]), dict(abl["policy"])
        rows = [{"uci": u, "san": self._san(u),
                 "p": pa.get(u, 0.0), "p_abl": pb.get(u, 0.0)}
                for u in pa.keys() | pb.keys()]
        rows.sort(key=lambda r: abs(r["p"] - r["p_abl"]), reverse=True)
        return {"layer": int(layer), "head": int(head), "rows": rows,
                "wdl": base["wdl"], "wdl_abl": abl["wdl"]}
