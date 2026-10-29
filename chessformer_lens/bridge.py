"""
bridge.py — the pywebview JS API (game state + the JSON methods it exposes).

`LensApi` is what gets exposed to the browser as `window.pywebview.api`: every
method returns a plain JSON-able dict. It owns the chess.Board, drives new
game / move / undo / analyze, and forwards the interp queries (policy, attention,
residual) to the engine an alias names — any registered model, the same method
surface either way. There is no default model: launched without one, the page
shows a picker (`models` / `load_model`). The model loads on a background
thread so the window opens instantly. This file is the app's glue — for
notebook work use interp_plot.py / interp_widget.py, or import an engine from
engine.py directly; none of them go through here.
"""
import itertools
import os
import random
import sys
import threading
from pathlib import Path

import chess

from .core import available_models, sample_from_logits
from .engine import (LeelaEngine, NeuronScan, find_lc0, load_engine, load_positions,
                     ordinary_positions, resolve_engine)

# Cached activations are working data, not package content: they land under the
# directory the app is launched from, so an installed copy is never written to.
ACT_DIR = Path.cwd() / "activations"

# Dumping a snapshot on every evaluation is how you get a real game's
# activations into a notebook, but it is also a few hundred KB per move AND per
# drag of the rating slider — an evening of play is gigabytes. Off unless asked:
#     CHESSFORMER_SAVE_ACTIVATIONS=1 chessformer_lens
SAVE_ACTIVATIONS = os.environ.get(
    "CHESSFORMER_SAVE_ACTIVATIONS", "").strip().lower() in {"1", "true", "yes", "on"}

# `random_position` walks back out of a position offering fewer choices than
# this: a board where the side to move has one forced reply makes for a dead
# policy panel, which is the one panel the button exists to fill.
RANDOM_MIN_LEGAL = 6

# "Analyze many positions" scans a pool of positions on a background thread,
# one batch per hold of the engine lock so the other panels keep answering.
# A scan that would run past this many seconds pauses and asks first.
SCAN_PAUSE_SECONDS = 60

# set-up mode's piece cycle, in order: each step puts down the piece after the one on the square
PIECE_CYCLE = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def side_name(turn):
    return "white" if turn == chess.WHITE else "black"


def _log(msg):
    print(f"[chessformer_lens] {msg}", file=sys.stderr, flush=True)


class LensApi:
    """The JS-callable surface. Methods return plain JSON-able dicts.

    `ui.py` drives most of these. `gab_coeffs`, `gab_bias`, `qk_scores` and
    `compare_residual` are not wired to any control — they're kept as a stable
    surface for poking at the model from the webview console."""

    def __init__(self, alias=None, engine=None, simple=False):
        """`engine`: an engine already loaded (a notebook's), used as is.
        `simple`: open in simple play (the page hides the interpretability
        panels and never asks for them; the engine only plays and evaluates)."""
        self.engine = engine
        self.simple = bool(simple)
        self.ready = engine is not None
        self.error = None
        if engine is not None:
            self.alias, self.target_name = engine.spec.name, engine.spec.display_name
        else:
            # which engine to load: explicit arg > env var > none, in which case
            # the page asks (see `models` / `load_model`)
            self.alias = (alias or os.environ.get("CHESSFORMER_MODEL")
                          or os.environ.get("MAIA3_ALIAS") or None)
            self.target_name = self._display_name(self.alias) if self.alias else None
        self.board = chess.Board()
        self.human = chess.WHITE
        self.human_both = False
        self.san_history = []
        self._lock = threading.Lock()
        self._loading = False
        self._game_return = None       # the game to go back to while a scanned or PV position is shown
        self._search = None            # the last search: {root board, result} (see `search`)
        self._scan = None              # the last "many positions" scan (see `scan_start`)
        self._scan_mutex = threading.Lock()
        self._scan_ids = itertools.count(1)
        if self.alias and not self.ready:
            self._start_load()

    @staticmethod
    def _display_name(alias):
        """Friendly name for `alias` without loading weights (for the loading
        screen). Falls back to the raw alias if it can't be resolved."""
        try:
            return resolve_engine(alias)[1]
        except Exception:
            return str(alias)

    def _start_load(self):
        # load the model off the UI thread so the window opens instantly. Bigger
        # models just take longer here; the UI polls info() until ready.
        self._loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        import traceback
        try:
            _log(f"loading {self.target_name} ({self.alias})…")
            self.engine = load_engine(self.alias, activation_dir=str(ACT_DIR))
            self.ready = True
            _log(f"model ready on {self.engine.device}: {self.engine.cfg.checkpoint_path}")
        except Exception as exc:  # surface load errors to the UI instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            _log("MODEL LOAD FAILED:\n" + traceback.format_exc())
        finally:
            self._loading = False

    # ----- choosing a model -------------------------------------------------
    def models(self):
        """Every registered model, for the page's picker:
        {models: [{family, alias, display_name, aliases, available, note}]}."""
        try:
            return {"models": available_models()}
        except Exception as exc:
            return {"models": [], "error": f"{type(exc).__name__}: {exc}"}

    def load_model(self, alias):
        """Start loading `alias` (any registered alias, an lc0 weights path, …).
        Validated first, so an unknown name comes back as an error instead of a
        failed load. One model per session: restart the app to switch."""
        if self.ready or self._loading:
            return {**self.info(), "error": "a model is already loaded or loading — "
                                            "restart the app to switch models"}
        alias = str(alias or "").strip()
        try:
            self.target_name = resolve_engine(alias)[1]
        except Exception as exc:
            return {**self.info(), "error": f"{type(exc).__name__}: {exc}"}
        self.alias, self.error = alias, None
        self._start_load()
        return self.info()

    # ----- introspection ----------------------------------------------------
    def info(self):
        eng = self.engine if self.ready else None
        out = {
            "ready": self.ready,
            "error": self.error,
            # no model named yet: the page shows the picker
            "needs_model": self.alias is None and not self.ready,
            "loading": self._loading,
            # `target` is known immediately (before load) so the loading screen
            # can name the model; `alias` is the confirmed name once ready.
            "target": self.target_name,
            "alias": eng.spec.display_name if eng else None,
            "device": eng.device if eng else None,
            "checkpoint": eng.cfg.checkpoint_path if eng else None,
            "num_blocks": eng.cfg.num_blocks if eng else None,
            "num_heads": eng.cfg.num_heads if eng else None,
            "dim_vit": eng.cfg.dim_vit if eng else None,
            "gen_size": eng.cfg.gab_gen_size if eng else None,
            # capability flags the UI degrades on: no rating card without a
            # rating input (Leela), a moves-left readout only where there is one
            "conditioning": eng.has_conditioning if eng else None,
            "mlp_dim": int(eng.cfg.dim_vit * eng.cfg.mlp_ratio) if eng else None,
            "has_mlh": eng.has_mlh if eng else None,
            # the Search card: Leela with lc0 installed
            "search": bool(eng and isinstance(eng, LeelaEngine) and find_lc0()),
            "activation_dir": str(ACT_DIR) if SAVE_ACTIVATIONS else None,
            # start in simple play: just the game, no interpretability panels
            "simple": self.simple,
        }
        if eng:
            layout = eng.layout(chess.Board())
            out.update({
                "caps": eng.caps.to_json(),
                "has_gab": eng.has_generated_bias,
                "value_kind": eng.caps.value,
                "attn_terms": list(eng.caps.attn_terms),
                "is_board": layout.is_board,
                "frame": layout.frame,
                "n_tokens": len(layout),
            })
        return out

    # ----- state ------------------------------------------------------------
    def _base(self):
        b = self.board
        over = b.is_game_over()
        engine_to_move = (not over) and (not self.human_both) and (b.turn != self.human)
        root = b.root()
        return {
            "fen": b.fen(),
            # where san_history starts, so the moves list numbers from there
            "root_fullmove": root.fullmove_number,
            "root_turn": side_name(root.turn),
            "browsing": self._game_return is not None,
            "turn": side_name(b.turn),
            "human_color": "both" if self.human_both else side_name(self.human),
            "human_to_move": (not over) and (self.human_both or b.turn == self.human),
            "engine_to_move": engine_to_move,
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
        self._game_return = None
        self.board = chess.Board()
        self.human_both = (human_color == "both")
        self.human = chess.BLACK if human_color == "black" else chess.WHITE
        self.san_history = []
        return self._base()

    def layout(self):
        """What each token of the current position stands for (TokenLayout)."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        return self.engine.layout(self.board).to_json()

    # ----- model-backed -----------------------------------------------------
    def _policy_for_current(self, elo, save=True):
        b = self.board
        with self._lock:
            res = self.engine.evaluate(b, self_elo=int(elo))
            act_file = None
            if save and SAVE_ACTIVATIONS and not b.is_game_over():
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
        return pol, res, act_file

    def policy(self, elo=1500, save=True):
        """Re-evaluate the current position at a given Elo (no move made).
        This is the slider probe + per-position activation dump."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        st = self._base()
        if st["game_over"]:
            return {**st, "policy": [], "wdl": None, "mlh": None, "value": None,
                    "activation_file": None}
        pol, res, act = self._policy_for_current(elo, save=save)
        return {**st, "policy": pol, "wdl": res["wdl"], "mlh": res.get("mlh"),
                "value": res["value"].to_json(), "activation_file": act,
                "layout": self.engine.layout(self.board).to_json()}

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

    def engine_move(self, elo=1500, temperature=1.0, components=None, scale=0.0):
        """The engine plays the side to move (sampled at `temperature`; a
        win-probability model plays its argmax, as DeepMind's do). With
        `components` it plays the steered model (engine.steer at `scale`)."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over() or self.human_both:
            return self._base()
        if b.turn == self.human:
            return {**self._base(), "error": "not the engine's turn"}

        # policy + activations for the position the engine is about to move in
        pol, res, act = self._policy_for_current(elo, save=True)
        t = None if self.engine.caps.policy == "win_prob" else float(temperature)
        with self._lock:
            if components:
                steered = self.engine.steer(b, int(elo), list(components), float(scale))
                if t is None:
                    t = 0.0
                mv = self.engine._idx_to_move(b, sample_from_logits(steered["_logits"], t))
            else:
                mv, _ = self.engine.select_move(b, self_elo=int(elo), temperature=t)
        if mv is None and pol:
            mv = chess.Move.from_uci(pol[0]["uci"])
        played = None
        if mv is not None:
            played = {"uci": mv.uci(), "san": b.san(mv)}
            self.san_history.append(played["san"])
            b.push(mv)
        return {**self._base(),
                "engine_move": played, "engine_policy": pol, "engine_wdl": res["wdl"],
                "engine_mlh": res.get("mlh"), "engine_value": res["value"].to_json(),
                "activation_file": act}

    def undo(self):
        """Step back to the previous position where it is the human's move
        (pops the engine's reply and your move). Pops at least one ply."""
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
        """Switch to set-up / analyze mode (you move both sides; the engine
        never auto-moves) without resetting the board."""
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
        self._game_return = None
        return self._base()

    def random_position(self, elo=1500, min_plies=12, max_plies=60):
        """Land on a random position by playing the engine against itself, then
        enter set-up / analyze mode there.

        The walk samples from the model's own policy at `elo`, so you get a
        board that rating actually reaches — a uniformly-random legal walk
        wanders off the training distribution within a dozen plies, and the
        policy over such a position reads as noise. Being a real walk and not a
        FEN jump, the move list and ← Back work from here too.

        Returns the usual state dict plus `walk`: one frame per ply, which the
        UI replays on the board so you watch the game arrive.
        """
        board, sans, frames = self._self_play(
            int(elo), random.randint(int(min_plies), int(max_plies)))
        # a walk can end in mate, or on a board with one forced reply. Step back
        # to the last position that poses a real choice — re-rolling the whole
        # walk would cost another N forward passes for the same result.
        while board.move_stack and (board.is_game_over()
                                    or board.legal_moves.count() < RANDOM_MIN_LEGAL):
            board.pop()
            sans.pop()
            frames.pop()
        self.board = board
        self.san_history = sans
        self.human_both = True
        self._game_return = None
        return {**self._base(), "walk": frames}

    def _self_play(self, elo, plies):
        """One self-play walk from the start position, `plies` long (shorter if
        the game ends first). Falls back to random legal moves if the model is
        not loaded — the app gates its UI on `ready`, so that path is only
        reachable when this is driven straight from the console."""
        board, sans, frames = chess.Board(), [], []
        for _ in range(plies):
            if board.is_game_over():
                break
            mv = None
            if self.ready:
                with self._lock:
                    mv, _ = self.engine.select_move(
                        board, self_elo=elo, temperature=None)       # the model's own default
            if mv is None or mv not in board.legal_moves:
                mv = random.choice(list(board.legal_moves))
            san = board.san(mv)
            board.push(mv)
            sans.append(san)
            frames.append({"fen": board.fen(), "last_move": mv.uci(),
                           "san": san, "in_check": board.is_check()})
        return board, sans, frames

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
        self._game_return = None
        return self._base()

    def cycle_square(self, sq):
        """One step of set-up mode's piece cycle on `sq`: the piece after the one
        on it in PIECE_CYCLE. An empty square gets a white pawn; after the black
        king the square is empty again. Stays in analyze mode."""
        try:
            sq = chess.parse_square(sq)
        except Exception:
            return self._base()
        piece = self.board.piece_at(sq)
        if piece is None:
            next_idx = 0
        else:
            next_idx = PIECE_CYCLE.index(piece.symbol()) + 1
        if next_idx < len(PIECE_CYCLE):
            self.board.set_piece_at(sq, chess.Piece.from_symbol(PIECE_CYCLE[next_idx]))
        else:
            self.board.remove_piece_at(sq)
        self.human_both = True
        self.san_history = []
        self._game_return = None
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
        the semantic half of attention, before any bias is added."""
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
        """Skill diff on internals: per-token ||x_A − x_B|| of the running
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

    def neuron(self, elo=1500, layer=0, neuron=0):
        """One MLP unit's activation on every token (engine.neuron_activation)."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        if self.board.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                return self.engine.neuron_activation(self.board, self_elo=int(elo),
                                                     layer=int(layer), neuron=int(neuron))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def neurons_overview(self, elo=1500, k=12):
        """Every layer's most active MLP units (engine.neuron_overview)."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        if self.board.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                return self.engine.neuron_overview(self.board, self_elo=int(elo), k=int(k))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _versus(self, base, abl):
        """Rows of a clean pass vs an ablated one, sorted by |Δp|, plus both values."""
        pa, pb = dict(base["policy"]), dict(abl["policy"])
        rows = [{"uci": u, "san": self._san(u),
                 "p": pa.get(u, 0.0), "p_abl": pb.get(u, 0.0)}
                for u in pa.keys() | pb.keys()]
        rows.sort(key=lambda r: abs(r["p"] - r["p_abl"]), reverse=True)
        return {"rows": rows, "wdl": base["wdl"], "wdl_abl": abl["wdl"],
                "value": base["value"].to_json(), "value_abl": abl["value"].to_json()}

    def ablate_neuron(self, elo=1500, layer=0, neuron=0, scale=0.0):
        """Remove one MLP unit's exact write on every token (engine.ablate_neuron),
        or scale it (engine.steer_neuron), and report the policy + value against a clean pass — same
        shape as `ablate`."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                base = self.engine.evaluate(b, self_elo=int(elo))
                abl = (self.engine.ablate_neuron(b, self_elo=int(elo), layer=int(layer), neuron=int(neuron))
                       if not scale else self.engine.steer_neuron(b, int(elo), int(layer), int(neuron), float(scale)))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"layer": int(layer), "neuron": int(neuron), "scale": float(scale), **self._versus(base, abl)}

    def carrier_neurons(self, elo=1500, uci=None, top_k=12):
        """The carrier-neuron table of one move (engine.carrier_neurons): every
        MLP unit scored at once by attribution patching, each with its
        per-token footprint."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        if not uci or not self._is_legal(uci):
            return {"error": f"not a legal move here: {uci}"}
        try:
            with self._lock:
                return self.engine.carrier_neurons(b, self_elo=int(elo), move=uci, top_k=int(top_k))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ----- search (Leela, through lc0) -----------------------------------------
    def search(self, nodes=400):
        """lc0's search on the current position (`LeelaEngine.search`): every
        move's visits and Q beside the raw policy, the PV, and whether search
        overruled the network's top move. Kept for `search_pv`."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        if not self.info()["search"]:
            return {"error": "search needs a Leela model and lc0 installed"}
        if self.board.is_game_over():
            return {**self._base(), "error": "game over"}
        try:
            with self._lock:
                res = self.engine.search(self.board, int(nodes))
                raw_top = self.engine.evaluate_batch([self.board])[0]["policy"][0][0]
        except Exception as exc:
            return {**self._base(), "error": f"{type(exc).__name__}: {exc}"}
        self._search = {"root": self.board.copy(), "res": res, "raw_top": raw_top}
        return {**self._base(), "search": self._search_payload()}

    def _search_payload(self):
        sc = self._search
        root, res = sc["root"], sc["res"]
        pv, b = [], root.copy()
        for u in res["pv"]:
            mv = chess.Move.from_uci(u)
            num = f"{b.fullmove_number}." if b.turn == chess.WHITE else \
                (f"{b.fullmove_number}…" if not pv else "")
            pv.append({"uci": u, "san": b.san(mv), "num": num})
            b.push(mv)
        return {**res, "pv_moves": pv, "root_fen": root.fen(), "raw_top": sc["raw_top"],
                "overruled": res["bestmove"] != sc["raw_top"]}

    def search_pv(self, k=0):
        """Show the position `k` plies along the last search's PV (0 = the
        searched position) in every panel, history kept, in analyze mode.
        `go_back` returns to the game as it was."""
        if self._search is None:
            return {**self._base(), "error": "no search yet"}
        res = self._search["res"]
        k = max(0, min(int(k), len(res["pv"])))
        if self._game_return is None:
            self._game_return = (self.board.copy(), list(self.san_history),
                                 self.human_both, self.human)
        board = self._search["root"].copy()
        for u in res["pv"][:k]:
            board.push_uci(u)
        self.board, self.san_history, self.human_both = board, self._sans(board), True
        return {**self._base(), "pv_index": k, "search": self._search_payload()}

    # ----- many positions (engine.top_neurons & co) ---------------------------
    def go_back(self):
        """Leave a scanned or PV position and restore the game exactly as it was."""
        if self._game_return is not None:
            self.board, self.san_history, self.human_both, self.human = self._game_return
            self._game_return = None
        out = self._base()
        if self._search is not None:
            out["search"] = self._search_payload()
        return out

    def _game(self):
        """The game being played: while a scanned position is shown, the one
        to go back to."""
        return self._game_return[0] if self._game_return is not None else self.board

    def _pool(self, pool, fens=None):
        """(positions, label) for a pool name: the bundled ordinary positions,
        every position of the game (its history kept), or `fens`, a list of
        FEN strings the page read from a file."""
        if pool == "fens":
            lines = [l.strip() for l in (fens or []) if l.strip() and not l.lstrip().startswith("#")]
            if not lines:
                raise ValueError("no FENs in the file")
            for i, line in enumerate(lines, 1):
                try:
                    chess.Board(line)
                except ValueError:
                    raise ValueError(f"line {i} is not a FEN: {line[:60]}") from None
            boards = load_positions(lines)
            return boards, f"{len(boards):,} positions from the file"
        if pool == "game":
            game = self._game()
            root, boards = game.root(), []
            b = root.copy()
            boards.append(b.copy())
            for mv in game.move_stack:
                b.push(mv)
                boards.append(b.copy())
            return boards, f"this game ({len(boards)} positions)"
        boards = ordinary_positions()
        return boards, f"{len(boards):,} ordinary positions"

    def scan_start(self, pool="ordinary", elo=1500, fens=None):
        """Scan a pool for the neurons that fire most (`NeuronScan`), on a
        background thread a batch at a time: "ordinary" (the bundled
        positions), "game", or "fens" (`fens`: the lines of a file of FENs).
        The same pool and rating again returns the scan already made."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        elo = int(elo) if self.engine.has_conditioning else None
        game = self._game()
        key = (pool, elo, (game.root().fen(), tuple(m.uci() for m in game.move_stack))
               if pool == "game" else tuple(fens) if pool == "fens" else None)
        with self._scan_mutex:                  # one scan at a time, even with web.py's threads
            sc = self._scan
            if sc is not None and sc["key"] == key and sc["state"] in ("running", "paused", "done"):
                return self.scan_status()
            if sc is not None:
                self._halt(sc)
            try:
                boards, label = self._pool(pool, fens)
                batch = 8 if str(self.engine.device) == "cpu" else 32
                scan = NeuronScan(self.engine, boards, elo=elo or 0, batch_size=batch)
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
            sc = self._scan = {"id": next(self._scan_ids), "key": key, "pool": pool,
                               "label": label, "elo": elo, "scan": scan, "state": "running",
                               "seconds": 0.0, "stop": False, "long_ok": False, "error": None,
                               "go": threading.Event()}
            threading.Thread(target=self._scan_run, args=(sc,), daemon=True).start()
        return self.scan_status()

    @staticmethod
    def _halt(sc):
        """Stop a scan's thread, paused or not."""
        sc["stop"] = True
        sc["go"].set()

    def _scan_run(self, sc):
        import time
        scan, t0 = sc["scan"], time.monotonic()
        try:
            while not scan.finished and not sc["stop"]:
                with self._lock:
                    scan.step()
                if sc["stop"]:
                    break
                sc["seconds"] = time.monotonic() - t0
                rate = sc["seconds"] / max(1, scan.done)
                if not sc["long_ok"] and rate * scan.total > SCAN_PAUSE_SECONDS \
                        and not scan.finished:
                    sc["state"] = "paused"
                    sc["go"].wait()
                    sc["go"].clear()
                    if sc["stop"]:
                        break
                    sc["state"], t0 = "running", time.monotonic() - sc["seconds"]
                time.sleep(0)                 # let a waiting panel take the lock
            sc["state"] = "done" if scan.finished else "stopped"
        except Exception as exc:
            sc["state"], sc["error"] = "error", f"{type(exc).__name__}: {exc}"

    def scan_status(self):
        """Where the scan is: {state (none, running, paused, done, stopped,
        error), pool, label, elo, done, total, seconds, eta}. Takes no lock."""
        sc = self._scan
        if sc is None:
            return {"state": "none"}
        scan = sc["scan"]
        rate = sc["seconds"] / max(1, scan.done)
        return {"state": sc["state"], "id": sc["id"], "pool": sc["pool"], "label": sc["label"],
                "elo": sc["elo"],
                "done": scan.done, "total": scan.total, "seconds": sc["seconds"],
                "eta": rate * (scan.total - scan.done) if scan.done else None,
                "error": sc["error"]}

    def scan_continue(self):
        """Let a paused scan run to the end."""
        sc = self._scan
        if sc is not None and sc["state"] == "paused":
            sc["long_ok"] = True
            sc["go"].set()
        return self.scan_status()

    def scan_stop(self):
        """Stop the scan; what it scanned so far stays usable."""
        sc = self._scan
        if sc is not None and sc["state"] in ("running", "paused"):
            self._halt(sc)
        return self.scan_status()

    def scan_report(self, k=10):
        """Each layer's `k` strongest neurons over the positions scanned so far:
        {n, total, label, elo, layers, grid: [[{neuron, mean}] per layer]}."""
        sc = self._scan
        if sc is None or not sc["scan"].done:
            return {"error": "nothing scanned yet"}
        with self._lock:
            rep = sc["scan"].report(k=int(k), n_best=0)
        grid = [[{"neuron": r["neuron"], "mean": r["mean"]} for r in rep.top(L)]
                for L in rep.layers]
        return {"n": len(rep.positions), "total": sc["scan"].total, "label": sc["label"],
                "elo": sc["elo"], "layers": rep.layers, "grid": grid}

    def neuron_positions(self, layer, neuron, k=8, lo=None, hi=None):
        """One neuron over the scanned positions: its mean / sd / max peak and
        rank in its layer, where it fires on average (64 values, the model's
        frame), a histogram of its peaks, and `k` positions with the
        activation on every square: its strongest, or, with `lo`/`hi`, `k`
        spread evenly over those whose peak lies in [lo, hi] (a histogram bin)."""
        sc = self._scan
        if sc is None or not sc["scan"].done:
            return {"error": "nothing scanned yet"}
        import torch
        scan = sc["scan"]
        L, n = int(layer), int(neuron)
        if L not in scan.layers or not 0 <= n < scan.peaks.size(2):
            return {"error": f"no neuron L{L}·n{n}"}
        j = scan.layers.index(L)
        with self._lock:
            peaks = scan.peaks[:scan.done, j, n].clone()
            layer_means = scan.peaks[:scan.done, j].mean(0)
            squares = (scan._square_sum[j, :, n] / scan.done).tolist()
            ranked = peaks.argsort(descending=True).tolist()
            in_bin = None
            if lo is not None and hi is not None:
                ranked = [i for i in ranked if float(lo) <= float(peaks[i]) <= float(hi)]
                in_bin = len(ranked)
                pick = sorted({round(x * (len(ranked) - 1) / max(1, int(k) - 1))
                               for x in range(min(int(k), len(ranked)))})
                order = [ranked[x] for x in pick]
            else:
                order = ranked[:int(k)]
            boards = [scan.positions[i] for i in order]
            tiles = self.engine.top_positions(L, n, boards, k=None, elo=sc["elo"] or 0) \
                if boards else []
        hist = torch.histc(peaks, bins=24, min=float(peaks.min()), max=float(peaks.max()))
        out = []
        for t in tiles:
            i = order[t["index"]]
            out.append({"index": i, "fen": t["fen"], "turn": side_name(t["board"].turn),
                        "peak": t["peak"], "square": t["square"], "peak_canon": t["peak_canon"],
                        "act": t["act"]})
        return {"layer": L, "neuron": n, "n": scan.done, "label": sc["label"], "scan_id": sc["id"],
                "bin": None if in_bin is None else {"lo": float(lo), "hi": float(hi), "count": in_bin},
                "mean": float(peaks.mean()), "std": float(peaks.std(unbiased=False)),
                "max": float(peaks.max()), "min": float(peaks.min()),
                "rank": int((layer_means > layer_means[n]).sum()) + 1,
                "squares": squares, "hist": hist.tolist(), "tiles": out}

    def neuron_effects(self, layer, neuron, indices, scan_id=None):
        """What the neuron does on scanned positions `indices`
        (`eng.neuron_effects`): per position the top move with and
        without it, whether it changes, and the moves it pushes up and down
        (SAN, Δ probability)."""
        sc = self._scan
        if sc is None or (scan_id is not None and int(scan_id) != sc["id"]):
            return {"error": "that scan is gone"}
        boards = [sc["scan"].positions[int(i)] for i in indices]
        try:
            with self._lock:
                eff = self.engine.neuron_effects(int(layer), int(neuron), boards,
                                                 elo=sc["elo"] or 0)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        def san(b, u):
            return b.san(chess.Move.from_uci(u))
        out = []
        for i, b, e in zip(indices, boards, eff):
            out.append({"index": int(i), "top": san(b, e["top"]), "p_top": e["p_top"],
                        "top_ablated": san(b, e["top_ablated"]),
                        "p_top_ablated": e["p_top_ablated"], "changed": e["changed"],
                        "supports": [{"san": san(b, u), "dp": d} for u, d in e["supports"]],
                        "opposes": [{"san": san(b, u), "dp": d} for u, d in e["opposes"]]})
        return {"layer": int(layer), "neuron": int(neuron), "effects": out}

    def open_position(self, index, scan_id=None):
        """Show scanned position `index`, with its history, in every panel (in
        analyze mode). `scan_id` (from `neuron_positions`) makes sure it is
        that scan's position. `go_back` returns to the game."""
        sc = self._scan
        if sc is None or not 0 <= int(index) < sc["scan"].total \
                or (scan_id is not None and int(scan_id) != sc["id"]):
            return {**self._base(), "error": "that position belongs to an earlier scan"}
        if self._game_return is None:
            self._game_return = (self.board.copy(), list(self.san_history),
                               self.human_both, self.human)
        board = sc["scan"].positions[int(index)].copy()
        self.board, self.san_history, self.human_both = board, self._sans(board), True
        return {**self._base(), "browse_index": int(index)}

    @staticmethod
    def _sans(board):
        """The SAN of every move in `board`'s history, from its root."""
        replay, out = board.root(), []
        for mv in board.move_stack:
            out.append(replay.san(mv))
            replay.push(mv)
        return out

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
            return {**out, "rows": [], "wdl_a": None, "wdl_b": None,
                    "value_a": None, "value_b": None}
        b = self.board
        with self._lock:
            ra = self.engine.evaluate(b, self_elo=int(elo_a))
            rb = self.engine.evaluate(b, self_elo=int(elo_b))
        pa, pb = dict(ra["policy"]), dict(rb["policy"])
        rows = [{"uci": u, "san": self._san(u),
                 "p_a": pa.get(u, 0.0), "p_b": pb.get(u, 0.0)}
                for u in pa.keys() | pb.keys()]
        rows.sort(key=lambda r: max(r["p_a"], r["p_b"]), reverse=True)
        return {**out, "rows": rows, "wdl_a": ra["wdl"], "wdl_b": rb["wdl"],
                "value_a": ra["value"].to_json(), "value_b": rb["value"].to_json()}

    def ablate(self, elo=1500, layer=0, head=0, scale=0.0):
        """Remove one attention head's exact residual write (engine.ablate_head),
        or scale it by `scale` (engine.steer_head: 2 = doubled, -1 = flipped), and
        report how the policy + value move vs a clean forward pass. Rows are
        sorted by |Δp| so the moves the head matters for come first."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        try:
            with self._lock:
                base = self.engine.evaluate(b, self_elo=int(elo))
                abl = (self.engine.ablate_head(b, self_elo=int(elo), layer=int(layer), head=int(head))
                       if not scale else self.engine.steer_head(b, int(elo), int(layer), int(head), float(scale)))
        except Exception as exc:  # e.g. the head-write reconstruction assert
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"layer": int(layer), "head": int(head), "scale": float(scale), **self._versus(base, abl)}

    def steer(self, elo=1500, components=(), scale=0.0):
        """Several heads and neurons steered at once (engine.steer: each write
        recomputed live, exact across layers) against a clean pass — same shape as
        `ablate`. `components` is [{kind, layer, index}, ...]; `scale` 0 removes them."""
        if not self.ready:
            return {"error": self.error or "model still loading"}
        b = self.board
        if b.is_game_over():
            return {"error": "game over"}
        if not components:
            return {"error": "nothing picked"}
        arch = self.engine.arch
        for c in components:
            width = arch.n_heads if c.get("kind") == "head" else arch.d_mlp
            if c.get("kind") not in ("head", "neuron") or not 0 <= int(c["layer"]) < arch.n_layers \
                    or not 0 <= int(c["index"]) < width:
                return {"error": f"no such component on {arch.display_name}: {c}"}
        try:
            with self._lock:
                base = self.engine.evaluate(b, self_elo=int(elo))
                abl = self.engine.steer(b, int(elo), list(components), float(scale))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"n": len(components), "scale": float(scale), **self._versus(base, abl)}
