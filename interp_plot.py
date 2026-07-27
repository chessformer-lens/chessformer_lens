"""
The app's views as matplotlib figures — same layouts, same colours, same words.

`interp_widget.py` keeps the two panels whose whole point is sweeping a space
(attention over blocks/heads/queries, and the GAB template mixture). Everything
else in the app is a view you read: the position and its policy, the residual
film, one move's depth curve, the carrier-head grid. Those are figures, and this
module draws them from `engine.py` alone — no UI deps, so it imports cleanly in a
notebook, a script, or a paper pipeline:

    import chess
    from engine import MaiaEngine
    from interp_plot import plot_position, plot_residual_film

    eng = MaiaEngine()
    plot_residual_film(eng, chess.Board(), 1500)

  plot_position         the board, "Policy over N legal moves", and the
                        Win / Draw / Loss bar — with a second rating overlaid
                        when you pass `elo_b`
  plot_residual_film    "Residual stream across depth · this position": the
                        per-square ‖Δ‖ each structure writes, with the logit-lens
                        top move drawn on top of every readout point
  plot_move_microscope  "Move microscope": one move's logit through depth (or up
                        to 4 moves compared, or one move at two ratings), beside
                        the carrier heads that hold it up
  plot_carrier_heads    the carrier grid alone · Δlogit = ablated − clean
  plot_attention        "Live attention · this position": semantic QKᵀ, geometric
                        GAB, and the head's final attention, for one query square
  plot_gab_mixture      "How L·H's GAB is generated": the decomposition readout,
                        the generated mixing coefficients, and the template bank
  plot_gab_templates    the template vocabulary on its own
  plot_skill_diff       where two ratings diverge inside the stream, per square

`plot_attention` and `plot_gab_mixture` are the same two panels `interp_widget`
ships as live widgets, rendered as one slice: take the figure when you want a
single frame for a paper or a static export, take the widget when you want to
sweep blocks, heads and query squares.

Every plotter takes `figsize` and returns the Figure; the defaults are tuned to
the app's proportions, and the ones that wrap a row of boards size themselves
from the model's depth.

Frames follow the app: the attention boards and the position are drawn in real
board orientation (White at the bottom), while the residual film and the skill
diff are in the model's own canonical side-to-move frame — that is the frame the
engine returns those tensors in, and the app draws them the same way.
"""
from __future__ import annotations

import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from piece_art import draw_piece

# ---------------------------------------------------------------------------
# the app's palette (ui.py :root) — so a figure and the app read the same
# ---------------------------------------------------------------------------
# the full :root set, mirrored even where a figure has no use for it (PANEL2)
BG, PANEL, PANEL2, LINE = "#0e1014", "#161a21", "#1b2029", "#262c37"
TEXT, MUTED = "#e7eaf0", "#8b93a3"
ACCENT, ACCENT2 = "#6ea8fe", "#7bd88f"
SQ_LIGHT, SQ_DARK = "#c9d1dc", "#6b7686"
WIN, DRAW, LOSS = "#5fb878", "#6b7480", "#d9606a"
HL, QRING = "#f5d56b", "#ff5d6c"
CHART_BG = "#0f131a"
KIND_COL = {"emb": "#8a93a3", "attn": "#f0a35e", "mlp": "#6fb3ff", "enc": "#5ac878"}
MLCOLORS = ["#6ea8fe", "#7bd88f", "#f0a35e", "#c98bff"]   # compared moves
MLMAX = 4                                                 # how many at once

_MID, _BLUE, _ORANGE = (24, 28, 36), (64, 132, 234), (244, 134, 58)
POLBAR = LinearSegmentedColormap.from_list("polbar", [ACCENT, ACCENT2])
DIVMAP = LinearSegmentedColormap.from_list(
    "divmap", [np.array(_BLUE) / 255, np.array(_MID) / 255, np.array(_ORANGE) / 255])
MONO = "DejaVu Sans Mono"


def _divmap(v: float) -> tuple:
    """The app's divmap(): 0 = panel dark, negative = blue, positive = orange."""
    v = max(-1.0, min(1.0, float(v)))
    end = _BLUE if v < 0 else _ORANGE
    return tuple((m + (e - m) * abs(v)) / 255 for m, e in zip(_MID, end))


def _canon(square: int, turn: bool) -> int:
    """python-chess square -> the model's canonical index (side-to-move frame,
    square = rank*8 + file). The app's realToCanon(). Duplicated verbatim in
    interp_widget.py so each module imports standalone — keep the three in sync."""
    rank, file = chess.square_rank(square), chess.square_file(square)
    return (rank if turn == chess.WHITE else 7 - rank) * 8 + file


def _fig(figsize):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig


def _panel(ax, *, face=PANEL, radius=True):
    ax.set_facecolor(face)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    return ax


def _h2(ax, text, *, x=0.0, y=1.0, color=MUTED, size=9.5):
    """The app's card heading: uppercase and muted. (The `.6px` tracking in
    ui.py has no matplotlib equivalent, so it isn't reproduced.)"""
    ax.text(x, y, text.upper(), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=size, color=color, fontweight="600", family="sans-serif")


def _hint(ax, text, *, x=0.0, y=1.0, size=7.5, color=MUTED):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=size, color=color, family=MONO)


def _runs(fig, x, y, runs, *, va="top"):
    """Draw a single line made of differently styled runs — the app writes its
    titles and readouts that way ("Move microscope · **Qxf7#** f3f7 · elo 1500"),
    so each run has to be measured to know where the next one starts.
    `runs` is [(text, kwargs)]; returns the x the line ended at, in figure
    fraction. `_twidth` is the axes-fraction counterpart."""
    r = fig.canvas.get_renderer()
    for text, kw in runs:
        t = fig.text(x, y, text, va=va, **kw)
        x += t.get_window_extent(renderer=r).width / fig.bbox.width
    return x


def _inch_y(fig, inches_from_top):
    """Figure-fraction y that sits a fixed number of inches below the top edge —
    header blocks stay put instead of drifting with the figure's height."""
    return 1 - inches_from_top / fig.get_size_inches()[1]


# ---------------------------------------------------------------------------
# boards
# ---------------------------------------------------------------------------
# Pieces are the app's own artwork (pieces.py / python-chess "cburnett") — see
# piece_art.py. A chess font is not a substitute: the proportions and the
# interior linework are different, and it shows.


def _empty_board(ax):
    """Bare 8x8 axes: limits, equal aspect, no spines or ticks. Paints nothing —
    the caller fills every square (see `draw_board`)."""
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_board(ax, board: chess.Board, *, move=None, heat=None, cmap=None,
               query=None, pieces=True, coords=True, canonical=False,
               checker=True, piece_size=.88):
    """One board panel.

    `heat` is 64 values indexed by the model's canonical square (the frame every
    engine tensor uses); `canonical` also mirrors the *pieces* into that frame,
    which is what the residual film does. `move` draws the app's markers: the
    moving piece on the from-square and a green ring on the destination.

    Kwargs:
      cmap        any callable float -> RGBA: a matplotlib Colormap, or a plain
                  function like `_divmap`. Required whenever `heat` is given.
      query       canonical square to ring in red (the app's query marker).
      pieces      draw the pieces at all; False for a pure heat board.
      coords      file/rank labels around the edge.
      checker     overlay the app's faint checkerboard on top of `heat`, so the
                  squares stay readable under a flat colour.
      piece_size  fraction of a square the artwork occupies.
    """
    _empty_board(ax)
    turn = board.turn
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        canon = _canon(sq, turn)
        x, y = (f, canon // 8) if canonical else (f, r)
        if heat is not None:
            face = cmap(float(heat[canon]))
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, lw=0, fc=face, zorder=1))
            if checker:                       # the app's faint checkerboard on top
                over = (1, 1, 1, .05) if (f + r) % 2 else (0, 0, 0, .16)
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, lw=0, fc=over, zorder=2))
        else:
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, lw=0, zorder=1,
                                       fc=SQ_LIGHT if (f + r) % 2 else SQ_DARK))
        pc = board.piece_at(sq)
        if pieces and pc is not None:
            draw_piece(ax, pc.symbol(), x, y, size=piece_size)
    if move is not None:                       # (from_canon, to_canon, piece symbol)
        frm, to, sym = move
        if to is not None:
            ax.add_patch(plt.Rectangle((to % 8 - .5, to // 8 - .5), 1, 1, lw=2.2,
                                       ec=(90 / 255, 200 / 255, 120 / 255, .95),
                                       fc="none", zorder=5))
        if frm is not None and sym:
            draw_piece(ax, sym, frm % 8, frm // 8, size=piece_size, zorder=6)
    if query is not None:
        qx, qy = (query % 8, query // 8) if canonical else _real_xy(query, turn)
        ax.add_patch(plt.Rectangle((qx - .5, qy - .5), 1, 1, lw=2,
                                   ec=QRING, fc="none", zorder=6))
    if coords:
        for f in range(8):
            name = chess.FILE_NAMES[f]
            ax.text(f + .40, -.46, name, ha="right", va="bottom", fontsize=5.5,
                    color="#ffffff5c", family=MONO, zorder=7)
        for r in range(8):
            rank = r + 1 if (canonical or turn == chess.WHITE) else 8 - r
            ax.text(-.46, r + .40, str(rank), ha="left", va="top", fontsize=5.5,
                    color="#ffffff5c", family=MONO, zorder=7)


def _real_xy(canon: int, turn: bool):
    """canonical index -> (x, y) on a White-at-bottom board."""
    rank0, file = divmod(canon, 8)
    return file, (rank0 if turn == chess.WHITE else 7 - rank0)


def _canon_name(canon: int, turn: bool) -> str:
    x, y = _real_xy(canon, turn)
    return chess.FILE_NAMES[x] + str(y + 1)


# ---------------------------------------------------------------------------
# position · policy · win/draw/loss
# ---------------------------------------------------------------------------
def plot_position(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                  elo_b: int | None = None, played: str | None = None,
                  max_moves: int = 14, figsize=(11, 6.2)):
    """The app's left and right columns: the position, its policy, its eval.

    With `elo_b`, the policy becomes the app's compare mode —
    "Policy · A (blue) vs B (green)", one thin bar per rating and the signed
    delta, and a Win / Draw / Loss row per rating.

    Kwargs:
      oppo_elo    the opponent's rating; defaults to `elo` for both sides.
      elo_b       second rating -> compare mode (see above).
      played      a uci move to mark in the policy list as the one played.
      max_moves   how many policy rows to draw. The app's list scrolls and shows
                  essentially every legal move; a figure can't, so it truncates."""
    res = eng.evaluate(board, elo, oppo_elo)
    pol = res["policy"]
    if not pol:
        raise ValueError("no legal moves in this position")
    res_b = eng.evaluate(board, elo_b, oppo_elo) if elo_b is not None else None

    fig = _fig(figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.15], height_ratios=[1, .17],
                  hspace=.28, wspace=.16, left=.03, right=.97, top=.9, bottom=.06)
    ax_b = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    mv = chess.Move.from_uci(played) if played else chess.Move.from_uci(pol[0][0])
    draw_board(ax_b, board)
    for sq in (mv.from_square, mv.to_square):        # the app's played-move highlight
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ax_b.add_patch(plt.Rectangle((f - .5, r - .5), 1, 1, lw=0,
                                     fc=(*_hex(HL), .28), zorder=3))
    ax_b.set_title(f"{'White' if board.turn else 'Black'} to move · {elo} Elo"
                   + (f" vs {elo_b}" if elo_b is not None else ""),
                   fontsize=9.5, color=MUTED, pad=8, family=MONO)

    # ---- policy ------------------------------------------------------------
    ax_p.axis("off")
    if res_b is None:
        _h2(ax_p, f"Policy over {len(pol)} legal moves")
        rows = pol[:max_moves]
        _policy_rows(ax_p, board, [(u, p, None) for u, p in rows], played)
        extra = len(pol) - len(rows)
        if extra > 0:
            ax_p.text(0, -.02, f"+{extra} more legal moves", transform=ax_p.transAxes,
                      fontsize=7, color=MUTED, va="top")
    else:
        _h2(ax_p, f"Policy · {elo} (blue) vs {elo_b} (green)")
        pb = dict(res_b["policy"])
        rows = [(u, p, pb.get(u, 0.0)) for u, p in pol[:max_moves]]
        _policy_rows(ax_p, board, rows, played)
        extra = len(pol) - len(rows)
        if extra > 0:
            ax_p.text(0, -.02, f"+{extra} more legal moves · Δ = p({elo_b}) − p({elo})",
                      transform=ax_p.transAxes, fontsize=7, color=MUTED, va="top")

    # ---- win / draw / loss -------------------------------------------------
    ax_w.axis("off")
    _h2(ax_w, "Win / Draw / Loss · side to move", y=1.15)
    if res_b is None:
        _wdl_bar(ax_w, res["wdl"], y=.05, h=.62)
    else:
        _wdl_bar(ax_w, res["wdl"], y=.52, h=.36, tag=str(elo))
        _wdl_bar(ax_w, res_b["wdl"], y=.06, h=.36, tag=str(elo_b))
    return fig


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _policy_rows(ax, board, rows, played):
    """The app's .prow list: SAN · bar · percentage (or dual bars + Δ)."""
    n = len(rows)
    step = 1.0 / max(n, 1)
    size = float(np.clip(90 / max(n, 1), 6.0, 9.5))      # keep rows legible as n grows
    for i, (uci, pa, pb) in enumerate(rows):
        y = 1 - (i + .5) * step
        top = i == 0
        san = board.san(chess.Move.from_uci(uci))
        ax.text(.005, y, san, transform=ax.transAxes, va="center", fontsize=size,
                family=MONO, color=ACCENT2 if top else TEXT,
                fontweight="bold" if top else "normal")
        x0, w = .16, .68
        if pb is None:
            ax.add_patch(plt.Rectangle((x0, y - step * .30), w, step * .60,
                                       transform=ax.transAxes, fc=CHART_BG,
                                       ec=HL if (played and uci == played) else LINE,
                                       lw=1.4 if (played and uci == played) else .8,
                                       zorder=1))
            bw = max(.012, pa) * w
            ax.imshow(np.linspace(0, 1, 128).reshape(1, -1), cmap=POLBAR,
                      aspect="auto", zorder=2, transform=ax.transAxes,
                      extent=(x0, x0 + bw, y - step * .28, y + step * .28))
            ax.text(.995, y, f"{pa * 100:.1f}%", transform=ax.transAxes, va="center",
                    ha="right", fontsize=size - .5, family=MONO,
                    color=ACCENT2 if top else MUTED)
        else:
            for j, (p, col) in enumerate(((pa, ACCENT), (pb, ACCENT2))):
                yy = y + (step * .16 if j == 0 else -step * .16)
                ax.add_patch(plt.Rectangle((x0, yy - step * .09), max(.008, p) * w,
                                           step * .18, transform=ax.transAxes,
                                           fc=col, lw=0, zorder=2))
            d = (pb - pa) * 100
            ax.text(.995, y, f"{d:+.1f}", transform=ax.transAxes, va="center",
                    ha="right", fontsize=size - .5, family=MONO,
                    color=ACCENT2 if d >= 0 else LOSS)


def _wdl_bar(ax, wdl, *, y, h, tag=None):
    """The app's .wdl bar: win / draw / loss, labelled when the slice is wide."""
    x0 = 0.0
    if tag is not None:
        ax.text(0, y + h / 2, tag, transform=ax.transAxes, va="center", ha="left",
                fontsize=7, color=MUTED, family=MONO)
        x0 = .075
    w, d = round(wdl["win"] * 100), round(wdl["draw"] * 100)
    l = max(0, 100 - w - d)
    span = 1 - x0
    for val, col, ink in ((w, WIN, "#0d130d"), (d, DRAW, "#10141a"), (l, LOSS, "#0d130d")):
        width = span * val / 100
        ax.add_patch(plt.Rectangle((x0, y), width, h, transform=ax.transAxes,
                                   fc=col, lw=0, clip_on=False))
        if val > (12 if tag else 8):
            ax.text(x0 + width / 2, y + h / 2, f"{val}%", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=ink, family=MONO)
        x0 += width


# ---------------------------------------------------------------------------
# residual stream across depth  (the app's film strip)
# ---------------------------------------------------------------------------
def plot_residual_film(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                       per_row: int = 9, figsize=None):
    """"Residual stream across depth · this position" — per-square ‖Δ‖ each
    structure writes into the residual stream with the logit-lens top move on top.

    One mini board per readout point, in the app's order and colours: viridis heat
    (emb on its own scale, attn + MLP sharing one), the moving piece drawn on the
    lens move's from-square and a green ring on its destination. `enc` has no
    additive write, so it shows the lens only. Canonical side-to-move frame.

    Kwargs:
      oppo_elo    the opponent's rating; defaults to `elo` for both sides.
      per_row     boards per row before wrapping — the figure grows to fit."""
    res = eng.residual_stream(board, elo, oppo_elo)
    moves, delta = res["moves"], res["delta"]
    n = len(moves)
    rows = int(np.ceil(n / per_row))
    figsize = figsize or (1.35 * min(n, per_row) + .6, 1.85 * rows + 1.05)

    lo = hi = None
    elo_lo = elo_hi = None
    for c in delta:                                   # emb on its own scale
        vals = c["norm"]
        if c["kind"] == "emb":
            elo_lo = min(vals) if elo_lo is None else min(elo_lo, min(vals))
            elo_hi = max(vals) if elo_hi is None else max(elo_hi, max(vals))
        else:
            lo = min(vals) if lo is None else min(lo, min(vals))
            hi = max(vals) if hi is None else max(hi, max(vals))
    span, espan = (hi - lo) or 1, ((elo_hi - elo_lo) or 1) if elo_hi is not None else 1

    fig = _fig(figsize)
    gs = GridSpec(rows, per_row, figure=fig, hspace=.34, wspace=.10,
                  left=.02, right=.98, top=.845 if rows > 1 else .78, bottom=.03)
    viridis = plt.get_cmap("viridis")
    for i, mvd in enumerate(moves):
        ax = fig.add_subplot(gs[i // per_row, i % per_row])
        d = delta[i] if i < len(delta) else None
        kind = mvd["kind"]
        if d is not None:
            base = elo_lo if d["kind"] == "emb" else lo
            sp = espan if d["kind"] == "emb" else span
            heat = [(v - base) / sp for v in d["norm"]]
            draw_board(ax, board, heat=heat, cmap=viridis, pieces=False,
                       canonical=True, coords=False,
                       move=(mvd["from"], mvd["to"], mvd["piece"]))
        else:                                          # enc: lens only
            draw_board(ax, board, heat=[0] * 64, cmap=lambda v: CHART_BG,
                       pieces=False, canonical=True, coords=False,
                       move=(mvd["from"], mvd["to"], mvd["piece"]))
        col = KIND_COL.get(kind, MUTED)
        ax.plot([-.5, 7.5], [7.62, 7.62], color=col, lw=3, clip_on=False, zorder=8)
        label = mvd["label"] + (f" {mvd['san']}" if mvd["san"] else "")
        ax.text(3.5, -1.15, label, ha="center", va="top", fontsize=8,
                color=col, family=MONO)

    fig.text(.02, _inch_y(fig, .22), "Residual stream across depth · this position",
             fontsize=11, color=TEXT, fontweight="600", va="top")
    fig.text(.02, _inch_y(fig, .44), "per-square ‖Δ‖ each structure writes into the "
             "residual stream with the logit-lens top move on top",
             fontsize=8, color=MUTED, family=MONO, va="top")
    y = _inch_y(fig, .68)                               # the drawer's .residlegend
    x = .02
    for key, txt in (("emb", "emb (input)"), ("attn", "attn add"),
                     ("mlp", "MLP add"), ("enc", "enc (final norm)")):
        fig.patches.append(plt.Rectangle((x, y - .004), .007, .010,
                                         transform=fig.transFigure, fc=KIND_COL[key], lw=0))
        x = _runs(fig, x + .011, y, [(txt, dict(fontsize=7.5, color=MUTED, family=MONO))],
                  va="center") + .018
    return fig


# ---------------------------------------------------------------------------
# move microscope
# ---------------------------------------------------------------------------
def plot_move_microscope(eng, board: chess.Board, elo: int = 1500, ucis=None, *,
                         oppo_elo=None, elo_b: int | None = None,
                         carrier: bool = True, figsize=(13, 5.0)):
    """"Move microscope" — one move's logit through depth, and the heads that
    carry it.

    `ucis` is one move or up to four to overlay (the app's MLMAX); the first is
    the primary, whose carrier-head grid is drawn. `elo_b` overlays the same move
    at a second rating, which — as in the app — only applies to a single move.
    The dashed line marks the snap: the first point of the final rank-1 run.

    Kwargs:
      oppo_elo    the opponent's rating; defaults to `elo` for both sides.
      elo_b       overlay the primary move at a second rating (single move only).
      carrier     draw the carrier-head grid on the right. False leaves only the
                  depth curve — the whole right-hand half of the figure goes
                  away, and with it the ~num_blocks·(num_heads+1) ablation
                  passes it costs."""
    if ucis is None:
        ucis = [eng.evaluate(board, elo, oppo_elo)["policy"][0][0]]
    if isinstance(ucis, str):
        ucis = [ucis]
    ucis = list(ucis)[:MLMAX]
    series = [eng.move_logit_lens(board, elo, u, oppo_elo) for u in ucis]
    single = len(series) == 1
    data_b = (eng.move_logit_lens(board, elo_b, ucis[0], oppo_elo)
              if (single and elo_b is not None) else None)

    fig = _fig(figsize)
    if carrier:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, .42], wspace=.13,
                      left=.05, right=.97, top=.80, bottom=.22)   # room for .mlnote
        ax = fig.add_subplot(gs[0, 0])
        ax_g = fig.add_subplot(gs[0, 1])
    else:
        ax = fig.add_axes([.05, .13, .92, .67])
        ax_g = None

    steps = series[0]["steps"]
    x = np.arange(len(steps))
    _panel(ax, face=CHART_BG)
    ax.grid(axis="y", color=LINE, lw=.9)
    ax.set_axisbelow(True)

    for m, color in zip(series, MLCOLORS):
        y = [s["logit"] for s in m["steps"]]
        ax.plot(x, y, color=color, lw=2.2 if m is series[0] else 1.5, zorder=3,
                label=m["san"])
        if not single:
            ax.scatter(x, y, s=14, color=color, zorder=4)
    if single:                                   # writer-coloured dots
        for i, s in enumerate(steps):
            ax.scatter([i], [s["logit"]], s=26, zorder=4,
                       color=KIND_COL.get(s["kind"], "#fff"))
    if data_b is not None:
        ax.plot(x, [s["logit"] for s in data_b["steps"]], color=ACCENT2, lw=1.4, zorder=2)
        ax.scatter(x, [s["logit"] for s in data_b["steps"]], s=10, color=ACCENT2,
                   alpha=.85, zorder=3)
        ax.text(.995, 1.03, f"━ {elo}   ━ {elo_b}", transform=ax.transAxes, ha="right",
                fontsize=8, family=MONO, color=ACCENT2)

    if single:                                   # the snap
        ranks = [s["rank"] for s in steps]
        snap = -1
        if ranks[-1] == 1:
            snap = len(ranks) - 1
            while snap > 0 and ranks[snap - 1] == 1:
                snap -= 1
        if snap > 0:
            ax.axvline(snap, color=QRING, ls=(0, (4, 3)), lw=1.2, zorder=2)
            ax.text(snap + .15, ax.get_ylim()[1], f"top from {steps[snap]['label']}",
                    color=QRING, fontsize=8, family=MONO, va="top")

    ax.set_xticks(x, [s["label"].replace(" mlp", "") if s["kind"] == "mlp" else
                      (s["label"] if s["kind"] in ("emb", "enc") else "")
                      for s in steps], fontsize=8, family=MONO)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for lab in ax.get_yticklabels():
        lab.set_family(MONO)
    ax.set_xlim(-.4, len(steps) - .6)
    ax.set_ylabel("logit", color=MUTED, fontsize=8, family=MONO)
    if not single:
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=len(series))
        for t, color in zip(leg.get_texts(), MLCOLORS):   # the app's .mlchip colours
            t.set_family(MONO)
            t.set_color(color)

    pm = series[0]
    y = _inch_y(fig, .26)
    _runs(fig, .05, y, [
        ("Move microscope · ", dict(fontsize=11.5, color=TEXT, fontweight="600")),
        (pm["san"], dict(fontsize=11.5, color=ACCENT2, fontweight="600")),
        (f"  {pm['uci']} · elo {elo}" + (f" vs {elo_b}" if data_b is not None else ""),
         dict(fontsize=8.5, color=MUTED, family=MONO)),
    ])
    fig.text(.05, _inch_y(fig, .52),
             "click policy move(s) to see its logits through depth, and the heads "
             "that carry it" if len(series) > 1 else
             f"logit at each of the {len(steps)} readout points · "
             f"rank 1 of {pm['n_legal']} legal moves = currently the top move",
             fontsize=8, color=MUTED, family=MONO, va="top")

    if ax_g is not None:
        _carrier_grid(ax_g, eng, board, elo, pm["uci"], oppo_elo)
    return fig


def plot_carrier_heads(eng, board: chess.Board, elo: int = 1500, uci: str | None = None,
                       *, oppo_elo=None, figsize=(6.4, 5.6)):
    """"carrier heads · Δlogit = ablated − clean" on its own.

    `uci` defaults to the model's own move. Costs ~num_blocks·(num_heads+1)
    forward passes (72 on the 5M), so seconds, not ms."""
    if uci is None:
        uci = eng.evaluate(board, elo, oppo_elo)["policy"][0][0]
    fig = _fig(figsize)
    ax = fig.add_axes([.09, .16, .86, .74])
    _carrier_grid(ax, eng, board, elo, uci, oppo_elo)
    return fig


def _carrier_grid(ax, eng, board, elo, uci, oppo_elo):
    """The app's ablation grid: num_heads columns x num_blocks rows, coloured by
    Δlogit = ablated − clean, with the final block excluded from the scale."""
    g = eng.ablate_grid(board, elo, uci, oppo_elo)
    d = np.array(g["deltas"])
    nb, nh = d.shape
    no_carrier = nb - 1                       # writes straight to the logits

    scored = d[:no_carrier]
    m = float(np.abs(scored).max()) or 1e-9
    sL, sH = np.unravel_index(int(np.argmax(np.abs(scored))), scored.shape)

    ax.set_xlim(0, nh)
    ax.set_ylim(nb, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for L in range(nb):
        for h in range(nh):
            if L == no_carrier:               # dimmed + striped, out of attribution
                ax.add_patch(plt.Rectangle((h, L), 1, 1, fc="#151a22", ec="#2b3444",
                                           lw=0, hatch="////", alpha=.5))
                continue
            ax.add_patch(plt.Rectangle((h, L), 1, 1, ec=PANEL, lw=.6,
                                       fc=_divmap(d[L, h] / m)))
            if (L, h) == (sL, sH):
                ax.add_patch(plt.Rectangle((h, L), 1, 1, fc="none", ec=QRING, lw=2, zorder=3))
    for h in range(nh):
        ax.text(h + .5, -.15, f"h{h}", ha="center", va="bottom", fontsize=7,
                color=MUTED, family=MONO)
    for L in range(nb):
        ax.text(-.15, L + .5, f"b{L}", ha="right", va="center", fontsize=7,
                color=MUTED, family=MONO)
    ax.text(0, -.85, "carrier heads · Δlogit = ablated − clean", fontsize=8.5,
            color=MUTED, family=MONO, fontweight="600")
    v = d[sL, sH]
    ax.text(0, nb + .55,                      # the drawer's .mlnote, wrapped to the grid
            f"base logit {g['base_logit']:.2f} · strongest b{sL}·h{sH} {v:+.2f}\n"
            f"blue = ablating the head drops {g['san']}'s logit (carrier)\n"
            f"orange = raises it (suppressor)\n"
            f"b{no_carrier} excluded (writes straight to the logits)",
            fontsize=6.8, color=MUTED, family=MONO, va="top", linespacing=1.7)
    return g


# ---------------------------------------------------------------------------
# live attention
# ---------------------------------------------------------------------------
def plot_attention(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                   layer: int = 0, head: int = 0, query: str | None = None,
                   figsize=(11.5, 4.6)):
    """"Live attention · this position" — the app's three boards for one head:
    semantic attention (QKᵀ), geometric attention (GAB), and the head's final
    attention matrix. `query` is a square name; it defaults to the from-square of
    the model's move. Real board orientation (White at the bottom).

    Kwargs:
      oppo_elo     the opponent's rating; defaults to `elo` for both sides.
      layer, head  which head to show — this is one slice of the space
                   `interp_widget.attention_widget` lets you sweep.
      query        square name ('e4') whose attention row is drawn."""
    att = eng.attention(board, elo, oppo_elo, layer=layer, head=head)
    if query is None:
        best = eng.evaluate(board, elo, oppo_elo)["policy"][0][0]
        q = _canon(chess.Move.from_uci(best).from_square, board.turn)
    else:
        q = _canon(chess.parse_square(query), board.turn)

    qk, gab, attn = (np.array(att[k])[q] for k in ("qk", "gab", "attn"))

    fig = _fig(figsize)
    gs = GridSpec(1, 3, figure=fig, wspace=.08, left=.03, right=.97, top=.70, bottom=.14)
    viridis = plt.get_cmap("viridis")
    labels = ["semantic attention (QKᵀ)", "geometric attention (GAB)",
              "final head attention matrix (scaled softmax(QKᵀ + GAB))"]
    for i, (row, lab) in enumerate(zip((qk, gab, attn), labels)):
        ax = fig.add_subplot(gs[0, i])
        if i < 2:
            mx = float(np.abs(row).max()) or 1
            draw_board(ax, board, heat=row / mx, cmap=_divmap, pieces=False,
                       query=q, coords=True)
        else:
            mx = float(row.max()) or 1
            draw_board(ax, board, heat=row / mx, cmap=viridis, pieces=False,
                       query=q, coords=True)
        ax.set_title(lab, fontsize=8, color=MUTED, family=MONO, fontweight="600", pad=6)

    fig.text(.03, _inch_y(fig, .26), "Live attention · this position", fontsize=11.5,
             color=TEXT, fontweight="600", va="top")
    fig.text(.03, _inch_y(fig, .52), f"Layer {layer} · Head {head} · query "
             f"{_canon_name(q, board.turn)} · elo {elo}",
             fontsize=8.5, color=MUTED, family=MONO, va="top")
    # Two legends, each under the boards it describes: diverging for the
    # pre-softmax logits, 0 -> max for the head's actual attention. The app ships
    # only the viridis one — a static figure has no hover readout, so the
    # diverging scale has to be spelled out here.
    span = (.97 - .03) / 3
    _legbar(fig, .03 + span * .55, .055, span * .9, DIVMAP, "−max", "+max")
    _legbar(fig, .03 + span * 2.08, .055, span * .82, plt.get_cmap("viridis"), "0", "max")
    fig.text(.03 + span * 2.08, .028, "query's 64 weights (one per key square) sum to 1",
             fontsize=7.5, color=MUTED, family=MONO, va="top")
    return fig


def _legbar(fig, x, y, w, cmap, left, right, h=.018):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.text(-.04, .5, left, transform=ax.transAxes, ha="right", va="center",
            fontsize=7, color=MUTED, family=MONO)
    ax.text(1.04, .5, right, transform=ax.transAxes, ha="left", va="center",
            fontsize=7, color=MUTED, family=MONO)


# ---------------------------------------------------------------------------
# the GAB generator
# ---------------------------------------------------------------------------
def plot_gab_mixture(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                     layer: int = 0, head: int = 0, query: str | None = None,
                     target: str | None = None, top: int = 4, figsize=(12, 7.4)):
    """"How L·H's GAB is generated" — the app's drawer as a figure: the pair
    decomposition readout, the generated mixing coefficients, and the template
    vocabulary with each template's live coefficient.

    Kwargs:
      oppo_elo     the opponent's rating; defaults to `elo` for both sides.
      layer, head  which head's mixture to decompose — one slice of the space
                   `interp_widget.gab_widget` lets you sweep.
      query        square name ('e4') for the query half of the pair; defaults
                   to the from-square of the model's move.
      target       square name for the key half. Defaults to the key with the
                   largest |bias| on that query's row — so this chooses which
                   pair the readout sentence is about.
      top          how many templates the readout names, largest |contribution|
                   first."""
    T = eng.gab_templates().numpy()
    coeffs = eng.gab_coeffs(board, elo, oppo_elo, layer=layer).numpy()[head]
    gab = eng.gab_bias(board, elo, oppo_elo, layer=layer, head=head).numpy()

    if query is None:
        best = eng.evaluate(board, elo, oppo_elo)["policy"][0][0]
        q = _canon(chess.Move.from_uci(best).from_square, board.turn)
    else:
        q = _canon(chess.parse_square(query), board.turn)
    k = (_canon(chess.parse_square(target), board.turn) if target
         else int(np.argmax(np.abs(gab[q]))))

    n = len(coeffs)
    terms = sorted(((i, coeffs[i] * T[i, q, k]) for i in range(n)),
                   key=lambda t: -abs(t[1]))[:top]
    total = float(gab[q, k])
    rest = total - sum(v for _, v in terms)

    fig = _fig(figsize)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[.20, .22, .07, 1], hspace=.34,
                  left=.04, right=.97, top=.86, bottom=.03)

    # ---- readout -----------------------------------------------------------
    ax_r = fig.add_subplot(gs[0]); ax_r.axis("off")
    _panel(ax_r, face=CHART_BG)
    ax_r.add_patch(plt.Rectangle((0, .1), 1, .8, transform=ax_r.transAxes,
                                 fc=CHART_BG, ec=LINE, lw=.8, clip_on=False))
    parts = [(f"GAB {_canon_name(q, board.turn)}→{_canon_name(k, board.turn)} = ", MUTED),
             (f"{total:+.2f}", _ORANGE if total >= 0 else _BLUE), ("  =  ", MUTED)]
    x = .015
    for txt, col in parts:
        c = col if isinstance(col, str) else tuple(v / 255 for v in col)
        t = ax_r.text(x, .5, txt, transform=ax_r.transAxes, va="center", fontsize=9,
                      family=MONO, color=c, fontweight="bold" if txt[0] in "+-−" else "normal")
        x += _twidth(fig, t)
    for i, v in terms:
        t = ax_r.text(x, .5, f"{v:+.2f}", transform=ax_r.transAxes, va="center",
                      fontsize=9, family=MONO,
                      color=tuple(c / 255 for c in (_ORANGE if v >= 0 else _BLUE)))
        x += _twidth(fig, t)
        t = ax_r.text(x, .5, f"·#{i} ", transform=ax_r.transAxes, va="center",
                      fontsize=9, family=MONO, color=ACCENT)
        x += _twidth(fig, t)
    ax_r.text(x, .5, f" {rest:+.2f} rest", transform=ax_r.transAxes, va="center",
              fontsize=9, family=MONO, color=MUTED, alpha=.6)

    # ---- coefficient strip -------------------------------------------------
    ax_c = fig.add_subplot(gs[1])
    _panel(ax_c, face=CHART_BG)
    m = float(np.abs(coeffs).max()) or 1
    ax_c.bar(np.arange(n), coeffs, width=.62,
             color=["#f0a35e" if v >= 0 else "#6fb3ff" for v in coeffs])
    ax_c.set_xlim(-.8, n - .2)
    ax_c.set_ylim(-m * 1.12, m * 1.12)
    ax_c.set_yticks([])
    ax_c.set_xticks(range(0, n, 8))
    ax_c.tick_params(colors=MUTED, labelsize=7, length=0)
    for lab in ax_c.get_xticklabels():
        lab.set_family(MONO)
    ax_c.axhline(0, color=LINE, lw=.8)
    _hint(ax_c, f"generated mixing coefficients · template #0–{n - 1}", y=1.06)

    # ---- template gallery --------------------------------------------------
    ax_lab = fig.add_subplot(gs[2]); ax_lab.axis("off")
    _hint(ax_lab, f"template vocabulary · the {n} static stencils "
                  f"(row = query sq, col = key sq)", y=.1, size=8)
    per_row = 16
    rows = int(np.ceil(n / per_row))
    inner = GridSpecFromSubplotSpec(rows, per_row, subplot_spec=gs[3],
                                    hspace=.55, wspace=.10)
    for i in range(n):
        ax_t = fig.add_subplot(inner[i // per_row, i % per_row])
        s = float(np.abs(T[i]).max()) or 1
        ax_t.imshow(T[i] / s, cmap=DIVMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax_t.set_xticks([]); ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_color(ACCENT if i in [t[0] for t in terms] else LINE)
            sp.set_linewidth(1.6 if i in [t[0] for t in terms] else .6)
        ax_t.set_xlabel(f"#{i} {coeffs[i]:+.2f}", fontsize=5.6, family=MONO, labelpad=2,
                        color="#f0a35e" if coeffs[i] >= 0 else "#6fb3ff")
    _runs(fig, .04, _inch_y(fig, .28), [
        ("How ", dict(fontsize=11.5, color=TEXT, fontweight="600")),
        (f"L{layer}·H{head}", dict(fontsize=11.5, color=ACCENT2, fontweight="600")),
        ("'s GAB is generated", dict(fontsize=11.5, color=TEXT, fontweight="600")),
    ])
    fig.text(.04, _inch_y(fig, .54), f"a generator reads this position and emits, per "
             f"head, a bias of the {n} coefficients over a static bank of {n} 64×64 "
             f"square-pair templates shared by every layer", fontsize=8, color=MUTED,
             family=MONO, va="top")
    return fig


def _twidth(fig, t):
    """Width of a drawn text in axes fraction — see `_runs`, which does the same
    job in figure fraction. Forces a draw, so don't call it in a tight loop."""
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.transformed(t.axes.transAxes.inverted()).width


def plot_gab_templates(eng, *, per_row: int = 16, figsize=None):
    """"template vocabulary · the N static stencils (row = query sq, col = key sq)"
    — the whole shared bank, position-independent."""
    T = eng.gab_templates().numpy()
    n = len(T)
    rows = int(np.ceil(n / per_row))
    fig = _fig(figsize or (per_row * .78, rows * .86 + .9))
    gs = GridSpec(rows, per_row, figure=fig, hspace=.5, wspace=.1,
                  left=.02, right=.98, top=.86, bottom=.03)
    for i in range(n):
        ax = fig.add_subplot(gs[i // per_row, i % per_row])
        s = float(np.abs(T[i]).max()) or 1
        ax.imshow(T[i] / s, cmap=DIVMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE); sp.set_linewidth(.6)
        ax.set_xlabel(f"#{i}", fontsize=6, family=MONO, color=MUTED, labelpad=2)
    fig.text(.02, .965, "template vocabulary", fontsize=11.5, color=TEXT,
             fontweight="600", va="top")
    fig.text(.02, .925, f"the {n} static stencils (row = query sq, col = key sq) — "
             f"shared by every layer", fontsize=8, color=MUTED, family=MONO, va="top")
    return fig


# ---------------------------------------------------------------------------
# skill diff on internals
# ---------------------------------------------------------------------------
def plot_skill_diff(eng, board: chess.Board, elo_a: int = 1500, elo_b: int = 1100, *,
                    per_row: int = 9, figsize=None):
    """Skill diff on INTERNALS: per-square ‖x_A − x_B‖ at every readout point,
    with each rating's logit-lens move — where on the board, and at what depth,
    the two skill levels diverge. Canonical side-to-move frame.

    Both runs set the opponent's rating equal to their own, so each board is the
    diff between two whole skill settings (see engine.compare_residual).
    `per_row` sets how many boards before wrapping."""
    d = eng.compare_residual(board, elo_a, elo_b)
    steps = d["steps"]
    n = len(steps)
    rows = int(np.ceil(n / per_row))
    figsize = figsize or (1.35 * min(n, per_row) + .6, 1.95 * rows + 1.05)

    hi = max(max(s["norm"]) for s in steps) or 1
    fig = _fig(figsize)
    gs = GridSpec(rows, per_row, figure=fig, hspace=.42, wspace=.10,
                  left=.02, right=.98, top=.845 if rows > 1 else .78, bottom=.03)
    magma = plt.get_cmap("magma")
    for i, s in enumerate(steps):
        ax = fig.add_subplot(gs[i // per_row, i % per_row])
        draw_board(ax, board, heat=[v / hi for v in s["norm"]], cmap=magma,
                   pieces=False, canonical=True, coords=False)
        col = KIND_COL.get(s["kind"], MUTED)
        ax.plot([-.5, 7.5], [7.62, 7.62], color=col, lw=3, clip_on=False, zorder=8)
        ax.text(3.5, -1.05, s["label"], ha="center", va="top", fontsize=8,
                color=col, family=MONO)
        agree = s["same"]
        ax.text(3.5, -2.15,
                (s["move_a"]["san"] or "—") if agree else
                f"{s['move_a']['san'] or '—'} / {s['move_b']['san'] or '—'}",
                ha="center", va="top", fontsize=7.5, family=MONO,
                color=MUTED if agree else QRING)
    fig.text(.02, .965, f"Where {elo_a} and {elo_b} diverge inside the stream",
             fontsize=11.5, color=TEXT, fontweight="600", va="top")
    fig.text(.02, .928, f"per-square ‖x_{elo_a} − x_{elo_b}‖ at every readout point · "
             f"under each board the logit-lens move of both runs "
             f"(red = they disagree)", fontsize=8, color=MUTED, family=MONO, va="top")
    fig.text(.02, .898, "Elo enters as an embedding on every square token, so at emb the "
             "diff is one constant skill vector on all 64 squares (flat heat) — the "
             "structure is how depth localizes it",
             fontsize=7.5, color=MUTED, family=MONO, va="top", alpha=.75)
    return fig
