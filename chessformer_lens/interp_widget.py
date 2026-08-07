"""
The app's two interactive panels, each as a self-contained notebook cell.

`interp_plot.py` covers everything in the app you read once — the position and
its policy, the residual film, a move's depth curve, the carrier grid. These
two are different: they're spaces you sweep (num_blocks × num_heads × 64 query
squares of attention; a bias built from a bank of static templates). So they
keep the app's own interaction, CSS and wording, rewired from "fetch from
window.pywebview.api" to "read the injected data":

    from chessformer_lens.interp_widget import attention_widget, gab_widget

    attention_widget(eng, board, elo=1500, layer=5, head=3)   # Live attention
    gab_widget(eng, board, elo=1500, layer=5, head=3)         # the GAB generator

Each renders exactly its panel from the app, nothing else. What they ship:

  attention_widget  the scaled QKᵀ logits and the generated GAB of every block
                    and head, fp16 (~1.4 MB); the JS takes softmax(QKᵀ + GAB)
                    itself, so every block, head and query square is live
  gab_widget        the static template bank plus every head's mixing
                    coefficients (~750 KB); `gab_bias == coeffs @ templates`
                    holds exactly (engine.gab_coeffs asserts it), so the JS
                    rebuilds any head's bias and decomposes any square pair

The position and the Elo are baked in — only Python has the model, so changing
those means re-running the cell.

Where it renders: each widget is an <iframe srcdoc> — Colab, JupyterLab, VS Code
and classic Jupyter all execute scripts inside one, whereas a bare <script> in
`display(HTML(...))` is silently dropped by JupyterLab. GitHub's .ipynb viewer
drops the iframe entirely, so on GitHub these cells look empty — the srcdoc
fallback never renders. Open in Colab or Jupyter.
`*_widget_html()` returns the same page as a standalone .html file for sharing.

`interp_plot.plot_attention` / `plot_gab_mixture` render these same two panels as
static figures — one slice each, for a paper or a batch run.

GAB / smolgen: GAB is the bias itself; "smolgen" is the upstream name for the
tiny generator that emits the mixing coefficients. This file says GAB throughout,
including in both pages' JS.
"""
from __future__ import annotations

import base64
import html as _html
import json

import chess
import numpy as np

from .pieces import PIECE_URI


def _canon(square: int, turn: bool) -> int:
    """python-chess square -> the model's canonical index (side-to-move frame,
    square = rank*8 + file). The app's realToCanon(). Duplicated verbatim in
    interp_plot.py so each module imports standalone — keep both, and the
    app's JS, in sync."""
    rank, file = chess.square_rank(square), chess.square_file(square)
    return (rank if turn == chess.WHITE else 7 - rank) * 8 + file


def _f16(t) -> str:
    """Tensor/array -> base64 float16. Half precision costs ~1e-3 on a GAB bias
    and ~1e-4 on an attention weight — invisible at the 2-3 decimals shown, and
    half the bytes of fp32 in a notebook that has to carry them in its output."""
    return base64.b64encode(np.asarray(t, dtype=np.float16).tobytes()).decode()


def _common(eng, board: chess.Board, elo: int, oppo_elo) -> dict:
    """Everything both pages need that isn't the big tensors — spread into each
    payload, so `attention_payload` / `gab_payload` carry all of it too:

      cells      the 64 squares in display order (White at the bottom), each
                 carrying the canonical index every engine tensor is indexed by,
                 so the JS never has to know about the side-to-move mirror
      pieces     data URIs for only the piece types actually on this board
      fen, elo, turn, n_blocks, n_heads
      best_san   the model's own move, for the header
      query      the default query square: the from-square of that move, or
                 canonical 28 (real e4 for White) when there is no legal move"""
    res = eng.evaluate(board, elo, oppo_elo)
    best = res["policy"][0][0] if res["policy"] else None
    mv = chess.Move.from_uci(best) if best else None
    cells = []
    for rank in range(7, -1, -1):                               # rank 8 -> 1
        for file in range(8):
            sq = chess.square(file, rank)
            pc = board.piece_at(sq)
            cells.append({"name": chess.square_name(sq),
                          "canon": _canon(sq, board.turn),
                          "piece": pc.symbol() if pc else None,
                          "light": (file + rank) % 2 == 1})
    return {
        "cells": cells,
        "pieces": {k: v for k, v in PIECE_URI.items()
                   if any(c["piece"] == k for c in cells)},
        "fen": board.fen(),
        "elo": int(elo),
        "turn": "White" if board.turn else "Black",
        "n_heads": int(eng.cfg.num_heads),
        "n_blocks": int(eng.cfg.num_blocks),
        "best_san": board.san(mv) if mv else None,
        "query": _canon(mv.from_square, board.turn) if mv else 28,
    }


def attention_payload(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None) -> dict:
    """The attention panel's data: scaled QKᵀ and the generated GAB for every
    block and head, base64 fp16 (n_blocks, n_heads, 64, 64) each — the JS takes
    the softmax itself. Plus every key from `_common`."""
    nb = eng.cfg.num_blocks
    qk = np.stack([eng.qk_scores(board, elo, oppo_elo, layer=L).numpy() for L in range(nb)])
    gab = np.stack([eng.gab_bias(board, elo, oppo_elo, layer=L).numpy() for L in range(nb)])
    return {**_common(eng, board, elo, oppo_elo), "qk": _f16(qk), "gab": _f16(gab)}


def gab_payload(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None) -> dict:
    """The GAB drawer's data: the static template bank (base64 fp16
    (gen, 64, 64)) and [block][head][gen] generated mixing coefficients. Plus
    every key from `_common`.

    A single block built without GAB comes back as null coefficients; a model
    built entirely without GAB has no template bank at all, so gab_templates()
    raises RuntimeError here — use `attention_widget` on those."""
    coeffs = []
    for L in range(eng.cfg.num_blocks):
        try:
            c = eng.gab_coeffs(board, elo, oppo_elo, layer=L)
        except RuntimeError:                                    # use_gab=False
            coeffs.append(None)
            continue
        coeffs.append([[round(float(v), 5) for v in row] for row in c])
    T = eng.gab_templates()
    return {**_common(eng, board, elo, oppo_elo),
            "templates": _f16(T), "n_gen": int(T.shape[0]), "coeffs": coeffs}


def _page(template: str, data: dict, layer: int, head: int) -> str:
    data = {**data, "layer": int(layer), "head": int(head)}
    return template.replace("__PAYLOAD__", json.dumps(data).replace("<", "\\u003c"))


def _frame(doc: str, height: int):
    import warnings

    from IPython.display import HTML

    markup = (
        f'<iframe srcdoc="{_html.escape(doc, quote=True)}" width="100%" '
        f'height="{height}" frameborder="0" style="border:0;border-radius:10px;'
        f'color-scheme:dark">This notebook viewer strips JavaScript — run the '
        f'cell in Colab or Jupyter for the interactive panel.</iframe>')
    with warnings.catch_warnings():
        # HTML() nudges you toward IPython.display.IFrame whenever it sees an
        # <iframe>, but IFrame only takes a src URL — the whole point here is
        # srcdoc, which needs no server and no file next to the notebook.
        warnings.filterwarnings("ignore", message=".*IFrame.*")
        return HTML(markup)


def attention_widget_html(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                          layer: int = 0, head: int = 0) -> str:
    """"Live attention · this position" as a standalone HTML page."""
    return _page(_ATTENTION_PAGE, attention_payload(eng, board, elo, oppo_elo=oppo_elo),
                 layer, head)


def gab_widget_html(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                    layer: int = 0, head: int = 0) -> str:
    """"How L·H's GAB is generated" as a standalone HTML page."""
    return _page(_GAB_PAGE, gab_payload(eng, board, elo, oppo_elo=oppo_elo), layer, head)


def attention_widget(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
                     layer: int = 0, head: int = 0, height: int = 590):
    """The app's live-attention panel in the current cell: pick a layer and head,
    click a square to set the query. Returns an IPython HTML display object — let
    it be the cell's last expression. An iframe can't size itself to its content,
    so `height` is fixed in pixels; raise it if the panel is clipped, or if a
    narrow cell makes the boards wrap onto another row."""
    return _frame(attention_widget_html(eng, board, elo, oppo_elo=oppo_elo,
                                        layer=layer, head=head), height)


def gab_widget(eng, board: chess.Board, elo: int = 1500, *, oppo_elo=None,
               layer: int = 0, head: int = 0, height: int = 880):
    """The app's GAB-generator drawer in the current cell: the live decomposition
    of one square pair, the generated mixing coefficients, and the template
    vocabulary. Click a template to inspect it. Returns an IPython HTML display
    object — let it be the cell's last expression. `height` is fixed in pixels
    (an iframe can't size itself); raise it if the template gallery is clipped."""
    return _frame(gab_widget_html(eng, board, elo, oppo_elo=oppo_elo,
                                  layer=layer, head=head), height)


# --------------------------------------------------------------------------
# Shared chrome: the app's palette, board cells, colormaps and fp16 unpacking.
# Both pages are built from these plus their own panel.
# --------------------------------------------------------------------------
_CSS = r"""
<meta charset="utf-8">
<style>
  /* Neutrals lifted one step from pitch black (2026-07), to match the paper
     figures and ui.py — see interp_plot.py's palette note. To revert:
       --bg:#0e1014; --panel:#161a21; --panel2:#1b2029; --line:#262c37;
       --chart-bg:#0f131a; */
  :root{
    --bg:#181c24; --panel:#20252f; --panel2:#262c38; --line:#333b4a;
    --chart-bg:#1a1f29;   /* insets: pair readout, gab readout, coeff strip */
    --text:#e7eaf0; --muted:#8b93a3; --accent:#6ea8fe; --accent2:#7bd88f;
    --sq-light:#c9d1dc; --sq-dark:#6b7686;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:16px;background:var(--bg);color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
  .card h2{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--muted);
           text-transform:uppercase;letter-spacing:.6px}
  .sub{font-size:10px;color:var(--muted);font-family:var(--mono);margin:-4px 0 10px}
  .sub b{color:var(--text)}
  /* chips */
  .chiprow{display:flex;align-items:flex-start;gap:8px;margin-bottom:8px}
  .chiprow .lbl{flex:0 0 38px;padding-top:4px;font-size:11px;color:var(--muted)}
  .chips{display:flex;flex:1;min-width:0;flex-wrap:wrap;gap:4px}
  .chip{padding:3px 8px;font-size:11px;border:1px solid var(--line);border-radius:6px;
        background:var(--panel2);cursor:pointer;font-family:var(--mono);color:var(--text)}
  .chip:hover{border-color:var(--accent)}
  .chip.active{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:700}
  /* boards */
  .attlabel{font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:4px;
            text-align:center;font-weight:600}
  .attpair .attlabel{min-height:26px;display:flex;align-items:flex-end;justify-content:center}
  .attboard{width:100%;aspect-ratio:1/1;display:grid;grid-template-columns:repeat(8,1fr);
            grid-template-rows:repeat(8,1fr);border:1px solid var(--line);border-radius:5px;
            overflow:hidden;background:#10141b}
  .attcell{cursor:pointer;position:relative}
  .attcell::after{content:"";position:absolute;inset:0;pointer-events:none}
  .attcell.dk::after{background:rgba(0,0,0,.16)}
  .attcell.lt::after{background:rgba(255,255,255,.05)}
  .attcell.q{box-shadow:inset 0 0 0 2px #ff5d6c;z-index:2}
  .attcell.k{box-shadow:inset 0 0 0 2px #ffffffcc;z-index:2}
  .attcell img{position:absolute;inset:6%;width:88%;height:88%;z-index:3;pointer-events:none;
               filter:drop-shadow(0 1px 2px rgba(0,0,0,.45))}
  .attcoord{position:absolute;z-index:1;font-size:6px;line-height:1;font-family:var(--mono);
            color:#ffffff5c;pointer-events:none}
  .attcoord.f{right:1px;bottom:0} .attcoord.r{left:1px;top:0}
  .attcap{font-size:10px;color:var(--muted);margin-bottom:10px;line-height:1.45}
  .attlegend{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:9px;
             color:var(--muted);font-family:var(--mono)}
  .legbar{flex:1;height:8px;border-radius:4px;border:1px solid var(--line);
          background:linear-gradient(90deg, rgb(68,1,84), rgb(59,82,139), rgb(33,144,141),
                     rgb(93,200,99), rgb(253,231,37))}
  .legbar.div{background:linear-gradient(90deg, rgb(64,132,234), rgb(24,28,36), rgb(244,134,58))}
  .leghint{font-size:8px;color:var(--muted);margin-top:3px;font-family:var(--mono)}
</style>
"""

_JS_COMMON = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = id => document.getElementById(id);
const NH = D.n_heads, NB = D.n_blocks;

/* fp16 -> Float32Array */
function unpack(b64){
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const dv = new DataView(raw.buffer), out = new Float32Array(raw.length / 2);
  for(let i = 0; i < out.length; i++){
    const h = dv.getUint16(i * 2, true);
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0  ? s * 6.103515625e-5 * (f / 1024)
           : e === 31 ? (f ? NaN : s * Infinity)
           : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

/* the app's colormaps */
const MID = [24,28,36], BLUE = [64,132,234], ORANGE = [244,134,58];
const lerp3 = (a, b, t) =>
  `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;
function divmap(v){ v = Math.max(-1, Math.min(1, v)); return lerp3(MID, v < 0 ? BLUE : ORANGE, Math.abs(v)); }
function viridis(t){
  t = Math.max(0, Math.min(1, t));
  const s = [[68,1,84],[59,82,139],[33,144,141],[93,200,99],[253,231,37]];
  const x = t * (s.length - 1), i = Math.min(Math.floor(x), s.length - 2);
  return lerp3(s[i], s[i+1], x - i);
}
const symScale = row => {
  let m = 0; for(const v of row){ const a = Math.abs(v); if(a > m) m = a; }
  m = m || 1; return v => divmap(v / m);
};
const nameOf = canon => D.cells.find(c => c.canon === canon).name;

/* boards — 64 heat cells; click = set query, hover = pick the key square */
function fillAttCells(el, {pieces = false, interactive = true} = {}){
  el.innerHTML = '';
  D.cells.forEach((c, i) => {
    const d = document.createElement('div');
    d.className = 'attcell ' + (c.light ? 'lt' : 'dk');
    d.dataset.canon = c.canon;
    if(pieces && c.piece){
      const im = document.createElement('img'); im.src = D.pieces[c.piece]; d.appendChild(im);
    }
    if(i % 8 === 0){ const s = document.createElement('span'); s.className='attcoord r'; s.textContent = c.name[1]; d.appendChild(s); }
    if(i >= 56){ const s = document.createElement('span'); s.className='attcoord f'; s.textContent = c.name[0]; d.appendChild(s); }
    if(interactive){
      d.onclick = () => { q = c.canon; k = null; renderAll(); };
      d.onmouseenter = () => { k = c.canon; onHover(); };
    }
    el.appendChild(d);
  });
}
function paintRow(el, row, colf){
  [...el.children].forEach(d => { d.style.background = colf(row[+d.dataset.canon]); });
}
function marks(){
  document.querySelectorAll('.attcell').forEach(d => {
    const c = +d.dataset.canon;
    d.classList.toggle('q', c === q);
    d.classList.toggle('k', c === k && c !== q);
  });
}
function buildChips(el, n, cur, pick){
  el.innerHTML = '';
  for(let i = 0; i < n; i++){
    const b = document.createElement('div');
    b.className = 'chip' + (i === cur ? ' active' : '');
    b.textContent = i;
    b.onclick = () => pick(i);
    el.appendChild(b);
  }
}
"""

# --------------------------------------------------------------------------
# Live attention · this position
# --------------------------------------------------------------------------
_ATTENTION_PAGE = _CSS + r"""
<style>
  .attset{display:flex;flex-direction:column;gap:12px}
  .row{display:grid;grid-template-columns:200px 1fr;gap:16px;align-items:start}
  .attpair{display:flex;gap:10px}
  .attpair>div{flex:1;min-width:0}
  @media (max-width:760px){ .row{grid-template-columns:1fr} }
  .pos .attcell{cursor:pointer}
  .pos .attcell.lt{background:var(--sq-light)} .pos .attcell.dk{background:var(--sq-dark)}
  .pair{font-family:var(--mono);font-size:11px;line-height:1.7;background:var(--chart-bg);
        border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin-top:10px}
  .pair .k{color:var(--muted)} .pair b{color:var(--text)}
</style>

<div class="card">
  <h2>Live attention · this position</h2>
  <div class="sub" id="sub"></div>
  <div class="chiprow"><span class="lbl">Layer</span><div class="chips" id="layerChips"></div></div>
  <div class="chiprow"><span class="lbl">Head</span><div class="chips" id="headChips"></div></div>
  <div class="attcap">Click a square to set the query.</div>
  <div class="row">
    <div>
      <div class="attlabel">position</div>
      <div class="attboard pos" id="bpos"></div>
    </div>
    <div class="attset">
      <div class="attpair">
        <div><div class="attlabel">semantic attention (QKᵀ) </div><div class="attboard" id="att_qk"></div></div>
        <div><div class="attlabel">geometric attention (GAB) </div><div class="attboard" id="att_gab"></div></div>
        <div><div class="attlabel">final head attention matrix (scaled softmax(QKᵀ + GAB)) </div><div class="attboard" id="att_sum"></div></div>
      </div>
      <div>
        <div class="attlegend" style="margin-top:0"><span>0</span><div class="legbar"></div><span>max</span></div>
        <div class="leghint">query's 64 weights (one per key square) sum to 1</div>
      </div>
      <div class="pair" id="pair">—</div>
    </div>
  </div>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
""" + _JS_COMMON + r"""
const QK = unpack(D.qk), GAB = unpack(D.gab);     // (NB, NH, 64, 64) each
let layer = D.layer, head = D.head, q = D.query, k = null;

const rowOf = (buf, qq) => {
  const off = ((layer * NH + head) * 64 + qq) * 64;
  return buf.subarray(off, off + 64);
};
function softmax(a, b){                            // the head's actual attention
  const z = new Float64Array(64);
  let mx = -Infinity;
  for(let i = 0; i < 64; i++){ z[i] = a[i] + (b ? b[i] : 0); if(z[i] > mx) mx = z[i]; }
  let s = 0;
  for(let i = 0; i < 64; i++){ z[i] = Math.exp(z[i] - mx); s += z[i]; }
  for(let i = 0; i < 64; i++) z[i] /= s;
  return z;
}
function renderPair(){
  const el = $('pair');
  if(k === null){ el.innerHTML = '<span class="k">hover a square to read that pair</span>'; return; }
  const qk = rowOf(QK, q), gab = rowOf(GAB, q);
  const at = softmax(qk, gab), plain = softmax(qk);
  const f = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
  el.innerHTML =
    `<b>${nameOf(q)} → ${nameOf(k)}</b> &nbsp; <span class="k">QKᵀ</span> ${f(qk[k])}` +
    ` &nbsp; <span class="k">GAB</span> ${f(gab[k])}` +
    ` &nbsp; <span class="k">sum</span> ${f(qk[k] + gab[k])} &nbsp;→&nbsp; ` +
    `<b>${(at[k] * 100).toFixed(1)}%</b> of this query's attention ` +
    `<span class="k">(without GAB ${(plain[k] * 100).toFixed(1)}%)</span>`;
}
function onHover(){ renderPair(); marks(); }
function renderAll(){
  buildChips($('layerChips'), NB, layer, i => { layer = i; k = null; renderAll(); });
  buildChips($('headChips'),  NH, head,  i => { head  = i; k = null; renderAll(); });
  // QKᵀ and GAB are pre-softmax logits: diverging scale centered on 0, symmetric
  // per board (blue = negative, orange = positive).
  const qk = rowOf(QK, q), gab = rowOf(GAB, q);
  paintRow($('att_qk'),  qk,  symScale(qk));
  paintRow($('att_gab'), gab, symScale(gab));
  // Third board = the head's ACTUAL attention. It's a probability distribution
  // (this query's row sums to 1), so use a sequential viridis scale from 0 to the
  // row's peak weight, not the logit scale.
  const at = softmax(qk, gab);
  let am = 0; for(const v of at){ if(v > am) am = v; }
  paintRow($('att_sum'), at, v => viridis(v / (am || 1)));
  renderPair(); marks();
}
$('sub').innerHTML = `${D.turn} to move · Maia rating (self_elo) ${D.elo}` +
  (D.best_san ? ` · plays <b>${D.best_san}</b>` : '') + `<br>${D.fen}`;
fillAttCells($('bpos'), {pieces: true});
['att_qk','att_gab','att_sum'].forEach(id => fillAttCells($(id)));
renderAll();
</script>
"""

# --------------------------------------------------------------------------
# How L·H's GAB is generated
# --------------------------------------------------------------------------
_GAB_PAGE = _CSS + r"""
<style>
  .mlhead{display:flex;align-items:baseline;gap:12px;margin-bottom:8px;flex-wrap:wrap}
  .mltitle{font-size:13px;font-weight:600}
  .mltitle b{color:var(--accent2)}
  .mlhint{font-size:10px;color:var(--muted);font-family:var(--mono);flex:1;min-width:0}
  .gabrow{display:grid;grid-template-columns:180px 1fr;gap:16px;align-items:start}
  @media (max-width:760px){ .gabrow{grid-template-columns:1fr} }
  .gabreadout{font-family:var(--mono);font-size:10px;line-height:1.7;background:var(--chart-bg);
    border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:6px 0 10px;min-height:30px}
  .gabreadout b{color:var(--text)}
  .gabreadout .pos{color:#f0a35e} .gabreadout .neg{color:#6fb3ff}
  .gabreadout .tref{color:var(--accent);cursor:pointer}
  .gabreadout .tref:hover{text-decoration:underline}
  .coeffstrip{display:flex;gap:1px;height:54px;background:var(--chart-bg);border:1px solid var(--line);
    border-radius:6px;padding:3px;align-items:center}
  .cbar{flex:1;height:100%;position:relative;cursor:pointer;border-radius:2px}
  .cbar:hover{background:#ffffff12} .cbar.selt{background:#ffffff1f}
  .cbar span{position:absolute;left:1px;right:1px;border-radius:1px}
  .gallery{display:grid;grid-template-columns:repeat(16,1fr);gap:4px}
  @media (max-width:760px){ .gallery{grid-template-columns:repeat(8,1fr)} }
  .gtile{cursor:pointer;text-align:center}
  .gtile canvas{width:100%;aspect-ratio:1/1;image-rendering:pixelated;display:block;
    border:1px solid var(--line);border-radius:4px}
  .gtile.selt canvas{border-color:var(--accent)}
  .gtlbl{font-size:8px;color:var(--muted);margin-top:2px;font-family:var(--mono)}
  .gabdetail{border:1px solid var(--line);border-radius:8px;background:var(--panel2);
    padding:10px;margin:8px 0 10px;display:flex;gap:14px;align-items:flex-start}
  .gabdetail.hidden{display:none}
  .gabdetail canvas#gdcv{width:140px;height:140px;image-rendering:pixelated;
    border:1px solid var(--line);border-radius:6px}
  .gdboard{width:140px}
  .gdinfo{flex:1;font-size:10px;color:var(--muted);font-family:var(--mono);line-height:1.6}
  .gdinfo b{color:var(--text)}
  .gdclose{float:right;background:var(--panel2);border:1px solid var(--line);color:var(--muted);
    border-radius:5px;cursor:pointer;font-size:10px;padding:2px 8px;font-family:var(--mono)}
  .gdclose:hover{border-color:var(--accent)}
</style>

<div class="card">
  <div class="mlhead">
    <span class="mltitle">How <b id="gabhead">L0·H0</b>'s GAB is generated</span>
    <span class="mlhint">a generator reads this position and emits, per head, a bias of the
      <span id="gabNcoeff">64</span> coefficients over a static bank of
      <span id="gabNtmpl">64</span> 64×64 square-pair templates shared by every layer</span>
  </div>
  <div class="chiprow"><span class="lbl">Layer</span><div class="chips" id="layerChips"></div></div>
  <div class="chiprow"><span class="lbl">Head</span><div class="chips" id="headChips"></div></div>
  <div class="attcap">Click a square to set the query, hover another to decompose that pair.</div>
  <div class="gabrow">
    <div>
      <div class="attlabel">generated GAB · query row</div>
      <div class="attboard" id="bgab"></div>
    </div>
    <div>
      <div class="gabreadout" id="gabreadout">—</div>
      <div class="attlabel" style="text-align:left">generated mixing coefficients · template
        <span id="gabRange">#0–63</span> · click to inspect</div>
      <div class="coeffstrip" id="coeffstrip"></div>
    </div>
  </div>
  <div class="gabdetail hidden" id="gabdetail"></div>
  <div class="attlabel" style="text-align:left;margin-top:10px">template vocabulary · the
    <span id="gabNtmpl2">64</span> static stencils (row = query sq, col = key sq)</div>
  <div class="gallery" id="gallery"></div>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
""" + _JS_COMMON + r"""
const N = D.n_gen;
const T = unpack(D.templates);                     // (N, 64, 64)
const gabtMaxAbs = new Float32Array(N);            // per-template |max|, for painting
for(let g = 0; g < N; g++){
  let m = 1e-9;
  for(let i = g * 4096; i < (g + 1) * 4096; i++){ const a = Math.abs(T[i]); if(a > m) m = a; }
  gabtMaxAbs[g] = m;
}
let layer = D.layer, head = D.head, q = D.query, k = null, selTemplate = null;
const coeffs = () => D.coeffs[layer] ? D.coeffs[layer][head] : null;

/* gab_bias[h][q][:] == sum_i coeffs[i] * template_i[q][:] — exactly */
function gabRow(qq){
  const c = coeffs(), row = new Float32Array(64);
  if(!c) return row;
  for(let g = 0; g < N; g++){
    const cv = c[g]; if(cv === 0) continue;
    const off = g * 4096 + qq * 64;
    for(let kk = 0; kk < 64; kk++) row[kk] += cv * T[off + kk];
  }
  return row;
}
function paintTemplate(cv, g){                     // 64×64 pair-matrix -> one pixel per (query, key)
  const ctx = cv.getContext('2d'), img = ctx.createImageData(64, 64), s = gabtMaxAbs[g];
  for(let i = 0; i < 4096; i++){
    const v = Math.max(-1, Math.min(1, T[g*4096+i] / s)), c = v < 0 ? BLUE : ORANGE, a = Math.abs(v);
    const p = i * 4;
    for(let j = 0; j < 3; j++) img.data[p+j] = Math.round(MID[j] + (c[j]-MID[j]) * a);
    img.data[p+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
const f = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
const cls = v => v >= 0 ? 'pos' : 'neg';

function defaultGabTarget(){                       // strongest |GAB| target for the query
  const row = gabRow(q); let bi = 0, bv = -1;
  for(let i = 0; i < 64; i++){ const a = Math.abs(row[i]); if(a > bv){ bv = a; bi = i; } }
  return bi;
}
function renderGabReadout(){
  const el = $('gabreadout'), c = coeffs();
  if(!c){ el.textContent = 'this block was built without GAB'; return; }
  if(k === null) k = defaultGabTarget();
  const terms = []; let total = 0;
  for(let g = 0; g < N; g++){ const v = c[g] * T[g*4096 + q*64 + k]; terms.push({i: g, v}); total += v; }
  terms.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top = terms.slice(0, 4);
  let rest = total; for(const t of top) rest -= t.v;
  el.innerHTML = `GAB ${nameOf(q)}→${nameOf(k)} = <b class="${cls(total)}">${f(total)}</b> &nbsp;=&nbsp; ` +
    top.map(t => `<span class="${cls(t.v)}">${f(t.v)}</span>·<span class="tref" data-i="${t.i}">#${t.i}</span>`).join(' ') +
    ` <span style="opacity:.6">${f(rest)} rest</span>`;
  el.querySelectorAll('.tref').forEach(x => { x.onclick = () => toggleTemplate(+x.dataset.i); });
}
function buildCoeffStrip(){
  const s = $('coeffstrip'); s.innerHTML = '';
  for(let i = 0; i < N; i++){
    const b = document.createElement('div');
    b.className = 'cbar'; b.appendChild(document.createElement('span'));
    b.onclick = () => toggleTemplate(i);
    s.appendChild(b);
  }
}
function renderCoeffStrip(){
  const c = coeffs();
  let m = 1e-9; if(c) for(const v of c){ const a = Math.abs(v); if(a > m) m = a; }
  [...$('coeffstrip').children].forEach((b, i) => {
    const v = c ? c[i] : 0, sp = b.firstChild;
    sp.style.background = v >= 0 ? '#f0a35e' : '#6fb3ff';
    sp.style.height = Math.max(3, Math.abs(v) / m * 48) + '%';
    if(v >= 0){ sp.style.bottom = '50%'; sp.style.top = 'auto'; }
    else      { sp.style.top = '50%';    sp.style.bottom = 'auto'; }
    b.title = 'template #' + i + ' · coeff ' + v.toFixed(3);
    b.classList.toggle('selt', selTemplate === i);
  });
}
function buildGallery(){
  const g = $('gallery'); g.innerHTML = '';
  for(let i = 0; i < N; i++){
    const tile = document.createElement('div'); tile.className = 'gtile';
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    paintTemplate(cv, i);
    const lb = document.createElement('div'); lb.className = 'gtlbl';
    tile.appendChild(cv); tile.appendChild(lb);
    tile.onclick = () => toggleTemplate(i);
    g.appendChild(tile);
  }
}
function renderGalleryBadges(){                    // live coefficients on the static vocabulary
  const c = coeffs();
  let m = 1e-9; if(c) for(const v of c){ const a = Math.abs(v); if(a > m) m = a; }
  [...$('gallery').children].forEach((tile, i) => {
    const lb = tile.querySelector('.gtlbl'), cv = tile.querySelector('canvas');
    lb.innerHTML = c ? `<b>#${i}</b> <span style="color:${c[i] >= 0 ? '#f0a35e' : '#6fb3ff'}">${f(c[i])}</span>`
                     : `<b>#${i}</b>`;
    cv.style.borderColor = c ? divmap(c[i] / m) : '';
    tile.classList.toggle('selt', selTemplate === i);
  });
}
function renderTemplateDetail(){
  const box = $('gabdetail');
  if(selTemplate === null){ box.classList.add('hidden'); box.innerHTML = ''; return; }
  const i = selTemplate, c = coeffs(), v = c ? c[i] : null;
  let rank = null;
  if(v !== null){ rank = 1; for(const x of c) if(Math.abs(x) > Math.abs(v)) rank++; }
  box.classList.remove('hidden');
  box.innerHTML =
    `<div><canvas id="gdcv" width="64" height="64"></canvas>` +
      `<div class="gtlbl">full 64×64 · row = query, col = key</div></div>` +
    `<div class="gdboard"><div class="attboard" id="gdrowboard"></div>` +
      `<div class="gtlbl">its row at query ${nameOf(q)}</div></div>` +
    `<div class="gdinfo"><button class="gdclose" id="gdclose">close</button>` +
      `template <b>#${i}</b> — static stencil, shared by every layer &amp; head.` +
      (v === null ? '' : ` In L${layer}·H${head} on this board (elo ${D.elo}) its coefficient is ` +
        `<b>${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}</b> — #${rank} of ${N} by |coeff|.`) +
    `</div>`;
  paintTemplate($('gdcv'), i);
  const rb = $('gdrowboard');
  fillAttCells(rb, {interactive: false});
  const row = new Float32Array(64);
  for(let kk = 0; kk < 64; kk++) row[kk] = T[i*4096 + q*64 + kk];
  paintRow(rb, row, symScale(row));
  marks();
  $('gdclose').onclick = () => toggleTemplate(i);
}
function toggleTemplate(i){
  selTemplate = (selTemplate === i) ? null : i;
  renderCoeffStrip(); renderGalleryBadges(); renderTemplateDetail();
}
function onHover(){ renderGabReadout(); marks(); }
function renderAll(){
  $('gabhead').textContent = 'L' + layer + '·H' + head;
  buildChips($('layerChips'), NB, layer, i => { layer = i; k = null; renderAll(); });
  buildChips($('headChips'),  NH, head,  i => { head  = i; k = null; renderAll(); });
  renderGabReadout();
  const row = gabRow(q);
  paintRow($('bgab'), row, symScale(row));
  renderCoeffStrip(); renderGalleryBadges(); renderTemplateDetail();
  marks();
}
['gabNcoeff','gabNtmpl','gabNtmpl2'].forEach(id => { $(id).textContent = N; });
$('gabRange').textContent = '#0–' + (N - 1);
fillAttCells($('bgab'));
buildCoeffStrip(); buildGallery(); renderAll();
</script>
"""
