<!-- ===========================================================================
     WRITING SCAFFOLD.
     Notes to self live in HTML comments, so they do NOT render on GitHub —
     safe to push mid-draft. Delete each note once its section is written.
     Sections with real content below are already accurate; leave them alone.
     Screenshot staging + verification steps: see the plan file.
     =========================================================================== -->

# Chessformer interp

<!-- TITLE + PITCH — TO WRITE (one line under the title)
     · It's a workbench / toolkit, not an activity. The app is the headline
       feature, not the whole product.
     · The old title ("Chessformer (Maia 3) mechanistic interpretability app.")
       said "app" and pinned to Maia-3 — both narrower than what this is.
     · Keep it to one sentence. Hero image goes immediately after, before prose.
-->

<img width="1440" height="900" alt="Screenshot 2026-07-09 at 6 09 56 PM" src="https://github.com/user-attachments/assets/42218edf-d8b2-4d89-81d3-c750837d3df3" />

<!-- HERO IMAGE = SHOT 1 — REPLACE THE IMAGE ABOVE
     Whole window, all three columns, no drawer open. Play ~8 plies into an open
     position so the policy list and the Moves card have content.
     The tag above is the stale Jul-09 hosted shot. The 5 files in Screenshots/
     are also stale (five-board attention panel; today ships three).
-->

<!-- WHAT THIS IS — TO WRITE. Three short paragraphs:
     1. WHY chess transformers are good interp subjects. They read the board as
        64 square tokens and emit a policy over moves, so every activation is
        indexed by a *square* — an attention row, a residual delta, an ablation
        effect can all be drawn back onto the board and just looked at. This is
        the argument the whole project rests on; the README has never said it.
     2. ONE ENGINE, THREE FRONTENDS. engine.py captures the residual stream at
        every sub-layer and owns all the analysis; the app, the figure layer and
        the widgets are views onto it. Nothing in engine.py imports the UI.
     3. SCOPE, HONESTLY. The methods are architecture-general; the bundled
        loader targets the Maia-3 family (Monroe et al., ICLR 2026), every
        released size 3M-79M. Any checkpoint with the same Post-LN block
        structure drops into MaiaEngine.
-->

## What it can do

<!-- TO WRITE — one bullet per method. Facts for each:

     · LOGIT LENS ACROSS DEPTH — decode the running residual stream at all
       2·num_blocks + 2 readout points (18 on every current size) and watch a
       move form sub-layer by sub-layer.

     · PER-HEAD CAUSAL ABLATION — removes a head's *exact* residual write,
       reconstructed from the block's own weights. Worth saying why that
       matters: the hooked attn_NN activation is post-out_proj, where heads are
       already mixed across channels, so slicing channels does NOT isolate a
       head. Sweep every head -> the carrier heatmap.

     · GAB / "smolgen" — Maia-3 GENERATES its positional bias instead of
       storing it. Three pieces to expose: the static template bank every layer
       and head shares, the per-head mixing coefficients generated live for the
       position, and any single square pair decomposed into its top templates.
       gab_bias == coeffs @ templates holds to 1e-4 — which is what lets the
       browser rebuild it client-side.

     · ACTIVATION PATCHING — run_with_hooks takes arbitrary intervention
       functions on any readout point, so patching and steering work, not just
       zero-ablation.

     · THE SKILL AXIS — Maia-3 conditions on a rating, so you can run one
       position at two ratings and diff the INTERNALS, per square and per depth.
       The genuinely unusual capability here; almost no other model has a skill
       axis to do interp across. Give it its own sentence.

     · CAVEAT, stated once, here — intermediate lens readouts skip the head's
       final LayerNorm, so compare readout points against each other rather
       than against the real policy. (Matches engine.logit_lens's docstring.)
-->

## The app

<!-- THE LARGEST SECTION. Three sub-parts below. Every quoted string is the
     app's ACTUAL label, verified against ui.py — use them verbatim rather than
     paraphrasing, so the README and the app share one vocabulary. -->

<!-- PLAY OR SET UP A POSITION — TO WRITE
     · Click-to-move. NO drag-and-drop (pieces are explicitly draggable=false).
     · Promotion picker modal: `Promote to`.
     · `New game`, `← Back`, and a dropdown:
       `You play White` / `You play Black` / `Set up position`.
     · FEN box, placeholder `paste a FEN to load`, plus a `Load` button. The box
       is also an OUTPUT — it rewrites itself after every move.
     · In setup mode both sides are yours; clicks move pieces ignoring legality,
       and clicking the same square twice deletes the piece.
     · Header line reads `Maia3 5M · cpu · 8 blocks × 8 heads × 256d`.
     · THE ONE GAP TO BE UPFRONT ABOUT: no piece palette, so FEN is the only
       way to ADD material.
     · Also: gold last-move arrow, legal-move dots, captures drawn as rings,
       `Moves` card with SAN history.
-->

<!-- READ THE POSITION — TO WRITE
     · `Win / Draw / Loss · side to move` — stacked bar.
     · `Policy over N legal moves` — scrollable ranked list, caps at 40 rows
       then `+N more legal moves`.
     · `Maia rating (self_elo)` slider: 600-2800, step 25, default 1500.
       Dragging re-evaluates the same position (180 ms debounce).
     · `compare with a second rating` checkbox reveals a `second rating` slider
       (default 1100). What responds: policy rows become paired blue/green bars
       with a signed Δ; the WDL bar splits into one labelled row per rating; the
       move microscope overlays a second curve ONLY when exactly one move is
       selected. The attention / GAB / residual panels always use slider A.
-->

<!-- TAKE THE MODEL APART  [= SHOT 2] — TO WRITE
     · `Live attention · this position`, with `Layer` and `Head` chip rows and
       the caption `Click a square to set the query.`
     · The three boards, exact labels:
         `semantic attention (QKᵀ)`
         `geometric attention (GAB)`
         `final head attention matrix (scaled softmax(QKᵀ + GAB))`
     · BEST SINGLE DETAIL IN THE APP: hover any heat cell and the GAB drawer
       decomposes that square pair live —
         GAB d4→e5 = +1.84 = +0.91·#12 +0.44·#3 … +0.12 rest
       — with every #N clickable to open that template.
     · `Ablate this head` (its note: `removes its exact residual write`) gives
       the top 8 moves by |Δp| as `Nf3  12.4 → 3.1%  -9.3`, plus
       `W/D/L 41/38/21 → 33/44/23`. Self-expires when position, rating, layer
       or head changes.
-->

<!-- THE THREE DRAWERS  [= SHOT 3, SHOT 4] — TO WRITE
     One open at a time, each peeking at the bottom with a `▲ pull up` grip,
     `Escape` closes.

     · `Residual stream across depth · this position` — a filmstrip of mini
       boards, one per readout point, top border colour-coded by writer kind
       (legend: `emb (input)` / `attn add` / `MLP add` / `enc (final norm)`),
       heat = per-square ‖Δ‖, logit-lens top move drawn on top. Opened by the
       `Watch residual stream` button.

     · `Move microscope` — click up to 4 policy moves to overlay their depth
       curves (4 fixed colours); dashed red "snap" marker at the start of the
       final rank-1 run; per-dot hover
       `b3 mlp · logit 4.21 · p 38.2% · rank 1/31`.

     · `carrier heads · Δlogit = ablated − clean` — the blocks×heads grid beside
       the curve. Hover gives `b3·h5  Δ -1.42 — supports Nf3`; the strongest
       cell is ringed red. CLICK A CELL TO JUMP THE ATTENTION PANEL TO THAT
       HEAD — be precise, clicking does not ablate. The final block is dimmed
       and striped, `excluded from carrier attribution`, because it writes
       straight to the logits. The grid re-fits to the model's head count.

     · `How L0·H0's GAB is generated` —
       `generated mixing coefficients · template #0–63 · click to inspect`,
       then `template vocabulary · the 64 static stencils (row = query sq,
       col = key sq)` over a gallery of canvas tiles.
-->

<!-- TWO THINGS TO DISCLOSE + THE POLISH NOTES — TO WRITE

     · SWEEP COST: num_blocks·(num_heads+1) forward passes — 72 on 5M, 264 on
       79M — and it blocks the microscope while it runs.

     · UNDISCLOSED TODAY: every position you analyze is dumped to activations/
       as ply016_white_elo1500.pt — a torch.save of every hook point's
       (64, dim) tensor plus {fen, self_elo, side_to_move, ply}. Reads as a
       feature if the README says so, a surprise if not. Worth adding that the
       filename keys on ply+side+elo only, so re-analysis silently overwrites,
       and nothing is pruned (146 files in the repo now).

     · POLISH, a line each: the model loads on a background thread so the window
       opens instantly (staged `Loading Maia model…` -> `Connecting to Python
       bridge…` -> `Loading Maia3 5M…`); the layout reads the loaded model's
       counts, so one UI drives 6/8/16/32-head models; attention board
       coordinates re-label per side-to-move so heatmaps stay aligned when Black
       is to move; native window, dark palette matched end to end.

     · DON'T CLAIM: no export button, no PGN, no session save, no URL/permalink
       (it's a native window, no address bar), no temperature control, no
       in-app model switcher (launch-time flag), no redo.
-->

## Figures for a paper or a batch run

<!-- TO WRITE: a sentence or two framing interp_plot as the app's read-once
     views as static figures. The code block below is accurate — keep as-is. -->

```python
import chess
from engine import MaiaEngine
import interp_plot as ip

eng = MaiaEngine()
board = chess.Board("<your fen>")

ip.plot_position(eng, board, 1500)            # board · policy · win/draw/loss
ip.plot_residual_film(eng, board, 1500)       # residual stream across depth
ip.plot_move_microscope(eng, board, 1500)     # one move's depth curve + carrier heads
```

## The two live panels in a notebook

The app's two interactive panels also run as self-contained notebook cells (or
standalone HTML pages) via `interp_widget.attention_widget` / `gab_widget` —
the app's own JS with the data injected instead of fetched. (GitHub's .ipynb
viewer drops the iframe entirely, so on GitHub these cells look empty — the
srcdoc fallback never renders. Open in Colab or Jupyter.)

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first launch downloads the Maia3-5M transformer weights (~20 MB) from
Hugging Face and a native window opens (no browser needed).

### Choosing a model size

The same interface drives every Maia-3 size — the GUI reads the loaded model's
block / head / dimension / GAB-template counts and lays itself out to match, so
`3m`, `5m`, `23m` and `79m` all just work.

| alias | blocks | heads | dim | GAB templates | ablation sweep |
|---|---|---|---|---|---|
| `3m` | 8 | 6 | 192 | 64 | 56 passes |
| `5m` (default) | 8 | 8 | 256 | 64 | 72 passes |
| `23m` | 8 | 16 | 512 | 128 | 136 passes |
| `79m` | 8 | 32 | 1024 | 128 | 264 passes |

<!-- Table verified against maia3.model_registry. Note `3m`'s registry display
     name is "Maia3 3M ablation" — don't call it plain 3M. -->

Pick one at launch:

```bash
python app.py 23m               # built-in alias: 3m / 5m / 23m / 79m
python app.py --model maia3-79m # full alias, HF repo id, or HF URL
MAIA3_ALIAS=23m python app.py   # env var (still supported)
```

Weights for the chosen size download on first use and are cached. Larger models
are heavier per position — on CPU the 23M/79M attention and head-ablation views
are noticeably slower than 5M; set `MAIA3_DEVICE=mps` (Apple Silicon) or run on
CUDA to speed them up. Model weights: <https://huggingface.co/UofTCSSLab>

## Use the engine directly

<!-- TO WRITE: a sentence or two. Point at engine.py's module docstring as the
     real reference rather than duplicating it — it's accurate now. Two facts
     worth stating: no UI dependency, and read paths return CPU tensors in the
     model's canonical side-to-move frame (square = rank*8 + file). -->

```python
import chess
from engine import MaiaEngine

eng = MaiaEngine()                      # or "maia3-23m"; downloads on first use
board = chess.Board("<your fen>")

out, cache = eng.run_with_cache(board, self_elo=1500)
eng.logit_lens(cache["postattn_04"], board)     # top legal move at that point
eng.run_with_hooks(board, 1500, fwd_hooks=[("attn_05", lambda a: a * 0)])
```

## Code layout

- `engine.py` — `MaiaEngine`: the interp core (model + hooks + logit lens + head ablation). No UI deps; imports cleanly in a notebook.
- `interp_plot.py` — the app's read-once views as matplotlib figures, same layouts and wording: `plot_position`, `plot_residual_film`, `plot_move_microscope`, `plot_carrier_heads`, `plot_attention`, `plot_gab_mixture`, `plot_gab_templates`, `plot_skill_diff`. Engine + matplotlib only.
- `piece_art.py` — draws `pieces.py`'s cburnett pieces in matplotlib from PNGs rasterised offline into `piece_art/`, so figures use the app's actual artwork with no native dependency at runtime (`piece_art/regenerate.py` rebuilds them, and needs cairosvg).
- `interp_widget.py` — the app's two interactive panels as a self-contained notebook cell or standalone HTML page: `attention_widget` (Live attention) and `gab_widget` (the GAB generator), plus their `*_html` twins; `ui.py`'s CSS, colormaps and interaction, reading injected data instead of the pywebview API. These are the sweep-a-space versions of `interp_plot`'s static `plot_attention` / `plot_gab_mixture` — take the figure for one frame, the widget to explore.
- `bridge.py` — game state + the JSON API the UI calls.
- `ui.py` — the whole interface (HTML/CSS/JS) as one string.
- `pieces.py` — SVG piece set as data URIs.
- `app.py` — launcher (native window via pywebview).
- `requirements.txt` — everything the app needs (matplotlib is only for the figure/notebook layer).

## Built on

Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>

Please don't hesitate to give feedback by email or at davidlitman.com

## License

AGPLv3 — see [LICENSE](LICENSE). This app is built on the
[CSSLab/maia3](https://github.com/CSSLab/maia3) model code (also AGPLv3);
model weights from [Hugging Face](https://huggingface.co/UofTCSSLab/Maia3-5M).
Chess piece artwork is python-chess's built-in "cburnett" set
(Colin M.L. Burnett, GFDL/BSD/GPL).

## Future implementations:

- Activation patching beyond zero-ablation.
- Decompose a move's logit into per head/MLP layer contributions directly
- Add batching plots
- Max-activating positions per head: feed many boards, show what most excites the selected head.
- Linear probes
- Skill-acquisition plot highlight which heads shift most between 600 and 2800 on the same position and more interpretability across a skill axis experiments.
