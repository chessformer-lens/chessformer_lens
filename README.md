# Chessformer interpretability
A toolkit + visualizer designed for mech interp enthusiasts working with chess models that treat each square as a token (MAIA-3 is completed; Leela will be completed soon).

Since beginning to do interpretability work on chess engines, I was struck by the lack of supporting infrastructure. I could hardly find a decent GUI to display general UCI reading engines. So, I decided to build an interface both for interactive visualization and rigorous mechanistic interpretability functionality—especially inspired by Neel Nanda's fantastic transformer_lens library.

![Hero Image](Screenshots/Screenshot1.png)

Chess tranformers are good interp subjects because **[copy from my paper]**

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

- **Logit Lens Across Depth**: decode the running residual stream at all readout points {embed, attention_i, mlp_j,unembed}, and watch a move form layer by layer
- **Per Head Causal Ablation**: remove a head's exact residual write, reconstructed from the layer's own weights. Uses the models' placed hooks on attention layers. Also, sweep every head to get a carrier/supressor heatmap.
- **Dissect positional encoding GAB heads**: A small generator network reads
the board and emits, per head, linear combinations of a bank of 64×64 square-pair templates shared across every layer and head for its geometric attention head called GAB (a head does not
have a fixed geometry). Expose the static template bank each layer shares, the linear combination of these templates each head learns per position, and any single (to, from) square pair decomposed into its top templates.
- **Activation Patching**: run_with_hooks takes arbitrary intervention functions on any readout point, allowing for patching and steering.
- **Skill Axis**: Maia-3 conditions on a rating, so you can run one position at two ratings and compare the internals. This is a genuinely unusual and inspiring capaility; almost no other top model has a skill axis that allows for meaningful mech interp with skill acquisition. Highly recommended.



## Quickstart in colab:
<-------------------------------------------------------------------->
```python
!pip install -q git+https://github.com/CSSLab/maia3
!pip install -q "chessformer_lens[plot] @ git+https://github.com/chessformer-lens/chessformer_lens"

#Interact with the attention widget
import chess
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_widget as iw

eng = MaiaEngine()                      
board = "...fill in FEN for position..."
board = chess.Board(board)                   

iw.attention_widget(eng, board, 1500,layer=4,head=3) # click between different layers and heads in the widget
```
<-------------------------------------------------------------------->


## The standalone app

**Play or Set up a Position**
![alt text](Screenshots/Screenshot7.png)
Click to move your pieces. Select an Elo for the engine (recall that it mimics human play at that strength, with all of our biases)
and use `New game`, `← Back`, and a dropdown:`You play White` / `You play Black` / `Set up position`.
It is also possible to `paste a FEN to load` a position with the `Load` button. 

In setup mode both sides are yours; clicks move pieces ignoring legality, and clicking the same square twice deletes the piece. 
Select a color to continue as it from that position.

The last move is indicated by a gold arrow, legal moves for a selected piece are indicated by dots, captures are drawn as rings. The games
moves are recorded in SAN notation.

Header line reads `Maia3 5M · cpu · 8 blocks × 8 heads × 256d`. **[THIS IS OUTDATED both heads instead of layers and hard coded num heads]**

**Read the Position**

At the top in the center there is a `Win / Draw / Loss · side to move` stacked bar. 
Under it is the `Maia rating (self_elo)` slider: 600-2800, step 25, default 1500; Dragging reevaluates the same position. 
Under that is the scrollable ranked list: `Policy over N legal moves`. 

There is a `compare with a second rating` checkbox which reveals a `second rating` slider
(default 1100). Setting it makes the policy rows become paired blue/green bars showing the compared policy and evaluation. This second rating does not affect the attention or GAB or residual panel app features.

**Take the Model Apart**
Get the `Live attention · this position`, with `Layer` and `Head` chip rows.
Click a square to set the query for the three boards, labeled:
-`semantic attention (QKᵀ)`
-`geometric attention (GAB)`
-`final head attention matrix (scaled softmax(QKᵀ + GAB))`
`Ablate this head` (its note: `removes its exact residual write`) gives the top 8 moves by |Δp|.
![alt text](Screenshots/Screenshot2.png)
Unique to the app and Maia-3, hover over any attention square and the GAB drawer decomposes that square pair live, with every head clickable to open that template:
![alt text](Screenshots/Screenshot8.png)
 
**The Three Drawers**
![alt text](Screenshots/Screenshot4.png)
![alt text](Screenshots/Screenshot5.png)+![alt text](Screenshots/Screenshot3.png)
![alt text](Screenshots/Screenshot6.png)
One open at a time, each peeking at the bottom with a `▲ pull up` grip, `Escape` closes.

· `Residual stream across depth · this position` — creates a filmstrip of mini
  boards, one per readout point, top border color coded by writer kind, **[SAY MORE HERE]**
  heat = per-square ‖Δ‖, logit-lens top move in miniature display on top. Opened by the
  `Watch residual stream` button.

· `Move microscope (and Carrier Heads)` — click up to 4 policy moves to overlay their depth curves (essentially logit lens narrowed to one policy); marker shows when a move sustains rank-1; per-dot hover
ex: `b3 mlp · logit 4.21 · p 38.2% · rank 1/31`.
`carrier heads` runs (layers x heads) forward passes with different ablated heads and record: `Δlogit = ablated − clean`.Hover for exact values. The largest Δlogit head is ringed red. The final layer is dimmed and striped, `excluded from carrier attribution`, because it writes straight to the logits and muddles meaningful results from earlier layers. 

· `How L[i]H[j] GAB is generated` — see the network's generated linear combination of each template #0–63. `click to inspect`, and under
see `template vocabulary · the 64 static stencils (row = query sq, col = key sq)` as a gallery.


## Addional Notes
The model loads on a background thread so the window opens instantly; the layout reads the loaded model's counts so all sizes of Maia-3 with different head counts work the same. As of now, it opens in a native window and there is no export button or permalink for a session.

## Install as library (this and below is not necessarily fully functional yet)

`chessformer_lens` is a normal Python package, but it needs one thing installed
first: `engine.py` imports `maia3`, the Maia-3 model code, which is **not on
PyPI**, so it can't be declared as a dependency and has to come from GitHub.

```bash
pip install git+https://github.com/CSSLab/maia3
pip install "chessformer_lens[all] @ git+https://github.com/chessformer-lens/chessformer_lens"
```

The install also installs torch, numpy, python-chess, huggingface-hub and
ipython for `engine.py` and the widgets. 

| extra | adds | needed for |
|---|---|---|
| *(none)* | — | `engine.py`, `interp_widget.py` |
| `[plot]` | matplotlib | `interp_plot.py`, `piece_art.py` — the static figures |
| `[app]` | pywebview | `app.py` — the native window |
| `[all]` | both | everything |

From a clone, `pip install -e ".[all]"` does the same thing; `pip install -r
requirements.txt` is the older flat equivalent and still works.

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
chessformer-lens
```

Or without installing, `python -m chessformer_lens.app`


The first launch downloads the Maia-3 5m model weights (~20 MB) from
Hugging Face and a native window opens (no browser needed).

### Choosing a model size

The same interface drives every Maia-3 size — the GUI reads the loaded model's
block / head / dimension / GAB-template counts and lays itself out to match, so `5m`, `23m` and `79m` all work. The higher parameter models have 16 and 32 heads per layer, and 512 and 1024 dim of the model (d_model).


Various ways to launch:

```bash
chessformer-lens 23m               
chessformer-lens --model maia3-79m 
MAIA3_ALIAS=5m chessformer-lens   
```

<!-- understand about set `MAIA3_DEVICE=mps` (Apple Silicon) or run on
CUDA to speed higher parameter models up. Model weights: <https://huggingface.co/UofTCSSLab> -->

## Use the engine directly


<!-- TO WRITE: a sentence or two. Point at engine.py's module docstring as the
     real reference rather than duplicating it — it's accurate now. Two facts
     worth stating: no UI dependency, and read paths return CPU tensors in the
     model's canonical side-to-move frame (square = rank*8 + file). -->


See the residual stream film and move microscope for the famous Paul Morphy "Opera Game"
```python
import chess
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_plot as ip

#opera game FEN right at legendary queen sacrifice 
fen = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16" 
import matplotlib.pyplot as plt

eng = MaiaEngine('23m')
board = chess.Board(fen)                   
move = eng.to_move(board,'Qb8+')
ip.plot_position(eng, board,2600)
plt.show()
print("\n\n")
ip.plot_residual_film(eng, board, 2600) #exceptional Elo rating to match Morphy's skill
plt.show()
print("\n\n")
ip.plot_move_microscope(eng, board, 2600, move)     # one move's depth curve + carrier heads
plt.show()
plt.close()
```

## Code layout

Everything importable lives in the `chessformer_lens/` package; the paths below
are relative to it. `import chessformer_lens` itself is cheap — the submodules
resolve on first use, so nothing drags in torch, matplotlib or pywebview until
you actually touch that layer.

- `__init__.py` — the package surface: `MaiaEngine`, `build_cfg`, `pick_device`, `attention_widget`, `gab_widget`, `__version__`.
- `engine.py` — `MaiaEngine`: the interp core (model + hooks + logit lens + head ablation + list of values through depth). No UI deps; imports cleanly in a notebook.
- `interp_plot.py` — the app's read-once views as matplotlib figures, same layouts and wording: `plot_position`, `plot_residual_film`, `plot_move_microscope`, `plot_carrier_heads`, `plot_attention`, `plot_gab_mixture`, `plot_gab_templates`, `plot_skill_diff`. Engine + matplotlib only.
- `interp_widget.py` — the app's two interactive panels as a self-contained notebook cell or standalone HTML page: `attention_widget` (Live attention) and `gab_widget` (the GAB generator), plus their `*_html` twins; `ui.py`'s CSS, colormaps and interaction, reading injected data instead of the pywebview API. These are the interactive versions of `interp_plot`'s static `plot_attention` / `plot_gab_mixture`. Take the figure for one frame, the widget to explore.
- `piece_art.py` — draws `pieces.py`'s cburnett pieces in matplotlib from PNGs rasterised offline into `piece_art/`, so figures use the app's actual artwork with no native dependency at runtime (`piece_art/regenerate.py` rebuilds them, and needs cairosvg).
- `bridge.py` — the game state + the JSON API the UI calls.
- `ui.py` — the whole interface (HTML/CSS/JS) as one string including guide to change template.
- `pieces.py` — SVG piece set as data URIs.
- `app.py` — launcher (native window via pywebview); the `chessformer-lens` command is its `main()`.

At the repo root: `pyproject.toml` (packaging metadata, deps and extras),
`requirements.txt` (the flat everything-at-once install, including `maia3`),
`CITATION.cff`, `LICENSE` and `THIRD-PARTY-NOTICES.md`.

## Built on

Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>

Please don't hesitate to give feedback by email or at davidlitman.com

## License

**MIT** — see [LICENSE](LICENSE). Use it, fork it, build on it.

Two things MIT can't cover, both spelled out in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md):

- It runs on [python-chess](https://github.com/niklasf/python-chess) (GPLv3+)
  and [CSSLab/maia3](https://github.com/CSSLab/maia3) (AGPLv3), which pip
  installs separately — no code from either is in this repo. Installing and
  using this project is unaffected, but if you redistribute a *bundle* that
  ships those libraries alongside it (Docker image, frozen binary, vendored
  copy), the bundle as a whole is subject to their terms, not MIT.
- Chess piece artwork is the "cburnett" set by Colin M.L. Burnett, used under
  the BSD option of its GFDL/BSD/GPL triple license.

Model weights come from [Hugging Face](https://huggingface.co/UofTCSSLab) under
their own terms.
<!--
## Future implementations:

- Activation patching beyond zero-ablation.
- Decompose a move's logit into per head/MLP layer contributions directly
- Add batching plots
- Max-activating positions per head: feed many boards, show what most excites the selected head.
- Linear probes
- Skill-acquisition plot highlight which heads shift most between 600 and 2800 on the same position and more interpretability across a skill axis experiments.
- **Add the next engine functionality: Leela Lc0-BT**: 
     - How to: Lc0 uses .pb so I need to first convert to PyTorch (community converters do exist for this). Lc0 uses smolgen not GAB which is a learned per position attention bias. Worth researching if smolgen plays the role GAB does perhaps. For logit lens, Maia-3 and Lc0 differ by pre-LN vs post-LN at readout points so be wary of that. Also, Lc0 does not have a skill panel, BUT IT IS VERY WIDELY USED and worked on in interp (e.g. Erik Jenner et al paper).
-->
## Citing

If `chessformer_lens` contributes to published work, a citation is very appreciated and helps others find it! Please see
`CITATION.cff` and GitHub's "Cite this repository" button.
