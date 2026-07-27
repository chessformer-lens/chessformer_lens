# Chessformer (Maia 3) mechanistic interpretability app.

<img width="1440" height="900" alt="Screenshot 2026-07-09 at 6 09 56 PM" src="https://github.com/user-attachments/assets/42218edf-d8b2-4d89-81d3-c750837d3df3" />

Play a transformer based chess bot (Maia 3) trained to mimic human play and analyze its move policy, its attention (semantic QKᵀ, the unique geometric bias GAB, and their sum), 
and how its residual stream evolves with depth or watch it LIVE through widgets or the dedicated app. 

Use mechanistic interpretability tools like logit lens and head ablations.

Take the geometric attention bias (GAB / "smolgen") apart live, see the static templates
every layer shares (`gab_gen_size` of them — 64 on 3m/5m, 128 on 23m/79m), and the linear combination of templates used for a pair of squares or go move centric with move microscope: one move's logit at all residual readout points. Ablate any head with a click and see the causal effects. Drag the ELO slider to re-evaluate a position at different skill levels.



## Try it in your own notebook

To make the app's figures in a notebook or script:

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
`3m`, `5m`, `23m` and `79m` all just work. Pick one at launch:

```bash
python app.py 23m               # built-in alias: 3m / 5m / 23m / 79m
python app.py --model maia3-79m # full alias, HF repo id, or HF URL
MAIA3_ALIAS=23m python app.py   # env var (still supported)
```

Weights for the chosen size download on first use and are cached. Larger models
are heavier per position — on CPU the 23M/79M attention and head-ablation views
are noticeably slower than 5M; set `MAIA3_DEVICE=mps` (Apple Silicon) or run on
CUDA to speed them up. Model weights: <https://huggingface.co/UofTCSSLab>

Built on
 Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>

Please don't hesitate to give feedback by email or at davidlitman.com

## License

AGPLv3 — see [LICENSE](LICENSE). This app is built on the
[CSSLab/maia3](https://github.com/CSSLab/maia3) model code (also AGPLv3);
model weights from [Hugging Face](https://huggingface.co/UofTCSSLab/Maia3-5M).
Chess piece artwork is python-chess's built-in "cburnett" set
(Colin M.L. Burnett, GFDL/BSD/GPL).

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



## Future implementations:

- Activation patching beyond zero-ablation.
- Decompose a move's logit into per head/MLP layer contributions directly
- Add batching plots
- Max-activating positions per head: feed many boards, show what most excites the selected head.
- Linear probes
- Skill-acquisition plot highlight which heads shift most between 600 and 2800 on the same position and more interpretability across a skill axis experiments.