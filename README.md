# Chessformer (Maia 3) mechanistic interpretability app.

<img width="1440" height="900" alt="Screenshot 2026-07-09 at 6 09 56 PM" src="https://github.com/user-attachments/assets/42218edf-d8b2-4d89-81d3-c750837d3df3" />

Play a transformer based chess bot (Maia 3) trained to mimic human play and watch its move policy, 
its attention (self attention vs unique geometric attention), 
and how its residual stream evolves with depth LIVE. 

Visualize mechanistic interpretability tools like logit lens and head ablations.

Take the geometric attention bias (GAB / "smolgen") apart live: softmax(QKᵀ)
vs softmax(QKᵀ+GAB) side by side per head, the 64 static square-pair templates
every layer shares, the generated mixing coefficients, and a per-square-pair
decomposition ("GAB b5→h5 = −0.34·#5 −0.17·#31 …") for any position.

Go move-centric with the move microscope (click any policy row): one move's
logit at all 18 residual readout points — where it snaps into the plan — plus
its carrier-head heatmap (every head ablated; Δ = ablated − clean everywhere
in the app). The residual filmstrip lives in a second bottom drawer, with the
structure-write heat and the per-point logit-lens move combined in one strip.
(`compare_residual(elo_a, elo_b)` — per-square internals diff between two
ratings — stays available in engine.py / bridge.py for notebook work.)

Drag the ELO slider to re-evaluate a position at
different skill levels.



## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first launch downloads the Maia3-5M transformer weights (~20 MB) from
Hugging Face and a native window opens (no browser needed).

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
- `bridge.py` — game state + the JSON API the UI calls.
- `ui.py` — the whole interface (HTML/CSS/JS) as one string.
- `pieces.py` — SVG piece set as data URIs.
- `app.py` — launcher (native window via pywebview).
