# Chessformer (Maia 3) mechanistic interpretability app.

<img width="1440" height="900" alt="Screenshot 2026-07-09 at 6 09 56 PM" src="https://github.com/user-attachments/assets/42218edf-d8b2-4d89-81d3-c750837d3df3" />

Play a transformer based chess bot (Maia 3) trained to mimic human play and watch its move policy, its attention (self attention vs unique geometric attention), 
and how its residual stream evolves with depth LIVE. 

Visualize mechanistic interpretability tools like logit lens and head ablations.

Take the geometric attention bias (GAB / "smolgen") apart live, see the 64 static templates
every layer shares, and the linear combination of templates used for a pair of squares or go move centric with move microscope: one move's logit at all residual readout points. Ablate any head with a click and see the causal effects. Drag the ELO slider to re-evaluate a position at different skill levels.



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



# Later will implement:
Grouped by how much they'd help people form and test hypotheses. I'd start with the first three — they're the highest leverage for a shared community tool and mostly build on hooks you already have.

Reproducibility & sharing (do first)

State in the URL — encode FEN + ELO + layer/head/query + open window + compared moves into a shareable link. Essential so someone can say "look at b3·h5 on this position" and others land on the exact view.
Export the analysis — download the current attention matrices, residual deltas, ablation grid, and smolgen coefficients as JSON/NPY so people can continue in a notebook. You already persist activations; just surface a button.
Causal experiments (the core of the field)

Activation patching / causal tracing — patch a head's output or a residual position from a "corrupted" position into a "clean" one and watch the policy move. This is the gold-standard experiment and your hooks already support the ablation version; patching is the natural next step beyond zero-ablation.
Direct logit attribution — decompose a move's logit into per-(head/MLP) contributions along the clean path, not just ablation deltas (the two disagree, and showing both is instructive).
Concept discovery & aggregation

Batch/dataset mode — run a probe over a PGN or a tactics/opening suite and aggregate: which heads consistently carry captures, checks, castling, pins? Turns single-position observations into testable claims.
Max-activating positions per head — a "what does this head do?" dashboard: feed many boards, show the square-pairs/motifs that most excite the selected head (the attention analogue of a neuron feature card).
Linear probes on the residual stream — train tiny probes for board-state features (is square X attacked? king safety? piece identity) and plot accuracy by depth — this is exactly how the "chess world-model" results were shown, and it pairs beautifully with your residual film.
Unique-to-Maia angle (very publishable)

Skill-delta view — you already condition on ELO; add a mode that highlights which heads/features shift most between, say, 600 and 2800 on the same position. Almost no other model lets you do interpretability across a skill axis — lean into it.
Happy to implement any of these — the URL-state sharing and activation patching are the two I'd prioritize if you want maximum community traction. Want me to start on one?