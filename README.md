# Maia-3 · play & probe

A tiny desktop app to play chess against **Maia-3 (5M)** — a human-like neural
network — and watch, live, what it's doing inside: its full move policy, its
attention (the semantic **QKᵀ** vs. the geometric **GAB** bias), and how its
internal representation evolves layer by layer. Drag the **ELO slider** to
re-evaluate the same position at different skill levels.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

That's it. The first launch downloads the Maia3-5M weights (~20 MB) from
Hugging Face and a native window opens (no browser needed).

Requires **Python 3.10+** and an internet connection on first run.

Built on Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>
