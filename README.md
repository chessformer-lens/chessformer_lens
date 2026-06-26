# Chessformer (Maia 3) interpretability app.


Play a transformer based chess bot (Maia-3) trained to mimic human play and watch its move policy, 
its attention (regular self attention vs unique geometric GAB), 
and how its residual stream evolves with depth. 

Drag the ELO slider to re-evaluate a position at
different skill levels (e.g. eval at very low ELO for King and Queen vs King comes out to
roughly 75% chance draw).

I'm hoping to get the transcoders to analyze too soon.

<img width="1440" height="900" alt="Screenshot 2026-06-25 at 12 16 03 PM" src="https://github.com/user-attachments/assets/ff7de900-cf28-4e0c-8661-0aac43409325" />

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first launch downloads the Maia3-5M transformer weights (~20 MB) from
Hugging Face and a native window opens (no browser needed).

Requires **Python 3.10+** and an internet connection on first run.

Built on Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>
