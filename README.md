# Chessformer (Maia 3) mechanistic interpretability app.

<img width="1440" height="900" alt="Screenshot 2026-07-09 at 6 09 56 PM" src="https://github.com/user-attachments/assets/42218edf-d8b2-4d89-81d3-c750837d3df3" />

Play a transformer based chess bot (Maia 3) trained to mimic human play and watch its move policy, 
its attention (self attention vs unique geometric attention), 
and how its residual stream evolves with depth LIVE. 

Visualize mechanistic interpretability tools like logit lens and head ablations.

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
