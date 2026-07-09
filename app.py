"""
Chessformer (Maia 3) interpretability app — launcher.

Play a transformer-based chess bot (Maia-3) trained to mimic human play and watch
its move policy, its attention (regular self-attention vs the unique geometric
GAB), and how its residual stream evolves with depth. Drag the ELO slider to
re-evaluate a position at different skill levels (e.g. at very low ELO, King and
Queen vs King comes out to roughly 75% draw).

The code is split into a few small, single-purpose modules:

  engine.py  — MaiaEngine: the interp core (model + hooks + analysis). No UI deps,
               so it imports cleanly into Colab/Jupyter. Start here for interp work.
  bridge.py  — MaiaApi: game state + the methods the UI calls (window.pywebview.api).
  ui.py      — INDEX_HTML: the whole interface (HTML + CSS + JS) as one string.
  app.py     — this file: opens the native window.

Run:  python app.py        (see README.md)
"""
import sys

from bridge import MaiaApi
from ui import INDEX_HTML


def main():
    try:
        import webview  # pywebview
    except ImportError:
        sys.exit("pywebview is not installed.  Run:  pip install -r requirements.txt")
    api = MaiaApi()
    webview.create_window(
        "Chessformer (Maia 3) Interpretability App",
        html=INDEX_HTML,
        js_api=api,
        width=1400, height=920, min_size=(1240, 840),
        background_color="#0e1014",
    )
    webview.start()


if __name__ == "__main__":
    main()
