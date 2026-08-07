"""
Chessformer (Maia 3) interpretability app — launcher.

Play a transformer-based chess bot (Maia 3) trained to mimic human play and watch
its move policy, its attention (semantic QKᵀ, the unique geometric GAB, and
the head's raw logits), and how its residual stream evolves with
depth. The GAB itself is taken apart live in its own drawer: the static
square-pair template bank and the generated smolgen mixing coefficients that
combine it into each head's bias. Drag the ELO slider to
re-evaluate a position at different skill levels (e.g. at very low ELO, King and
Queen vs King comes out to roughly 75% draw).

The app itself is four small, single-purpose modules:

  engine.py  — MaiaEngine: the interp core (model + hooks + analysis). No UI deps,
               so it imports cleanly into Colab/Jupyter. Start here for interp work.
  bridge.py  — MaiaApi: the JSON API exposed as window.pywebview.api, plus game state.
  ui.py      — INDEX_HTML: the whole interface (HTML + CSS + JS) as one string.
  app.py     — this file: opens the native window.

Alongside it, the same engine drives a notebook/figure layer that needs no UI:
interp_plot.py (the read-once views as matplotlib figures), interp_widget.py (the
two interactive panels as a notebook cell), and pieces.py / piece_art.py for the
piece artwork both of them share with the app. See README.md.

Run:  python app.py                 # default model (Maia3 5M)
      python app.py 23m             # any built-in alias: 3m / 5m / 23m / 79m
      python app.py --model maia3-79m
      MAIA3_ALIAS=23m python app.py # env var still works

The GUI adapts to whatever loads (block/head/dim counts, GAB template bank), so
the same interface drives every model size.
"""
import argparse
import os
import sys

from bridge import MaiaApi
from ui import INDEX_HTML


def resolve_alias():
    """Pick the model from the CLI (positional or --model), else MAIA3_ALIAS,
    else the 5M default. Validated against the registry so a typo fails fast
    with the list of aliases instead of a mid-load traceback."""
    ap = argparse.ArgumentParser(description="Chessformer (Maia 3) interpretability app")
    ap.add_argument("model", nargs="?", default=None,
                    help="model alias or HF repo/URL (e.g. 3m, 5m, 23m, 79m); "
                         "overrides $MAIA3_ALIAS")
    ap.add_argument("--model", dest="model_opt", default=None,
                    help="same as the positional argument")
    args = ap.parse_args()

    alias = args.model_opt or args.model or os.environ.get("MAIA3_ALIAS") or "maia3-5m"

    # Validate now (no download) so a bad name is caught before the window opens.
    from maia3.model_registry import resolve_model_spec, ModelResolutionError, format_model_list
    try:
        resolve_model_spec(alias)
    except ModelResolutionError as exc:
        sys.exit(f"Unknown model {alias!r}: {exc}\n\n{format_model_list()}")
    return alias


def main():
    alias = resolve_alias()
    try:
        import webview  # pywebview
    except ImportError:
        sys.exit("pywebview is not installed.  Run:  pip install -r requirements.txt")
    api = MaiaApi(alias=alias)
    webview.create_window(
        "Chessformer (Maia 3) Interpretability App",
        html=INDEX_HTML,
        js_api=api,
        width=1400, height=840, min_size=(1360, 780),
        background_color="#0e1014",   # matches ui.py's --bg
    )
    webview.start()


if __name__ == "__main__":
    main()
