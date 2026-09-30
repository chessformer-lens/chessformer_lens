"""
App Guide: see app_guide.md

From repo run: python -m chessformer_lens.app
"""
import argparse
import os
import sys

from .bridge import MaiaApi
from .ui import INDEX_HTML


def resolve_alias():
    """Pick the engine from the CLI. Engines available: 
            bt4, leela-bt4, maia3-3m, maia3-5m, maia3-23m, maia3-79m
    or HF repo/URL or a path to an lc0 network (.pb.gz).
    Defaults to maia3-5m."""
    ap = argparse.ArgumentParser(description="Chessformer interpretability app")
    ap.add_argument("model", nargs="?", default=None,
                    help="engine alias: a Maia-3 size (3m, 5m, 23m, 79m) or HF repo/URL, "
                         "bt4 / leela-bt4, or a path to an lc0 network (.pb.gz); "
                         "overrides $CHESSFORMER_MODEL")
    ap.add_argument("--model", dest="model_opt", default=None,
                    help="same as the positional argument")
    args = ap.parse_args()

    alias = (args.model_opt or args.model or os.environ.get("CHESSFORMER_MODEL")
             or os.environ.get("MAIA3_ALIAS") or "maia3-5m")

    # Validate now (no weights load) so a bad name is caught before the window opens.
    from .engine import resolve_engine
    try:
        resolve_engine(alias)
    except ValueError as exc:
        sys.exit(str(exc))
    return alias


def main():
    alias = resolve_alias()
    try:
        import webview  # pywebview
    except ImportError:
        sys.exit("pywebview is not installed.  Run:  pip install --upgrade chessformer_lens")
    api = MaiaApi(alias=alias)
    # Size the window to the screen: the board is sized from the window's height
    # (the side panels run 1.5in under it and the drawer peeks below that), so a
    # fixed 880px window left a small board on a large display.
    width, height = 1560, 880
    try:
        scr = webview.screens[0]
        width = max(1400, min(1880, scr.width - 40))
        height = max(780, scr.height - 90)       # menu bar + dock
    except Exception:
        pass
    webview.create_window(
        "Chessformer Interpretability App",
        html=INDEX_HTML,
        js_api=api,
        width=width, height=height, min_size=(1400, 780),
        background_color="#0e1014",   # matches ui.py's --bg
    )
    try:
        webview.start()
    except Exception as exc:
        # pywebview renders through a system webview that pip cannot install
        # "Alternatively, skip the app and use the notebook panels: "
        # "chessformer_lens.interp_widget.attention_widget()."
        if sys.platform.startswith("linux"):
            sys.exit(
                f"Could not open the app window: {exc}\n\n")
        raise


if __name__ == "__main__":
    main()
