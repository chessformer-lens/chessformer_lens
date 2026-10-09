"""
App Guide: see app_guide.md

From repo run: python -m chessformer_lens.app
"""
import argparse
import os
import sys

from .bridge import LensApi
from .ui import INDEX_HTML


def resolve_alias():
    """The engine named on the command line (or $CHESSFORMER_MODEL), or None:
    there is no default model, and without one the window opens on a model
    picker. `--list` prints every registered engine and exits."""
    ap = argparse.ArgumentParser(description="Chessformer interpretability app")
    ap.add_argument("model", nargs="?", default=None,
                    help="engine alias, e.g. bt4 or maia3-23m, or a path to an lc0 "
                         "network (.pb.gz); see --list. Overrides $CHESSFORMER_MODEL. "
                         "Without one the app opens a model picker")
    ap.add_argument("--model", dest="model_opt", default=None,
                    help="same as the positional argument")
    ap.add_argument("--list", action="store_true",
                    help="print every registered engine and its aliases, then exit")
    args = ap.parse_args()

    from .engine import format_engine_list, resolve_engine
    if args.list:
        print(format_engine_list())
        sys.exit(0)

    alias = (args.model_opt or args.model or os.environ.get("CHESSFORMER_MODEL")
             or os.environ.get("MAIA3_ALIAS") or None)
    if alias is None:
        return None
    # Validate now (no weights load) so a bad name is caught before the window opens.
    try:
        resolve_engine(alias)
    except (ValueError, ImportError) as exc:
        sys.exit(str(exc))
    return alias


def main():
    alias = resolve_alias()
    try:
        import webview  # pywebview
    except ImportError:
        sys.exit("pywebview is not installed.  Run:  pip install --upgrade chessformer_lens")
    api = LensApi(alias=alias)
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
