"""
App Guide: see app_README.md
"""
import argparse
import os
import sys

from .bridge import MaiaApi
from .ui import INDEX_HTML


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
        sys.exit("pywebview is not installed.  Run:  pip install 'chessformer_lens[app]'")
    api = MaiaApi(alias=alias)
    webview.create_window(
        "Chessformer (Maia 3) Interpretability App",
        html=INDEX_HTML,
        js_api=api,
        width=1400, height=840, min_size=(1360, 780),
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
