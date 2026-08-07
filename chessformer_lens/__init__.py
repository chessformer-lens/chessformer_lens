"""chessformer_lens — an interpretability lens and visualizer for square-token chess transformers.

Reads the internals of chess models that treat the board as 64 square tokens with
a (from, to) policy head. The Maia-3 backend is complete; Leela Lc0-BT is next.

Play a transformer-based chess bot trained to mimic human play and watch its move
policy, its attention (semantic QKᵀ vs the geometric GAB bias), and how its
residual stream evolves with depth — live. Decode the running residual with a
logit lens, ablate any head and see the causal effect, take the GAB apart into the
64 static templates every layer shares, and drag the ELO slider to re-evaluate the
same position at different skill levels.

One engine, three frontends:

    from chessformer_lens import MaiaEngine            # the interp core, no UI deps
    from chessformer_lens import interp_plot as ip     # read-once views as figures
    from chessformer_lens import interp_widget as iw   # the two live panels, in a notebook

    chessformer-lens                                   # the native app

`interp_plot` needs matplotlib and the app needs pywebview; both are extras
(`pip install 'chessformer_lens[plot,app]'`). Importing this package pulls in
neither — the names below resolve on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

try:  # populated once the distribution is installed
    from importlib.metadata import version

    __version__ = version("chessformer-lens")
except Exception:  # running from a source checkout without an install
    __version__ = "0.1.0"

# Attribute -> the submodule it lives in. Resolved lazily so that `import
# chessformer_lens` (and `__version__`) costs nothing: engine.py pulls in torch
# and maia3, interp_plot pulls in matplotlib, app pulls in pywebview.
_LAZY = {
    "MaiaEngine": "chessformer_lens.engine",
    "build_cfg": "chessformer_lens.engine",
    "pick_device": "chessformer_lens.engine",
    "attention_widget": "chessformer_lens.interp_widget",
    "gab_widget": "chessformer_lens.interp_widget",
}

_SUBMODULES = ("engine", "interp_plot", "interp_widget", "bridge", "ui", "app",
               "pieces", "piece_art")


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    if name in _SUBMODULES:
        return import_module(f"chessformer_lens.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*__all__, *_SUBMODULES})


if TYPE_CHECKING:  # so type checkers and IDEs still see the real symbols
    from .engine import MaiaEngine, build_cfg, pick_device
    from .interp_widget import attention_widget, gab_widget

__all__ = ["MaiaEngine", "build_cfg", "pick_device",
           "attention_widget", "gab_widget", "__version__"]
