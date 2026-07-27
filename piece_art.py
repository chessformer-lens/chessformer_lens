"""
The app's chess pieces, drawn in matplotlib.

`pieces.py` carries the artwork the app uses — python-chess's built-in "cburnett"
set, as SVG data URIs. A figure should use that same artwork, not a lookalike:
unicode chess glyphs are a different typeface with different proportions and no
interior detail, which is why they read as shoddy next to the app.

Matplotlib cannot read SVG (its SVG support is write-only) and the rasterisers
that could — cairosvg, svglib — are native-code dependencies that would have to
install cleanly on every Colab. So the pieces were rasterised once, offline, into
`piece_art/` at 256px on transparent backgrounds; drawing is then just `imshow`.
`piece_art/regenerate.py` rebuilds them from `pieces.py` if the set ever changes.

Pieces keep their own colours (white with a black outline, black), so a board
here looks like a board there.

    from piece_art import draw_piece
    draw_piece(ax, "N", x=4, y=0, size=.88)     # one unit square per board square
"""
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent / "piece_art"


@functools.lru_cache(maxsize=None)
def piece_image(symbol: str):
    """RGBA array for one piece symbol ('P'…'k'; uppercase = white)."""
    name = ("w" if symbol.isupper() else "b") + symbol.lower()
    return plt.imread(ART / f"{name}.png")


def draw_piece(ax, symbol: str, x: float, y: float, *, size: float = .88,
               zorder: int = 4, alpha: float = 1.0):
    """Draw one cburnett piece centred on (x, y) in data coordinates, occupying
    `size` of a one-unit square — matching the app's miniboards, whose
    `.miniboard img.pc` is 88% (the main board's `.sq img.pc` is 87%).

    `zorder` and `alpha` pass through to the imshow, so a caller can stack a
    piece over a heat square or dim one that's out of attribution."""
    half = size / 2
    im = ax.imshow(piece_image(symbol), zorder=zorder, alpha=alpha,
                   extent=(x - half, x + half, y - half, y + half),
                   interpolation="antialiased")
    im.set_clip_on(False)
