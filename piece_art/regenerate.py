"""Rebuild the piece PNGs from `pieces.py`. Run only if the piece set changes.

Needs an SVG rasteriser, which is exactly why the app itself does not:
    pip install cairosvg && python3 piece_art/regenerate.py
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cairosvg

from pieces import PIECE_URI

OUT = Path(__file__).resolve().parent
PX = 256

for sym, uri in PIECE_URI.items():
    svg = base64.b64decode(uri.split(",", 1)[1])
    name = ("w" if sym.isupper() else "b") + sym.lower()
    cairosvg.svg2png(bytestring=svg, write_to=str(OUT / f"{name}.png"),
                     output_width=PX, output_height=PX)
    print(name)
