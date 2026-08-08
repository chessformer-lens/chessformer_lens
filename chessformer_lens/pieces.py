"""
pieces.py — SVG chess pieces as data URIs. The one source of piece artwork.

The artwork is python-chess's built-in set (chess.svg.PIECES) by Colin M.L. Burnett. 
"""
import base64

import chess.svg


def _uri(fragment: str) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45">'
           + fragment + "</svg>")
    return ("data:image/svg+xml;base64,"
            + base64.b64encode(svg.encode("utf-8")).decode("ascii"))


PIECE_URI = {sym: _uri(frag) for sym, frag in chess.svg.PIECES.items()}
