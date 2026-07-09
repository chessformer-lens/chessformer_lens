"""
pieces.py — SVG chess pieces as data URIs for the app's <img> tags.

The artwork is python-chess's built-in set (chess.svg.PIECES), i.e. the
"cburnett" pieces by Colin M.L. Burnett (GFDL/BSD/GPL — same license terms as
they ship with python-chess). Keys are piece symbols ('P','n',… — uppercase =
white); values are self-contained data:image/svg+xml URIs, so the page needs
no network and no files on disk.
"""
import base64

import chess.svg


def _uri(fragment: str) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45">'
           + fragment + "</svg>")
    return ("data:image/svg+xml;base64,"
            + base64.b64encode(svg.encode("utf-8")).decode("ascii"))


PIECE_URI = {sym: _uri(frag) for sym, frag in chess.svg.PIECES.items()}
