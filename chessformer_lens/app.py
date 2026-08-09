"""
App Guide:

**Play or Set up a Position**
![alt text](Screenshots/Screenshot7.png)
Click to move your pieces. Select an Elo for the engine (Maia-3 is created to mimic how HUMANS play at that strength, with all of our innate biases)
and use `New game`, and a dropdown:`You play White` / `You play Black` / `Set up position`.
It is also possible to `paste a FEN to load` a position with the `Load` button. 

In setup mode both sides are yours; clicks move pieces ignoring legality, and clicking the same square twice deletes the piece. 
Select a color to continue as it from that position.

The last move is indicated by a gold arrow, legal moves for a selected piece are indicated by dots, captures are drawn as rings. The games
moves are recorded in SAN notation.

**Read the Position**

At the top in the center there is a `Win / Draw / Loss · side to move` stacked bar. 
Under it is the `Maia rating (self_elo)` slider: 600-2800, step 25, default 1500; Dragging reevaluates the same position. 
Under that is the scrollable ranked list: `Policy over N legal moves`. 

There is a `compare with a second rating` checkbox which reveals a `second rating` slider
(default 1100). Setting it makes the policy rows become paired blue/green bars showing the compared policy and evaluation. This second rating does not affect the attention or GAB or residual panel app features.

**Take the Model Apart**
Get the `Live attention · this position`, with `Layer` and `Head` chip rows.
Click a square to set the query for the three boards, labeled:
-`semantic attention (QKᵀ)`
-`geometric attention (GAB)`
-`final head attention matrix (scaled softmax(QKᵀ + GAB))`
`Ablate this head` (its note: `removes its exact residual write`) gives the top 8 moves by |Δp|.
![alt text](Screenshots/Screenshot2.png)
Unique to the app and Maia-3, hover over any attention square and the GAB drawer decomposes that square pair live, with every head clickable to open that template:
![alt text](Screenshots/Screenshot8.png)
 
**The Three Drawers**
![alt text](Screenshots/Screenshot4.png)
![alt text](Screenshots/Screenshot5.png)+![alt text](Screenshots/Screenshot3.png)
![alt text](Screenshots/Screenshot6.png)
One open at a time, each peeking at the bottom with a `▲ pull up` grip, `Escape` closes.

· `Residual stream across depth · this position` — creates a filmstrip of mini
  boards, one per readout point, top border color coded by writer kind, **[SAY MORE HERE]**
  heat = per-square ‖Δ‖, logit-lens top move in miniature display on top. Opened by the
  `Watch residual stream` button.

· `Move microscope (and Carrier Heads)` — click up to 4 policy moves to overlay their depth curves (essentially logit lens narrowed to one policy); marker shows when a move sustains rank-1; per-dot hover
ex: `b3 mlp · logit 4.21 · p 38.2% · rank 1/31`.
`carrier heads` runs (layers x heads) forward passes with different ablated heads and record: `Δlogit = ablated − clean`.Hover for exact values. The largest Δlogit head is ringed red. The final layer is dimmed and striped, `excluded from carrier attribution`, because it writes straight to the logits and muddles meaningful results from earlier layers. 

· `How L[i]H[j] GAB is generated` — see the network's generated linear combination of each template #0–63. `click to inspect`, and under
see `template vocabulary · the 64 static stencils (row = query sq, col = key sq)` as a gallery.

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
    webview.start()


if __name__ == "__main__":
    main()
