# chessformer_lens

#### A toolkit + visualizer library  for mechanistic interpretability of transformer based chess models.

`pip install chessformer_lens`, then `import chessformer_lens` or launch `chessformer_lens`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877655.svg)](https://doi.org/10.5281/zenodo.21877655)

![The app: board, policy, live attention and the GAB decomposition](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/Screenshot7.png)

---
## Case study: Knight Fork Carrier Head
**Download chessformer_lens locally:**


Pip install:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/CSSLab/maia3
pip install "chessformer_lens[all]"
```

Clone install:
```bash
git clone https://github.com/chessformer-lens/chessformer_lens
cd chessformer_lens
pip install -r requirements.txt
```

Quick interpretability demo (n=10): **attention layer 5 head 5 seems to be the carrier head for knight forks**

In python:
```python
# Gather ten positions with knights forking King and Queen, then ablate every
# head in the model to see which one is most causally linked to the output.
import chess
import matplotlib.pyplot as plt
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_plot as ip

eng = MaiaEngine()

knight_forks = [
{"fen": '1r2k3/3qn3/3p4/p2P1Pp1/PpP3Np/1P3Q1P/3K1PP1/7R w - - 0 37', "move":'g4f6'},
{"fen": '5b2/6p1/3p2k1/2p3pn/7r/8/P2NKPQ1/R6R b - - 2 25', "move":'h5f4'},
{"fen": '6k1/5p1p/p3p1p1/1p2n3/1P1q4/P2p2P1/3Q1N1P/6K1 b - - 2 37', "move":'e5f3'},
{"fen": 'rn1q1knr/pb2p1b1/6p1/5pN1/1pPP3p/1P6/P1B2PPP/1RQ2RK1 w - - 2 20', "move":'g5e6'},
{"fen": '5r1k/3nrBp1/b1pp1n1p/p3q2P/Pp2PN2/3PB1Q1/1PP3P1/2KR3R w - - 1 22', "move":'f4g6'},
{"fen": '6k1/1pq2ppp/p7/3p4/1P1Nn3/P2QPP2/1B4Pb/7K b - - 1 26', "move":'e4f2'},
{"fen": '4r1k1/6b1/3p2pp/3Pnp2/4N2Q/6P1/P1q2P1P/4R1K1 b - - 0 26',"move": 'e5f3'},
{"fen": '5rk1/pp5p/2p1p1p1/6N1/3PQBP1/2q4P/P7/3n3K b - - 1 25', "move":'d1f2'},
{"fen": 'r2q1rk1/ppp3p1/3ppn1B/2b1p3/3nP3/3P2QN/PPP2PPP/RN3RK1 b - - 2 11',"move": 'd4e2'},
{"fen": 'r1b2rk1/1pp2p1p/4p1p1/2PqP3/p1nPN2B/P1PQ4/6PP/R4RK1 w - - 2 21',"move": 'e4f6'}]

for pos in knight_forks:
    ip.plot_move_report(eng, chess.Board(pos["fen"]), 2400, pos["move"])
    plt.show()
```

![A knight-fork position's move report](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/screenshot10.png)![A second knight-fork move report](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/screenshot11.png)

---

There is wonderful prior chess-interp work, for instance: McGrath et al. on AlphaZero concepts, Jenner et al. on lookahead in Leela, Karvonen on chess-GPT—but there is no general infrastructure for rigorous interpretability work or interactive visualization for these models.
 
Chessformer_lens is built to do both, especially inspired by Neel Nanda's transformer_lens library.

Chess is unusually suited for AI interpretability because it requires complex and structured reasoning in a compact domain, and it has a hierarchy of human concepts (squares -> threats -> tactics -> strategy) which can provide deep insight into how AI carves features.

Transformer based chess models ("chessformer") are a good lens for studying LLMs because outputs have a clear ground source for the evaluation of a move and so avoid the vagueness of language, because attention functions with clearly interpretable roles on a chess board, and because informative chessformer results are straightforward to scale and investigate in LLMs.

---

This repo's core is **one engine** with **three frontends**:
`engine.py` is the interp core (model + hooks + logit lens + head ablation + list of values through depth). It does all of the analysis for:
- `app.py`
- `interp_plot.py`
- `interp_widget.py`

Users are encouraged to read the user guides for each of these modules which can be found at the top of the respective scripts.
![A move's logit curve across depth, beside its carrier-head grid](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/Screenshot9.png)
*interp_plot.py functions **plot_logit_curve** and **plot_carrier_heads***

---
## Quickstart in colab or notebook:

Recall that FEN is the modern notation for a chess position
```python
!pip install -q git+https://github.com/CSSLab/maia3
!pip install -q "chessformer_lens[plot]"

import chess
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_widget as iw

eng = MaiaEngine()          
# Opera Game FEN right at legendary queen sacrifice 
fen = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16" 
# [or insert another FEN string]

try:
    board = chess.Board(fen)
except ValueError as e:
    print(f"Invalid FEN: {e}")   

#Interact with the attention widget
#click between different layers and heads in the widget
iw.attention_widget(eng, board, 1500,layer=4,head=3) 
```

---
## The app



To me, the app is the pièce de résistance and usage should be rather intuitive, nonetheless, a detailed guide can be found in [`app_README.md`](https://github.com/chessformer-lens/chessformer_lens/blob/main/chessformer_lens/app_README.md).


Play a transformer-based chess bot ("chessformer") and watch
its move policy, its attention (both semantic QKᵀ and unique geometric GAB), and its residual stream 
evolve with model depth. The model loads on a background thread so the window opens instantly.


```bash
#Various ways to launch, note that higher parameter models have 16 and 32 heads per layer, and 512 and 1024 dim of the model (d_model):
chessformer_lens
chessformer_lens 23m               
chessformer_lens --model maia3-79m 
MAIA3_ALIAS=5m chessformer_lens   
```


<!-- understand about set `MAIA3_DEVICE=mps` (Apple Silicon) or run on
CUDA to speed higher parameter models up. Model weights: <https://huggingface.co/UofTCSSLab> -->

---

### Code layout: all found within the chessformer_lens package
- `__init__.py` — the package surface: `MaiaEngine`, `build_cfg`, `pick_device`, `attention_widget`, `gab_widget`, `__version__`.
- `engine.py` — `MaiaEngine`: the interp core (model + hooks + logit lens + head ablation + list of values through depth). No UI deps; imports cleanly in a notebook.
- `interp_plot.py` — matplotlib figures that the app uses
- `interp_widget.py` — the app's two interactive panels as a self-contained notebook cell or standalone HTML page
- `piece_art.py` — draws `pieces.py`'s cburnett pieces in matplotlib from PNGs
- `bridge.py` — the game state + the JSON API the UI calls.
- `ui.py` — the whole interface (HTML/CSS/JS) as one string including guide to change template.
- `pieces.py` — SVG piece set as data URIs.
- `app.py` — app launcher (native window via pywebview); the `chessformer_lens` command is its `main()`.

---
### Notes


Maia-3 comes from:
Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>


The **Maia-3** model interpreter is completed; **Leela** will be completed soon. Both of these treat each square as a token—which allows for beautiful board readable attention patterns. Eventually other tokenization schemes will be tackled.


Please don't hesitate to give me feedback or thoughts by email or at **davidlitman.com**. I hope for this to be a useful and intuitive tool for the community!

---
### Citing
If `chessformer_lens` contributes to published work, a citation is very appreciated and helps others find it! Use GitHub's "Cite this repository" button, or DOI [10.5281/zenodo.21877655](https://doi.org/10.5281/zenodo.21877655).
