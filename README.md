## Chessformer_lens
##### A toolkit + visualizer designed for mech interp enthusiasts working with chess models that treat each square as a token—which allows for beautiful board readable attention patterns.


This project was spurred by the surprising lack of infrastructure for analyzing chess engines. Chessformer_lens is both an interface for interactive engine visualization and rigorous mechanistic interpretability functionality—especially inspired by Neel Nanda's fantastic transformer_lens library.

(The Maia-3 engine interpreter is completed; Leela will be completed soon; and eventually other tokenization schemes will be tackled).
<-------------------------------------------------------------------->

![Hero Image](Screenshots/Screenshot1.png)

Chess is unusually good for AI interpretability because it requires complex and structured reasoning in a compact domain, and it has a hiearchy of human concepts (squares -> threats -> tactics -> strategy) which can provide deep insight into how AI carves features.

Transformer based chess models ("chessformer") are a good lens for studying LLMs because outputs have a clear ground source for the evaluation of a move and so avoid the vagueness of language, because attention functions with clearly interpretable roles on a chess board, and because informative chessformer results are straightforward to scale and investigate in LLMs.

<-------------------------------------------------------------------->

This repo's core is **one engine** with **three frontends**:
`engine.py` is the interp core (model + hooks + logit lens + head ablation + list of values through depth). It does all of the analysis for:
- `app.py`
- `interp_plot.py`
- `interp_widget.py`

Users are encouraged to read the user guides for each of these modules which can be found at the top of the respective scripts.


## Quickstart in colab:
<-------------------------------------------------------------------->
```python
!pip install -q git+https://github.com/CSSLab/maia3
!pip install -q "chessformer_lens[plot] @ git+https://github.com/chessformer-lens/chessformer_lens"

#Interact with the attention widget
import chess
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_widget as iw

eng = MaiaEngine()                      
board = "...fill in FEN for position..."
board = chess.Board(board)                   

iw.attention_widget(eng, board, 1500,layer=4,head=3) # click between different layers and heads in the widget
```
<-------------------------------------------------------------------->
## The app

To me, the app is the pièce de résistance and usage should be rather intuitive, nonetheless, a detailed guide can be found in `app.py`.


Play a transformer-based chess bot ("chessformer") and watch
its move policy, its attention (both semantic QKᵀ and unique geometric GAB), and its residual stream 
evolve with model depth. The model loads on a background thread so the window opens instantly.
<-------------------------------------------------------------------->

### Run locally
`chessformer_lens` is a normal Python package, but it needs one thing installed
first: `engine.py` imports `maia3`: the Maia-3 model code, which is **not on
PyPI**, so it can't be declared as a dependency and has to come from GitHub.
The install also installs torch, numpy, python-chess, huggingface-hub and
ipython for `engine.py` and the widgets. 

```bash
pip install git+https://github.com/CSSLab/maia3
pip install "chessformer_lens[all] @ git+https://github.com/chessformer-lens/chessformer_lens"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#Various ways to launch, note that higher parameter models have 16 and 32 heads per layer, and 512 and 1024 dim of the model (d_model):
chessformer-lens
chessformer-lens 23m               
chessformer-lens --model maia3-79m 
MAIA3_ALIAS=5m chessformer-lens   
```

<!-- understand about set `MAIA3_DEVICE=mps` (Apple Silicon) or run on
CUDA to speed higher parameter models up. Model weights: <https://huggingface.co/UofTCSSLab> -->

### Example usage directly from engine
See the residual stream film and move microscope for the famous Paul Morphy "Opera Game"
```python
import chess
from chessformer_lens import MaiaEngine
import chessformer_lens.interp_plot as ip

#opera game FEN right at legendary queen sacrifice 
fen = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16" 
import matplotlib.pyplot as plt

eng = MaiaEngine('23m')
board = chess.Board(fen)                   
move = eng.to_move(board,'Qb8+')
ip.plot_position(eng, board,2600)
plt.show()
print("\n\n")
ip.plot_residual_film(eng, board, 2600) #exceptional Elo rating to match Morphy's skill
plt.show()
print("\n\n")
ip.plot_move_microscope(eng, board, 2600, move)     # one move's depth curve + carrier heads
plt.show()
plt.close()
```

### Code layout: all found within the chessformer_lens package
- `__init__.py` — the package surface: `MaiaEngine`, `build_cfg`, `pick_device`, `attention_widget`, `gab_widget`, `__version__`.
- `engine.py` — `MaiaEngine`: the interp core (model + hooks + logit lens + head ablation + list of values through depth). No UI deps; imports cleanly in a notebook.
- `interp_plot.py` — matplotlib figures that the app uses
- `interp_widget.py` — the app's two interactive panels as a self-contained notebook cell or standalone HTML page
- `piece_art.py` — draws `pieces.py`'s cburnett pieces in matplotlib from PNGs
- `bridge.py` — the game state + the JSON API the UI calls.
- `ui.py` — the whole interface (HTML/CSS/JS) as one string including guide to change template.
- `pieces.py` — SVG piece set as data URIs.
- `app.py` — app launcher (native window via pywebview); the `chessformer-lens` command is its `main()`.


### Notes

**Please don't hesitate to give me feedback by email or at davidlitman.com**. I intend for this to be a useful and intuitive tool for the community.


Maia-3 comes from:
Chessformer / Maia-3 (Monroe et al., ICLR 2026).
Model weights: <https://huggingface.co/UofTCSSLab/Maia3-5M>


### Citing
If `chessformer_lens` contributes to published work, a citation is very appreciated and helps others find it! Please see
`CITATION.cff` and GitHub's "Cite this repository" button.
