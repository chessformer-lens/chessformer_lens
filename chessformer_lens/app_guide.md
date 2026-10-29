# App Guide 

**Background, if chess transformers are new to you**
Each model here is a neural network that looks at one chess position and, in a single calculation, outputs a probability for every legal move (the `policy`) it can play and a guess at the current evaluation (`Win / Draw / Loss`). Maia-3 is trained on human games, so at a given `Rating` it predicts what a human of that strength would play, blunders included. Leela's nets are trained to play at a world competitive level. Other models may differ but each has a transformer neural network backing it, the same architecture that backs modern LLMs like Claude.

Inside, the position is 64 tokens, one per square. They pass through a stack of layers, and each layer has two parts: `attention heads`, where every square gathers information from the other squares (like a knight's square looking at the squares it attacks), and an `MLP` of many `neurons` that is trained to perform some sort of calculations per token. Each part adds its contribution to a running vector per square, the `residual stream`, until at the end a policy head reads the moves off that stream. A move's `logit` is its raw score before the scores are turned into probabilities.

This app takes that whole process apart and let's you play around with the internals. If you would just like to play against a specific model without the internals, click `Simple Play`.

**Play or Set up a Position**
Click to move your pieces. Select an Elo for the engine (Maia-3 is created to mimic how HUMANS play at that strength, with all of our innate biases)
and use `New game`, and a dropdown:`You play White` / `You play Black` / `Set up position` / `Random position`.
It is also possible to `paste a FEN to load` a position with the `Load` button. 

`Random position` generates a position through a random walk of the engine's policies. It is ideal for testing mechanistic hypothesis on realistic positions. 

In setup mode both sides are yours; clicks move pieces ignoring legality.
Double clicking a piece removes it. Double clicking an empty square cycles through
different pieces to add to the position.
Select a color to continue as it from that position.

The last move is indicated by a gold arrow, legal moves for a selected piece are indicated by dots, captures are drawn as rings. The games
moves are recorded in SAN notation.

**Read the Position**

At the top in the center there is a `Win / Draw / Loss · side to move` stacked bar (`Expected score · side to move` for models with a single value output). 
Under it is the `Rating (self_elo)` slider: 600-2800, step 25, default 1500; Dragging reevaluates the same position. 
Under that is the scrollable ranked list: `Policy · N legal moves`: every legal move gets a row (`Win probability · N legal moves` on DeepMind's searchless models, which state a win probability per move and play the best one; they have no logit lens, so the depth curves in "Analyze one move" do not apply to them). 


![app view](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/SS11.png)
*App view: board, evaluation, policy, live attention, live neurons*

**Take the Model Apart**
Get the `Live attention · this position`, with `Layer` and `Head` chip rows.
Select a square on the right to set the query for the three boards, labeled:
-`semantic attention (QKᵀ)`
-`geometric attention (GAB)`
-`final head attention matrix (scaled softmax(QKᵀ + GAB))`
`Ablate / steer this head` redraws the policy list with the ablated pass in red over the clean pass in blue, re-sorted by
signed `Δ = p(ablated) − p(clean)` — the moves the head was suppressing rise to the top, the ones it was holding up sink.
While the toggle is on, a slider beside it sets the head's strength instead of removing it outright (activation steering): 0 removes the write, 1 is the clean model, 2 doubles it, −1 flips it — a knight-move head at ×3 makes the model see forks everywhere. The Neurons panel has the same slider for one neuron.

The `Neurons` panel to the right does the same for one MLP neuron. A network diagram shows one column of dots per MLP layer. Click a column to pick the layer, a dot to pick the neuron. The dots in the selected layer are its most active units on this position (by peak activation over the squares, the same score the many-positions tab ranks by). Below, the neuron's activation on every square, and `Ablate / steer this neuron` removes its exact write on every square.

For models whose tokens aren't board squares, the attention boards and the neuron's activation board are replaced by a "not available yet" notice; the chips and both ablation toggles still work.



**Analyze One Move**
![Analyze one move](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/SS12.png)
*Analyze one move: `Logit lens`, `causal heads`, `causal neurons`*



One drawer, `Analyze one move`, docked at the bottom with a `▲ pull up` grip. The side panels run 1.5 inches below the board; when the drawer opens it rises an inch above the board's bottom edge and the whole layout above scales down to make room, so nothing is ever covered. `Escape` closes it. Click up to 4 policy moves to open it. Its three sections sit side by side and all describe the primary move (click a move chip to change it):

`Logit lens` — the move's logit at every readout point of the forward pass (the logit lens narrowed to one policy entry); a marker shows from where the move sustains rank 1; per-dot hover, ex: `b3 mlp · logit 4.21 · p 38.2% · rank 1/31`.

`Causal heads` — (layers × heads) forward passes, each with one head's exact residual write removed, recording `Δlogit = ablated − clean`. Hover for exact values; the largest-|Δ| head is ringed red; click a cell to open that head in the attention panel. The final layer is dimmed and striped, excluded from carrier attribution, because it writes straight to the logits and muddles the earlier structure.

`Causal neurons` — the same question at neuron grain. Ablating every MLP unit one by one is computationally unfeasible so every unit is scored at once by attribution patching, `Δlogit ≈ −∂logit/∂h · h`, the first-order effect of removing it, from one backward pass. Each row carries a mini-board of where on the board the unit does its work (not shown for models whose tokens aren't squares). Same colors as the head grid: blue carries the move, orange suppresses it. 

Under the logit-lens curve, a strip shows the model's own top move at every readout point, so you can see when the played move takes over from what the early layers favored.

**Analyze Many Positions**

The drawer's second tab. Pick a pool — the 1,000 ordinary positions bundled with the package (real Lichess games), this game, or a file of FENs (one per line) — and press `Scan`: every MLP neuron's peak activation on every position is recorded on a background thread, a batch at a time, so the rest of the app keeps answering (a scan that would take over a minute pauses and asks first). The grid on the left is each layer's ten strongest neurons by mean peak (brighter = higher; circles for neurons, squares for heads elsewhere). Click one, or press `Where else does it fire? ▸` in the Neurons panel, and the middle board shows where it fires on average (the model's frame, side to move at the bottom) with its mean, spread, max and rank in the layer; on the right are the eight positions where it fires most. Click a position to see it in every panel; a strip under the board steps through the eight, and `↩ back to the game` restores the game exactly as it was.

**Steer a circuit**
Shift+click heads in the causal-heads grid and rows in the causal-neurons list (or turn on `Pick for steering` in the drawer and click them) to build a steer set; it appears above the board as chips with one strength slider (×0 removes every picked write, ×2 doubles it, ×−1 flips it). 

**Search (Leela)**

With a Leela model and the `lc0` program installed (`brew install lc0`; or set `$CHESSFORMER_LC0`), a `Search` card appears above the Win / Draw / Loss bar. Pick 100, 400 or 1600 nodes and press `Search`: lc0 runs its search on the same net, and the policy list shows each move's raw network probability (blue) over its share of the search's visits (green), with visits and Q on hover. The note says whether search agreed with the network's top move or *overruled* it. The principal variation appears as chips; click one to see that position in every panel, and `↩ back to the game` returns.
