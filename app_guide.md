# App Guide 
*(If you cloned the repo look at Screenshots/chessformer_lens_demo.mp4)*:

**Play or Set up a Position**
Click to move your pieces. Select an Elo for the engine (Maia-3 is created to mimic how HUMANS play at that strength, with all of our innate biases)
and use `New game`, and a dropdown:`You play White` / `You play Black` / `Set up position` / `Random position`.
It is also possible to `paste a FEN to load` a position with the `Load` button. 

`Random position` generates a position through a random walk of the engine's policies. It is ideal for testing mechanistic hypothesis on realistic positions. 

In setup mode both sides are yours; clicks move pieces ignoring legality, and clicking the same square twice deletes the piece. 
Select a color to continue as it from that position.

The last move is indicated by a gold arrow, legal moves for a selected piece are indicated by dots, captures are drawn as rings. The games
moves are recorded in SAN notation.

**Read the Position**

At the top in the center there is a `Win / Draw / Loss · side to move` stacked bar. 
Under it is the `Maia rating (self_elo)` slider: 600-2800, step 25, default 1500; Dragging reevaluates the same position. 
Under that is the scrollable ranked list: `Policy over N legal moves`: every legal move gets a row. 


![app view](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/SS11.png)
*App view: board, evaluation, policy, live attention, live neurons*

**Take the Model Apart**
Get the `Live attention · this position`, with `Layer` and `Head` chip rows.
Select a square on the right to set the query for the three boards, labeled:
-`semantic attention (QKᵀ)`
-`geometric attention (GAB)`
-`final head attention matrix (scaled softmax(QKᵀ + GAB))`
`Ablate this head` redraws the policy list with the ablated pass in red over the clean pass in blue, re-sorted by
signed `Δ = p(ablated) − p(clean)` — the moves the head was suppressing rise to the top, the ones it was holding up sink.

The `Neurons` panel to the right does the same for one MLP neuron. A network diagram shows one column of dots per MLP layer. Click a column to pick the layer, a dot to pick the neuron. The dots in the selected layer are its most active units on this position. Below, the neuron's activation on every square, and `Ablate this neuron` removes its exact write on every square.



**Analyze One Move**
![Analyze one move](https://raw.githubusercontent.com/chessformer-lens/chessformer_lens/main/Screenshots/SS12.png)
*Analyze one move: `Logit lens`, `causal heads`, `causal neurons`*



One drawer, `Analyze one move`, docked at the bottom with a `▲ pull up` grip. The side panels run 1.5 inches below the board; when the drawer opens it rises an inch above the board's bottom edge and the whole layout above scales down to make room, so nothing is ever covered. `Escape` closes it. Click up to 4 policy moves to open it. Its three sections sit side by side and all describe the primary move (click a move chip to change it):

`Logit lens` — the move's logit at every readout point of the forward pass (the logit lens narrowed to one policy entry); a marker shows from where the move sustains rank 1; per-dot hover, ex: `b3 mlp · logit 4.21 · p 38.2% · rank 1/31`.

`Causal heads` — (layers × heads) forward passes, each with one head's exact residual write removed, recording `Δlogit = ablated − clean`. Hover for exact values; the largest-|Δ| head is ringed red; click a cell to open that head in the attention panel. The final layer is dimmed and striped, excluded from carrier attribution, because it writes straight to the logits and muddles the earlier structure.

`Causal neurons` — the same question at neuron grain. Ablating every MLP unit one by one is computationally unfeasible so every unit is scored at once by attribution patching, `Δlogit ≈ −∂logit/∂h · h`, the first-order effect of removing it, from one backward pass. Each row carries a mini-board of where on the board the unit does its work. Same colours as the head grid: blue carries the move, orange suppresses it. 

Under the logit-lens curve, a strip shows the model's own top move at every readout point, so you can see when the played move takes over from what the early layers favoured.
