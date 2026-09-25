# Engine Guide
*(`engine.py`: the interpretability core behind the app, plots, and widgets)*

An interpretability core for chessformers that treat the 64 squares as tokens.
`MaiaEngine` loads a Maia-3 checkpoint, runs forward passes, and captures the
residual stream at every layer.

`engine.py` imports cleanly into a notebook and is called by `interp_plot.py`
(static figures), `interp_widget.py` (interactive panels), and the standalone
app in `bridge.py` / `app.py` / `ui.py`.

*The next release will do the same for Leela (LC0 BT4).*


**Run the Model**

- `evaluate`: one forward pass yielding the full normalized policy over legal
  moves (in descending order) and the W/D/L for the side to move.
- `select_move`: pick a move at a rating (temperature 0 = argmax), through the
  released engine's own sampler.
- `tokens`: the position as the model's input tokens, padded to fill `history`.
- `run_with_cache`: forward pass returning `(out, cache)`, the whole residual
  stream, TransformerLens style.
- `run_with_hooks`: forward pass with intervention hooks for activation
  patching, ablation, and steering.
- `logit_lens`: decode any residual activation at a readout point by pushing it
  through the policy head.

**Take Attention Apart**

- `attention`: the 64×64 components of one (layer, head): semantic `QKᵀ`,
  geometric `GAB`, and the softmax the model actually runs.
- `qk_scores`: one layer's scaled `QKᵀ` logits.
- `gab_bias`: one layer's generated bias.
- `gab_coeffs`: the coefficients head h applies to the GAB bank.
- `gab_templates`: the static square-pair template bank every layer shares.
- `head_writes`: the exact residual writes of one layer's attention.

**Interventions**

- `ablate_head`: forward pass with one head's write removed, exactly.
- `ablate_grid`: the carrier heatmap of one move. Every head is ablated in turn,
  `Δlogit = ablated − clean` (negative = the head carried it).
- `ablate_grid_batch`: `ablate_grid` over a batch of positions at once;
  minimizes CPU syncs to take advantage of GPUs.

**Tracing**

- `residual_stream`: per-square views of how the stream is built up, one row
  per readout point.
- `compare_residual`: the same position at two ratings, differenced. Shows where
  skill diverges *inside* the stream, not just in the output.
- `depth_points`: the readout points as `[{label, kind}]`, the x axis in every
  depth plot.
- `move_logit_lens`: one move's depth curve: logit, probability, and rank at
  every readout point.
  - `logit_per_depth`: that curve's raw logit alone, unmasked.
  - `policy_per_depth`: that curve after the softmax over legal moves.
  - `rank_per_depth`: that curve as a rank, 1 being the top move at that depth.

**Moves**

- `move_info`: one move in every representation at once, as a dictionary.
  **The** method to use when converting between any of the five move forms.
- `to_move`: any move form → `chess.Move` (used throughout the other files).

**Others**

- `save_activations`: persist the most recent forward pass's snapshot to disk.
- `remove_hooks`: detach the capture hooks, for a bare forward with no CPU copies.

**More info**

Every tensor a read path returns is on CPU, in the model's canonical
side-to-move frame (square = rank*8 + file). Depth reads the same as in the
figures: emb, then aN/mN for layer N's attention and MLP sub-layers, then
enc — depth_points hands you that axis directly.

Module level, beside the class: build_cfg builds the args-namespace the model
expects from a registry alias, and pick_device resolves the torch device
(explicit > $MAIA3_DEVICE > cuda > cpu).
