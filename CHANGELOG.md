# Changes from 1.5 to 3.0

Every change to the engine, the figures and the app since v1.5 (bdc7961).

## engine.py

**Structure (2.0)**
- `ChessformerEngine` no longer contains a model. It works over a `ModelAdapter` (new `core.py`), which supplies tokens, the forward pass, the policy vocabulary, a `TokenLayout`, the hook points, the policy head, each layer's q/k/v and additive score terms, and each MLP's output projection. The BT4/Maia model classes that lived in engine.py (`forward`, `policy_logits`, `value_logits`, `moves_left`, `dense`/`ffn`/`lin`, `put`) moved to `adapters/lc0.py`, `adapters/maia3.py`, `adapters/smolgen.py`.
- `MaiaEngine` and `LeelaEngine` are thin wrappers; neither is the package default. `load_engine(alias)` resolves through the adapter registry (`register_adapter`, entry-point group `chessformer_lens.adapters`); `resolve_engine`, `format_engine_list`, `build_cfg`, `pick_device` at module level.
- Leela no longer needs the `maia3` package (independent smolgen block). `maia3` is imported only when a Maia-3 model loads.
- Leela nets download on first use: `bt4` (base BT4) and `bt4-332` (policytune-332) from lczero.org into `~/.cache/chessformer_lens` (`$CHESSFORMER_CACHE`), sha256 checked, `CHESSFORMER_NO_DOWNLOAD=1` refuses; a renamed file is recognised by size and named correctly. Lookup order: `$CHESSFORMER_WEIGHTS`, `./weights`, the cache.
- Ratings: `Conditioning` replaces the positional `(self_elo, oppo_elo)` pair (which still works); a model with no rating input ignores them. `ValueEstimate` covers scalar value heads; activations are `(tokens, d_model)`.

**New methods (2.0)**
- `layout(board)` — what each token of the residual stream stands for.
- `evaluate_batch(boards, …)` — `evaluate` over many positions, leaving the activation snapshot alone.
- `no_capture()` — context in which forward passes leave the snapshot alone; snapshot reads inside it now raise instead of answering for the previous position.
- `attention_terms(board, layer)` — every score term of one layer, all heads, in one pass (QKᵀ, GAB where the model has it).
- Properties `has_conditioning`, `has_generated_bias`, `has_mlh`; `arch` (n_layers, n_heads, d_model, d_mlp), `caps`.
- Maia-3's think-time head exposed as `aux["ponder"]`.

**New methods (3.0)**
- `top_neurons(positions, k=10, …)` → `NeuronReport` — every layer's MLP neurons ranked by peak activation over the squares, averaged over a list of positions; `report.table()`, `report.top(layer)`, `report.peaks_of(layer, neuron)`.
- `NeuronScan(eng, positions)` — the same a batch at a time (`step`, `run`, `report`), for progress bars and stopping early.
- `top_positions(layer, neuron, positions=None, k=8)` — the positions where one neuron fires most (default pool: the bundled positions).
- `neuron_effects(layer, neuron, positions)` — what one neuron does on each position, by exact ablation: the moves it pushes up and down, whether the top move changes.
- Module level: `load_positions` (FENs, boards or a PGN file → boards), `positions_from_pgn` (one position per game, seeded), `ordinary_positions()` (1,000 bundled Lichess positions, CC0, in `ordinary_positions.txt`).
- `LeelaEngine.search(board, nodes=400)` — lc0's search on the same net through python-chess: bestmove, PV, root Q/WDL, per-move visits/share/prior/Q. `find_lc0()` finds the program (`$CHESSFORMER_LC0`, PATH, Homebrew).
- `neuron_overview` ranks a layer's units by peak activation (was: norm over tokens).
- `steer_head(board, elo, layer, head, scale)` / `steer_neuron(…)` — activation steering: the head's or unit's exact write multiplied by `scale` (1 = clean, 0 = the ablation, 2 doubles it, −1 flips it).

## interp_plot.py
- Theme globals annotated for type checkers (`f8e9d47`, yours).
- Model-neutral (2.0): `plot_attention` / `plot_attention_layer` read the model's own score terms (`_att_terms`) instead of assuming QKᵀ + GAB; a scalar value head draws an expected-score bar (`_value_bar`, `_bar_parts`) beside the W/D/L bar; a model whose tokens are not squares gets a clear error from the board figures (`_board_only`).
- New figures (3.0): `plot_top_neurons` (each layer's strongest neurons as a viridis grid of circles, and where one fires on average), `plot_top_positions` (the positions where a neuron fires most, as plain boards titled with side · peak · square), `plot_search` (raw policy vs search visits and Q, the PV, W/D/L after search, "overruled" when search plays something else).
- The carrier grid labels heads `h0…` instead of `H0…`.

## The app

**app.py**
- No default model: `chessformer_lens` with nothing named opens a picker; `--list` prints every alias by family.
- `--web [--port] [--host] [--no-browser]` serves the same app to a web browser.

**bridge.py** (`MaiaApi` → `LensApi`)
- Model-neutral names: `engine_move` (was `maia_move`), `engine_to_move`, `engine_policy`, …; payloads carry the expected score beside W/D/L and the token layout; `LensApi(engine=eng)` reuses an engine already loaded.
- Picker: `models()`, `load_model(alias)`; `info()` reports `needs_model`, `loading`, capability flags (`conditioning`, `has_mlh`, `search`).
- `_base()` carries `root_fullmove` / `root_turn` (moves list numbered from a loaded position) and `browsing`.
- `ablate` / `ablate_neuron` take a `scale` (steering) and report it.
- Search: `search(nodes)`, `search_pv(k)`; many positions: `scan_start(pool, elo, fens)`, `scan_status`, `scan_continue`, `scan_stop`, `scan_report`, `neuron_positions`, `neuron_effects`, `open_position`; `go_back()` restores the game after a PV or scanned position.

**ui.py**
- Picker overlay ("Choose a model"); the rating card is "Rating (self_elo)" and hides for models without a rating input; "not available yet" notices for models whose tokens aren't squares; the piece SVGs (`PIECE_URI`) now live here (`pieces.py` is gone).
- Search card (Leela + lc0): nodes chips, green visit bars over the blue policy bars, "overruled" note, PV chips that open the position in every panel, back to the game.
- Drawer tab "Analyze many positions": pools (1,000 ordinary / this game / a file of FENs), background scan with progress and a pause for long scans, the neuron grid (viridis circles), one neuron's mean board and stats, its eight strongest positions as boards; "Where else does it fire? ▸" in the Neurons panel; opening a position shows it in every panel with a strip under the board to step through.
- Network diagram tooltips show peak activation.
- A strength slider (−2 … +3) beside each Ablate button while it is on: the red overlay shows the policy with the head or neuron scaled instead of removed.

**web.py** (new, 2.0)
- `ThreadingHTTPServer` serving the page with the pywebview bridge shimmed onto `POST /api/<name>`; only public `LensApi` methods reachable; loopback host check and a per-server token header; `web.serve(alias | engine=eng)` for notebooks, Colab via the kernel's port proxy.

## Also
- `piece_art.py`: regeneration code removed (PNGs kept for the figures). `pieces.py` deleted.
- `core.py`, `adapters/` new (2.0). Notebooks in `notebooks/` (`app_quickstart.ipynb`, `demo.ipynb` with the two many-positions cells).
- Tests: conformance (adapter contract, toy non-square model), golden regression (local recordings), frontend, web, headless-Chrome UI tests (opt-in), many-positions, lc0 search.
