"""Positions in, MLP neurons out. The batched readings must equal
the engine's own single-position `neuron_activation`, on Maia-3 (when its 5M
weights are cached) and on BT4 (when its weights are here)."""
import chess
import pytest

import battery
from chessformer_lens.engine import load_positions, ordinary_positions, positions_from_pgn, NeuronScan
from test_conformance import _maia_cached

OPERA = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16"
BLACK = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4"

PGN = """[Event "a"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 *

[Event "too short"]
[Result "*"]

1. d4 d5 *

[Event "set up"]
[SetUp "1"]
[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]
[Result "*"]

1. e4 Kd7 2. e5 Ke6 3. Ke2 Kxe5 4. Ke3 Kd5 5. Kd3 Kc5 6. Kc3 Kb5 7. Kb3 *

[Event "b"]
[Result "*"]

1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 Nb6 7. O-O Be7 8. d3 O-O *
"""


def test_ordinary_positions():
    boards = ordinary_positions()
    assert len(boards) == 1000 and len(ordinary_positions(10)) == 10
    assert all(len(b.move_stack) <= 8 and not b.is_game_over() for b in boards)
    turns = {b.turn for b in boards}
    assert turns == {chess.WHITE, chess.BLACK}
    assert boards[0].fen() == ordinary_positions(1)[0].fen()        # fixed, not resampled


def test_load_positions(tmp_path):
    b = chess.Board()
    b.push_uci("e2e4")
    boards = load_positions([OPERA, b])
    assert [x.fen() for x in boards] == [chess.Board(OPERA).fen(), b.fen()]
    assert boards[1].move_stack == b.move_stack and boards[1] is not b
    assert len(load_positions(OPERA)) == 1
    pgn = tmp_path / "games.pgn"
    pgn.write_text(PGN)
    got = load_positions(pgn, 5, min_ply=4, max_ply=30)
    assert len(got) == 2                                  # one too short, one set up from a FEN
    assert all(4 <= len(x.move_stack) <= 16 for x in got)       # game "a" has 17 plies
    again = positions_from_pgn(pgn, 5, min_ply=4, max_ply=30)
    assert [x.fen() for x in got] == [x.fen() for x in again]          # seeded
    with pytest.raises(TypeError):
        load_positions([42])


def _check_engine(eng, elo, tol):
    positions = [OPERA, BLACK] + ordinary_positions(4)
    rep = eng.top_neurons(positions, k=3, elo=elo, batch_size=4)
    n_layers = eng.arch.n_layers
    assert rep.layers == list(range(n_layers)) and len(rep.rows) == 3 * n_layers
    for L in rep.layers:
        means = [r["mean"] for r in rep.top(L)]
        assert means == sorted(means, reverse=True) and [r["rank"] for r in rep.top(L)] == [1, 2, 3]
    for r in rep.rows[::5]:                                 # the batch equals one at a time
        acts = [eng.neuron_activation(b, elo, r["layer"], r["neuron"])["act"] for b in rep.positions]
        peaks = [max(a) for a in acts]
        assert r["mean"] == pytest.approx(sum(peaks) / len(peaks), abs=tol)
        assert r["max"] == pytest.approx(max(peaks), abs=tol)
        squares = [sum(a[s] for a in acts) / len(acts) for s in range(64)]
        assert r["squares"] == pytest.approx(squares, abs=tol)
        assert r["best"][0] == max(range(len(peaks)), key=peaks.__getitem__)
        assert rep.peaks_of(r["layer"], r["neuron"]) == pytest.approx(peaks, abs=tol)
    assert "positions" in rep.table() and len(repr(rep)) < 400
    one = eng.top_neurons(positions, k=2, layers=[1], elo=elo)
    assert one.layers == [1] and [r["neuron"] for r in one.rows] == [r["neuron"] for r in rep.top(1, 2)]

    L, u = rep.rows[0]["layer"], rep.rows[0]["neuron"]
    top = eng.top_positions(L, u, positions, k=3, elo=elo)
    assert [t["peak"] for t in top] == sorted((t["peak"] for t in top), reverse=True)
    for t in top:                                           # real squares, either side to move
        act = eng.neuron_activation(t["board"], elo, L, u)["act"]
        canon = max(range(64), key=act.__getitem__)
        assert t["peak"] == pytest.approx(act[canon], abs=tol)
        assert t["square"] == chess.square_name(eng._frame_square(canon, t["board"]))
        assert t["peak_canon"] == canon
    every = eng.top_positions(L, u, positions, k=None, elo=elo)
    assert len(every) == len(positions) and every[:3] == top
    with pytest.raises(ValueError):
        eng.top_positions(L, u, [], elo=elo)
    with pytest.raises(ValueError):
        eng.top_neurons(positions, layers=[n_layers], elo=elo)
    return rep


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_maia(tmp_path):
    from chessformer_lens.engine import MaiaEngine
    eng = MaiaEngine("maia3-5m", device="cpu", activation_dir=str(tmp_path))
    rep = _check_engine(eng, 1500, 1e-4)
    assert rep.elo == 1500
    low = eng.top_neurons(rep.positions, k=3, elo=800)
    assert low.peaks.shape == rep.peaks.shape and not low.peaks.equal(rep.peaks)   # rating matters


@pytest.mark.skipif(battery.bt4_weights() is None, reason="BT4 weights not found")
def test_leela(tmp_path):
    from chessformer_lens.engine import LeelaEngine
    eng = LeelaEngine("bt4", checkpoint_path=battery.bt4_weights(), activation_dir=str(tmp_path))
    eng.evaluate(chess.Board(OPERA), 0)
    before = {k: v.clone() for k, v in eng._activations.items()}
    eng.top_neurons([BLACK], k=1)
    eng.top_positions(2, 5, [OPERA, BLACK], k=1)
    assert all(eng._activations[k].equal(v) for k, v in before.items())    # snapshot untouched
    rep = _check_engine(eng, 0, 1e-3)
    assert rep.elo is None


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_scan_in_steps(tmp_path):
    from chessformer_lens.engine import MaiaEngine
    eng = MaiaEngine("maia3-5m", device="cpu", activation_dir=str(tmp_path))
    positions = ordinary_positions(10)
    scan = NeuronScan(eng, positions, batch_size=4)
    assert (scan.done, scan.total, scan.finished) == (0, 10, False)
    with pytest.raises(ValueError):
        scan.report()
    assert scan.step() == 4
    part = scan.report(k=2)
    assert len(part.positions) == 4
    scan.run()
    whole, direct = scan.report(k=2), eng.top_neurons(positions, k=2, batch_size=3)
    assert [(r["layer"], r["neuron"]) for r in whole.rows] == \
        [(r["layer"], r["neuron"]) for r in direct.rows]
    assert whole.peaks.allclose(direct.peaks, atol=1e-4)


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import chessformer_lens.interp_plot as ip
    from chessformer_lens.engine import MaiaEngine
    eng = MaiaEngine("maia3-5m", device="cpu", activation_dir=str(tmp_path))
    texts = lambda fig: " ".join(t.get_text() for ax in [fig, *fig.axes] for t in ax.texts)
    fig = ip.plot_top_neurons(eng, [OPERA, BLACK], k=4)
    assert len(fig.axes) == 2 and "Top neurons" in texts(fig) and "elo 1500" in texts(fig)
    plt.close(fig)
    rep = eng.top_neurons([OPERA], k=3)
    r = rep.top(2, 3)[2]
    fig = ip.plot_top_neurons(eng, report=rep, layer=2, neuron=r["neuron"], width="page")
    assert f"L2·n{r['neuron']}" in texts(fig)
    plt.close(fig)
    with pytest.raises(ValueError):
        ip.plot_top_neurons(eng, report=rep, layer=2, neuron=10 ** 6)
    fig = ip.plot_top_positions(eng, 3, 7, ordinary_positions(30), k=4)
    assert len(fig.axes) == 4 and "L3·n7" in texts(fig)
    plt.close(fig)
    found = eng.top_positions(3, 7, ordinary_positions(30), k=None)
    fig = ip.plot_top_positions(eng, 3, 7, results=found, k=2)
    assert len(fig.axes) == 2
    plt.close(fig)


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_the_apps_scan(tmp_path):
    import time
    from chessformer_lens.bridge import LensApi
    from chessformer_lens.engine import MaiaEngine
    api = LensApi(engine=MaiaEngine("maia3-5m", device="cpu", activation_dir=str(tmp_path)))
    assert api.scan_status() == {"state": "none"} and "error" in api.scan_report()
    api.analyze()                                                     # move both sides
    for u in "e2e4 e7e5 g1f3 b8c6 f1b5".split():
        assert "error" not in api.human_move(u)
    st = api.scan_start("game", 1500)
    for _ in range(300):
        if api.scan_status()["state"] == "done":
            break
        time.sleep(.05)
    st = api.scan_status()
    assert st["state"] == "done" and st["total"] == 6 and st["elo"] == 1500
    assert api.scan_start("game", 1500)["state"] == "done"            # the same scan, kept
    rep = api.scan_report(4)
    assert rep["n"] == 6 and len(rep["grid"]) == len(rep["layers"]) and len(rep["grid"][0]) == 4
    L, n = rep["layers"][3], rep["grid"][3][0]["neuron"]
    d = api.neuron_positions(L, n, 3)
    assert d["rank"] == 1 and len(d["tiles"]) == 3 and len(d["squares"]) == 64
    assert [t["peak"] for t in d["tiles"]] == sorted((t["peak"] for t in d["tiles"]), reverse=True)
    t = d["tiles"][0]
    game = api.board.copy()
    opened = api.open_position(t["index"])
    assert opened["fen"] == t["fen"] and opened["root_fullmove"] == 1 and opened["human_color"] == "both"
    assert len(opened["san_history"]) == t["index"]                   # this game's history, to there
    back = api.go_back()
    assert back["fen"] == game.fen() and back["san_history"] == api._sans(game)
    assert "error" in api.neuron_positions(L, 10 ** 6)
    lo, hi = sorted(t["peak"] for t in d["tiles"])[0], d["max"]   # a range, as a histogram bin
    binned = api.neuron_positions(L, n, 2, lo, hi)
    assert binned["bin"]["count"] >= 2 and len(binned["tiles"]) == 2
    assert all(lo - 1e-4 <= t["peak"] <= hi + 1e-4 for t in binned["tiles"])   # batch numerics
    eff = api.neuron_effects(L, n, [t["index"] for t in d["tiles"]], d["scan_id"])
    assert len(eff["effects"]) == 3 and {"top", "top_ablated", "changed"} <= set(eff["effects"][0])
    assert "error" in api.neuron_effects(L, n, [0], d["scan_id"] + 1)


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_neuron_effects_are_the_engines_ablation(tmp_path):
    from chessformer_lens.engine import MaiaEngine
    eng = MaiaEngine("maia3-5m", device="cpu", activation_dir=str(tmp_path))
    positions = [OPERA, BLACK] + ordinary_positions(3)
    eff = eng.neuron_effects(4, 17, positions, k=100, batch_size=2)
    assert len(eff) == len(positions)
    for b, e in zip(load_positions(positions), eff):
        clean = dict(eng.evaluate(b, 1500)["policy"])
        ablated = dict(eng.ablate_neuron(b, 1500, 4, 17)["policy"])
        for uci, dp in e["supports"] + e["opposes"]:
            assert dp == pytest.approx(clean[uci] - ablated[uci], abs=1e-5)
        assert e["top"] == max(clean, key=clean.get)
        assert e["changed"] == (e["top"] != max(ablated, key=ablated.get))
        assert all(d > 0 for _, d in e["supports"]) and all(d < 0 for _, d in e["opposes"])
