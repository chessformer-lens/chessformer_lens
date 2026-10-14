"""Invariants of the interp core, checked on models that need no download.

Every test here is structural: it holds for any weights, so the models are
randomly initialised.

  maia   Maia-3 5M architecture, random weights (needs the maia3 package;
         skipped without it)
  leela  a tiny BT4-style net (2 layers, 4 heads, 64-d), random weights,
         built in-process (no lc0 file, no download)

What is checked, in the order a reader would doubt it:
  the logit lens is exact at the last readout point
  per-head writes sum back to each layer's attention output
  head ablation agrees with an independent re-implementation of attention
  ablate_grid, ablate_grid_batch, ablate_head and steer_head agree
  carrier_neurons' exact deltas equal ablate_neuron's
  GAB is exactly coefficients x templates
  every legal move round-trips through every move form, both colours
  the frontends: the demo notebook names no undefined variable, SAE.py
  imports, and the web server refuses foreign callers

Run:  pytest            (from the repo root; pyproject points pytest here)
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

import chess
import pytest
import torch
import torch.nn.functional as F

from chessformer_lens.core import ArchSpec
from chessformer_lens.engine import ChessformerEngine, MaiaEngine, ordinary_positions

ROOT = Path(__file__).resolve().parents[1]
ELO = 1500

FENS = {
    "start": chess.STARTING_FEN,
    "italian": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "opera": "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16",
    "black_promo_castle": "r3k2r/pPppqppp/8/8/8/8/PpPPQPPP/R3K2R b KQkq - 0 1",
    "en_passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "lone_promo": "8/P7/8/8/8/8/6k1/4K3 w - - 0 1",
}
BOARDS = [chess.Board(f) for f in FENS.values()]


# ----- models ---------------------------------------------------------------------

def _sharpen(model, factor=3.0):
    """Scale the weight matrices up so attention is far from uniform and
    ablations move the logits by O(1); default init leaves them nearly flat."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.mul_(factor)


@pytest.fixture(scope="session")
def maia(tmp_path_factory):
    pytest.importorskip("maia3")
    from maia3.models import MAIA3Model
    from chessformer_lens.adapters.maia3 import build_cfg

    torch.manual_seed(0)
    cfg, _ = build_cfg("maia3-5m", "cpu")
    model = MAIA3Model(cfg)
    _sharpen(model)
    path = tmp_path_factory.mktemp("weights") / "maia3-5m-random.pt"
    torch.save(model.state_dict(), path)
    return MaiaEngine("maia3-5m", device="cpu", checkpoint_path=str(path))


@pytest.fixture(scope="session")
def leela():
    from chessformer_lens.adapters.lc0 import BT4Model, Lc0Adapter, Lc0Vocab, _leela_all_moves
    from chessformer_lens.adapters.smolgen import GabAccess

    torch.manual_seed(0)
    model = BT4Model(num_blocks=2, dim=64, num_heads=4, ff=96, gen=16, per_square=4,
                     intermediate=32, pre=8, policy_dim=64, value_dim=8, mlh_dim=4)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    model.eval()

    a = Lc0Adapter()
    a.device, a.model = "cpu", model
    a.cfg = types.SimpleNamespace(**vars(model.cfg), checkpoint_path="tiny", device="cpu",
                                  history=8, trust_checkpoint=False)
    a.spec = types.SimpleNamespace(name="lc0-tiny", display_name="tiny BT4", repo_id=None,
                                   checkpoint_filename="tiny", aliases=(), config={})
    a.arch = ArchSpec(n_layers=2, n_heads=4, d_model=64, d_mlp=96, name="lc0-tiny",
                      display_name="tiny BT4")
    a.vocab = Lc0Vocab(_leela_all_moves())
    a._gab = GabAccess([blk.self_attn for blk in model.transformer.layers],
                       model.gab_shared_weight)
    return ChessformerEngine(a)


@pytest.fixture(params=["maia", "leela"])
def eng(request):
    return request.getfixturevalue(request.param)


def _top_move(eng, board):
    return eng.evaluate(board, ELO)["policy"][0][0]


# ----- forward pass ---------------------------------------------------------------

@pytest.mark.parametrize("board", BOARDS, ids=list(FENS))
def test_policy_covers_exactly_the_legal_moves(eng, board):
    out = eng.evaluate(board, ELO)
    assert {u for u, _ in out["policy"]} == {m.uci() for m in board.legal_moves}
    assert sum(p for _, p in out["policy"]) == pytest.approx(1.0, abs=1e-4)
    probs = [p for _, p in out["policy"]]
    assert probs == sorted(probs, reverse=True)


def test_batch_matches_single(eng):
    for board, batched in zip(BOARDS, eng.evaluate_batch(BOARDS, ELO)):
        single = dict(eng.evaluate(board, ELO)["policy"])
        for uci, p in batched["policy"]:
            assert p == pytest.approx(single[uci], abs=1e-4)


# ----- logit lens -----------------------------------------------------------------

@pytest.mark.parametrize("board", BOARDS, ids=list(FENS))
def test_logit_lens_is_exact_at_the_last_readout(eng, board):
    """unembed on the stream the heads read must reproduce the real policy
    logits; otherwise every depth curve is measured against the wrong thing."""
    with torch.no_grad():
        logits, _ = eng._forward(board, ELO)
        last = eng._lens_steps()[-1][0]
        lens = eng.adapter.unembed(eng._snap(last)[0])
    torch.testing.assert_close(lens, logits, atol=1e-5, rtol=1e-5)


def test_move_logit_lens_shape_and_snap(eng):
    board = BOARDS[2]
    res = eng.move_logit_lens(board, ELO, _top_move(eng, board))
    assert [s["label"] for s in res["steps"]] == [d["label"] for d in eng.depth_points()]
    assert len(res["steps"]) == 2 * eng.arch.n_layers + 1 + ("encoder_out" in eng.hook_points)
    assert res["steps"][-1]["rank"] == 1


def test_compare_residual_embedding_diff(maia, leela):
    """Maia-3: the rating enters every token identically, so at `emb` the diff
    has the same size on all 64 squares. BT4 has no rating: all zeros."""
    emb = torch.tensor(maia.compare_residual(BOARDS[1], 1100, 2200)["steps"][0]["norm"])
    assert float(emb.min()) > 0
    torch.testing.assert_close(emb, emb.mean().expand(64), atol=1e-4, rtol=1e-4)
    flat = leela.compare_residual(BOARDS[1], 1100, 2200)["steps"]
    assert all(max(s["norm"]) == 0 for s in flat)


# ----- attention: decomposition and ablation ------------------------------------

def test_head_writes_sum_back_on_every_layer(eng):
    """head_writes asserts its own reconstruction; this runs it everywhere."""
    for board in BOARDS[:3]:
        for layer in range(eng.arch.n_layers):
            w = eng.head_writes(board, ELO, layer=layer)
            assert w.shape == (eng.arch.n_heads, 64, eng.arch.d_model)


def test_attention_and_gab_decomposition(eng):
    board, layer, head = BOARDS[1], 1, 2
    att = eng.attention(board, ELO, layer=layer, head=head)
    A = torch.tensor(att["attn"])
    torch.testing.assert_close(A.sum(-1), torch.ones(64), atol=1e-5, rtol=0)
    gab = eng.gab_bias(board, ELO, layer=layer, head=head)
    torch.testing.assert_close(torch.tensor(att["gab"]), gab, atol=1e-5, rtol=1e-5)
    coeffs = eng.gab_coeffs(board, ELO, layer=layer)
    recon = (coeffs[head][:, None, None] * eng.gab_templates()).sum(0)
    torch.testing.assert_close(recon, gab, atol=1e-4, rtol=1e-4)


def _independent_ablated_logit(eng, board, layer, head, idx):
    """Head ablation from first principles: recompute the layer's attention
    from its packed QKV weights and the model's own GAB generator, zero head
    `head`'s slice before the output projection, patch that in."""
    attn = eng.model.transformer.layers[layer].self_attn
    mha = attn.mha
    with torch.no_grad():
        x = eng._block_input(board, ELO, None, layer)
        d, H = x.size(-1), mha.num_heads
        dh = d // H
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, -1)
        q, k, v = (t.view(1, 64, H, dh).transpose(1, 2) for t in (q, k, v))
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh) + attn._sq_bias(x), -1)
        o = (A @ v).transpose(1, 2).reshape(1, 64, d)
        clean_out, _ = attn(x)
        torch.testing.assert_close(F.linear(o, mha.out_proj.weight, mha.out_proj.bias),
                                   clean_out, atol=1e-4, rtol=1e-4)
        o[..., head * dh:(head + 1) * dh] = 0
        ablated = F.linear(o, mha.out_proj.weight, mha.out_proj.bias)
        logits, _ = eng.run_with_hooks(board, ELO, fwd_hooks=[(f"attn_{layer:02d}", lambda a: ablated)],
                                       return_type="logits")
    return float(logits[idx])


def test_head_ablation_matches_independent_reimplementation(eng):
    board = BOARDS[2]
    idx = eng.move_info(board, _top_move(eng, board))["idx"]
    layer, head = eng.arch.n_layers - 1, 1
    mine, _ = eng.ablate_head(board, ELO, layer, head, return_type="logits")
    assert float(mine[idx]) == pytest.approx(_independent_ablated_logit(eng, board, layer, head, idx),
                                             abs=1e-4)


def test_ablation_paths_agree(eng):
    board = BOARDS[2]
    move = _top_move(eng, board)
    idx = eng.move_info(board, move)["idx"]
    grid = eng.ablate_grid(board, ELO, move)
    deltas = torch.tensor(grid["deltas"])
    assert deltas.shape == (eng.arch.n_layers, eng.arch.n_heads)
    assert float(deltas.abs().max()) > 1e-3, "ablations should move the logit at all"

    layer, head = eng.arch.n_layers - 1, 1
    ablated, _ = eng.ablate_head(board, ELO, layer, head, return_type="logits")
    assert float(ablated[idx]) - grid["base_logit"] == pytest.approx(deltas[layer, head].item(), abs=1e-4)

    zero, _ = eng.steer_head(board, ELO, layer, head, 0.0, return_type="logits")
    one, _ = eng.steer_head(board, ELO, layer, head, 1.0, return_type="logits")
    assert float(zero[idx]) == pytest.approx(float(ablated[idx]), abs=1e-4)
    assert float(one[idx]) == pytest.approx(grid["base_logit"], abs=1e-4)

    other = BOARDS[1]
    batch = eng.ablate_grid_batch([(other, _top_move(eng, other)), (board, move)], ELO)
    torch.testing.assert_close(torch.tensor(batch[1]["deltas"]), deltas, atol=1e-3, rtol=1e-3)


# ----- neurons ----------------------------------------------------------------------

def test_carrier_neurons_exact_equals_ablate_neuron(eng):
    board = BOARDS[2]
    move = _top_move(eng, board)
    idx = eng.move_info(board, move)["idx"]
    res = eng.carrier_neurons(board, ELO, move, top_k=3, verify=True)
    assert len(res["top"]) == 3
    for t in res["top"]:
        logits, _ = eng.ablate_neuron(board, ELO, t["layer"], t["neuron"], return_type="logits")
        assert float(logits[idx]) - res["base_logit"] == pytest.approx(t["exact"], abs=1e-4)


def test_steer_neuron_endpoints(eng):
    board = BOARDS[1]
    idx = eng.move_info(board, _top_move(eng, board))["idx"]
    with torch.no_grad():
        clean, _ = eng._forward(board, ELO)
    ablated, _ = eng.ablate_neuron(board, ELO, 0, 5, return_type="logits")
    zero, _ = eng.steer_neuron(board, ELO, 0, 5, 0.0, return_type="logits")
    one, _ = eng.steer_neuron(board, ELO, 0, 5, 1.0, return_type="logits")
    assert float(zero[idx]) == pytest.approx(float(ablated[idx]), abs=1e-4)
    assert float(one[idx]) == pytest.approx(float(clean[idx]), abs=1e-4)


def test_many_position_tools(eng):
    pos = ordinary_positions(16)
    assert len(pos) == 16
    rep = eng.top_neurons(pos, k=2, layers=[0, eng.arch.n_layers - 1])
    assert len(rep.rows) == 4
    neuron = rep.top(0, 1)[0]["neuron"]
    top = eng.top_positions(0, neuron, pos, k=3)
    assert [t["peak"] for t in top] == sorted((t["peak"] for t in top), reverse=True)
    effects = eng.neuron_effects(0, neuron, pos[:4])
    assert len(effects) == 4 and all(e["top"] in {m.uci() for m in chess.Board(e["fen"]).legal_moves}
                                     for e in effects)


# ----- moves and frames ---------------------------------------------------------------

@pytest.mark.parametrize("board", BOARDS, ids=list(FENS))
def test_every_legal_move_round_trips(eng, board):
    for move in board.legal_moves:
        info = eng.move_info(board, move)
        assert eng.to_move(board, info["idx"]) == move
        assert eng.to_move(board, move.uci()) == move
        assert eng.to_move(board, info["san"]) == move
        pair = eng.to_move(board, (info["from"], info["to"]))
        assert (pair.from_square, pair.to_square) == (move.from_square, move.to_square)


def test_king_takes_rook_castling_is_normalised(eng):
    """lc0 encodes castling as king takes rook. Every form of it must resolve to
    the standard move, or uci lookups into the policy miss (and on Maia, e8h8
    would index the plain king-to-h8 square pair instead of castling)."""
    board = chess.Board(FENS["black_promo_castle"])
    castle = chess.Move.from_uci("e8g8")
    assert eng.to_move(board, "e8h8") == castle
    assert eng.to_move(board, chess.Move.from_uci("e8h8")) == castle
    assert eng.move_info(board, "e8h8")["idx"] == eng.move_info(board, castle)["idx"]
    assert eng.move_info(board, "e8h8")["uci"] == "e8g8"


def test_leela_planes_history_and_mirror():
    from chessformer_lens.adapters.lc0 import _leela_planes

    board = chess.Board()
    board.push_uci("e2e4")                                 # Black to move: mirrored view
    planes = _leela_planes(board).view(112, 64)
    assert planes[108].min() == 1                          # side-to-move plane
    ours = {i for i in range(64) if planes[0, i]}          # Black's pawns, seen from Black
    assert ours == set(range(8, 16))
    slot1 = planes[13:26]                                  # one ply ago = the start position
    assert slot1[0].sum() == 8 and slot1[6].sum() == 8
    assert planes[26:104].sum() == 0                       # nothing before move 1


# ----- frontends ------------------------------------------------------------------------

def _notebook_source(path):
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    lines = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
    return "\n".join(lines)


@pytest.mark.parametrize("nb", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.name)
def test_notebooks_parse_and_name_nothing_undefined(nb):
    """Catches a cell that uses a variable no earlier cell defines (the demo's
    last cell once passed an undefined `positions`)."""
    src = _notebook_source(nb)
    ast.parse(src)
    pyflakes = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    import io
    out = io.StringIO()
    pyflakes.check(src, nb.name, Reporter(out, out))
    undefined = [l for l in out.getvalue().splitlines() if "undefined name" in l]
    assert not undefined, undefined


def test_notebooks_ship_without_local_paths():
    for nb in (ROOT / "notebooks").glob("*.ipynb"):
        text = nb.read_text(encoding="utf-8")
        assert "/Users/" not in text and "\\Users\\" not in text, nb.name


def test_sae_script_imports():
    spec = importlib.util.spec_from_file_location("sae_script", ROOT / "SAE.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.main([])                                       # usage, not a crash


def test_web_server_refuses_foreign_callers():
    from chessformer_lens.bridge import LensApi
    from chessformer_lens.web import make_server

    server = make_server(LensApi())
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(name, token=server.token, host=None):
        headers = {"X-Chessformer-Token": token}
        if host:
            headers["Host"] = host
        req = urllib.request.Request(server.url + "api/" + name, data=b"[]", method="POST",
                                     headers=headers)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert call("info") == 200
        assert call("info", token="wrong") == 403
        assert call("info", host="evil.example:80") == 403          # DNS rebinding
        assert call("_start_load") == 404                           # private method
    finally:
        server.shutdown()
        server.server_close()
