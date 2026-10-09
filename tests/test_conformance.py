"""The contract every model adapter must keep, checked on each one available:
the toy models always, Maia-3 when its weights are cached, BT4 when its
weights are present. A new adapter passing this file works with every engine
method and frontend."""
import os
import subprocess
import sys

import pytest
import torch

import battery
from chessformer_lens.core import Conditioning, real_square
from chessformer_lens.engine import load_engine


def _maia_cached() -> bool:
    try:
        from maia3 import model_registry as r
        r.resolve_checkpoint_path(r.resolve_model_spec("maia3-5m"), local_files_only=True)
        return True
    except Exception:
        return False


ALIASES = ["toy", "toy-causal"]
if _maia_cached():
    ALIASES.append("maia3-5m")
if battery.bt4_weights() is not None:
    ALIASES.append("bt4")

POSITIONS = ["start", "ruy_black", "promo_black"]


@pytest.fixture(scope="module", params=ALIASES)
def eng(request, tmp_path_factory):
    kw = {"checkpoint_path": battery.bt4_weights()} if request.param == "bt4" else {}
    return load_engine(request.param, device="cpu",
                       activation_dir=str(tmp_path_factory.mktemp("act")), **kw)


def _rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / (b.norm() + 1e-12))


@pytest.mark.parametrize("pos", POSITIONS)
def test_layout_matches_activations(eng, pos):
    b = battery.board_for(pos)
    layout = eng.layout(b)
    _, cache = eng.run_with_cache(b, 1500)
    for name, t in cache.items():
        assert t.shape[0] == len(layout), f"{name}: {tuple(t.shape)} vs {len(layout)} tokens"
    if layout.is_board:
        assert len(layout) == 64
        for i, tok in enumerate(layout.tokens):
            assert tok.kind == "square" and tok.square == real_square(i, b.turn)


@pytest.mark.parametrize("pos", POSITIONS)
def test_head_writes_reconstruct_attention(eng, pos):
    b = battery.board_for(pos)
    for L in range(eng.arch.n_layers):
        writes = eng.head_writes(b, 1500, layer=L)                 # asserts internally too
        dec = eng.adapter.decompose_attention(
            L, eng._activations[eng.adapter.attn_input_key(L)].to(eng.device), Conditioning(1500))
        recon = writes.sum(0) + (dec.b_O.cpu() if dec.b_O is not None else 0)
        assert _rel(recon, eng._activations[f"attn_{L:02d}"][0]) < 1e-5


def test_neuron_writes_sum_to_mlp(eng):
    b = battery.board_for("ruy_black")
    for L in range(eng.arch.n_layers):
        hidden = eng._mlp_hidden(b, 1500, layers=[L])[L]
        out = eng.adapter.mlp_out(L)
        recon = hidden @ out.weight.t() + (out.bias if out.bias is not None else 0)
        assert _rel(recon.cpu(), eng._activations[f"mlp_{L:02d}"][0]) < 1e-5
        assert hidden.shape[1] == eng.arch.d_mlp


@pytest.mark.parametrize("pos", POSITIONS)
def test_unembed_at_final_readout_is_the_policy(eng, pos):
    b = battery.board_for(pos)
    logits, _ = eng.run_with_hooks(b, 1500, return_type="logits")
    _, cache = eng.run_with_cache(b, 1500)
    final = eng._lens_steps()[-1][0]
    lens = eng.logit_lens(cache[final])
    assert lens.shape == logits.shape == (len(eng.vocab),)
    assert torch.allclose(lens.cpu(), logits.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("pos", POSITIONS + ["castle_both", "promo_white"])
def test_every_legal_move_round_trips(eng, pos):
    b = battery.board_for(pos)
    for mv in b.legal_moves:
        info = eng.move_info(b, mv)
        assert eng.to_move(b, info["idx"]) == mv
        # A square pair carries no promotion piece, and lc0 encodes castling as
        # king-takes-rook (e1h1, which python-chess also reads as castling).
        back = eng.to_move(b, (info["from"], info["to"]))
        assert back in b.legal_moves
        assert mv.promotion or b.san(back) == b.san(mv)
        assert eng.to_move(b, info["san"]) == mv


def test_policy_covers_legal_moves_and_value_is_normalized(eng):
    b = battery.board_for("ruy_black")
    res = eng.evaluate(b, 1500)
    assert {u for u, _ in res["policy"]} == {m.uci() for m in b.legal_moves}
    assert abs(sum(p for _, p in res["policy"]) - 1) < 1e-4
    v = res["value"]
    assert 0.0 <= v.expected <= 1.0
    assert (v.wdl is None) == (eng.caps.value == "scalar")
    assert (res["wdl"] is None) == (eng.caps.value == "scalar")
    assert set(res["aux"]) == set(eng.caps.aux_heads)


def test_conditioning_object_matches_legacy_pair(eng):
    b = battery.board_for("castle_both")
    a = eng.evaluate(b, 1500, 1300)["_logits"]
    c = eng.evaluate(b, Conditioning(self_elo=1500, oppo_elo=1300))["_logits"]
    assert torch.equal(a, c)


def test_evaluate_batch_matches_and_keeps_snapshot(eng):
    b1, b2, b3 = (battery.board_for(p) for p in ("start", "ruy_black", "promo_white"))
    eng.evaluate(b1, 1500)
    snap = {k: v.clone() for k, v in eng._activations.items()}
    aux = dict(eng._aux)
    batch = eng.evaluate_batch([b2, b3], 1500)
    assert eng._aux == aux and snap.keys() == eng._activations.keys()
    assert all(torch.equal(snap[k], eng._activations[k]) for k in snap)
    for b, got in zip((b2, b3), batch):
        want = eng.evaluate(b, 1500)
        assert [u for u, _ in got["policy"]][:3] == [u for u, _ in want["policy"]][:3]
        pw = dict(want["policy"])
        assert all(abs(p - pw[u]) < 1e-4 for u, p in got["policy"])
        assert abs(got["value"].expected - want["value"].expected) < 1e-4


def test_no_capture_leaves_snapshot(eng):
    eng.evaluate(battery.board_for("start"), 1500)
    snap = {k: v.clone() for k, v in eng._activations.items()}
    with eng.no_capture():
        res = eng.evaluate(battery.board_for("promo_black"), 1500)
    assert res["policy"]
    assert all(torch.equal(snap[k], eng._activations[k]) for k in snap)


def test_attention_is_a_distribution_over_its_terms(eng):
    b = battery.board_for("ruy_black")
    for L in {0, eng.arch.n_layers - 1}:
        a = eng.attention(b, 1500, layer=L, head=eng.arch.n_heads - 1)
        attn = torch.tensor(a["attn"])
        assert attn.shape == (len(eng.layout(b)),) * 2
        assert torch.allclose(attn.sum(-1), torch.ones(attn.shape[0]), atol=1e-5)
        assert a["term_names"] == list(eng.caps.attn_terms)
        assert (a["gab"] is not None) == ("gab" in a["term_names"])
        if eng.caps.causal:
            assert torch.all(attn.triu(1) == 0)
            assert a["mask"] is not None


def test_gab_methods_follow_the_capability(eng):
    b = battery.board_for("start")
    if eng.has_generated_bias:
        T = eng.gab_templates()
        c = eng.gab_coeffs(b, 1500, layer=0)
        bias = eng.gab_bias(b, 1500, layer=0)
        recon = torch.einsum("hi,iqk->hqk", c, T)
        assert _rel(recon, bias) < 1e-4
    else:
        with pytest.raises(RuntimeError):
            eng.gab_templates()
        with pytest.raises(RuntimeError):
            eng.gab_bias(b, 1500, layer=0)


def test_works_without_maia3():
    """Everything but Maia-3 itself must import and run with maia3 absent, and
    asking for a Maia-3 model must say how to install it."""
    code = f"""
import sys
sys.modules["maia3"] = None            # import maia3 -> ModuleNotFoundError
sys.path.insert(0, {str(battery.REPO / "tests")!r})
import toy_adapter                     # registers the toy model
import chess
from chessformer_lens.engine import load_engine, resolve_engine, format_engine_list
eng = load_engine("toy", device="cpu")
assert eng.evaluate(chess.Board(), 1500)["policy"]
try:
    resolve_engine("maia3-5m")
except ModuleNotFoundError as exc:
    assert "pip install" in str(exc), exc
else:
    raise SystemExit("resolving a Maia-3 alias without maia3 should fail loudly")
assert "maia3 package" in format_engine_list()
bt4 = {str(battery.bt4_weights()) if battery.bt4_weights() else None!r}
if bt4:
    from chessformer_lens.engine import LeelaEngine
    assert LeelaEngine(device="cpu", checkpoint_path=bt4).evaluate(chess.Board())["policy"]
assert "maia3" not in sys.modules or sys.modules["maia3"] is None
print("ok")
"""
    env = {**os.environ, "PYTHONPATH": str(battery.REPO)}
    r = subprocess.run([sys.executable, "-c", code], cwd=battery.REPO, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]
