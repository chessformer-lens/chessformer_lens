"""The frontends — app bridge, notebook widgets, figures — on both kinds of
model: the board views where tokens are the 64 squares, a plain "not
available yet" where they are not (the toy model), and everything else live."""
import time

import chess
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import battery  # noqa: E402
from chessformer_lens import interp_plot as ip  # noqa: E402
from chessformer_lens import interp_widget as iw  # noqa: E402
from chessformer_lens.bridge import LensApi  # noqa: E402
from chessformer_lens.engine import load_engine  # noqa: E402
from chessformer_lens.ui import INDEX_HTML  # noqa: E402
from test_conformance import _maia_cached  # noqa: E402

NOT_YET = "not available yet"


@pytest.fixture(scope="module")
def toy():
    return load_engine("toy", device="cpu")


@pytest.fixture(scope="module")
def maia():
    if not _maia_cached():
        pytest.skip("Maia-3 5M weights not cached")
    return load_engine("maia3-5m", device="cpu")


@pytest.fixture
def no_env_model(monkeypatch):
    monkeypatch.delenv("CHESSFORMER_MODEL", raising=False)
    monkeypatch.delenv("MAIA3_ALIAS", raising=False)


# ----- the app bridge ------------------------------------------------------------

def test_app_has_no_default_model(no_env_model):
    api = LensApi()
    info = api.info()
    assert info["needs_model"] and not info["ready"] and not info["loading"]
    families = {m["family"] for m in api.models()["models"]}
    assert {"Leela Chess Zero", "Maia-3"} <= families
    assert "error" in api.load_model("no-such-model")
    assert not api.info()["loading"]


def test_picked_model_loads_once(no_env_model):
    api = LensApi()
    assert api.load_model("toy")["loading"]
    for _ in range(200):
        if api.info()["ready"] or api.info()["error"]:
            break
        time.sleep(.05)
    info = api.info()
    assert info["ready"], info["error"]
    assert info["is_board"] is False and info["value_kind"] == "scalar" and not info["has_gab"]
    assert "restart" in api.load_model("toy-causal")["error"]


def test_bridge_payloads_are_model_neutral(toy):
    api = battery._bridge(toy)
    p = api.policy(1500, save=False)
    assert p["wdl"] is None and 0 <= p["value"]["expected"] <= 1
    assert p["layout"]["is_board"] is False and len(p["layout"]["tokens"]) == 8
    api.board, api.human, api.human_both = chess.Board(), chess.BLACK, False
    mv = api.engine_move(1500, 1.0)
    assert mv["engine_move"] and mv["engine_value"]["wdl"] is None
    for payload in (p, mv):
        assert not [k for k in payload if k.startswith("maia")]
    assert not hasattr(api, "maia_move")
    ab = api.ablate(1500, 0, 1)
    assert ab["value"]["wdl"] is None and ab["value_abl"]["wdl"] is None
    cp = api.compare_policy(1500, 900)
    assert cp["value_a"] and cp["value_b"]
    assert "error" in api.gab_templates()


def test_ui_page_is_model_neutral():
    # "maia_" / "MaiaApi" cannot occur in the base64 piece art ("_" is not base64)
    for name in ("maia_", "MaiaApi", "Maia rating", "Maia ("):
        assert name not in INDEX_HTML
    assert "Attention view not available yet" in INDEX_HTML
    assert "engine_move" in INDEX_HTML and "load_model" in INDEX_HTML


# ----- notebook widgets ---------------------------------------------------------------

def test_widgets_refuse_non_board_models(toy):
    b = battery.board_for("ruy_black")
    with pytest.raises(ValueError, match=NOT_YET):
        iw.attention_payload(toy, b)
    with pytest.raises(ValueError, match=NOT_YET):
        iw.gab_payload(toy, b)


def test_widget_payload_names_its_terms(maia):
    p = iw.attention_payload(maia, battery.board_for("ruy_black"))
    assert p["term_names"] == ["content", "gab"]
    assert p["gab"] and p["terms"] == {} and p["mask"] is None


# ----- figures ------------------------------------------------------------------------

@pytest.mark.parametrize("fn", ["plot_attention", "plot_attention_layer", "plot_attention_atlas"])
def test_board_figures_refuse_non_board_models(toy, fn):
    with pytest.raises(ValueError, match=NOT_YET):
        getattr(ip, fn)(toy, battery.board_for("start"), 1500)


FIGURES = {"plot_position": {"elo_b": 900}, "plot_logit_curve": {}, "plot_policy_curve": {},
           "plot_rank_curve": {}, "plot_move_microscope": {"elo_b": 900},
           "plot_carrier_heads": {}, "plot_move_report": {}}


@pytest.mark.parametrize("fn", sorted(FIGURES))
def test_other_figures_work_on_non_board_models(toy, fn):
    plt.close(getattr(ip, fn)(toy, battery.board_for("ruy_black"), 1500, **FIGURES[fn]))


@pytest.mark.parametrize("fn", sorted(FIGURES) + ["plot_attention", "plot_attention_layer",
                                                 "plot_attention_atlas", "plot_gab_mixture"])
def test_every_figure_works_on_board_models(maia, fn):
    kw = FIGURES.get(fn, {"layer": 1})
    plt.close(getattr(ip, fn)(maia, battery.board_for("ruy_black"), 1500, **kw))
