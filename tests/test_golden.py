"""Every legacy output of both engines must be unchanged: bit for bit when the
torch version matches the goldens', to a tolerance otherwise. See battery.py."""
import os

import pytest
import torch

import battery

pytestmark = pytest.mark.golden

ENGINES = [n for n in ("maia3-5m", "bt4") if (battery.GOLDEN_DIR / f"{n}.json.gz").exists()]


@pytest.fixture(scope="module", params=ENGINES)
def golden_run(request, tmp_path_factory):
    name = request.param
    if name == "bt4" and battery.bt4_weights() is None:
        pytest.skip("BT4 weights not found (set CHESSFORMER_WEIGHTS)")
    meta, tree, arrays = battery.load(name)
    eng = battery.make_engine(name, tmp_path_factory.mktemp("act"))
    new_tree, new_arrays = battery.encode(battery.run(eng))
    rtol = os.environ.get("CHESSFORMER_GOLDEN_RTOL")
    if rtol is None and meta.get("torch") != torch.__version__:
        rtol = "1e-5"
    return name, tree, arrays, new_tree, new_arrays, (float(rtol) if rtol else None)


def _sections():
    keys = set()
    for n in ENGINES:
        _, tree, _ = battery.load(n)
        keys |= set(tree)
    return sorted(keys)


@pytest.mark.parametrize("section", _sections())
def test_section_unchanged(golden_run, section):
    name, tree, arrays, new_tree, new_arrays, rtol = golden_run
    if section not in tree:
        pytest.skip(f"{name} has no {section}")
    problems = battery.diff(tree[section], arrays, new_tree.get(section), new_arrays,
                            path=f"{name}:{section}", rtol=rtol)
    assert not problems, f"{len(problems)} mismatches:\n" + "\n".join(problems[:40])
