"""The app's real page in headless Chrome, its pywebview bridge shimmed onto a
real LensApi over HTTP (ui_server.py). Opt-in — it needs Chrome and ~20 s a
model:

    CHESSFORMER_BROWSER_TESTS=1 pytest tests/test_ui_browser.py

Set CHESSFORMER_CHROME to the browser binary if it is not in a usual place.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from test_conformance import _maia_cached

_CANDIDATES = [os.environ.get("CHESSFORMER_CHROME"),
               "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
               shutil.which("google-chrome"), shutil.which("chromium"),
               shutil.which("chromium-browser")]
CHROME = next((c for c in _CANDIDATES if c and Path(c).exists()), None)
pytestmark = pytest.mark.skipif(
    not (os.environ.get("CHESSFORMER_BROWSER_TESTS") and CHROME),
    reason="opt-in: set CHESSFORMER_BROWSER_TESTS=1 (needs Chrome)")

HERE = Path(__file__).resolve().parent

# After boot: open the move microscope on the top move, ablate head L0·h0, then
# report what the page shows.
INTERACT = r"""
(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = id => document.getElementById(id);
  const vis = id => $(id) ? getComputedStyle($(id)).display : 'missing';
  for (let i = 0; i < 400 && !booted; i++) await wait(100);
  await wait(2500);
  const top = document.querySelector('#policy .prow');
  if (top) top.click();
  await wait(6000);
  $('ablbtn').click();
  await wait(4000);
  console.log('DIAG ' + JSON.stringify({
    is_board: IS_BOARD,
    att_cells: $('att_qk').children.length + $('att_gab').children.length + $('att_sum').children.length,
    neuron_cells: $('att_neuron').children.length,
    attmsg: vis('attmsg'), attset: vis('attset'), neumsg: vis('neumsg'), neuboard: vis('neuboard'),
    wdltitle: $('wdltitle').textContent, poltitle: $('poltitle').textContent,
    grid_cells: document.querySelectorAll('#ablgrid .agc.cell').length,
    curve_points: document.querySelectorAll('#mlsvg circle').length,
    carrier_rows: document.querySelectorAll('#nrows .nrow').length,
    carrier_canvases: document.querySelectorAll('#nrows canvas').length,
  }));
})();
"""

# No model named: read the picker, try a bad alias, then pick the toy.
PICK = r"""
(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = id => document.getElementById(id);
  for (let i = 0; i < 200 && $('picker').style.display !== 'flex'; i++) await wait(100);
  await wait(300);
  const families = [...document.querySelectorAll('.pkfam')].map(e => e.textContent);
  await chooseModel('no-such-model');
  const err = $('pickererr').textContent;
  await chooseModel('toy');
  for (let i = 0; i < 400 && !booted; i++) await wait(100);
  await wait(1500);
  console.log('DIAG ' + JSON.stringify({families, err, picker: $('picker').style.display,
    model: $('modelinfo').textContent, attmsg: getComputedStyle($('attmsg')).display}));
})();
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_page(tmp_path, alias, script, budget_ms=60000):
    """Load the app for `alias` ("-" = none) with `script` appended; returns
    (the script's DIAG dict, uncaught JS errors, the bridge methods called)."""
    port = _free_port()
    js, calls, log, shot = (tmp_path / n for n in ("extra.js", "calls.log", "chrome.log", "shot.png"))
    js.write_text(script)
    calls.write_text("")
    server = subprocess.Popen([sys.executable, str(HERE / "ui_server.py"), str(port), alias,
                               str(js), str(calls)],
                              env={**os.environ, "CHESSFORMER_DEVICE": "cpu"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except OSError:
                time.sleep(.1)
        with open(log, "w") as fh:
            chrome = subprocess.Popen(
                [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                 "--no-default-browser-check", f"--user-data-dir={tmp_path / 'profile'}",
                 "--window-size=1560,900", "--hide-scrollbars", f"--virtual-time-budget={budget_ms}",
                 "--enable-logging=stderr", "--v=0", f"--screenshot={shot}",
                 f"http://127.0.0.1:{port}/"],
                stdout=subprocess.DEVNULL, stderr=fh)
            deadline = time.time() + 240
            while time.time() < deadline and not (shot.exists() and shot.stat().st_size):
                time.sleep(.2)
            time.sleep(.5)
            chrome.kill()
            chrome.wait()
    finally:
        server.kill()
        server.wait()
    text = log.read_text(errors="replace")
    m = re.search(r'DIAG (\{.*\})", source', text)
    assert m, "the page never reported:\n" + text[-3000:]
    errors = [ln for ln in text.splitlines() if re.search("UNCAUGHT|UNHANDLED|Uncaught", ln)]
    return json.loads(m.group(1)), errors, calls.read_text().split()


def test_non_board_model_shows_notices_and_draws_no_board(tmp_path):
    diag, errors, calls = run_page(tmp_path, "toy", INTERACT)
    assert not errors
    assert diag["is_board"] is False
    assert diag["att_cells"] == 0 and diag["neuron_cells"] == 0
    assert diag["attmsg"] != "none" and diag["attset"] == "none"
    assert diag["neumsg"] != "none" and diag["neuboard"] == "none"
    assert diag["wdltitle"].startswith("Expected score")
    assert "ablated" in diag["poltitle"]
    assert diag["grid_cells"] > 0 and diag["curve_points"] > 0
    assert diag["carrier_rows"] > 0 and diag["carrier_canvases"] == 0
    assert "attention" not in calls and "neuron" not in calls


def test_board_model_draws_its_boards(tmp_path):
    if not _maia_cached():
        pytest.skip("Maia-3 5M weights not cached")
    diag, errors, calls = run_page(tmp_path, "maia3-5m", INTERACT)
    assert not errors
    assert diag["is_board"] is True
    assert diag["att_cells"] == 3 * 64 and diag["neuron_cells"] == 64
    assert diag["attmsg"] == "none" and diag["neumsg"] == "none"
    assert diag["wdltitle"].startswith("Win / Draw / Loss")
    assert diag["carrier_canvases"] == diag["carrier_rows"] > 0
    assert "attention" in calls and "neuron" in calls


def test_picker_reports_bad_names_and_loads_the_choice(tmp_path):
    diag, errors, _ = run_page(tmp_path, "-", PICK)
    assert not errors
    assert diag["families"] == sorted(diag["families"], key=str.lower)
    assert {"Leela Chess Zero", "Maia-3"} <= set(diag["families"])
    assert "Unknown model" in diag["err"]
    assert diag["picker"] == "none" and diag["model"].startswith("Toy")


MANY = r"""
(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = id => document.getElementById(id);
  for (let i = 0; i < 1200 && !booted; i++) await wait(100);
  for (let i = 0; i < 300 && busy; i++) await wait(100);
  const start = cur.fen;
  $('mpwhere').click();
  const tab = $('mpbody').style.display, empty = $('mptilegrid').textContent;
  await mpScan();
  for (let i = 0; i < 6000; i++) { const s = await API.scan_status(); if (s.state === 'done') break; await wait(20); }
  await mpTick();
  for (let i = 0; i < 200 && !mpData; i++) await wait(50);
  const cardNeuron = mpData && mpData.layer === 0 && mpData.neuron === 0;   // the card's L0·n0
  document.querySelectorAll('#mpgrid .agc.cell')[0].click();             // L0's strongest
  for (let i = 0; i < 200 && !(mpData && mpData.neuron === mpReport.grid[0][0].neuron); i++) await wait(50);
  const cells = document.querySelectorAll('#mpgrid .agc.cell').length,
        strong = document.querySelectorAll('#mpgrid .strong').length,
        picked = $('nidx').value == mpReport.grid[0][0].neuron && $('nlayer').value == 0,
        tiles = document.querySelectorAll('.mptile').length, stat = $('mpstat').textContent,
        pieces = document.querySelectorAll('.mptile img.pc').length;
  const first = mpData.tiles[0];
  await mpOpenTile(0);
  for (let i = 0; i < 300 && busy; i++) await wait(100);
  const opened = cur.fen, strip = $('mpstrip').offsetHeight, drawer = $('mlens').classList.contains('open');
  await mpBack();
  for (let i = 0; i < 300 && busy; i++) await wait(100);
  console.log('DIAG ' + JSON.stringify({tab, empty, cardNeuron, picked, cells, strong, tiles, stat, pieces, first: first.fen,
    opened, strip, drawer, back: cur.fen === start, stripAfter: $('mpstrip').offsetHeight,
    layers: MODEL_INFO.num_blocks}));
})();
"""


@pytest.mark.skipif(not _maia_cached(), reason="Maia-3 5M weights not cached")
def test_many_positions_tab(tmp_path):
    diag, errors, calls = run_page(tmp_path, "maia3-5m", MANY, budget_ms=240000)
    assert not errors
    assert diag["tab"] == "flex" and "scan a pool" in diag["empty"]
    assert diag["cardNeuron"]                    # opened on the Neurons card's neuron
    assert diag["cells"] == 10 * diag["layers"] and diag["strong"] == 1 and diag["picked"]
    assert diag["tiles"] == 8 and diag["pieces"] > 8 * 10
    assert "scanned 1,000 of 1,000 ordinary positions" in diag["stat"] and "elo 1500" in diag["stat"]
    assert diag["opened"] == diag["first"] and diag["strip"] > 0 and not diag["drawer"]
    assert diag["back"] and diag["stripAfter"] == 0
    assert {"scan_start", "scan_status", "scan_report", "neuron_positions",
            "open_position", "go_back"} <= set(calls)
