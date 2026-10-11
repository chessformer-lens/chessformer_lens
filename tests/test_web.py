"""The app over HTTP (web.py), without a browser: the page, the bridge calls,
and what the server refuses."""
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from chessformer_lens import web


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.delenv("CHESSFORMER_MODEL", raising=False)
    monkeypatch.delenv("MAIA3_ALIAS", raising=False)
    monkeypatch.setenv("CHESSFORMER_NO_DOWNLOAD", "1")
    monkeypatch.setenv("CHESSFORMER_CACHE", str(tmp_path / "cache"))
    srv = web.serve(open_browser=False)              # no model: the picker
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, name, *args, token=True, host=None, body=None):
    req = urllib.request.Request(srv.url + "api/" + name, method="POST",
                                 data=body if body is not None else json.dumps(args).encode())
    if token:
        req.add_header(web.TOKEN_HEADER, srv.token)
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_carries_the_shim_and_its_token(server):
    html = urllib.request.urlopen(server.url, timeout=10).read().decode()
    assert "window.pywebview" in html and server.token in html
    assert "Choose a model" in html                     # ui.py's own page


def test_bridge_calls(server):
    code, info = call(server, "info")
    assert code == 200 and info["needs_model"] and not info["ready"]
    code, models = call(server, "models")
    assert code == 200 and {"Leela Chess Zero", "Maia-3"} <= {m["family"] for m in models["models"]}
    code, res = call(server, "load_model", "no-such-model")
    assert code == 200 and "Unknown model" in res["error"]
    code, res = call(server, "info", 1, 2, 3)           # a Python error rejects the call
    assert code == 200 and res["__exc__"].startswith("TypeError")


def test_refusals(server):
    assert call(server, "info", token=False)[0] == 403                  # another site
    assert call(server, "info", host="evil.example:80")[0] == 403       # DNS rebinding
    assert call(server, "_load")[0] == 404                              # private
    assert call(server, "no_such_method")[0] == 404
    assert call(server, "info", body=b"{not json")[0] == 400
    assert call(server, "info", body=b'{"a": 1}')[0] == 400            # args must be a list
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server.url + "secret.txt", timeout=10)
    assert e.value.code == 404


def test_public_methods_are_the_bridge():
    from chessformer_lens.bridge import LensApi
    allowed = web.public_methods(LensApi())
    assert {"info", "models", "load_model", "policy"} <= allowed
    assert not any(n.startswith("_") for n in allowed)


def test_command_line(tmp_path):
    env = {"CHESSFORMER_NO_DOWNLOAD": "1", "CHESSFORMER_CACHE": str(tmp_path / "cache"),
           "PATH": "/usr/bin:/bin"}
    proc = subprocess.Popen([sys.executable, "-m", "chessformer_lens.app", "--web",
                             "--no-browser"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    try:
        line = proc.stdout.readline()
        url = re.search(r"http://127\.0\.0\.1:\d+/", line).group(0)
        html = urllib.request.urlopen(url, timeout=10).read().decode()
        assert "window.pywebview" in html
    finally:
        proc.kill()
        proc.wait()


def test_serve_an_engine_already_loaded():
    import battery
    if battery.bt4_weights() is None:
        pytest.skip("BT4 weights not found")
    from chessformer_lens.engine import LeelaEngine
    eng = LeelaEngine("bt4", checkpoint_path=battery.bt4_weights())
    srv = web.serve(engine=eng, open_browser=False)
    try:
        code, info = call(srv, "info")
        assert info["ready"] and not info["needs_model"] and not info["loading"]
        assert info["alias"] == eng.spec.display_name
        code, pol = call(srv, "policy", 0)
        assert code == 200 and pol["policy"] and pol["wdl"]
    finally:
        srv.shutdown()
        srv.server_close()
