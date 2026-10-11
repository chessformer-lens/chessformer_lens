"""Finding, recognising and downloading lc0 nets, without touching the network:
the downloader is pointed at a small file on a local HTTP server."""
import hashlib
import http.server
import threading
from functools import partial

import pytest

from chessformer_lens import engine
from chessformer_lens.adapters import lc0


def _sized(path, size):
    """A sparse file of exactly `size` bytes (recognition is by size)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(size)
    return path


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """No ./weights, an empty cache, no $CHESSFORMER_WEIGHTS."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHESSFORMER_WEIGHTS", raising=False)
    monkeypatch.setenv("CHESSFORMER_CACHE", str(tmp_path / "cache"))
    return tmp_path


def test_nets_are_recognised_by_size(dirs):
    for name, net in lc0._LC0_NETS.items():
        assert lc0.identify_lc0_net(_sized(dirs / f"{name}.pb.gz", net["size"])) == name
    assert lc0.identify_lc0_net(_sized(dirs / "other.pb.gz", 1234)) is None
    assert lc0.identify_lc0_net(dirs / "missing.pb.gz") is None


def test_lookup_order_and_the_old_file_name(dirs, monkeypatch):
    base, tuned = lc0._LC0_NETS["bt4"], lc0._LC0_NETS["bt4-332"]
    assert lc0._find_lc0_net("bt4") is None
    # the old fixed name counts only for the net its size says it is
    legacy = _sized(dirs / "weights" / lc0._LEELA_LEGACY_FILE, tuned["size"])
    assert lc0._find_lc0_net("bt4-332") == legacy.relative_to(dirs)
    assert lc0._find_lc0_net("bt4") is None
    cached = _sized(dirs / "cache" / base["file"], base["size"])
    assert lc0._find_lc0_net("bt4") == cached
    own = _sized(dirs / "mine" / tuned["file"], tuned["size"])
    monkeypatch.setenv("CHESSFORMER_WEIGHTS", str(dirs / "mine"))
    assert lc0._find_lc0_net("bt4-332") == own          # $CHESSFORMER_WEIGHTS comes first


@pytest.fixture
def served(tmp_path):
    """A local HTTP server with one small file; yields (url, payload)."""
    root = tmp_path / "srv"
    root.mkdir()
    payload = bytes(range(256)) * 4000                      # ~1 MB
    (root / "net.pb.gz").write_bytes(payload)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/net.pb.gz", payload
    srv.shutdown()


def test_download_checks_the_hash(dirs, served, monkeypatch):
    url, payload = served
    fake = {**lc0._LC0_NETS["bt4"], "url": url, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(lc0._LC0_NETS, "bt4", fake)
    path = lc0.download_lc0_net("bt4")
    assert path.read_bytes() == payload and path.parent == dirs / "cache"
    assert not list((dirs / "cache").glob("*.part"))

    monkeypatch.setitem(lc0._LC0_NETS, "bt4", {**fake, "sha256": "0" * 64})
    path.unlink()
    with pytest.raises(RuntimeError, match="checksum"):
        lc0.download_lc0_net("bt4")
    assert not list((dirs / "cache").iterdir())             # nothing half-written left behind


def test_no_download_refuses(dirs, monkeypatch):
    monkeypatch.setenv("CHESSFORMER_NO_DOWNLOAD", "1")
    with pytest.raises(FileNotFoundError, match="CHESSFORMER_NO_DOWNLOAD"):
        engine.LeelaEngine("bt4", device="cpu")
