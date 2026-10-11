"""
web.py — the app in a web browser instead of a native window.

    chessformer_lens --web                     # opens http://127.0.0.1:<port>/
    chessformer_lens bt4 --web --port 8765

    from chessformer_lens import web           # in a notebook or Colab
    server = web.serve("bt4")                  # returns at once; the server runs on
    server.url                                 # a background thread
    server.shutdown()
    web.serve(engine=eng)                      # the engine your notebook already loaded

The page is ui.py's own, with `window.pywebview.api` shimmed onto HTTP: every
`api.<method>(...)` the page makes becomes `POST /api/<method>` with the
arguments as a JSON list, answered by a `LensApi` (bridge.py) in this process.
Only LensApi's public methods are reachable. In Colab the page opens through
Colab's port proxy in a new window (the app needs about 1400x780, more than a
cell gives it); elsewhere in your default browser.

One model and one board per server: a local tool, not a site for many users.
It listens on 127.0.0.1 unless told otherwise. `host="0.0.0.0"` puts the
whole API on your network, including loading models, which can download
weights. Other websites open in your browser can't drive it: every call
carries a per-server token in a custom header (so a cross-site request would
need a CORS preflight, which this server never grants), and on 127.0.0.1 a
request must name the loopback host (no DNS rebinding).
"""
import json
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .bridge import LensApi
from .ui import INDEX_HTML

# pywebview's bridge, over fetch: a Python exception rejects the call's
# promise, as pywebview does.
SHIM = """<script>
window.pywebview = {api: new Proxy({}, {get: (_, name) => async (...args) => {
  const r = await fetch('/api/' + name, {method: 'POST', body: JSON.stringify(args),
                                         headers: {'X-Chessformer-Token': '__TOKEN__'}});
  const j = await r.json();
  if (!r.ok || (j && j.__exc__)) throw new Error(j && (j.__exc__ || j.error) || r.statusText);
  return j;
}})};
</script>"""
TOKEN_HEADER = "X-Chessformer-Token"
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


def page(token: str, extra_js: str = "") -> str:
    """ui.py's page with the shim in its head (and `extra_js` at the end)."""
    html = INDEX_HTML.replace("<head>", "<head>" + SHIM.replace("__TOKEN__", token), 1)
    if extra_js:
        html = html.replace("</body>", "<script>" + extra_js + "</script></body>", 1)
    return html


def public_methods(api) -> frozenset:
    """The bridge methods the page may call: LensApi's public methods."""
    cls = type(api)
    return frozenset(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))


class Handler(BaseHTTPRequestHandler):
    """GET / serves the page; POST /api/<method> calls the bridge. A server
    made by `make_server` sets `api`, `html`, `allowed`, `token` and
    `check_host` on its own subclass."""
    api = None
    html = ""
    allowed = frozenset()
    token = ""
    check_host = True

    def log_message(self, *_):
        pass

    def _send(self, body: str, ctype: str, code: int = 200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, default=str), "application/json", code)

    def _host_ok(self):
        """On loopback, the request must name a loopback host: a page on some
        other domain that resolves to 127.0.0.1 (DNS rebinding) is refused."""
        if not self.check_host:
            return True
        host = self.headers.get("Host") or ""
        host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if host in LOOPBACK:
            return True
        self._json({"error": "forbidden host"}, 403)
        return False

    def do_GET(self):
        if not self._host_ok():
            return
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            self._send(self.html, "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._host_ok():
            return
        if not secrets.compare_digest(self.headers.get(TOKEN_HEADER, ""), self.token):
            self._json({"error": "missing or wrong token"}, 403)
            return
        name = self.path[len("/api/"):] if self.path.startswith("/api/") else ""
        if name not in self.allowed:
            self._json({"error": f"no bridge method {name!r}"}, 404)
            return
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))
                              or b"[]")
            if not isinstance(args, list):
                raise ValueError("arguments must be a JSON list")
        except ValueError as exc:
            self._json({"error": f"bad request: {exc}"}, 400)
            return
        self.called(name)
        try:
            res = getattr(self.api, name)(*args)
        except Exception as exc:                   # the page's promise rejects
            res = {"__exc__": f"{type(exc).__name__}: {exc}"}
        self._json(res)

    def called(self, name):
        """Hook for subclasses (the browser tests log every call)."""


def make_server(api=None, host="127.0.0.1", port=0, *, handler=Handler, extra_js=""):
    """A ThreadingHTTPServer for `api` (a LensApi; a new one, with no model, if
    None). port=0 picks a free port; the server's `url` says which, and
    `token` is what the page sends with every call."""
    api = api if api is not None else LensApi()
    token = secrets.token_urlsafe(24)
    loopback = host in ("127.0.0.1", "localhost", "::1")
    cls = type("BoundHandler", (handler,), {
        "api": api, "html": page(token, extra_js), "allowed": public_methods(api),
        "token": token, "check_host": loopback and not _colab()})
    server = ThreadingHTTPServer((host, port), cls)
    server.daemon_threads = True
    shown = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    server.url = f"http://{shown}:{server.server_port}/"
    server.api, server.token = api, token
    return server


def _colab():
    """In a Colab kernel (which always has google.colab imported)."""
    return "google.colab" in sys.modules


def serve(alias=None, *, engine=None, host="127.0.0.1", port=0, open_browser=True,
          block=False):
    """Serve the app for `alias` (no alias: the page opens on the model picker),
    or for `engine`, one already loaded, and open it. The app and your own
    cells then share that engine, so let a panel finish before running a cell
    on it. With block=False (the default, for notebooks) the server runs on a
    daemon thread and is returned; `server.shutdown()` stops it. With
    block=True this serves until interrupted."""
    server = make_server(LensApi(alias=alias, engine=engine), host, port)
    print(f"chessformer_lens: the app is at {server.url}", flush=True)
    if _colab():
        from google.colab import output
        output.serve_kernel_port_as_window(server.server_port, path="/",
                                           anchor_text="Open the chessformer_lens app")
    elif open_browser:
        webbrowser.open(server.url)
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
