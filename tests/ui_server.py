"""Serve the app's page over HTTP with `window.pywebview.api` shimmed onto a
real LensApi, so a headless browser can drive the real UI (test_ui_browser.py).

    python tests/ui_server.py PORT ALIAS|- [EXTRA_JS] [CALLS_LOG]

ALIAS "-" starts with no model (the picker). EXTRA_JS is appended to the page;
every bridge call's method name is appended to CALLS_LOG.
"""
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import toy_adapter  # noqa: E402,F401  registers the toy models
from chessformer_lens.bridge import LensApi  # noqa: E402
from chessformer_lens.ui import INDEX_HTML  # noqa: E402

port = int(sys.argv[1])
alias = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
extra = Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else ""
calls = Path(sys.argv[4]) if len(sys.argv) > 4 else None
api = LensApi(alias=alias)

SHIM = """<script>
window.pywebview = {api: new Proxy({}, {get: (_, name) => async (...args) => {
  const r = await fetch('/api/' + name, {method: 'POST', body: JSON.stringify(args)});
  const j = await r.json();
  if (j && j.__exc__) throw new Error(j.__exc__);
  return j;
}})};
window.addEventListener('error', e => console.error('UNCAUGHT ' + e.message + ' @' + e.lineno));
window.addEventListener('unhandledrejection',
  e => console.error('UNHANDLED ' + (e.reason && e.reason.stack || e.reason)));
</script>"""
PAGE = INDEX_HTML.replace("<head>", "<head>" + SHIM, 1).replace(
    "</body>", "<script>" + extra + "</script></body>", 1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, body, ctype):
        b = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        name = self.path[len("/api/"):]
        if calls is not None:
            with open(calls, "a") as fh:
                fh.write(name + "\n")
        args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "[]")
        try:
            res = getattr(api, name)(*args)
        except Exception as exc:
            traceback.print_exc()
            res = {"__exc__": f"{type(exc).__name__}: {exc}"}
        self._send(json.dumps(res), "application/json")


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
