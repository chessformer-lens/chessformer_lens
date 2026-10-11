"""Serve the app's page through chessformer_lens.web for test_ui_browser.py,
with JS errors reported to the console and every bridge call logged.

    python tests/ui_server.py PORT ALIAS|- [EXTRA_JS] [CALLS_LOG]

ALIAS "-" starts with no model (the picker). EXTRA_JS is appended to the page;
every bridge call's method name is appended to CALLS_LOG.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import toy_adapter  # noqa: E402,F401  registers the toy models
from chessformer_lens.bridge import LensApi  # noqa: E402
from chessformer_lens.web import Handler, make_server  # noqa: E402

port = int(sys.argv[1])
alias = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
extra = Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else ""
calls = Path(sys.argv[4]) if len(sys.argv) > 4 else None

ERRORS = """<script>
window.addEventListener('error', e => console.error('UNCAUGHT ' + e.message + ' @' + e.lineno));
window.addEventListener('unhandledrejection',
  e => console.error('UNHANDLED ' + (e.reason && e.reason.stack || e.reason)));
</script>"""


class Logged(Handler):
    def called(self, name):
        if calls is not None:
            with open(calls, "a") as fh:
                fh.write(name + "\n")


server = make_server(LensApi(alias=alias), "127.0.0.1", port, handler=Logged, extra_js=extra)
server.RequestHandlerClass.html = server.RequestHandlerClass.html.replace("<head>", "<head>" + ERRORS, 1)
server.serve_forever()
