"""Self-check of the panel's static server.

Serving /js and /css turned _static() from "one known filename" into "a path
from the URL", which is a trust boundary: ../../etc/passwd and an absolute path
both have to be refused. These are the checks for that, plus that the split's
entry point and its stylesheet are actually reachable.

    python3 test_static.py
"""

import sys
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

import server


class StaticHandler(server.Handler):
    """Capture what would go on the wire instead of writing it."""

    def __init__(self):  # noqa: D107 - deliberately not calling super()
        self.status = None
        self.headers: dict = {}
        self.body = b""

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass

    def wfile_write(self, data):
        self.body += data

    # BaseHTTPRequestHandler writes through self.wfile, which is a property.
    @property
    def wfile(self):
        import io
        return _Sink(self)


class _Sink:
    def __init__(self, handler):
        self.handler = handler

    def write(self, data):
        self.handler.body += data

    def flush(self):
        pass


def get(name):
    h = StaticHandler()
    with mock.patch.object(server.Handler, "_json",
                           lambda self, payload, code=200: setattr(self, "status", code) or None):
        server.Handler._static(h, name)
    return h


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    root = server.HERE.resolve()

    # --- the split's files are reachable ------------------------------------
    for name in ("index.html", "js/core.js", "js/boot.js", "css/panel.css"):
        h = get(name)
        check(h.status == 200 and h.body, f"{name} deberia servirse 200 con contenido")

    # --- a missing file is a 404, not a crash -------------------------------
    h = get("js/no-existe.js")
    check(h.status == 404, f"un archivo ausente deberia dar 404, dio {h.status}")

    # --- path traversal ------------------------------------------------------
    for evil in ("../server.py", "../../etc/passwd", "js/../../server.py",
                 "/etc/passwd", "js/../../../etc/shadow"):
        h = get(evil)
        check(h.status == 404, f"traversal {evil!r} deberia dar 404, dio {h.status}")
        check(b"root:" not in h.body and b"def main" not in h.body,
              f"traversal {evil!r} devolvio contenido")

    # --- a symlink that points outside is still refused ---------------------
    link = root / "js" / "escape.js"
    created = False
    if not link.exists():
        try:
            link.symlink_to("/etc/hostname")
            created = True
        except OSError:
            pass
    if created:
        h = get("js/escape.js")
        check(h.status == 404, "un symlink fuera del panel deberia dar 404")
        link.unlink()

    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"{len(fails)} FALLAS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())