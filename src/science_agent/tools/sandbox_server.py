"""Sandbox runner: executes model-written Python for Pandora, one script at a time.

Meant to run in its own container with no network access, a read-only filesystem, a
non-root user and memory/CPU/process limits (see docker-compose.yml). Pandora reaches it
over an internal Docker network and calls POST /run with {"code": ..., "timeout": ...}.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from .sandbox import run_python

MAX_CODE_BYTES = 200_000
MAX_TIMEOUT = 300.0

log = logging.getLogger(__name__)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._json({"ok": True})
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/run":
            return self._json({"error": "Not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_CODE_BYTES:
            return self._json({"error": "Script too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            code = str(body["code"])
            timeout = min(float(body.get("timeout") or 60), MAX_TIMEOUT)
        except (ValueError, KeyError, TypeError):
            return self._json({"error": "Expected {code, timeout}"}, 400)
        r = run_python(code, timeout)
        self._json({"ok": r.ok, "stdout": r.stdout, "stderr": r.stderr, "result": r.result,
                    "timed_out": r.timed_out})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox-runner", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Single-threaded on purpose: scripts run one after another, never side by side.
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Sandbox runner listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
