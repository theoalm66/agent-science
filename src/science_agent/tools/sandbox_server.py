"""Sandbox runner: executes model-written Python for Pandora, one script at a time.

Meant to run in its own container with no network at all, a read-only filesystem, a
non-root user and memory/CPU/process limits (see docker-compose.yml). Pandora reaches it
through a Unix socket on a shared volume (--socket) and calls POST /run with
{"code": ..., "timeout": ...}. Without a network the scripts cannot reach Pandora's API, where
they could otherwise claim any identity in the Tailscale-User-Login header.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

from .sandbox import run_python

MAX_CODE_BYTES = 200_000
MAX_TIMEOUT = 300.0

log = logging.getLogger(__name__)


class Handler(BaseHTTPRequestHandler):
    def address_string(self) -> str:
        # Over a Unix socket client_address is "" rather than (host, port).
        return self.client_address[0] if isinstance(self.client_address, tuple) else "unix"

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


class UnixHTTPServer(socketserver.UnixStreamServer):
    def __init__(self, path: str, handler):
        if os.path.exists(path):
            os.unlink(path)  # left over from a previous run
        super().__init__(path, handler)
        os.chmod(path, 0o666)  # Pandora connects as another uid

    def server_close(self) -> None:
        super().server_close()
        if os.path.exists(self.server_address):
            os.unlink(self.server_address)


def make_server(host: str, port: int, socket_path: str | None):
    # Single-threaded on purpose: scripts run one after another, never side by side.
    if socket_path:
        return UnixHTTPServer(socket_path, Handler), socket_path
    return HTTPServer((host, port), Handler), f"{host}:{port}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox-runner", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--socket", help="listen on this Unix socket instead of host:port")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server, where = make_server(args.host, args.port, args.socket)
    print(f"Sandbox runner listening on {where}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
