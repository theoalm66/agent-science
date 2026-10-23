"""Run LLM-written Python in a separate process with time and memory limits.

run_python is containment against accidents (runaway loops, huge arrays), NOT a
security boundary: the script runs as your user and can reach the network and
filesystem. When other people can start runs, use RemoteSandbox with the sandbox
runner in its own locked-down container (see docker-compose.yml).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

RESULT_MARKER = "RESULT_JSON:"
MAX_OUTPUT_CHARS = 20_000
MEMORY_LIMIT_BYTES = 4 * 1024**3


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    result: dict | None
    timed_out: bool = False

    def summary(self) -> str:
        if self.timed_out:
            return "Script timed out."
        if not self.ok:
            return f"Script failed:\n{self.stderr[-3000:]}"
        return self.stdout[-6000:]


def _limit_resources():  # pragma: no cover - runs in the child process
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def parse_result(stdout: str) -> dict | None:
    """Return the JSON after the last RESULT_JSON: marker line, if any."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                value = json.loads(line[len(RESULT_MARKER) :].strip())
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else {"value": value}
    return None


def run_python(code: str, timeout: float = 60) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="science-agent-") as tmp:
        script = Path(tmp) / "experiment.py"
        script.write_text(code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLBACKEND": "Agg",
            # One BLAS thread keeps virtual memory well under the address-space limit.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        # Windows needs these for os.urandom, temp files and numpy's DLL loading.
        for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            if name in os.environ:
                env[name] = os.environ[name]
        # A new session lets us kill the whole process group afterwards, so a script cannot leave
        # background processes behind (important when one sandbox serves several people).
        popen_kwargs = {"start_new_session": True, "preexec_fn": _limit_resources} if os.name == "posix" else {}
        # Output goes to files, not pipes: a background child inheriting a pipe would keep
        # communicate() waiting until the timeout.
        out_path, err_path = Path(tmp) / "stdout.txt", Path(tmp) / "stderr.txt"
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                # -I ignores PYTHON* env vars (so PYTHONIOENCODING would be dropped); -X utf8
                # forces UTF-8 stdio, otherwise Windows uses cp1252 and print("≈") crashes.
                [sys.executable, "-I", "-X", "utf8", str(script)],
                cwd=tmp,
                env=env,
                stdout=out,
                stderr=err,
                **popen_kwargs,
            )
            try:
                proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _kill_group(proc)
                proc.wait()
        stdout = _tail(out_path)
        stderr = _tail(err_path)
    if timed_out:
        return RunResult(False, stdout, "", None, timed_out=True)
    return RunResult(proc.returncode == 0, stdout, stderr, parse_result(stdout))


def _tail(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4 * MAX_OUTPUT_CHARS))
        return f.read().decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:]


def _kill_group(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        proc.kill()


class RemoteSandbox:
    """Runs scripts in a sandbox runner over HTTP (see sandbox_server.py). Same call shape as run_python."""

    def __init__(self, url: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/") + "/run"
        self.client = client or httpx.Client()

    def __call__(self, code: str, timeout: float = 60) -> RunResult:
        try:
            resp = self.client.post(self.url, json={"code": code, "timeout": timeout}, timeout=timeout + 30)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            return RunResult(False, "", f"Sandbox runner unreachable at {self.url}: {e}", None)
        return RunResult(
            bool(data.get("ok")), str(data.get("stdout") or ""), str(data.get("stderr") or ""),
            data.get("result") if isinstance(data.get("result"), dict) else None,
            timed_out=bool(data.get("timed_out")),
        )
