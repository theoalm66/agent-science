"""Run LLM-written Python in a separate process with time and memory limits.

This is containment against accidents (runaway loops, huge arrays), NOT a
security boundary: the script runs as your user and can reach the network and
filesystem. For untrusted models run the whole agent inside a container.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
        try:
            proc = subprocess.run(
                # -I ignores PYTHON* env vars (so PYTHONIOENCODING would be dropped); -X utf8
                # forces UTF-8 stdio, otherwise Windows uses cp1252 and print("≈") crashes.
                [sys.executable, "-I", "-X", "utf8", str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return RunResult(False, out[-MAX_OUTPUT_CHARS:], "", None, timed_out=True)
    stdout = proc.stdout[-MAX_OUTPUT_CHARS:]
    stderr = proc.stderr[-MAX_OUTPUT_CHARS:]
    return RunResult(proc.returncode == 0, stdout, stderr, parse_result(stdout))
