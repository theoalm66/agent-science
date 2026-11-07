"""Interim results of runs that have not finished, so an interrupted run can continue.

A run writes its partial assessment here after every stage. If it dies (network drop, model
server restart, Pandora restart) the file stays; "Resume" starts a new run that skips the stages
already done. The file is removed when the run finishes. No API keys are ever stored in it.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

MAX_AGE_DAYS = 14
_ID = re.compile(r"^[0-9a-f]{6,32}$")


class CheckpointStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, run_id: str) -> Path | None:
        return self.directory / f"{run_id}.json" if _ID.match(run_id or "") else None

    def save(self, run_id: str, owner: str, data: dict) -> None:
        """Atomic write: a crash during the write leaves the previous checkpoint intact."""
        path = self._path(run_id)
        if path is None:
            raise ValueError("Invalid run id")
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": run_id, "owner": owner, "saved_at": time.time(), **data}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def load(self, run_id: str) -> dict | None:
        path = self._path(run_id)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set_error(self, run_id: str, message: str) -> None:
        """Note why the run stopped, if it has a checkpoint."""
        data = self.load(run_id)
        if data is not None:
            data["error"] = message[:500]
            self.save(run_id, data.pop("owner"), {k: v for k, v in data.items() if k not in ("run_id", "saved_at", "owner")})

    def exists(self, run_id: str) -> bool:
        path = self._path(run_id)
        return bool(path and path.exists())

    def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        if path is not None:
            path.unlink(missing_ok=True)

    def list(self, can_see) -> list[dict]:
        """Short descriptions of the unfinished runs `can_see(owner)` allows, newest first.
        Checkpoints older than MAX_AGE_DAYS are deleted."""
        if not self.directory.is_dir():
            return []
        out = []
        for path in self.directory.glob("*.json"):
            data = self.load(path.stem)
            if data is None:
                continue
            if time.time() - float(data.get("saved_at") or 0) > MAX_AGE_DAYS * 86400:
                path.unlink(missing_ok=True)
                continue
            if not can_see(data.get("owner")):
                continue
            assessment = data.get("assessment") or {}
            done = [k for k in ("formalization", "literature", "limits", "computation", "critique") if assessment.get(k)]
            out.append({
                "run_id": data["run_id"],
                "idea": str(data.get("idea") or assessment.get("idea") or "")[:300],
                "saved_at": data.get("saved_at"),
                "done": done,
                "error": data.get("error") or "",
            })
        return sorted(out, key=lambda r: r["saved_at"] or 0, reverse=True)
