"""Version history for the agent's editable text files: stage prompts, Claude Code skills and subagents.

The file on disk is always the live text, so the CLI, Claude Code and Pandora all use the same
prompt. The store keeps every saved version; activating a version writes it to the file. A version
counts as active when its text equals the file, so an edit made outside Pandora (an editor, a
git pull) shows up as "unversioned" until it is saved as a version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml

STAGE_HEADER = re.compile(r"^# Stage: (\w+)\s*$")


def normalize(text: str) -> str:
    """Line endings to \\n and exactly one trailing newline, so cosmetic differences are not versions."""
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_preserving_newlines(path: Path, text: str) -> None:
    newline = "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline=newline)
    os.replace(tmp, path)


def validate(key: str, text: str) -> None:
    """Reject edits that would break the pipeline or Claude Code; raises ValueError."""
    kind, name = key.split(":", 1)
    if not text.strip():
        raise ValueError("The text is empty")
    if kind == "prompt" and name != "system":
        first = next((line for line in text.splitlines() if line.strip()), "")
        m = STAGE_HEADER.match(first)
        if not m or m.group(1) != name:
            raise ValueError(f"A stage prompt must start with the line '# Stage: {name}'")
    if kind in ("skill", "agent"):
        if not text.startswith("---"):
            raise ValueError("A skill or agent file must start with YAML front matter (---)")
        parts = text.split("---", 2)
        try:
            meta = yaml.safe_load(parts[1]) if len(parts) == 3 else None
        except yaml.YAMLError as e:
            raise ValueError(f"The front matter is not valid YAML: {e}") from e
        if not isinstance(meta, dict) or not meta.get("name") or not meta.get("description"):
            raise ValueError("The front matter needs at least 'name' and 'description'")


class VersionStore:
    def __init__(self, path: Path, prompts_dir: Path, project_dir: Path):
        self.path = path
        self.prompts_dir = prompts_dir
        self.project_dir = project_dir
        self._lock = threading.Lock()

    # -- files -----------------------------------------------------------------

    def files(self) -> dict[str, Path]:
        out = {f"prompt:{p.stem}": p for p in sorted(self.prompts_dir.glob("*.md"))}
        out.update({f"skill:{p.parent.name}": p for p in sorted(self.project_dir.glob(".claude/skills/*/SKILL.md"))})
        out.update({f"agent:{p.stem}": p for p in sorted(self.project_dir.glob(".claude/agents/*.md"))})
        return out

    def _path(self, key: str) -> Path:
        path = self.files().get(key)
        if path is None:
            raise KeyError(f"Unknown item {key!r}")
        return path

    def read(self, key: str) -> str:
        return normalize(self._path(key).read_text(encoding="utf-8"))

    # -- store -----------------------------------------------------------------

    def _load(self) -> dict:
        if not self.path.exists():
            return {"items": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("items", {})
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _entry(n: int, text: str, note: str, source: str, created: str | None = None) -> dict:
        return {"n": n, "text": text, "note": note, "source": source, "created": created or _now()}

    def item(self, key: str, data: dict | None = None) -> dict:
        text = self.read(key)
        versions = (data or self._load())["items"].get(key, [])
        active = next((v["n"] for v in reversed(versions) if v["text"] == text), None)
        return {"key": key, "text": text, "active": active, "versions": versions}

    def _record_current(self, versions: list[dict], current: str) -> None:
        """Keep the file's present text as a version before it is replaced, so nothing is lost."""
        if not versions:
            versions.append(self._entry(1, current, "original", "file"))
        elif all(v["text"] != current for v in versions):
            versions.append(self._entry(versions[-1]["n"] + 1, current, "changed outside Pandora", "file"))

    def save_version(self, key: str, text: str, note: str = "") -> dict:
        text = normalize(text)
        validate(key, text)
        with self._lock:
            data = self._load()
            versions = data["items"].setdefault(key, [])
            self._record_current(versions, self.read(key))
            if all(v["text"] != text for v in versions):
                versions.append(self._entry(versions[-1]["n"] + 1, text, note.strip(), "edit"))
            self._save(data)
            _write_preserving_newlines(self._path(key), text)
            return self.item(key, data)

    def activate(self, key: str, n: int) -> dict:
        with self._lock:
            data = self._load()
            versions = data["items"].get(key, [])
            version = next((v for v in versions if v["n"] == n), None)
            if version is None:
                raise KeyError(f"{key} has no version {n}")
            validate(key, version["text"])
            self._record_current(versions, self.read(key))
            self._save(data)
            _write_preserving_newlines(self._path(key), version["text"])
            return self.item(key, data)

    def delete(self, key: str, n: int) -> dict:
        with self._lock:
            data = self._load()
            versions = data["items"].get(key, [])
            current = self.read(key)
            version = next((v for v in versions if v["n"] == n), None)
            if version is None:
                raise KeyError(f"{key} has no version {n}")
            if version["text"] == current:
                raise ValueError("The active version cannot be deleted; activate another one first")
            if n == versions[0]["n"]:
                raise ValueError("The original version cannot be deleted")
            versions.remove(version)
            self._save(data)
            return self.item(key, data)

    # -- runs, export, import ----------------------------------------------------

    def snapshot_prompts(self) -> tuple[dict[str, str], dict[str, dict]]:
        """The prompt texts to freeze for one run, and which version each one is."""
        data = self._load()
        texts, versions = {}, {}
        for key in self.files():
            if key.startswith("prompt:"):
                item = self.item(key, data)
                name = key.split(":", 1)[1]
                texts[name] = item["text"]
                versions[name] = {"version": item["active"], "hash": text_hash(item["text"])}
        return texts, versions

    def export(self) -> dict:
        return {"items": self._load()["items"]}

    def import_versions(self, exported: dict) -> int:
        """Add versions from an export that are not here yet. Never changes the live files."""
        known = self.files()
        added = 0
        with self._lock:
            data = self._load()
            for key, incoming in (exported.get("items") or {}).items():
                if key not in known or not isinstance(incoming, list):
                    continue
                versions = data["items"].setdefault(key, [])
                self._record_current(versions, self.read(key))
                for v in incoming:
                    text = normalize(str(v.get("text") or ""))
                    if not text.strip() or any(x["text"] == text for x in versions):
                        continue
                    note = str(v.get("note") or "")
                    versions.append(self._entry(
                        versions[-1]["n"] + 1, text, f"imported: {note}" if note else "imported", "import",
                        created=str(v.get("created") or "") or None,
                    ))
                    added += 1
            self._save(data)
        return added
