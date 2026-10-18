"""Pandora: a local web UI for the science agent.

Serves one HTML page plus a small JSON API on top of the pipeline: connection
settings (endpoint, model, API key), starting an assessment, live progress with
every prompt and reply, versioned editing of the prompts and Claude Code skills
and agents (see versions.py), views of knowledge bases, benchmarks and past
runs, and JSON export/import of versions and runs.

Built on the standard-library HTTP server, so it adds no dependencies. It binds
to 127.0.0.1 by default: a run executes LLM-written Python on this machine, so
only expose it (--host 0.0.0.0) on a network you trust.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

import httpx
import yaml

from . import knowledge
from .bench import load_benchmark
from .config import Config, LLMConfig, load_config
from .jsonutil import JSONExtractionError, extract_json
from .llm import LLM
from .pipeline import Pipeline, check_idea_length, compose_source, normalize_clarifications
from .report import render_markdown, save
from .tools.literature import LiteratureSearch
from .tools.sandbox import RunResult, run_python
from .versions import VersionStore

log = logging.getLogger(__name__)

STAGE_RE = re.compile(r"# Stage: (\w+)")
PROMPT_ORDER = ("system", "formalize", "literature", "limits", "compute", "critic", "verdict")
LLM_FIELDS = {"provider", "base_url", "model", "temperature", "max_tokens", "timeout", "json_mode", "effort"}
MAX_BODY = 1_000_000
MAX_IMPORT = 200_000_000
KEEP_RUNS = 10
EXPORT_FORMAT = "pandora-export"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}")


class RunCancelled(Exception):
    pass


class Run:
    """One assessment in a background thread; the browser polls its event list."""

    def __init__(self, idea: str, clarifications: list[dict] | None = None, parent_run: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.idea = idea
        self.clarifications = clarifications or []
        self.parent_run = parent_run
        self.status = "running"  # running | done | error | cancelled
        self.events: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.cancel = threading.Event()
        self._lock = threading.Lock()

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"i": len(self.events), "t": time.time(), "kind": kind, **data})

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            events = self.events[since:]
        return {
            "id": self.id,
            "idea": self.idea,
            "clarifications": self.clarifications,
            "parent_run": self.parent_run,
            "status": self.status,
            "events": events,
            "result": self.result if self.status == "done" else None,
            "error": self.error,
        }


class _RecordingLLM:
    """Passes calls through and records the exact prompt and reply of each one."""

    def __init__(self, inner: LLM, run: Run):
        self.inner, self.run, self.name = inner, run, inner.name

    def complete(self, system: str, user: str) -> str:
        if self.run.cancel.is_set():
            raise RunCancelled()
        m = STAGE_RE.search(user)
        stage = m.group(1) if m else "?"
        self.run.emit("llm_request", stage=stage, model=self.name, system=system, prompt=user)
        start = time.monotonic()
        text = self.inner.complete(system, user)
        try:
            parsed = extract_json(text)
        except JSONExtractionError:
            parsed = None
        self.run.emit(
            "llm_response", stage=stage, seconds=round(time.monotonic() - start, 1), response=text, parsed=parsed
        )
        return text


class _RecordingSearch:
    def __init__(self, inner: LiteratureSearch, run: Run):
        self.inner, self.run = inner, run

    def search(self, queries: list[str]):
        self.run.emit("search_start", queries=queries)
        papers = self.inner.search(queries)
        self.run.emit("search_result", count=len(papers), papers=[p.to_dict() for p in papers[:15]])
        return papers


class _RunLogHandler(logging.Handler):
    """Forwards warnings from the run's own thread (e.g. a rate-limited search) to the trace."""

    def __init__(self, run: Run, thread_id: int):
        super().__init__(logging.WARNING)
        self.run, self.thread_id = run, thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread_id:
            self.run.emit("log", level=record.levelname, logger=record.name, message=record.getMessage())


def _project_dir() -> Path:
    """The repository root (for benchmarks/ and .claude/), from the cwd or the package location."""
    for p in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (p / "benchmarks").is_dir() or (p / ".claude").is_dir():
            return p
    return Path.cwd()


def _front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                return yaml.safe_load(parts[1]) or {}, parts[2].lstrip("\n")
            except yaml.YAMLError:
                pass
    return {}, text


def _variables(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\$(\w+)", text)))


def library_entry(store: VersionStore, key: str) -> dict:
    """One editable file with its version history, shaped for the UI."""
    item = store.item(key)
    kind, name = key.split(":", 1)
    path = store.files()[key]
    try:
        file = path.relative_to(store.project_dir).as_posix()
    except ValueError:
        file = path.as_posix()
    entry = {**item, "kind": kind, "name": name, "file": file}
    if kind == "prompt":
        original = item["versions"][0]["text"] if item["versions"] else item["text"]
        entry["variables"] = _variables(item["text"])
        entry["expected_variables"] = _variables(original)
    else:
        meta, body = _front_matter(item["text"])
        entry.update(
            title=str(meta.get("name") or name), description=str(meta.get("description") or ""),
            tools=str(meta.get("tools") or ""), body=body,
        )
    return entry


def library(store: VersionStore) -> dict:
    """Everything the agent is made of, for the views and editors in the UI."""
    keys = list(store.files())
    prompt_names = [k.split(":", 1)[1] for k in keys if k.startswith("prompt:")]
    ordered = [n for n in PROMPT_ORDER if n in prompt_names] + [n for n in prompt_names if n not in PROMPT_ORDER]
    prompts = [library_entry(store, f"prompt:{n}") for n in ordered]
    skills = [library_entry(store, k) for k in keys if not k.startswith("prompt:")]

    domains = []
    for name in knowledge.available_domains():
        d = knowledge.load_domain(name)
        domains.append({"name": name, "description": d.get("description", ""), "limits": d["limits"]})

    root = store.project_dir
    benchmarks = []
    for path in sorted((root / "benchmarks").glob("*.yaml")):
        try:
            domain, items = load_benchmark(path)
        except (OSError, KeyError, TypeError, yaml.YAMLError) as e:
            log.warning("Skipping benchmark %s: %s", path, e)
            continue
        for item in items:
            benchmarks.append({
                "id": item.id, "idea": " ".join(item.idea.split()), "acceptable": item.acceptable,
                "why": item.why, "domain": domain, "file": path.relative_to(root).as_posix(),
            })
    return {"prompts": prompts, "domains": domains, "skills": skills, "benchmarks": benchmarks}


def list_reports(output_dir: Path) -> list[dict]:
    if not output_dir.is_dir():
        return []
    reports = []
    for md in sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        entry = {"name": md.stem, "idea": "", "verdict": None, "confidence": None, "started_at": "",
                 "models": {}, "meta": {}}
        js = md.with_suffix(".json")
        if js.exists():
            try:
                data = json.loads(js.read_text(encoding="utf-8-sig"))
                v = data.get("verdict") or {}
                entry.update(
                    idea=data.get("idea", ""), verdict=v.get("verdict"), confidence=v.get("confidence"),
                    started_at=data.get("started_at", ""), models=data.get("models") or {},
                    meta=data.get("meta") or {},
                )
            except (OSError, ValueError):
                pass
        entry["idea_key"] = " ".join(entry["idea"].lower().split())
        reports.append(entry)
    return reports


def list_models(provider: str, base_url: str, key: str | None) -> list[dict]:
    if provider == "anthropic":
        key = key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No API key: enter one or set ANTHROPIC_API_KEY")
        resp = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            params={"limit": 100},
            timeout=15,
        )
    else:
        if not base_url:
            raise ValueError("No endpoint URL")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return [
        {"id": m.get("id"), "context": m.get("max_model_len") or m.get("context_length") or m.get("max_input_tokens")}
        for m in resp.json().get("data") or []
        if m.get("id")
    ]


class PandoraServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        config_path: str | None = None,
        llm_factory: Callable[[Config], dict[str, LLM]] | None = None,
        sandbox: Callable[[str, float], RunResult] = run_python,
        literature_factory: Callable[[Config, str | None], LiteratureSearch] | None = None,
        store: VersionStore | None = None,
    ):
        super().__init__(address, Handler)
        self.config_path = config_path
        if store is None:
            project = _project_dir()
            store = VersionStore(
                project / "pandora-data" / "versions.json",
                Path(str(resources.files("science_agent") / "prompts")),
                project,
            )
        self.store = store
        self.llm_factory = llm_factory or Pipeline._build_llms
        self.sandbox = sandbox
        self.literature_factory = literature_factory or (
            lambda cfg, key: LiteratureSearch(max_results=cfg.literature.max_results, s2_api_key=key)
        )
        self.runs: dict[str, Run] = {}
        self.current: Run | None = None
        self.lock = threading.Lock()

    def config(self) -> Config:
        return load_config(self.config_path)

    def config_file(self) -> Path:
        return Path(self.config_path or os.environ.get("SCIENCE_AGENT_CONFIG") or "config.yaml").resolve()

    def host_allowed(self, host: str) -> bool:
        """Reject foreign Host headers (DNS rebinding) unless deliberately bound to all interfaces."""
        bound, port = self.server_address[:2]
        if bound in ("0.0.0.0", "::", ""):
            return True
        return host in {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}", f"{bound}:{port}"}

    # -- runs ----------------------------------------------------------------

    def run_config(self, body: dict) -> tuple[Config, str | None]:
        cfg = self.config()
        llm = body.get("llm") or {}
        update = {k: v for k, v in llm.items() if k in LLM_FIELDS and v not in (None, "")}
        key = str(llm.get("api_key") or "").strip()
        if key:
            update["api_key_value"] = key
        cfg.llm = LLMConfig.model_validate({**cfg.llm.model_dump(), **update})
        if body.get("domain"):
            if body["domain"] not in knowledge.available_domains():
                raise ValueError(f"Unknown domain {body['domain']!r}")
            cfg.domain = body["domain"]
        if "literature" in body:
            cfg.literature.enabled = bool(body["literature"])
        if "compute" in body:
            cfg.compute.enabled = bool(body["compute"])
        env = cfg.literature.semantic_scholar_api_key_env
        s2_key = str(body.get("s2_api_key") or "").strip() or (os.environ.get(env) if env else None)
        return cfg, s2_key

    def start_run(self, body: dict) -> Run:
        idea = str(body.get("idea") or "").strip()
        if not idea:
            raise ValueError("The idea is empty")
        cfg, s2_key = self.run_config(body)
        raw = body.get("clarifications") or []
        if not isinstance(raw, list):
            raise ValueError("clarifications must be a list of {question, answer}")
        clarifications = normalize_clarifications(raw)
        parent_run = str(body.get("parent_run") or "")[:64] or None
        check_idea_length(compose_source(idea, clarifications), cfg.max_idea_chars)
        # Freeze the prompts now: activating another version mid-run must not mix versions.
        prompts, prompt_versions = self.store.snapshot_prompts()
        with self.lock:
            if self.current and self.current.status == "running":
                raise RuntimeError("A run is already in progress")
            run = Run(idea, clarifications, parent_run)
            self.current = run
            self.runs[run.id] = run
            for old in list(self.runs)[:-KEEP_RUNS]:
                del self.runs[old]
        run.emit(
            "start", provider=cfg.llm.provider, model=cfg.llm.model, base_url=cfg.llm.base_url,
            domain=cfg.domain, literature=cfg.literature.enabled, compute=cfg.compute.enabled,
            prompt_versions=prompt_versions,
        )
        threading.Thread(
            target=self._work, args=(run, cfg, s2_key, prompts, prompt_versions), daemon=True, name=f"run-{run.id}"
        ).start()
        return run

    @staticmethod
    def run_meta(run: Run, cfg: Config, prompt_versions: dict) -> dict:
        """What is needed to tell runs of the same idea apart and to reproduce one. Never the keys."""
        llm = cfg.llm.model_dump(include={"provider", "base_url", "model", "temperature", "max_tokens", "json_mode"})
        if cfg.llm.provider == "anthropic":
            llm["effort"] = cfg.llm.effort
        return {
            "source": "pandora",
            "run_id": run.id,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "llm": llm,
            "stage_overrides": {s: {k: v for k, v in o.items() if "key" not in k} for s, o in cfg.stages.items()},
            "domain": cfg.domain,
            "literature": cfg.literature.enabled,
            "compute": cfg.compute.enabled,
            "prompt_versions": prompt_versions,
            "parent_run": run.parent_run,
        }

    def _work(self, run: Run, cfg: Config, s2_key: str | None, prompts: dict, prompt_versions: dict) -> None:
        handler = _RunLogHandler(run, threading.get_ident())
        logging.getLogger().addHandler(handler)
        try:
            shared: dict[int, _RecordingLLM] = {}
            llms = {
                stage: shared.setdefault(id(llm), _RecordingLLM(llm, run))
                for stage, llm in self.llm_factory(cfg).items()
            }

            def sandbox(code: str, timeout: float) -> RunResult:
                if run.cancel.is_set():
                    raise RunCancelled()
                run.emit("sandbox_start", code=code)
                result = self.sandbox(code, timeout)
                run.emit("sandbox_result", ok=result.ok, timed_out=result.timed_out,
                         output=result.summary(), result=result.result)
                return result

            literature = None
            if cfg.literature.enabled:
                literature = _RecordingSearch(self.literature_factory(cfg, s2_key), run)
            pipeline = Pipeline(
                cfg, llms=llms, literature=literature, sandbox=sandbox,
                progress=lambda stage: run.emit("stage", stage=stage), prompts=prompts,
            )
            assessment = pipeline.run(run.idea, run.clarifications)
            assessment.meta = self.run_meta(run, cfg, prompt_versions)
            path = save(assessment, cfg.output_dir)
            run.result = {
                "assessment": assessment.to_dict(),
                "markdown": render_markdown(assessment),
                "report": path.stem,
            }
            run.status = "done"
        except RunCancelled:
            run.status = "cancelled"
        except Exception as e:  # report every failure to the browser instead of dying silently
            run.error = f"{type(e).__name__}: {e}"
            run.emit("error", message=run.error, traceback=traceback.format_exc()[-4000:])
            run.status = "error"
        finally:
            logging.getLogger().removeHandler(handler)
            run.emit("end", status=run.status)

    # -- export / import -------------------------------------------------------

    def export_bundle(self, runs: str | list[str] = "all", versions: bool = True) -> dict:
        output_dir = self.config().output_dir
        names = [r["name"] for r in list_reports(output_dir)]
        if runs != "all":
            wanted = set(runs or [])
            names = [n for n in names if n in wanted]
        exported = []
        for name in names:
            js = output_dir / f"{name}.json"
            exported.append({
                "name": name,
                "markdown": (output_dir / f"{name}.md").read_text(encoding="utf-8-sig"),
                "assessment": json.loads(js.read_text(encoding="utf-8-sig")) if js.exists() else None,
            })
        return {
            "format": EXPORT_FORMAT,
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": self.store.export() if versions else {"items": {}},
            "runs": exported,
        }

    def import_bundle(self, bundle: dict) -> dict:
        """Merge an export: new versions are added (live files untouched), runs never overwrite."""
        if bundle.get("format") != EXPORT_FORMAT:
            raise ValueError("Not a Pandora export file")
        versions_added = self.store.import_versions(bundle.get("versions") or {})
        output_dir = self.config().output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        added = skipped = 0
        for r in bundle.get("runs") or []:
            name, md, assessment = str(r.get("name") or ""), r.get("markdown"), r.get("assessment")
            if not SAFE_NAME.fullmatch(name) or ".." in name or not isinstance(md, str):
                skipped += 1
                continue
            existing = output_dir / f"{name}.md"
            if existing.exists() and existing.read_text(encoding="utf-8-sig") == md:
                skipped += 1  # already here
                continue
            stem, n = name, 0
            while (output_dir / f"{stem}.md").exists() or (output_dir / f"{stem}.json").exists():
                n += 1
                stem = f"{name}-imported" + (f"-{n}" if n > 1 else "")
            (output_dir / f"{stem}.md").write_text(md, encoding="utf-8")
            if isinstance(assessment, dict):
                (output_dir / f"{stem}.json").write_text(
                    json.dumps(assessment, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            added += 1
        return {"versions_added": versions_added, "runs_added": added, "runs_skipped": skipped}


class Handler(BaseHTTPRequestHandler):
    server: PandoraServer

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _guard(self) -> bool:
        if not self.server.host_allowed(self.headers.get("Host", "")):
            self._error(403, "Host not allowed")
            return False
        return True

    def do_GET(self) -> None:
        if not self._guard():
            return
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = parse_qs(url.query)
        try:
            if not parts or parts == ["index.html"]:
                html = (resources.files("science_agent") / "static" / "pandora.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if parts == ["api", "config"]:
                return self._json(self._config_view())
            if parts == ["api", "library"]:
                return self._json(library(self.server.store))
            if parts == ["api", "reports"]:
                return self._json(list_reports(self.server.config().output_dir))
            if len(parts) == 3 and parts[:2] == ["api", "reports"]:
                return self._report(parts[2])
            if parts == ["api", "runs", "current"]:
                run = self.server.current
                return self._json({"run": run.snapshot() if run else None})
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = self.server.runs.get(parts[2])
                if not run:
                    return self._error(404, "Unknown run")
                since = int((query.get("since") or ["0"])[0] or 0)
                return self._json(run.snapshot(max(since, 0)))
            self._error(404, "Not found")
        except Exception as e:
            log.exception("GET %s failed", self.path)
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self) -> None:
        if not self._guard():
            return
        # A JSON content type forces a CORS preflight, which this server never approves,
        # so other web pages cannot start runs here behind your back.
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._error(415, "Expected application/json")
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host', '')}":
            return self._error(403, "Origin not allowed")
        parts = [p for p in urlsplit(self.path).path.split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        if length > (MAX_IMPORT if parts == ["api", "import"] else MAX_BODY):
            return self._error(413, "Request too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._error(400, "Invalid JSON")
        if not isinstance(body, dict):
            return self._error(400, "Expected a JSON object")
        try:
            if parts == ["api", "models"]:
                return self._models(body)
            if len(parts) == 3 and parts[:2] == ["api", "versions"]:
                return self._versions(parts[2], body)
            if parts == ["api", "export"]:
                return self._json(self.server.export_bundle(body.get("runs", "all"), bool(body.get("versions", True))))
            if parts == ["api", "import"]:
                try:
                    return self._json(self.server.import_bundle(body))
                except ValueError as e:
                    return self._error(400, str(e))
            if parts == ["api", "runs"]:
                try:
                    run = self.server.start_run(body)
                except RuntimeError as e:
                    return self._error(409, str(e))
                except ValueError as e:
                    return self._error(400, str(e))
                return self._json({"id": run.id})
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                run = self.server.runs.get(parts[2])
                if not run:
                    return self._error(404, "Unknown run")
                run.cancel.set()
                return self._json({"ok": True})
            self._error(404, "Not found")
        except Exception as e:
            log.exception("POST %s failed", self.path)
            self._error(500, f"{type(e).__name__}: {e}")

    def _config_view(self) -> dict:
        cfg = self.server.config()
        llm = cfg.llm.model_dump(exclude={"api_key_value"})
        return {
            "config_file": str(self.server.config_file()),
            "config_exists": self.server.config_file().exists(),
            "llm": llm,
            "llm_key_from_env": bool(cfg.llm.api_key_env and os.environ.get(cfg.llm.api_key_env)),
            "stages": cfg.stages,
            "domain": cfg.domain,
            "domains": knowledge.available_domains(),
            "literature": cfg.literature.model_dump(),
            "s2_key_from_env": bool(
                cfg.literature.semantic_scholar_api_key_env
                and os.environ.get(cfg.literature.semantic_scholar_api_key_env)
            ),
            "compute": cfg.compute.model_dump(),
            "output_dir": str(cfg.output_dir.resolve()),
        }

    def _models(self, body: dict) -> None:
        provider = str(body.get("provider") or "openai_compatible")
        base_url = str(body.get("base_url") or "").strip()
        key = str(body.get("api_key") or "").strip() or None
        if not key:  # fall back to the config's env key, but only for the endpoint it belongs to
            cfg = self.server.config().llm
            if cfg.provider == provider and (provider == "anthropic" or cfg.base_url == base_url):
                key = cfg.api_key()
        try:
            self._json({"models": list_models(provider, base_url, key)})
        except (ValueError, httpx.HTTPError) as e:
            self._error(502, str(e) or type(e).__name__)

    def _versions(self, action: str, body: dict) -> None:
        store, key = self.server.store, str(body.get("key") or "")
        try:
            if action == "save":
                store.save_version(key, str(body.get("text") or ""), str(body.get("note") or ""))
            elif action == "activate":
                store.activate(key, int(body.get("n")))
            elif action == "delete":
                store.delete(key, int(body.get("n")))
            else:
                return self._error(404, "Not found")
        except KeyError as e:
            return self._error(404, str(e.args[0]) if e.args else "Not found")
        except (ValueError, TypeError) as e:
            return self._error(400, str(e))
        self._json({"entry": library_entry(store, key)})

    def _report(self, name: str) -> None:
        output_dir = self.server.config().output_dir
        if name not in {r["name"] for r in list_reports(output_dir)}:
            return self._error(404, "Unknown report")
        md = (output_dir / f"{name}.md").read_text(encoding="utf-8-sig")
        js = output_dir / f"{name}.json"
        assessment = json.loads(js.read_text(encoding="utf-8-sig")) if js.exists() else None
        self._json({"name": name, "markdown": md, "assessment": assessment})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pandora", description="Pandora: web UI for the science agent.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default 127.0.0.1; 0.0.0.0 exposes it to your network)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", help="YAML config (default: $SCIENCE_AGENT_CONFIG or ./config.yaml)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    server = PandoraServer((args.host, args.port), config_path=args.config)
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{server.server_address[1]}/"
    print(f"Pandora is running at {url}  (Ctrl+C to stop)", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: reachable from your network. Anyone who can reach it can run LLM-written code here.",
              flush=True)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
