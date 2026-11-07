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
import collections
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
from typing import Callable, NamedTuple
from urllib.parse import parse_qs, urlsplit

import httpx
import yaml

from . import knowledge
from .bench import load_benchmark
from .config import Config, LLMConfig, load_config
from .jsonutil import JSONExtractionError, extract_json
from .llm import LLM
from .netguard import EndpointNotAllowed, check_public_endpoint, safe_client
from .checkpoints import CheckpointStore
from .pipeline import Assessment, Pipeline, check_idea_length, compose_source, normalize_clarifications
from .profiles import Profile, ProfileError, ProfileStore, clean_profile, load_master_key
from .report import render_markdown, save
from .tools.literature import LiteratureSearch
from .tools.sandbox import RemoteSandbox, RunResult, run_python
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


class User(NamedTuple):
    login: str
    admin: bool


LOCAL_USER = User("local", True)


class Run:
    """One assessment in a background thread; the browser polls its event list."""

    def __init__(
        self,
        idea: str,
        clarifications: list[dict] | None = None,
        parent_run: str | None = None,
        owner: str = "local",
    ):
        self.id = uuid.uuid4().hex[:12]
        self.idea = idea
        self.clarifications = clarifications or []
        self.parent_run = parent_run
        self.owner = owner
        self.resumable = False  # an interim result is on disk, so "Resume" can skip the finished stages
        self.resumed_from: str | None = None
        self.live: dict | None = None  # what the model is writing right now (streamed models only)
        self.status = "queued"  # queued | running | done | error | cancelled
        self.queue_position: int | None = None
        self.lane = "local"  # "local": the server's own model (one at a time); "own": a guest's API
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
            "queue_position": self.queue_position,
            "events": events,
            "result": self.result if self.status == "done" else None,
            "error": self.error,
            "resumable": self.resumable,
            "live": self.live if self.status == "running" else None,
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
        if hasattr(self.inner, "on_progress"):
            def progress(info: dict) -> None:
                if self.run.cancel.is_set():
                    raise RunCancelled()  # closes the stream, so the model server stops generating
                self.run.live = {"stage": stage, **info}

            self.inner.on_progress = progress
        try:
            text = self.inner.complete(system, user)
        finally:
            if hasattr(self.inner, "on_progress"):
                self.inner.on_progress = None
            self.run.live = None
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


def list_models(provider: str, base_url: str, key: str | None, client: httpx.Client | None = None) -> list[dict]:
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
        resp = (client or httpx).get(base_url.rstrip("/") + "/models", headers=headers, timeout=15)
    if not resp.is_success:
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return [
        {
            # Gemini lists "models/gemini-..."; chat completions take the bare id.
            "id": m["id"].removeprefix("models/"),
            "context": m.get("max_model_len") or m.get("context_length") or m.get("max_input_tokens"),
        }
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
        sandbox: Callable[[str, float], RunResult] | None = None,
        literature_factory: Callable[[Config, str | None], LiteratureSearch] | None = None,
        store: VersionStore | None = None,
        profiles: ProfileStore | None = None,
        checkpoints: CheckpointStore | None = None,
    ):
        super().__init__(address, Handler)
        self.config_path = config_path
        project = _project_dir()
        self.data_dir = Path(os.environ.get("PANDORA_DATA_DIR") or project / "pandora-data")
        self._profiles = profiles
        self.checkpoints = checkpoints or CheckpointStore(self.data_dir / "checkpoints")
        if store is None:
            store = VersionStore(
                self.data_dir / "versions.json",
                Path(str(resources.files("science_agent") / "prompts")),
                project,
            )
        self.store = store
        self.llm_factory = llm_factory or Pipeline._build_llms
        self.custom_llm_factory = llm_factory is not None
        self.sandbox = sandbox
        self.literature_factory = literature_factory or (
            lambda cfg, key: LiteratureSearch(max_results=cfg.literature.max_results, s2_api_key=key)
        )
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self.queue_ready = threading.Condition(self.lock)
        # Two lanes, each a FIFO queue. "local" runs on the server's model, strictly one at a time.
        # "own" runs use a guest's own API and do not load the server, so a few may run side by side.
        self.lanes: dict[str, collections.deque] = {"local": collections.deque(), "own": collections.deque()}
        try:
            external = max(1, self.config().server.external_workers)
        except Exception:  # an unreadable config is reported when a run starts
            external = 1
        for lane, workers in (("local", 1), ("own", external)):
            for i in range(workers):
                threading.Thread(target=self._worker, args=(lane,), daemon=True, name=f"pandora-{lane}-{i}").start()

    @property
    def profiles(self) -> ProfileStore:
        """Encrypted guest profiles; created on first use so the key file only appears when needed."""
        with self.lock:
            if self._profiles is None:
                self._profiles = ProfileStore(self.data_dir / "profiles", load_master_key(self.data_dir))
            return self._profiles

    def current_run_for(self, user: "User") -> "Run | None":
        """The newest queued or running run of this person, for picking a run up after a page reload."""
        with self.lock:
            mine = [r for r in self.runs.values() if r.owner == user.login and r.status in ("queued", "running")]
        return mine[-1] if mine else None

    def config(self) -> Config:
        return load_config(self.config_path)

    def config_file(self) -> Path:
        return Path(self.config_path or os.environ.get("SCIENCE_AGENT_CONFIG") or "config.yaml").resolve()

    def user_for(self, headers) -> User | None:
        """Who is calling. Local mode: always the admin. Shared mode: the Tailscale login that
        Tailscale Serve put in the identity header, or None when the header is missing."""
        server = self.config().server
        if server.mode == "local":
            return LOCAL_USER
        login = (headers.get(server.identity_header) or "").strip()
        if not login:
            return None
        return User(login, login.lower() in {a.strip().lower() for a in server.admins})

    def can_see(self, user: User, owner: str | None) -> bool:
        return user.admin or (owner is not None and owner == user.login)

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

    def own_api_config(self, body: dict, user: User) -> tuple[Config, str | None, str]:
        """Run settings for a guest using their own saved API: their endpoint, model and keys,
        no per-stage overrides of the server. Returns (config, semantic scholar key, endpoint)."""
        base = self.config()
        if not base.server.guests_may_use_own_api:
            raise ValueError("Using your own API is switched off on this server")
        profile = self.profiles.load(user.login)
        if profile is None:
            raise ValueError("Save your own API settings first (⚙)")
        cfg, s2_key = self.run_config({k: v for k, v in body.items() if k not in ("llm", "s2_api_key")})
        allowed = cfg.server.guest_allowed_hosts
        base_url = profile.base_url
        if profile.provider == "openai_compatible":
            try:
                base_url = check_public_endpoint(profile.base_url, allowed)
            except EndpointNotAllowed as e:
                raise ValueError(str(e)) from e
        cfg.llm = LLMConfig(
            provider=profile.provider, base_url=base_url or None, model=profile.model,
            temperature=profile.temperature, max_tokens=min(profile.max_tokens, cfg.server.guest_max_tokens),
            timeout=min(profile.timeout, cfg.server.guest_max_timeout), json_mode=profile.json_mode,
            effort=profile.effort, api_key_value=profile.api_key or None,
        )
        cfg.stages = {}
        return cfg, (profile.s2_api_key or s2_key), base_url

    def start_run(self, body: dict, user: User = LOCAL_USER) -> Run:
        resume_from = str(body.get("resume_from") or "")
        resume = None
        if resume_from:
            # Continue an interrupted run: idea, answers and finished stages come from its checkpoint.
            resume = self.checkpoints.load(resume_from)
            if resume is None or not self.can_see(user, resume.get("owner")):
                raise ValueError("There is no interrupted run to resume")
            with self.lock:
                if any(r.status in ("queued", "running") and resume_from in (r.id, r.resumed_from)
                       for r in self.runs.values()):
                    raise ValueError("That run is already being resumed")
            body = {**body, "idea": resume["idea"], "clarifications": resume.get("clarifications") or [],
                    "parent_run": resume.get("parent_run")}
        idea = str(body.get("idea") or "").strip()
        if not idea:
            raise ValueError("The idea is empty")
        own = (not user.admin) and str(body.get("llm_source") or "server") == "own"
        if own:
            cfg, s2_key, _ = self.own_api_config(body, user)
        else:
            if not user.admin:
                # On the server's model guests use its endpoint and keys, nothing from the browser.
                body = {k: v for k, v in body.items() if k not in ("llm", "s2_api_key")}
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
            active = [r for r in self.runs.values() if r.owner == user.login and r.status in ("queued", "running")]
            limit = cfg.server.max_runs_per_user
            if len(active) >= limit:
                raise RuntimeError(f"You already have {len(active)} run(s) queued or running (limit {limit})")
            run = Run(idea, clarifications, parent_run, owner=user.login)
            run.resumed_from = resume_from or None
            run.lane = "own" if own else "local"
            self.runs[run.id] = run
            finished = [k for k, r in self.runs.items() if r.status not in ("queued", "running")]
            for old in finished[: max(0, len(self.runs) - KEEP_RUNS)]:
                del self.runs[old]
            self.lanes[run.lane].append((run, (cfg, s2_key, prompts, prompt_versions, resume)))
            self._renumber_queue(run.lane)
            self.queue_ready.notify_all()
        run.emit(
            "start", provider=cfg.llm.provider, model=cfg.llm.model,
            # A guest never sees the server's endpoint, but always their own.
            base_url=cfg.llm.base_url if (user.admin or own) else None,
            source="own" if own else "server",
            domain=cfg.domain, literature=cfg.literature.enabled, compute=cfg.compute.enabled,
            prompt_versions=prompt_versions, queue_position=run.queue_position,
        )
        if resume is not None:
            stages = Assessment.from_dict(resume["assessment"]).done_stages() if resume.get("assessment") else []
            run.emit(
                "resume", from_run=resume_from, stages=stages,
                prompts_changed=bool(resume.get("prompt_versions")) and resume.get("prompt_versions") != prompt_versions,
            )
        return run

    def _renumber_queue(self, lane: str) -> None:
        """Positions as people see them: 1 = next to start. Call with the lock held."""
        for i, (queued, _) in enumerate(self.lanes[lane], 1):
            queued.queue_position = i

    def cancel_run(self, run: Run) -> None:
        with self.lock:
            run.cancel.set()
            queue = self.lanes[run.lane]
            queued = [item for item in queue if item[0] is run]
            if not queued:
                return  # running: stops at its next model call
            queue.remove(queued[0])
            self._renumber_queue(run.lane)
            run.status, run.queue_position = "cancelled", None
        run.emit("end", status="cancelled")

    def _worker(self, lane: str) -> None:
        queue = self.lanes[lane]
        while True:
            with self.lock:
                while not queue:
                    self.queue_ready.wait()
                run, args = queue.popleft()
                self._renumber_queue(lane)
                run.status, run.queue_position = "running", None
            run.emit("dequeued")
            self._work(run, *args)

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
            "owner": run.owner,
            "llm_source": "own-api" if run.lane == "own" else "server",
        }

    def _checkpoint(self, run: Run, prompt_versions: dict, assessment: Assessment) -> None:
        """Keep the finished stages on disk. Once the first checkpoint of a resumed run is written,
        it contains everything the old one had, so the old file goes."""
        self.checkpoints.save(run.id, run.owner, {
            "idea": run.idea, "clarifications": run.clarifications, "parent_run": run.parent_run,
            "prompt_versions": prompt_versions, "assessment": assessment.to_dict(),
        })
        run.resumable = True
        if run.resumed_from:
            self.checkpoints.delete(run.resumed_from)
            run.resumed_from = None

    def _work(self, run: Run, cfg: Config, s2_key: str | None, prompts: dict, prompt_versions: dict,
              resume: dict | None = None) -> None:
        handler = _RunLogHandler(run, threading.get_ident())
        logging.getLogger().addHandler(handler)
        guard = None
        try:
            shared: dict[int, _RecordingLLM] = {}
            if run.lane == "own" and not self.custom_llm_factory:
                # The guest chose the endpoint: allow only public addresses (see netguard.py).
                guard = safe_client(cfg.server.guest_allowed_hosts, cfg.llm.timeout)
                raw_llms = Pipeline._build_llms(cfg, guard)
            else:
                raw_llms = self.llm_factory(cfg)
            llms = {stage: shared.setdefault(id(llm), _RecordingLLM(llm, run)) for stage, llm in raw_llms.items()}

            base_sandbox = self.sandbox or (
                RemoteSandbox(cfg.compute.sandbox_url) if cfg.compute.sandbox_url else run_python
            )

            def sandbox(code: str, timeout: float) -> RunResult:
                if run.cancel.is_set():
                    raise RunCancelled()
                run.emit("sandbox_start", code=code)
                result = base_sandbox(code, timeout)
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
            assessment = pipeline.run(
                run.idea, run.clarifications,
                resume=Assessment.from_dict(resume["assessment"]) if resume and resume.get("assessment") else None,
                checkpoint=lambda a: self._checkpoint(run, prompt_versions, a),
            )
            assessment.meta = self.run_meta(run, cfg, prompt_versions)
            path = save(assessment, cfg.output_dir)
            self.checkpoints.delete(run.id)
            if run.resumed_from:
                self.checkpoints.delete(run.resumed_from)
            run.resumable = False
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
            self.checkpoints.set_error(run.id, run.error)
            run.resumable = self.checkpoints.exists(run.id)
        finally:
            logging.getLogger().removeHandler(handler)
            if guard is not None:
                guard.close()
            run.emit("end", status=run.status)

    # -- export / import -------------------------------------------------------

    def export_bundle(self, runs: str | list[str] = "all", versions: bool = True, user: User = LOCAL_USER) -> dict:
        output_dir = self.config().output_dir
        names = [r["name"] for r in list_reports(output_dir) if self.can_see(user, r["meta"].get("owner"))]
        versions = versions and user.admin
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
        user = self.server.user_for(self.headers)
        if user is None:
            self._error(403, "Pandora is shared through Tailscale: open it with its Tailscale (https://….ts.net) address")
            return False
        self.user = user
        return True

    def _admin_only(self) -> bool:
        if not self.user.admin:
            self._error(403, "Only the Pandora admin can do this")
            return False
        return True

    def _visible_run(self, run_id: str):
        run = self.server.runs.get(run_id)
        return run if run and self.server.can_see(self.user, run.owner) else None

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
                reports = list_reports(self.server.config().output_dir)
                return self._json([r for r in reports if self.server.can_see(self.user, r["meta"].get("owner"))])
            if len(parts) == 3 and parts[:2] == ["api", "reports"]:
                return self._report(parts[2])
            if parts == ["api", "runs", "current"]:
                run = self.server.current_run_for(self.user)
                return self._json({"run": run.snapshot() if run else None})
            if parts == ["api", "profile"]:
                return self._profile_get()
            if parts == ["api", "unfinished"]:
                return self._json(self.server.checkpoints.list(lambda owner: self.server.can_see(self.user, owner)))
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = self._visible_run(parts[2])
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
        host = self.headers.get("Host", "")
        allowed = {f"http://{host}", f"https://{host}", *self.server.config().server.allowed_origins}
        if origin and origin.rstrip("/") not in {o.rstrip("/") for o in allowed}:
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
            if parts == ["api", "profile"]:
                return self._profile_save(body)
            if parts == ["api", "profile", "delete"]:
                return self._profile_delete()
            if parts == ["api", "unfinished", "discard"]:
                data = self.server.checkpoints.load(str(body.get("run_id") or ""))
                if data is None or not self.server.can_see(self.user, data.get("owner")):
                    return self._error(404, "Unknown interrupted run")
                self.server.checkpoints.delete(data["run_id"])
                return self._json({"ok": True})
            if parts == ["api", "models"]:
                return self._models(body)
            if len(parts) == 3 and parts[:2] == ["api", "versions"]:
                return self._admin_only() and self._versions(parts[2], body)
            if parts == ["api", "export"]:
                return self._json(self.server.export_bundle(
                    body.get("runs", "all"), bool(body.get("versions", True)), self.user))
            if parts == ["api", "import"]:
                if not self._admin_only():
                    return
                try:
                    return self._json(self.server.import_bundle(body))
                except ValueError as e:
                    return self._error(400, str(e))
            if parts == ["api", "runs"]:
                try:
                    run = self.server.start_run(body, self.user)
                except RuntimeError as e:
                    return self._error(409, str(e))
                except ValueError as e:
                    return self._error(400, str(e))
                return self._json({"id": run.id})
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                run = self._visible_run(parts[2])
                if not run:
                    return self._error(404, "Unknown run")
                self.server.cancel_run(run)
                return self._json({"ok": True})
            self._error(404, "Not found")
        except Exception as e:
            log.exception("POST %s failed", self.path)
            self._error(500, f"{type(e).__name__}: {e}")

    def _config_view(self) -> dict:
        cfg = self.server.config()
        identity = {"mode": cfg.server.mode, "user": self.user.login, "admin": self.user.admin}
        if not self.user.admin:
            # Guests see which model answers them, not where the server and its files are.
            return {
                **identity,
                "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model},
                "stages": {},
                "domain": cfg.domain,
                "domains": knowledge.available_domains(),
                "literature": {"enabled": cfg.literature.enabled},
                "compute": {"enabled": cfg.compute.enabled},
                "max_runs_per_user": cfg.server.max_runs_per_user,
            }
        llm = cfg.llm.model_dump(exclude={"api_key_value"})
        return {
            **identity,
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
        if not self.user.admin:
            return self._guest_models(provider, base_url, key)
        if not key:  # fall back to the config's env key, but only for the endpoint it belongs to
            cfg = self.server.config().llm
            if cfg.provider == provider and (provider == "anthropic" or cfg.base_url == base_url):
                key = cfg.api_key()
        try:
            self._json({"models": list_models(provider, base_url, key)})
        except (ValueError, httpx.HTTPError) as e:
            self._error(502, str(e) or type(e).__name__)

    # -- guests' own API profile -------------------------------------------------

    def _guest_only(self) -> bool:
        if self.user.admin:
            return self._error(403, "Only guests have a saved profile; the admin sets the connection in the settings") or False
        if not self.server.config().server.guests_may_use_own_api:
            return self._error(403, "Using your own API is switched off on this server") or False
        return True

    def _guest_models(self, provider: str, base_url: str, key: str | None) -> None:
        """Model list from the guest's own endpoint. Never falls back to the server's keys, and
        only talks to public addresses."""
        if not self._guest_only():
            return
        cfg = self.server.config()
        try:
            stored = self.server.profiles.load(self.user.login)
            if not key and stored and stored.provider == provider and (provider == "anthropic" or stored.base_url == base_url.rstrip("/")):
                key = stored.api_key or None  # the key they saved for exactly this endpoint
            if provider == "anthropic":
                if not key:  # list_models would otherwise fall back to the server's own Anthropic key
                    raise ProfileError("Enter your Anthropic API key first")
                return self._json({"models": list_models("anthropic", "", key)})
            url = check_public_endpoint(base_url, cfg.server.guest_allowed_hosts)
            with safe_client(cfg.server.guest_allowed_hosts, 15) as client:
                self._json({"models": list_models(provider, url, key, client)})
        except (ValueError, ProfileError, httpx.HTTPError) as e:
            self._error(400 if isinstance(e, (EndpointNotAllowed, ProfileError)) else 502, str(e) or type(e).__name__)

    def _profile_get(self) -> None:
        cfg = self.server.config()
        info = {
            "enabled": bool(cfg.server.guests_may_use_own_api) and not self.user.admin,
            "server_model": {"provider": cfg.llm.provider, "model": cfg.llm.model},
            "limits": {"max_tokens": cfg.server.guest_max_tokens, "timeout": cfg.server.guest_max_timeout},
            "profile": None,
        }
        if info["enabled"]:
            try:
                profile = self.server.profiles.load(self.user.login)
            except ProfileError as e:
                info["error"] = str(e)
                profile = None
            info["profile"] = profile.public() if profile else None
        self._json(info)

    def _profile_save(self, body: dict) -> None:
        if not self._guest_only():
            return
        cfg = self.server.config()
        allowed = cfg.server.guest_allowed_hosts
        try:
            try:
                current = self.server.profiles.load(self.user.login)
            except ProfileError:
                current = None  # unreadable old file: saving replaces it
            profile = clean_profile(
                body, current, lambda url: check_public_endpoint(url, allowed),
                cfg.server.guest_max_tokens, cfg.server.guest_max_timeout,
            )
            self.server.profiles.save(self.user.login, profile)
        except (EndpointNotAllowed, ProfileError) as e:
            return self._error(400, str(e))
        self._json({"profile": profile.public()})

    def _profile_delete(self) -> None:
        if not self._guest_only():
            return
        self._json({"deleted": self.server.profiles.delete(self.user.login)})

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
        visible = {r["name"] for r in list_reports(output_dir) if self.server.can_see(self.user, r["meta"].get("owner"))}
        if name not in visible:
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
