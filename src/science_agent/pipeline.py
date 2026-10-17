"""The assessment pipeline: formalize -> literature -> limits -> compute -> critic -> verdict.

Control flow lives in code, not in the model. Each stage is one prompt that
must return a JSON object, which keeps the pipeline usable with models that
have weak or no native tool calling.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from string import Template
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from . import knowledge
from .config import STAGES, Config
from .jsonutil import JSONExtractionError, extract_json
from .llm import LLM, make_llm
from .models import (
    Computation,
    ComputePlan,
    Critique,
    Formalization,
    LimitsCheck,
    LiteratureAssessment,
    VerdictResult,
)
from .tools.literature import LiteratureSearch, Paper
from .tools.sandbox import RunResult, run_python

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)


class StageError(RuntimeError):
    pass


def load_prompt(name: str) -> str:
    return (resources.files("science_agent") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def _dump(model: BaseModel | None) -> str:
    if model is None:
        return "(stage skipped)"
    return json.dumps(model.model_dump(exclude={"code"}), indent=2, ensure_ascii=False)


@dataclass
class Assessment:
    idea: str
    domain: str
    started_at: str
    formalization: Formalization | None = None
    papers: list[Paper] = field(default_factory=list)
    literature: LiteratureAssessment | None = None
    limits: LimitsCheck | None = None
    computation: Computation | None = None
    critique: Critique | None = None
    verdict: VerdictResult | None = None
    models: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def dump(m):
            return m.model_dump() if m is not None else None

        return {
            "idea": self.idea,
            "domain": self.domain,
            "started_at": self.started_at,
            "formalization": dump(self.formalization),
            "papers": [p.to_dict() for p in self.papers],
            "literature": dump(self.literature),
            "limits": dump(self.limits),
            "computation": dump(self.computation),
            "critique": dump(self.critique),
            "verdict": dump(self.verdict),
            "models": self.models,
            "seconds": self.seconds,
        }


class Pipeline:
    def __init__(
        self,
        config: Config,
        llms: dict[str, LLM] | None = None,
        literature: LiteratureSearch | None = None,
        sandbox: Callable[[str, float], RunResult] = run_python,
        progress: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.llms = llms or self._build_llms(config)
        self.domain = knowledge.load_domain(config.domain)
        self.system_prompt = load_prompt("system")
        self.sandbox = sandbox
        self.progress = progress or (lambda msg: None)
        if literature is None and config.literature.enabled:
            env = config.literature.semantic_scholar_api_key_env
            literature = LiteratureSearch(
                max_results=config.literature.max_results,
                s2_api_key=os.environ.get(env) if env else None,
            )
        self.literature_search = literature if config.literature.enabled else None

    @staticmethod
    def _build_llms(config: Config) -> dict[str, LLM]:
        """One client per distinct LLM config, shared by the stages that use it."""
        cache: dict[str, LLM] = {}
        llms = {}
        for stage in STAGES:
            cfg = config.llm_for(stage)
            key = cfg.model_dump_json()
            if key not in cache:
                cache[key] = make_llm(cfg)
            llms[stage] = cache[key]
        return llms

    # -- core helper -------------------------------------------------------

    def ask(self, stage: str, model_cls: type[M], max_attempts: int = 2, **variables) -> M:
        """Render the stage prompt, call the LLM, parse and validate JSON; retry once on bad output."""
        prompt = Template(load_prompt(stage)).safe_substitute(
            {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in variables.items()}
        )
        llm = self.llms[stage]
        last_error = ""
        for _ in range(max_attempts):
            text = llm.complete(self.system_prompt, prompt + last_error)
            try:
                return model_cls.model_validate(extract_json(text))
            except (JSONExtractionError, ValidationError) as e:
                log.warning("Stage %s returned invalid output: %s", stage, e)
                last_error = (
                    "\n\nYOUR PREVIOUS REPLY WAS INVALID:\n"
                    f"{str(e)[:1500]}\n"
                    "Reply again with ONLY the JSON object, matching the schema exactly."
                )
        raise StageError(f"Stage {stage!r} produced no valid JSON after {max_attempts} attempts")

    def _timed(self, assessment: Assessment, stage: str, fn: Callable[[], M]) -> M:
        self.progress(stage)
        start = time.monotonic()
        try:
            return fn()
        finally:
            assessment.seconds[stage] = round(time.monotonic() - start, 1)
            assessment.models[stage] = self.llms[stage].name

    # -- stages ------------------------------------------------------------

    def run(self, idea: str) -> Assessment:
        a = Assessment(
            idea=idea.strip(),
            domain=self.domain.get("domain", self.config.domain),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        a.formalization = self._timed(a, "formalize", lambda: self.ask(
            "formalize", Formalization, idea=a.idea, knowledge_index=knowledge.render_index(self.domain)
        ))
        f = a.formalization
        if f.underspecified:
            a.verdict = VerdictResult(
                verdict="underspecified",
                confidence=1.0,
                deciding_argument="The idea cannot be assessed as stated.",
                next_step="Answer the clarifying questions, then re-run.",
                summary="Open questions:\n" + "\n".join(f"- {q}" for q in f.clarifying_questions),
            )
            return a

        if self.literature_search and f.search_queries:
            self.progress("search")
            a.papers = self.literature_search.search(f.search_queries[:5])
        a.literature = self._timed(a, "literature", lambda: self.ask(
            "literature",
            LiteratureAssessment,
            claim=f.claim,
            standard_terms=", ".join(f.standard_terms) or "(none given)",
            papers=self._render_papers(a.papers),
        ))

        a.limits = self._timed(a, "limits", lambda: self.ask(
            "limits",
            LimitsCheck,
            claim=f.claim,
            assumptions="; ".join(f.assumptions) or "(none stated)",
            predicted_quantity=f.predicted_quantity or "(not stated)",
            knowledge=knowledge.render_full(self.domain),
        ))

        if self.config.compute.enabled and a.limits.quantitative_question:
            a.computation = self._timed(a, "compute", lambda: self._compute(f, a.limits))

        a.critique = self._timed(a, "critic", lambda: self.ask(
            "critic",
            Critique,
            claim=f.claim,
            literature=_dump(a.literature),
            limits=_dump(a.limits),
            computation=self._render_computation(a.computation),
        ))

        a.verdict = self._timed(a, "verdict", lambda: self.ask(
            "verdict",
            VerdictResult,
            claim=f.claim,
            literature=_dump(a.literature),
            limits=_dump(a.limits),
            computation=self._render_computation(a.computation),
            critique=_dump(a.critique),
        ))
        return a

    def _compute(self, f: Formalization, limits: LimitsCheck) -> Computation:
        comp = Computation(question=limits.quantitative_question)
        previous_error = ""
        limit_names = ", ".join(lim.name for lim in limits.applicable_limits) or "(none)"
        for attempt in range(1, self.config.compute.max_attempts + 1):
            comp.attempts = attempt
            try:
                plan = self.ask(
                    "compute",
                    ComputePlan,
                    question=limits.quantitative_question,
                    claim=f.claim,
                    limits=limit_names,
                    previous_error=previous_error,
                )
            except StageError as e:
                comp.output = str(e)
                return comp
            comp.purpose, comp.code = plan.purpose, plan.code
            run = self.sandbox(plan.code, self.config.compute.timeout)
            comp.ok, comp.output, comp.result = run.ok, run.summary(), run.result
            if run.ok:
                return comp
            previous_error = (
                "\nYOUR PREVIOUS SCRIPT FAILED. Fix it.\n"
                f"Previous script:\n{plan.code[:4000]}\nError:\n{run.summary()[-2000:]}\n"
            )
        return comp

    @staticmethod
    def _render_papers(papers: list[Paper]) -> str:
        if not papers:
            return "(no search results)"
        lines = []
        for i, p in enumerate(papers[:15], 1):
            lines.append(f"[{i}] {p.title} ({p.year or 'n.d.'}, {p.source}) {p.url}\n    {p.abstract[:400]}")
        return "\n".join(lines)

    @staticmethod
    def _render_computation(comp: Computation | None) -> str:
        if comp is None:
            return "(no computation was run)"
        status = "succeeded" if comp.ok else "FAILED (do not rely on it)"
        return (
            f"Question: {comp.question}\nStatus: {status}\nPurpose: {comp.purpose}\n"
            f"Result: {json.dumps(comp.result, ensure_ascii=False) if comp.result else '(none)'}\n"
            f"Output tail:\n{comp.output[-2000:]}"
        )
