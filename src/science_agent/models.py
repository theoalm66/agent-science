"""Typed results of each pipeline stage.

Fields default to empty so that slightly incomplete output from a small local
model still validates; the critic and verdict stages see what is missing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["wrong", "known", "plausible_but_hard", "promising", "underspecified"]
VERDICTS: tuple[str, ...] = Verdict.__args__


def _normalize_label(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_").replace(" ", "_")
    return value


class Formalization(BaseModel):
    claim: str
    domain: str = ""
    assumptions: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    predicted_quantity: str = ""
    standard_terms: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    underspecified: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    established_view: str = ""
    what_must_be_true: list[str] = Field(default_factory=list)


class PriorWork(BaseModel):
    title: str
    url: str = ""
    relevance: str = ""


class LiteratureAssessment(BaseModel):
    established_field: str = ""
    novelty: Literal["known", "partially_known", "novel", "unclear"] = "unclear"
    closest_prior_work: list[PriorWork] = Field(default_factory=list)
    summary: str = ""

    @field_validator("novelty", mode="before")
    @classmethod
    def normalize_label(cls, v: object) -> object:
        return _normalize_label(v)


class AppliedLimit(BaseModel):
    name: str
    statement: str = ""
    implication: str = ""


class LimitsCheck(BaseModel):
    applicable_limits: list[AppliedLimit] = Field(default_factory=list)
    relaxed_assumptions: list[str] = Field(default_factory=list)
    violates_known_limit: bool = False
    violation_explanation: str = ""
    dimensional_check: str = ""
    quantitative_question: str = ""


class ComputePlan(BaseModel):
    purpose: str = ""
    code: str


class Computation(BaseModel):
    question: str
    purpose: str = ""
    code: str = ""
    ok: bool = False
    output: str = ""
    result: dict | None = None
    attempts: int = 0


class Objection(BaseModel):
    objection: str
    severity: Literal["fatal", "major", "minor"] = "major"
    basis: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_label(cls, v: object) -> object:
        return _normalize_label(v)


class Critique(BaseModel):
    objections: list[Objection] = Field(default_factory=list)
    errors_in_analysis: list[str] = Field(default_factory=list)
    strongest_objection: str = ""
    steelman: str = ""


class VerdictResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(default=0.5, ge=0, le=1)
    deciding_argument: str = ""
    relaxed_assumption: str = ""
    salvage: str = ""
    next_step: str = ""
    summary: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_label(cls, v: object) -> object:
        return _normalize_label(v)
