"""Calibration benchmark: run the pipeline on ideas with known verdicts and score it."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import yaml

from .pipeline import Assessment


@dataclass
class BenchItem:
    id: str
    idea: str
    acceptable: list[str]
    why: str = ""


@dataclass
class BenchOutcome:
    id: str
    expected: list[str]
    got: str | None
    confidence: float | None
    correct: bool
    error: str = ""


def load_benchmark(path: str | Path) -> tuple[str | None, list[BenchItem]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [BenchItem(**item) for item in data["ideas"]]
    return data.get("domain"), items


def run_benchmark(
    items: list[BenchItem],
    assess: Callable[[str], Assessment],
    on_result: Callable[[BenchOutcome], None] | None = None,
) -> list[BenchOutcome]:
    outcomes = []
    for item in items:
        try:
            verdict = assess(item.idea).verdict
            got = verdict.verdict if verdict else None
            outcome = BenchOutcome(
                item.id, item.acceptable, got, verdict.confidence if verdict else None, got in item.acceptable
            )
        except Exception as e:  # one broken idea must not abort the whole run
            outcome = BenchOutcome(item.id, item.acceptable, None, None, False, error=str(e)[:300])
        outcomes.append(outcome)
        if on_result:
            on_result(outcome)
    return outcomes


def summarize(outcomes: list[BenchOutcome]) -> dict:
    n = len(outcomes)
    correct = sum(o.correct for o in outcomes)
    by_expected: dict[str, Counter] = {}
    for o in outcomes:
        by_expected.setdefault(o.expected[0], Counter())[o.got or "error"] += 1
    wrong_ideas = [o for o in outcomes if o.expected == ["wrong"]]
    real_ideas = [o for o in outcomes if "wrong" not in o.expected]
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
        # Two failure modes reported separately: gullible vs. dismissive.
        "rejected_wrong_ideas": sum(o.got == "wrong" for o in wrong_ideas),
        "wrong_ideas": len(wrong_ideas),
        "falsely_rejected_real_ideas": sum(o.got == "wrong" for o in real_ideas),
        "real_ideas": len(real_ideas),
        "confusion": {k: dict(v) for k, v in by_expected.items()},
        "outcomes": [asdict(o) for o in outcomes],
    }


def save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
