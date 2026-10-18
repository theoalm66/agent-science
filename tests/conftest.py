import json
import re

import pytest

from science_agent.config import STAGES, Config

STAGE_RE = re.compile(r"# Stage: (\w+)")

CANNED = {
    "formalize": {
        "claim": "A modulation exceeds B log2(1+SNR) on a SISO AWGN channel.",
        "domain": "information theory",
        "assumptions": ["SISO", "AWGN", "fixed bandwidth and power"],
        "variables": ["B: bandwidth [Hz]"],
        "predicted_quantity": "rate above Shannon capacity",
        "standard_terms": ["channel capacity"],
        "search_queries": ["AWGN channel capacity converse"],
        "underspecified": False,
        "clarifying_questions": [],
        "input_type": "derivation",
        "key_content": ["C = B log2(1 + S/N)", "S/N = 1000"],
    },
    "literature": {
        "established_field": "channel coding",
        "novelty": "known",
        "closest_prior_work": [{"title": "A Mathematical Theory of Communication", "url": "", "relevance": "defines C"}],
        "summary": "Capacity is proven.",
    },
    "limits": {
        "applicable_limits": [{"name": "Shannon-Hartley", "statement": "C = B log2(1+SNR)", "implication": "no rate above C"}],
        "relaxed_assumptions": [],
        "violates_known_limit": True,
        "violation_explanation": "No assumption is relaxed.",
        "dimensional_check": "consistent",
        "quantitative_question": "Capacity of a 1 MHz channel at 20 dB SNR",
    },
    "compute": {
        "purpose": "capacity",
        "code": "import math\nc = 1e6*math.log2(1+100)\nprint('C', c)\nprint('RESULT_JSON: ' + __import__('json').dumps({'value': c}))\n",
    },
    "critic": {
        "objections": [{"objection": "Converse theorem forbids it", "severity": "Fatal", "basis": "Shannon 1948"}],
        "errors_in_analysis": [],
        "strongest_objection": "Converse",
        "steelman": "",
    },
    "verdict": {
        "verdict": "wrong",
        "confidence": 0.95,
        "deciding_argument": "Converse of the channel coding theorem.",
        "relaxed_assumption": "none",
        "salvage": "MIMO",
        "next_step": "Read Cover & Thomas ch. 9",
        "summary": "Wrong.",
    },
}


class ScriptedLLM:
    """Answers each stage prompt from a dict; records every call."""

    name = "scripted"

    def __init__(self, responses=None):
        self.responses = {k: v for k, v in CANNED.items()}
        self.responses.update(responses or {})
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        stage = STAGE_RE.search(user).group(1)
        self.calls.append((stage, user))
        reply = self.responses[stage]
        if isinstance(reply, list):  # sequence of replies for retries
            reply = reply.pop(0)
        return reply if isinstance(reply, str) else "```json\n" + json.dumps(reply) + "\n```"


@pytest.fixture
def offline_config(tmp_path):
    cfg = Config()
    cfg.literature.enabled = False
    cfg.output_dir = tmp_path
    return cfg


@pytest.fixture
def scripted():
    llm = ScriptedLLM()
    return llm, {stage: llm for stage in STAGES}
