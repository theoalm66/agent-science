"""Extract JSON objects from free-form LLM output (fences, prose, trailing commas)."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONExtractionError(ValueError):
    pass


def _balanced_objects(text: str):
    """Yield every top-level {...} substring, respecting strings and escapes."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def _try_parse(candidate: str):
    for variant in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            value = json.loads(variant)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_json(text: str) -> dict:
    """Return the first JSON object found in `text`; prefers fenced blocks, then the largest object."""
    for block in _FENCE.findall(text):
        parsed = _try_parse(block.strip())
        if parsed is not None:
            return parsed
    for candidate in sorted(_balanced_objects(text), key=len, reverse=True):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
    raise JSONExtractionError(f"No JSON object found in model output: {text[:300]!r}")
