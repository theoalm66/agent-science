"""Render an Assessment as a Markdown report and save it next to its JSON trace."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .pipeline import Assessment

VERDICT_LABELS = {
    "wrong": "WRONG: contradicts a known limit",
    "known": "KNOWN: correct, but already established",
    "plausible_but_hard": "PLAUSIBLE BUT HARD",
    "promising": "PROMISING: worth deeper research",
    "underspecified": "UNDERSPECIFIED: needs more detail",
}


def _bullets(items, empty="(none)") -> str:
    items = [i for i in items if i]
    return "\n".join(f"- {i}" for i in items) if items else empty


def render_markdown(a: Assessment) -> str:
    v, f = a.verdict, a.formalization
    out = [f"# Idea assessment\n\n> {a.idea}\n"]
    out.append(f"*Domain:* {a.domain} · *Date:* {a.started_at}\n")

    if v:
        out.append(f"## Verdict: {VERDICT_LABELS.get(v.verdict, v.verdict)}\n")
        out.append(f"**Confidence:** {v.confidence:.0%}\n")
        out.append(f"**Deciding argument:** {v.deciding_argument}\n")
        if v.relaxed_assumption:
            out.append(f"**Relaxed assumption:** {v.relaxed_assumption}\n")
        if v.salvage:
            out.append(f"**Closest legitimate idea:** {v.salvage}\n")
        out.append(f"**Next step:** {v.next_step}\n")
        out.append(f"{v.summary}\n")

    if f:
        out.append("## 1. Formalized claim\n")
        out.append(f"{f.claim}\n")
        out.append(f"**Predicted quantity:** {f.predicted_quantity or '(not stated)'}\n")
        out.append(f"**Assumptions:**\n{_bullets(f.assumptions)}\n")
        if f.variables:
            out.append(f"**Variables:**\n{_bullets(f.variables)}\n")
        if f.clarifying_questions:
            out.append(f"**Clarifying questions:**\n{_bullets(f.clarifying_questions)}\n")

    if a.literature:
        lit = a.literature
        out.append("## 2. Literature\n")
        out.append(f"**Established field:** {lit.established_field} · **Novelty:** {lit.novelty}\n")
        out.append(f"{lit.summary}\n")
        works = [
            f"[{w.title}]({w.url}) — {w.relevance}" if w.url else f"{w.title} — {w.relevance}"
            for w in lit.closest_prior_work
        ]
        out.append(f"**Closest prior work:**\n{_bullets(works)}\n")
        if a.papers:
            found = [f"[{p.title}]({p.url}) ({p.year or 'n.d.'}, {p.source})" for p in a.papers[:15]]
            out.append(f"<details><summary>Search results ({len(a.papers)})</summary>\n\n{_bullets(found)}\n\n</details>\n")

    if a.limits:
        lim = a.limits
        out.append("## 3. Fundamental limits\n")
        out.append(_bullets(f"**{x.name}:** {x.implication}" for x in lim.applicable_limits) + "\n")
        out.append(f"**Violates a known limit:** {'yes' if lim.violates_known_limit else 'no'}\n")
        if lim.violation_explanation:
            out.append(f"{lim.violation_explanation}\n")
        out.append(f"**Relaxed assumptions:**\n{_bullets(lim.relaxed_assumptions)}\n")
        out.append(f"**Dimensional check:** {lim.dimensional_check}\n")

    if a.computation:
        c = a.computation
        out.append("## 4. Computation\n")
        out.append(f"**Question:** {c.question}\n")
        out.append(f"**Status:** {'succeeded' if c.ok else 'failed'} after {c.attempts} attempt(s)\n")
        if c.result:
            out.append(f"```json\n{json.dumps(c.result, indent=2, ensure_ascii=False)}\n```\n")
        if c.code:
            out.append(f"<details><summary>Script</summary>\n\n```python\n{c.code}\n```\n\n</details>\n")
        if c.output:
            out.append(f"<details><summary>Output</summary>\n\n```\n{c.output[-4000:]}\n```\n\n</details>\n")

    if a.critique:
        cr = a.critique
        out.append("## 5. Adversarial critique\n")
        out.append(_bullets(f"**[{o.severity}]** {o.objection} *({o.basis})*" for o in cr.objections) + "\n")
        if cr.errors_in_analysis:
            out.append(f"**Errors found in the analysis:**\n{_bullets(cr.errors_in_analysis)}\n")
        if cr.steelman:
            out.append(f"**Steelman:** {cr.steelman}\n")

    timings = ", ".join(f"{k} {s}s ({a.models.get(k, '?')})" for k, s in a.seconds.items())
    out.append(f"---\n*Stages:* {timings}\n")
    out.append(
        "*This is an automated first-pass triage by a language model. "
        "Check the deciding argument yourself before relying on it.*\n"
    )
    return "\n".join(out)


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "idea"


def save(a: Assessment, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.started_at[:10]}-{slugify(a.idea)}"
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(render_markdown(a), encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(
        json.dumps(a.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return md_path
