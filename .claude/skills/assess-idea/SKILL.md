---
name: assess-idea
description: Triage a scientific idea or hypothesis from first principles - decide whether it is wrong, already known, plausible but hard, or promising, and write a report. Use when the user asks to assess, check, sanity-check or evaluate a scientific idea, hypothesis or claim (e.g. "can we beat Shannon by...", "what about higher dimensions for data transmission?").
---

# Assess a scientific idea

This is the Claude Code version of the `science-agent` Python pipeline in this repository.
It runs the same six stages with the same prompts and knowledge base, but uses Claude Code's
own tools and subagents instead of a configured LLM endpoint. Both produce reports in `reports/`
in the same format, so results can be compared.

## Inputs

The idea is in the user's message (or `$ARGUMENTS`). Determine the domain; default is
`information_theory`. Knowledge bases live in `src/science_agent/knowledge/<domain>.yaml`.

## Procedure

Read `src/science_agent/prompts/system.md` first and hold yourself to its rules throughout.
For each stage, read the matching prompt file in `src/science_agent/prompts/` and produce the JSON
it asks for (keep it in your working notes; you don't need to show the JSON to the user).

1. **Formalize** (`formalize.md`): restate the idea as a precise claim. Never swap the author's
   setup for the textbook one. If it is underspecified, stop and ask the user the clarifying
   questions, but still state the established view and what would have to be true if the idea
   is correct.

2. **Literature** (`literature.md`): delegate to the `literature-scout` subagent with the claim,
   standard terms and search queries. It returns established field, novelty and closest prior work
   with real URLs. Do not cite papers it did not find unless you are certain they exist.

3. **Limits** (`limits.md`): read the domain knowledge base in full and check the claim against
   every applicable limit. The key question: which assumption of each limit does the idea
   genuinely relax? If none, it cannot beat that limit.

4. **Compute** (`compute.md`): write a short Python script answering the most decisive
   quantitative question and run it with `.venv/bin/python` (or `python3` with numpy, scipy, sympy).
   Print the parameters and end with a `RESULT_JSON:` line. Fix and re-run once if it fails.
   Never report a number you did not compute or cite.

5. **Critic** (`critic.md`): delegate to the `critic` subagent, passing the claim and your
   literature, limits and computation results. It works in a fresh context so it is not
   anchored on your reasoning. Take its objections seriously; also check it for errors.

6. **Verdict** (`verdict.md`): choose exactly one of `wrong`, `known`, `plausible_but_hard`,
   `promising`, `underspecified`, with a calibrated confidence, the deciding argument, the
   relaxed assumption (or "none"), the closest legitimate idea, and the cheapest next step.

## Output

Write the report to `reports/<YYYY-MM-DD>-<slug>.md` using the section structure produced by
`src/science_agent/report.py` (Verdict, 1. Formalized claim, 2. Literature, 3. Fundamental limits,
4. Computation with the script, 5. Adversarial critique). Then give the user a short summary:
verdict, confidence, deciding argument, next step, and the report path.

## Calibration

To check this skill against known answers, run it on ideas from
`benchmarks/information_theory.yaml` and compare with `acceptable`. An assessor that calls
everything wrong is as useless as one that calls everything promising.
