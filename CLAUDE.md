# Science Agent

LLM-agnostic pipeline that triages scientific ideas (wrong / known / plausible_but_hard / promising).
See README.md for usage.

- Setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev,anthropic]"`
- Tests: `.venv/bin/pytest` (fully offline: scripted fake LLM, mocked HTTP)
- Layout: `src/science_agent/` — `pipeline.py` (stages), `llm.py` (providers), `prompts/*.md`
  (one per stage, `$var` placeholders), `knowledge/*.yaml` (limits per domain), `tools/`
  (literature search, sandbox), `bench.py` + `benchmarks/*.yaml` (calibration),
  `web.py` + `static/pandora.html` (Pandora web UI: stdlib HTTP server around the pipeline),
  `versions.py` (version history of prompts/skills/agents; the active version is the file on disk).
- Tests must never edit the real prompt or `.claude/` files: give `PandoraServer`/`VersionStore` copies (see tests/test_web.py).
- Keep the LLM interface text-in/text-out: no native tool calling, so local models keep working.
- Every prompt file must start with `# Stage: <name>`; the test fake LLM dispatches on it.
- When changing prompts or knowledge, run `science-agent bench` against a real model if one is available.
- To assess an idea inside Claude Code, use the `/assess-idea` skill.
