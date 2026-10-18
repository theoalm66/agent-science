import json

import pytest

from science_agent.config import STAGES
from science_agent.pipeline import Pipeline, StageError
from science_agent.report import render_markdown, save
from science_agent.tools.sandbox import RunResult

from .conftest import CANNED, ScriptedLLM


def test_full_run_produces_verdict_and_report(offline_config, scripted, tmp_path):
    llm, llms = scripted
    a = Pipeline(offline_config, llms=llms).run("Beat Shannon with fractal modulation")

    assert [stage for stage, _ in llm.calls] == ["formalize", "literature", "limits", "compute", "critic", "verdict"]
    assert a.verdict.verdict == "wrong"
    assert a.computation.ok and a.computation.result["value"] == pytest.approx(1e6 * 6.6582, rel=1e-3)
    assert a.critique.objections[0].severity == "fatal"  # normalized from "Fatal"

    # Later stages see the earlier stages' output.
    verdict_prompt = llm.calls[-1][1]
    assert "Converse theorem forbids it" in verdict_prompt and "succeeded" in verdict_prompt

    path = save(a, tmp_path)
    md = path.read_text()
    assert "## Verdict: WRONG" in md and "Shannon-Hartley" in md
    assert json.loads(path.with_suffix(".json").read_text())["verdict"]["verdict"] == "wrong"


def test_invalid_json_is_retried_with_error_feedback(offline_config):
    llm = ScriptedLLM({"verdict": ["I think it's wrong.", CANNED["verdict"]]})
    a = Pipeline(offline_config, llms={s: llm for s in STAGES}).run("idea")
    verdict_calls = [prompt for stage, prompt in llm.calls if stage == "verdict"]
    assert len(verdict_calls) == 2
    assert "YOUR PREVIOUS REPLY WAS INVALID" in verdict_calls[1]
    assert a.verdict.verdict == "wrong"


def test_stage_fails_after_repeated_invalid_output(offline_config):
    llm = ScriptedLLM({"formalize": ["nope", "still nope"]})
    with pytest.raises(StageError):
        Pipeline(offline_config, llms={s: llm for s in STAGES}).run("idea")


def test_underspecified_idea_stops_early(offline_config):
    llm = ScriptedLLM({"formalize": {**CANNED["formalize"], "underspecified": True,
                                     "clarifying_questions": ["Which channel?"],
                                     "established_view": "AWGN capacity is 9.97 Mbit/s here.",
                                     "what_must_be_true": ["The noise is not Gaussian"]}})
    a = Pipeline(offline_config, llms={s: llm for s in STAGES}).run("higher dimensions?")
    assert a.verdict.verdict == "underspecified"
    assert [s for s, _ in llm.calls] == ["formalize"]
    # Stops to ask, but still reports the established view and what would have to hold.
    for text in (a.verdict.summary, render_markdown(a)):
        assert "Which channel?" in text
        assert "AWGN capacity is 9.97 Mbit/s here." in text
        assert "The noise is not Gaussian" in text


def test_failed_script_is_retried_with_error(offline_config, scripted):
    llm, llms = scripted
    runs = iter([RunResult(False, "", "NameError: x", None), RunResult(True, "ok\nRESULT_JSON: {}", "", {})])
    a = Pipeline(offline_config, llms=llms, sandbox=lambda code, t: next(runs)).run("idea")
    compute_prompts = [p for s, p in llm.calls if s == "compute"]
    assert len(compute_prompts) == 2 and "NameError: x" in compute_prompts[1]
    assert a.computation.ok and a.computation.attempts == 2


def test_compute_disabled_skips_stage(offline_config, scripted):
    llm, llms = scripted
    offline_config.compute.enabled = False
    a = Pipeline(offline_config, llms=llms).run("idea")
    assert a.computation is None
    assert "compute" not in [s for s, _ in llm.calls]
    assert "(no computation was run)" in llm.calls[-1][1]


def test_literature_results_reach_prompt(offline_config, scripted):
    from science_agent.tools.literature import Paper

    class FakeSearch:
        def search(self, queries):
            return [Paper("Capacity of MIMO", "https://x", 1999, "abstract", "arxiv")]

    llm, llms = scripted
    offline_config.literature.enabled = True
    a = Pipeline(offline_config, llms=llms, literature=FakeSearch()).run("idea")
    lit_prompt = next(p for s, p in llm.calls if s == "literature")
    assert "Capacity of MIMO" in lit_prompt and len(a.papers) == 1


def test_idea_length_limit_is_configurable(offline_config, scripted):
    llm, llms = scripted
    offline_config.max_idea_chars = 50
    with pytest.raises(ValueError, match="max_idea_chars"):
        Pipeline(offline_config, llms=llms).run("x" * 51)
    assert llm.calls == []
    offline_config.max_idea_chars = 30_000
    assert Pipeline(offline_config, llms=llms).run("y" * 25_000).verdict.verdict == "wrong"


def test_long_sources_reach_the_stages_that_check_them(offline_config, scripted):
    llm, llms = scripted
    original = "Derivation:\n(1) C = B log2(1 + S/N)\n(2) therefore C = 2 B log2(S/N)  <- wrong step\n" + "detail " * 500
    Pipeline(offline_config, llms=llms).run(original)
    prompts = {stage: prompt for stage, prompt in llm.calls}
    # Verbatim key content goes to the analysis stages ...
    for stage in ("limits", "compute"):
        assert "- C = B log2(1 + S/N)" in prompts[stage]
    # ... and the judges read the full original text, not just the one-line claim.
    for stage in ("critic", "verdict"):
        assert "therefore C = 2 B log2(S/N)  <- wrong step" in prompts[stage]
        assert prompts[stage].count("detail") >= 500
