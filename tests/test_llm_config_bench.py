import json

import httpx
import pytest

from science_agent.bench import BenchItem, load_benchmark, run_benchmark, summarize
from science_agent.config import Config, LLMConfig, load_config
from science_agent.knowledge import available_domains, load_domain
from science_agent.llm import LLMError, OpenAICompatibleLLM, strip_think_tags
from science_agent.models import VerdictResult
from science_agent.pipeline import Assessment


def test_openai_compatible_request_shape(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-test")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>{\"a\": 1}"}}]})

    cfg = LLMConfig(base_url="http://localhost:11434/v1/", model="m", api_key_env="MY_KEY", json_mode=True)
    llm = OpenAICompatibleLLM(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.complete("sys", "user") == '{"a": 1}'
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_error_raises():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no model")))
    with pytest.raises(LLMError, match="404"):
        OpenAICompatibleLLM(LLMConfig(), client=client).complete("s", "u")


def test_strip_think_tags_variants():
    assert strip_think_tags("<think>a</think>\nanswer") == "answer"
    assert strip_think_tags("reasoning only closed</think>answer") == "answer"
    assert strip_think_tags("plain") == "plain"


def test_stage_override_merges_and_validates(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  model: small\nstages:\n  critic:\n    model: big\n")
    cfg = load_config(p)
    assert cfg.llm_for("critic").model == "big"
    assert cfg.llm_for("critic").base_url == cfg.llm.base_url
    assert cfg.llm_for("limits").model == "small"


def test_unknown_stage_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stages:\n  critics: {model: x}\n")
    with pytest.raises(ValueError, match="critics"):
        load_config(p)


def test_missing_default_config_gives_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCIENCE_AGENT_CONFIG", raising=False)
    assert isinstance(load_config(), Config)


def test_bundled_knowledge_and_benchmark_are_valid():
    assert "information_theory" in available_domains()
    domain = load_domain("information_theory")
    assert len(domain["limits"]) >= 10
    for lim in domain["limits"]:
        assert {"id", "name", "statement", "assumptions"} <= lim.keys()
    name, items = load_benchmark("benchmarks/information_theory.yaml")
    assert name == "information_theory" and len(items) >= 10
    valid = set(VerdictResult.model_fields["verdict"].annotation.__args__)
    assert all(set(i.acceptable) <= valid for i in items)


def test_benchmark_scoring_separates_failure_modes():
    items = [
        BenchItem("a", "x", ["wrong"]),
        BenchItem("b", "y", ["known"]),
        BenchItem("c", "z", ["known", "plausible_but_hard"]),
        BenchItem("d", "boom", ["wrong"]),
    ]
    answers = {"x": "wrong", "y": "wrong", "z": "plausible_but_hard"}

    def assess(idea):
        if idea == "boom":
            raise RuntimeError("model down")
        a = Assessment(idea=idea, domain="d", started_at="t")
        a.verdict = VerdictResult(verdict=answers[idea], confidence=0.8)
        return a

    s = summarize(run_benchmark(items, assess))
    assert (s["correct"], s["total"]) == (2, 4)
    assert s["rejected_wrong_ideas"] == 1 and s["wrong_ideas"] == 2
    assert s["falsely_rejected_real_ideas"] == 1 and s["real_ideas"] == 2
    assert s["outcomes"][3]["error"] == "model down"


def test_rh_koeln_api_is_called_like_any_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("RHK_API_KEY", "rhk-token")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    cfg = LLMConfig(base_url="https://api.ai.rh-koeln.de/v1", model="qwen3-30b-a3b-instruct-2507",
                    api_key_env="RHK_API_KEY", max_tokens=2048)
    llm = OpenAICompatibleLLM(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.complete("sys", "user") == '{"ok": true}'
    assert seen["url"] == "https://api.ai.rh-koeln.de/v1/chat/completions"
    assert seen["auth"] == "Bearer rhk-token"
    assert seen["body"]["model"] == "qwen3-30b-a3b-instruct-2507" and seen["body"]["max_tokens"] == 2048
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_any_2xx_reply_is_accepted():
    # The RH Köln gateway answers chat completions with 201 Created.
    reply = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(201, json=reply)))
    assert OpenAICompatibleLLM(LLMConfig(), client=client).complete("s", "u") == '{"ok": true}'


def _chat_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_endpoint_uses_max_completion_tokens():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    cfg = LLMConfig(base_url="https://api.openai.com/v1", model="gpt-x", max_tokens=4000)
    OpenAICompatibleLLM(cfg, client=_chat_client(handler)).complete("s", "u")
    assert bodies[0]["max_completion_tokens"] == 4000 and "max_tokens" not in bodies[0]


def test_rejected_parameters_are_adapted_and_retried():
    # Reasoning models behind other gateways reject max_tokens and a custom temperature.
    bodies = []
    errors = iter([
        httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}}),
        httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2. Only the default (1) value is supported."}}),
    ])

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(errors, httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))

    llm = OpenAICompatibleLLM(LLMConfig(base_url="https://gateway.example/v1"), client=_chat_client(handler))
    assert llm.complete("s", "u") == "{}"
    assert "max_tokens" in bodies[0] and "temperature" in bodies[0]
    assert "max_completion_tokens" in bodies[2] and "temperature" not in bodies[2]
    llm.complete("s", "u")  # the adaptation sticks for later calls
    assert len(bodies) == 4


def test_empty_reply_after_token_limit_explains_itself():
    reply = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    client = _chat_client(lambda r: httpx.Response(200, json=reply))
    with pytest.raises(LLMError, match="raise max_tokens"):
        OpenAICompatibleLLM(LLMConfig(), client=client).complete("s", "u")


def test_gemini_model_ids_lose_their_prefix(monkeypatch):
    from science_agent import web

    payload = {"data": [{"id": "models/gemini-x-flash"}, {"id": "plain-model"}]}
    monkeypatch.setattr(web.httpx, "get", lambda *a, **k: httpx.Response(200, json=payload))
    ids = [m["id"] for m in web.list_models("openai_compatible", "https://generativelanguage.googleapis.com/v1beta/openai", "k")]
    assert ids == ["gemini-x-flash", "plain-model"]
