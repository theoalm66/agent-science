import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from science_agent.config import STAGES
from science_agent.tools.sandbox import RunResult
from science_agent.versions import VersionStore
from science_agent.web import PandoraServer

from .conftest import ScriptedLLM

REPO = Path(__file__).resolve().parents[1]


def make_server(root: Path):
    """A server on a free port whose prompts, skills and reports are copies under `root`."""
    project = root / "project"
    shutil.copytree(REPO / ".claude", project / ".claude")
    shutil.copytree(REPO / "benchmarks", project / "benchmarks")
    prompts = root / "prompts"
    shutil.copytree(REPO / "src" / "science_agent" / "prompts", prompts)
    cfg = root / "config.yaml"
    cfg.write_text(
        f"output_dir: {json.dumps(str(root / 'reports'))}\nliterature:\n  enabled: false\n",
        encoding="utf-8",
    )
    llm = ScriptedLLM()
    seen_configs = []

    def factory(config):
        seen_configs.append(config)
        return {stage: llm for stage in STAGES}

    srv = PandoraServer(
        ("127.0.0.1", 0),
        config_path=str(cfg),
        llm_factory=factory,
        sandbox=lambda code, timeout: RunResult(True, "C 6.6e6\n", "", {"value": 6.6e6}),
        store=VersionStore(root / "data" / "versions.json", prompts, project),
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.seen_configs, srv.llm, srv.prompts_dir = seen_configs, llm, prompts
    return srv


@pytest.fixture
def server(tmp_path):
    srv = make_server(tmp_path)
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def other_server(tmp_path):
    srv = make_server(tmp_path / "other")
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, headers=None):
    url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if body is not None and "Content-Type" not in (headers or {}):
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if "json" in resp.headers["Content-Type"] else raw.decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for_end(srv, run_id):
    deadline = time.time() + 10
    while time.time() < deadline:
        _, snap = call(srv, "GET", f"/api/runs/{run_id}")
        if any(e["kind"] == "end" for e in snap["events"]):
            return snap
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_serves_page_and_library(server):
    status, html = call(server, "GET", "/")
    assert status == 200 and "<title>Pandora" in html
    _, lib = call(server, "GET", "/api/library")
    names = [p["name"] for p in lib["prompts"]]
    assert names[:7] == ["system", "formalize", "literature", "limits", "compute", "critic", "verdict"]
    assert "idea" in lib["prompts"][1]["variables"]
    assert lib["domains"][0]["limits"]
    assert {s["name"] for s in lib["skills"]} >= {"assess-idea", "critic"}
    assert any(b["id"] == "mimo_scaling" for b in lib["benchmarks"])


def test_run_records_prompts_and_saves_report(server):
    status, res = call(server, "POST", "/api/runs", {
        "idea": "A modulation that beats Shannon.",
        "llm": {"base_url": "http://example.invalid/v1", "model": "m", "api_key": "secret-key"},
        "compute": True,
    })
    assert status == 200
    snap = wait_for_end(server, res["id"])
    assert snap["status"] == "done", snap["error"]
    assert snap["result"]["assessment"]["verdict"]["verdict"] == "wrong"

    kinds = [e["kind"] for e in snap["events"]]
    requests = [e for e in snap["events"] if e["kind"] == "llm_request"]
    assert [e["stage"] for e in requests] == ["formalize", "literature", "limits", "compute", "critic", "verdict"]
    assert "A modulation that beats Shannon." in requests[0]["prompt"]
    assert "sandbox_result" in kinds and kinds[-1] == "end"

    # The key reaches the LLM config but never the event stream or the report.
    assert server.seen_configs[-1].llm.api_key() == "secret-key"
    assert "secret-key" not in json.dumps(snap)
    _, reports = call(server, "GET", "/api/reports")
    assert reports[0]["verdict"] == "wrong"
    _, report = call(server, "GET", f"/api/reports/{reports[0]['name']}")
    assert "secret-key" not in report["markdown"] + json.dumps(report["assessment"])


def test_rejects_foreign_host_and_non_json_posts(server):
    status, _ = call(server, "GET", "/api/config", headers={"Host": "evil.example:80"})
    assert status == 403
    status, _ = call(server, "POST", "/api/runs", {"idea": "x"}, headers={"Content-Type": "text/plain"})
    assert status == 415
    port = server.server_address[1]
    status, _ = call(server, "POST", "/api/runs", {"idea": "x"},
                     headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
    assert status == 403
    assert call(server, "GET", "/api/config", headers={"Host": f"localhost:{port}"})[0] == 200


def run_idea(srv, idea="A modulation that beats Shannon."):
    status, res = call(srv, "POST", "/api/runs", {"idea": idea, "llm": {"model": "m"}})
    assert status == 200, res
    snap = wait_for_end(srv, res["id"])
    assert snap["status"] == "done", snap["error"]
    return snap


def test_prompt_edit_is_versioned_and_used_by_the_next_run(server):
    _, lib = call(server, "GET", "/api/library")
    limits = next(p for p in lib["prompts"] if p["key"] == "prompt:limits")
    assert limits["versions"] == [] and "claim" in limits["expected_variables"]

    status, res = call(server, "POST", "/api/versions/save", {
        "key": "prompt:limits", "text": limits["text"] + "\nALWAYS CHECK UNITS.\n", "note": "units",
    })
    assert status == 200
    entry = res["entry"]
    assert entry["active"] == 2 and [v["note"] for v in entry["versions"]] == ["original", "units"]
    assert "ALWAYS CHECK UNITS." in (server.prompts_dir / "limits.md").read_text(encoding="utf-8")

    snap = run_idea(server)
    sent = next(e["prompt"] for e in snap["events"] if e["kind"] == "llm_request" and e["stage"] == "limits")
    assert "ALWAYS CHECK UNITS." in sent
    meta = snap["result"]["assessment"]["meta"]
    assert meta["prompt_versions"]["limits"]["version"] == 2
    assert meta["llm"]["model"] == "m" and meta["source"] == "pandora"

    assert call(server, "POST", "/api/versions/activate", {"key": "prompt:limits", "n": 1})[1]["entry"]["active"] == 1
    status, res = call(server, "POST", "/api/versions/save", {"key": "prompt:limits", "text": "no header"})
    assert status == 400 and "# Stage: limits" in res["error"]
    assert call(server, "POST", "/api/versions/activate", {"key": "prompt:nope", "n": 1})[0] == 404


def test_repeated_runs_form_a_history(server):
    run_idea(server, "Same idea twice.")
    run_idea(server, "Same idea twice.")
    _, reports = call(server, "GET", "/api/reports")
    assert len(reports) == 2 and len({r["name"] for r in reports}) == 2
    assert reports[0]["idea_key"] == reports[1]["idea_key"] == "same idea twice."


def test_export_import_round_trip(server, other_server):
    _, lib = call(server, "GET", "/api/library")
    critic = next(s for s in lib["skills"] if s["key"] == "agent:critic")
    call(server, "POST", "/api/versions/save", {"key": "agent:critic", "text": critic["text"] + "\nBe terse.\n"})
    run_idea(server)

    _, bundle = call(server, "POST", "/api/export", {"runs": "all", "versions": True})
    assert bundle["format"] == "pandora-export" and len(bundle["runs"]) == 1
    assert "secret" not in json.dumps(bundle)

    _, res = call(other_server, "POST", "/api/import", bundle)
    assert res == {"versions_added": 1, "runs_added": 1, "runs_skipped": 0}
    _, again = call(other_server, "POST", "/api/import", bundle)
    assert again["runs_added"] == 0 and again["runs_skipped"] == 1 and again["versions_added"] == 0
    # Imported versions are history only: the other installation's file is unchanged.
    _, other_lib = call(other_server, "GET", "/api/library")
    other_critic = next(s for s in other_lib["skills"] if s["key"] == "agent:critic")
    assert "Be terse." not in other_critic["text"] and other_critic["active"] == 1

    assert call(other_server, "POST", "/api/import", {"format": "something-else"})[0] == 400
    bad = {"format": "pandora-export", "runs": [{"name": "../escape", "markdown": "x"}]}
    assert call(other_server, "POST", "/api/import", bad)[1]["runs_skipped"] == 1


def test_validates_input_and_unknown_reports(server):
    assert call(server, "POST", "/api/runs", {"idea": "  "})[0] == 400
    assert call(server, "POST", "/api/runs", {"idea": "x", "domain": "../../secrets"})[0] == 400
    assert call(server, "GET", "/api/reports/..%2Fconfig")[0] == 404


def test_answers_to_clarifying_questions_rerun_the_idea(server):
    first = run_idea(server)
    status, res = call(server, "POST", "/api/runs", {
        "idea": "A modulation that beats Shannon.",
        "llm": {"model": "m"},
        "parent_run": first["id"],
        "clarifications": [
            {"question": "Baseband or passband?", "answer": "Passband, 1 MHz wide."},
            {"question": "Skipped question", "answer": "  "},
        ],
    })
    assert status == 200, res
    snap = wait_for_end(server, res["id"])
    assert snap["status"] == "done", snap["error"]
    a = snap["result"]["assessment"]
    # Only answered questions are kept; the idea itself stays the original text.
    assert a["clarifications"] == [{"question": "Baseband or passband?", "answer": "Passband, 1 MHz wide."}]
    assert a["idea"] == "A modulation that beats Shannon."
    assert a["meta"]["parent_run"] == first["id"]
    prompts = {e["stage"]: e["prompt"] for e in snap["events"] if e["kind"] == "llm_request"}
    for stage in ("formalize", "critic", "verdict"):
        assert "A: Passband, 1 MHz wide." in prompts[stage]
    assert "Skipped question" not in prompts["formalize"]
    assert "Passband, 1 MHz wide." in snap["result"]["markdown"]
    # Both runs belong to the same idea in the history.
    _, reports = call(server, "GET", "/api/reports")
    assert len({r["idea_key"] for r in reports}) == 1


def test_rejects_malformed_clarifications(server):
    assert call(server, "POST", "/api/runs", {"idea": "x", "clarifications": "nope"})[0] == 400
