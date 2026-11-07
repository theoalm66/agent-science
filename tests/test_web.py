import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from science_agent.checkpoints import CheckpointStore
from science_agent.config import STAGES
from science_agent.tools.sandbox import RunResult
from science_agent.profiles import Profile, ProfileStore
from science_agent.versions import VersionStore
from science_agent.web import PandoraServer

from .conftest import ScriptedLLM

REPO = Path(__file__).resolve().parents[1]


def make_server(root: Path, extra_config: str = ""):
    """A server on a free port whose prompts, skills and reports are copies under `root`."""
    project = root / "project"
    shutil.copytree(REPO / ".claude", project / ".claude")
    shutil.copytree(REPO / "benchmarks", project / "benchmarks")
    prompts = root / "prompts"
    shutil.copytree(REPO / "src" / "science_agent" / "prompts", prompts)
    cfg = root / "config.yaml"
    cfg.write_text(
        f"output_dir: {json.dumps(str(root / 'reports'))}\nliterature:\n  enabled: false\n" + extra_config,
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
        profiles=ProfileStore(root / "data" / "profiles", Fernet.generate_key()),
        checkpoints=CheckpointStore(root / "data" / "checkpoints"),
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


def wait_for_end(srv, run_id, headers=None):
    deadline = time.time() + 10
    while time.time() < deadline:
        _, snap = call(srv, "GET", f"/api/runs/{run_id}", headers=headers)
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


SHARED = """server:
  mode: shared
  admins: [boss@example.com]
  allowed_origins: [https://spark.tailnet.ts.net]
  max_runs_per_user: 1
"""


@pytest.fixture
def shared(tmp_path):
    srv = make_server(tmp_path, SHARED)
    yield srv
    srv.shutdown()
    srv.server_close()


def as_user(login):
    return {"Tailscale-User-Login": login}


def guest_run(srv, login, idea="A modulation that beats Shannon.", **extra):
    status, res = call(srv, "POST", "/api/runs", {"idea": idea, **extra}, headers={
        "Content-Type": "application/json", **as_user(login)})
    assert status == 200, res
    _, snap = call(srv, "GET", f"/api/runs/{res['id']}", headers=as_user(login))
    deadline = time.time() + 10
    while not any(e["kind"] == "end" for e in snap["events"]) and time.time() < deadline:
        time.sleep(0.05)
        _, snap = call(srv, "GET", f"/api/runs/{res['id']}", headers=as_user(login))
    return res["id"], snap


def test_shared_mode_requires_the_tailscale_identity(shared):
    status, res = call(shared, "GET", "/api/config")
    assert status == 403 and "Tailscale" in res["error"]
    _, cfg = call(shared, "GET", "/api/config", headers=as_user("guest@example.com"))
    assert cfg["mode"] == "shared" and cfg["admin"] is False
    assert "base_url" not in cfg["llm"] and "output_dir" not in cfg  # no server internals for guests
    _, cfg = call(shared, "GET", "/api/config", headers=as_user("Boss@Example.com"))
    assert cfg["admin"] is True and "output_dir" in cfg


def test_guests_cannot_change_the_server_or_its_prompts(shared):
    g = {"Content-Type": "application/json", **as_user("guest@example.com")}
    assert call(shared, "POST", "/api/versions/save", {"key": "prompt:system", "text": "x"}, headers=g)[0] == 403
    assert call(shared, "POST", "/api/import", {"format": "pandora-export"}, headers=g)[0] == 403
    status, res = call(shared, "POST", "/api/models", {"base_url": "http://192.168.0.1/v1"}, headers=g)
    assert status == 400 and "private" in res["error"].lower() or "https" in res["error"].lower()
    _, bundle = call(shared, "POST", "/api/export", {"runs": "all", "versions": True}, headers=g)
    assert bundle["versions"] == {"items": {}}


def test_guests_use_the_server_model_and_see_only_their_own_runs(shared):
    run_a, snap = guest_run(shared, "alice@example.com", llm={"base_url": "http://evil.example/v1", "model": "x"},
                            s2_api_key="stolen")
    assert snap["status"] == "done", snap["error"]
    # The browser's endpoint, model and keys were ignored for a guest.
    assert shared.seen_configs[-1].llm.base_url != "http://evil.example/v1"
    assert shared.seen_configs[-1].llm.model != "x"
    start = next(e for e in snap["events"] if e["kind"] == "start")
    assert start["base_url"] is None
    assert snap["result"]["assessment"]["meta"]["owner"] == "alice@example.com"

    bob = as_user("bob@example.com")
    assert call(shared, "GET", f"/api/runs/{run_a}", headers=bob)[0] == 404
    assert call(shared, "GET", "/api/reports", headers=bob)[1] == []
    name = call(shared, "GET", "/api/reports", headers=as_user("alice@example.com"))[1][0]["name"]
    assert call(shared, "GET", f"/api/reports/{name}", headers=bob)[0] == 404
    assert call(shared, "GET", f"/api/reports/{name}", headers=as_user("boss@example.com"))[0] == 200
    assert call(shared, "GET", "/api/runs/current", headers=bob)[1] == {"run": None}


def test_tailscale_origin_is_accepted_and_others_are_not(shared):
    base = {"Content-Type": "application/json", **as_user("alice@example.com")}
    ok = call(shared, "POST", "/api/export", {"runs": []}, headers={**base, "Origin": "https://spark.tailnet.ts.net"})
    assert ok[0] == 200
    bad = call(shared, "POST", "/api/export", {"runs": []}, headers={**base, "Origin": "https://evil.example"})
    assert bad[0] == 403


def test_runs_queue_up_with_positions_and_a_per_user_limit(tmp_path):
    srv = make_server(tmp_path, SHARED.replace("max_runs_per_user: 1", "max_runs_per_user: 2"))
    gate = threading.Event()
    original = srv.llm.complete

    def slow(system, user):
        gate.wait(5)
        return original(system, user)

    srv.llm.complete = slow
    try:
        h = {"Content-Type": "application/json", **as_user("carol@example.com")}
        first = call(srv, "POST", "/api/runs", {"idea": "one"}, headers=h)[1]["id"]
        second = call(srv, "POST", "/api/runs", {"idea": "two"}, headers=h)[1]["id"]
        status, res = call(srv, "POST", "/api/runs", {"idea": "three"}, headers=h)
        assert status == 409 and "limit 2" in res["error"]
        time.sleep(0.2)
        snap = call(srv, "GET", f"/api/runs/{second}", headers=as_user("carol@example.com"))[1]
        assert snap["status"] == "queued" and snap["queue_position"] == 1
        # Cancelling a queued run removes it at once.
        assert call(srv, "POST", f"/api/runs/{second}/cancel", {}, headers=h)[0] == 200
        snap = call(srv, "GET", f"/api/runs/{second}", headers=as_user("carol@example.com"))[1]
        assert snap["status"] == "cancelled"
        gate.set()
        assert wait_for_end(srv, first, as_user("carol@example.com"))["status"] == "done"
    finally:
        gate.set()
        srv.shutdown()
        srv.server_close()


# -- guests with their own API -----------------------------------------------------------------

OWN_URL = "https://93.184.216.34/v1"  # a public IP literal: no DNS lookup needed in tests
OWN_KEY = "sk-guest-key-0123456789"


def guest_headers(login):
    return {"Content-Type": "application/json", **as_user(login)}


def save_profile(srv, login, **fields):
    body = {"provider": "openai_compatible", "base_url": OWN_URL, "model": "own-model",
            "api_key": OWN_KEY, **fields}
    return call(srv, "POST", "/api/profile", body, headers=guest_headers(login))


def test_profile_is_private_to_the_guest_and_encrypted_on_disk(shared, tmp_path):
    status, res = save_profile(shared, "alice@example.com", s2_api_key="s2k-guest")
    assert status == 200, res
    assert res["profile"]["has_api_key"] and "api_key" not in res["profile"] and OWN_KEY not in json.dumps(res)

    _, mine = call(shared, "GET", "/api/profile", headers=as_user("alice@example.com"))
    assert mine["enabled"] and mine["profile"]["model"] == "own-model" and mine["profile"]["api_key_hint"] == "…6789"
    assert OWN_KEY not in json.dumps(mine)
    assert call(shared, "GET", "/api/profile", headers=as_user("bob@example.com"))[1]["profile"] is None

    stored = b"".join(p.read_bytes() for p in (tmp_path / "data" / "profiles").iterdir())
    assert stored and OWN_KEY.encode() not in stored and b"93.184.216.34" not in stored and b"s2k-guest" not in stored

    # Updating without retyping the key keeps it.
    status, res = save_profile(shared, "alice@example.com", api_key="", model="other-model")
    assert status == 200 and res["profile"]["has_api_key"] and res["profile"]["model"] == "other-model"

    assert call(shared, "POST", "/api/profile/delete", {}, headers=guest_headers("alice@example.com"))[1] == {"deleted": True}
    assert call(shared, "GET", "/api/profile", headers=as_user("alice@example.com"))[1]["profile"] is None


@pytest.mark.parametrize("url", [
    "https://192.168.2.84:8008/v1",
    "https://127.0.0.1:8765/v1",
    "https://100.116.110.73/v1",
    "http://93.184.216.34/v1",
])
def test_guests_cannot_point_pandora_at_internal_or_unencrypted_endpoints(shared, url):
    status, res = save_profile(shared, "mallory@example.com", base_url=url)
    assert status == 400, res
    assert call(shared, "GET", "/api/profile", headers=as_user("mallory@example.com"))[1]["profile"] is None
    status, res = call(shared, "POST", "/api/models", {"base_url": url}, headers=guest_headers("mallory@example.com"))
    assert status == 400


def test_admin_allowed_hosts_may_be_internal(tmp_path):
    srv = make_server(tmp_path, SHARED + "  guest_allowed_hosts: [api.hochschule.test]\n")
    try:
        status, res = save_profile(srv, "carol@example.com", base_url="http://api.hochschule.test/v1")
        assert status == 200, res
    finally:
        srv.shutdown()
        srv.server_close()


def test_admin_has_no_profile_and_the_feature_can_be_switched_off(shared, tmp_path):
    assert call(shared, "POST", "/api/profile", {"model": "m"}, headers=guest_headers("boss@example.com"))[0] == 403
    assert call(shared, "GET", "/api/profile", headers=as_user("boss@example.com"))[1]["enabled"] is False

    off = make_server(tmp_path / "off", SHARED + "  guests_may_use_own_api: false\n")
    try:
        assert save_profile(off, "alice@example.com")[0] == 403
        assert call(off, "GET", "/api/profile", headers=as_user("alice@example.com"))[1]["enabled"] is False
    finally:
        off.shutdown()
        off.server_close()


def test_a_run_on_the_guests_own_api_uses_only_their_settings(shared):
    save_profile(shared, "alice@example.com", s2_api_key="s2k-guest", max_tokens=5000, timeout=120)
    run_id, snap = guest_run(shared, "alice@example.com", llm_source="own",
                             llm={"base_url": "http://evil.example/v1", "model": "evil", "api_key": "stolen"})
    assert snap["status"] == "done", snap["error"]
    cfg = shared.seen_configs[-1]
    assert (cfg.llm.base_url, cfg.llm.model, cfg.llm.api_key()) == (OWN_URL, "own-model", OWN_KEY)
    assert cfg.llm.max_tokens == 5000 and cfg.llm.timeout == 120
    assert cfg.stages == {}  # the server's per-stage models (and their keys) do not apply
    start = next(e for e in snap["events"] if e["kind"] == "start")
    assert start["base_url"] == OWN_URL and start["source"] == "own"
    meta = snap["result"]["assessment"]["meta"]
    assert meta["llm_source"] == "own-api" and meta["owner"] == "alice@example.com"
    assert OWN_KEY not in json.dumps(snap) and "s2k-guest" not in json.dumps(snap)
    # The server's model is untouched for the next run.
    _, snap = guest_run(shared, "alice@example.com")
    assert shared.seen_configs[-1].llm.model != "own-model"


def test_own_api_run_needs_a_saved_profile_and_rechecks_the_endpoint(shared):
    h = guest_headers("dave@example.com")
    status, res = call(shared, "POST", "/api/runs", {"idea": "x", "llm_source": "own"}, headers=h)
    assert status == 400 and "Save your own API settings" in res["error"]
    save_profile(shared, "dave@example.com")
    # The administrator tightens the rules afterwards: the saved endpoint is refused at run time.
    shared.profiles.save("dave@example.com", Profile(base_url="https://10.0.0.7/v1", model="m", api_key=OWN_KEY))
    status, res = call(shared, "POST", "/api/runs", {"idea": "x", "llm_source": "own"}, headers=h)
    assert status == 400 and "private or internal" in res["error"]


def test_server_keys_are_never_used_for_a_guests_requests(shared, monkeypatch):
    from science_agent import web

    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-anthropic-key")
    calls = []
    monkeypatch.setattr(web, "list_models", lambda *a, **k: calls.append(a) or [])
    status, res = call(shared, "POST", "/api/models", {"provider": "anthropic"}, headers=guest_headers("eve@example.com"))
    assert status == 400 and "your Anthropic API key" in res["error"] and calls == []

    save_profile(shared, "eve@example.com")
    call(shared, "POST", "/api/models", {"base_url": OWN_URL}, headers=guest_headers("eve@example.com"))
    assert calls[-1][2] == OWN_KEY  # the key she saved for exactly this endpoint
    call(shared, "POST", "/api/models", {"base_url": "https://8.8.8.8/v1"}, headers=guest_headers("eve@example.com"))
    assert calls[-1][2] is None  # a different endpoint does not get her key


def test_runs_on_the_guests_own_api_do_not_wait_for_the_server_model(tmp_path):
    srv = make_server(tmp_path, SHARED.replace("max_runs_per_user: 1", "max_runs_per_user: 3"))
    gate = threading.Event()
    slow, fast = ScriptedLLM(), ScriptedLLM()
    original = slow.complete
    slow.complete = lambda system, user: (gate.wait(10), original(system, user))[1]
    srv.llm_factory = lambda cfg: {s: (fast if cfg.llm.model == "own-model" else slow) for s in STAGES}
    try:
        save_profile(srv, "carol@example.com")
        h = guest_headers("carol@example.com")
        blocked = call(srv, "POST", "/api/runs", {"idea": "on the server model"}, headers=h)[1]["id"]
        mine = call(srv, "POST", "/api/runs", {"idea": "on my own api", "llm_source": "own"}, headers=h)[1]["id"]
        done = wait_for_end(srv, mine, as_user("carol@example.com"))
        assert done["status"] == "done"
        still = call(srv, "GET", f"/api/runs/{blocked}", headers=as_user("carol@example.com"))[1]
        assert still["status"] == "running"  # the server-model run is still going
        gate.set()
        assert wait_for_end(srv, blocked, as_user("carol@example.com"))["status"] == "done"
    finally:
        gate.set()
        srv.shutdown()
        srv.server_close()


def test_an_interrupted_run_resumes_after_its_last_finished_stage(server):
    from .conftest import CANNED

    server.llm.responses["critic"] = ["not json", "still not json"]  # the critic stage fails
    status, res = call(server, "POST", "/api/runs", {"idea": "A modulation that beats Shannon.", "llm": {"model": "m"}})
    failed = wait_for_end(server, res["id"])
    assert failed["status"] == "error" and failed["resumable"]
    _, unfinished = call(server, "GET", "/api/unfinished")
    assert [u["run_id"] for u in unfinished] == [res["id"]]
    assert unfinished[0]["done"] == ["formalization", "literature", "limits", "computation"]

    # The model is back: resuming skips everything that was done.
    server.llm.responses["critic"] = CANNED["critic"]
    server.llm.calls.clear()
    status, again = call(server, "POST", "/api/runs", {"resume_from": res["id"], "llm": {"model": "m"}})
    assert status == 200, again
    snap = wait_for_end(server, again["id"])
    assert snap["status"] == "done", snap["error"]
    assert [stage for stage, _ in server.llm.calls] == ["critic", "verdict"]
    resume_event = next(e for e in snap["events"] if e["kind"] == "resume")
    assert resume_event["from_run"] == res["id"] and "limits" in resume_event["stages"]
    assert snap["result"]["assessment"]["idea"] == "A modulation that beats Shannon."
    assert snap["result"]["assessment"]["computation"]["ok"]
    assert call(server, "GET", "/api/unfinished")[1] == []  # nothing left over


def test_resuming_an_unknown_run_is_refused_and_discarding_works(server):
    status, res = call(server, "POST", "/api/runs", {"resume_from": "abcdef123456", "llm": {"model": "m"}})
    assert status == 400 and "no interrupted run" in res["error"].lower()
    server.checkpoints.save("abcdef123456", "local", {"idea": "x", "clarifications": [], "assessment": {}})
    assert len(call(server, "GET", "/api/unfinished")[1]) == 1
    assert call(server, "POST", "/api/unfinished/discard", {"run_id": "abcdef123456"})[0] == 200
    assert call(server, "GET", "/api/unfinished")[1] == []
