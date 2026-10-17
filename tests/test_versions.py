import shutil
from pathlib import Path

import pytest

from science_agent.pipeline import Assessment
from science_agent.report import save
from science_agent.versions import VersionStore, normalize

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    prompts = tmp_path / "prompts"
    shutil.copytree(REPO / "src" / "science_agent" / "prompts", prompts)
    shutil.copytree(REPO / ".claude", tmp_path / "project" / ".claude")
    return VersionStore(tmp_path / "data" / "versions.json", prompts, tmp_path / "project")


def test_files_cover_prompts_skills_and_agents(store):
    keys = set(store.files())
    assert {"prompt:system", "prompt:limits", "skill:assess-idea", "agent:critic"} <= keys


def test_save_keeps_original_and_writes_file(store):
    original = store.read("prompt:limits")
    edited = original.replace("This is the core", "This is the very core")
    item = store.save_version("prompt:limits", edited, "stronger wording")
    assert [v["n"] for v in item["versions"]] == [1, 2]
    assert item["versions"][0]["note"] == "original" and item["versions"][0]["text"] == original
    assert item["active"] == 2
    assert store.read("prompt:limits") == normalize(edited)

    item = store.activate("prompt:limits", 1)
    assert item["active"] == 1 and store.read("prompt:limits") == original
    # Saving a text that already exists does not duplicate it.
    assert len(store.save_version("prompt:limits", edited)["versions"]) == 2


def test_outside_edit_is_recorded_before_it_is_replaced(store):
    path = store.files()["prompt:critic"]
    store.save_version("prompt:critic", store.read("prompt:critic") + "\nExtra rule.\n")
    path.write_text(path.read_text(encoding="utf-8") + "Edited in an editor.\n", encoding="utf-8")
    assert store.item("prompt:critic")["active"] is None  # unversioned change
    item = store.activate("prompt:critic", 1)
    assert [v["note"] for v in item["versions"]][-1] == "changed outside Pandora"
    assert any("Edited in an editor." in v["text"] for v in item["versions"])


def test_validation_and_delete_rules(store):
    with pytest.raises(ValueError, match="# Stage: limits"):
        store.save_version("prompt:limits", "Check the claim.\n")
    with pytest.raises(ValueError, match="front matter"):
        store.save_version("agent:critic", "no front matter\n")
    store.save_version("prompt:verdict", store.read("prompt:verdict") + "\nBe brief.\n")
    with pytest.raises(ValueError, match="active"):
        store.delete("prompt:verdict", 2)
    with pytest.raises(ValueError, match="original"):
        store.delete("prompt:verdict", 1)
    store.activate("prompt:verdict", 1)
    assert [v["n"] for v in store.delete("prompt:verdict", 2)["versions"]] == [1]


def test_crlf_files_stay_crlf(store):
    path = store.files()["prompt:system"]
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    store.save_version("prompt:system", store.read("prompt:system") + "- One more rule.\n")
    raw = path.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_snapshot_and_import(store, tmp_path):
    store.save_version("prompt:formalize", store.read("prompt:formalize") + "\nKeep it short.\n", "short")
    texts, versions = store.snapshot_prompts()
    assert "Keep it short." in texts["formalize"]
    assert versions["formalize"]["version"] == 2 and versions["limits"]["version"] is None

    other = VersionStore(tmp_path / "other.json", store.prompts_dir, store.project_dir)
    assert other.import_versions(store.export()) == 1  # v2 is new there; the original is already the file
    assert other.import_versions(store.export()) == 0


def test_repeated_runs_of_an_idea_do_not_overwrite(tmp_path):
    a = Assessment(idea="Same idea", domain="d", started_at="2026-10-17T10:00:00+00:00")
    first, second = save(a, tmp_path), save(a, tmp_path)
    assert first.stem == "2026-10-17-same-idea" and second.stem == "2026-10-17-same-idea-2"
