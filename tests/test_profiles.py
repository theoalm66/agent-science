import os
import stat

import pytest
from cryptography.fernet import Fernet

from science_agent.profiles import (
    Profile,
    ProfileError,
    ProfileStore,
    clean_profile,
    load_master_key,
)

KEY = "sk-live-0123456789abcdef"


def ok_endpoint(url):
    return url.strip().rstrip("/")


def make(body, current=None):
    return clean_profile(body, current, ok_endpoint, max_tokens_cap=64_000, timeout_cap=600)


def test_master_key_from_environment_wins_and_is_validated(tmp_path):
    key = Fernet.generate_key().decode()
    assert load_master_key(tmp_path, {"PANDORA_SECRET_KEY": key}) == key.encode()
    assert not (tmp_path / "secret.key").exists()
    with pytest.raises(ProfileError, match="not a valid key"):
        load_master_key(tmp_path, {"PANDORA_SECRET_KEY": "too short"})


def test_master_key_file_is_created_private_and_reused(tmp_path):
    first = load_master_key(tmp_path / "data", {})
    path = tmp_path / "data" / "secret.key"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_master_key(tmp_path / "data", {}) == first
    Fernet(first)  # a usable key


def test_profile_round_trip_is_encrypted_on_disk(tmp_path):
    store = ProfileStore(tmp_path / "profiles", Fernet.generate_key())
    profile = Profile(provider="openai_compatible", base_url="https://api.openai.com/v1", model="gpt-x",
                      api_key=KEY, s2_api_key="s2k-secret")
    store.save("Alice@Example.com", profile)

    loaded = store.load("alice@example.com")  # logins are case-insensitive
    assert loaded == profile
    files = list((tmp_path / "profiles").iterdir())
    assert len(files) == 1 and files[0].suffix == ".enc"
    raw = files[0].read_bytes()
    for secret in (KEY, "s2k-secret", "api.openai.com", "alice", "gpt-x"):
        assert secret.encode() not in raw
    assert stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600
    assert "alice" not in files[0].name  # the file name does not reveal the login either


def test_other_users_and_missing_profiles(tmp_path):
    store = ProfileStore(tmp_path, Fernet.generate_key())
    store.save("alice@example.com", Profile(model="m", api_key=KEY))
    assert store.load("bob@example.com") is None
    assert store.delete("bob@example.com") is False
    assert store.delete("alice@example.com") is True
    assert store.load("alice@example.com") is None


def test_wrong_secret_key_or_swapped_file_is_refused(tmp_path):
    store = ProfileStore(tmp_path, Fernet.generate_key())
    store.save("alice@example.com", Profile(model="m", api_key=KEY))
    with pytest.raises(ProfileError, match="cannot be read"):
        ProfileStore(tmp_path, Fernet.generate_key()).load("alice@example.com")
    # Alice's file copied under Bob's file name must not be accepted as Bob's profile.
    (tmp_path / store._path("bob@example.com").name).write_bytes(store._path("alice@example.com").read_bytes())
    with pytest.raises(ProfileError, match="cannot be read"):
        store.load("bob@example.com")


def test_browser_view_never_contains_the_keys():
    view = Profile(model="m", api_key=KEY, s2_api_key="s2k-secret").public()
    assert KEY not in repr(view) and "s2k-secret" not in repr(view)
    assert view["has_api_key"] and view["has_s2_api_key"] and view["api_key_hint"] == "…cdef"
    assert "api_key" not in view and "s2_api_key" not in view
    assert Profile(model="m", api_key="short").public()["api_key_hint"] == "…"  # too short to reveal any part
    assert KEY not in repr(Profile(model="m", api_key=KEY))  # nor in logs


def test_blank_key_keeps_the_stored_one_and_a_new_key_replaces_it():
    current = make({"base_url": "https://api.example.com/v1", "model": "m", "api_key": KEY})
    kept = make({"base_url": "https://api.example.com/v1", "model": "m2", "api_key": ""}, current)
    assert kept.api_key == KEY and kept.model == "m2"
    assert make({"model": "m", "api_key": "sk-new-key-123456"}, current).api_key == "sk-new-key-123456"
    assert make({"model": "m", "clear_api_key": True}, current).api_key == ""


def test_anthropic_needs_a_key_and_has_no_endpoint():
    with pytest.raises(ProfileError, match="own API key"):
        make({"provider": "anthropic", "model": "claude-x"})
    p = make({"provider": "anthropic", "model": "claude-x", "api_key": KEY, "base_url": "http://evil.example"})
    assert p.base_url == ""  # the SDK talks to its own fixed host


@pytest.mark.parametrize("body, message", [
    ({"model": ""}, "model name"),
    ({"model": "m", "provider": "evil"}, "provider"),
    ({"model": "m", "api_key": "sk-abc\nX-Injected: 1"}, "spaces or line breaks"),
    ({"model": "m", "api_key": "has space"}, "spaces or line breaks"),
    ({"model": "m", "temperature": 7}, "Temperature"),
    ({"model": "m", "max_tokens": 10}, "Max output tokens"),
    ({"model": "m", "max_tokens": 10**7}, "Max output tokens"),
    ({"model": "m", "timeout": 99999}, "Timeout"),
    ({"model": "m", "timeout": "soon"}, "numbers"),
    ({"model": "m", "effort": "turbo"}, "effort"),
])
def test_invalid_profiles_are_rejected(body, message):
    with pytest.raises(ProfileError, match=message):
        make({"base_url": "https://api.example.com/v1", **body})


def test_endpoint_check_is_applied():
    def refuse(url):
        raise ValueError("internal address")

    with pytest.raises(ValueError, match="internal address"):
        clean_profile({"base_url": "https://192.168.2.84/v1", "model": "m"}, None, refuse, 64_000, 600)
