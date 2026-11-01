"""Per-user API profiles for guests: endpoint, model and keys, encrypted at rest.

One file per Tailscale login under <data dir>/profiles/. The file holds a Fernet token
(AES-128-CBC with HMAC-SHA256); the secret that protects it comes from the PANDORA_SECRET_KEY
environment variable, or, if that is not set, from a key file next to the data that is created
on first use. Keep the environment variable outside the data volume (docker-compose reads it
from .env): then a copied volume or backup alone does not reveal anyone's keys.

Keys are never returned to the browser, only whether one is set and its last four characters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)

SECRET_ENV = "PANDORA_SECRET_KEY"
KEY_FILE = "secret.key"
PROVIDERS = ("openai_compatible", "anthropic")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")


class ProfileError(ValueError):
    pass


def load_master_key(data_dir: Path, environ: dict | None = None) -> bytes:
    """The Fernet key: from PANDORA_SECRET_KEY, else from <data_dir>/secret.key (created if missing)."""
    env = environ if environ is not None else os.environ
    value = (env.get(SECRET_ENV) or "").strip()
    if value:
        try:
            Fernet(value.encode())
        except (ValueError, TypeError) as e:
            raise ProfileError(
                f"{SECRET_ENV} is not a valid key. Create one with: "
                'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
            ) from e
        return value.encode()
    path = data_dir / KEY_FILE
    if path.exists():
        return path.read_bytes().strip()
    log.warning(
        "%s is not set: generated %s next to the data. Set %s from outside the data volume "
        "to keep stored API keys unreadable when the volume is copied.", SECRET_ENV, path, SECRET_ENV,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


@dataclass
class Profile:
    provider: str = "openai_compatible"
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    s2_api_key: str = field(default="", repr=False)
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: float = 300
    json_mode: bool = False
    effort: str = "high"

    def public(self) -> dict:
        """What the browser may see: everything except the keys, which are only hinted at."""
        d = {k: v for k, v in self.__dict__.items() if k not in ("api_key", "s2_api_key")}
        d["has_api_key"] = bool(self.api_key)
        d["api_key_hint"] = ("…" + self.api_key[-4:]) if len(self.api_key) >= 12 else ("…" if self.api_key else "")
        d["has_s2_api_key"] = bool(self.s2_api_key)
        return d


def _secret_text(value, what: str) -> str:
    text = str(value or "").strip()
    if _CONTROL.search(text):
        raise ProfileError(f"The {what} must not contain spaces or line breaks")
    return text


def clean_profile(body: dict, current: Profile | None, check_endpoint, max_tokens_cap: int, timeout_cap: float) -> Profile:
    """A validated Profile from the form fields. A blank key keeps the stored one; keys are only
    replaced when a new one is typed. `check_endpoint(url) -> url` validates the endpoint."""
    base = current or Profile()
    provider = str(body.get("provider") or base.provider)
    if provider not in PROVIDERS:
        raise ProfileError("Unknown provider")
    model = str(body.get("model") if "model" in body else base.model).strip()
    if not model:
        raise ProfileError("Enter a model name (Load models lists the ones your endpoint offers)")
    api_key = _secret_text(body.get("api_key"), "API key") or base.api_key
    s2 = _secret_text(body.get("s2_api_key"), "Semantic Scholar key") or base.s2_api_key
    if body.get("clear_api_key"):
        api_key = ""
    if body.get("clear_s2_api_key"):
        s2 = ""
    if provider == "anthropic":
        base_url = ""
        if not api_key:
            raise ProfileError("The Anthropic API needs your own API key")
    else:
        base_url = check_endpoint(str(body.get("base_url") if "base_url" in body else base.base_url))
    try:
        temperature = float(body.get("temperature", base.temperature))
        max_tokens = int(body.get("max_tokens", base.max_tokens))
        timeout = float(body.get("timeout", base.timeout))
    except (TypeError, ValueError) as e:
        raise ProfileError("Temperature, token limit and timeout must be numbers") from e
    if not 0 <= temperature <= 2:
        raise ProfileError("Temperature must be between 0 and 2")
    if not 256 <= max_tokens <= max_tokens_cap:
        raise ProfileError(f"Max output tokens must be between 256 and {max_tokens_cap}")
    if not 10 <= timeout <= timeout_cap:
        raise ProfileError(f"Timeout must be between 10 and {int(timeout_cap)} seconds")
    effort = str(body.get("effort") or base.effort)
    if effort not in EFFORTS:
        raise ProfileError("Unknown effort level")
    return Profile(
        provider=provider, base_url=base_url, model=model, api_key=api_key, s2_api_key=s2,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        json_mode=bool(body.get("json_mode", base.json_mode)), effort=effort,
    )


class ProfileStore:
    def __init__(self, directory: Path, key: bytes):
        self.dir = Path(directory)
        self.fernet = Fernet(key)
        self.lock = threading.Lock()

    def _path(self, login: str) -> Path:
        return self.dir / (hashlib.sha256(login.strip().lower().encode()).hexdigest() + ".enc")

    def load(self, login: str) -> Profile | None:
        path = self._path(login)
        try:
            token = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(self.fernet.decrypt(token))
        except (InvalidToken, ValueError):
            log.error("Cannot decrypt the profile file %s (wrong %s?)", path.name, SECRET_ENV)
            raise ProfileError("Your saved settings cannot be read. Save them again.") from None
        if payload.get("login", "").lower() != login.strip().lower():
            raise ProfileError("Your saved settings cannot be read. Save them again.")
        fields = {k: v for k, v in payload.get("profile", {}).items() if k in Profile.__dataclass_fields__}
        return Profile(**fields)

    def save(self, login: str, profile: Profile) -> None:
        payload = json.dumps({"login": login.strip().lower(), "profile": profile.__dict__}).encode()
        token = self.fernet.encrypt(payload)
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self._path(login)
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "wb") as f:
                f.write(token)
            os.replace(tmp, path)

    def delete(self, login: str) -> bool:
        try:
            self._path(login).unlink()
            return True
        except FileNotFoundError:
            return False
