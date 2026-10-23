"""Configuration loading. One YAML file, with per-stage LLM overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

STAGES = ("formalize", "literature", "limits", "compute", "critic", "verdict")


class LLMConfig(BaseModel):
    provider: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    base_url: str | None = "http://localhost:11434/v1"
    model: str = "qwen3:32b"
    api_key_env: str | None = None
    # A literal key, used by the web UI so keys typed there never touch disk. Takes precedence.
    api_key_value: str | None = Field(default=None, repr=False)
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: float = 600
    json_mode: bool = False
    effort: str | None = "high"  # anthropic only

    def api_key(self) -> str | None:
        if self.api_key_value:
            return self.api_key_value
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class LiteratureConfig(BaseModel):
    enabled: bool = True
    max_results: int = 5
    semantic_scholar_api_key_env: str | None = "S2_API_KEY"


class ComputeConfig(BaseModel):
    enabled: bool = True
    timeout: float = 60
    max_attempts: int = 2
    # URL of the sandbox runner (python -m science_agent.tools.sandbox_server). When set, model-written
    # scripts run there instead of in a local subprocess, e.g. in the isolated container of docker-compose.
    sandbox_url: str | None = None


class ServerConfig(BaseModel):
    """How Pandora is shared. "local": one user on this machine. "shared": several people behind
    Tailscale Serve, identified by the Tailscale-User-Login header that Serve sets (and strips from
    clients), so Pandora must only be reachable through Serve (bind it to localhost)."""

    mode: Literal["local", "shared"] = "local"
    admins: list[str] = Field(default_factory=list)  # Tailscale logins with full rights
    identity_header: str = "Tailscale-User-Login"
    # Origins a browser may POST from besides http://<Host>, e.g. the https://<machine>.<tailnet>.ts.net URL.
    allowed_origins: list[str] = Field(default_factory=list)
    max_runs_per_user: int = 2  # queued plus running, per person


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    domain: str = "information_theory"
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    output_dir: Path = Path("reports")
    # About 4 characters per token: 200k characters is ~50k tokens, which the model's
    # context window (e.g. vLLM --max-model-len) must also hold next to the prompt.
    max_idea_chars: int = 200_000

    def llm_for(self, stage: str) -> LLMConfig:
        """LLM settings for a stage: the base config with that stage's overrides applied."""
        override = self.stages.get(stage) or {}
        if not override:
            return self.llm
        return LLMConfig.model_validate({**self.llm.model_dump(), **override})


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path`, $SCIENCE_AGENT_CONFIG, or ./config.yaml; defaults otherwise."""
    candidate = path or os.environ.get("SCIENCE_AGENT_CONFIG") or "config.yaml"
    p = Path(candidate)
    if not p.exists():
        if path:  # explicitly requested but missing
            raise FileNotFoundError(f"Config file not found: {p}")
        return Config()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    unknown = set(data.get("stages") or {}) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s) in config: {sorted(unknown)}; valid: {STAGES}")
    return Config.model_validate(data)
