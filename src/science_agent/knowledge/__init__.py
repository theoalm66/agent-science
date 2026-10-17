"""Domain knowledge bases: known limits, their assumptions and legitimate workarounds."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml


def available_domains() -> list[str]:
    return sorted(
        p.name.removesuffix(".yaml")
        for p in resources.files(__package__).iterdir()
        if p.name.endswith(".yaml")
    )


def load_domain(name_or_path: str) -> dict:
    """Load a bundled domain by name, or any YAML file by path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        resource = resources.files(__package__) / f"{name_or_path}.yaml"
        if not resource.is_file():
            raise ValueError(f"Unknown domain {name_or_path!r}; available: {available_domains()}")
        text = resource.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    data.setdefault("limits", [])
    return data


def render_index(domain: dict) -> str:
    """One line per limit: enough for the formalize stage to pick vocabulary."""
    return "\n".join(f"- {lim['name']}" for lim in domain["limits"]) or "(none)"


def render_full(domain: dict) -> str:
    """Full limit entries for the limits stage."""
    return yaml.safe_dump(domain["limits"], sort_keys=False, allow_unicode=True, width=100)
