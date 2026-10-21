"""Provider-neutral LLM interface.

The pipeline only needs "system + user prompt in, text out". Keeping the
interface that small is what makes local models usable: no reliance on native
tool calling, which many local models implement poorly.
"""

from __future__ import annotations

import re
import time
from typing import Protocol

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
    """Chat-completions client for Ollama, LM Studio, vLLM, llama.cpp, OpenRouter, OpenAI."""

    RETRY_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, cfg: LLMConfig, client: httpx.Client | None = None, max_retries: int = 3):
        if not cfg.base_url:
            raise LLMError("openai_compatible provider needs llm.base_url")
        self.cfg = cfg
        self.name = f"{cfg.model} @ {cfg.base_url}"
        headers = {"Content-Type": "application/json"}
        key = cfg.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.client = client or httpx.Client(timeout=cfg.timeout)
        self.headers = headers
        self.max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        body: dict = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.post(url, json=body, headers=self.headers)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"Cannot reach {url}: {e}") from e
                time.sleep(2**attempt)
                continue
            if resp.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue
            if not resp.is_success:  # some gateways (e.g. RH Köln) answer 201
                raise LLMError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"Unexpected response shape: {str(data)[:500]}") from e
            return strip_think_tags(content or "")
        raise LLMError("unreachable")  # pragma: no cover


class AnthropicLLM:
    """Claude via the official Anthropic SDK (optional dependency)."""

    def __init__(self, cfg: LLMConfig, client=None):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on install
            raise LLMError('Install the extra: pip install -e ".[anthropic]"') from e
        self.anthropic = anthropic
        self.cfg = cfg
        self.name = cfg.model
        key = cfg.api_key()
        self.client = client or anthropic.Anthropic(
            **({"api_key": key} if key else {}), timeout=cfg.timeout
        )

    def complete(self, system: str, user: str) -> str:
        kwargs: dict = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # On a safety decline, let the API re-run the request on a suitable model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        try:
            response = self.client.beta.messages.create(**kwargs)
        except self.anthropic.APIConnectionError as e:
            raise LLMError(f"Cannot reach Anthropic API: {e}") from e
        except self.anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        if response.stop_reason == "refusal":
            raise LLMError(f"Model declined the request: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (Qwen, DeepSeek-R1) emit inline."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Some servers drop the opening tag and return only "...reasoning</think>answer".
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def make_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg)
    return OpenAICompatibleLLM(cfg)
