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
    """Chat-completions client for vLLM, LM Studio, Ollama, llama.cpp, OpenAI, Gemini, OpenRouter, ..."""

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
        # OpenAI's current models reject max_tokens in favour of max_completion_tokens.
        self.token_param = "max_completion_tokens" if "api.openai.com" in cfg.base_url else "max_tokens"
        self.send_temperature = True
        # "chat" (/chat/completions) or "responses" (/responses). OpenAI serves some models, such
        # as the *-pro ones, only through the Responses API; we switch when the server says so.
        self.api = "chat"

    def _body(self, system: str, user: str) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self.token_param: self.cfg.max_tokens,
        }
        if self.send_temperature:
            body["temperature"] = self.cfg.temperature
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _responses_body(self, system: str, user: str) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.cfg.max_tokens,
        }
        if self.send_temperature:
            body["temperature"] = self.cfg.temperature
        if self.cfg.json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        return body

    def _adapt(self, error_text: str) -> bool:
        """Drop or rename a parameter the server rejected (reasoning models); True if retrying helps."""
        text = error_text.lower()
        if self.api == "chat" and "not a chat model" in text:
            self.api = "responses"
            return True
        if self.api == "chat" and self.token_param == "max_tokens" and "max_tokens" in text and (
            "max_completion_tokens" in text or "unsupported" in text
        ):
            self.token_param = "max_completion_tokens"
            return True
        if self.send_temperature and "temperature" in text and any(
            w in text for w in ("unsupported", "not supported", "does not support", "only the default")
        ):
            self.send_temperature = False
            return True
        return False

    def complete(self, system: str, user: str) -> str:
        attempt = adaptations = 0
        while True:
            if self.api == "responses":
                url, body = self.cfg.base_url.rstrip("/") + "/responses", self._responses_body(system, user)
            else:
                url, body = self.cfg.base_url.rstrip("/") + "/chat/completions", self._body(system, user)
            try:
                resp = self.client.post(url, json=body, headers=self.headers)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"Cannot reach {url}: {e}") from e
                attempt += 1
                time.sleep(2**attempt)
                continue
            if resp.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                attempt += 1
                time.sleep(2**attempt)
                continue
            if resp.status_code in (400, 404) and adaptations < 3 and self._adapt(resp.text):
                adaptations += 1
                continue
            if not resp.is_success:  # some gateways (e.g. RH Köln) answer 201
                raise LLMError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if self.api == "responses":
                return self._responses_text(data)
            try:
                choice = data["choices"][0]
                content = choice["message"].get("content")
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                raise LLMError(f"Unexpected response shape: {str(data)[:500]}") from e
            if not content and choice.get("finish_reason") == "length":
                raise LLMError(
                    f"{self.cfg.model} used all {self.cfg.max_tokens} tokens before answering "
                    "(reasoning models count their thinking); raise max_tokens"
                )
            return strip_think_tags(content or "")


    def _responses_text(self, data: dict) -> str:
        """Collect the output_text parts of the message items; reasoning items come first and are skipped."""
        parts = [
            part.get("text") or ""
            for item in data.get("output") or []
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        ]
        content = "".join(parts)
        if not content and data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
            raise LLMError(
                f"{self.cfg.model} stopped before answering ({reason}); reasoning models count their "
                f"thinking against the {self.cfg.max_tokens}-token limit, so raise max_tokens"
            )
        if not content:
            raise LLMError(f"Unexpected response shape: {str(data)[:500]}")
        return strip_think_tags(content)


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
