"""Multi-provider async Structured Output clients (Gemini, OpenAI, Anthropic, Ollama).

- Gemini uses the official `google-genai` SDK (`response_schema`).
- OpenAI, Anthropic, and local Ollama use stdlib `urllib.request` via `asyncio.to_thread`
  so zero extra heavyweight SDK dependencies are required.
"""

from __future__ import annotations

import asyncio
import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import (
    ANTHROPIC_API_KEY,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

TModel = TypeVar("TModel", bound=BaseModel)
ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]


@dataclass(frozen=True)
class ModelSpec:
    """Curated model slot metadata for the UI Tier Slider Slot."""

    slot_label: str
    provider: ProviderName
    model_id: str
    tier: str
    speed: str
    cost_tier: str
    stability: str
    summary: str


PROVIDER_LABELS: dict[ProviderName, str] = {
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic Claude",
    "ollama": "Local Ollama",
}

# Ordered left-to-right on the Tier Slider: Local/Fast -> Balanced -> Deep Reasoning
MODEL_TIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        slot_label="Ollama · Qwen 4B",
        provider="ollama",
        model_id="qwen3.5:4b",
        tier="Tier 1 · Local",
        speed="Local GPU/CPU",
        cost_tier="$0 Local",
        stability="Local",
        summary="Runs offline via Ollama with native JSON schema output. Zero API cost.",
    ),
    ModelSpec(
        slot_label="Ollama · Qwen 7B",
        provider="ollama",
        model_id="qwen2.5:7b",
        tier="Tier 1 · Local+",
        speed="Local GPU",
        cost_tier="$0 Local",
        stability="Local",
        summary="Stronger local reasoning & instruction following for multi-round debate.",
    ),
    ModelSpec(
        slot_label="Gemini · 3.5 Lite",
        provider="gemini",
        model_id="gemini-3.5-flash-lite",
        tier="Tier 2 · Fast",
        speed="~45s / 2 rounds",
        cost_tier="$ Low",
        stability="Verified",
        summary="Live-verified default. Fast, stable structured outputs with low latency.",
    ),
    ModelSpec(
        slot_label="OpenAI · 4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        tier="Tier 2 · Fast",
        speed="~40s / 2 rounds",
        cost_tier="$ Low",
        stability="Stable",
        summary="Reliable JSON adherence and crisp debate personas at low token cost.",
    ),
    ModelSpec(
        slot_label="Gemini · 3.7 Flash",
        provider="gemini",
        model_id="gemini-3.7-flash",
        tier="Tier 3 · Balanced",
        speed="~55s / 2 rounds",
        cost_tier="$$ Mid",
        stability="Watch 503",
        summary="Richer financial nuance; may occasionally hit 503 during peak demand.",
    ),
    ModelSpec(
        slot_label="Claude · 3.5 Haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-latest",
        tier="Tier 3 · Balanced",
        speed="~50s / 2 rounds",
        cost_tier="$$ Mid",
        stability="Stable",
        summary="Fast Anthropic model with sharp persona contrast via forced tool schema.",
    ),
    ModelSpec(
        slot_label="OpenAI · GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        tier="Tier 4 · Deep",
        speed="~75s / 2 rounds",
        cost_tier="$$$ High",
        stability="Stable",
        summary="High-conviction multi-turn argumentation and nuanced risk synthesis.",
    ),
    ModelSpec(
        slot_label="Claude · 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-latest",
        tier="Tier 4 · Deep",
        speed="~80s / 2 rounds",
        cost_tier="$$$ High",
        stability="Stable",
        summary="Flagship editorial writing quality and disciplined behavioral debate.",
    ),
)

MODEL_BY_SLOT: dict[str, ModelSpec] = {spec.slot_label: spec for spec in MODEL_TIERS}


def find_default_slot_label(provider: str, model_id: str) -> str:
    """Resolve the best matching Tier Slider slot for a configured provider/model."""
    norm_provider = provider.strip().lower()
    norm_model = model_id.strip()
    for spec in MODEL_TIERS:
        if spec.provider == norm_provider and spec.model_id == norm_model:
            return spec.slot_label
    for spec in MODEL_TIERS:
        if spec.model_id == norm_model:
            return spec.slot_label
    for spec in MODEL_TIERS:
        if spec.provider == norm_provider:
            return spec.slot_label
    return "Gemini · 3.5 Lite"


class MissingAPIKeyError(ValueError):
    """Raised when a required provider API key is missing or still a placeholder."""


class LLMGenerationError(RuntimeError):
    """Raised when a structured LLM call fails after retries."""


class StructuredLLMClient(Protocol):
    """Minimal protocol enabling deterministic unit-test fakes without mocking SDK internals."""

    model_name: str

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        ...


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    retry_signals = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "resource exhausted",
        "unavailable",
        "overloaded",
        "timeout",
    )
    return any(signal in msg for signal in retry_signals)


def _format_exc(exc: Exception | None, timeout_seconds: float) -> str:
    """Always produce a non-empty diagnostic message (asyncio.TimeoutError stringifies to '')."""
    if exc is None:
        return "Unknown error"
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return f"Timed out after {timeout_seconds:.0f}s"
    detail = str(exc).strip()
    return detail or type(exc).__name__


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _post_json_sync(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {err_body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Connection failed to {url}: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:240]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON root type from {url}: {type(data).__name__}")
    return data


def _extract_validated_model(response: types.GenerateContentResponse, response_schema: type[TModel]) -> TModel:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, response_schema):
        return parsed
    if isinstance(parsed, dict):
        return response_schema.model_validate(parsed)

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return response_schema.model_validate_json(_strip_json_fences(text))

    raise LLMGenerationError(
        f"Model returned an empty or unparseable response for schema {response_schema.__name__}."
    )


class GeminiLLMClient:
    """Production async client over `google-genai` with structured Pydantic output enforcement."""

    provider: ProviderName = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
    ) -> None:
        resolved_key = (api_key if api_key is not None else GEMINI_API_KEY).strip()
        if not resolved_key or resolved_key == "your_gemini_api_key_here":
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not configured. Add it to `.env` or enter a session key in the sidebar."
            )

        self.model_name = model_name.strip() or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = genai.Client(api_key=resolved_key)

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        """Execute one async Gemini call returning a validated Pydantic model instance."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )
                return _extract_validated_model(response, response_schema)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    break
                delay = (0.75 * (2**attempt)) + random.uniform(0.0, 0.35)
                await asyncio.sleep(delay)

        raise LLMGenerationError(
            f"Gemini call failed for model '{self.model_name}' "
            f"(schema={response_schema.__name__}): {_format_exc(last_error, self.timeout_seconds)}"
        ) from last_error

    async def aclose(self) -> None:
        aio_client = getattr(self._client, "aio", None)
        if aio_client is not None and hasattr(aio_client, "aclose"):
            await aio_client.aclose()
        if hasattr(self._client, "close"):
            self._client.close()

    async def __aenter__(self) -> GeminiLLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


class OpenAILLMClient:
    """Async OpenAI (and OpenAI-compatible) client enforcing Pydantic JSON outputs."""

    provider: ProviderName = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = OPENAI_BASE_URL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
    ) -> None:
        resolved_key = (api_key if api_key is not None else OPENAI_API_KEY).strip()
        if not resolved_key or resolved_key.startswith("your_"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not configured. Add it to `.env` or enter a session key in the sidebar."
            )
        self.api_key = resolved_key
        self.model_name = model_name.strip() or "gpt-4o-mini"
        self.base_url = (base_url or OPENAI_BASE_URL).strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
        enforced_system = (
            f"{system_prompt}\n\n"
            "OUTPUT FORMAT REQUIREMENT:\n"
            "Return ONLY a single valid JSON object conforming strictly to this JSON Schema "
            "(no markdown fences, no commentary):\n"
            f"{schema_json}"
        )
        payload = {
            "model": self.model_name,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": enforced_system},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = await asyncio.to_thread(
                    _post_json_sync, url, payload, headers, self.timeout_seconds
                )
                choices = data.get("choices") or []
                content = choices[0]["message"]["content"] if choices else ""
                if not isinstance(content, str) or not content.strip():
                    raise LLMGenerationError("OpenAI returned an empty message content.")
                return response_schema.model_validate_json(_strip_json_fences(content))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    break
                delay = (0.75 * (2**attempt)) + random.uniform(0.0, 0.35)
                await asyncio.sleep(delay)

        raise LLMGenerationError(
            f"OpenAI call failed for model '{self.model_name}' "
            f"(schema={response_schema.__name__}): {_format_exc(last_error, self.timeout_seconds)}"
        ) from last_error

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> OpenAILLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


class AnthropicLLMClient:
    """Async Anthropic Claude client enforcing Pydantic outputs via forced tool schema."""

    provider: ProviderName = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-3-5-sonnet-latest",
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
    ) -> None:
        resolved_key = (api_key if api_key is not None else ANTHROPIC_API_KEY).strip()
        if not resolved_key or resolved_key.startswith("your_"):
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not configured. Add it to `.env` or enter a session key in the sidebar."
            )
        self.api_key = resolved_key
        self.model_name = model_name.strip() or "claude-3-5-sonnet-latest"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        tool_name = "emit_structured_output"
        payload = {
            "model": self.model_name,
            "max_tokens": 2048,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [
                {
                    "name": tool_name,
                    "description": f"Return structured {response_schema.__name__} JSON.",
                    "input_schema": response_schema.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        url = "https://api.anthropic.com/v1/messages"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = await asyncio.to_thread(
                    _post_json_sync, url, payload, headers, self.timeout_seconds
                )
                content_blocks = data.get("content") or []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return response_schema.model_validate(block.get("input", {}))
                raise LLMGenerationError("Anthropic response did not include a tool_use block.")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    break
                delay = (0.75 * (2**attempt)) + random.uniform(0.0, 0.35)
                await asyncio.sleep(delay)

        raise LLMGenerationError(
            f"Anthropic call failed for model '{self.model_name}' "
            f"(schema={response_schema.__name__}): {_format_exc(last_error, self.timeout_seconds)}"
        ) from last_error

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> AnthropicLLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


class OllamaLLMClient:
    """Async Local Ollama client using native JSON Schema structured outputs."""

    provider: ProviderName = "ollama"

    def __init__(
        self,
        model_name: str = "qwen3.5:4b",
        base_url: str = OLLAMA_BASE_URL,
        timeout_seconds: float = max(LLM_TIMEOUT_SECONDS, 120.0),
        max_retries: int = LLM_MAX_RETRIES,
    ) -> None:
        self.model_name = model_name.strip() or "qwen3.5:4b"
        self.base_url = (base_url or OLLAMA_BASE_URL).strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        payload = {
            "model": self.model_name,
            "stream": False,
            "format": response_schema.model_json_schema(),
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        url = f"{self.base_url}/api/chat"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = await asyncio.to_thread(
                    _post_json_sync, url, payload, {}, self.timeout_seconds
                )
                content = (data.get("message") or {}).get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise LLMGenerationError("Ollama returned an empty message content.")
                return response_schema.model_validate_json(_strip_json_fences(content))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    break
                delay = (0.75 * (2**attempt)) + random.uniform(0.0, 0.35)
                await asyncio.sleep(delay)

        raise LLMGenerationError(
            f"Ollama call failed for model '{self.model_name}' at {self.base_url} "
            f"(schema={response_schema.__name__}): {_format_exc(last_error, self.timeout_seconds)}"
        ) from last_error

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> OllamaLLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def create_llm_client(
    *,
    provider: ProviderName,
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
) -> GeminiLLMClient | OpenAILLMClient | AnthropicLLMClient | OllamaLLMClient:
    """Instantiate the appropriate structured LLM client for the chosen provider."""
    norm = provider.strip().lower()
    if norm == "gemini":
        return GeminiLLMClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if norm == "openai":
        return OpenAILLMClient(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url or OPENAI_BASE_URL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if norm == "anthropic":
        return AnthropicLLMClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if norm == "ollama":
        return OllamaLLMClient(
            model_name=model_name,
            base_url=base_url or OLLAMA_BASE_URL,
            timeout_seconds=max(timeout_seconds, 120.0),
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported LLM provider: '{provider}'")
