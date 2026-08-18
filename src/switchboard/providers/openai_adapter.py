"""Native ``[openai]`` adapter.

This adapter uses the official OpenAI SDK directly instead of going through
Instructor or LiteLLM. The SDK import is lazy, and a prebuilt ``client`` /
``aclient`` can be supplied for tests, custom base URLs or hosted gateways.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from switchboard._models import capabilities_for
from switchboard.engine.schema import strict_compatible_json_schema
from switchboard.errors import (
    ConfigError,
    MissingDependencyError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimit,
    ProviderTimeout,
    SwitchboardError,
)
from switchboard.providers.base import ClientCapabilities, LLMResult, TokenLP, Usage

if TYPE_CHECKING:
    from switchboard.providers.base import LLMRequest

__all__ = ["OpenAIAdapter"]

_EXTRA = "openai"
_PACKAGE = "openai"
_TOP_LOGPROBS = 10
_AUTH_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "UnauthorizedError"})
_RATE_ERRORS = frozenset({"RateLimitError", "TooManyRequests"})
_TIMEOUT_ERRORS = frozenset(
    {
        "APITimeoutError",
        "Timeout",
        "TimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "APIStatusError",
    }
)


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised in bare installs
        raise MissingDependencyError(_EXTRA, _PACKAGE) from exc
    return openai


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(seq: Any) -> Any:
    if isinstance(seq, Sequence) and not isinstance(seq, str | bytes) and seq:
        return seq[0]
    return None


def _first_int(obj: Any, *keys: str) -> int | None:
    for key in keys:
        value = _get(obj, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _extract_usage(response: Any) -> Usage:
    usage = _get(response, "usage")
    if usage is None:
        return Usage()
    prompt = _first_int(usage, "prompt_tokens")
    output = _first_int(usage, "completion_tokens") or 0
    details = _get(usage, "prompt_tokens_details")
    cached = _first_int(details, "cached_tokens") or 0
    return Usage(
        input_tokens=max(0, (prompt or 0) - cached),
        cached_input_tokens=max(0, cached),
        output_tokens=max(0, output),
    )


def _extract_text(response: Any) -> str:
    message = _get(_first(_get(response, "choices")), "message")
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        parts = [_get(part, "text") for part in content]
        return "\n".join(part for part in parts if isinstance(part, str))
    return ""


def _extract_logprobs(response: Any) -> list[TokenLP] | None:
    content = _get(_get(_first(_get(response, "choices")), "logprobs"), "content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return None
    out: list[TokenLP] = []
    for item in content:
        token = _get(item, "token")
        logprob = _get(item, "logprob")
        if not isinstance(token, str) or not isinstance(logprob, int | float):
            continue
        alternatives = _get(item, "top_logprobs")
        top: list[tuple[str, float]] = []
        if isinstance(alternatives, Sequence) and not isinstance(alternatives, str | bytes):
            for alt in alternatives:
                alt_token = _get(alt, "token")
                alt_logprob = _get(alt, "logprob")
                if isinstance(alt_token, str) and isinstance(alt_logprob, int | float):
                    top.append((alt_token, float(alt_logprob)))
        out.append(TokenLP(token=token, logprob=float(logprob), top=top))
    return out or None


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = _get(response, "headers")
    value = _get(headers, "retry-after") or _get(headers, "Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _map_provider_error(exc: BaseException) -> ProviderError:
    name = type(exc).__name__
    if name in _AUTH_ERRORS or getattr(exc, "status_code", None) in (401, 403):
        return ProviderAuthError(str(exc))
    if name in _RATE_ERRORS or getattr(exc, "status_code", None) == 429:
        return ProviderRateLimit(str(exc), retry_after=_retry_after(exc))
    if name in _TIMEOUT_ERRORS or getattr(exc, "status_code", None) in (500, 502, 503, 504):
        return ProviderTimeout(str(exc))
    return ProviderError(f"{name}: {exc}")


class OpenAIAdapter:
    """OpenAI SDK-backed client implementing both provider protocols."""

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        aclient: Any = None,
        capabilities: ClientCapabilities | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ConfigError("OpenAIAdapter requires a model, e.g. 'gpt-5-nano'.")
        self.model = model.strip()
        self.model_name = self.model.rsplit("/", 1)[-1]
        self.provider = "openai"
        sdk = None if (client is not None and aclient is not None) else _import_openai()
        self._openai = sdk
        if client is not None:
            self._client = client
        else:
            if sdk is None:  # pragma: no cover - defensive; branch above imports when client is absent
                sdk = _import_openai()
            self._client = sdk.OpenAI()
        self._aclient = aclient
        self._extra_kwargs = dict(extra_kwargs or {})
        self.capabilities = capabilities or capabilities_for(self.model_name) or ClientCapabilities(
            structured="grammar",
            logprobs=True,
            caching="implicit",
            reasoning_toggle=True,
        )

    def _aclient_or_build(self) -> Any:
        if self._aclient is None:
            sdk = self._openai or _import_openai()
            self._aclient = sdk.AsyncOpenAI()
        return self._aclient

    @staticmethod
    def _messages(request: LLMRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        current_role: str | None = None
        current: list[str] = []
        for segment in request.segments:
            if segment.role != current_role:
                if current_role is not None:
                    messages.append({"role": current_role, "content": "\n\n".join(current)})
                current_role = segment.role
                current = [segment.content]
            else:
                current.append(segment.content)
        if current_role is not None:
            messages.append({"role": current_role, "content": "\n\n".join(current)})
        return messages

    @staticmethod
    def _response_format(request: LLMRequest) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.output_schema.__name__,
                "strict": True,
                "schema": strict_compatible_json_schema(request.output_schema),
            },
        }

    def _call_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            "response_format": self._response_format(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.want_logprobs and self.capabilities.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = _TOP_LOGPROBS
        if self.capabilities.reasoning_toggle:
            kwargs["reasoning_effort"] = "minimal"
        kwargs.update(self._extra_kwargs)
        return kwargs

    @staticmethod
    def _parse(raw_text: str, request: LLMRequest) -> tuple[Any, str | None]:
        if not raw_text.strip():
            return None, "empty response"
        try:
            payload = json.loads(raw_text)
        except (ValueError, TypeError) as exc:
            return None, f"json: {exc}"
        try:
            return request.output_schema.model_validate(payload), None
        except ValidationError as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _to_result(self, response: Any, request: LLMRequest) -> LLMResult[Any]:
        raw_text = _extract_text(response)
        parsed, schema_error = self._parse(raw_text, request)
        meta: dict[str, Any] = {"adapter": "openai", "provider": "openai"}
        response_id = _get(response, "id")
        if response_id is not None:
            meta["response_id"] = response_id
        finish = _get(_first(_get(response, "choices")), "finish_reason")
        if finish is not None:
            meta["finish_reason"] = finish
        if schema_error is not None:
            meta["schema_error"] = schema_error
        return LLMResult(
            parsed=parsed,
            raw_text=raw_text,
            token_logprobs=_extract_logprobs(response),
            usage=_extract_usage(response),
            model_id=str(_get(response, "model") or self.model),
            attempts=1,
            provider_meta=meta,
        )

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """One synchronous routing call."""
        try:
            response = self._client.chat.completions.create(**self._call_kwargs(request))
        except SwitchboardError:
            raise
        except Exception as exc:
            raise _map_provider_error(exc) from exc
        return self._to_result(response, request)

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """One asynchronous routing call."""
        try:
            response = await self._aclient_or_build().chat.completions.create(**self._call_kwargs(request))
        except SwitchboardError:
            raise
        except Exception as exc:
            raise _map_provider_error(exc) from exc
        return self._to_result(response, request)

    def __repr__(self) -> str:
        return f"OpenAIAdapter({self.model!r}, structured={self.capabilities.structured!r})"
