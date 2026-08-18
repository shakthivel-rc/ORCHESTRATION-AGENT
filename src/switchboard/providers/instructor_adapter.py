"""``[instructor]`` adapter — the recommended v0.1 path (plan §4.2, §13 ruling #13).

Design points the plan pins down, and why:

* **Instructor's own retries are disabled** (``max_retries=0``). switchboard owns
  the retry loop (§4.5) because every attempt — count, raw text, error — has to
  land in the :class:`~switchboard.core.audit.AuditRecord`. Letting instructor
  silently re-ask would make per-provider schema-failure rates, an eval metric,
  unmeasurable.
* **``create_with_completion``**, not ``create``: we need the raw completion back
  to recover usage, logprobs and ``raw_text``. ``LLMResult.raw_text`` is kept
  even on success (audit + repair prompt + distillation).
* **Mode auto-picked per provider** (TOOLS_STRICT / ANTHROPIC_TOOLS / GEMINI or
  GenAI structured outputs), resolved by name against ``instructor.Mode`` so a
  version that renamed a member degrades to instructor's own default instead of
  crashing at import.
* **Reasoning forced off** (§4.3): routing on a reasoning mode is a
  misconfiguration (13s vs 0.33s TTFT).
* A schema failure **degrades** to ``LLMResult(parsed=None, ...)`` — it is not a
  ``ProviderError``. Transport failures map onto the §3.8 tree; adapters never
  raise a bare ``ImportError`` (§13 ruling #2).

The ``instructor`` import is lazy (inside :func:`_import_instructor`), so this
module imports cleanly with the SDK absent — which is exactly the state of the
bare-venv CI guard in §2.4.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from switchboard._models import capabilities_for, lookup
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from switchboard.providers.base import LLMRequest

__all__ = ["InstructorAdapter"]

_EXTRA = "instructor"
_PACKAGE = "instructor"

#: How many ``top_logprobs`` alternatives to ask for; the margin signal (§6.2) wants ~10.
_TOP_LOGPROBS = 10

#: Anthropic caps ``cache_control`` breakpoints at 4; we place them on stable segments (§4.6).
_MAX_CACHE_BREAKPOINTS = 4

#: Vendor aliases → the provider token ``instructor.from_provider`` understands.
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "google-genai": "google",
    "googleai": "google",
    "genai": "google",
    "vertexai": "vertexai",
    "vertex_ai": "vertexai",
    "azure": "azure_openai",
    "azure-openai": "azure_openai",
}

#: ``instructor.Mode`` member names to try per provider, best rung first (plan §4.2, §4.3).
_MODE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "openai": ("TOOLS_STRICT", "TOOLS", "JSON"),
    "azure_openai": ("TOOLS_STRICT", "TOOLS", "JSON"),
    "anthropic": ("ANTHROPIC_TOOLS", "ANTHROPIC_JSON"),
    "google": ("GENAI_STRUCTURED_OUTPUTS", "GENAI_TOOLS", "GEMINI_JSON", "GEMINI_TOOLS"),
    "vertexai": ("VERTEXAI_TOOLS", "GEMINI_JSON"),
    "mistral": ("MISTRAL_TOOLS", "JSON"),
    "cohere": ("COHERE_TOOLS", "COHERE_JSON_SCHEMA"),
    "groq": ("TOOLS", "JSON"),
}

#: Provider SDK a given provider token needs, for a precise MissingDependencyError.
_PROVIDER_SDK: dict[str, tuple[str, str]] = {
    # provider -> (importable package, pip extra to suggest alongside [instructor])
    "openai": ("openai", "instructor,openai"),
    "azure_openai": ("openai", "instructor,openai"),
    "anthropic": ("anthropic", "instructor,anthropic"),
    "google": ("google.genai", "instructor,gemini"),
    "vertexai": ("google.genai", "instructor,gemini"),
}

#: Call kwargs we inject opportunistically. If the provider rejects one, it is dropped and the
#: call is retried once — see ``_drop_rejected``. Keeps an adapter we cannot test here from
#: hard-failing on a provider that simply does not know a knob.
_SOFT_KWARGS = frozenset({"logprobs", "top_logprobs", "seed", "reasoning_effort", "generation_config"})

_AUTH_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "UnauthorizedError"})
_RATE_ERRORS = frozenset({"RateLimitError", "TooManyRequests", "ResourceExhausted"})
_TIMEOUT_ERRORS = frozenset(
    {
        "APITimeoutError",
        "Timeout",
        "TimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "APIStatusError",
        "OverloadedError",
    }
)
#: Exception type names instructor raises when the model's output would not validate.
_SCHEMA_ERRORS = frozenset(
    {"InstructorRetryException", "IncompleteOutputException", "ValidationError", "JSONDecodeError"}
)


def _import_instructor() -> Any:
    """Import ``instructor`` or raise :class:`MissingDependencyError` (§13 ruling #2)."""
    try:
        import instructor  # deliberate local import: the whole point is a lazy optional SDK
    except ImportError as exc:  # pragma: no cover - exercised only in a bare venv
        raise MissingDependencyError(_EXTRA, _PACKAGE) from exc
    return instructor


# --------------------------------------------------------------------------- #
# Tolerant accessors. Provider SDKs return pydantic models, dataclasses, plain
# dicts and namespaces interchangeably across versions; every read below goes
# through these so a shape change degrades a field to None instead of raising.
# --------------------------------------------------------------------------- #


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(seq: Any) -> Any:
    """First element of a sequence-ish value, or ``None``. Never raises on a scalar."""
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


def _extract_usage(completion: Any) -> Usage:
    """Recover token accounting across OpenAI / Anthropic / GenAI usage shapes.

    ``Usage.input_tokens`` is *uncached* input, because ``cached_input_tokens``
    is priced separately (§8.4). OpenAI reports ``prompt_tokens`` **inclusive**
    of the cached prefix, Anthropic reports ``input_tokens`` **exclusive** of it,
    so the subtraction is conditional on which shape we matched.
    """
    usage = _get(completion, "usage")
    if usage is None:
        usage = _get(completion, "usage_metadata")
    if usage is None:
        return Usage()

    prompt = _first_int(usage, "prompt_tokens", "prompt_token_count")
    exclusive = _first_int(usage, "input_tokens")  # Anthropic: excludes cache reads
    output = _first_int(usage, "completion_tokens", "output_tokens", "candidates_token_count") or 0

    cached = _first_int(usage, "cache_read_input_tokens", "cached_content_token_count")
    cached_is_inclusive = False
    if cached is None:
        details = _get(usage, "prompt_tokens_details")
        cached = _first_int(details, "cached_tokens")
        cached_is_inclusive = cached is not None
    cached = cached or 0

    base = (prompt - cached if cached_is_inclusive else prompt) if prompt is not None else (exclusive or 0)

    return Usage(
        input_tokens=max(0, base),
        cached_input_tokens=max(0, cached),
        output_tokens=max(0, output),
    )


def _text_from_blocks(blocks: Sequence[Any]) -> str | None:
    """Join Anthropic-style content blocks; a ``tool_use`` block yields its JSON input."""
    parts: list[str] = []
    for block in blocks:
        kind = _get(block, "type")
        if kind == "tool_use" or _get(block, "input") is not None:
            payload = _get(block, "input")
            if payload is not None:
                parts.append(json.dumps(payload, default=str, ensure_ascii=False))
                continue
        text = _get(block, "text")
        if isinstance(text, str) and text:
            parts.append(text)
    joined = "\n".join(parts).strip()
    return joined or None


def _extract_text(completion: Any) -> str | None:
    """Best-effort raw model text across chat-completion, Anthropic and GenAI shapes."""
    choices = _get(completion, "choices")
    if isinstance(choices, Sequence) and not isinstance(choices, str | bytes) and choices:
        message = _get(choices[0], "message", choices[0])
        content = _get(message, "content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, Sequence) and not isinstance(content, str | bytes):
            text = _text_from_blocks(content)
            if text:
                return text
        tool_calls = _get(message, "tool_calls")
        if isinstance(tool_calls, Sequence) and tool_calls:
            arguments = _get(_get(tool_calls[0], "function"), "arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
            if arguments is not None:
                return json.dumps(arguments, default=str, ensure_ascii=False)

    content = _get(completion, "content")  # Anthropic Message
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        text = _text_from_blocks(content)
        if text:
            return text

    candidates = _get(completion, "candidates")  # google-genai GenerateContentResponse
    if isinstance(candidates, Sequence) and candidates:
        parts = _get(_get(candidates[0], "content"), "parts")
        if isinstance(parts, Sequence):
            text = _text_from_blocks(parts)
            if text:
                return text
    direct = _get(completion, "text")
    return direct if isinstance(direct, str) and direct.strip() else None


def _extract_logprobs(completion: Any) -> list[TokenLP] | None:
    """Map an OpenAI-shaped ``logprobs.content`` array onto :class:`TokenLP`.

    Returns ``None`` — never an empty list — when the provider path carries no
    logprobs, because ``None`` is what tells the confidence engine the signal is
    absent rather than zero (§6.1).
    """
    choices = _get(completion, "choices")
    if not (isinstance(choices, Sequence) and choices):
        return None
    content = _get(_get(choices[0], "logprobs"), "content")
    if not isinstance(content, Sequence):
        return None

    out: list[TokenLP] = []
    for item in content:
        token = _get(item, "token")
        logprob = _get(item, "logprob")
        if token is None or not isinstance(logprob, int | float):
            continue
        top: list[tuple[str, float]] = []
        alternatives = _get(item, "top_logprobs")
        if isinstance(alternatives, Sequence):
            for alt in alternatives:
                alt_token = _get(alt, "token")
                alt_logprob = _get(alt, "logprob")
                if alt_token is not None and isinstance(alt_logprob, int | float):
                    top.append((str(alt_token), float(alt_logprob)))
        out.append(TokenLP(token=str(token), logprob=float(logprob), top=top))
    return out or None


def _retry_after(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after", None)
    if value is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if isinstance(headers, Mapping):
            value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _map_provider_error(exc: BaseException) -> ProviderError:
    """Translate an SDK exception onto the §3.8 tree.

    Connection failures and 5xx map to :class:`ProviderTimeout` deliberately: the
    tree exposes exactly one retryable transport class, and the retry driver
    consulting ``retryable`` matters more than the class name reading literally.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if name in _AUTH_ERRORS or status in (401, 403):
        return ProviderAuthError(str(exc))
    if name in _RATE_ERRORS or status == 429:
        return ProviderRateLimit(str(exc), retry_after=_retry_after(exc))
    if name in _TIMEOUT_ERRORS or (status is not None and status >= 500):
        return ProviderTimeout(str(exc))
    return ProviderError(f"{name}: {exc}")


def _is_schema_failure(exc: BaseException) -> bool:
    if isinstance(exc, ValidationError):
        return True
    return any(cls.__name__ in _SCHEMA_ERRORS for cls in type(exc).__mro__)


class InstructorAdapter:
    """``instructor``-backed client implementing both provider protocols (plan §4.2).

    Args:
        model: ``"provider/model"`` (e.g. ``"openai/gpt-5-nano"``). A bare model
            id is accepted when ``_models.lookup`` knows its vendor.
        client: a pre-built **sync** instructor client. Supplying one skips
            ``instructor.from_provider`` entirely — the hook for custom base
            URLs, gateways and tests.
        aclient: a pre-built **async** instructor client. Built on first
            ``acomplete`` from ``from_provider(..., async_client=True)`` when omitted.
        mode: an ``instructor.Mode`` member; auto-picked per provider otherwise.
        capabilities: override the static ``_models`` table (v0.2 replaces the
            table with a runtime canary probe, §4.3).
        enable_cache_control: place Anthropic ``cache_control`` breakpoints on
            the stable segments (§4.6). Turn off for gateways that reject blocks.
        extra_kwargs: merged into every ``create`` call, last — an escape hatch
            that also wins over switchboard's own kwargs.

    Attributes:
        capabilities: drives the §4.3 degradation ladder.
        model: the full ``provider/model`` spec.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        aclient: Any = None,
        mode: Any = None,
        capabilities: ClientCapabilities | None = None,
        enable_cache_control: bool = True,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ConfigError("InstructorAdapter requires a model, e.g. 'openai/gpt-5-nano'.")

        self._instructor = _import_instructor()  # eager: a missing extra must fail at construction
        self.model = model.strip()
        self.provider, self.model_name = self._split(self.model)
        self._enable_cache_control = enable_cache_control
        self._extra_kwargs = dict(extra_kwargs or {})
        self._disabled_kwargs: set[str] = set()

        self._mode = mode if mode is not None else self._pick_mode(self.provider)
        self._client = client if client is not None else self._build_client(async_client=False)
        self._aclient = aclient

        self.capabilities = capabilities or self._resolve_capabilities()

    # ----------------------------------------------------------- construction

    @staticmethod
    def _split(spec: str) -> tuple[str, str]:
        """``"openai/gpt-5-nano"`` → ``("openai", "gpt-5-nano")``, with vendor inference."""
        if "/" in spec:
            head, _, tail = spec.partition("/")
            provider = head.strip().lower()
            return _PROVIDER_ALIASES.get(provider, provider), tail.strip()

        info = lookup(spec)
        if info is None:
            raise ConfigError(
                f"Model {spec!r} has no provider prefix and is not in switchboard's model table, "
                f"so the instructor provider cannot be inferred. Use 'provider/model', "
                f"e.g. 'openai/gpt-5-nano' or 'anthropic/claude-haiku-4-5'."
            )
        provider = _PROVIDER_ALIASES.get(info.provider, info.provider)
        return provider, spec

    def _pick_mode(self, provider: str) -> Any:
        """First ``instructor.Mode`` member that exists for this provider, else instructor's default.

        Resolved by *name* rather than imported directly: instructor has renamed
        Mode members across minor versions, and a rename must not turn into an
        AttributeError at Router construction.
        """
        mode_enum = getattr(self._instructor, "Mode", None)
        if mode_enum is None:
            return None
        for candidate in _MODE_CANDIDATES.get(provider, ("TOOLS", "JSON")):
            member = getattr(mode_enum, candidate, None)
            if member is not None:
                return member
        return None

    def _build_client(self, *, async_client: bool) -> Any:
        from_provider = getattr(self._instructor, "from_provider", None)
        if from_provider is None:
            raise ConfigError(
                "instructor.from_provider is unavailable; switchboard needs instructor>=1.7 "
                "(pip install 'switchboard[instructor]'). Alternatively build the client yourself "
                "and pass it as InstructorAdapter(..., client=...)."
            )
        kwargs: dict[str, Any] = {"async_client": async_client}
        if self._mode is not None:
            kwargs["mode"] = self._mode
        try:
            return from_provider(f"{self.provider}/{self.model_name}", **kwargs)
        except ImportError as exc:
            package, extra = _PROVIDER_SDK.get(self.provider, (self.provider, f"instructor,{self.provider}"))
            raise MissingDependencyError(extra, package) from exc
        except SwitchboardError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"instructor could not build a client for {self.model!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def _aclient_or_build(self) -> Any:
        """Async client, built on first use.

        Deliberately lazy: building both clients at construction doubles the
        connection pools for the (common) case of a router that only ever uses
        one driver. The extras check already happened on the sync path, so
        laziness here does not weaken the §2.4 fail-fast guarantee.
        """
        if self._aclient is None:
            self._aclient = self._build_client(async_client=True)
        return self._aclient

    def _resolve_capabilities(self) -> ClientCapabilities:
        caps = capabilities_for(self.model_name) or self._provider_default_caps()
        if caps.caching == "explicit" and not self._enable_cache_control:
            # We are the only thing that would place the breakpoints; without them there is no
            # cache at all on Anthropic, so claiming "explicit" would mislead the engine.
            caps = caps.model_copy(update={"caching": "none"})
        return caps

    def _provider_default_caps(self) -> ClientCapabilities:
        """Conservative capabilities for a model missing from the static table (§4.3)."""
        if self.provider in ("openai", "azure_openai"):
            return ClientCapabilities(structured="tool_strict", logprobs=False, caching="implicit")
        if self.provider == "anthropic":
            return ClientCapabilities(structured="tool_strict", logprobs=False, caching="explicit")
        if self.provider in ("google", "vertexai"):
            return ClientCapabilities(structured="json_mode", logprobs=False, caching="implicit")
        return ClientCapabilities()

    # --------------------------------------------------------------- request

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Fold segments into one system + one user message, order preserved.

        Merging per role (rather than one message per segment) is what keeps the
        rendered prefix byte-identical across requests: segments A+B are the
        system message and C+D the user message, so the cached prefix is the
        whole system block (§4.6). On Anthropic the system content becomes a
        block list so ``cache_control`` breakpoints can sit after A and after B.
        """
        system = [s for s in request.segments if s.role == "system"]
        user = [s for s in request.segments if s.role != "system"]
        messages: list[dict[str, Any]] = []

        if system:
            wants_blocks = self._enable_cache_control and self.capabilities.caching == "explicit"
            if wants_blocks:
                stable_indexes = [i for i, seg in enumerate(system) if seg.cache == "stable"]
                breakpoints = set(stable_indexes[-_MAX_CACHE_BREAKPOINTS:])
                blocks: list[dict[str, Any]] = []
                for index, segment in enumerate(system):
                    block: dict[str, Any] = {"type": "text", "text": segment.content}
                    if index in breakpoints:
                        block["cache_control"] = {"type": "ephemeral"}
                    blocks.append(block)
                messages.append({"role": "system", "content": blocks})
            else:
                messages.append({"role": "system", "content": "\n\n".join(s.content for s in system)})

        messages.append({"role": "user", "content": "\n\n".join(s.content for s in user)})
        return messages

    def _reasoning_kwargs(self) -> dict[str, Any]:
        """Force reasoning OFF (plan §4.3). Empty when the provider reasons off by default."""
        if not self.capabilities.reasoning_toggle:
            return {}
        if self.provider in ("openai", "azure_openai"):
            return {"reasoning_effort": "minimal"}
        if self.provider in ("google", "vertexai"):
            return {"generation_config": {"thinking_config": {"thinking_budget": 0}}}
        return {}  # Anthropic: thinking is off unless explicitly requested

    def _call_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.want_logprobs and self.capabilities.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = _TOP_LOGPROBS
        kwargs.update(self._reasoning_kwargs())
        for key in self._disabled_kwargs:
            kwargs.pop(key, None)
        kwargs.update(self._extra_kwargs)  # caller's escape hatch wins
        return kwargs

    def _drop_rejected(self, kwargs: dict[str, Any], exc: BaseException) -> bool:
        """Remove soft kwargs the provider complained about; remember the removal.

        Optional knobs (``logprobs``, ``reasoning_effort``, ``generation_config``,
        ``seed``) are unevenly supported across the matrix instructor fronts. If
        the provider names one in its error, dropping it and retrying once is
        strictly better than failing a routing decision over a nice-to-have.
        """
        message = str(exc).lower()
        removed = {key for key in list(kwargs) if key in _SOFT_KWARGS and key.lower() in message}
        if not removed:
            return False
        if "logprobs" in removed:
            # top_logprobs is meaningless — and on OpenAI an outright error — without logprobs.
            removed.add("top_logprobs")
        for key in removed:
            kwargs.pop(key, None)
        self._disabled_kwargs |= removed
        return True

    # ---------------------------------------------------------------- result

    def _to_result(self, parsed: Any, completion: Any, request: LLMRequest) -> LLMResult[Any]:
        raw_text = _extract_text(completion)
        if raw_text is None and parsed is not None:
            raw_text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else str(parsed)
        meta: dict[str, Any] = {
            "adapter": "instructor",
            "provider": self.provider,
            "mode": str(getattr(self._mode, "name", self._mode)),
        }
        finish = _get(_first(_get(completion, "choices")), "finish_reason")
        if finish is not None:
            meta["finish_reason"] = finish
        avg = _get(_first(_get(completion, "candidates")), "avg_logprobs")
        if isinstance(avg, int | float):
            meta["avg_logprobs"] = float(avg)  # Gemini's coarse fallback signal (§6.1)

        if parsed is not None and not isinstance(parsed, request.output_schema):
            # instructor honored a different response_model than we asked for; treat the payload
            # as unvalidated text and let the repair loop own it rather than lying about `parsed`.
            parsed = None

        return LLMResult(
            parsed=parsed,
            raw_text=raw_text or "",
            token_logprobs=_extract_logprobs(completion),
            usage=_extract_usage(completion),
            model_id=str(_get(completion, "model") or self.model),
            attempts=1,
            provider_meta=meta,
        )

    def _schema_failure(self, exc: BaseException) -> LLMResult[Any] | None:
        """Turn an instructor validation failure into a degraded result, never an exception.

        Plan §3.8: unusable model output degrades, it does not raise. Returning
        ``parsed=None`` with the offending text hands control to the §4.5 repair
        loop, which is where the retry budget and the audit trail live.
        """
        if not _is_schema_failure(exc):
            return None
        completion = getattr(exc, "last_completion", None)
        raw_text = _extract_text(completion) if completion is not None else None
        return LLMResult(
            parsed=None,
            raw_text=raw_text if raw_text is not None else str(exc),
            token_logprobs=_extract_logprobs(completion),
            usage=_extract_usage(completion),
            model_id=str(_get(completion, "model") or self.model),
            attempts=1,
            provider_meta={
                "adapter": "instructor",
                "provider": self.provider,
                "schema_error": f"{type(exc).__name__}: {exc}",
            },
        )

    # ------------------------------------------------------------ LLMClient

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """One synchronous routing call (plan §4.1)."""
        messages = self._messages(request)
        kwargs = self._call_kwargs(request)
        create = self._client.chat.completions.create_with_completion

        for attempt in range(2):  # attempt 2 exists only to retry after dropping a rejected kwarg
            try:
                parsed, completion = create(
                    model=self.model_name,
                    response_model=request.output_schema,
                    messages=messages,
                    max_retries=0,  # switchboard owns the retry loop (§4.5)
                    **kwargs,
                )
            except SwitchboardError:
                raise
            except Exception as exc:
                degraded = self._schema_failure(exc)
                if degraded is not None:
                    return degraded
                if attempt == 0 and self._drop_rejected(kwargs, exc):
                    continue
                raise _map_provider_error(exc) from exc
            return self._to_result(parsed, completion, request)

        raise ProviderError("instructor call exhausted its kwarg-compatibility retry")  # unreachable

    # ------------------------------------------------------- AsyncLLMClient

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """One asynchronous routing call — the async twin of :meth:`complete` (plan §2.5)."""
        messages = self._messages(request)
        kwargs = self._call_kwargs(request)
        create = self._aclient_or_build().chat.completions.create_with_completion

        for attempt in range(2):
            try:
                parsed, completion = await create(
                    model=self.model_name,
                    response_model=request.output_schema,
                    messages=messages,
                    max_retries=0,
                    **kwargs,
                )
            except SwitchboardError:
                raise
            except Exception as exc:
                degraded = self._schema_failure(exc)
                if degraded is not None:
                    return degraded
                if attempt == 0 and self._drop_rejected(kwargs, exc):
                    continue
                raise _map_provider_error(exc) from exc
            return self._to_result(parsed, completion, request)

        raise ProviderError("instructor call exhausted its kwarg-compatibility retry")  # unreachable

    def __repr__(self) -> str:
        return f"InstructorAdapter({self.model!r}, mode={getattr(self._mode, 'name', self._mode)!r})"
