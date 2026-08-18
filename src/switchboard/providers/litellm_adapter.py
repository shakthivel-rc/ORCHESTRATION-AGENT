"""``[litellm]`` adapter — broadest provider matrix, weakest guarantees (plan §4.2).

LiteLLM is the v0.1 escape hatch: one spec string reaches ~100 providers, at the
cost of a capability surface that varies per model. So this adapter does not
assume anything — it **asks**:

* ``litellm.supports_response_schema(model)`` and
  ``litellm.get_supported_openai_params(model)`` decide the structure rung and
  whether logprobs are worth requesting, falling back to the static
  ``switchboard._models.capabilities_for()`` table (§4.3) when LiteLLM has no
  opinion, and to all-conservative defaults when neither does. Guessing high
  would flatten the degradation ladder in the wrong direction (report risk #3).
* ``drop_params=True`` is sent by default so a knob one provider lacks is
  dropped by LiteLLM rather than failing the routing decision.
* ``anthropic/*`` targets get ``cache_control`` breakpoints on the stable
  segments (§4.6) — without them there is no prompt cache on that path at all.
* Reasoning is forced **off** (§4.3): ``reasoning_effort="minimal"`` on the
  OpenAI reasoning family, ``"disable"`` on Gemini/Vertex (LiteLLM maps it to
  ``thinkingBudget=0``); Anthropic thinking is already off unless requested.

Schema failures **degrade** to ``LLMResult(parsed=None, ...)`` for the §4.5
repair loop; transport failures map onto the §3.8 tree. The ``litellm`` import
is lazy, so this module imports cleanly with the SDK absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from switchboard._models import CachingMode, StructuredMode, capabilities_for, lookup
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

__all__ = ["LiteLLMAdapter"]

_EXTRA = "litellm"
_PACKAGE = "litellm"

_TOP_LOGPROBS = 10
_MAX_CACHE_BREAKPOINTS = 4

#: LiteLLM route prefixes that speak the Anthropic wire format (explicit cache_control).
_ANTHROPIC_PREFIXES = ("anthropic", "bedrock", "vertex_ai-anthropic")

#: LiteLLM route prefixes served by Google, where reasoning is disabled via reasoning_effort.
_GOOGLE_PREFIXES = ("gemini", "vertex_ai", "vertex_ai-language-models", "google")

_AUTH_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "UnauthorizedError"})
_RATE_ERRORS = frozenset({"RateLimitError", "TooManyRequests", "ResourceExhausted"})
_TIMEOUT_ERRORS = frozenset(
    {
        "Timeout",
        "APITimeoutError",
        "TimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIError",
        "OverloadedError",
    }
)


def _import_litellm() -> Any:
    """Import ``litellm`` or raise :class:`MissingDependencyError` (§13 ruling #2)."""
    try:
        import litellm  # deliberate local import: the whole point is a lazy optional SDK
    except ImportError as exc:  # pragma: no cover - exercised only in a bare venv
        raise MissingDependencyError(_EXTRA, _PACKAGE) from exc
    return litellm


# --------------------------------------------------------------------------- #
# Tolerant accessors — LiteLLM returns its own ModelResponse, a dict, or the
# upstream SDK object depending on provider and version.
# --------------------------------------------------------------------------- #


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
    """Recover usage; ``input_tokens`` is the *uncached* remainder (§8.4).

    OpenAI-shaped ``prompt_tokens`` includes the cached prefix and Anthropic's
    ``input_tokens`` excludes it, so the subtraction is conditional on which key
    supplied the cached count.
    """
    usage = _get(response, "usage")
    if usage is None:
        return Usage()

    prompt = _first_int(usage, "prompt_tokens")
    exclusive = _first_int(usage, "input_tokens")
    output = _first_int(usage, "completion_tokens", "output_tokens") or 0

    cached = _first_int(usage, "cache_read_input_tokens")
    cached_is_inclusive = False
    if cached is None:
        cached = _first_int(_get(usage, "prompt_tokens_details"), "cached_tokens")
        cached_is_inclusive = cached is not None
    cached = cached or 0

    base = (prompt - cached if cached_is_inclusive else prompt) if prompt is not None else (exclusive or 0)

    return Usage(
        input_tokens=max(0, base),
        cached_input_tokens=max(0, cached),
        output_tokens=max(0, output),
    )


def _extract_text(response: Any) -> str | None:
    """Raw assistant text, falling back to a tool call's arguments."""
    message = _get(_first(_get(response, "choices")), "message")
    content = _get(message, "content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        parts = [str(_get(block, "text")) for block in content if _get(block, "text") is not None]
        if parts:
            return "\n".join(parts)
    tool_calls = _get(message, "tool_calls")
    first_call = _first(tool_calls)
    if first_call is not None:
        arguments = _get(_get(first_call, "function"), "arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
        if arguments is not None:
            return json.dumps(arguments, default=str, ensure_ascii=False)
    return None


def _extract_logprobs(response: Any) -> list[TokenLP] | None:
    """Map ``choices[0].logprobs.content`` onto :class:`TokenLP`; ``None`` when absent (§6.1)."""
    content = _get(_get(_first(_get(response, "choices")), "logprobs"), "content")
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
        headers = getattr(exc, "headers", None) or getattr(
            getattr(exc, "response", None), "headers", None
        )
        if isinstance(headers, Mapping):
            value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _map_provider_error(exc: BaseException) -> ProviderError:
    """Translate a LiteLLM exception onto the §3.8 tree.

    LiteLLM re-raises OpenAI-shaped exception classes for every provider, so
    matching on class name plus ``status_code`` covers the whole matrix.
    Connection failures and 5xx map to :class:`ProviderTimeout` because that is
    the tree's single retryable transport class.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if names & _AUTH_ERRORS or status in (401, 403):
        return ProviderAuthError(str(exc))
    if names & _RATE_ERRORS or status == 429:
        return ProviderRateLimit(str(exc), retry_after=_retry_after(exc))
    if names & _TIMEOUT_ERRORS or (status is not None and status >= 500):
        return ProviderTimeout(str(exc))
    return ProviderError(f"{type(exc).__name__}: {exc}")


class LiteLLMAdapter:
    """LiteLLM-backed client implementing both provider protocols (plan §4.2).

    Args:
        model: a LiteLLM model string, e.g. ``"gemini/gemini-2.5-flash-lite"``,
            ``"anthropic/claude-haiku-4-5"``, ``"openai/gpt-5-nano"``.
        capabilities: skip probing and declare capabilities outright (v0.2
            replaces probing with a runtime canary, §4.3).
        enable_cache_control: place Anthropic ``cache_control`` breakpoints on
            stable segments (§4.6).
        drop_params: let LiteLLM drop params a provider does not support instead
            of erroring. On by default — a routing decision should not fail over
            an unsupported ``seed``.
        extra_kwargs: merged into every call, last; wins over switchboard's own.

    Attributes:
        capabilities: drives the §4.3 degradation ladder.
        model: the LiteLLM model string, as given.
    """

    def __init__(
        self,
        model: str,
        *,
        capabilities: ClientCapabilities | None = None,
        enable_cache_control: bool = True,
        drop_params: bool = True,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ConfigError("LiteLLMAdapter requires a model, e.g. 'gemini/gemini-2.5-flash-lite'.")

        self._litellm = _import_litellm()  # eager: a missing extra must fail at construction
        self.model = model.strip()
        self.route_prefix, self.model_name = self._split(self.model)
        self._enable_cache_control = enable_cache_control
        self._drop_params = drop_params
        self._extra_kwargs = dict(extra_kwargs or {})

        self.capabilities = capabilities or self._probe_capabilities()

    # ----------------------------------------------------------- construction

    @staticmethod
    def _split(spec: str) -> tuple[str, str]:
        """``"gemini/gemini-2.5-flash-lite"`` → ``("gemini", "gemini-2.5-flash-lite")``.

        A bare id keeps an empty prefix and the vendor is inferred from the
        static table when one of the behavioural branches needs it.
        """
        if "/" in spec:
            head, _, tail = spec.partition("/")
            return head.strip().lower(), tail.strip()
        return "", spec

    @property
    def _vendor(self) -> str:
        """Normalized vendor: the route prefix, or the static table's provider."""
        if self.route_prefix:
            return self.route_prefix
        info = lookup(self.model_name)
        return info.provider if info is not None else ""

    def _is_anthropic(self) -> bool:
        vendor = self._vendor
        if vendor.startswith(_ANTHROPIC_PREFIXES):
            # bedrock/* is only Anthropic-shaped for Claude models.
            return not vendor.startswith("bedrock") or "claude" in self.model_name.lower()
        return False

    def _is_google(self) -> bool:
        return self._vendor.startswith(_GOOGLE_PREFIXES)

    def _probe_capabilities(self) -> ClientCapabilities:
        """Per-model capabilities from LiteLLM, backed by the static table (plan §4.2, §4.3).

        Precedence is "definite answer wins": LiteLLM's per-model helpers are
        authoritative when they answer, the static table fills the gaps, and an
        unknown model lands on the all-conservative default — which arms the
        repair loop and leaves confidence inert rather than claiming a rung the
        model may not have.
        """
        static = capabilities_for(self.model_name) or capabilities_for(self.model)
        supports_schema = self._probe_bool("supports_response_schema")
        params = self._probe_params()

        # --- structure rung -------------------------------------------------
        structured: StructuredMode
        if supports_schema is True:
            # LiteLLM only reports "this model accepts a JSON schema"; whether that is true
            # constrained decoding is a per-model fact only the static table knows.
            structured = static.structured if static and static.structured != "none" else "json_mode"
        elif params is not None and "response_format" in params:
            structured = "json_mode"
        elif supports_schema is False and params is not None:
            structured = "none"
        else:
            structured = static.structured if static else "none"

        # --- logprobs -------------------------------------------------------
        logprobs = ("logprobs" in params) if params is not None else bool(static and static.logprobs)

        # --- caching --------------------------------------------------------
        caching: CachingMode
        if self._is_anthropic():
            caching = "explicit" if self._enable_cache_control else "none"
        elif static is not None:
            caching = static.caching
        else:
            caching = "none"

        reasoning_toggle = static.reasoning_toggle if static else bool(self._probe_bool("supports_reasoning"))

        return ClientCapabilities(
            structured=structured,
            logprobs=logprobs,
            caching=caching,
            reasoning_toggle=reasoning_toggle,
        )

    def _probe_bool(self, helper: str) -> bool | None:
        """Call a ``litellm.supports_*`` helper; ``None`` when it is absent or unsure."""
        fn = getattr(self._litellm, helper, None)
        if fn is None:
            return None
        try:
            result = fn(model=self.model)
        except Exception:  # an unknown model must not break Router construction
            return None
        return bool(result) if isinstance(result, bool) else None

    def _probe_params(self) -> frozenset[str] | None:
        """``litellm.get_supported_openai_params(model)`` as a set; ``None`` when unavailable."""
        fn = getattr(self._litellm, "get_supported_openai_params", None)
        if fn is None:
            return None
        try:
            params = fn(model=self.model)
        except Exception:  # unknown models raise here on some LiteLLM versions
            return None
        if not isinstance(params, Sequence) or isinstance(params, str | bytes):
            return None
        return frozenset(str(p) for p in params)

    # --------------------------------------------------------------- request

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """One system + one user message, segment order preserved (§4.6).

        Merging per role keeps the cached prefix byte-stable. On Anthropic-shaped
        targets the system content becomes a block list so ``cache_control`` can
        sit after segments A and B — LiteLLM passes those blocks through.
        """
        system = [s for s in request.segments if s.role == "system"]
        user = [s for s in request.segments if s.role != "system"]
        messages: list[dict[str, Any]] = []

        if system:
            if self._enable_cache_control and self.capabilities.caching == "explicit":
                stable = [i for i, seg in enumerate(system) if seg.cache == "stable"]
                breakpoints = set(stable[-_MAX_CACHE_BREAKPOINTS:])
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

    def _response_format(self, request: LLMRequest) -> Any:
        """Highest structure rung this model supports (§4.3 ladder).

        A Pydantic class is handed straight to LiteLLM on the schema rungs: it
        translates the model into each provider's native schema form (OpenAI
        ``json_schema``, Gemini ``responseSchema``, a forced tool on Anthropic),
        which is precisely the per-provider mapping we do not want to reimplement.
        """
        if self.capabilities.structured in ("grammar", "tool_strict"):
            return request.output_schema
        if self.capabilities.structured == "json_mode":
            return {"type": "json_object"}
        return None

    def _reasoning_kwargs(self) -> dict[str, Any]:
        """Force reasoning OFF (plan §4.3)."""
        if not self.capabilities.reasoning_toggle:
            return {}
        if self._is_google():
            return {"reasoning_effort": "disable"}  # LiteLLM maps this to thinkingBudget=0
        if self._is_anthropic():
            return {}  # thinking is off unless explicitly requested
        return {"reasoning_effort": "minimal"}

    def _call_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        response_format = self._response_format(request)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.want_logprobs and self.capabilities.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = _TOP_LOGPROBS
        if self._drop_params:
            kwargs["drop_params"] = True
        kwargs.update(self._reasoning_kwargs())
        kwargs.update(self._extra_kwargs)  # caller's escape hatch wins
        return kwargs

    # ---------------------------------------------------------------- result

    def _to_result(self, response: Any, request: LLMRequest) -> LLMResult[Any]:
        raw_text = _extract_text(response) or ""
        parsed, schema_error = self._parse(raw_text, request)

        meta: dict[str, Any] = {"adapter": "litellm", "vendor": self._vendor}
        finish = _get(_first(_get(response, "choices")), "finish_reason")
        if finish is not None:
            meta["finish_reason"] = finish
        if schema_error is not None:
            meta["schema_error"] = schema_error
        hidden = _get(response, "_hidden_params")
        if isinstance(hidden, Mapping):
            for key in ("custom_llm_provider", "response_cost"):
                if key in hidden:
                    meta[key] = hidden[key]

        return LLMResult(
            parsed=parsed,
            raw_text=raw_text,
            token_logprobs=_extract_logprobs(response),
            usage=_extract_usage(response),
            model_id=str(_get(response, "model") or self.model),
            attempts=1,
            provider_meta=meta,
        )

    @staticmethod
    def _parse(raw_text: str, request: LLMRequest) -> tuple[Any, str | None]:
        """Validate the raw text against the wire schema.

        A failure is **not** an exception: ``parsed=None`` plus the offending
        text is what the §4.5 repair loop consumes, and §3.8 says unusable model
        output degrades rather than raising.
        """
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

    # ------------------------------------------------------------ LLMClient

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """One synchronous routing call (plan §4.1)."""
        try:
            response = self._litellm.completion(**self._call_kwargs(request))
        except SwitchboardError:
            raise
        except Exception as exc:
            raise _map_provider_error(exc) from exc
        return self._to_result(response, request)

    # ------------------------------------------------------- AsyncLLMClient

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """One asynchronous routing call — the async twin of :meth:`complete` (plan §2.5)."""
        try:
            response = await self._litellm.acompletion(**self._call_kwargs(request))
        except SwitchboardError:
            raise
        except Exception as exc:
            raise _map_provider_error(exc) from exc
        return self._to_result(response, request)

    def __repr__(self) -> str:
        return f"LiteLLMAdapter({self.model!r}, structured={self.capabilities.structured!r})"
