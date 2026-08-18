"""OTel GenAI span emission (plan §8.1; [v0.1] per §13 ruling #12).

The **only** module in the package that touches ``opentelemetry``, and it does so
lazily, inside a function, behind ``try/except ImportError`` (plan §2.4). The
``[otel]`` extra is API-only (``opentelemetry-api``): switchboard emits spans into
whatever tracer provider the host application already configured and never
installs an SDK, an exporter or a global provider of its own.

**Absent-dependency contract.** With ``opentelemetry`` not installed this module
still imports cleanly, :attr:`OTelEmitter.available` is ``False``, and every span
context manager degrades to a no-op that yields an inert handle — semantically
``contextlib.nullcontext``, but with the handle's methods present so callers need
no ``if emitter.available`` branches. Nothing in the routing loop changes shape.

**Span topology (plan §8.1)**::

    invoke_agent switchboard.router            (parent: the decision)
    ├── embeddings {model}                     (optional shortlist)
    ├── chat {model}                           (LLM decision call; xn under self-consistency)
    └── execute_tool {route}                   (opened by execute_tool_span() if the caller executes)

**Attribute discipline.** Only *standard* ``gen_ai.*`` semconv keys are emitted;
everything switchboard-specific rides under the ``switchboard.*`` application
namespace. No ``gen_ai.*`` key is ever invented — in particular there is no
standard GenAI cost attribute, so cost appears only as ``switchboard.cost.usd``
(plan §8.1). The parent span's attributes come verbatim from
:meth:`~switchboard.core.audit.AuditRecord.as_otel_attributes`, so the span and
the audit row can never disagree (§13 ruling #6).

**Content capture is double-keyed (plan §8.1, §7.4).** Message content appears on
a span only when *both*

1. the router's ``content_mode`` is not ``"none"`` (and the record has already
   been through :func:`~switchboard.telemetry.emitter.apply_content_mode`, so the
   redactor ran first), **and**
2. the ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment
   variable is truthy.

Either one alone is not enough. The env var is read at emission time, so it can
be flipped per process without rebuilding the router.

**Errors.** On an exception the span records ``error.type``, is marked
``StatusCode.ERROR``, and is *still ended* — span creation and closing are
bracketed by an ``ExitStack``, matching the plan §2.1 rule that the audit/telemetry
tap runs in a ``finally``. Following the same privacy posture as
``AuditDraft.set_error`` (which stores the exception *type*, never the message,
because messages can carry payload text), the exception's message and stack trace
are attached only when content capture is permitted.

**Metrics.** Metrics use only ``opentelemetry-api`` instruments, just like spans:
Switchboard obtains a meter from the host provider and records token usage,
operation duration, decision counts and cost when a finalized audit record is
available. No SDK, exporter or global provider is installed by this package.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from switchboard.core.audit import canonical_json
from switchboard.telemetry.emitter import RateLimitedLogger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from switchboard.core.audit import AuditRecord
    from switchboard.core.decision import Decision
    from switchboard.providers.base import LLMRequest, LLMResult
    from switchboard.telemetry.emitter import ContentMode

__all__ = [
    "CAPTURE_CONTENT_ENV",
    "INSTRUMENTATION_NAME",
    "DecisionSpan",
    "LLMSpan",
    "OTelEmitter",
    "SpanHandle",
    "otel_available",
]

logger = logging.getLogger("switchboard.telemetry.otel")

_warner = RateLimitedLogger(logger)
"""Span-emission failures are logged at most once per minute per failure site —
telemetry must never turn into a log flood, and never into a raised exception."""

INSTRUMENTATION_NAME = "switchboard"
"""Instrumentation scope name reported to the tracer provider."""

CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
"""The semconv env var that forms the second half of the content double-key (§8.1)."""

_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})

_ATTR_SCALARS = (str, bool, int, float)


# --------------------------------------------------------------------------- #
# Lazy, optional opentelemetry import (plan §2.4: lazy, inside a function).
# --------------------------------------------------------------------------- #


class _OTelAPI(NamedTuple):
    """The handful of ``opentelemetry-api`` objects this module uses."""

    trace: Any
    metrics: Any
    span_kind: Any
    status: Any
    status_code: Any


_API_UNSET: Any = object()
_api_cache: Any = _API_UNSET


def _api() -> _OTelAPI | None:
    """Import ``opentelemetry`` on first use; return ``None`` when it is absent.

    The import is deliberately inside this function: importing
    ``switchboard.telemetry.otel`` must never pull in an optional SDK, and the
    bare-venv CI guard asserts ``opentelemetry`` is absent from ``sys.modules``
    after a routing call (plan §2.4).
    """
    global _api_cache
    if _api_cache is _API_UNSET:
        try:
            from opentelemetry import metrics as _metrics
            from opentelemetry import trace as _trace
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except ImportError:
            _api_cache = None
        else:
            _api_cache = _OTelAPI(_trace, _metrics, SpanKind, Status, StatusCode)
    return _api_cache


def otel_available() -> bool:
    """Whether the ``[otel]`` extra is installed and importable."""
    return _api() is not None


def _reset_api_cache() -> None:
    """Forget the cached import result. Test-only hook."""
    global _api_cache
    _api_cache = _API_UNSET


def _env_capture_content() -> bool:
    """Read the semconv content-capture env var (truthy check, at emission time)."""
    return os.environ.get(CAPTURE_CONTENT_ENV, "").strip().lower() in _TRUTHY


@contextlib.contextmanager
def _never_fails(operation: str) -> Iterator[None]:
    """Swallow and rate-limit-log any telemetry failure (never break ``route()``).

    ``BaseException``s that are not ``Exception`` — cancellation, ``KeyboardInterrupt`` —
    pass through untouched.
    """
    try:
        yield
    except Exception as exc:
        _warner.warn(
            f"otel:{operation}",
            f"switchboard: OTel {operation} failed; the span was skipped and routing was unaffected",
            exc,
        )


def _coerce_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce to OTel-legal attribute values (scalars and homogeneous sequences).

    ``None`` values are dropped rather than emitted — an absent attribute is the
    semconv way to say "unknown".
    """
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, _ATTR_SCALARS):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = tuple(v if isinstance(v, _ATTR_SCALARS) else str(v) for v in value if v is not None)
        else:
            out[key] = str(value)
    return out


# --------------------------------------------------------------------------- #
# Span handles. Every method is safe to call when the underlying span is absent.
# --------------------------------------------------------------------------- #


class SpanHandle:
    """A thin, always-safe wrapper around one OTel span.

    ``_span`` is ``None`` whenever the ``[otel]`` extra is missing, the emitter is
    disabled, or span creation failed — in which case every method here is a
    no-op and :attr:`trace_id` / :attr:`span_id` are ``None``. Callers therefore
    never branch on availability.
    """

    __slots__ = ("_emitter", "_errored", "_span")

    def __init__(self, span: Any, emitter: OTelEmitter) -> None:
        self._span = span
        self._emitter = emitter
        self._errored = False

    @property
    def span(self) -> Any:
        """The raw OTel span, or ``None``. Escape hatch for host instrumentation."""
        return self._span

    @property
    def recording(self) -> bool:
        """Whether a live, recording span backs this handle."""
        if self._span is None:
            return False
        try:
            return bool(self._span.is_recording())
        except Exception:
            return False

    @property
    def trace_id(self) -> str | None:
        """32-hex trace id for ``AuditRecord.trace_id`` (plan §8.2), or ``None``."""
        return self._context_id("trace_id", 32)

    @property
    def span_id(self) -> str | None:
        """16-hex span id for ``AuditRecord.span_id`` (plan §8.2), or ``None``."""
        return self._context_id("span_id", 16)

    def _context_id(self, attribute: str, width: int) -> str | None:
        if self._span is None:
            return None
        try:
            ctx = self._span.get_span_context()
            if ctx is None or not ctx.is_valid:
                return None
            return format(getattr(ctx, attribute), f"0{width}x")
        except Exception:
            return None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one attribute, coercing the value and dropping ``None``."""
        self.set_attributes({key: value})

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Set many attributes at once. Never raises."""
        if self._span is None:
            return
        with _never_fails("set_attributes"):
            coerced = _coerce_attributes(attributes)
            if coerced:
                self._span.set_attributes(coerced)

    def record_error(self, exc: BaseException) -> None:
        """Mark the span failed: ``error.type`` + ``StatusCode.ERROR`` (plan §8.1).

        The exception's message and stack trace are attached (via
        ``record_exception``) only when content capture is permitted, because an
        exception message can carry payload text — the same posture
        ``AuditDraft.set_error`` takes by storing the type name alone (§7.4).
        """
        if self._span is None or self._errored:
            return
        self._errored = True
        with _never_fails("record_error"):
            api = _api()
            error_type = type(exc).__qualname__
            self._span.set_attribute("error.type", error_type)
            if api is not None:
                self._span.set_status(api.status(api.status_code.ERROR, error_type))
            if self._emitter.capture_content:
                self._span.record_exception(exc)

    def set_error_type(self, error_type: str) -> None:
        """Mark the span failed from a recorded error *name* (no exception object)."""
        if self._span is None or self._errored:
            return
        self._errored = True
        with _never_fails("set_error_type"):
            api = _api()
            self._span.set_attribute("error.type", error_type)
            if api is not None:
                self._span.set_status(api.status(api.status_code.ERROR, error_type))


class DecisionSpan(SpanHandle):
    """The parent ``invoke_agent switchboard.router`` span (plan §8.1)."""

    __slots__ = ()

    def set_record(self, record: AuditRecord) -> None:
        """Attach the finalized audit record's attributes to the span.

        Pass the record **after**
        :func:`~switchboard.telemetry.emitter.apply_content_mode` has run: this
        method applies the second half of the content double-key (the env var),
        but the first half (``content_mode`` + redactor) belongs to the record.

        Also propagates ``error.type`` / ``StatusCode.ERROR`` when the loop ended
        in a recorded error but no exception reached this span (e.g. a provider
        failure that degraded to ``abstain``).
        """
        if self._span is None:
            return
        with _never_fails("set_record"):
            self._span.set_attributes(
                _coerce_attributes(
                    record.as_otel_attributes(
                        capture_content=self._emitter.capture_content,
                        hash_tenant=self._emitter.hash_tenant,
                    )
                )
            )
        if record.error is not None:
            self.set_error_type(record.error)


class LLMSpan(SpanHandle):
    """A child ``chat {model}`` / ``embeddings {model}`` span (plan §8.1)."""

    __slots__ = ()

    def set_result(self, result: LLMResult[Any]) -> None:
        """Attach response attributes from an :class:`~switchboard.providers.base.LLMResult`.

        ``gen_ai.response.id`` and ``gen_ai.response.finish_reasons`` are read
        from ``provider_meta`` (adapters put them there; nothing is invented when
        they are absent).
        """
        if self._span is None:
            return
        attrs: dict[str, Any] = {
            "gen_ai.usage.input_tokens": result.usage.input_tokens,
            "gen_ai.usage.output_tokens": result.usage.output_tokens,
        }
        if result.model_id:
            attrs["gen_ai.response.model"] = result.model_id
        if result.usage.cached_input_tokens:
            attrs["switchboard.usage.cached_input_tokens"] = result.usage.cached_input_tokens
        if result.attempts > 1:
            attrs["switchboard.llm.attempts"] = result.attempts

        meta = result.provider_meta or {}
        response_id = meta.get("response_id") or meta.get("id")
        if response_id is not None:
            attrs["gen_ai.response.id"] = str(response_id)
        finish = meta.get("finish_reasons", meta.get("finish_reason"))
        if isinstance(finish, str):
            attrs["gen_ai.response.finish_reasons"] = (finish,)
        elif isinstance(finish, (list, tuple)) and finish:
            attrs["gen_ai.response.finish_reasons"] = tuple(str(f) for f in finish)

        if self._emitter.capture_content and result.raw_text:
            attrs["gen_ai.output.messages"] = canonical_json(
                [{"role": "assistant", "content": result.raw_text}]
            )
        self.set_attributes(attrs)

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        cached_input_tokens: int = 0,
    ) -> None:
        """Attach token counts directly — the embeddings path has no ``LLMResult``."""
        attrs: dict[str, Any] = {
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        }
        if cached_input_tokens:
            attrs["switchboard.usage.cached_input_tokens"] = cached_input_tokens
        self.set_attributes(attrs)


_S = TypeVar("_S", bound=SpanHandle)


# --------------------------------------------------------------------------- #
# The emitter.
# --------------------------------------------------------------------------- #


class OTelEmitter:
    """Opens the plan §8.1 span topology; no-ops when ``[otel]`` is not installed.

    One instance lives on the ``Router`` and is shared across threads and tasks
    (it holds no per-request state, matching plan §2.5's thread-safety rule).
    Every span method has an ``a``-prefixed async twin; the twins delegate to the
    sync context managers because OTel's current-span propagation is contextvar
    based and therefore already task-correct.

    Usage in the router's decision loop::

        with emitter.decision_span(conversation_id=ctx.conversation_id) as span:
            draft.trace_id, draft.span_id = span.trace_id, span.span_id
            try:
                with emitter.chat_span(model, provider=provider, request=req) as call:
                    result = client.complete(req)
                    call.set_result(result)
                ...
            finally:
                record = apply_content_mode(draft.finalize(), content_mode, redactor)
                span.set_record(record)
                safe_emit(sink, record)

    Args:
        content_mode: the router's ``content_mode`` (plan §7.4) — first half of the
            content double-key. ``"none"`` (the default) means no content attribute
            is ever emitted, whatever the environment says.
        hash_tenant: hash ``switchboard.tenant.id`` on spans. Default ``True`` per
            §7.4: raw tenant ids stay in the audit record, not in the tracing backend.
        tracer_provider: an explicit provider; by default the globally configured
            one is used (switchboard never installs a provider).
        enabled: hard off switch — ``False`` makes :attr:`available` ``False`` even
            with the extra installed.
        agent_name / agent_id: ``gen_ai.agent.name`` / ``gen_ai.agent.id``. The
            name is also the parent span's suffix (``invoke_agent {agent_name}``).
    """

    available: bool = False
    """``True`` only on instances that found a usable ``opentelemetry`` install.

    Declared on the class as ``False`` so the "extra absent" answer is correct
    even before an instance exists; ``__init__`` shadows it per instance.
    """

    def __init__(
        self,
        *,
        content_mode: ContentMode = "none",
        hash_tenant: bool = True,
        tracer_provider: Any | None = None,
        enabled: bool = True,
        agent_name: str = "switchboard.router",
        agent_id: str | None = None,
    ) -> None:
        self.content_mode: ContentMode = content_mode
        self.hash_tenant = hash_tenant
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.enabled = enabled
        self._tracer_provider = tracer_provider
        self._tracer: Any = None
        self._meter: Any = None
        self._duration_histogram: Any = None
        self._token_histogram: Any = None
        self._decision_counter: Any = None
        self._cost_histogram: Any = None
        self.available = enabled and otel_available()

    def __repr__(self) -> str:
        return (
            f"OTelEmitter(available={self.available}, content_mode={self.content_mode!r}, "
            f"agent_name={self.agent_name!r})"
        )

    # -- gating ------------------------------------------------------------- #

    @property
    def capture_content(self) -> bool:
        """The content double-key (plan §8.1): ``content_mode`` **and** the env var.

        Evaluated per call so ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
        can be flipped without rebuilding the router.
        """
        return self.content_mode != "none" and _env_capture_content()

    def _get_tracer(self) -> Any:
        """Fetch (and memoize) the tracer, or ``None`` when unavailable."""
        if not self.available:
            return None
        if self._tracer is None:
            api = _api()
            if api is None:
                return None
            with _never_fails("get_tracer"):
                provider = self._tracer_provider
                if provider is None:
                    self._tracer = api.trace.get_tracer(INSTRUMENTATION_NAME)
                else:
                    self._tracer = provider.get_tracer(INSTRUMENTATION_NAME)
        return self._tracer

    def _get_meter(self) -> Any:
        """Fetch (and memoize) the meter, or ``None`` when unavailable."""
        if not self.available:
            return None
        if self._meter is None:
            api = _api()
            if api is None:
                return None
            with _never_fails("get_meter"):
                provider = self._tracer_provider
                if provider is not None and hasattr(provider, "get_meter"):
                    self._meter = provider.get_meter(INSTRUMENTATION_NAME)
                else:
                    self._meter = api.metrics.get_meter(INSTRUMENTATION_NAME)
        return self._meter

    def _ensure_instruments(self) -> bool:
        """Create metric instruments once. Returns whether metrics are usable."""
        meter = self._get_meter()
        if meter is None:
            return False
        if self._duration_histogram is not None:
            return True
        with _never_fails("create_metrics"):
            self._duration_histogram = meter.create_histogram(
                "gen_ai.client.operation.duration",
                unit="s",
                description="Duration of a switchboard routing operation.",
            )
            self._token_histogram = meter.create_histogram(
                "gen_ai.client.token.usage",
                unit="{token}",
                description="Input/output token usage for switchboard routing calls.",
            )
            self._decision_counter = meter.create_counter(
                "switchboard.decision.count",
                unit="{decision}",
                description="Number of switchboard decisions.",
            )
            self._cost_histogram = meter.create_histogram(
                "switchboard.cost.usd",
                unit="USD",
                description="Resolved switchboard decision cost in USD.",
            )
        return self._duration_histogram is not None

    def record_metrics(self, record: AuditRecord) -> None:
        """Record metrics from a finalized audit record."""
        if not self._ensure_instruments():
            return
        attrs = _coerce_attributes(
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": self.agent_name,
                "switchboard.decision.kind": record.kind,
                "switchboard.decision.path": record.decision_path,
                "switchboard.registry.version": record.registry_version,
                "gen_ai.provider.name": record.provider,
                "gen_ai.request.model": record.request_model,
            }
        )
        with _never_fails("record_metrics"):
            self._decision_counter.add(1, attrs)
            self._duration_histogram.record(record.latency.total_ms / 1000.0, attrs)
            if record.usage.input_tokens:
                self._token_histogram.record(
                    record.usage.input_tokens,
                    {**attrs, "gen_ai.token.type": "input"},
                )
            if record.usage.cached_input_tokens:
                self._token_histogram.record(
                    record.usage.cached_input_tokens,
                    {**attrs, "gen_ai.token.type": "input_cached"},
                )
            if record.usage.output_tokens:
                self._token_histogram.record(
                    record.usage.output_tokens,
                    {**attrs, "gen_ai.token.type": "output"},
                )
            if record.cost is not None and record.cost.usd is not None:
                self._cost_histogram.record(record.cost.usd, attrs)

    # -- span plumbing ------------------------------------------------------ #

    @contextlib.contextmanager
    def _open(
        self,
        handle_cls: type[_S],
        name: str,
        kind_name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[_S]:
        """Open one span as the current span and always end it (plan §2.1 finally rule).

        The ``ExitStack`` guarantees the span is closed on the raise path too, and
        a failure to *create* the span degrades to an inert handle instead of
        propagating — telemetry never decides whether routing succeeds.
        """
        with contextlib.ExitStack() as stack:
            raw: Any = None
            tracer = self._get_tracer()
            api = _api()
            if tracer is not None and api is not None:
                try:
                    raw = stack.enter_context(
                        tracer.start_as_current_span(
                            name,
                            kind=getattr(api.span_kind, kind_name),
                            record_exception=False,
                            set_status_on_exception=False,
                        )
                    )
                except Exception as exc:
                    _warner.warn(
                        "otel:start_span",
                        f"switchboard: could not start OTel span {name!r}; continuing without it",
                        exc,
                    )
                    raw = None
            handle = handle_cls(raw, self)
            if attributes:
                handle.set_attributes(attributes)
            try:
                yield handle
            except BaseException as exc:
                handle.record_error(exc)
                raise

    # -- parent span -------------------------------------------------------- #

    @contextlib.contextmanager
    def decision_span(
        self,
        *,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[DecisionSpan]:
        """Open the parent ``invoke_agent switchboard.router`` span (plan §8.1).

        Yields a :class:`DecisionSpan`: read :attr:`~SpanHandle.trace_id` /
        :attr:`~SpanHandle.span_id` off it into the ``AuditDraft`` at the start,
        and hand it the finalized record via :meth:`DecisionSpan.set_record` at
        the end (in the router's ``finally``, so an exception still produces a
        fully attributed span).
        """
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": self.agent_name,
        }
        resolved_agent_id = agent_id or self.agent_id
        if resolved_agent_id is not None:
            attrs["gen_ai.agent.id"] = resolved_agent_id
        if conversation_id is not None:
            attrs["gen_ai.conversation.id"] = conversation_id
        if attributes:
            attrs.update(attributes)
        with self._open(DecisionSpan, f"invoke_agent {self.agent_name}", "INTERNAL", attrs) as span:
            yield span

    @contextlib.asynccontextmanager
    async def adecision_span(
        self,
        *,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[DecisionSpan]:
        """Async twin of :meth:`decision_span` (plan §2.5 sync/async parity).

        OTel's current-span propagation is contextvar-based, so the sync context
        manager is already correct inside a task; this twin exists so ``aroute()``
        reads symmetrically and so the async API is complete.
        """
        with self.decision_span(
            conversation_id=conversation_id, agent_id=agent_id, attributes=attributes
        ) as span:
            yield span

    # -- child spans -------------------------------------------------------- #

    @contextlib.contextmanager
    def chat_span(
        self,
        model: str,
        *,
        provider: str | None = None,
        request: LLMRequest | None = None,
        choice_count: int = 1,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[LLMSpan]:
        """Open a child ``chat {model}`` span for one LLM decision call (plan §8.1).

        Opened once per provider call, so self-consistency voting and repair
        retries each get their own span under the same parent.
        ``gen_ai.request.temperature/.max_tokens/.seed`` are read off ``request``;
        ``gen_ai.request.choice.count`` is emitted only when ``choice_count > 1``
        (the vote fan-out, [v0.2]).

        Prompt content is emitted only under the content double-key. The segments
        are already redacted at assembly — the redactor's enforcement site 2 in
        plan §7.4 — so no second redaction pass happens here.
        """
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.output.type": "json",
        }
        if provider is not None:
            attrs["gen_ai.provider.name"] = provider
        if request is not None:
            attrs["gen_ai.request.temperature"] = request.temperature
            attrs["gen_ai.request.max_tokens"] = request.max_tokens
            if request.seed is not None:
                attrs["gen_ai.request.seed"] = request.seed
            if self.capture_content:
                attrs["gen_ai.input.messages"] = canonical_json(
                    [{"role": s.role, "content": s.content} for s in request.segments]
                )
        if choice_count > 1:
            attrs["gen_ai.request.choice.count"] = choice_count
        if attributes:
            attrs.update(attributes)
        with self._open(LLMSpan, f"chat {model}", "CLIENT", attrs) as span:
            yield span

    @contextlib.asynccontextmanager
    async def achat_span(
        self,
        model: str,
        *,
        provider: str | None = None,
        request: LLMRequest | None = None,
        choice_count: int = 1,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMSpan]:
        """Async twin of :meth:`chat_span`."""
        with self.chat_span(
            model,
            provider=provider,
            request=request,
            choice_count=choice_count,
            attributes=attributes,
        ) as span:
            yield span

    @contextlib.contextmanager
    def embeddings_span(
        self,
        model: str,
        *,
        provider: str | None = None,
        input_count: int | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[LLMSpan]:
        """Open a child ``embeddings {model}`` span for the shortlist pass (plan §8.1).

        Only the embedding-backed shortlisters open this; the pure-Python BM25
        backend does no network I/O and therefore emits no child span.
        """
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "embeddings",
            "gen_ai.request.model": model,
        }
        if provider is not None:
            attrs["gen_ai.provider.name"] = provider
        if input_count is not None:
            # No semconv key for embedded-input count: application namespace only.
            attrs["switchboard.embeddings.input_count"] = input_count
        if attributes:
            attrs.update(attributes)
        with self._open(LLMSpan, f"embeddings {model}", "CLIENT", attrs) as span:
            yield span

    @contextlib.asynccontextmanager
    async def aembeddings_span(
        self,
        model: str,
        *,
        provider: str | None = None,
        input_count: int | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMSpan]:
        """Async twin of :meth:`embeddings_span`."""
        with self.embeddings_span(
            model, provider=provider, input_count=input_count, attributes=attributes
        ) as span:
            yield span

    # -- execution span (opened by the caller, not by route()) -------------- #

    @contextlib.contextmanager
    def execute_tool_span(
        self,
        decision: Decision,
        *,
        route: str | None = None,
        description: str | None = None,
        call_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        """Open an ``execute_tool {route}`` span around the caller's execution (plan §8.1).

        switchboard **decides, never executes** (§7.4), so this span is opened by
        the application that runs the chosen route — the ``decision.otel_execute_span()``
        hook of plan §8.1, living on the emitter so ``core/`` keeps no telemetry
        dependency. Wrapping execution here links the side effect to the decision
        that caused it: ``gen_ai.tool.call.id`` defaults to the decision's ULID, so
        span, audit row and training example all share one key.

        Args:
            decision: the decision being executed; the route name comes from it.
            route: override for multi-route decisions, where one span per invoked
                route is the right shape.
            description: the ``Route.description`` — ``gen_ai.tool.description``.
                Not carried on the Decision, so callers pass it from the registry.
            call_id: override for ``gen_ai.tool.call.id``.
        """
        audit = getattr(decision, "audit", None)
        decision_id = getattr(audit, "decision_id", None)
        name = route or getattr(decision, "route", None)
        if name is None:
            routes = getattr(audit, "routes", ()) or ()
            name = routes[0] if routes else "unknown"

        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
        }
        if description is not None:
            attrs["gen_ai.tool.description"] = description
        resolved_call_id = call_id or decision_id
        if resolved_call_id is not None:
            attrs["gen_ai.tool.call.id"] = resolved_call_id
        if decision_id is not None:
            attrs["switchboard.audit.id"] = decision_id
        if attributes:
            attrs.update(attributes)
        with self._open(SpanHandle, f"execute_tool {name}", "INTERNAL", attrs) as span:
            yield span

    @contextlib.asynccontextmanager
    async def aexecute_tool_span(
        self,
        decision: Decision,
        *,
        route: str | None = None,
        description: str | None = None,
        call_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[SpanHandle]:
        """Async twin of :meth:`execute_tool_span`."""
        with self.execute_tool_span(
            decision,
            route=route,
            description=description,
            call_id=call_id,
            attributes=attributes,
        ) as span:
            yield span
