"""Audit emission and OTel spans (plan §8.1, §8.3, §7.4).

Two modules, one boundary:

* ``emitter.py`` — the :class:`DecisionSink` Protocol, the v0.1 sinks
  (:class:`InMemorySink`, :class:`JSONLSink`, :class:`CallbackSink`,
  :class:`MultiSink`, :class:`QueuedSink`, :class:`NoopSink`), the guarded ``safe_*`` call helpers that
  implement the plan §8.3 delivery contract, and :func:`apply_content_mode` — the
  single place audit payload text is stripped or redacted (plan §7.4). Pydantic +
  stdlib only.
* ``otel.py`` — ``gen_ai.*`` span emission and API-only metrics behind the
  ``[otel]`` extra. One of the few modules permitted to import a third-party
  package, and only lazily, inside a function, behind ``try/except ImportError``
  (plan §2.4). With the extra absent it imports fine, reports
  ``available=False``, and every span/metric path becomes a no-op.

Importing this package is always safe in a bare venv: nothing here imports
``opentelemetry`` at module scope, so the CI bare-venv guard (plan §2.4) still
sees an empty deny-list after a routing call.

The remaining v0.2 addition named by plan §8.3 — ``OTLPSink`` — is deliberately
absent (§13 rulings #12 and #20); the queue hardening seam is implemented as
:class:`QueuedSink`, and metrics are implemented in :class:`OTelEmitter`.
"""

from __future__ import annotations

from switchboard.telemetry.emitter import (
    BaseSink,
    CallbackSink,
    ContentMode,
    DecisionSink,
    InMemorySink,
    JSONLSink,
    MultiSink,
    NoopSink,
    QueuedSink,
    RateLimitedLogger,
    Redactor,
    apply_content_mode,
    resolve_sink,
    safe_aemit,
    safe_close,
    safe_emit,
    safe_flush,
)
from switchboard.telemetry.otel import (
    CAPTURE_CONTENT_ENV,
    DecisionSpan,
    LLMSpan,
    OTelEmitter,
    SpanHandle,
    otel_available,
)

__all__ = [
    "CAPTURE_CONTENT_ENV",
    "BaseSink",
    "CallbackSink",
    "ContentMode",
    "DecisionSink",
    "DecisionSpan",
    "InMemorySink",
    "JSONLSink",
    "LLMSpan",
    "MultiSink",
    "NoopSink",
    "OTelEmitter",
    "QueuedSink",
    "RateLimitedLogger",
    "Redactor",
    "SpanHandle",
    "apply_content_mode",
    "otel_available",
    "resolve_sink",
    "safe_aemit",
    "safe_close",
    "safe_emit",
    "safe_flush",
]
