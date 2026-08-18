"""``DecisionSink`` and the v0.1 core sinks (plan §8.3), plus the content-mode
gate (plan §7.4).

Two things live here, and both are stdlib + Pydantic only (plan §2.4 — this
module must stay importable in a bare venv):

1. **Sinks.** :class:`DecisionSink` is the Protocol; :class:`InMemorySink`,
   :class:`JSONLSink`, :class:`CallbackSink`, :class:`MultiSink` and
   :class:`NoopSink` are the v0.1 implementations. ``OTLPSink`` and the bounded
   queue + daemon-drain machinery are [v0.2] (plan §8.3, §13 ruling #20).

2. **The content gate.** :func:`apply_content_mode` is the *single* place that
   strips or redacts ``input_text`` / ``args`` / ``rationale`` on an
   :class:`~switchboard.core.audit.AuditRecord`. Both the sink path and the OTel
   span path (``telemetry/otel.py``) go through it, so there is no path where
   raw payload text leaks because a second hook was forgotten (plan §7.4).

**Delivery contract (plan §8.3, normative).** ``route()`` never blocks on and
never fails from sink I/O. Sink exceptions are logged at ``WARNING`` through a
rate limiter (once per sink per 60 s, with a suppression count) and are **never**
re-raised. Enforcement is doubled deliberately:

* every shipped sink guards its own body (:meth:`BaseSink._guard`), so even a
  direct ``sink.emit(record)`` in a router ``finally`` block cannot raise; and
* :func:`safe_emit` / :func:`safe_aemit` / :func:`safe_flush` / :func:`safe_close`
  guard *arbitrary* user-supplied sinks. The router should call these.

v0.1 emission is **synchronous** on purpose. Plan §8.3 defers the non-blocking
bounded queue (10 k entries, ``drop_oldest`` + a visible counter) drained by a
daemon thread, and the ``fsync``-per-record compliance mode, to v0.2 hardening —
JSONL and callback writes are fast enough that the MVP does not pay for the
machinery (§13 ruling #20).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from switchboard.core.audit import canonical_json, sha256_hex
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import TracebackType

    from switchboard.core.audit import AuditRecord

__all__ = [
    "BaseSink",
    "CallbackSink",
    "ContentMode",
    "DecisionSink",
    "InMemorySink",
    "JSONLSink",
    "MultiSink",
    "NoopSink",
    "QueuedSink",
    "RateLimitedLogger",
    "Redactor",
    "apply_content_mode",
    "resolve_sink",
    "safe_aemit",
    "safe_close",
    "safe_emit",
    "safe_flush",
]

logger = logging.getLogger("switchboard.telemetry")

SINK_ERROR_LOG_INTERVAL_S = 60.0
"""Seconds between repeated failure warnings for one sink (plan §8.3: "logged
(rate-limited), never raised"). A misconfigured sink must not turn every
``route()`` into a log line."""


# --------------------------------------------------------------------------- #
# Rate-limited failure reporting.
# --------------------------------------------------------------------------- #


class RateLimitedLogger:
    """Emit at most one message per key per ``interval`` seconds.

    Used for the sink delivery contract (plan §8.3) and reused by
    ``telemetry/otel.py`` for span-emission failures. Suppressed occurrences are
    counted and reported on the next message that gets through, so a silently
    broken sink is still discoverable from the log.

    Thread-safe: the router is documented as safe across threads and tasks
    (plan §2.5), so the limiter's bookkeeping is lock-guarded.
    """

    def __init__(self, log: logging.Logger, interval: float = SINK_ERROR_LOG_INTERVAL_S) -> None:
        self.interval = interval
        self._log = log
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        self._lock = threading.Lock()

    def warn(self, key: str, message: str, exc: BaseException | None = None) -> bool:
        """Log ``message`` at WARNING unless ``key`` fired within ``interval``.

        Returns whether the message was actually emitted (tests assert on this;
        callers should ignore it).
        """
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and (now - last) < self.interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False
            suppressed = self._suppressed.pop(key, 0)
            self._last[key] = now
        if suppressed:
            message = f"{message} [{suppressed} similar failure(s) suppressed in the last {self.interval:g}s]"
        self._log.warning(message, exc_info=exc)
        return True

    def reset(self) -> None:
        """Forget all rate-limiting state (tests, and long-lived process reuse)."""
        with self._lock:
            self._last.clear()
            self._suppressed.clear()


_sink_warner = RateLimitedLogger(logger)


def _sink_key(sink: object) -> str:
    """Stable-enough identity for rate limiting: type + instance id."""
    kind = type(sink)
    return f"{kind.__module__}.{kind.__qualname__}#{id(sink):x}"


def _warn_sink_failure(sink: object, operation: str, exc: BaseException) -> None:
    """Report a swallowed sink failure. Never raises (plan §8.3)."""
    _sink_warner.warn(
        _sink_key(sink),
        f"switchboard: decision sink {type(sink).__name__}.{operation}() failed; "
        f"the record was dropped and routing was unaffected",
        exc,
    )


# --------------------------------------------------------------------------- #
# Content modes (plan §7.4).
# --------------------------------------------------------------------------- #

ContentMode = Literal["none", "redacted", "full"]
"""Tri-state audit content capture (plan §7.4, Appendix A default ``"none"``).

A boolean cannot express "redacted", which is why this is the single knob:

* ``"none"``    - ``inputs_hash``/``args_hash`` only; no payload text anywhere.
* ``"redacted"`` - payload text passed through ``Router(redactor=...)`` first.
* ``"full"``    - payload text verbatim.

Signals, candidates, the chosen route, ``tenant_id``, ``registry_version``,
latency and cost are captured in **every** mode; payload text is the only
conditional part.
"""

Redactor = Callable[[str], str]
"""Text-in/text-out redaction hook as seen by the audit site.

``Router(redactor=...)`` is documented as ``Callable[[str, RequestContext], str]``
(plan §3.3); the router binds the context once per request (e.g.
``functools.partial(redactor, ctx=ctx)`` or a closure) and hands this narrowed
form to :func:`apply_content_mode`, which has no request context of its own.
"""


def _redact_value(value: Any, redactor: Redactor) -> Any:
    """Recursively redact every string *value* inside ``args``.

    Mapping keys are left alone: they are ``args_model`` field names — trusted
    config, not user payload (plan §7.4).
    """
    if isinstance(value, str):
        return redactor(value)
    if isinstance(value, dict):
        return {k: _redact_value(v, redactor) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        redacted = [_redact_value(v, redactor) for v in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


def apply_content_mode(
    record: AuditRecord,
    mode: ContentMode = "none",
    redactor: Redactor | None = None,
) -> AuditRecord:
    """Return ``record`` with its payload fields stripped or redacted (plan §7.4).

    The ONE enforcement point for audit-side content capture: sinks and OTel
    spans both consume the result, so a record can never reach a sink in a
    stronger content mode than the router was configured for.

    Behaviour:

    * ``"none"`` - ``input_text``, ``args`` and ``rationale`` are cleared.
      ``args_hash`` is preserved (and computed here if the loop did not set it)
      so records stay joinable and dedupable without any payload (plan §8.2).
    * ``"redacted"`` - each payload field is passed through ``redactor``; string
      values nested inside ``args`` are redacted recursively.
    * ``"full"`` - payload fields are kept verbatim.

    **Fails closed.** ``mode="redacted"`` with ``redactor=None``, or a redactor
    that raises, degrades to ``"none"`` (and logs once per interval) rather than
    emitting unredacted text — the safe direction for a PII control.

    ``AuditRecord`` is frozen, so this returns a copy; the original is returned
    unchanged when nothing needs to change.
    """
    updates: dict[str, Any] = {}

    if record.args is not None and record.args_hash is None:
        # Always-present-when-args-exist (plan §8.2): compute before we drop args.
        updates["args_hash"] = sha256_hex(canonical_json(record.args))

    effective: ContentMode = mode
    if mode == "redacted" and redactor is None:
        _sink_warner.warn(
            "content_mode:redacted:no_redactor",
            "switchboard: content_mode='redacted' but no redactor is configured; "
            "audit payload text is being dropped instead (failing closed, plan §7.4)",
        )
        effective = "none"

    if effective == "redacted" and redactor is not None:
        try:
            if record.input_text is not None:
                updates["input_text"] = redactor(record.input_text)
            if record.rationale is not None:
                updates["rationale"] = redactor(record.rationale)
            if record.args is not None:
                updates["args"] = _redact_value(record.args, redactor)
        except Exception as exc:  # user code: fail closed, never break the loop
            _sink_warner.warn(
                "content_mode:redactor_failed",
                "switchboard: the configured redactor raised; audit payload text is being "
                "dropped instead (failing closed, plan §7.4)",
                exc,
            )
            effective = "none"
            for key in ("input_text", "rationale", "args"):
                updates.pop(key, None)

    if effective == "none":
        if record.input_text is not None:
            updates["input_text"] = None
        if record.args is not None:
            updates["args"] = None
        if record.rationale is not None:
            updates["rationale"] = None

    return record.model_copy(update=updates) if updates else record


# --------------------------------------------------------------------------- #
# The sink Protocol (plan §8.3, verbatim) and a convenience base class.
# --------------------------------------------------------------------------- #


@runtime_checkable
class DecisionSink(Protocol):
    """Where finalized :class:`~switchboard.core.audit.AuditRecord` s go (plan §8.3).

    Structural, not nominal: any object with these four methods is a sink, so
    users can hand in their own Kafka/Postgres/Datadog writer without importing
    anything from switchboard. :class:`BaseSink` exists only as a convenience.

    Implementations MUST NOT raise from :meth:`emit` — the router's delivery
    contract is that a broken sink degrades to a log line, never to a failed
    ``route()``. The router additionally wraps calls in :func:`safe_emit`.
    """

    def emit(self, record: AuditRecord) -> None:
        """Deliver one record. Must not raise; must not block for long."""
        ...

    async def aemit(self, record: AuditRecord) -> None:
        """Async twin of :meth:`emit`. Default implementation: ``to_thread(emit)``."""
        ...

    def flush(self) -> None:
        """Push anything buffered. Must not raise."""
        ...

    def close(self) -> None:
        """Release resources. Must not raise."""
        ...


class BaseSink:
    """Optional base class supplying the boring half of :class:`DecisionSink`.

    Subclasses override :meth:`emit` (and :meth:`aemit` when they have a genuinely
    async fast path). Defaults:

    * :meth:`aemit` - ``asyncio.to_thread(self.emit, record)``, exactly as plan
      §8.3 specifies, so blocking sinks never stall an event loop;
    * :meth:`flush` / :meth:`close` - no-ops;
    * :meth:`aflush` / :meth:`aclose` - async twins (full sync/async parity,
      plan §2.5). They are *not* on the :class:`DecisionSink` Protocol, which the
      plan pins to four methods; the router uses the sync forms.

    :meth:`emit` defaults to a no-op rather than raising ``NotImplementedError``:
    a half-written sink must degrade to dropping records, never to breaking a
    routing call (plan §8.3).
    """

    def emit(self, record: AuditRecord) -> None:
        """Deliver one record. The base implementation drops it."""

    async def aemit(self, record: AuditRecord) -> None:
        """Deliver one record from async code (default: ``to_thread(emit)``)."""
        await asyncio.to_thread(self.emit, record)

    def flush(self) -> None:
        """Push anything buffered. No-op by default."""

    async def aflush(self) -> None:
        """Async twin of :meth:`flush`."""
        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        """Release resources. No-op by default."""

    async def aclose(self) -> None:
        """Async twin of :meth:`close`."""
        await asyncio.to_thread(self.close)

    # -- delivery contract -------------------------------------------------- #

    @contextlib.contextmanager
    def _guard(self, operation: str) -> Iterator[None]:
        """Swallow and rate-limit-log any failure inside the block (plan §8.3).

        ``BaseException`` subclasses that are not ``Exception``
        (``KeyboardInterrupt``, ``asyncio.CancelledError``, ``SystemExit``) pass
        through untouched — swallowing cancellation would be a much worse bug
        than losing an audit record.
        """
        try:
            yield
        except Exception as exc:
            _warn_sink_failure(self, operation, exc)

    # -- ergonomics --------------------------------------------------------- #

    def __enter__(self) -> BaseSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Guarded call helpers — the router's entry points (plan §8.3 delivery contract).
# --------------------------------------------------------------------------- #


def safe_emit(sink: DecisionSink, record: AuditRecord) -> bool:
    """``sink.emit(record)`` that can never raise. Returns whether it delivered.

    This is what ``router.route()`` calls from its ``finally`` block (plan §2.1,
    §8.3): telemetry is a tap, and a tap that bursts must not take the pipe with
    it.
    """
    try:
        sink.emit(record)
    except Exception as exc:
        _warn_sink_failure(sink, "emit", exc)
        return False
    return True


async def safe_aemit(sink: DecisionSink, record: AuditRecord) -> bool:
    """Async twin of :func:`safe_emit` — used by ``router.aroute()``.

    ``asyncio.CancelledError`` propagates (it is a ``BaseException``): a
    cancelled request must stay cancelled.
    """
    try:
        await sink.aemit(record)
    except Exception as exc:
        _warn_sink_failure(sink, "aemit", exc)
        return False
    return True


def safe_flush(sink: DecisionSink) -> bool:
    """``sink.flush()`` that can never raise. Returns whether it succeeded."""
    try:
        sink.flush()
    except Exception as exc:
        _warn_sink_failure(sink, "flush", exc)
        return False
    return True


def safe_close(sink: DecisionSink) -> bool:
    """``sink.close()`` that can never raise. Returns whether it succeeded."""
    try:
        sink.close()
    except Exception as exc:
        _warn_sink_failure(sink, "close", exc)
        return False
    return True


# --------------------------------------------------------------------------- #
# v0.1 sinks (plan §8.3 table).
# --------------------------------------------------------------------------- #


class NoopSink(BaseSink):
    """Drops every record. The default sink, so ``Router()`` needs no wiring.

    Spans are emitted by ``telemetry/otel.py`` independently of the sink, so a
    ``NoopSink`` router is still fully observable when the ``[otel]`` extra is
    installed (plan §8.3).
    """

    async def aemit(self, record: AuditRecord) -> None:
        """Drop the record without paying for a thread hop."""


class InMemorySink(BaseSink):
    """Bounded ring buffer of the most recent records (plan §8.3).

    Backs the test suite and the eval harness (§9), which want the last N
    decisions without touching a filesystem. Oldest records are evicted once
    ``maxlen`` is reached; :attr:`dropped` counts the evictions so a test can
    tell "nothing was emitted" from "the buffer wrapped".

    Thread-safe. ``maxlen=None`` makes it unbounded — only sane for tests.
    """

    def __init__(self, maxlen: int | None = 1000) -> None:
        self._records: deque[AuditRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def maxlen(self) -> int | None:
        """Ring-buffer capacity; ``None`` when unbounded."""
        return self._records.maxlen

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """Immutable snapshot, oldest first."""
        with self._lock:
            return tuple(self._records)

    @property
    def dropped(self) -> int:
        """How many records the ring buffer evicted."""
        return self._dropped

    @property
    def latest(self) -> AuditRecord | None:
        """The most recently emitted record, or ``None``."""
        with self._lock:
            return self._records[-1] if self._records else None

    def emit(self, record: AuditRecord) -> None:
        """Append to the ring buffer, evicting the oldest record if full."""
        with self._guard("emit"), self._lock:
            if self._records.maxlen is not None and len(self._records) == self._records.maxlen:
                self._dropped += 1
            self._records.append(record)

    async def aemit(self, record: AuditRecord) -> None:
        """Append in-process — a deque append needs no worker thread."""
        self.emit(record)

    def clear(self) -> None:
        """Empty the buffer and reset :attr:`dropped`."""
        with self._lock:
            self._records.clear()
            self._dropped = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self.records)


class JSONLSink(BaseSink):
    """One JSON object per line, appended to ``path`` (plan §8.3).

    This is the distillation input format (§8.5) and the format the eval harness
    replays. ``decision_id`` is a ULID, so a file written by this sink is
    lexicographically time-ordered and range-scannable without an index (§8.2).

    * The file (and its parent directory) is created **lazily**, on the first
      record — constructing a router must not touch the filesystem.
    * Opened in append mode, line-buffered, so each record is visible to other
      readers as soon as it is written. There is **no ``fsync``**: the
      durability-first mode (fsync per record + an unbounded queue) is the
      compliance-strict opt-in at [v0.2] (plan §8.3).
    * Writes are serialized by a lock; emission is **synchronous** in v0.1, again
      per plan §8.3 (the bounded queue drained by a daemon thread is [v0.2]
      hardening, §13 ruling #20). :meth:`aemit` inherits the ``to_thread``
      default so an event loop never blocks on the write.

    After :meth:`close`, further records are dropped with a rate-limited warning
    rather than silently re-opening the file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: Any = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def _handle(self) -> Any:
        """Open the file on first use. Caller must hold the lock."""
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        return self._fh

    def emit(self, record: AuditRecord) -> None:
        """Append ``record`` as one JSON line."""
        with self._guard("emit"):
            line = record.model_dump_json()
            with self._lock:
                if self._closed:
                    _warn_sink_failure(
                        self,
                        "emit",
                        RuntimeError(f"JSONLSink({self.path}) is closed; record dropped"),
                    )
                    return
                self._handle().write(line + "\n")

    def flush(self) -> None:
        """Flush the OS-level buffer (line buffering usually got there first)."""
        with self._guard("flush"), self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        """Flush and close the file. Idempotent."""
        with self._guard("close"), self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
            self._closed = True


class CallbackSink(BaseSink):
    """Wraps a user callable — the BYO escape hatch (plan §8.3).

    Sync and async callables are auto-detected (``functools.partial`` chains and
    objects with an ``async def __call__`` included), mirroring the BYO-callable
    provider adapter's detection (§4.2):

    * async callable + :meth:`aemit` - awaited directly;
    * sync callable + :meth:`aemit` - ``asyncio.to_thread``;
    * sync callable + :meth:`emit` - called inline;
    * async callable + :meth:`emit` - scheduled on the running loop as a
      fire-and-forget task when there is one (a sync ``route()`` inside async
      code), otherwise driven to completion with ``asyncio.run``. Prefer
      ``aroute()`` with an async callback; this path is the compatibility shim.

    The callable's exceptions are swallowed and rate-limit-logged like every
    other sink failure.
    """

    def __init__(self, fn: Callable[[AuditRecord], Any]) -> None:
        self._fn = fn
        self.is_async = _is_async_callable(fn)
        self._pending: set[asyncio.Task[Any]] = set()

    @property
    def fn(self) -> Callable[[AuditRecord], Any]:
        """The wrapped callable."""
        return self._fn

    def emit(self, record: AuditRecord) -> None:
        """Invoke the callback synchronously (see the class docstring for async)."""
        with self._guard("emit"):
            result = self._fn(record)
            if inspect.isawaitable(result):
                self._drive(result)

    async def aemit(self, record: AuditRecord) -> None:
        """Invoke the callback from async code, awaiting it when it is awaitable."""
        with self._guard("aemit"):
            if self.is_async:
                result = self._fn(record)
                if inspect.isawaitable(result):
                    await result
            else:
                await asyncio.to_thread(self._fn, record)

    def _drive(self, awaitable: Any) -> None:
        """Run an awaitable produced from sync code without blocking a loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_await(awaitable))
            return
        task = loop.create_task(_await(awaitable))
        # Keep a strong reference: the loop only holds a weak one, so an
        # unreferenced task can be garbage-collected mid-flight.
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


async def _await(awaitable: Any) -> None:
    """Await ``awaitable``, discarding its result (module-level, picklable-ish)."""
    await awaitable


def _is_async_callable(fn: Any) -> bool:
    """True when calling ``fn`` returns a coroutine (partials and ``__call__`` included)."""
    target = fn
    while isinstance(target, functools.partial):
        target = target.func
    if inspect.iscoroutinefunction(target):
        return True
    # Look the descriptor up on the *type*: an instance with `async def __call__`
    # is an async callable even though the instance itself is not a function.
    try:
        call = type(fn).__call__
    except AttributeError:
        return False
    return inspect.iscoroutinefunction(call)


class MultiSink(BaseSink):
    """Fan out to several sinks with **per-sink error isolation** (plan §8.3).

    One sink raising, hanging on a bad file descriptor, or being closed cannot
    stop the others from receiving the record, and cannot surface in ``route()``.
    Delivery order is the construction order in both the sync and async paths, so
    tests are deterministic.

    The bounded, drop-oldest queue drained by a daemon thread is [v0.2] (plan
    §8.3, §13 ruling #20); v0.1 fans out synchronously.
    """

    def __init__(self, *sinks: DecisionSink) -> None:
        self._sinks: tuple[DecisionSink, ...] = tuple(sinks)

    @property
    def sinks(self) -> tuple[DecisionSink, ...]:
        """The wrapped sinks, in delivery order."""
        return self._sinks

    def emit(self, record: AuditRecord) -> None:
        """Deliver to every sink; failures are isolated and logged."""
        for sink in self._sinks:
            safe_emit(sink, record)

    async def aemit(self, record: AuditRecord) -> None:
        """Async twin of :meth:`emit`, preserving order and isolation."""
        for sink in self._sinks:
            await safe_aemit(sink, record)

    def flush(self) -> None:
        """Flush every sink; failures are isolated and logged."""
        for sink in self._sinks:
            safe_flush(sink)

    def close(self) -> None:
        """Close every sink; failures are isolated and logged."""
        for sink in self._sinks:
            safe_close(sink)

    def __len__(self) -> int:
        return len(self._sinks)

    def __iter__(self) -> Iterator[DecisionSink]:
        return iter(self._sinks)


class QueuedSink(BaseSink):
    """Bounded background sink wrapper with visible drop accounting.

    This is the v0.2 hardening seam implemented without changing the default
    delivery contract: ``emit()`` enqueues quickly, one daemon worker drains to the
    wrapped sink with :func:`safe_emit`, and routing still never fails because a
    sink did. When the queue is full, the default is ``drop_oldest`` so recent
    incidents remain visible; :attr:`dropped` is the operator-visible counter.

    ``flush()`` waits until the queue and current in-flight write are empty, then
    flushes the wrapped sink. ``close()`` stops accepting new records, drains what
    remains, flushes, and closes the wrapped sink. Both are deterministic and
    idempotent.
    """

    def __init__(
        self,
        sink: DecisionSink,
        *,
        maxlen: int = 10_000,
        drop_oldest: bool = True,
        daemon: bool = True,
    ) -> None:
        if maxlen < 1:
            raise ConfigError(f"QueuedSink(maxlen={maxlen}) must be >= 1")
        self._sink = sink
        self._maxlen = maxlen
        self._drop_oldest = drop_oldest
        self._queue: deque[AuditRecord] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._dropped = 0
        self._inflight = 0
        self._worker = threading.Thread(
            target=self._drain,
            name="switchboard-queued-sink",
            daemon=daemon,
        )
        self._worker.start()

    @property
    def sink(self) -> DecisionSink:
        """The wrapped sink."""
        return self._sink

    @property
    def maxlen(self) -> int:
        """Maximum records buffered before dropping starts."""
        return self._maxlen

    @property
    def drop_oldest(self) -> bool:
        """Whether a full queue drops the oldest record; otherwise it drops newest."""
        return self._drop_oldest

    @property
    def dropped(self) -> int:
        """How many records were dropped because the queue was full or closed."""
        with self._cv:
            return self._dropped

    @property
    def queued(self) -> int:
        """Current queued record count."""
        with self._cv:
            return len(self._queue)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has stopped accepting new records."""
        with self._cv:
            return self._closed

    def emit(self, record: AuditRecord) -> None:
        """Enqueue ``record`` or drop according to the configured policy."""
        with self._guard("emit"), self._cv:
            if self._closed:
                self._dropped += 1
                _warn_sink_failure(
                    self,
                    "emit",
                    RuntimeError("QueuedSink is closed; record dropped"),
                )
                return
            if len(self._queue) >= self._maxlen:
                self._dropped += 1
                if self._drop_oldest:
                    self._queue.popleft()
                else:
                    return
            self._queue.append(record)
            self._cv.notify()

    async def aemit(self, record: AuditRecord) -> None:
        """Async enqueue path; no thread hop is needed for a deque append."""
        self.emit(record)

    def flush(self) -> None:
        """Wait for the queue to drain, then flush the wrapped sink."""
        with self._guard("flush"):
            with self._cv:
                while self._queue or self._inflight:
                    self._cv.wait()
            safe_flush(self._sink)

    def close(self) -> None:
        """Stop accepting records, drain the queue, flush and close the wrapped sink."""
        with self._guard("close"):
            with self._cv:
                already_closed = self._closed
                self._closed = True
                self._cv.notify_all()
            if not already_closed and threading.current_thread() is not self._worker:
                self._worker.join()
            safe_flush(self._sink)
            safe_close(self._sink)

    def _drain(self) -> None:
        """Worker loop; all downstream failures are isolated per sink contract."""
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if not self._queue and self._closed:
                    self._cv.notify_all()
                    return
                record = self._queue.popleft()
                self._inflight += 1
            try:
                safe_emit(self._sink, record)
            finally:
                with self._cv:
                    self._inflight -= 1
                    if not self._queue and not self._inflight:
                        self._cv.notify_all()


def resolve_sink(sink: DecisionSink | Iterable[DecisionSink] | None) -> DecisionSink:
    """Normalize the ``Router(sink=...)`` argument to exactly one sink.

    ``None`` becomes a :class:`NoopSink` (the documented default), an iterable of
    sinks becomes a :class:`MultiSink`, and a single sink passes through — so the
    router holds one ``DecisionSink`` and never branches on arity.
    """
    if sink is None:
        return NoopSink()
    if isinstance(sink, DecisionSink):
        return sink
    return MultiSink(*sink)
