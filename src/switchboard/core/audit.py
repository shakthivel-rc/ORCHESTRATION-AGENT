"""``AuditRecord`` — the canonical decision artifact (plan §8.2; §13 ruling #6).

One model serves three jobs simultaneously:

1. the source of OTel GenAI span attributes (§8.1),
2. the compliance audit row (§8.3 sinks),
3. the distillation training example (§8.5).

Sinks, spans, cost rollups and the dataset builder are all *views* over this
record — there is no second audit schema (§13 ruling #6).

Emission is a **tap, not a stage** (plan §2.1): every loop step appends to a
mutable :class:`AuditDraft`, and :meth:`AuditDraft.finalize` freezes it exactly
once after the Decision is final. The router emits in a ``finally`` block so a
raised provider error still produces a record with ``error`` set.

This module also owns the codebase's single hashing utility pair
(:func:`canonical_json` / :func:`sha256_hex`) and the ULID generator
(:func:`new_decision_id`). Everything that hashes anything — route content
hashes, registry versions, view hashes, ``inputs_hash``, ``args_hash``, the
seeded shuffle — goes through here so identical inputs dedupe across processes
(plan §8.2, §7.3, §5.6).

Note on placement: :data:`AbstainReason` and :class:`ConfidenceReport` are
*defined* here and re-exported from ``core/decision.py`` (their documented import
path, plan §3.4). ``AuditRecord`` needs both as field types while
``_DecisionBase`` needs ``AuditRecord``, so the definitions must sit on this side
of the edge to keep the import graph acyclic.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Runtime import, deliberately not deferred to TYPE_CHECKING: `Candidate` is a
# Pydantic field type on AuditRecord, so Pydantic must resolve the real class at
# class-construction time to build the validator. Moving it breaks import.
from switchboard.core.candidates import Candidate  # noqa: TC001
from switchboard.providers.base import Usage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "AbstainReason",
    "AuditDraft",
    "AuditRecord",
    "ConfidenceReport",
    "CostBlock",
    "DecisionKind",
    "DecisionPath",
    "LatencyBlock",
    "canonical_json",
    "new_decision_id",
    "sha256_hex",
]


# --------------------------------------------------------------------------- #
# Canonical hashing (plan §8.2: "SHA-256 over canonical (sorted-key, UTF-8) JSON
# so identical inputs dedupe across processes").
# --------------------------------------------------------------------------- #


def _json_default(obj: Any) -> Any:
    """Deterministic fallback encoder for :func:`canonical_json`.

    Deliberately never falls back to ``repr()``/``str()`` for unknown objects —
    those embed memory addresses and would make content hashes unstable across
    processes, defeating the whole point (§7.3).
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", "replace")
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    return f"<{type(obj).__module__}.{type(obj).__qualname__}>"


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to canonical JSON: sorted keys, no whitespace, UTF-8.

    The ONE canonicalisation used by the whole codebase (plan §8.2). Total by
    construction — unknown types degrade to a stable type name rather than
    raising, so hashing never becomes a failure mode inside the loop.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def sha256_hex(*parts: Any) -> str:
    """Return the SHA-256 hex digest of ``parts``.

    ``str``/``bytes`` parts are hashed as-is (UTF-8 for text); anything else is
    canonicalised via :func:`canonical_json` first. Multiple parts are separated
    by an unambiguous delimiter so ``sha256_hex("ab", "c") != sha256_hex("a", "bc")``.

    The ONE hashing utility in the codebase (plan §7.3, §8.2): route content
    hashes, ``registry_version``, ``view_hash``, ``inputs_hash``, ``args_hash``
    and the shuffle seed all derive from it.
    """
    digest = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            digest.update(b"\x1f")
        if isinstance(part, (bytes, bytearray)):
            digest.update(bytes(part))
        elif isinstance(part, str):
            digest.update(part.encode("utf-8"))
        else:
            digest.update(canonical_json(part).encode("utf-8"))
    return digest.hexdigest()


_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_decision_id() -> str:
    """Generate a ULID: 26 Crockford-base32 chars, lexicographically sortable.

    Plan §8.2 requires ``decision_id`` to be a ULID precisely so JSONL sinks are
    range-scannable by time without an index. Layout is the ULID spec: a 48-bit
    big-endian millisecond timestamp followed by 80 random bits, rendered as 130
    bits of base32 (the top 2 bits are always zero for timestamps this century).

    Pure stdlib — no ``ulid-py`` dependency (plan §2.4).
    """
    timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    value = (timestamp_ms << 80) | secrets.randbits(80)
    out = ["0"] * 26
    for i in range(25, -1, -1):
        out[i] = _CROCKFORD32[value & 0x1F]
        value >>= 5
    return "".join(out)


def _utcnow() -> datetime:
    """Timezone-aware UTC now — audit timestamps are RFC3339 UTC (plan §8.2)."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Shared decision vocabulary (plan §3.4). Re-exported by core/decision.py.
# --------------------------------------------------------------------------- #

AbstainReason = Literal[
    "no_eligible_routes",
    "unparseable_output",
    "invalid_route_reference",
    "invalid_args",
    "low_confidence",
    "provider_error",
    "model_elected",
]
"""The ONE closed abstain vocabulary (plan §3.4; §13 ruling #4).

Four competing vocabularies were collapsed into this literal. The audit record,
distillation filters and eval fixtures all key on it, so adding a member is a
schema-breaking change (§10.1).
"""

DecisionKind = Literal["route", "multi_route", "clarify", "abstain", "plan"]
"""Discriminator values of the :data:`~switchboard.core.decision.Decision` union."""

DecisionPath = Literal["llm", "distilled", "fallback"]
"""How the decision was produced (plan §3.4): the LLM loop, the [v0.5] distilled
fast path, or fallback substitution on a terminal abstain (§6.6)."""


class ConfidenceReport(BaseModel):
    """The ONE confidence model (plan §3.4, §6.3).

    The identical object appears on ``Decision.confidence`` and in
    ``AuditRecord.confidence``, so recalibration and distillation see the raw
    evidence rather than just the fused scalar.

    ``score`` is explicitly **not** a calibrated probability — it is a
    conservative fusion (``min`` of the available signals, §6.3) intended for
    threshold comparison only.
    """

    score: float
    """Fused scalar in [0, 1]. NOT a calibrated probability (documented, §6.3)."""
    method: str
    """e.g. ``"logprobs"``, ``"logprobs+vote:n=3"``, ``"verbalized"``, ``"none"``."""
    p_route: float | None = None
    """Geometric-mean token probability of the committed route value (§6.2). [v0.1]"""
    margin: float | None = None
    """First-divergent-token probability gap; ``None`` when uncomputable. [v0.2]"""
    agreement: float | None = None
    """Self-consistency ``winner_votes / n``. [v0.2]"""
    vote_overturned: bool = False
    """A majority overturned the greedy vote — a strong clarify indicator. [v0.2]"""
    stated: float | None = None
    """Verbalized confidence, raw. Recorded, damped x0.85, never solely trusted."""
    thresholds: dict[str, float] = Field(default_factory=dict)
    """Thresholds as applied, e.g. ``{"clarify": 0.55, "abstain": 0.30}``."""


# --------------------------------------------------------------------------- #
# AuditRecord and its blocks (plan §8.2).
# --------------------------------------------------------------------------- #


class LatencyBlock(BaseModel):
    """Per-stage timings in milliseconds (plan §8.2)."""

    total_ms: float
    shortlist_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_total_ms: float | None = None
    validation_ms: float | None = None


class CostBlock(BaseModel):
    """Resolved cost for one decision (plan §8.2, §8.4). [v0.2]

    ``usd`` is ``None`` when the model is absent from the ``PriceTable`` — cost
    is **never guessed**. ``price_table_version`` is stored so historical records
    can be re-costed when prices move.
    """

    usd: float | None = None
    price_table_version: str = ""
    breakdown: dict[str, float] = Field(default_factory=dict)
    """Keys: ``llm_input``, ``llm_cached``, ``llm_output``, ``embed``."""


class AuditRecord(BaseModel):
    """The canonical, immutable record of one routing decision (plan §8.2).

    ``Decision.audit`` is this type; :meth:`as_otel_attributes` and
    :meth:`as_training_example` are its methods (§13 ruling #6). Build it through
    :class:`AuditDraft` rather than constructing it field-by-field in the loop.

    Content fields (``input_text``, ``args``, ``rationale``) are populated only
    when ``Router(content_mode=...)`` permits, post-redaction; the hash fields
    beside them are always present so records stay joinable in ``"none"`` mode
    (plan §7.4).
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    """Versioned artifact: drift is a breaking change (plan §10.1)."""

    decision_id: str
    """ULID — sortable, so JSONL sinks are range-scannable (see :func:`new_decision_id`)."""
    ts_start: datetime
    ts_end: datetime
    trace_id: str | None = None
    span_id: str | None = None

    tenant_id: str | None = None
    """Raw in the record (the user's own infrastructure and rollup key); hashed on
    the OTel span by default (§7.4)."""
    user_id_hash: str | None = None

    inputs_hash: str
    """sha256 over canonical(query + context metadata) — the dedup key (§8.5)."""
    input_text: str | None = None
    """Only when ``content_mode`` permits; already redacted (§7.4)."""

    registry_version: str
    """12-hex content hash of the registry — hence content hashing is [v0.1]
    (§13 ruling #11)."""
    candidates_total: int = 0
    """Registry size pre-filter."""
    candidates_entitled: int = 0
    """Size after the entitlement filter."""
    shortlist: list[Candidate] = Field(default_factory=list)
    shortlist_skipped: bool = False
    weak_retrieval: bool = False
    shuffle_seed: int | None = None
    """The sha256-derived seed used to order candidates — replayable (§5.6)."""

    kind: DecisionKind
    routes: list[str] = Field(default_factory=list)
    """Committed route name(s); empty for clarify/abstain."""
    args: dict[str, Any] | None = None
    """Only when ``content_mode`` permits."""
    args_hash: str | None = None
    """Always present when args exist, regardless of ``content_mode``."""
    args_schema_fingerprint: str | None = None

    rationale: str | None = None
    """``content_mode``-gated."""
    validation_retries: int = 0

    confidence: ConfidenceReport | None = None
    decision_path: DecisionPath = "llm"
    downgraded_from: str | None = None
    abstain_reason: AbstainReason | None = None
    """Pre-fallback reason, preserved even when a fallback turned the outcome
    into ``kind="route"`` (§3.4, §13 ruling #5)."""

    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    response_id: str | None = None

    usage: Usage = Field(default_factory=Usage)
    """Summed over every LLM call the decision made (votes, retries)."""
    cost: CostBlock | None = None
    latency: LatencyBlock

    outcome: str | None = None
    """Joined later via ``router.record_outcome()``. [v0.5]"""
    error: str | None = None
    """Exception type name when the loop raised; drives ``error.type`` on the span."""
    prev_record_hash: str | None = None
    """Per-tenant tamper-evident hash chain. [v1.0]"""

    # -- projections ------------------------------------------------------- #

    def as_otel_attributes(
        self,
        capture_content: bool = False,
        *,
        hash_tenant: bool = True,
    ) -> dict[str, Any]:
        """Project to OTel span attributes for the parent decision span (plan §8.1).

        Only *standard* ``gen_ai.*`` semconv attributes are emitted; every
        switchboard-specific field rides under the ``switchboard.*`` application
        namespace and is never grafted onto ``gen_ai.*``. Attributes whose values
        are unknown are omitted rather than emitted as ``None``.

        Args:
            capture_content: emit the ``gen_ai.*.messages`` content attributes.
                The caller must have already double-keyed this on
                ``content_mode`` **and** ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
                (§7.4, §8.1); this method does not read the environment.
            hash_tenant: hash ``tenant.id`` on the span. Default True per §7.4 —
                the raw id stays in the record, not the telemetry backend.

        Returns:
            A flat, OTel-safe attribute mapping (str/int/float/bool/sequence values).
        """
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "switchboard.router",
            "gen_ai.output.type": "json",
            "gen_ai.usage.input_tokens": self.usage.input_tokens,
            "gen_ai.usage.output_tokens": self.usage.output_tokens,
            "switchboard.audit.id": self.decision_id,
            "switchboard.decision.kind": self.kind,
            "switchboard.decision.path": self.decision_path,
            "switchboard.registry.version": self.registry_version,
            "switchboard.candidates.count": self.candidates_entitled,
            "switchboard.shortlist.size": len(self.shortlist),
        }
        if self.provider is not None:
            attrs["gen_ai.provider.name"] = self.provider
        if self.request_model is not None:
            attrs["gen_ai.request.model"] = self.request_model
        if self.response_model is not None:
            attrs["gen_ai.response.model"] = self.response_model
        if self.response_id is not None:
            attrs["gen_ai.response.id"] = self.response_id
        if self.error is not None:
            attrs["error.type"] = self.error

        if self.tenant_id is not None:
            attrs["switchboard.tenant.id"] = (
                sha256_hex(self.tenant_id)[:16] if hash_tenant else self.tenant_id
            )
        if self.routes:
            attrs["switchboard.decision.route"] = self.routes[0]
            if len(self.routes) > 1:
                attrs["switchboard.decision.routes"] = tuple(self.routes)
        if self.confidence is not None:
            attrs["switchboard.confidence.score"] = self.confidence.score
        if self.abstain_reason is not None:
            attrs["switchboard.abstain.reason"] = self.abstain_reason
        if self.validation_retries:
            attrs["switchboard.validation.retries"] = self.validation_retries
        if self.cost is not None and self.cost.usd is not None:
            # There is no standard gen_ai cost metric, so cost never fakes one (§8.1).
            attrs["switchboard.cost.usd"] = self.cost.usd

        if capture_content:
            if self.input_text is not None:
                attrs["gen_ai.input.messages"] = canonical_json(
                    [{"role": "user", "content": self.input_text}]
                )
            if self.rationale is not None or self.args is not None:
                body: dict[str, Any] = {"kind": self.kind, "routes": list(self.routes)}
                if self.rationale is not None:
                    body["rationale"] = self.rationale
                if self.args is not None:
                    body["args"] = self.args
                attrs["gen_ai.output.messages"] = canonical_json(
                    [{"role": "assistant", "content": body}]
                )
        return attrs

    def as_training_example(self) -> dict[str, Any]:
        """Project to a distillation training row (plan §8.5).

        "The AuditRecord **is** the training example, projected." Returns an
        empty mapping when ``input_text`` is absent — hashes don't train models,
        so a record captured under ``content_mode="none"`` is simply not
        eligible (§8.5 step 1).

        The remaining eligibility rules (``kind=="route"``, ``score >= 0.8`` or
        ``outcome=="success"``, ``decision_path=="llm"``, no error, dedup on
        ``inputs_hash``, per-route caps) belong to the dataset builder, which
        filters a stream of these rows.
        """
        if self.input_text is None:
            return {}
        label = self.routes[0] if (self.kind == "route" and self.routes) else self.kind
        if self.outcome == "success":
            weight = 1.0
        elif self.confidence is not None:
            weight = float(self.confidence.score)
        else:
            weight = 1.0
        return {
            "text": self.input_text,
            "label": label,
            "registry_version": self.registry_version,
            "tenant": self.tenant_id,
            "weight": weight,
        }


# --------------------------------------------------------------------------- #
# AuditDraft — the mutable accumulator (plan §2.1: "audit is a tap, not a stage").
# --------------------------------------------------------------------------- #


@dataclass
class AuditDraft:
    """Mutable accumulator that every loop stage appends to (plan §2.1, §8.2).

    Deliberately a plain dataclass, not a Pydantic model: it is written to on a
    hot path, has no validation duties, and must never reject a partial state.
    Every field is optional and order-independent — the entitlement filter, the
    shortlister, the provider call, the validator and the policy stage each set
    the slice they own, whenever they run, in any order. Stages that never run
    simply leave their fields at defaults.

    :meth:`finalize` freezes the draft into an :class:`AuditRecord` exactly once
    and caches it, so the router can call it from a ``finally`` block without
    worrying about double-emission or about a second call producing a different
    record.

    Example::

        draft = AuditDraft(registry_version=registry.version, tenant_id=ctx.tenant_id)
        draft.set_inputs(query, {"locale": ctx.locale})
        with draft.timing("shortlist"):
            result = shortlister.shortlist(...)
        draft.add_candidates(result.candidates)
        draft.add_usage(llm_result.usage)
        record = draft.finalize()
    """

    decision_id: str = field(default_factory=new_decision_id)
    ts_start: datetime = field(default_factory=_utcnow)
    ts_end: datetime | None = None
    trace_id: str | None = None
    span_id: str | None = None

    tenant_id: str | None = None
    user_id_hash: str | None = None

    inputs_hash: str | None = None
    input_text: str | None = None

    registry_version: str = ""
    candidates_total: int = 0
    candidates_entitled: int = 0
    shortlist: list[Candidate] = field(default_factory=list)
    shortlist_skipped: bool = False
    weak_retrieval: bool = False
    shuffle_seed: int | None = None

    kind: DecisionKind = "abstain"
    routes: list[str] = field(default_factory=list)
    args: dict[str, Any] | None = None
    args_hash: str | None = None
    args_schema_fingerprint: str | None = None

    rationale: str | None = None
    validation_retries: int = 0

    confidence: ConfidenceReport | None = None
    decision_path: DecisionPath = "llm"
    downgraded_from: str | None = None
    abstain_reason: AbstainReason | None = None

    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    response_id: str | None = None

    usage: Usage = field(default_factory=Usage)
    cost: CostBlock | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)
    """Accumulated stage timings. Recognised keys: ``total``, ``shortlist``,
    ``llm_ttft``, ``llm_total``, ``validation``. Unknown keys are ignored by
    :meth:`finalize` rather than erroring, so stages may record freely."""

    outcome: str | None = None
    error: str | None = None
    prev_record_hash: str | None = None

    _record: AuditRecord | None = field(default=None, repr=False, compare=False)

    # -- accumulation helpers ---------------------------------------------- #

    def set_inputs(
        self,
        query: str,
        context_metadata: dict[str, Any] | None = None,
        *,
        input_text: str | None = None,
    ) -> None:
        """Compute :attr:`inputs_hash` from the query and context metadata.

        The hash is over canonical JSON so identical inputs dedupe across
        processes (plan §8.2) — it is the dedup key the distillation dataset
        builder groups on (§8.5).

        ``input_text`` is set separately and only when ``content_mode`` permits;
        the caller is responsible for having applied the redactor first (§7.4).
        """
        self.inputs_hash = sha256_hex(
            canonical_json({"query": query, "context": context_metadata or {}})
        )
        if input_text is not None:
            self.input_text = input_text

    def add_usage(self, usage: Usage) -> None:
        """Accumulate token usage from one more LLM call (votes, repair retries)."""
        self.usage = self.usage + usage

    def add_candidates(self, candidates: Iterable[Candidate]) -> None:
        """Append shortlist candidates; safe to call more than once (e.g. pinned)."""
        self.shortlist.extend(candidates)

    def add_latency(self, stage: str, ms: float) -> None:
        """Accumulate ``ms`` into ``stage`` — use for stages that repeat.

        ``llm_total`` under self-consistency voting or repair retries is a sum of
        several calls, so this adds rather than replaces.
        """
        self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + ms

    def set_latency(self, stage: str, ms: float) -> None:
        """Set ``stage`` outright — use for point measurements like ``llm_ttft``."""
        self.stage_ms[stage] = ms

    @contextlib.contextmanager
    def timing(self, stage: str) -> Iterator[None]:
        """Time a block and accumulate it into ``stage``, even if it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_latency(stage, (time.perf_counter() - started) * 1000.0)

    def set_error(self, error: BaseException | str | None) -> None:
        """Record the failure that ended the loop.

        Stores the exception *type* name (not the message, which may carry
        payload text) so it can be emitted as the OTel ``error.type`` attribute.
        """
        if error is None:
            self.error = None
        elif isinstance(error, str):
            self.error = error
        else:
            self.error = type(error).__name__

    # -- freeze ------------------------------------------------------------- #

    @property
    def finalized(self) -> bool:
        """Whether :meth:`finalize` has already frozen this draft."""
        return self._record is not None

    def finalize(self, *, ts_end: datetime | None = None) -> AuditRecord:
        """Freeze the draft into an :class:`AuditRecord`, exactly once.

        Idempotent: repeated calls return the first record, so a router that
        finalizes in a ``finally`` block can also finalize on the happy path
        without emitting two different records for one decision.

        ``latency.total_ms`` is derived from the wall clock between
        :attr:`ts_start` and ``ts_end`` unless a ``"total"`` stage timing was
        recorded explicitly.
        """
        if self._record is not None:
            return self._record

        end = ts_end or self.ts_end or _utcnow()
        total = self.stage_ms.get("total")
        if total is None:
            total = max((end - self.ts_start).total_seconds() * 1000.0, 0.0)

        record = AuditRecord(
            decision_id=self.decision_id,
            ts_start=self.ts_start,
            ts_end=end,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tenant_id=self.tenant_id,
            user_id_hash=self.user_id_hash,
            inputs_hash=self.inputs_hash or sha256_hex(""),
            input_text=self.input_text,
            registry_version=self.registry_version,
            candidates_total=self.candidates_total,
            candidates_entitled=self.candidates_entitled,
            shortlist=list(self.shortlist),
            shortlist_skipped=self.shortlist_skipped,
            weak_retrieval=self.weak_retrieval,
            shuffle_seed=self.shuffle_seed,
            kind=self.kind,
            routes=list(self.routes),
            args=self.args,
            args_hash=self.args_hash,
            args_schema_fingerprint=self.args_schema_fingerprint,
            rationale=self.rationale,
            validation_retries=self.validation_retries,
            confidence=self.confidence,
            decision_path=self.decision_path,
            downgraded_from=self.downgraded_from,
            abstain_reason=self.abstain_reason,
            provider=self.provider,
            request_model=self.request_model,
            response_model=self.response_model,
            response_id=self.response_id,
            usage=self.usage,
            cost=self.cost,
            latency=LatencyBlock(
                total_ms=total,
                shortlist_ms=self.stage_ms.get("shortlist"),
                llm_ttft_ms=self.stage_ms.get("llm_ttft"),
                llm_total_ms=self.stage_ms.get("llm_total"),
                validation_ms=self.stage_ms.get("validation"),
            ),
            outcome=self.outcome,
            error=self.error,
            prev_record_hash=self.prev_record_hash,
        )
        self.ts_end = end
        self._record = record
        return record
