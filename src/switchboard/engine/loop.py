"""The canonical decision loop — ONE pipeline, two drivers (plan §2.1, §2.5).

Plan §2.5 is the whole design of this module: *"All steps in ``engine/loop.py``
are pure ``(LoopState) -> LoopState`` functions; the only I/O points are the
provider call and telemetry flush. ``route()`` and ``aroute()`` are ~30-line
drivers over identical steps — parity is structural."*

So every stage of §2.1's loop lives here as a step function, and
:class:`~switchboard.router.Router` is a thin driver that calls them in order.
The sync and async drivers differ **only** at the provider call; nothing else in
the pipeline is duplicated, which is why sync/async parity cannot silently rot.

Pipeline order (plan §2.1)::

    snapshot_registry
      -> filter_entitlements          -- empty set: abstain(no_eligible_routes), NO LLM call
      -> run_shortlist                -- I/O-free for the default BM25 backend
      -> order_and_build_prompt       -- seeded shuffle; query LAST
      -> build_schema                 -- rationale-first wire schema
      -> run_decision_sync / _async   -- THE I/O POINT: provider call + §4.5 repair loop
           -> validate_output         -- (called per attempt, from inside the repair loop)
      -> score_confidence
      -> resolve_policy               -- thresholds (downgrade-only), degradation table
      -> apply_fallback
      -> finalize_audit               -- content gate + freeze; the router emits in a `finally`

**On "pure".** Plan §2.1 specifies a *mutable* ``LoopState`` that every step
appends to ("the audit emitter is a tap, not a stage"), so the steps mutate the
state they are given and return it. "Pure" here means what §2.5 needs it to
mean: **no I/O, no hidden global state, no dependence on the driver**. Every step
except ``run_decision_*`` is deterministic given its input state, which is what
makes the two drivers interchangeable and the loop replayable from an audit
record.

**Raise vs degrade (plan §3.8, §13 ruling #2).** Schema and validation exhaustion
*never* raise — they degrade to ``abstain(unparseable_output)`` /
``abstain(invalid_route_reference)`` / ``clarify``, uniformly, whether or not a
fallback is configured. Provider transport failure raises by default after the
retry budget, and abstains only under ``on_provider_error="abstain"``.
``ProviderAuthError`` is never retried.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from switchboard.core.audit import (
    AuditDraft,
    AuditRecord,
    ConfidenceReport,
    canonical_json,
    sha256_hex,
)
from switchboard.core.candidates import Candidate, ShortlistResult
from switchboard.engine.confidence import build_confidence_report
from switchboard.engine.entitlements import EntitlementResult, filter_routes
from switchboard.engine.policy import (
    ThresholdPolicy,
    abstain,
    clarify,
    resolve_decision,
)
from switchboard.engine.policy import apply_fallback as substitute_fallback
from switchboard.engine.prompt import (
    build_repair_segment,
    build_segments,
    order_candidates,
)
from switchboard.engine.schema import build_wire_schema, resolve_schema_mode
from switchboard.engine.shortlist import effective_k
from switchboard.engine.validate import (
    ValidationError,
    check_route_reference,
    extract_json_object,
    format_error_for_repair,
    parse_wire_output,
    validate_args,
)
from switchboard.errors import ConfigError, ProviderError
from switchboard.providers.base import LLMRequest, LLMResult
from switchboard.telemetry.emitter import apply_content_mode
from switchboard.telemetry.otel import OTelEmitter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from switchboard.core.context import RequestContext
    from switchboard.core.decision import Decision
    from switchboard.core.registry import Registry, RegistryView
    from switchboard.core.route import Route
    from switchboard.engine.prompt import CandidateOrder
    from switchboard.engine.schema import SchemaModeSetting, WireSchemaMode
    from switchboard.engine.shortlist import Shortlister
    from switchboard.providers.base import ClientCapabilities, PromptSegment
    from switchboard.telemetry.emitter import ContentMode, Redactor

__all__ = [
    "AttemptRecord",
    "ConfidenceSource",
    "LoopConfig",
    "LoopState",
    "OnProviderError",
    "RetrySpec",
    "apply_fallback",
    "arun_shortlist",
    "build_schema",
    "filter_entitlements",
    "finalize_audit",
    "order_and_build_prompt",
    "resolve_policy",
    "run_decision_async",
    "run_decision_sync",
    "run_pipeline_async",
    "run_pipeline_sync",
    "run_shortlist",
    "score_confidence",
    "snapshot_registry",
    "validate_output",
]


ConfidenceSource = Literal["logprobs", "verbalized", "none"]
"""Resolved ``Router(confidence=...)`` source (plan §3.3, §6.2).

``"logprobs+vote:n=3"`` and the margin gate are [v0.2] (§13 ruling #9); the
parser in ``router.py`` rejects them with a phase-tagged
:class:`~switchboard.errors.ConfigError` rather than silently degrading."""

OnProviderError = Literal["raise", "abstain"]
"""``Router(on_provider_error=...)`` (plan §3.3, §3.8). Default ``"raise"``."""

_ARGS_REPAIR_BUDGET = 1
"""Plan §4.5: *"an args-only failure gets one repair pass, then downgrades to
clarify"*. Deliberately separate from ``schema_attempts``: a sound route choice
with a missing argument is a user question, not a model failure, and spending the
whole schema budget re-asking for a value the user never supplied only burns
tokens."""

_FINGERPRINT_LEN = 12
"""Width of ``args_schema_fingerprint``, matching ``registry_version`` (§7.3)."""


# --------------------------------------------------------------------------- #
# RetrySpec (plan §3.3, Appendix A)
# --------------------------------------------------------------------------- #


class RetrySpec(BaseModel):
    """Retry budgets for the two independent failure modes (plan §3.3, §4.5).

    ``RetrySpec(schema_attempts=2, provider_attempts=3, backoff="expo_jitter")``
    is Appendix A's authoritative default. The two budgets are separate because
    the failures are: a *schema* failure means the model produced something
    unusable and is retried with a corrective repair segment (§4.5); a *provider*
    failure means the transport broke and is retried with the identical request
    after a backoff (§3.8).

    Attributes:
        schema_attempts: repair retries after the first call, so the loop makes at
            most ``1 + schema_attempts`` provider calls per decision.
        provider_attempts: total transport attempts per call, including the first.
            :class:`~switchboard.errors.ProviderAuthError` is never retried
            regardless (``retryable`` is a class attribute on the error).
        backoff: ``"expo_jitter"`` (the default), ``"expo"`` (deterministic — for
            reproducible tests) or ``"none"``.
        base_delay: first backoff delay in seconds.
        max_delay: ceiling for the exponential ramp, in seconds.
    """

    model_config = ConfigDict(frozen=True)

    schema_attempts: int = 2
    provider_attempts: int = 3
    backoff: Literal["expo_jitter", "expo", "none"] = "expo_jitter"
    base_delay: float = 0.5
    max_delay: float = 8.0

    @model_validator(mode="after")
    def _check(self) -> RetrySpec:
        """Reject budgets that cannot produce a decision (plan §3.8)."""
        if self.schema_attempts < 0:
            raise ConfigError(
                f"RetrySpec.schema_attempts must be >= 0, got {self.schema_attempts}"
            )
        if self.provider_attempts < 1:
            raise ConfigError(
                f"RetrySpec.provider_attempts must be >= 1 (the first call counts), "
                f"got {self.provider_attempts}"
            )
        if self.base_delay < 0 or self.max_delay < 0:
            raise ConfigError("RetrySpec delays must be >= 0")
        return self

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before transport attempt ``attempt + 1`` (plan §3.8).

        ``Retry-After`` always wins when the adapter recovered one from a
        :class:`~switchboard.errors.ProviderRateLimit` — the provider knows more
        than our exponential ramp does. Otherwise the delay is
        ``base_delay * 2**(attempt-1)``, clamped to ``max_delay``, and multiplied
        by a uniform jitter factor in ``[0.5, 1.0]`` under ``"expo_jitter"`` so a
        fleet of workers that failed together does not retry together.
        """
        if retry_after is not None and retry_after >= 0:
            return float(retry_after)
        if self.backoff == "none":
            return 0.0
        ramp: float = min(self.base_delay * (2.0 ** max(attempt - 1, 0)), self.max_delay)
        if self.backoff == "expo":
            return ramp
        return ramp * (0.5 + random.random() * 0.5)


# --------------------------------------------------------------------------- #
# Loop configuration and state (plan §2.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class LoopConfig:
    """Everything the loop needs that does **not** vary per request (plan §2.5).

    Built once in ``Router.__init__`` and shared, unmodified, by every request on
    every thread and task. That is the mechanical reason a single ``Router`` is
    thread-safe: all per-request mutable state lives in :class:`LoopState`, and
    this object is frozen.
    """

    registry: Registry
    capabilities: ClientCapabilities
    emitter: OTelEmitter = field(default_factory=OTelEmitter)

    # -- retrieval (plan §5) ------------------------------------------------- #
    shortlister: Shortlister | None = None
    shortlist_k: int | None = None
    allow_oversized_k: bool = False
    max_candidates: int = 25
    candidate_order: CandidateOrder = "shuffle"

    # -- decision surface (plan §3.3) ---------------------------------------- #
    multi_route: bool = False
    allow_clarify: bool = True
    allow_plan: bool = False
    fallback: str | None = None

    # -- confidence and thresholds (plan §6) --------------------------------- #
    thresholds: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    confidence_source: ConfidenceSource = "logprobs"
    confidence_fusion: Callable[[ConfidenceReport], float] | None = None
    thresholds_on_verbalized: bool = False

    # -- provider call (plan §4) --------------------------------------------- #
    retry: RetrySpec = field(default_factory=RetrySpec)
    wire_schema: SchemaModeSetting = "auto"
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    on_provider_error: OnProviderError = "raise"
    provider: str | None = None
    request_model: str | None = None

    # -- prompt / privacy (plan §4.6, §7.4) ---------------------------------- #
    system_prompt: str | None = None
    content_mode: ContentMode = "none"
    default_visibility: Literal["allow", "deny"] = "allow"

    @property
    def verbalized(self) -> bool:
        """Whether the wire schema should carry ``stated_confidence`` (§6.2).

        Only on the verbalized rung — no logprobs and confidence not switched off.
        Asking a model to score itself when a better signal exists costs tokens
        and, worse, gets fused into the ``min`` (§6.3).
        """
        return self.confidence_source != "none" and not self.capabilities.logprobs

    @property
    def want_logprobs(self) -> bool:
        """``LLMRequest.want_logprobs``; adapters map it down when unsupported."""
        return self.confidence_source == "logprobs"


@dataclass
class AttemptRecord:
    """One provider round trip inside the §4.5 repair loop.

    Plan §4.5: *"Every attempt (count, raw text, error) lands in the audit
    record"*. ``AuditRecord`` (§8.2) carries the count as ``validation_retries``;
    the per-attempt raw text and error live here on the state, where the eval
    harness's per-provider schema-failure metric reads them, rather than widening
    the canonical audit schema (§13 ruling #6 pins that schema).
    """

    index: int
    raw_text: str
    stage: str | None = None
    error: str | None = None


@dataclass
class LoopState:
    """The mutable accumulator every step appends to (plan §2.1).

    One instance per ``route()`` / ``aroute()`` call, never shared. Fields are
    filled in pipeline order; a stage that does not run leaves its slice at the
    default, which is what makes an early ``abstain("no_eligible_routes")`` (no
    LLM call at all) produce a well-formed audit record anyway.
    """

    query: str
    """The caller's query, verbatim and unredacted. Never sent anywhere."""
    config: LoopConfig
    draft: AuditDraft
    ctx: RequestContext | None = None
    redacted_query: str = ""
    """The query after ``Router(redactor=...)``. This is the string that reaches
    the shortlister, the prompt and the shuffle seed — the single redaction, done
    once by the Router (plan §7.4)."""
    bound_redactor: Redactor | None = None
    """``redactor`` with the request context already bound, for the audit egress
    site (plan §7.4 site 3)."""

    # -- catalog ------------------------------------------------------------- #
    view: RegistryView | None = None
    entitlement: EntitlementResult | None = None
    entitled_routes: tuple[Route, ...] = ()
    route_index: dict[str, Route] = field(default_factory=dict)
    allowed: frozenset[str] = frozenset()

    # -- retrieval ----------------------------------------------------------- #
    shortlist: ShortlistResult | None = None
    candidates: tuple[Candidate, ...] = ()
    candidate_names: tuple[str, ...] = ()
    shuffle_seed: int | None = None

    # -- prompt / schema ----------------------------------------------------- #
    segments: tuple[PromptSegment, ...] = ()
    schema_mode: WireSchemaMode = "static"
    wire_model: type[BaseModel] | None = None
    request: LLMRequest | None = None

    # -- provider ------------------------------------------------------------ #
    llm_results: list[LLMResult[Any]] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    provider_error: ProviderError | None = None

    # -- validation ---------------------------------------------------------- #
    parsed: BaseModel | None = None
    args: BaseModel | None = None
    member_args: dict[str, BaseModel | None] = field(default_factory=dict)
    validation_error: ValidationError | None = None

    # -- outcome ------------------------------------------------------------- #
    confidence: ConfidenceReport | None = None
    decision: Decision | None = None
    record: AuditRecord | None = None

    started: float = field(default_factory=time.perf_counter)

    # -- derived views ------------------------------------------------------- #

    @property
    def registry_version(self) -> str:
        """The registry content hash this decision was taken against (§7.3)."""
        return self.config.registry.version

    @property
    def view_hash(self) -> str:
        """The cohort key for prompt-cache segment B (§4.6, §7.2)."""
        return self.view.view_hash if self.view is not None else ""

    @property
    def committed_route(self) -> str | None:
        """The route name the model committed to, if any."""
        value = getattr(self.parsed, "route", None)
        return value if isinstance(value, str) and value else None

    @property
    def weak_retrieval(self) -> bool:
        """Whether retrieval found nothing convincing (§5.5)."""
        return bool(self.shortlist is not None and self.shortlist.weak_retrieval)

    @property
    def last_result(self) -> LLMResult[Any] | None:
        """The most recent provider response, or ``None`` if none was made."""
        return self.llm_results[-1] if self.llm_results else None


# --------------------------------------------------------------------------- #
# Step 1 — registry snapshot (plan §2.1)
# --------------------------------------------------------------------------- #


def run_pipeline_sync(state: LoopState, client: Any) -> LoopState:
    """Run the canonical decision pipeline synchronously.

    This is the single orchestration sequence shared with
    :func:`run_pipeline_async`: only shortlisting and provider I/O differ. The
    Router owns span/sink finalization around this helper so telemetry still runs
    in its ``finally`` block.
    """
    state = snapshot_registry(state)
    state = filter_entitlements(state)
    if state.decision is None:
        state = run_shortlist(state)
    if state.decision is None:
        state = order_and_build_prompt(state)
        state = build_schema(state)
        state = run_decision_sync(state, client)
        state = score_confidence(state)
        state = resolve_policy(state)
    return apply_fallback(state)


async def run_pipeline_async(state: LoopState, client: Any) -> LoopState:
    """Async twin of :func:`run_pipeline_sync` with awaited I/O stages."""
    state = snapshot_registry(state)
    state = filter_entitlements(state)
    if state.decision is None:
        state = await arun_shortlist(state)
    if state.decision is None:
        state = order_and_build_prompt(state)
        state = build_schema(state)
        state = await run_decision_async(state, client)
        state = score_confidence(state)
        state = resolve_policy(state)
    return apply_fallback(state)


def snapshot_registry(state: LoopState) -> LoopState:
    """Freeze the catalog for this request and stamp its identity (plan §2.1, §7.3).

    ``registry_version`` is a content hash, so taking the snapshot here means the
    prompt-cache prefix, the shortlist index key and the audit record all name the
    *same* catalog even if the caller swaps registries mid-flight.

    Also computes ``inputs_hash`` over the **redacted** query — the string that
    actually drove the decision, so an audit record replays to the same prompt
    (§5.6) — and stores the raw query as ``input_text`` for
    :func:`finalize_audit` to gate through ``content_mode`` (§7.4).
    """
    cfg = state.config
    ctx = state.ctx
    state.view = cfg.registry.view(ctx)
    draft = state.draft
    draft.registry_version = cfg.registry.version
    draft.candidates_total = len(cfg.registry)
    draft.provider = cfg.provider
    draft.request_model = cfg.request_model
    if ctx is not None:
        draft.tenant_id = ctx.tenant_id
        if ctx.user_id:
            draft.user_id_hash = sha256_hex(ctx.user_id)[:16]
        if ctx.trace_id and draft.trace_id is None:
            draft.trace_id = ctx.trace_id
    draft.set_inputs(
        state.redacted_query,
        _context_metadata(ctx),
        input_text=state.query,
    )
    return state


def _context_metadata(ctx: RequestContext | None) -> dict[str, Any]:
    """The context fields that participate in ``inputs_hash`` (plan §8.2).

    Only what can change the decision: entitlements (they change the candidate
    set), locale (it changes the clarify language) and the caller's cohort key.
    ``user_id`` and ``trace_id`` are deliberately excluded — two identical
    requests from two users must hash the same, or distillation dedup (§8.5) and
    the replay cache (§9.6) both stop working.
    """
    if ctx is None:
        return {}
    return {
        "tenant_id": ctx.tenant_id,
        "entitlements": sorted(ctx.entitlements),
        "entitlement_key": ctx.entitlement_key,
        "locale": ctx.locale,
    }


# --------------------------------------------------------------------------- #
# Step 2 — entitlement filter (plan §7.1; §13 ruling #3)
# --------------------------------------------------------------------------- #


def filter_entitlements(state: LoopState) -> LoopState:
    """Filter the catalog to what this context may see, pre-LLM (plan §7.1).

    An **empty** entitled set is a normal outcome, not an error: it degrades to
    ``abstain(reason="no_eligible_routes")`` **with no LLM call** (§13 ruling #3),
    which this step performs directly by setting ``state.decision``. The driver
    checks for that and short-circuits the rest of the pipeline.
    """
    cfg = state.config
    view = state.view
    routes = view.routes if view is not None else cfg.registry.routes
    result = filter_routes(routes, state.ctx, cfg.default_visibility)

    state.entitlement = result
    state.entitled_routes = result.routes
    state.route_index = {route.name: route for route in result.routes}
    state.allowed = result.allowed
    state.draft.candidates_entitled = len(result.routes)

    if not result.routes:
        state.decision = abstain("no_eligible_routes", draft=state.draft)
    return state


# --------------------------------------------------------------------------- #
# Step 3 — shortlist (plan §5)
# --------------------------------------------------------------------------- #


def run_shortlist(state: LoopState) -> LoopState:
    """Retrieve this request's candidate set (plan §5.1, §5.3, §5.4).

    I/O-free for the default ``BM25Shortlister`` — pure Python scoring over an
    in-memory index — which is why it sits among the pure steps. An
    :class:`~switchboard.engine.shortlist.EmbeddingShortlister` backed by a
    network embedder does perform I/O; :func:`arun_shortlist` is its async twin so
    ``aroute()`` never blocks the event loop on one.

    Three invariants are enforced here rather than in the backend:

    * the index is built over the **full registry** (IDF is corpus-wide and
      tenant-stable, §5.2) while ``allowed`` restricts scoring to the entitled
      set — filter *before* truncate (§5.9);
    * candidates are re-intersected with the entitled set afterwards, because
      every shortlister is **untrusted on entitlements** (§5.1): a buggy backend
      may degrade recall, never leak a route;
    * the result is capped at ``max_candidates``, keeping pinned routes (§5.5).
    """
    cfg = state.config
    if cfg.shortlister is None:
        return _absorb_shortlist(state, _full_entitled_result(state))

    cfg.shortlister.build(
        cfg.registry.routes, registry_version=cfg.registry.version
    )
    with state.draft.timing("shortlist"):
        result = cfg.shortlister.shortlist(
            state.redacted_query,
            allowed=state.allowed,
            k=_resolve_k(state),
            ctx=state.ctx,
        )
    return _absorb_shortlist(state, result)


async def arun_shortlist(state: LoopState) -> LoopState:
    """Async twin of :func:`run_shortlist` (plan §2.5).

    Uses ``abuild``/``ashortlist``, which the BM25 backend implements by direct
    delegation (no thread hop for CPU-only work) and the embedding backend
    implements by awaiting or offloading the embed callable. The single-flight
    build lock (§5.4) lives in the backend, so concurrent first requests after a
    registry change rebuild once, not N times.
    """
    cfg = state.config
    if cfg.shortlister is None:
        return _absorb_shortlist(state, _full_entitled_result(state))

    await cfg.shortlister.abuild(
        cfg.registry.routes, registry_version=cfg.registry.version
    )
    with state.draft.timing("shortlist"):
        result = await cfg.shortlister.ashortlist(
            state.redacted_query,
            allowed=state.allowed,
            k=_resolve_k(state),
            ctx=state.ctx,
        )
    return _absorb_shortlist(state, result)


def _resolve_k(state: LoopState) -> int:
    """This request's K, from the §5.3 band table or the configured override."""
    cfg = state.config
    configured = cfg.shortlist_k
    if configured is None:
        configured = getattr(cfg.shortlister, "top_k", None)
    return effective_k(
        len(state.entitled_routes),
        configured,
        allow_oversized_k=cfg.allow_oversized_k,
    )


def _full_entitled_result(state: LoopState) -> ShortlistResult:
    """The whole entitled catalog as a skipped shortlist (``shortlist=None``).

    Byte-identical, from the prompt's point of view, to the small-catalog bypass:
    full registry, canonical order, never shuffled, stable cached prefix (§5.3,
    §5.6). The Router refuses at construction to pair ``shortlist=None`` with a
    catalog larger than ``max_candidates``, so this can never silently truncate.
    """
    return ShortlistResult(
        candidates=[
            Candidate(route_name=route.name, score=0.0, rank=rank, source="all")
            for rank, route in enumerate(state.entitled_routes)
        ],
        skipped=True,
    )


def _absorb_shortlist(state: LoopState, result: ShortlistResult) -> LoopState:
    """Re-intersect with entitlements, cap at ``max_candidates``, record (§5.1)."""
    cfg = state.config
    allowed = state.allowed
    candidates = [c for c in result.candidates if c.route_name in allowed]

    if len(candidates) > cfg.max_candidates:
        pinned = [c for c in candidates if c.source == "pinned"]
        rest = [c for c in candidates if c.source != "pinned"]
        keep = max(cfg.max_candidates - len(pinned), 0)
        candidates = (rest[:keep] + pinned)[: cfg.max_candidates]

    result = result.model_copy(update={"candidates": candidates})
    state.shortlist = result
    state.draft.shortlist_skipped = result.skipped
    state.draft.weak_retrieval = result.weak_retrieval

    if not candidates:
        # Retrieval returned nothing that survives entitlement. There is nothing
        # to put in front of a model, so this is the §13 ruling #3 outcome again
        # rather than a call that can only hallucinate.
        state.decision = abstain("no_eligible_routes", draft=state.draft)
    return state


# --------------------------------------------------------------------------- #
# Step 4 — candidate ordering + prompt assembly (plan §4.6, §5.6)
# --------------------------------------------------------------------------- #


def order_and_build_prompt(state: LoopState) -> LoopState:
    """Order the candidates and assemble the A/B/C/D prompt (plan §4.6, §5.6).

    When the shortlist was **skipped** (small-catalog bypass or
    ``shortlist=None``) segment C is omitted entirely and the candidates keep
    canonical order: the full entitled directory in segment B *is* the option
    set, and shuffling a cached prefix would trade the ~90% caching discount for
    a position-bias benefit that does not exist at N < 25 (§5.6).

    Otherwise the shortlist is seeded-shuffled. The seed is a stable sha256
    digest of ``(redacted query, registry_version)`` — never ``hash()`` — so the
    same request produces the same prompt across processes and the replay cache
    works (§13 ruling #8). Seed and pre-shuffle ranks both land in the audit
    record, which is what makes the prompt reconstructible.
    """
    cfg = state.config
    result = state.shortlist
    if result is None:  # pragma: no cover - driver always runs the shortlist first
        raise ConfigError("order_and_build_prompt() called before run_shortlist()")

    if result.skipped:
        ordered: Sequence[Candidate] = result.candidates
        seed: int | None = None
        prompt_candidates: Sequence[Candidate] | None = None
    else:
        ordered, seed = order_candidates(
            result.candidates,
            cfg.candidate_order,
            state.redacted_query,
            state.registry_version,
        )
        prompt_candidates = ordered

    state.candidates = tuple(ordered)
    state.candidate_names = tuple(c.route_name for c in ordered)
    state.shuffle_seed = seed
    state.draft.shuffle_seed = seed
    state.draft.add_candidates(ordered)

    state.segments = build_segments(
        routes=state.entitled_routes,
        query=state.redacted_query,
        registry_version=state.registry_version,
        view_hash=state.view_hash,
        candidates=prompt_candidates,
        ctx=state.ctx,
        allow_clarify=cfg.allow_clarify,
        multi_route=cfg.multi_route,
        allow_plan=cfg.allow_plan,
        weak_retrieval=result.weak_retrieval,
        system_prompt=cfg.system_prompt,
    )
    return state


# --------------------------------------------------------------------------- #
# Step 5 — wire schema + request (plan §4.4)
# --------------------------------------------------------------------------- #


def build_schema(state: LoopState) -> LoopState:
    """Build this call's wire schema and the :class:`LLMRequest` (plan §4.4).

    ``wire_schema="auto"`` resolves to ``dynamic`` only on the ``grammar`` rung,
    where a per-query ``Literal`` route enum makes hallucinated names
    unrepresentable at no caching cost; on ``tool_strict``/``json_mode``/``none``
    it resolves to ``static``, because there the schema sits inside the cached
    prefix and a per-query enum would void the ~90% discount on every request
    (§13 ruling #14). The validator enforces ``route ∈ candidates`` in both
    modes, so ``static`` loses only the decoding-time guarantee.
    """
    cfg = state.config
    state.schema_mode = resolve_schema_mode(cfg.capabilities, cfg.wire_schema)
    state.wire_model = build_wire_schema(
        state.candidate_names,
        mode=state.schema_mode,
        allow_clarify=cfg.allow_clarify,
        multi_route=cfg.multi_route,
        allow_plan=cfg.allow_plan,
        verbalized=cfg.verbalized,
    )
    state.request = LLMRequest(
        segments=state.segments,
        output_schema=state.wire_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        want_logprobs=cfg.want_logprobs,
        seed=cfg.seed,
    )
    return state


# --------------------------------------------------------------------------- #
# Step 6 — validation of one attempt (plan §4.5)
# --------------------------------------------------------------------------- #


def validate_output(state: LoopState) -> LoopState:
    """Run §4.5's three validation layers over the newest provider result.

    Called once per attempt from inside the repair loop, so the repair segment can
    quote the exact failure. Layers, in order, each with its own §3.8 outcome:

    1. **schema parse** — tolerant extraction (fences, outermost balanced object)
       then Pydantic validation against the per-call wire model;
    2. **route reference** — every name the output mentions must be in this
       request's candidate set, checked even under a ``Literal`` enum;
    3. **args** — the committed route's ``args_model``, or every member's under
       ``multi_route``.

    Sets exactly one of ``state.parsed`` (+ ``args`` / ``member_args``) or
    ``state.validation_error``. Never raises.
    """
    result = state.last_result
    wire_model = state.wire_model
    if result is None or wire_model is None:  # pragma: no cover - driver invariant
        raise ConfigError("validate_output() called before a provider call")

    started = time.perf_counter()
    try:
        state.args = None
        state.member_args = {}
        payload: Any = result.parsed if result.parsed is not None else result.raw_text

        parsed, error = parse_wire_output(payload, wire_model)
        if error is not None or parsed is None:
            state.parsed = None
            state.validation_error = _reclassify_enum_failure(state, error, result)
            return state

        state.parsed = parsed
        reference_error = check_route_reference(parsed, set(state.candidate_names))
        if reference_error is not None:
            state.validation_error = reference_error
            return state

        state.validation_error = _validate_all_args(state, parsed)
        return state
    finally:
        state.draft.add_latency("validation", (time.perf_counter() - started) * 1000.0)


def _reclassify_enum_failure(
    state: LoopState, error: ValidationError | None, result: LLMResult[Any]
) -> ValidationError | None:
    """Re-stage "hallucinated route name" from ``schema`` to ``route_ref`` (§3.8).

    Under ``wire_schema="dynamic"`` the ``route`` field is a ``Literal`` over this
    request's candidates (§4.4), so a name outside that set is rejected by
    Pydantic during parsing and never reaches
    :func:`~switchboard.engine.validate.check_route_reference`. Left alone it
    would be audited as ``unparseable_output``, but §3.8's degradation table calls
    it ``invalid_route_reference`` — and the distinction is load-bearing: it is
    exactly the split between "the model cannot emit valid JSON" and "the model
    picked a route that does not exist", which §9.2's retrieval-gap vs
    confusion-gap decomposition is computed from.

    So the raw payload is re-read leniently and, if it does name a non-candidate
    route, the failure is re-stamped. The verdict itself is unchanged — same
    repair, same abstain if the repairs are exhausted — only its *reason* becomes
    truthful. Returns ``error`` untouched on every other path.
    """
    if error is None or error.stage != "schema" or state.schema_mode != "dynamic":
        return error

    payload = _loose_payload(result)
    if payload is None:
        return error

    referenced: list[str] = []
    value = payload.get("route")
    if isinstance(value, str) and value:
        referenced.append(value)
    for member in payload.get("routes") or ():
        name = member.get("route") if isinstance(member, dict) else member
        if isinstance(name, str) and name:
            referenced.append(name)
    for name in payload.get("candidates") or ():
        if isinstance(name, str) and name:
            referenced.append(name)

    allowed = set(state.candidate_names)
    unknown = [name for name in dict.fromkeys(referenced) if name not in allowed]
    if not unknown:
        return error

    return ValidationError(
        "route_ref",
        "model named route(s) that are not candidates for this request: "
        + ", ".join(repr(name) for name in unknown),
        {"unknown": unknown, "allowed": sorted(allowed), "errors": error.detail.get("errors", [])},
    )


def _loose_payload(result: LLMResult[Any]) -> dict[str, Any] | None:
    """Re-read a rejected response as plain JSON, or ``None`` if it is not JSON."""
    blob = extract_json_object(result.raw_text)
    if blob is None:
        return None
    try:
        payload = json.loads(blob)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate_all_args(state: LoopState, parsed: BaseModel) -> ValidationError | None:
    """Validate args for the committed route, or for every multi-route member."""
    kind = getattr(parsed, "kind", "route")
    if kind == "route":
        name = state.committed_route
        route = state.route_index.get(name) if name else None
        if route is None:  # pragma: no cover - layer 2 already rejected this
            return None
        args, error = validate_args(getattr(parsed, "args", None), route)
        state.args = args
        return error
    if kind == "multi_route":
        for member in getattr(parsed, "routes", None) or ():
            name = getattr(member, "route", None)
            route = state.route_index.get(name) if isinstance(name, str) else None
            if route is None:
                continue
            args, error = validate_args(getattr(member, "args", None), route)
            if error is not None:
                return error
            state.member_args[route.name] = args
    return None


# --------------------------------------------------------------------------- #
# Step 7 — THE I/O POINT: provider call + repair loop (plan §4.5, §3.8)
# --------------------------------------------------------------------------- #


def run_decision_sync(state: LoopState, client: Any) -> LoopState:
    """Drive the provider call and the §4.5 validate-and-retry loop, synchronously.

    This is one of exactly two I/O points in the library (the other is the sink
    flush). Everything it does is mirrored verbatim by
    :func:`run_decision_async`; the only difference is ``client.complete`` versus
    ``await client.acomplete`` and ``time.sleep`` versus ``asyncio.sleep``.

    **Repair loop (plan §4.5).** At most ``1 + retry.schema_attempts`` calls (3 by
    default). On a failure the repair segment — the validation error plus the
    truncated prior output plus a re-ask — is appended **after** the query
    segment, never before it, so the stable A+B prefix still hits the provider
    cache on exactly the requests that already cost extra. Temperature is forced
    to 0 on every retry: the first sample was creative enough.

    **Args failures get their own, smaller budget** (one repair pass, §4.5): a
    sound route with a missing argument is a question for the user, and burning
    the schema budget on it just delays the ``clarify``.

    **Transport failures (plan §3.8).** ``ProviderTimeout`` / ``ProviderRateLimit``
    are retried up to ``retry.provider_attempts`` with exponential backoff plus
    jitter, honouring ``Retry-After`` when the adapter recovered one.
    ``ProviderAuthError`` is never retried. Exhaustion **raises** by default and
    degrades to ``abstain("provider_error")`` only under
    ``on_provider_error="abstain"``.
    """
    cfg = state.config
    request = _require_request(state)
    max_calls = 1 + max(cfg.retry.schema_attempts, 0)
    args_repairs = 0

    try:
        for attempt in range(1, max_calls + 1):
            result = _call_provider_sync(state, client, request)
            _absorb_result(state, result)
            state = validate_output(state)
            error = _record_attempt(state, attempt, result)
            if error is None:
                break
            if attempt >= max_calls:
                break
            if error.stage == "args":
                if args_repairs >= _ARGS_REPAIR_BUDGET:
                    break
                args_repairs += 1
            request = _repair_request(request, error, result)
    except ProviderError as exc:
        return _handle_provider_error(state, exc)

    state.draft.validation_retries = max(len(state.attempts) - 1, 0)
    return state


async def run_decision_async(state: LoopState, client: Any) -> LoopState:
    """Async twin of :func:`run_decision_sync` — identical logic, awaited I/O.

    A sync-only client is driven through ``asyncio.to_thread`` so it cannot block
    the event loop, which is the documented direction of plan §2.5's mismatch
    rule (the other direction — an async-only client under ``route()`` — is a
    :class:`~switchboard.errors.ConfigError`, raised by the Router).
    """
    cfg = state.config
    request = _require_request(state)
    max_calls = 1 + max(cfg.retry.schema_attempts, 0)
    args_repairs = 0

    try:
        for attempt in range(1, max_calls + 1):
            result = await _call_provider_async(state, client, request)
            _absorb_result(state, result)
            state = validate_output(state)
            error = _record_attempt(state, attempt, result)
            if error is None:
                break
            if attempt >= max_calls:
                break
            if error.stage == "args":
                if args_repairs >= _ARGS_REPAIR_BUDGET:
                    break
                args_repairs += 1
            request = _repair_request(request, error, result)
    except ProviderError as exc:
        return _handle_provider_error(state, exc)

    state.draft.validation_retries = max(len(state.attempts) - 1, 0)
    return state


def _require_request(state: LoopState) -> LLMRequest:
    if state.request is None:  # pragma: no cover - driver invariant
        raise ConfigError("run_decision_*() called before build_schema()")
    return state.request


def _record_attempt(
    state: LoopState, attempt: int, result: LLMResult[Any]
) -> ValidationError | None:
    """Log one attempt's (count, raw text, error) triple (plan §4.5)."""
    error = state.validation_error
    state.attempts.append(
        AttemptRecord(
            index=attempt,
            raw_text=result.raw_text,
            stage=error.stage if error is not None else None,
            error=error.message if error is not None else None,
        )
    )
    return error


def _repair_request(
    request: LLMRequest, error: ValidationError, result: LLMResult[Any]
) -> LLMRequest:
    """Append the repair segment after the query and force temperature to 0 (§4.5)."""
    segment = build_repair_segment(format_error_for_repair(error), result.raw_text)
    return request.model_copy(
        update={"segments": (*request.segments, segment), "temperature": 0.0}
    )


def _absorb_result(state: LoopState, result: LLMResult[Any]) -> None:
    """Accumulate usage and response identity from one provider call (§8.2)."""
    draft = state.draft
    state.llm_results.append(result)
    draft.add_usage(result.usage)
    if result.model_id:
        draft.response_model = result.model_id
    meta = result.provider_meta or {}
    response_id = meta.get("response_id") or meta.get("id")
    if response_id is not None:
        draft.response_id = str(response_id)


def _handle_provider_error(state: LoopState, exc: ProviderError) -> LoopState:
    """Apply ``on_provider_error`` (plan §3.8): raise by default, else abstain."""
    state.draft.set_error(exc)
    if state.config.on_provider_error == "raise":
        raise exc
    state.provider_error = exc
    state.draft.validation_retries = max(len(state.attempts) - 1, 0)
    return state


def _call_provider_sync(
    state: LoopState, client: Any, request: LLMRequest
) -> LLMResult[Any]:
    """One provider call with the §3.8 transport-retry ladder, synchronously."""
    cfg = state.config
    attempts = cfg.retry.provider_attempts
    for attempt in range(1, attempts + 1):
        try:
            return _timed_call_sync(state, client, request)
        except ProviderError as exc:
            if not type(exc).retryable or attempt >= attempts:
                raise
            delay = cfg.retry.delay_for(attempt, getattr(exc, "retry_after", None))
            if delay > 0:
                time.sleep(delay)
    raise ConfigError(  # pragma: no cover - unreachable: the loop always returns or raises
        "provider retry loop exhausted without a result"
    )


async def _call_provider_async(
    state: LoopState, client: Any, request: LLMRequest
) -> LLMResult[Any]:
    """One provider call with the §3.8 transport-retry ladder, asynchronously."""
    cfg = state.config
    attempts = cfg.retry.provider_attempts
    for attempt in range(1, attempts + 1):
        try:
            return await _timed_call_async(state, client, request)
        except ProviderError as exc:
            if not type(exc).retryable or attempt >= attempts:
                raise
            delay = cfg.retry.delay_for(attempt, getattr(exc, "retry_after", None))
            if delay > 0:
                await asyncio.sleep(delay)
    raise ConfigError(  # pragma: no cover - unreachable
        "provider retry loop exhausted without a result"
    )


def _timed_call_sync(
    state: LoopState, client: Any, request: LLMRequest
) -> LLMResult[Any]:
    """Invoke ``client.complete`` inside a ``chat`` span, timing it (§8.1, §8.2)."""
    cfg = state.config
    model = cfg.request_model or "unknown"
    started = time.perf_counter()
    with cfg.emitter.chat_span(model, provider=cfg.provider, request=request) as span:
        try:
            result: LLMResult[Any] = client.complete(request)
        finally:
            state.draft.add_latency("llm_total", (time.perf_counter() - started) * 1000.0)
        span.set_result(result)
    return result


async def _timed_call_async(
    state: LoopState, client: Any, request: LLMRequest
) -> LLMResult[Any]:
    """Async twin of :func:`_timed_call_sync`; offloads sync-only clients (§2.5)."""
    cfg = state.config
    model = cfg.request_model or "unknown"
    acomplete = getattr(client, "acomplete", None)
    started = time.perf_counter()
    async with cfg.emitter.achat_span(
        model, provider=cfg.provider, request=request
    ) as span:
        try:
            result: LLMResult[Any]
            if callable(acomplete):
                result = await acomplete(request)
            else:
                result = await asyncio.to_thread(client.complete, request)
        finally:
            state.draft.add_latency("llm_total", (time.perf_counter() - started) * 1000.0)
        span.set_result(result)
    return result


# --------------------------------------------------------------------------- #
# Step 8 — confidence (plan §6)
# --------------------------------------------------------------------------- #


def score_confidence(state: LoopState) -> LoopState:
    """Compute the one :class:`ConfidenceReport` for this decision (plan §6.3).

    ``p_route`` is the v0.1 signal: the geometric-mean token probability of the
    committed route value, measured *after* the rationale because the wire schema
    puts ``rationale`` first (§3.4). It is ``None`` — and the threshold machinery
    therefore inert — whenever the provider path has no logprobs, which is the
    no-signal rule (§6.3) rather than a fabricated number.

    The verbalized ``stated_confidence`` is read when the verbalized rung armed
    it; it is recorded raw and damped x0.85 at fusion, never solely trusted.
    """
    cfg = state.config
    thresholds = cfg.thresholds.as_thresholds()
    if cfg.confidence_source == "none":
        state.confidence = ConfidenceReport(
            score=0.0, method="none", thresholds=thresholds
        )
        return state

    result = state.last_result
    stated = getattr(state.parsed, "stated_confidence", None)
    meta = (result.provider_meta or {}) if result is not None else {}
    avg_logprob = meta.get("avg_logprobs", meta.get("avg_logprob"))

    state.confidence = build_confidence_report(
        token_logprobs=result.token_logprobs if result is not None else None,
        route_name=state.committed_route,
        stated=float(stated) if isinstance(stated, (int, float)) else None,
        avg_logprob=float(avg_logprob) if isinstance(avg_logprob, (int, float)) else None,
        thresholds=thresholds,
        fusion=cfg.confidence_fusion,
    )
    return state


# --------------------------------------------------------------------------- #
# Step 9 — policy (plan §3.8, §6.4)
# --------------------------------------------------------------------------- #


def resolve_policy(state: LoopState) -> LoopState:
    """Turn the validated output into the typed Decision (plan §3.8, §6.4).

    Implements §3.8's degradation table for everything the loop can produce.
    Nothing here raises: schema and validation exhaustion **always** degrade,
    whether or not a fallback is configured (§13 ruling #2).

    ==========================================  ===================================
    Condition                                   Outcome
    ==========================================  ===================================
    provider exhausted, ``on_provider_error``   ``abstain("provider_error")``
      ``="abstain"``
    schema unparseable after the repairs        ``abstain("unparseable_output")``
    non-candidate route after the repairs       ``abstain("invalid_route_reference")``
    args invalid after one repair pass          ``clarify`` (``abstain("invalid_args")``
                                                if clarify is off)
    otherwise                                   thresholds, downgrade-only (§6.4)
    ==========================================  ===================================

    **This step finalizes the audit draft** (via ``engine/policy.py``), so it
    first stamps every ``content_mode``-gated field onto the draft. Anything
    written to the draft after this point is lost.
    """
    cfg = state.config
    draft = state.draft
    _stamp_draft(state)
    report = state.confidence or ConfidenceReport(
        score=0.0, method="none", thresholds=cfg.thresholds.as_thresholds()
    )

    if state.provider_error is not None:
        state.decision = abstain(
            "provider_error", draft=draft, confidence=report
        )
        return state

    error = state.validation_error
    if error is not None:
        state.decision = _degrade(state, error, report)
        return state

    if state.parsed is None:
        state.decision = abstain(
            "unparseable_output", draft=draft, confidence=report
        )
        return state

    state.decision = resolve_decision(
        state.parsed,
        state.route_index,
        report,
        cfg.thresholds,
        draft,
        args=state.args,
        member_args=state.member_args,
        candidates=state.candidates,
        allow_clarify=cfg.allow_clarify,
        allow_multi_route=cfg.multi_route,
        weak_retrieval=state.weak_retrieval,
        thresholds_on_verbalized=cfg.thresholds_on_verbalized,
    )
    return state


def _degrade(
    state: LoopState, error: ValidationError, report: ConfidenceReport
) -> Decision:
    """Map an exhausted validation failure to its §3.8 outcome."""
    cfg = state.config
    draft = state.draft
    if error.stage == "schema":
        return abstain("unparseable_output", draft=draft, confidence=report)
    if error.stage == "route_ref":
        return abstain("invalid_route_reference", draft=draft, confidence=report)

    # Args-only failure: the route choice was sound, so this is a question for the
    # user rather than a routing failure (§3.8 row 4).
    name = state.committed_route
    if not cfg.allow_clarify:
        return abstain("invalid_args", draft=draft, confidence=report)
    return clarify(
        _missing_args_question(error.missing),
        draft=draft,
        confidence=report,
        candidates=(name,) if name else (),
        missing=error.missing,
        rationale=getattr(state.parsed, "rationale", "") or "",
        downgraded_from="route",
        reason="invalid_args",
    )


def _missing_args_question(missing: Sequence[str]) -> str:
    """Template the question for an args-only failure (plan §6.5).

    The model committed to a route, so nobody asked it to author a question;
    re-prompting for one would cost a second round trip on a request that has
    already spent its repair budget. Field names are the only material available,
    so they are de-snake-cased and asked for directly.
    """
    fields = [name.replace("_", " ").strip() for name in missing if name]
    if not fields:
        return "Could you give me a bit more detail so I can complete that request?"
    if len(fields) == 1:
        return f"What is the {fields[0]}?"
    return f"Could you tell me the {', '.join(fields[:-1])} and {fields[-1]}?"


def _stamp_draft(state: LoopState) -> None:
    """Write the last draft fields before ``policy`` freezes it (plan §8.2).

    ``rationale``, ``args`` and ``input_text`` are stored **raw** here;
    :func:`finalize_audit` runs them through the single content gate
    (``apply_content_mode``) afterwards. ``args_hash`` and
    ``args_schema_fingerprint`` are computed unconditionally so records stay
    joinable and dedupable under ``content_mode="none"`` (§7.4).
    """
    draft = state.draft
    parsed = state.parsed
    if parsed is not None:
        rationale = getattr(parsed, "rationale", None)
        if isinstance(rationale, str) and rationale:
            draft.rationale = rationale

    if state.args is None:
        return
    payload = state.args.model_dump(mode="json")
    draft.args = payload
    draft.args_hash = sha256_hex(canonical_json(payload))
    name = state.committed_route
    route = state.route_index.get(name) if name else None
    if route is not None and route.args_model is not None:
        draft.args_schema_fingerprint = sha256_hex(
            canonical_json(route.args_model.model_json_schema())
        )[:_FINGERPRINT_LEN]


# --------------------------------------------------------------------------- #
# Step 10 — fallback (plan §6.6, §13 ruling #5)
# --------------------------------------------------------------------------- #


def apply_fallback(state: LoopState) -> LoopState:
    """Substitute the configured fallback for a terminal abstain (plan §6.6).

    One shape everywhere: the result is a ``RouteDecision`` with
    ``decision_path="fallback"`` and ``args=None``, so a caller's ``kind ==
    "route"`` branch needs no fallback handling at all. The pre-fallback
    ``AbstainReason`` survives in ``audit.abstain_reason``.

    **Entitlement is never bypassed**: the entitled route set for *this* request
    is what the substitution is checked against, so a globally configured
    fallback that a tenant may not reach leaves the abstain standing (§6.6).
    """
    if state.decision is None:
        return state
    state.decision = substitute_fallback(
        state.decision,
        state.config.fallback,
        state.entitled_routes,
        draft=state.draft,
    )
    return state


# --------------------------------------------------------------------------- #
# Step 11 — audit (plan §2.1, §7.4, §8.2)
# --------------------------------------------------------------------------- #


def finalize_audit(state: LoopState) -> LoopState:
    """Freeze the draft, apply the content gate, re-attach it (plan §2.1, §7.4).

    Called from the driver's ``finally`` block, so a raised provider error still
    produces a record (and therefore a span with ``error.type`` set) exactly as
    §2.1 requires. ``AuditDraft.finalize`` is idempotent, so the happy path — where
    ``engine/policy.py`` already froze the draft to build ``Decision.audit`` — and
    the error path both yield exactly one record.

    :func:`~switchboard.telemetry.emitter.apply_content_mode` is the **single**
    place payload text is stripped or redacted (§7.4 egress site 3), so the record
    is gated here rather than in the loop, and the resulting record is grafted
    back onto the Decision — a caller reading ``decision.audit`` must never see
    content the router was configured not to capture.
    """
    cfg = state.config
    record = apply_content_mode(
        state.draft.finalize(), cfg.content_mode, state.bound_redactor
    )
    state.record = record
    decision = state.decision
    if decision is not None and decision.audit is not record:
        state.decision = decision.model_copy(update={"audit": record})
    return state
