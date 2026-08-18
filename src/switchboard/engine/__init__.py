"""Decision-loop internals (plan §2.1, §2.5).

Every stage of the canonical loop is a step function in :mod:`~switchboard.engine.loop`
with the signature ``(LoopState) -> LoopState``; the only I/O points in the whole
library are the provider call and the telemetry flush. That is what makes
``Router.route()`` and ``Router.aroute()`` structurally identical drivers over
**one** pipeline rather than two hand-maintained copies (plan §2.5).

Module map::

    loop.py          the pipeline: LoopState, LoopConfig, RetrySpec, the steps    [v0.1]
    entitlements.py  pre-LLM predicate filter -> eligible set + entitlement_key   [v0.1/v0.2]
    shortlist.py     Shortlister Protocol, pure-Python BM25, "auto" bypass        [v0.1]
    prompt.py        cache-stable A/B/C/D segments, seeded shuffle, repair text   [v0.1]
    schema.py        per-call wire schema, rationale-first, dynamic vs static     [v0.1]
    validate.py      parse -> route-reference -> args, as pure predicates         [v0.1]
    confidence.py    p_route [v0.1]; margin/vote/fusion [v0.2]; the no-signal rule
    policy.py        ThresholdPolicy, downgrade-only resolution, fallback         [v0.1]

Nothing here may import an optional third-party SDK, lazily or otherwise (plan
§2.4 import topology); the sole exception is ``shortlist.py``'s function-local
import of the packaged embedding backends, which live behind the ``[embed]``
extra in ``switchboard.shortlisters``.

These are internals. The supported public surface is
``from switchboard import ...`` (plan §3); names re-exported here are for
advanced users composing their own drivers and for the eval harness.
"""

from __future__ import annotations

from switchboard.engine.confidence import (
    build_confidence_report,
    compute_margin,
    compute_p_route,
    default_fusion,
    signals_are_actionable,
)
from switchboard.engine.entitlements import EntitlementResult, filter_routes
from switchboard.engine.loop import (
    AttemptRecord,
    LoopConfig,
    LoopState,
    RetrySpec,
    apply_fallback,
    arun_shortlist,
    build_schema,
    filter_entitlements,
    finalize_audit,
    order_and_build_prompt,
    resolve_policy,
    run_decision_async,
    run_decision_sync,
    run_shortlist,
    score_confidence,
    snapshot_registry,
    validate_output,
)
from switchboard.engine.policy import (
    ThresholdPolicy,
    abstain,
    clarify,
    resolve_decision,
    route_index,
    synthesize_clarify_question,
)
from switchboard.engine.prompt import (
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    build_repair_segment,
    build_segments,
    order_candidates,
    render_prompt,
    shuffle_seed,
)
from switchboard.engine.schema import (
    build_wire_schema,
    enabled_kinds,
    resolve_schema_mode,
    strict_compatible_json_schema,
)
from switchboard.engine.shortlist import (
    AutoShortlister,
    BM25Shortlister,
    EmbeddingShortlister,
    IndexStore,
    MemoryIndexStore,
    Shortlister,
    effective_k,
    resolve_shortlister,
    tokenize,
)
from switchboard.engine.validate import (
    ValidationError,
    check_route_reference,
    format_error_for_repair,
    parse_wire_output,
    validate_args,
)

__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "SYSTEM_PROMPT",
    "AttemptRecord",
    "AutoShortlister",
    "BM25Shortlister",
    "EmbeddingShortlister",
    "EntitlementResult",
    "IndexStore",
    "LoopConfig",
    "LoopState",
    "MemoryIndexStore",
    "RetrySpec",
    "Shortlister",
    "ThresholdPolicy",
    "ValidationError",
    "abstain",
    "apply_fallback",
    "arun_shortlist",
    "build_confidence_report",
    "build_repair_segment",
    "build_schema",
    "build_segments",
    "build_wire_schema",
    "check_route_reference",
    "clarify",
    "compute_margin",
    "compute_p_route",
    "default_fusion",
    "effective_k",
    "enabled_kinds",
    "filter_entitlements",
    "filter_routes",
    "finalize_audit",
    "format_error_for_repair",
    "order_and_build_prompt",
    "order_candidates",
    "parse_wire_output",
    "render_prompt",
    "resolve_decision",
    "resolve_policy",
    "resolve_schema_mode",
    "resolve_shortlister",
    "route_index",
    "run_decision_async",
    "run_decision_sync",
    "run_shortlist",
    "score_confidence",
    "shuffle_seed",
    "signals_are_actionable",
    "snapshot_registry",
    "strict_compatible_json_schema",
    "synthesize_clarify_question",
    "tokenize",
    "validate_args",
    "validate_output",
]
