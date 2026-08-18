"""Core type contracts (plan §3, §5.1, §8.2).

Everything in this package is Pydantic + stdlib only. Per the import-topology
rule (plan §2.4), ``core/`` may never import an optional third-party SDK — not at
module level, not lazily. The CI import-linter contract enforces it.

Import order within the package is acyclic by construction::

    context ─┐
    candidates ─┬─> audit ──> decision
    errors ─────┘      └────> route ──> registry
"""

from __future__ import annotations

from switchboard.core.audit import (
    AbstainReason,
    AuditDraft,
    AuditRecord,
    ConfidenceReport,
    CostBlock,
    DecisionKind,
    DecisionPath,
    LatencyBlock,
    canonical_json,
    new_decision_id,
    sha256_hex,
)
from switchboard.core.candidates import Candidate, ShortlistResult
from switchboard.core.context import RequestContext
from switchboard.core.decision import (
    AbstainDecision,
    ClarifyDecision,
    Decision,
    DecisionAdapter,
    MultiRouteDecision,
    PlanDecision,
    PlanStep,
    RoutedCall,
    RouteDecision,
)
from switchboard.core.registry import Registry, RegistryView
from switchboard.core.route import Route

# Groups, for orientation (the list itself is sorted, per lint):
#   catalog   - Route, Registry, RegistryView
#   request   - RequestContext
#   retrieval - Candidate, ShortlistResult
#   decision  - AbstainReason, ConfidenceReport, RoutedCall, *Decision, Decision,
#               DecisionAdapter, DecisionKind, DecisionPath, PlanStep
#   audit     - AuditRecord, AuditDraft, LatencyBlock, CostBlock
#   hashing   - canonical_json, sha256_hex, new_decision_id (plan §7.3, §8.2)
__all__ = [
    "AbstainDecision",
    "AbstainReason",
    "AuditDraft",
    "AuditRecord",
    "Candidate",
    "ClarifyDecision",
    "ConfidenceReport",
    "CostBlock",
    "Decision",
    "DecisionAdapter",
    "DecisionKind",
    "DecisionPath",
    "LatencyBlock",
    "MultiRouteDecision",
    "PlanDecision",
    "PlanStep",
    "Registry",
    "RegistryView",
    "RequestContext",
    "Route",
    "RouteDecision",
    "RoutedCall",
    "ShortlistResult",
    "canonical_json",
    "new_decision_id",
    "sha256_hex",
]
