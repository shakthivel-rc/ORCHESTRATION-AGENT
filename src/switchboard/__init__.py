"""switchboard — the bring-your-own-model decision layer (plan §3).

*One typed ``route()`` call that picks among your tools/routes, extracts args,
asks for clarification, or abstains — in any framework, with any provider, with
an audit trail you can distill.*

Quickstart (plan §3.7, ten lines, verbatim)::

    from switchboard import Route, Registry, Router, RequestContext
    from pydantic import BaseModel

    class RefundArgs(BaseModel):
        order_id: str
        reason: str | None = None

    registry = Registry([
        Route(name="refund", description="Issue or check a refund for an order",
              args_model=RefundArgs, examples=("I want my money back for order 123",),
              tags=frozenset({"billing"})),
        Route(name="track_order", description="Track shipment status for an existing order"),
        Route(name="human_handoff", description="Escalate to a human support agent", pinned=True),
    ])
    router = Router(registry=registry, client="instructor:openai/gpt-5-nano",
                    shortlist="auto", allow_clarify=True, fallback="human_handoff")
    decision = router.route("where is my package for order 123?",
                            context=RequestContext(tenant_id="acme"))
    if decision.kind == "route":
        print(decision.route, decision.args, decision.confidence.score)

**This module re-exports, and does nothing else** (plan §2.3). In particular it
never imports a provider adapter or any optional dependency: ``import
switchboard`` in a bare venv pulls in Pydantic and the standard library, and
nothing more. Adapters are imported lazily by ``providers.resolve_client`` at
``Router(...)`` construction, embedding backends by the embedding shortlister,
and ``opentelemetry`` by the span emitter — each behind a function-local import
(plan §2.4). CI asserts this with a bare-venv guard that runs a full routing call
and checks ``sys.modules`` against a deny-list.

Everything below is the supported public API. Names not listed in
:data:`__all__` — including the whole ``switchboard.engine`` package — are
internals and may change without a major version bump.
"""

from __future__ import annotations

from switchboard._version import __version__
from switchboard.core.audit import (
    AbstainReason,
    AuditRecord,
    ConfidenceReport,
    CostBlock,
    DecisionKind,
    DecisionPath,
    LatencyBlock,
)
from switchboard.core.candidates import Candidate, ShortlistResult
from switchboard.core.context import RequestContext
from switchboard.core.decision import (
    AbstainDecision,
    ClarifyDecision,
    Decision,
    MultiRouteDecision,
    PlanDecision,
    PlanStep,
    RoutedCall,
    RouteDecision,
)
from switchboard.core.registry import Registry, RegistryView
from switchboard.core.route import Route
from switchboard.engine.loop import RetrySpec
from switchboard.engine.policy import ThresholdPolicy
from switchboard.errors import (
    ConfigError,
    MissingDependencyError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimit,
    ProviderTimeout,
    RegistryError,
    SwitchboardError,
)
from switchboard.providers.base import (
    AsyncLLMClient,
    ClientCapabilities,
    LLMClient,
    LLMRequest,
    LLMResult,
    PromptSegment,
    TokenLP,
    Usage,
)
from switchboard.router import ClientSpec, ConfidenceSpec, Router, ShortlistSpec
from switchboard.telemetry.emitter import (
    CallbackSink,
    ContentMode,
    DecisionSink,
    InMemorySink,
    JSONLSink,
    MultiSink,
    NoopSink,
    QueuedSink,
)

# Groups, for orientation (the list itself is sorted, per lint):
#   catalog    - Route, Registry, RegistryView
#   request    - RequestContext
#   router     - Router, ClientSpec, ShortlistSpec, ConfidenceSpec, RetrySpec, ThresholdPolicy
#   decision   - Decision + the five variants, RoutedCall, PlanStep, AbstainReason,
#                ConfidenceReport, DecisionKind, DecisionPath
#   retrieval  - Candidate, ShortlistResult
#   audit      - AuditRecord, LatencyBlock, CostBlock
#   sinks      - DecisionSink, InMemorySink, JSONLSink, CallbackSink, MultiSink,
#                QueuedSink, NoopSink, ContentMode
#   providers  - LLMClient, AsyncLLMClient, LLMRequest, LLMResult, PromptSegment,
#                ClientCapabilities, TokenLP, Usage
#   errors     - the full §3.8 tree
__all__ = [
    "AbstainDecision",
    "AbstainReason",
    "AsyncLLMClient",
    "AuditRecord",
    "CallbackSink",
    "Candidate",
    "ClarifyDecision",
    "ClientCapabilities",
    "ClientSpec",
    "ConfidenceReport",
    "ConfidenceSpec",
    "ConfigError",
    "ContentMode",
    "CostBlock",
    "Decision",
    "DecisionKind",
    "DecisionPath",
    "DecisionSink",
    "InMemorySink",
    "JSONLSink",
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "LatencyBlock",
    "MissingDependencyError",
    "MultiRouteDecision",
    "MultiSink",
    "NoopSink",
    "PlanDecision",
    "PlanStep",
    "PromptSegment",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimit",
    "ProviderTimeout",
    "QueuedSink",
    "Registry",
    "RegistryError",
    "RegistryView",
    "RequestContext",
    "RetrySpec",
    "Route",
    "RouteDecision",
    "RoutedCall",
    "Router",
    "ShortlistResult",
    "ShortlistSpec",
    "SwitchboardError",
    "ThresholdPolicy",
    "TokenLP",
    "Usage",
    "__version__",
]
