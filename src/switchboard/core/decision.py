"""The ``Decision`` discriminated union — switchboard's return contract (plan §3.4).

Four kinds ship in v0.1 (``route``, ``multi_route``, ``clarify``, ``abstain``);
``plan`` is defined here but only produced at [v0.5]. Downstream code branches on
``decision.kind`` **alone** — that is the whole point of the union:

* clarify and abstain are *results*, not exceptions (§3, decision (c));
* a configured fallback arrives as ``kind="route"`` with
  ``decision_path="fallback"``, never as a special abstain variant
  (§13 ruling #5), so a caller's ``route`` branch needs no fallback handling;
* an empty entitled candidate set degrades to ``abstain("no_eligible_routes")``
  with **no LLM call**.

``confidence`` and ``audit`` are library-attached, not model-emitted: the wire
schema the LLM fills in is generated per call and is a different, smaller thing
(§3, decision (b); §4.4).

:data:`AbstainReason` and :class:`ConfidenceReport` are defined in
``core.audit`` and re-exported here — this module is their documented import
path (§3.4), but ``AuditRecord`` needs them as field types while
``_DecisionBase`` needs ``AuditRecord``, so the definitions live on the audit
side of that edge.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, TypeAdapter

from switchboard.core.audit import (
    AbstainReason,
    AuditRecord,
    ConfidenceReport,
    DecisionKind,
    DecisionPath,
)

__all__ = [
    "AbstainDecision",
    "AbstainReason",
    "ClarifyDecision",
    "ConfidenceReport",
    "Decision",
    "DecisionAdapter",
    "DecisionKind",
    "DecisionPath",
    "MultiRouteDecision",
    "PlanDecision",
    "PlanStep",
    "RouteDecision",
    "RoutedCall",
]

_ArgsModel = SerializeAsAny[BaseModel] | None
"""``args`` is an instance of the chosen route's ``args_model`` — an arbitrary
``BaseModel`` subclass. ``SerializeAsAny`` makes Pydantic serialise it with the
*runtime* type's serialiser; without it, ``model_dump()`` would serialise
against the bare ``BaseModel`` schema and silently emit ``{}``, breaking the
Google ADK integration in §3.7 (``d.model_dump(mode="json", exclude={"audit"})``).
"""


class _DecisionBase(BaseModel):
    """Fields common to every decision kind (plan §3.4)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    rationale: str
    """Free-text reasoning. Emitted by the model *before* the committed route
    token — the reason-then-commit ordering the evidence supports (§3.4, §4.4)."""
    confidence: ConfidenceReport
    """Library-computed, not model-stated (§6.3)."""
    audit: AuditRecord
    """The canonical record (§8.2): OTel span payload and distillation row."""
    decision_path: DecisionPath = "llm"
    downgraded_from: str | None = None
    """e.g. ``"route"`` when thresholds downgraded this to clarify. Thresholds are
    downgrade-only — they never upgrade a decision (§6.4)."""

    def model_dump_public(self, **kwargs: Any) -> dict[str, Any]:
        """JSON-mode dump with the audit record excluded.

        The projection framework integrations want: §3.7's Google ADK tool does
        ``d.model_dump(mode="json", exclude={"audit"})`` because the audit record
        belongs in the telemetry pipeline, not in an agent's context window.
        """
        kwargs.setdefault("mode", "json")
        exclude = kwargs.pop("exclude", None)
        exclude = {"audit"} if exclude is None else set(exclude) | {"audit"}
        return self.model_dump(exclude=exclude, **kwargs)


class RoutedCall(BaseModel):
    """One (route, args) pair inside a :class:`MultiRouteDecision` (plan §3.4)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    route: str
    args: _ArgsModel = None
    """An instance of that route's ``args_model``, fully validated."""


class RouteDecision(_DecisionBase):
    """A single committed route (plan §3.4).

    Also the shape a configured fallback takes, with ``decision_path="fallback"``
    and ``args=None``; the pre-fallback outcome and ``abstain_reason`` are
    preserved in :attr:`~_DecisionBase.audit` (§6.6, §13 ruling #5).
    """

    kind: Literal["route"] = "route"
    route: str
    """Guaranteed to be a member of the entitled candidate set — re-checked by the
    validator even under a ``Literal`` wire enum, as defence in depth (§4.5)."""
    args: _ArgsModel = None


class MultiRouteDecision(_DecisionBase):
    """Several routes to invoke for one query (plan §3.4)."""

    kind: Literal["multi_route"] = "multi_route"
    routes: tuple[RoutedCall, ...]
    """Order-independent and parallel-safe — switchboard expresses no ordering
    here; sequencing with dependencies is what ``PlanDecision`` is for."""


class ClarifyDecision(_DecisionBase):
    """The router is asking the user a question instead of guessing (plan §3.4).

    Reached three ways: the model elected it in-schema, an args-only validation
    failure downgraded a sound route choice, or fused confidence fell below
    ``clarify_below`` (§3.8). Clarifying is cheaper than mis-routing (§9.2).
    """

    kind: Literal["clarify"] = "clarify"
    question: str
    candidates: tuple[str, ...] = ()
    """Routes it was torn between; enum-constrained in-schema to the entitled set,
    so the model cannot offer a route the tenant may not see (§6.5)."""
    missing: tuple[str, ...] = ()
    """Arg fields it could not extract."""
    resume_token: str | None = None
    """Opaque token to re-enter with the prior shortlist pinned (§6.5). [v0.2]"""


class AbstainDecision(_DecisionBase):
    """The router declined to route (plan §3.4).

    Returned only when no fallback is configured, or the configured fallback route
    is not entitled for this context — entitlement is never bypassed (§6.6).
    """

    kind: Literal["abstain"] = "abstain"
    reason: AbstainReason


class PlanStep(BaseModel):
    """One step of a multi-step plan (plan §3.4). [v0.5]"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    route: str
    args: _ArgsModel = None
    depends_on: tuple[int, ...] = ()
    """Indices of prerequisite steps."""


class PlanDecision(_DecisionBase):
    """An ordered, dependency-aware plan (plan §3.4). [v0.5]

    Defined in v0.1 so the union is closed and callers' ``match`` statements are
    exhaustive from day one; only produced when ``Router(allow_plan=True)``,
    which is a [v0.5] capability.
    """

    kind: Literal["plan"] = "plan"
    steps: tuple[PlanStep, ...]
    """Topologically ordered."""


Decision = Annotated[
    RouteDecision | MultiRouteDecision | ClarifyDecision | AbstainDecision | PlanDecision,
    Field(discriminator="kind"),
]
"""The return type of ``router.route()`` / ``router.aroute()`` (plan §3.4, §3.6)."""

DecisionAdapter: TypeAdapter[Decision] = TypeAdapter(Decision)
"""Parse/serialise a :data:`Decision` without a wrapper model.

Used by the eval harness replay cache and by sinks that round-trip decisions.
Note that ``args`` round-trips lossily through JSON — the concrete ``args_model``
type is not recoverable from the payload alone; re-validate against the registry
when the typed instance is needed.
"""
