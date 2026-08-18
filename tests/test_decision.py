"""Contract tests for the ``Decision`` discriminated union (plan §3.4).

This union *is* switchboard's public return contract. Downstream code branches on
``decision.kind`` alone — the LangGraph, FastAPI and ADK snippets in §3.7 all do
exactly that — so the properties asserted here are what those integrations stand
on:

* the discriminator dispatches to the right concrete class, in both directions;
* ``args`` is a **real, validated instance** of the route's ``args_model``, and
  serialises with the runtime type's fields rather than as an empty ``{}``;
* ``model_dump_public()`` never leaks the audit record into an agent's context
  window;
* ``AbstainReason`` is a **closed** enum — the audit pipeline, the distillation
  filters and the eval fixtures all key on it, so an unknown reason must fail
  validation rather than silently propagate.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from switchboard import (
    AbstainDecision,
    AuditRecord,
    ClarifyDecision,
    ConfidenceReport,
    MultiRouteDecision,
    PlanDecision,
    PlanStep,
    RoutedCall,
    RouteDecision,
)
from switchboard.core.audit import AuditDraft
from switchboard.core.decision import AbstainReason, DecisionAdapter, DecisionKind, DecisionPath

ABSTAIN_REASONS = get_args(AbstainReason)
DECISION_KINDS = get_args(DecisionKind)
DECISION_PATHS = get_args(DecisionPath)


class RefundArgs(BaseModel):
    """A concrete args model — the thing ``args`` is an instance of."""

    order_id: str
    reason: str | None = None


@pytest.fixture
def audit() -> AuditRecord:
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", kind="route", routes=["refund"])
    draft.set_inputs("refund order A-123")
    return draft.finalize()


@pytest.fixture
def confidence() -> ConfidenceReport:
    return ConfidenceReport(score=0.91, method="logprobs", p_route=0.91, thresholds={"clarify": 0.55})


def _common(audit: AuditRecord, confidence: ConfidenceReport) -> dict[str, Any]:
    return {"rationale": "The query names an order and asks for money back.",
            "confidence": confidence,
            "audit": audit}


# --------------------------------------------------------------------------- #
# The closed reason vocabulary (plan §3.4, §13 ruling #4)
# --------------------------------------------------------------------------- #


def test_abstain_reason_vocabulary_is_exactly_the_plans_seven_members() -> None:
    assert set(ABSTAIN_REASONS) == {
        "no_eligible_routes",
        "unparseable_output",
        "invalid_route_reference",
        "invalid_args",
        "low_confidence",
        "provider_error",
        "model_elected",
    }


@pytest.mark.parametrize("reason", ABSTAIN_REASONS)
def test_every_documented_abstain_reason_is_accepted(
    reason: str, audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = AbstainDecision(**_common(audit, confidence), reason=reason)
    assert decision.reason == reason


@pytest.mark.parametrize(
    "reason",
    ["NO_ELIGIBLE_ROUTES", "no_eligible_route", "timeout", "", None, 0, "unknown"],
)
def test_an_unknown_abstain_reason_fails_validation(
    reason: Any, audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """Adding a member here is a schema-breaking change (§10.1) precisely because

    downstream filters key on the closed set. A typo must not become a new value.
    """
    with pytest.raises(ValidationError):
        AbstainDecision(**_common(audit, confidence), reason=reason)


def test_abstain_reason_is_required(audit: AuditRecord, confidence: ConfidenceReport) -> None:
    with pytest.raises(ValidationError):
        AbstainDecision(**_common(audit, confidence))


# --------------------------------------------------------------------------- #
# Kind discriminators and construction
# --------------------------------------------------------------------------- #


def test_the_union_covers_exactly_the_five_declared_kinds() -> None:
    assert set(DECISION_KINDS) == {"route", "multi_route", "clarify", "abstain", "plan"}


def test_each_class_pins_its_own_discriminator(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    common = _common(audit, confidence)
    assert RouteDecision(**common, route="refund").kind == "route"
    assert MultiRouteDecision(**common, routes=()).kind == "multi_route"
    assert ClarifyDecision(**common, question="Which order?").kind == "clarify"
    assert AbstainDecision(**common, reason="low_confidence").kind == "abstain"
    assert PlanDecision(**common, steps=()).kind == "plan"


def test_kind_cannot_be_overridden_to_another_variants_value(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    with pytest.raises(ValidationError):
        RouteDecision(**_common(audit, confidence), route="refund", kind="clarify")


def test_every_decision_is_frozen(audit: AuditRecord, confidence: ConfidenceReport) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    for field, value in (("route", "other"), ("rationale", "x"), ("decision_path", "fallback")):
        with pytest.raises(ValidationError):
            setattr(decision, field, value)


def test_rationale_confidence_and_audit_are_required_on_every_kind() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AbstainDecision(reason="low_confidence")
    missing = {error["loc"][0] for error in excinfo.value.errors()}
    assert {"rationale", "confidence", "audit"} <= missing


# --------------------------------------------------------------------------- #
# Round-tripping through the discriminated union (plan §3.4)
# --------------------------------------------------------------------------- #


def _all_kinds(audit: AuditRecord, confidence: ConfidenceReport) -> list[Any]:
    common = _common(audit, confidence)
    return [
        RouteDecision(**common, route="refund"),
        MultiRouteDecision(
            **common, routes=(RoutedCall(route="refund"), RoutedCall(route="track_order"))
        ),
        ClarifyDecision(
            **common,
            question="Which order do you mean?",
            candidates=("refund", "track_order"),
            missing=("order_id",),
        ),
        AbstainDecision(**common, reason="low_confidence"),
        PlanDecision(
            **common,
            steps=(PlanStep(route="track_order"), PlanStep(route="refund", depends_on=(0,))),
        ),
    ]


def test_the_adapter_dispatches_every_kind_back_to_its_own_class(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    for original in _all_kinds(audit, confidence):
        parsed = DecisionAdapter.validate_python(original.model_dump(mode="json"))
        assert type(parsed) is type(original), original.kind
        assert parsed.kind == original.kind


def test_each_kind_round_trips_field_for_field(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """``args`` is deliberately excluded from this assertion: the module docstring

    documents that it round-trips lossily (the concrete ``args_model`` type is not
    recoverable from the payload). Everything else must survive exactly.
    """
    for original in _all_kinds(audit, confidence):
        dumped = original.model_dump(mode="json")
        assert DecisionAdapter.validate_python(dumped).model_dump(mode="json") == dumped


def test_the_adapter_round_trips_through_json_bytes(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """Sinks and the eval replay cache go through JSON, not through Python objects."""
    for original in _all_kinds(audit, confidence):
        parsed = DecisionAdapter.validate_json(DecisionAdapter.dump_json(original))
        assert parsed.kind == original.kind
        assert parsed.audit.decision_id == audit.decision_id


def test_the_audit_record_survives_the_round_trip(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    parsed = DecisionAdapter.validate_python(decision.model_dump(mode="json"))
    assert parsed.audit.registry_version == "a1b2c3d4e5f6"
    assert parsed.audit.inputs_hash == audit.inputs_hash
    assert parsed.confidence.score == pytest.approx(0.91)


def test_an_unknown_kind_is_rejected_by_the_discriminator(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    payload = RouteDecision(**_common(audit, confidence), route="refund").model_dump(mode="json")
    payload["kind"] = "execute"
    with pytest.raises(ValidationError) as excinfo:
        DecisionAdapter.validate_python(payload)
    assert excinfo.value.errors()[0]["type"] == "union_tag_invalid"


def test_a_missing_kind_is_rejected_by_the_discriminator(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    payload = RouteDecision(**_common(audit, confidence), route="refund").model_dump(mode="json")
    del payload["kind"]
    with pytest.raises(ValidationError):
        DecisionAdapter.validate_python(payload)


# --------------------------------------------------------------------------- #
# args carries a real BaseModel instance (plan §3.4; SerializeAsAny)
# --------------------------------------------------------------------------- #


def test_args_holds_the_concrete_args_model_instance(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    args = RefundArgs(order_id="A-123", reason="damaged")
    decision = RouteDecision(**_common(audit, confidence), route="refund", args=args)
    assert decision.args is args
    assert isinstance(decision.args, RefundArgs)
    assert decision.args.order_id == "A-123"


def test_args_serialises_with_the_runtime_types_fields_not_an_empty_object(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """Without ``SerializeAsAny`` Pydantic would serialise against the *declared*

    bare ``BaseModel`` and emit ``{}`` — silently breaking §3.7's ADK integration,
    which ships ``d.model_dump(mode="json", exclude={"audit"})`` to the agent.
    """
    decision = RouteDecision(
        **_common(audit, confidence),
        route="refund",
        args=RefundArgs(order_id="A-123", reason="damaged"),
    )
    assert decision.model_dump(mode="json")["args"] == {"order_id": "A-123", "reason": "damaged"}
    assert decision.model_dump_public()["args"] == {"order_id": "A-123", "reason": "damaged"}


def test_args_defaults_to_none_for_argless_routes(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    assert decision.args is None
    assert decision.model_dump(mode="json")["args"] is None


def test_routed_call_and_plan_step_carry_args_the_same_way() -> None:
    args = RefundArgs(order_id="A-1")
    call = RoutedCall(route="refund", args=args)
    step = PlanStep(route="refund", args=args, depends_on=(0, 1))
    assert call.args is args
    assert step.args is args
    assert step.depends_on == (0, 1)
    assert call.model_dump(mode="json")["args"] == {"order_id": "A-1", "reason": None}
    assert step.model_dump(mode="json")["args"] == {"order_id": "A-1", "reason": None}


def test_multi_route_routes_is_an_immutable_tuple_of_routed_calls(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = MultiRouteDecision(
        **_common(audit, confidence),
        routes=[{"route": "refund"}, {"route": "track_order"}],
    )
    assert isinstance(decision.routes, tuple)
    assert all(isinstance(call, RoutedCall) for call in decision.routes)
    assert [call.route for call in decision.routes] == ["refund", "track_order"]


def test_clarify_tuple_fields_default_to_empty_and_coerce_from_lists(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    bare = ClarifyDecision(**_common(audit, confidence), question="Which order?")
    assert bare.candidates == () and bare.missing == () and bare.resume_token is None

    filled = ClarifyDecision(
        **_common(audit, confidence),
        question="Which order?",
        candidates=["refund", "track_order"],
        missing=["order_id"],
    )
    assert filled.candidates == ("refund", "track_order")
    assert filled.missing == ("order_id",)


# --------------------------------------------------------------------------- #
# model_dump_public() — the audit record never reaches an agent's context (§3.7)
# --------------------------------------------------------------------------- #


def test_model_dump_public_excludes_the_audit_record(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    public = decision.model_dump_public()
    assert "audit" not in public
    assert "audit" in decision.model_dump()


def test_model_dump_public_keeps_everything_else(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    public = decision.model_dump_public()
    assert public["kind"] == "route"
    assert public["route"] == "refund"
    assert public["decision_path"] == "llm"
    assert public["rationale"].startswith("The query names an order")
    assert public["confidence"]["score"] == pytest.approx(0.91)


def test_model_dump_public_defaults_to_json_mode(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """JSON mode is what the framework integrations need — the value must be

    directly serialisable, not a graph of Python objects.
    """
    import json

    decision = RouteDecision(
        **_common(audit, confidence), route="refund", args=RefundArgs(order_id="A-1")
    )
    json.dumps(decision.model_dump_public())  # would raise on a non-JSON value
    assert isinstance(decision.model_dump_public()["confidence"], dict)


def test_model_dump_public_merges_a_caller_supplied_exclude(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """A caller narrowing the payload must not accidentally re-admit ``audit``."""
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    public = decision.model_dump_public(exclude={"rationale"})
    assert "audit" not in public
    assert "rationale" not in public
    assert public["route"] == "refund"


def test_model_dump_public_forwards_other_kwargs(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    public = decision.model_dump_public(exclude_none=True)
    assert "audit" not in public
    assert "downgraded_from" not in public


@pytest.mark.parametrize("index", range(5))
def test_model_dump_public_works_on_every_kind(
    index: int, audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = _all_kinds(audit, confidence)[index]
    public = decision.model_dump_public()
    assert "audit" not in public
    assert public["kind"] == decision.kind


# --------------------------------------------------------------------------- #
# ConfidenceReport (plan §3.4, §6.3)
# --------------------------------------------------------------------------- #


def test_confidence_report_defaults() -> None:
    report = ConfidenceReport(score=0.7, method="logprobs")
    assert report.score == pytest.approx(0.7)
    assert report.method == "logprobs"
    assert report.p_route is None
    assert report.margin is None
    assert report.agreement is None
    assert report.stated is None
    assert report.vote_overturned is False
    assert report.thresholds == {}


def test_confidence_report_thresholds_default_is_not_shared_between_instances() -> None:
    first = ConfidenceReport(score=0.5, method="none")
    first.thresholds["clarify"] = 0.55
    assert ConfidenceReport(score=0.5, method="none").thresholds == {}


def test_confidence_report_requires_score_and_method() -> None:
    with pytest.raises(ValidationError):
        ConfidenceReport(score=0.5)
    with pytest.raises(ValidationError):
        ConfidenceReport(method="none")


def test_confidence_report_carries_the_v02_signals_when_supplied() -> None:
    report = ConfidenceReport(
        score=0.62,
        method="logprobs+vote:n=3",
        p_route=0.81,
        margin=0.12,
        agreement=0.67,
        vote_overturned=True,
        stated=0.9,
        thresholds={"clarify": 0.55, "abstain": 0.30},
    )
    assert report.agreement == pytest.approx(0.67)
    assert report.vote_overturned is True
    assert report.thresholds["abstain"] == pytest.approx(0.30)


def test_the_same_confidence_object_appears_on_the_decision_and_in_the_audit(
    confidence: ConfidenceReport,
) -> None:
    """§6.3: one confidence model, so recalibration and distillation see the raw

    evidence rather than only the fused scalar.
    """
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", kind="route", routes=["refund"])
    draft.confidence = confidence
    record = draft.finalize()
    decision = RouteDecision(
        rationale="r", confidence=confidence, audit=record, route="refund"
    )
    assert decision.confidence is decision.audit.confidence is confidence


# --------------------------------------------------------------------------- #
# decision_path / downgraded_from (plan §3.4, §6.4, §6.6)
# --------------------------------------------------------------------------- #


def test_decision_path_defaults_to_llm(audit: AuditRecord, confidence: ConfidenceReport) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund")
    assert decision.decision_path == "llm"
    assert decision.downgraded_from is None


def test_decision_path_vocabulary_is_closed_to_the_three_documented_values() -> None:
    assert set(DECISION_PATHS) == {"llm", "distilled", "fallback"}


@pytest.mark.parametrize("path", DECISION_PATHS)
def test_every_documented_decision_path_is_accepted(
    path: str, audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = RouteDecision(**_common(audit, confidence), route="refund", decision_path=path)
    assert decision.decision_path == path


@pytest.mark.parametrize("path", ["LLM", "cache", "", None, "distill"])
def test_an_unknown_decision_path_fails_validation(
    path: Any, audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    with pytest.raises(ValidationError):
        RouteDecision(**_common(audit, confidence), route="refund", decision_path=path)


def test_a_fallback_arrives_as_a_route_decision_not_a_special_abstain(
    confidence: ConfidenceReport,
) -> None:
    """§13 ruling #5 / §6.6: one shape, everywhere. The caller's ``route`` branch

    needs no fallback handling; the pre-fallback reason lives in the audit record.
    """
    draft = AuditDraft(
        registry_version="a1b2c3d4e5f6",
        kind="route",
        routes=["human_handoff"],
        decision_path="fallback",
        abstain_reason="low_confidence",
    )
    record = draft.finalize()
    decision = RouteDecision(
        rationale="Confidence below the abstain threshold; falling back.",
        confidence=confidence,
        audit=record,
        route="human_handoff",
        decision_path="fallback",
    )
    assert decision.kind == "route"
    assert decision.args is None
    assert decision.decision_path == "fallback"
    assert decision.audit.abstain_reason == "low_confidence"


def test_downgraded_from_records_the_pre_threshold_outcome(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    """§6.4: thresholds are downgrade-only. ``downgraded_from`` is the breadcrumb

    that says what the model actually committed to before the policy stage.
    """
    decision = ClarifyDecision(
        **_common(audit, confidence),
        question="Which order do you mean?",
        candidates=("refund", "track_order"),
        downgraded_from="route",
    )
    assert decision.kind == "clarify"
    assert decision.downgraded_from == "route"
    assert decision.model_dump_public()["downgraded_from"] == "route"


def test_downgraded_from_survives_the_round_trip(
    audit: AuditRecord, confidence: ConfidenceReport
) -> None:
    decision = AbstainDecision(
        **_common(audit, confidence), reason="low_confidence", downgraded_from="route"
    )
    parsed = DecisionAdapter.validate_python(decision.model_dump(mode="json"))
    assert parsed.downgraded_from == "route"
    assert parsed.decision_path == "llm"
