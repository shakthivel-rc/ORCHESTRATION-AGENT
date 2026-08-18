"""Validation, confidence and policy — the three stages after the model speaks.

Plan §4.5 (validate), §6 (confidence, fusion, thresholds) and §3.8/§6.6 (the
degradation table and fallback) form one chain: parse → check the reference →
check the args → score → apply downgrade-only thresholds → substitute a fallback.
The properties this module pins are the ones that would fail *silently* if they
regressed:

* **tolerance without laxity** — fenced or prose-wrapped JSON is recovered, but
  whatever is recovered is still validated against the per-call wire model, so
  the contract is identical on every rung of §4.3's ladder;
* **``p_route`` never guesses** — a failed token alignment returns ``None``, and
  ``None`` makes the threshold machinery inert rather than feeding it a
  fabricated number (§6.2);
* **fusion is conservative** — a ``min``, verbalized damped x0.85, overturn
  penalised x0.8, so adding a signal can only lower confidence (§6.3);
* **the no-signal rule** — verbalized-only confidence does *not* downgrade
  anything unless the caller explicitly opts in, because acting on the one
  signal the evidence distrusts would invert the report's Recommendation #4
  (§6.3);
* **thresholds only ever downgrade** — nothing here can promote a clarify into a
  commitment (§6.4); the single ``abstain -> route`` transition in the library is
  fallback *substitution*, which marks itself as such (§6.6, §13 ruling #5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from conftest import NO_SIGNAL, wire_output
from switchboard import (
    AbstainDecision,
    ClarifyDecision,
    ConfidenceReport,
    Route,
    RouteDecision,
    TokenLP,
)
from switchboard.core.audit import AuditDraft
from switchboard.engine.confidence import (
    build_confidence_report,
    compute_margin,
    compute_p_route,
    default_fusion,
    signals_are_actionable,
)
from switchboard.engine.policy import (
    ThresholdPolicy,
    abstain,
    apply_fallback,
    resolve_decision,
    route_index,
    synthesize_clarify_question,
)
from switchboard.engine.schema import build_wire_schema
from switchboard.engine.validate import (
    check_route_reference,
    format_error_for_repair,
    parse_wire_output,
    validate_args,
)
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from switchboard import Registry, Router

CANDIDATES = ("refund", "track_order", "human_handoff")


class OrderArgs(BaseModel):
    """Two required fields and one optional — enough to produce a missing list."""

    order_id: str
    amount: int
    note: str | None = None


@pytest.fixture
def wire_model() -> type[BaseModel]:
    """The per-call wire model in ``static`` mode (route is a plain string)."""
    return build_wire_schema(CANDIDATES, mode="static")


@pytest.fixture
def policy_routes() -> dict[str, Route]:
    """``{name: Route}`` over the entitled set, as :func:`resolve_decision` takes it."""
    return route_index(
        [
            Route(
                name="refund",
                description="Issue or check a refund for an order.",
                args_model=OrderArgs,
                clarify_label="a refund",
            ),
            Route(
                name="track_order",
                description="Track shipment status for an existing order.",
                clarify_label="tracking an order",
            ),
            Route(name="human_handoff", description="Escalate to a human agent.", pinned=True),
        ]
    )


def _stream(pairs: Sequence[tuple[str, float]]) -> list[TokenLP]:
    """Build a token stream from ``(token text, probability)`` pairs.

    The concatenation of the token texts is the model's raw output, which is the
    only property ``engine/confidence.py``'s alignment depends on (§6.2). Written
    out by hand rather than generated so the expected geometric mean below is
    arithmetic anyone can check.
    """
    return [
        TokenLP(token=text, logprob=math.log(prob), top=[(text, math.log(prob))])
        for text, prob in pairs
    ]


# --------------------------------------------------------------------------- #
# Layer 1 — tolerant schema parsing (plan §4.5)
# --------------------------------------------------------------------------- #


def test_parses_a_markdown_fenced_payload(wire_model: type[BaseModel]) -> None:
    """A ```` ```json ```` fence is what the ``structured="none"`` rung produces (§4.3)."""
    parsed, error = parse_wire_output(
        '```json\n{"rationale": "r", "kind": "route", "route": "refund"}\n```',
        wire_model,
    )

    assert error is None
    assert parsed is not None
    assert parsed.route == "refund"


def test_parses_a_payload_buried_in_prose(wire_model: type[BaseModel]) -> None:
    """Leading and trailing chatter is stripped by the balanced-brace scan (§4.5)."""
    parsed, error = parse_wire_output(
        'Sure! Here is the decision:\n'
        '{"rationale": "the request names an order", "kind": "route", '
        '"route": "track_order", "args": {"order_id": "A-1"}}\n'
        'Let me know if that helps.',
        wire_model,
    )

    assert error is None
    assert parsed is not None
    assert parsed.route == "track_order"
    assert parsed.args == {"order_id": "A-1"}


def test_a_brace_inside_the_rationale_does_not_truncate_the_object(
    wire_model: type[BaseModel],
) -> None:
    """Brace counting respects JSON string literals and escapes (plan §4.5)."""
    parsed, error = parse_wire_output(
        '{"rationale": "the user wrote {} and \\" in their message", '
        '"kind": "route", "route": "refund"}',
        wire_model,
    )

    assert error is None
    assert parsed is not None
    assert parsed.route == "refund"


def test_tolerance_is_not_laxity(wire_model: type[BaseModel]) -> None:
    """Whatever is extracted is still validated against the wire model (§4.5)."""
    parsed, error = parse_wire_output('{"kind": "route", "route": "refund"}', wire_model)

    assert parsed is None
    assert error is not None
    assert error.stage == "schema"
    assert "rationale" in error.missing


def test_unparseable_output_is_a_value_not_an_exception(
    wire_model: type[BaseModel],
) -> None:
    """"A model that emitted unusable output is a healthy system" (§3.8, §13 #2)."""
    parsed, error = parse_wire_output("I'm afraid I can't do that.", wire_model)

    assert parsed is None
    assert error is not None and error.stage == "schema"

    parsed, error = parse_wire_output(None, wire_model)
    assert parsed is None
    assert error is not None and error.stage == "schema"


# --------------------------------------------------------------------------- #
# Layer 2 — route reference (plan §4.5)
# --------------------------------------------------------------------------- #


def test_route_reference_rejects_a_non_candidate(wire_model: type[BaseModel]) -> None:
    """"``route ∈`` the entitlement-filtered candidate set", checked always (§4.5).

    Under ``static`` mode the prompt is the only thing pointing at the candidate
    list, and a prompt is not an enforcement mechanism. This check is.
    """
    parsed, _ = parse_wire_output(
        '{"rationale": "r", "kind": "route", "route": "billing_report"}', wire_model
    )
    assert parsed is not None

    error = check_route_reference(parsed, set(CANDIDATES))

    assert error is not None
    assert error.stage == "route_ref"
    assert error.detail["unknown"] == ["billing_report"]
    assert error.detail["allowed"] == sorted(CANDIDATES)


def test_route_reference_accepts_a_candidate(wire_model: type[BaseModel]) -> None:
    parsed, _ = parse_wire_output(
        '{"rationale": "r", "kind": "route", "route": "refund"}', wire_model
    )
    assert parsed is not None
    assert check_route_reference(parsed, set(CANDIDATES)) is None


def test_a_clarify_can_never_offer_an_unentitled_route(
    wire_model: type[BaseModel],
) -> None:
    """The clarify arm's ``candidates`` are re-checked too (plan §6.5, §7.4)."""
    parsed, _ = parse_wire_output(
        '{"rationale": "r", "kind": "clarify", "question": "Which?", '
        '"candidates": ["refund", "billing_report"]}',
        wire_model,
    )
    assert parsed is not None

    error = check_route_reference(parsed, set(CANDIDATES))

    assert error is not None
    assert error.detail["unknown"] == ["billing_report"]


def test_kind_route_with_no_route_is_a_reference_failure(
    wire_model: type[BaseModel],
) -> None:
    parsed, _ = parse_wire_output('{"rationale": "r", "kind": "route"}', wire_model)
    assert parsed is not None

    error = check_route_reference(parsed, set(CANDIDATES))

    assert error is not None
    assert error.stage == "route_ref"
    assert error.missing == ("route",)


def test_repair_text_names_the_legal_alternatives() -> None:
    """The repair segment states what *is* allowed, not just what was wrong (§4.5)."""
    parsed = build_wire_schema(CANDIDATES, mode="static").model_validate(
        {"rationale": "r", "kind": "route", "route": "billing_report"}
    )
    error = check_route_reference(parsed, set(CANDIDATES))
    assert error is not None

    text = format_error_for_repair(error)

    assert "billing_report" in text
    assert "refund" in text
    assert "clarify" in text and "abstain" in text


# --------------------------------------------------------------------------- #
# Layer 3 — args (plan §4.5, §3.8 row 4)
# --------------------------------------------------------------------------- #


def test_args_validation_reports_the_missing_fields(
    policy_routes: Mapping[str, Route],
) -> None:
    """The missing list is what §3.8 row 4 turns into a question for the user."""
    instance, error = validate_args({"order_id": "A-1"}, policy_routes["refund"])

    assert instance is None
    assert error is not None
    assert error.stage == "args"
    assert error.missing == ("amount",)
    assert error.detail["route"] == "refund"
    assert "amount" in error.message


def test_absent_args_are_validated_as_empty_not_short_circuited(
    policy_routes: Mapping[str, Route],
) -> None:
    """``None`` args on a route with required fields is a *missing* list (§4.5).

    Short-circuiting would silently return "no args" for a route that needs two,
    and the caller would execute it with nothing.
    """
    instance, error = validate_args(None, policy_routes["refund"])

    assert instance is None
    assert error is not None
    assert set(error.missing) == {"order_id", "amount"}


def test_valid_args_come_back_typed(policy_routes: Mapping[str, Route]) -> None:
    instance, error = validate_args(
        {"order_id": "A-1", "amount": 42}, policy_routes["refund"]
    )

    assert error is None
    assert isinstance(instance, OrderArgs)
    assert instance.order_id == "A-1"
    assert instance.amount == 42


def test_spurious_args_on_an_argless_route_are_noise_not_errors(
    policy_routes: Mapping[str, Route],
) -> None:
    """"A spurious argument on an argument-less route is noise" (plan §4.5)."""
    instance, error = validate_args({"whatever": 1}, policy_routes["track_order"])

    assert (instance, error) == (None, None)


# --------------------------------------------------------------------------- #
# Signal 1 — p_route (plan §6.2)
# --------------------------------------------------------------------------- #


def test_p_route_is_the_geometric_mean_of_the_committed_tokens() -> None:
    """``p_route = exp(mean(logprob(t) for t in route_name_tokens))`` (plan §6.2).

    The four route tokens carry 0.9, 0.8, 0.7 and 0.6. The answer is their
    geometric mean (0.7416) — not their arithmetic mean (0.75), and emphatically
    not their product (0.3024), which would punish a multi-token route name for
    being long.
    """
    tokens = _stream(
        [
            ('{"rationale":"r","kind":"route","route":"', 0.99),
            ("ref", 0.9),
            ("und", 0.8),
            ("_", 0.7),
            ("order", 0.6),
            ('","args":null}', 0.99),
        ]
    )

    assert compute_p_route(tokens, "refund_order") == pytest.approx(
        (0.9 * 0.8 * 0.7 * 0.6) ** 0.25
    )


def test_p_route_is_length_normalised() -> None:
    """"So ``escalate_to_human_agent`` is not punished for being longer" (§6.2)."""
    short = _stream(
        [('{"route":"', 0.99), ("a", 0.9), ("b", 0.9), ('","x":1}', 0.99)]
    )
    long = _stream(
        [
            ('{"route":"', 0.99),
            *[(letter, 0.9) for letter in "abcdef"],
            ('","x":1}', 0.99),
        ]
    )

    assert compute_p_route(short, "ab") == pytest.approx(0.9)
    assert compute_p_route(long, "abcdef") == pytest.approx(0.9)


def test_p_route_measures_the_commitment_not_the_rationale() -> None:
    """Anchored on the ``"route"`` key (plan §6.2).

    The rationale precedes the commitment and routinely names routes; an
    unanchored search would happily measure the model's musings instead of its
    decision. Here the rationale mentions the route at 0.99 and the commitment
    sits at 0.5 — the answer must be 0.5.
    """
    tokens = _stream(
        [
            ('{"rationale":"considered \\"', 0.99),
            ("refund_order", 0.99),
            ('\\" first","kind":"route","route":"', 0.99),
            ("refund_order", 0.5),
            ('","args":null}', 0.99),
        ]
    )

    assert compute_p_route(tokens, "refund_order") == pytest.approx(0.5)


def test_p_route_is_none_when_alignment_fails() -> None:
    """"A wrong ``p_route`` is worse than no ``p_route``" (plan §6.2)."""
    tokens = _stream([('{"route":"', 0.99), ("refund", 0.9), ('"}', 0.99)])

    assert compute_p_route(tokens, "some_other_route") is None
    assert compute_p_route(None, "refund") is None
    assert compute_p_route([], "refund") is None
    assert compute_p_route(tokens, None) is None


def test_a_none_p_route_makes_the_thresholds_inert() -> None:
    """The consequence of the previous test, stated as the rule (plan §6.3)."""
    report = build_confidence_report(token_logprobs=None, route_name="refund")

    assert report.p_route is None
    assert report.method == "none"
    assert signals_are_actionable(report) is False


def test_avg_logprob_is_the_documented_coarse_fallback() -> None:
    """Gemini's ``avg_logprobs``, marked as such so nothing mistakes it (§6.2)."""
    report = build_confidence_report(
        token_logprobs=None, route_name="refund", avg_logprob=math.log(0.8)
    )

    assert report.p_route == pytest.approx(0.8)
    assert report.method == "logprobs:avg"


# --------------------------------------------------------------------------- #
# Signal 2 — margin [v0.2] (plan §6.2)
# --------------------------------------------------------------------------- #


def _margin_stream(alternatives: Sequence[tuple[str, float]]) -> list[TokenLP]:
    """A stream committing to ``refund`` whose route token carries ``alternatives``."""
    tokens = _stream(
        [
            ('{"rationale":"r","kind":"route","route":"', 0.99),
            ("refund", 0.7),
            ('","args":null}', 0.99),
        ]
    )
    return [
        TokenLP(
            token=token.token,
            logprob=token.logprob,
            top=[(text, math.log(prob)) for text, prob in alternatives]
            if token.token == "refund"
            else list(token.top),
        )
        for token in tokens
    ]


def test_margin_is_the_renormalised_gap_at_the_first_divergent_token() -> None:
    """Mass renormalised over *trie-valid* continuations only (plan §6.2). [v0.2]"""
    tokens = _margin_stream([("refund", 0.7), ("reset", 0.2)])

    # 0.7/(0.7+0.2) - 0.2/(0.7+0.2)
    assert compute_margin(tokens, "refund", ["refund", "reset"]) == pytest.approx(5 / 9)


def test_margin_is_none_when_the_runner_up_is_absent_from_top_logprobs() -> None:
    """The **specified fallback** of plan §6.2, and the reason it is specified.

    Providers return only 5-20 alternatives. Assuming the leftover probability
    mass belongs to the runner-up is how a margin signal turns into noise on
    exactly the ambiguous queries it exists to catch — so there is no guessing:
    ``None``, and fusion ignores it.
    """
    tokens = _margin_stream([("refund", 0.7)])

    assert compute_margin(tokens, "refund", ["refund", "reset"]) is None


def test_margin_is_none_without_a_competing_candidate() -> None:
    """No divergence point means no gap to measure (plan §6.2)."""
    tokens = _margin_stream([("refund", 0.7), ("reset", 0.2)])

    assert compute_margin(tokens, "refund", ["refund"]) is None
    assert compute_margin(tokens, "refund", []) is None


def test_the_v01_loop_does_not_compute_a_margin(
    router_factory: Callable[..., Router]
) -> None:
    """``margin`` ships but is deliberately not called in v0.1 (plan §13 #9)."""
    router = router_factory([wire_output("refund", {"order_id": "A-1"})])

    decision = router.route("refund order A-1")

    assert decision.confidence.margin is None


# --------------------------------------------------------------------------- #
# Fusion (plan §6.3)
# --------------------------------------------------------------------------- #


def test_default_fusion_is_the_minimum_of_the_available_signals() -> None:
    """"Conservative *by construction*": adding a signal can only lower it (§6.3)."""
    report = ConfidenceReport(score=0.0, method="x", p_route=0.6, agreement=0.9)

    assert default_fusion(report) == pytest.approx(0.6)


def test_verbalized_confidence_is_damped_by_zero_eight_five() -> None:
    """x0.85 before it can influence anything (plan §6.3, Appendix A)."""
    assert default_fusion(
        ConfidenceReport(score=0.0, method="verbalized", stated=0.9)
    ) == pytest.approx(0.765)

    # And the damped value still goes through the min, so a confident model
    # cannot talk its way past a weak logprob signal.
    assert default_fusion(
        ConfidenceReport(score=0.0, method="x", p_route=0.9, stated=0.5)
    ) == pytest.approx(0.425)


def test_an_overturned_vote_costs_a_further_zero_eight() -> None:
    """A majority that overturned greedy is a strong clarify indicator (§6.2, §6.3)."""
    assert default_fusion(
        ConfidenceReport(score=0.0, method="x", p_route=0.5, vote_overturned=True)
    ) == pytest.approx(0.4)


def test_no_signals_fuse_to_zero() -> None:
    """Which is why :func:`signals_are_actionable` exists (plan §6.3)."""
    assert default_fusion(ConfidenceReport(score=0.0, method="none")) == 0.0


def test_a_custom_fusion_replaces_the_default() -> None:
    """``Router(confidence_fusion=...)`` is a documented seam (plan §6.3)."""
    report = build_confidence_report(
        token_logprobs=_stream(
            [('{"route":"', 0.99), ("refund", 0.5), ('","x":1}', 0.99)]
        ),
        route_name="refund",
        fusion=lambda _: 0.123,
    )

    assert report.score == pytest.approx(0.123)
    assert report.p_route == pytest.approx(0.5)  # the raw evidence survives


def test_the_report_records_the_thresholds_as_applied() -> None:
    """Recalibration reads raw evidence plus the bands it was judged against (§6.3)."""
    report = build_confidence_report(
        route_name="refund", thresholds=ThresholdPolicy().as_thresholds()
    )

    assert report.thresholds == {"abstain": 0.30, "clarify": 0.55}


# --------------------------------------------------------------------------- #
# THE NO-SIGNAL RULE (plan §6.3, §13 ruling #9)
# --------------------------------------------------------------------------- #


def test_signals_are_actionable_only_with_a_real_signal() -> None:
    """Logprobs, margin or votes make the machinery live; nothing else does."""
    assert signals_are_actionable(ConfidenceReport(score=0.9, method="x", p_route=0.9))
    assert signals_are_actionable(ConfidenceReport(score=0.9, method="x", agreement=0.9))
    assert signals_are_actionable(ConfidenceReport(score=0.9, method="x", margin=0.4))
    assert not signals_are_actionable(ConfidenceReport(score=0.0, method="none"))


def test_verbalized_only_confidence_does_not_downgrade_a_route(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """THE NO-SIGNAL RULE (plan §6.3).

    ``stated=0.2`` fuses to 0.17, which is below *both* bands. With logprobs that
    would be an abstain. With verbalized confidence alone the thresholds are
    inert and the model's commitment stands — because acting on exactly the
    signal the evidence distrusts would invert Recommendation #4 and turn the
    router into something that abstains whenever the model sounds unsure.
    """
    parsed = wire_model.model_validate(
        {"rationale": "r", "kind": "route", "route": "refund"}
    )
    report = ConfidenceReport(score=0.85 * 0.2, method="verbalized", stated=0.2)

    decision = resolve_decision(
        parsed, policy_routes, report, ThresholdPolicy(), AuditDraft()
    )

    assert isinstance(decision, RouteDecision)
    assert decision.route == "refund"
    assert decision.downgraded_from is None
    # Recorded, though — recalibration needs the raw evidence (§6.3).
    assert decision.confidence.stated == 0.2
    assert decision.confidence.method == "verbalized"


def test_thresholds_on_verbalized_opts_back_in(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """The explicit opt-in of plan §6.3, which the Router warns about once."""
    parsed = wire_model.model_validate(
        {"rationale": "r", "kind": "route", "route": "refund"}
    )
    report = ConfidenceReport(score=0.85 * 0.5, method="verbalized", stated=0.5)

    decision = resolve_decision(
        parsed,
        policy_routes,
        report,
        ThresholdPolicy(),
        AuditDraft(),
        thresholds_on_verbalized=True,
    )

    assert isinstance(decision, ClarifyDecision)
    assert decision.downgraded_from == "route"
    assert decision.audit.abstain_reason == "low_confidence"


def test_the_no_signal_rule_holds_end_to_end(
    router_factory: Callable[..., Router]
) -> None:
    """A bare BYO callable declares no logprobs, so the ladder drops a rung (§4.1)."""
    payload = wire_output("refund", {"order_id": "A-1"}, stated_confidence=0.2)

    inert = router_factory([payload], capabilities=NO_SIGNAL)
    decision = inert.route("refund order A-1")
    assert decision.kind == "route"
    assert decision.confidence.p_route is None
    assert decision.confidence.stated == 0.2

    with pytest.warns(UserWarning, match="thresholds_on_verbalized"):
        armed = router_factory(
            [payload], capabilities=NO_SIGNAL, thresholds_on_verbalized=True
        )
    assert armed.route("refund order A-1").kind == "abstain"


def test_model_elected_outcomes_pass_through_even_with_no_signals(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """"Model-elected clarify/abstain always pass through regardless" (§6.3)."""
    parsed = wire_model.model_validate(
        {"rationale": "r", "kind": "abstain", "reason": "model_elected"}
    )

    decision = resolve_decision(
        parsed, policy_routes, ConfidenceReport(score=0.0, method="none"),
        ThresholdPolicy(), AuditDraft(),
    )

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "model_elected"


# --------------------------------------------------------------------------- #
# Thresholds: downgrade-only (plan §6.4)
# --------------------------------------------------------------------------- #


def _route_output(wire_model: type[BaseModel]) -> BaseModel:
    return wire_model.model_validate(
        {
            "rationale": "the request names an order",
            "kind": "route",
            "route": "refund",
            "args": {"order_id": "A-1", "amount": 5},
        }
    )


@pytest.mark.parametrize(
    ("score", "expected_kind"),
    [(0.95, "route"), (0.55, "route"), (0.54, "clarify"), (0.30, "clarify"), (0.29, "abstain")],
)
def test_the_threshold_bands(
    score: float,
    expected_kind: str,
    policy_routes: Mapping[str, Route],
    wire_model: type[BaseModel],
) -> None:
    """Plan §6.4's table: ``< 0.30`` abstain, ``< 0.55`` clarify, else route."""
    report = ConfidenceReport(score=score, method="logprobs", p_route=score)

    decision = resolve_decision(
        _route_output(wire_model), policy_routes, report, ThresholdPolicy(), AuditDraft()
    )

    assert decision.kind == expected_kind


def test_a_downgrade_is_labelled_and_reasoned(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """``downgraded_from`` on the decision, ``low_confidence`` in the audit (§6.4)."""
    report = ConfidenceReport(score=0.4, method="logprobs", p_route=0.4)

    decision = resolve_decision(
        _route_output(wire_model), policy_routes, report, ThresholdPolicy(), AuditDraft()
    )

    assert isinstance(decision, ClarifyDecision)
    assert decision.downgraded_from == "route"
    assert decision.audit.abstain_reason == "low_confidence"
    assert decision.audit.kind == "clarify"
    assert decision.audit.routes == []
    # Threshold downgrades have no authored question, so one is synthesised from
    # the top-2 candidates by signal (§6.5).
    assert decision.question


def test_high_confidence_never_upgrades_a_model_elected_clarify(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """DOWNGRADE-ONLY (plan §6.4 rule 1).

    Nothing in the policy stage may promote a clarify into a route or invent a
    commitment the model did not make — no matter how confident the signals are.
    """
    parsed = wire_model.model_validate(
        {
            "rationale": "torn between two",
            "kind": "clarify",
            "question": "Refund or tracking?",
            "candidates": ["refund", "track_order"],
            "missing": ["order_id"],
        }
    )
    report = ConfidenceReport(score=0.99, method="logprobs", p_route=0.99)

    decision = resolve_decision(parsed, policy_routes, report, ThresholdPolicy(), AuditDraft())

    assert isinstance(decision, ClarifyDecision)
    assert decision.question == "Refund or tracking?"
    assert decision.candidates == ("refund", "track_order")
    assert decision.missing == ("order_id",)
    assert decision.audit.abstain_reason == "model_elected"
    assert decision.downgraded_from is None


def test_an_unconfigured_multi_route_is_treated_as_an_ambiguity(
    policy_routes: Mapping[str, Route]
) -> None:
    """Defence in depth for a commitment the router did not ask for (plan §3.4).

    ``multi_route=False`` removes the kind from the wire enum, so the loop
    normally never sees one. If one arrives anyway — a lower rung, a hand-rolled
    adapter — returning one member of a set the model deliberately emitted
    *together* would be guessing, so it becomes a question instead.
    """
    parsed = build_wire_schema(CANDIDATES, mode="static", multi_route=True).model_validate(
        {
            "rationale": "two independent asks",
            "kind": "multi_route",
            "routes": [{"route": "refund"}, {"route": "track_order"}],
        }
    )

    decision = resolve_decision(
        parsed,
        policy_routes,
        ConfidenceReport(score=0.95, method="logprobs", p_route=0.95),
        ThresholdPolicy(),
        AuditDraft(),
        allow_multi_route=False,
    )

    assert isinstance(decision, ClarifyDecision)
    assert set(decision.candidates) <= {"refund", "track_order"}


def test_clarify_becomes_abstain_when_clarify_is_disabled(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """``allow_clarify=False``: everything that would clarify abstains (§3.8)."""
    report = ConfidenceReport(score=0.4, method="logprobs", p_route=0.4)

    decision = resolve_decision(
        _route_output(wire_model),
        policy_routes,
        report,
        ThresholdPolicy(),
        AuditDraft(),
        allow_clarify=False,
    )

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "low_confidence"
    assert decision.downgraded_from == "route"


def test_weak_retrieval_widens_the_question_band_not_the_silence_band(
    policy_routes: Mapping[str, Route], wire_model: type[BaseModel]
) -> None:
    """§5.5 handed to the policy stage: a query nothing matched is a clarify.

    A score of 0.6 clears the ordinary 0.55 clarify band. Under weak retrieval
    the band rises and the same score becomes a question instead of a guess —
    while the *abstain* band drops to zero, so weak retrieval can never turn a
    question into silence.
    """
    report = ConfidenceReport(score=0.6, method="logprobs", p_route=0.6)

    ordinary = resolve_decision(
        _route_output(wire_model), policy_routes, report, ThresholdPolicy(), AuditDraft()
    )
    weak = resolve_decision(
        _route_output(wire_model),
        policy_routes,
        report,
        ThresholdPolicy(),
        AuditDraft(),
        weak_retrieval=True,
    )

    assert ordinary.kind == "route"
    assert weak.kind == "clarify"

    very_low = ConfidenceReport(score=0.05, method="logprobs", p_route=0.05)
    still_a_question = resolve_decision(
        _route_output(wire_model),
        policy_routes,
        very_low,
        ThresholdPolicy(),
        AuditDraft(),
        weak_retrieval=True,
    )
    assert still_a_question.kind == "clarify"


def test_a_surviving_hallucinated_reference_abstains(
    policy_routes: Mapping[str, Route]
) -> None:
    """Defence in depth: the policy stage re-checks membership too (§4.5)."""
    parsed = build_wire_schema(CANDIDATES, mode="static").model_validate(
        {"rationale": "r", "kind": "route", "route": "billing_report"}
    )

    decision = resolve_decision(
        parsed,
        policy_routes,
        ConfidenceReport(score=0.99, method="logprobs", p_route=0.99),
        ThresholdPolicy(),
        AuditDraft(),
    )

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "invalid_route_reference"


def test_threshold_policy_rejects_incoherent_bands() -> None:
    """"A policy where ``abstain_below > clarify_below`` has no interpretation" (§6.4)."""
    with pytest.raises(ConfigError, match="abstain_below"):
        ThresholdPolicy(abstain_below=0.9, clarify_below=0.2)
    with pytest.raises(ConfigError, match=r"\[0, 1\]"):
        ThresholdPolicy(abstain_below=1.5)


def test_threshold_policy_provenance() -> None:
    """A calibration is only valid for the model it was fitted against (§6.4)."""
    policy = ThresholdPolicy(model_id="gpt-5-nano", registry_version="aaaaaaaaaaaa")

    assert policy.validate_provenance(model_id="gpt-5-nano", registry_version="aaaaaaaaaaaa") is False
    # Registry drift: stale, warned about, the run continues.
    assert policy.validate_provenance(model_id="gpt-5-nano", registry_version="bbbbbbbbbbbb") is True
    # Model swap: silently wrong in a way nothing downstream would reveal.
    with pytest.raises(ConfigError, match="calibrated for model"):
        policy.validate_provenance(model_id="claude-haiku-4-5")


def test_clarify_questions_are_templated_from_the_top_two_candidates(
    policy_routes: Mapping[str, Route]
) -> None:
    """``"Did you mean {a.clarify_label} or {b.clarify_label}?"`` (plan §6.5)."""
    question = synthesize_clarify_question(["refund", "track_order"], policy_routes)

    assert question == "Did you mean a refund or tracking an order?"


# --------------------------------------------------------------------------- #
# Fallback (plan §6.6, §13 ruling #5)
# --------------------------------------------------------------------------- #


def test_fallback_turns_a_terminal_abstain_into_a_route_decision(
    policy_routes: Mapping[str, Route]
) -> None:
    """"One shape, everywhere" (plan §3.4, §6.6).

    A caller's ``kind == "route"`` branch needs no fallback handling at all;
    ``decision_path`` is how the two are told apart, and the pre-fallback reason
    survives in the audit record so the substitution stays auditable.
    """
    declined = abstain(
        "low_confidence",
        draft=AuditDraft(),
        confidence=ConfidenceReport(score=0.1, method="logprobs", p_route=0.1),
    )

    decision = apply_fallback(declined, "human_handoff", list(policy_routes.values()))

    assert isinstance(decision, RouteDecision)
    assert decision.route == "human_handoff"
    assert decision.decision_path == "fallback"
    assert decision.args is None
    assert decision.audit.kind == "route"
    assert decision.audit.routes == ["human_handoff"]
    assert decision.audit.abstain_reason == "low_confidence"
    assert decision.audit.decision_path == "fallback"
    # The confidence report is carried over untouched: a fallback is a routing
    # *policy*, not evidence, and inflating the score would poison recalibration.
    assert decision.confidence.p_route == pytest.approx(0.1)


def test_fallback_never_bypasses_entitlement(policy_routes: Mapping[str, Route]) -> None:
    """"If the tenant's view filters out the fallback route, the abstain stands" (§6.6)."""
    declined = abstain("low_confidence", draft=AuditDraft())

    entitled = [route for route in policy_routes.values() if route.name != "human_handoff"]
    decision = apply_fallback(declined, "human_handoff", entitled)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "low_confidence"
    assert decision.decision_path == "llm"


def test_fallback_only_fires_on_abstain(policy_routes: Mapping[str, Route]) -> None:
    """Anything that is not a terminal abstain is returned unchanged (plan §6.6)."""
    committed = resolve_decision(
        build_wire_schema(CANDIDATES, mode="static").model_validate(
            {"rationale": "r", "kind": "route", "route": "track_order"}
        ),
        policy_routes,
        ConfidenceReport(score=0.95, method="logprobs", p_route=0.95),
        ThresholdPolicy(),
        AuditDraft(),
    )

    assert apply_fallback(committed, "human_handoff", list(policy_routes.values())) is committed


def test_fallback_end_to_end_through_the_router(
    router_factory: Callable[..., Router], small_registry: Registry
) -> None:
    """The same substitution, driven by the loop (plan §6.6, §3.8 last row)."""
    router = router_factory(
        ["not json", "still not json", "never json"], fallback="human_handoff"
    )

    decision = router.route("something unroutable")

    assert isinstance(decision, RouteDecision)
    assert decision.route == "human_handoff"
    assert decision.decision_path == "fallback"
    assert decision.audit.abstain_reason == "unparseable_output"


def test_router_refuses_an_unregistered_fallback(
    small_registry: Registry, fake_client: Any
) -> None:
    """An unknown fallback route is a ConfigError at construction (plan §3.8)."""
    from switchboard import Router

    with pytest.raises(ConfigError, match="fallback"):
        Router(small_registry, client=fake_client, fallback="not_a_route", otel=False)
