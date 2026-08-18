"""Threshold policy, decision resolution and fallback (plan §3.8, §6.4, §6.6).

The last stage before the audit emit (§2.1). It takes a *validated* model output
plus a :class:`~switchboard.core.audit.ConfidenceReport` and produces the typed
:data:`~switchboard.core.decision.Decision` the caller sees, under three
invariants that the rest of the library depends on:

1. **Downgrade-only.** Thresholds move a decision along ``route → clarify →
   abstain`` and never the other way (§6.4). Nothing in this module can promote a
   clarify into a route or invent a commitment the model did not make. The single
   ``abstain → route`` transition in the library is :func:`apply_fallback`, which
   is a *substitution* with its own ``decision_path="fallback"`` marker, not an
   upgrade — a caller can always tell the two apart (§13 ruling #5).
2. **The model's own election is honoured.** If the model chose ``clarify`` or
   ``abstain`` in-schema, that passes through untouched with the reason
   ``model_elected`` (§3.8). Thresholds only ever apply to model-emitted
   ``route``/``multi_route``.
3. **Degrade, never raise.** Every path here returns a Decision. The only
   exceptions this module raises come from :class:`ThresholdPolicy` construction —
   broken configuration, which §3.8 says must raise immediately.

Audit contract: :func:`resolve_decision` and the :func:`abstain` / :func:`clarify`
constructors **finalize the audit draft**, because ``Decision.audit`` is a frozen
``AuditRecord`` and the decision is not a decision without it. Everything else —
``input_text``, ``args``, ``args_hash``, ``rationale`` and any other
``content_mode``-gated or latency field — must therefore be written to the draft
*before* calling in. The canonical record for emission is ``decision.audit``;
``draft.finalize()`` is idempotent and returns the same object, so a router that
also finalizes in its ``finally`` block emits exactly one record either way.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, model_validator

from switchboard.core.audit import ConfidenceReport
from switchboard.core.candidates import Candidate
from switchboard.core.decision import (
    AbstainDecision,
    ClarifyDecision,
    MultiRouteDecision,
    RoutedCall,
    RouteDecision,
)
from switchboard.engine.confidence import signals_are_actionable
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from switchboard.core.audit import AbstainReason, AuditDraft, DecisionPath
    from switchboard.core.decision import Decision
    from switchboard.core.route import Route

__all__ = [
    "WEAK_RETRIEVAL_CLARIFY_MARGIN",
    "ThresholdPolicy",
    "abstain",
    "apply_fallback",
    "clarify",
    "resolve_decision",
    "route_index",
    "synthesize_clarify_question",
]


WEAK_RETRIEVAL_CLARIFY_MARGIN = 0.15
"""How much weak retrieval widens the clarify band (plan §5.5).

§5.5 hands this rule to the policy stage without a number, so here is the number
and the reasoning. When the shortlister reports ``weak_retrieval`` — fewer than
three candidates scored above zero, i.e. *nothing in the catalog lexically or
semantically matched the query* — two adjustments apply, and only while the
confidence signals are actionable at all (the no-signal rule wins; §6.3):

* the clarify threshold rises from ``clarify_below`` (0.55) to 0.70, so a
  middling commitment over a catalog that matched nothing becomes a question
  rather than a guess — "a query nothing matches is a clarification candidate,
  not a guessing opportunity";
* the abstain threshold drops to 0.0 whenever ``allow_clarify`` is on, so the
  low-confidence band that would have produced *silence* produces a *question*
  instead. Weak retrieval widens the question band; it never widens the silence
  band, because silence teaches the user nothing about a catalog that may simply
  be described badly (§5.7).

A high-confidence commitment is left alone: the LLM is the decider and the
shortlister only controls what it sees (§5.1).
"""

_MAX_LABEL_CHARS = 72
"""Clarify labels are read aloud by voice UIs and pasted into chat bubbles; a
description's first clause past ~72 chars is a paragraph, not a label."""

_CLAUSE_SPLIT = re.compile("[.;:!?]|\\s[-\\u2013\\u2014]\\s|,\\s+(?=which|and then|then\\b)")
"""First-clause boundary for a ``clarify_label`` fallback (§6.5). Deliberately
conservative — it splits on sentence punctuation and em-dash asides, not on every
comma, because "Issue or check a refund for an order, including partial refunds"
loses its meaning at the comma."""

_ABSTAIN_RATIONALE: dict[str, str] = {
    "no_eligible_routes": "No routes are available to this request.",
    "unparseable_output": "The model's output could not be parsed after the allowed repair attempts.",
    "invalid_route_reference": "The model named a route that is not available for this request.",
    "invalid_args": "The arguments required by the chosen route could not be extracted.",
    "low_confidence": "Confidence in every candidate route was below the configured threshold.",
    "provider_error": "The model provider could not be reached.",
    "model_elected": "The model determined that no route fits this request.",
}
"""Default rationales, so an abstain is never returned with an empty explanation.
A caller-supplied rationale (usually the model's own, on ``model_elected``)
always wins."""


# --------------------------------------------------------------------------- #
# ThresholdPolicy (plan §6.4)
# --------------------------------------------------------------------------- #


class ThresholdPolicy(BaseModel):
    """Downgrade thresholds and their calibration provenance (plan §6.4). [v0.1]

    The defaults are Appendix A's authoritative numbers (abstain 0.30, clarify
    0.55) and are **not** calibrated for any particular model — they are
    deliberately conservative starting points. ``switchboard eval calibrate``
    [v0.2] sweeps them against a labelled run and writes back a policy stamped
    with the ``(model_id, registry_version)`` it was fitted on, which is what the
    provenance fields are for: a calibration is only valid for the model it was
    fitted against (§6.4).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    abstain_below: float = 0.30
    """``conf < 0.30`` → abstain (→ fallback, if one is configured and entitled)."""
    clarify_below: float = 0.55
    """``0.30 <= conf < 0.55`` → clarify."""
    margin_clarify_below: float = 0.10
    """Margin gate, applied only when ``margin`` is available. [v0.2]"""
    multi_route_member_min: float = 0.35
    """Drop weak multi-route members; all dropped → clarify. [v0.2]"""
    model_id: str | None = None
    """Model this policy was calibrated against; a mismatch is an **error**."""
    registry_version: str | None = None
    """Registry this policy was calibrated against; a mismatch is a **warning**."""

    @model_validator(mode="after")
    def _check_bounds(self) -> ThresholdPolicy:
        """Reject incoherent thresholds as ``ConfigError`` (plan §3.8).

        Raised rather than clamped: a policy where ``abstain_below >
        clarify_below`` has no meaningful interpretation, and silently reordering
        the bands would produce a router that abstains where the operator asked it
        to clarify.
        """
        for name in (
            "abstain_below",
            "clarify_below",
            "margin_clarify_below",
            "multi_route_member_min",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"ThresholdPolicy.{name} must be in [0, 1], got {value!r}")
        if self.abstain_below > self.clarify_below:
            raise ConfigError(
                f"ThresholdPolicy.abstain_below ({self.abstain_below}) must not exceed "
                f"clarify_below ({self.clarify_below}); the bands are ordered "
                f"abstain < clarify < route"
            )
        return self

    def as_thresholds(self) -> dict[str, float]:
        """The thresholds-as-applied dict recorded in every ``ConfidenceReport``.

        Only the v0.1 bands are reported. ``margin_clarify_below`` and
        ``multi_route_member_min`` join this dict when the [v0.2] signals that
        feed them are actually computed — recording a threshold that nothing was
        compared against would make the audit record lie to the recalibration
        sweep that reads it.
        """
        return {"abstain": self.abstain_below, "clarify": self.clarify_below}

    def validate_provenance(
        self,
        *,
        model_id: str | None = None,
        registry_version: str | None = None,
    ) -> bool:
        """Check this policy against the live model and registry (plan §6.4).

        Args:
            model_id: the model actually being called.
            registry_version: the live ``registry.version``.

        Returns:
            ``True`` when the calibration is **stale** — fitted against a different
            registry version. Route descriptions changed underneath it, so the
            thresholds are approximate; the run continues.

        Raises:
            ConfigError: the policy was calibrated against a different model.
                Calibration is model-specific — logprob distributions are not
                comparable across models — so applying one model's thresholds to
                another is silently wrong in a way no downstream signal would
                reveal.
        """
        if self.model_id is not None and model_id is not None and self.model_id != model_id:
            raise ConfigError(
                f"ThresholdPolicy was calibrated for model {self.model_id!r} but the router "
                f"is configured with {model_id!r}; confidence calibration is model-specific. "
                f"Re-run `switchboard eval calibrate` or clear ThresholdPolicy.model_id."
            )
        return (
            self.registry_version is not None
            and registry_version is not None
            and self.registry_version != registry_version
        )


# --------------------------------------------------------------------------- #
# Decision constructors (plan §3.4)
# --------------------------------------------------------------------------- #


def _no_signal_report() -> ConfidenceReport:
    """The report attached to decisions taken without ever calling a model."""
    return ConfidenceReport(score=0.0, method="none")


def abstain(
    reason: AbstainReason,
    *,
    draft: AuditDraft,
    confidence: ConfidenceReport | None = None,
    rationale: str = "",
    downgraded_from: str | None = None,
    decision_path: DecisionPath = "llm",
) -> AbstainDecision:
    """Build an :class:`~switchboard.core.decision.AbstainDecision` (plan §3.4, §3.8).

    Abstaining is a *result*, not an exception (§3, decision (c)): the caller gets
    a machine-readable ``reason`` from the one closed vocabulary and a full audit
    record, so "we declined to route" is as observable as any commitment.

    Fills the draft consistently — ``kind``, empty ``routes``, ``abstain_reason``
    (preserved even if a fallback later turns the outcome into a route, §13 ruling
    #5), ``confidence``, ``downgraded_from``, ``decision_path`` — then finalizes
    it. Use for the no-LLM-call paths too: the empty entitled set degrades to
    ``abstain("no_eligible_routes")`` here (§13 ruling #3).
    """
    report = confidence if confidence is not None else _no_signal_report()
    draft.kind = "abstain"
    draft.routes = []
    draft.abstain_reason = reason
    draft.confidence = report
    draft.downgraded_from = downgraded_from
    draft.decision_path = decision_path
    return AbstainDecision(
        reason=reason,
        rationale=rationale or _ABSTAIN_RATIONALE.get(reason, "The router declined to route."),
        confidence=report,
        audit=draft.finalize(),
        decision_path=decision_path,
        downgraded_from=downgraded_from,
    )


def clarify(
    question: str,
    *,
    draft: AuditDraft,
    confidence: ConfidenceReport | None = None,
    candidates: Sequence[str] = (),
    missing: Sequence[str] = (),
    rationale: str = "",
    downgraded_from: str | None = None,
    reason: AbstainReason | None = None,
    decision_path: DecisionPath = "llm",
) -> ClarifyDecision:
    """Build a :class:`~switchboard.core.decision.ClarifyDecision` (plan §3.4, §6.5).

    Clarifying is cheaper than mis-routing (§9.2), and it is reached three ways:
    the model elected it in-schema, an args-only validation failure downgraded an
    otherwise-sound route choice (§3.8 row 4, hence ``missing``), or the fused
    confidence landed in the clarify band (hence ``downgraded_from="route"``).

    ``reason`` records *why* in the audit record without changing the decision
    kind: §3.8's degradation table gives the same reason code to both arms of a
    row (``low_confidence`` for the threshold row, ``model_elected`` for the
    model's own election), so the record stays queryable by cause rather than only
    by outcome.
    """
    report = confidence if confidence is not None else _no_signal_report()
    draft.kind = "clarify"
    draft.routes = []
    draft.confidence = report
    draft.downgraded_from = downgraded_from
    draft.decision_path = decision_path
    if reason is not None:
        draft.abstain_reason = reason
    return ClarifyDecision(
        question=question,
        candidates=tuple(candidates),
        missing=tuple(missing),
        rationale=rationale or "More information is needed before a route can be chosen.",
        confidence=report,
        audit=draft.finalize(),
        decision_path=decision_path,
        downgraded_from=downgraded_from,
    )


def _route_decision(
    name: str,
    *,
    draft: AuditDraft,
    confidence: ConfidenceReport,
    args: BaseModel | None = None,
    rationale: str = "",
    decision_path: DecisionPath = "llm",
) -> RouteDecision:
    """Build a committed :class:`~switchboard.core.decision.RouteDecision`."""
    draft.kind = "route"
    draft.routes = [name]
    draft.confidence = confidence
    draft.downgraded_from = None
    draft.decision_path = decision_path
    return RouteDecision(
        route=name,
        args=args,
        rationale=rationale or f"Routing to {name}.",
        confidence=confidence,
        audit=draft.finalize(),
        decision_path=decision_path,
    )


# --------------------------------------------------------------------------- #
# Clarify-question synthesis (plan §6.5)
# --------------------------------------------------------------------------- #


def synthesize_clarify_question(
    candidates: Iterable[Candidate | str],
    routes: Mapping[str, Route],
) -> str:
    """Template a clarify question over the top-2 candidates (plan §6.5).

    Threshold-triggered downgrades have no authored question — the model already
    committed to a route, so nobody asked it to write one — and re-prompting for
    one would cost a second round trip on exactly the requests that are already
    slow. So the question is synthesised from the top-2 candidates *by signal*:
    :class:`~switchboard.core.candidates.Candidate` inputs are ordered by their
    pre-shuffle ``rank`` (the retriever's actual opinion, §5.6), plain names keep
    the order they were given in.

    Labels come from ``Route.clarify_label``, falling back to the first clause of
    the description — which is why §5.7's linting pushes descriptions to open with
    what the route *does*: the first clause is user-facing here, not just
    model-facing.

    The optional LLM rephrase (``Router(clarify_refiner=True)``) is [v0.2].
    """
    names = _ordered_candidate_names(candidates)
    labels = [_clarify_label(name, routes) for name in names[:2]]
    if not labels:
        return "Could you tell me a bit more about what you would like to do?"
    if len(labels) == 1:
        return f"Did you mean {labels[0]}?"
    return f"Did you mean {labels[0]} or {labels[1]}?"


def _ordered_candidate_names(candidates: Iterable[Candidate | str]) -> list[str]:
    """De-duplicated candidate names, strongest signal first."""
    items = list(candidates)
    if items and all(isinstance(item, Candidate) for item in items):
        ranked = sorted(
            (item for item in items if isinstance(item, Candidate)),
            key=lambda candidate: (candidate.rank, -candidate.score),
        )
        return list(dict.fromkeys(candidate.route_name for candidate in ranked))
    names = [item if isinstance(item, str) else item.route_name for item in items]
    return list(dict.fromkeys(name for name in names if name))


def _clarify_label(name: str, routes: Mapping[str, Route]) -> str:
    """Human phrase for ``name``: ``clarify_label``, else the description's first clause."""
    route = routes.get(name)
    if route is None:
        return name
    if route.clarify_label:
        return route.clarify_label
    clause = _CLAUSE_SPLIT.split(" ".join(route.description.split()), maxsplit=1)[0].strip()
    if not clause:
        return name
    if len(clause) > _MAX_LABEL_CHARS:
        cut = clause.rfind(" ", 0, _MAX_LABEL_CHARS)
        clause = clause[: cut if cut > 0 else _MAX_LABEL_CHARS].rstrip() + "..."
    head = clause.split(" ", 1)[0]
    if head.isupper() or not head.isalpha():
        return clause
    return clause[0].lower() + clause[1:]


# --------------------------------------------------------------------------- #
# Resolution (plan §3.8, §6.4)
# --------------------------------------------------------------------------- #


def route_index(routes: Iterable[Route]) -> dict[str, Route]:
    """Build the ``{name: Route}`` lookup :func:`resolve_decision` takes.

    ``Registry`` and ``RegistryView`` are sequence-shaped, not mapping-shaped
    (they iterate routes, not names), so this is the one-liner that adapts either
    of them — or an :class:`~switchboard.engine.entitlements.EntitlementResult` —
    into the lookup.
    """
    return {route.name: route for route in routes}


def resolve_decision(
    parsed: BaseModel,
    routes: Mapping[str, Route],
    confidence: ConfidenceReport,
    policy: ThresholdPolicy,
    draft: AuditDraft,
    *,
    args: BaseModel | None = None,
    member_args: Mapping[str, BaseModel | None] | None = None,
    rationale: str | None = None,
    candidates: Sequence[Candidate | str] = (),
    allow_clarify: bool = True,
    allow_multi_route: bool = False,
    weak_retrieval: bool = False,
    thresholds_on_verbalized: bool = False,
) -> Decision:
    """Turn a validated wire output into the final Decision (plan §3.8, §6.4).

    The threshold logic is **downgrade-only** and applies only to model-emitted
    ``route``/``multi_route``:

    ===================================== ==========================================
    Fused score                           Outcome
    ===================================== ==========================================
    ``< abstain_below``                   ``abstain("low_confidence")``
    ``< clarify_below``                   ``clarify`` (``abstain`` if clarify is off)
    otherwise                             the committed route, untouched
    ===================================== ==========================================

    and the whole table is **inert** unless
    :func:`~switchboard.engine.confidence.signals_are_actionable` says otherwise —
    on a verbalized-only path the committed route stands (§6.3). Weak retrieval
    widens the clarify band per :data:`WEAK_RETRIEVAL_CLARIFY_MARGIN`.

    Model-elected ``clarify``/``abstain`` bypasses all of it and is returned as-is
    with ``reason="model_elected"`` (§3.8).

    Args:
        parsed: the schema-valid, route-checked wire output.
        routes: ``{name: Route}`` over the **entitled** set — see :func:`route_index`.
        confidence: the fused report; its ``thresholds`` are stamped from ``policy``
            if the caller left them empty.
        policy: the thresholds to apply.
        draft: the audit draft. **Finalized by this call** — write every
            ``content_mode``-gated and latency field first (see the module docstring).
        args: the validated args instance for a single committed route.
        member_args: validated args per route name for a ``multi_route`` output;
            members absent from the mapping get ``args=None``. Args validation is
            the loop's job (§4.5), not the policy stage's.
        rationale: overrides the model's own ``rationale`` field.
        candidates: shortlist candidates, used to synthesise a clarify question
            and to populate ``ClarifyDecision.candidates``.
        allow_clarify: when ``False``, everything that would clarify abstains
            instead (§3.8).
        allow_multi_route: whether ``Router(multi_route=True)`` was configured.
        weak_retrieval: ``ShortlistResult.weak_retrieval`` (§5.5).
        thresholds_on_verbalized: opt into acting on verbalized-only confidence.

    Returns:
        A :data:`~switchboard.core.decision.Decision`. Never raises.
    """
    report = _stamp_thresholds(confidence, policy)
    kind = _str_attr(parsed, "kind") or "route"
    text = rationale if rationale is not None else (_str_attr(parsed, "rationale") or "")
    name = _str_attr(parsed, "route")

    if kind == "clarify":
        return _model_elected_clarify(parsed, routes, report, draft, text, candidates)
    if kind == "abstain":
        return abstain("model_elected", draft=draft, confidence=report, rationale=text)
    if kind == "multi_route":
        return _resolve_multi_route(
            parsed,
            routes,
            report,
            policy,
            draft,
            member_args=member_args,
            rationale=text,
            candidates=candidates,
            allow_clarify=allow_clarify,
            allow_multi_route=allow_multi_route,
            weak_retrieval=weak_retrieval,
            thresholds_on_verbalized=thresholds_on_verbalized,
        )
    if kind != "route" and not name:
        # An unrecognised kind with nothing to commit to (a [v0.5] "plan", or a kind
        # the model invented). It parsed, but it is not a decision this router can
        # express, and guessing at one would be exactly the wrong recovery.
        return abstain("unparseable_output", draft=draft, confidence=report, rationale=text)

    if not name or name not in routes:
        # Defence in depth: engine/validate.py already enforced this (§4.5 layer 2).
        return abstain("invalid_route_reference", draft=draft, confidence=report, rationale=text)

    outcome = _apply_thresholds(
        score=report.score,
        policy=policy,
        actionable=signals_are_actionable(
            report, thresholds_on_verbalized=thresholds_on_verbalized
        ),
        weak_retrieval=weak_retrieval,
        allow_clarify=allow_clarify,
    )
    if outcome == "route":
        return _route_decision(
            name, draft=draft, confidence=report, args=args, rationale=text
        )
    return _downgrade(
        outcome,
        draft=draft,
        confidence=report,
        rationale=text,
        candidates=candidates or (name,),
        routes=routes,
        allow_clarify=allow_clarify,
    )


def _resolve_multi_route(
    parsed: BaseModel,
    routes: Mapping[str, Route],
    report: ConfidenceReport,
    policy: ThresholdPolicy,
    draft: AuditDraft,
    *,
    member_args: Mapping[str, BaseModel | None] | None,
    rationale: str,
    candidates: Sequence[Candidate | str],
    allow_clarify: bool,
    allow_multi_route: bool,
    weak_retrieval: bool,
    thresholds_on_verbalized: bool,
) -> Decision:
    """Resolve a ``multi_route`` output (plan §3.4).

    Per-member confidence filtering (``multi_route_member_min``) is [v0.2]; in
    v0.1 the fused score gates the decision as a whole, exactly as it does for a
    single route. If the router was not configured for multi-route the output is
    treated as an ambiguity rather than a commitment — returning one member of a
    set the model deliberately emitted together would be guessing.
    """
    members = [name for name in _member_names(parsed) if name in routes]
    if not members:
        return abstain(
            "invalid_route_reference", draft=draft, confidence=report, rationale=rationale
        )
    if not allow_multi_route:
        return _downgrade(
            "clarify" if allow_clarify else "abstain",
            draft=draft,
            confidence=report,
            rationale=rationale,
            candidates=members,
            routes=routes,
            allow_clarify=allow_clarify,
        )

    outcome = _apply_thresholds(
        score=report.score,
        policy=policy,
        actionable=signals_are_actionable(
            report, thresholds_on_verbalized=thresholds_on_verbalized
        ),
        weak_retrieval=weak_retrieval,
        allow_clarify=allow_clarify,
    )
    if outcome != "route":
        return _downgrade(
            outcome,
            draft=draft,
            confidence=report,
            rationale=rationale,
            candidates=candidates or members,
            routes=routes,
            allow_clarify=allow_clarify,
        )

    supplied = member_args or {}
    draft.kind = "multi_route"
    draft.routes = list(members)
    draft.confidence = report
    draft.downgraded_from = None
    return MultiRouteDecision(
        routes=tuple(RoutedCall(route=name, args=supplied.get(name)) for name in members),
        rationale=rationale or f"Routing to {', '.join(members)}.",
        confidence=report,
        audit=draft.finalize(),
    )


def _model_elected_clarify(
    parsed: BaseModel,
    routes: Mapping[str, Route],
    report: ConfidenceReport,
    draft: AuditDraft,
    rationale: str,
    candidates: Sequence[Candidate | str],
) -> ClarifyDecision:
    """Pass a model-authored clarify through untouched (plan §3.8, §6.5)."""
    offered = [name for name in _str_sequence(parsed, "candidates") if name in routes]
    question = _str_attr(parsed, "question") or synthesize_clarify_question(
        offered or candidates, routes
    )
    return clarify(
        question,
        draft=draft,
        confidence=report,
        candidates=offered,
        missing=_str_sequence(parsed, "missing"),
        rationale=rationale,
        reason="model_elected",
    )


def _apply_thresholds(
    *,
    score: float,
    policy: ThresholdPolicy,
    actionable: bool,
    weak_retrieval: bool,
    allow_clarify: bool,
) -> str:
    """Map a fused score to ``"route"`` / ``"clarify"`` / ``"abstain"`` (plan §6.4)."""
    if not actionable:
        # The no-signal rule (§6.3): thresholds are inert, so the model's
        # commitment stands. Weak retrieval rides on the thresholds and is
        # therefore inert here too — it adjusts bands, it is not a signal.
        return "route"
    abstain_at = policy.abstain_below
    clarify_at = policy.clarify_below
    if weak_retrieval and allow_clarify:
        clarify_at = min(1.0, clarify_at + WEAK_RETRIEVAL_CLARIFY_MARGIN)
        abstain_at = 0.0
    if score < abstain_at:
        return "abstain"
    if score < clarify_at:
        return "clarify" if allow_clarify else "abstain"
    return "route"


def _downgrade(
    outcome: str,
    *,
    draft: AuditDraft,
    confidence: ConfidenceReport,
    rationale: str,
    candidates: Sequence[Candidate | str],
    routes: Mapping[str, Route],
    allow_clarify: bool,
) -> Decision:
    """Emit the downgraded decision, marking ``downgraded_from="route"`` (§6.4).

    Both arms record ``low_confidence`` in the audit record: §3.8's threshold row
    gives that one reason code to both its clarify and its abstain outcome, so the
    *cause* of a downgrade stays queryable independently of which outcome it
    produced — which is what the [v0.2] recalibration sweep joins on.
    """
    if outcome == "clarify" and allow_clarify:
        names = _ordered_candidate_names(candidates)
        return clarify(
            synthesize_clarify_question(candidates, routes),
            draft=draft,
            confidence=confidence,
            candidates=names[:2],
            rationale=rationale,
            downgraded_from="route",
            reason="low_confidence",
        )
    return abstain(
        "low_confidence",
        draft=draft,
        confidence=confidence,
        rationale=rationale,
        downgraded_from="route",
    )


def _stamp_thresholds(
    confidence: ConfidenceReport, policy: ThresholdPolicy
) -> ConfidenceReport:
    """Ensure the report carries the thresholds it was actually judged against."""
    if confidence.thresholds:
        return confidence
    return confidence.model_copy(update={"thresholds": policy.as_thresholds()})


# --------------------------------------------------------------------------- #
# Fallback (plan §6.6, §13 ruling #5)
# --------------------------------------------------------------------------- #


def apply_fallback(
    decision: Decision,
    fallback_route: str | None,
    view: Iterable[Route | str],
    *,
    draft: AuditDraft | None = None,
    rationale: str | None = None,
) -> Decision:
    """Substitute a configured fallback for a terminal abstain (plan §6.6).

    One shape, everywhere (§13 ruling #5): the result is a
    :class:`~switchboard.core.decision.RouteDecision` with
    ``decision_path="fallback"`` and ``args=None``, so a caller's ``kind == "route"``
    branch needs no fallback handling at all. The pre-fallback ``AbstainReason``
    survives in ``audit.abstain_reason``, and the confidence report is carried over
    untouched — a fallback is a routing *policy*, not evidence, and inflating the
    score to match the new outcome would poison recalibration.

    **Entitlement is never bypassed.** If ``fallback_route`` is not in the entitled
    view, the abstain stands. That is the whole reason this takes the view rather
    than trusting the router's configuration: a fallback configured globally is
    still a route some tenant may not be allowed to reach (§6.6, §7.4).

    Args:
        decision: any decision. Anything that is not an
            :class:`~switchboard.core.decision.AbstainDecision` is returned
            unchanged — fallback fires only on terminal abstain paths.
        fallback_route: the configured fallback name, or ``None``.
        view: the entitled routes — a ``Registry``/``RegistryView``, an
            ``EntitlementResult.routes``, a ``{name: Route}`` mapping, or a set of
            names.
        draft: the audit draft, mirrored for consistency when the router also
            emits from it. The authoritative record is ``decision.audit``.
        rationale: overrides the default explanation.

    Returns:
        The substituted ``RouteDecision``, or ``decision`` unchanged.
    """
    if not isinstance(decision, AbstainDecision) or not fallback_route:
        return decision
    if fallback_route not in _entitled_names(view):
        return decision

    preserved: AbstainReason = decision.reason
    audit = decision.audit.model_copy(
        update={
            "kind": "route",
            "routes": [fallback_route],
            "decision_path": "fallback",
            "abstain_reason": preserved,
            "args": None,
            "args_hash": None,
        }
    )
    if draft is not None:
        draft.kind = "route"
        draft.routes = [fallback_route]
        draft.decision_path = "fallback"
        draft.abstain_reason = preserved
        draft.args = None
        draft.args_hash = None

    return RouteDecision(
        route=fallback_route,
        args=None,
        rationale=rationale
        or f"No route could be committed to ({preserved}); falling back to {fallback_route}.",
        confidence=decision.confidence,
        audit=audit,
        decision_path="fallback",
        downgraded_from=decision.downgraded_from,
    )


def _entitled_names(view: Iterable[Route | str]) -> frozenset[str]:
    """Route names in an entitled view, however the caller spelled it."""
    return frozenset(item if isinstance(item, str) else item.name for item in view)


# --------------------------------------------------------------------------- #
# Tolerant wire-output readers
# --------------------------------------------------------------------------- #
#
# The wire schema is generated per call by engine/schema.py and its shape varies
# with the configured arms (multi_route on/off, clarify on/off, verbalized rung).
# Reading it by getattr rather than by static field access keeps this module from
# having to know which arms were compiled in — and keeps a schema change from
# turning into an AttributeError inside the policy stage.


def _str_attr(obj: object, name: str) -> str | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, str) and value else None


def _sequence_attr(obj: object, name: str) -> Sequence[Any]:
    value = getattr(obj, name, None)
    return value if isinstance(value, (list, tuple)) else ()


def _str_sequence(obj: object, name: str) -> list[str]:
    return [item for item in _sequence_attr(obj, name) if isinstance(item, str) and item]


def _member_names(parsed: BaseModel) -> list[str]:
    """Route names from a ``multi_route`` output's members (names or objects)."""
    names: list[str] = []
    for item in _sequence_attr(parsed, "routes"):
        if isinstance(item, str):
            if item:
                names.append(item)
            continue
        member = getattr(item, "route", None)
        if isinstance(member, str) and member:
            names.append(member)
    return list(dict.fromkeys(names))
