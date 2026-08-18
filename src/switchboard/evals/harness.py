"""The v0.1 eval runner: metrics, the naive baseline, and the delta (plan §9).

The report's adoption wedge is accuracy credibility *on the user's own catalog*,
and its ship rule is blunt: **if you cannot beat naive ``.with_structured_output()``
meaningfully on a 100+ route catalog, do not ship.** So the harness is a product
surface, not test infrastructure, and its headline number is a *delta against the
naive baseline* rather than an absolute accuracy that anyone can make look good by
choosing an easy catalog.

Two design rules follow, and everything here obeys them:

1. **No separate instrumentation path.** Every metric is read off the
   :data:`~switchboard.core.decision.Decision` and the
   :class:`~switchboard.core.audit.AuditRecord` that production already emits
   (§13 ruling #6). If a number cannot be derived from those, it is not measured —
   otherwise the harness would be scoring a code path users do not run.
2. **The baseline shares everything except architecture.** Same
   :class:`~switchboard.providers.base.LLMClient`, same model, same decision
   schema (§9.3). The only differences are the ones under test: no shortlist, the
   whole entitled catalog in the prompt, no rationale-first system block, no
   confidence machinery. A baseline handicapped on schema would flatter the
   candidate and prove nothing.

v0.1 metrics are §9.2's minimal set — route accuracy, schema validity,
hallucinated route names, shortlist recall@K, clarify/abstain confusion, and
latency percentiles — plus the proto-G1/G4 gates §9.5 runs in the v0.1 dogfood.
Bootstrap CIs, arg exact match, ECE/AURC, cost-per-decision and the full gate set
are [v0.2].

**Packaging.** Pydantic + stdlib only. :meth:`SuiteResult.summary_table` renders
with ``str.format``, not ``rich`` — the ``[eval]`` extra is the [v0.2] CLI (§13
ruling #17).
"""

from __future__ import annotations

import asyncio
import copy
import math
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from switchboard.core.context import RequestContext
from switchboard.errors import ConfigError
from switchboard.evals.cache import ReplayCache, ReplayClient, sample_scope
from switchboard.evals.fixtures import (
    EvalCase,
    EvalSuite,
    ExpectedClarify,
    ExpectedMultiRoute,
    ExpectedRoute,
)
from switchboard.router import Router

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from switchboard.core.decision import Decision

__all__ = [
    "BASELINES",
    "NAIVE_FULL_CATALOG",
    "NAIVE_SYSTEM_PROMPT",
    "ArmResult",
    "CaseResult",
    "GateResult",
    "SuiteMetrics",
    "SuiteResult",
    "arun_suite",
    "build_naive_baseline",
    "run_suite",
]

NAIVE_FULL_CATALOG = "naive-full-catalog"
"""The report's mandated bar (plan §9.3). The only baseline implemented in v0.1."""

BASELINES: tuple[str, ...] = (NAIVE_FULL_CATALOG,)
"""Baselines this build can run. ``embed-top1`` and ``shortlist-oracle`` are
[v0.2] (§9.3) and are refused by name rather than silently skipped."""

_V02_BASELINES: tuple[str, ...] = ("embed-top1", "shortlist-oracle")

_SCHEMA_FAILURE_REASONS = frozenset(
    {"unparseable_output", "invalid_route_reference", "invalid_args"}
)
"""Abstain reasons that mean *the model's output was not usable as written*.

Read off ``AuditRecord.abstain_reason``, which survives even when the outcome was
downgraded to clarify or substituted by a fallback (§13 ruling #5), so the
schema-validity rate does not quietly improve because a fallback covered for it.
"""

_ROUTE_KINDS = frozenset({"route", "multi_route"})

_MIN_BINDING_CASES = 200
"""Plan §9.2's statistical floor: gates bind at n >= 200, advisory below."""

_G1_MARGIN = 0.10
"""Plan §9.5 G1: accuracy >= naive + 10 pp."""

_G4_RECALL = 0.95
"""Plan §9.5 G4: shortlist recall@K >= 95%."""


NAIVE_SYSTEM_PROMPT = """\
You are a tool router. Below is a catalog of tools. Pick the one that matches the
user's request and fill in its arguments.

Reply with a single JSON object matching the required schema. Choose only from the
tool names listed in the catalog."""
"""Segment A for the :data:`NAIVE_FULL_CATALOG` baseline (plan §9.3).

What a developer hand-rolls: name the tools, ask for structured output, stop.
Deliberately missing everything the candidate's system block spends tokens on —
the reason-then-commit ordering, the "order carries no information" instruction,
the clarify-is-cheaper-than-guessing rule, the never-invent-an-argument rule and
the untrusted-input framing. Those are architecture, and architecture is what the
delta is supposed to be measuring.
"""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class CaseResult(BaseModel):
    """One case, scored (plan §9.2). Every field is derived from the audit record."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    expected_kind: str
    predicted_kind: str
    """The four :data:`~switchboard.core.audit.DecisionKind` values, or ``"error"``
    when the loop raised — an unhandled provider failure is a data point, not a
    reason to lose the run."""
    predicted_routes: list[str] = Field(default_factory=list)
    correct: bool = False
    schema_valid: bool = False
    validation_retries: int = 0
    hallucinated_route: bool = False
    """A committed route name that is not in the registry. **Must be 0 across a
    run**: the validator re-checks membership even under a ``Literal`` wire enum
    (§4.5), so a nonzero count is a library bug, not a model failure."""
    invalid_route_reference: bool = False
    """The model *named* an unknown route and the validator caught it. This is the
    metric that actually moves: it is the hallucination rate the enum suppresses."""
    gold_in_shortlist: bool | None = None
    """``None`` when the case has no gold route, or when retrieval was skipped
    (small-catalog bypass / ``shortlist=None``) and recall is 1.0 by construction."""
    shortlist_size: int = 0
    shortlist_skipped: bool = False
    weak_retrieval: bool = False
    decision_path: str = "llm"
    abstain_reason: str | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    decision_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile over ``values``.

    Nearest-rank rather than interpolated: at the tens-to-hundreds of cases an
    eval suite has, an interpolated p95 invents a latency no request experienced.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _ratio(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, or 0.0 when there is nothing to divide."""
    return numerator / denominator if denominator else 0.0


class SuiteMetrics(BaseModel):
    """The v0.1 metric set for one arm (plan §9.2 minimal)."""

    model_config = ConfigDict(frozen=True)

    n_cases: int = 0
    n_errors: int = 0

    accuracy: float = 0.0
    """Headline: correct decisions over **all** cases, where correctness is judged
    per gold kind (route in ``any_of``, multi-route set match, clarify, abstain).
    Clarify and abstain are gold labels (§9.1), so scoring only route cases would
    reward a router that never asks a question."""
    route_accuracy: float = 0.0
    """§9.2's primary metric verbatim: ``kind == "route"`` and route in ``any_of``,
    over the ``expected.kind == "route"`` cases only."""
    n_route_cases: int = 0

    schema_validity: float = 0.0
    """Fraction of cases whose model output was usable after the §4.5 repair loop."""
    first_pass_schema_validity: float = 0.0
    """...and usable with **zero** repair attempts. The gap between the two is the
    cost the validate-and-retry loop is paying on this provider."""
    schema_repairs: int = 0
    hallucinated_routes: int = 0
    """Committed routes absent from the registry. Must be 0."""
    invalid_route_references: int = 0

    recall_at_k: float | None = None
    """Gold route present in the shortlist (§9.2, §9.5 G4). ``None`` when no case
    exercised retrieval — the naive baseline has no shortlist, so its recall is
    not 1.0, it is *undefined*, and reporting 1.0 would fake a tie on G4."""
    n_recall_cases: int = 0
    n_shortlist_skipped: int = 0

    clarify_precision: float | None = None
    clarify_recall: float | None = None
    abstain_precision: float | None = None
    abstain_recall: float | None = None
    harmful_clarify: int = 0
    """Clarified when the gold label was an unambiguous route (§9.2)."""
    missed_clarify: int = 0
    over_abstain: int = 0
    missed_abstain: int = 0
    confusion: dict[str, int] = Field(default_factory=dict)
    """``"gold->predicted"`` counts, including ``"...->error"``."""

    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_mean_ms: float | None = None

    @classmethod
    def from_cases(cls, cases: Sequence[CaseResult]) -> SuiteMetrics:
        """Aggregate scored cases into the metric block (plan §9.2)."""
        total = len(cases)
        route_cases = [c for c in cases if c.expected_kind == "route"]
        recall_cases = [c for c in cases if c.gold_in_shortlist is not None]

        confusion: dict[str, int] = {}
        for case in cases:
            label = f"{case.expected_kind}->{case.predicted_kind}"
            confusion[label] = confusion.get(label, 0) + 1

        predicted_clarify = [c for c in cases if c.predicted_kind == "clarify"]
        gold_clarify = [c for c in cases if c.expected_kind == "clarify"]
        predicted_abstain = [c for c in cases if c.predicted_kind == "abstain"]
        gold_abstain = [c for c in cases if c.expected_kind == "abstain"]
        latencies = [c.latency_ms for c in cases if c.error is None]

        return cls(
            n_cases=total,
            n_errors=sum(1 for c in cases if c.error is not None),
            accuracy=_ratio(sum(1 for c in cases if c.correct), total),
            route_accuracy=_ratio(
                sum(1 for c in route_cases if c.correct), len(route_cases)
            ),
            n_route_cases=len(route_cases),
            schema_validity=_ratio(sum(1 for c in cases if c.schema_valid), total),
            first_pass_schema_validity=_ratio(
                sum(1 for c in cases if c.schema_valid and c.validation_retries == 0),
                total,
            ),
            schema_repairs=sum(c.validation_retries for c in cases),
            hallucinated_routes=sum(1 for c in cases if c.hallucinated_route),
            invalid_route_references=sum(1 for c in cases if c.invalid_route_reference),
            recall_at_k=(
                _ratio(sum(1 for c in recall_cases if c.gold_in_shortlist), len(recall_cases))
                if recall_cases
                else None
            ),
            n_recall_cases=len(recall_cases),
            n_shortlist_skipped=sum(1 for c in cases if c.shortlist_skipped),
            clarify_precision=(
                _ratio(sum(1 for c in predicted_clarify if c.expected_kind == "clarify"),
                       len(predicted_clarify))
                if predicted_clarify
                else None
            ),
            clarify_recall=(
                _ratio(sum(1 for c in gold_clarify if c.predicted_kind == "clarify"),
                       len(gold_clarify))
                if gold_clarify
                else None
            ),
            abstain_precision=(
                _ratio(sum(1 for c in predicted_abstain if c.expected_kind == "abstain"),
                       len(predicted_abstain))
                if predicted_abstain
                else None
            ),
            abstain_recall=(
                _ratio(sum(1 for c in gold_abstain if c.predicted_kind == "abstain"),
                       len(gold_abstain))
                if gold_abstain
                else None
            ),
            harmful_clarify=sum(
                1
                for c in cases
                if c.expected_kind in _ROUTE_KINDS and c.predicted_kind == "clarify"
            ),
            missed_clarify=sum(
                1
                for c in cases
                if c.expected_kind == "clarify" and c.predicted_kind in _ROUTE_KINDS
            ),
            over_abstain=sum(
                1
                for c in cases
                if c.expected_kind != "abstain" and c.predicted_kind == "abstain"
            ),
            missed_abstain=sum(
                1
                for c in cases
                if c.expected_kind == "abstain" and c.predicted_kind in _ROUTE_KINDS
            ),
            confusion=dict(sorted(confusion.items())),
            latency_p50_ms=_percentile(latencies, 0.50),
            latency_p95_ms=_percentile(latencies, 0.95),
            latency_mean_ms=(sum(latencies) / len(latencies)) if latencies else None,
        )


class ArmResult(BaseModel):
    """One arm of a run — the candidate router, or a baseline (plan §9.3)."""

    model_config = ConfigDict(frozen=True)

    name: str
    metrics: SuiteMetrics
    cases: list[CaseResult] = Field(default_factory=list)
    model: str | None = None
    elapsed_s: float = 0.0

    def by_id(self) -> dict[str, CaseResult]:
        """Per-case results keyed by case id, for a per-case diff against another arm."""
        return {case.case_id: case for case in self.cases}


class GateResult(BaseModel):
    """One proto-gate outcome (plan §9.5; G1/G4 run in the v0.1 dogfood)."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    value: float | None
    threshold: float
    passed: bool | None
    """``None`` when the gate is not computable — no baseline ran, or no case
    exercised retrieval. Never silently ``True``."""
    binding: bool
    """False below :data:`_MIN_BINDING_CASES`; the gate is advisory then (§9.2)."""
    signed: bool = False
    """Render the value with an explicit sign. True for margin gates (G1's ``+10
    pp`` is a difference), false for level gates (G4's 95% is not)."""
    detail: str = ""

    def line(self) -> str:
        """One rendered row for :meth:`SuiteResult.summary_table`."""
        status = "N/A " if self.passed is None else ("PASS" if self.passed else "FAIL")
        if not self.binding and self.passed is not None:
            status += " (advisory)"
        style = "+.2%" if self.signed else ".2%"
        shown = "--" if self.value is None else format(self.value, style)
        bar = format(self.threshold, style)
        return (
            f"  {self.id}  {status:<16} {self.name}: {shown} (bar {bar}) {self.detail}"
        ).rstrip()


class SuiteResult(BaseModel):
    """A complete run: the candidate arm, its baselines, and the delta (plan §9)."""

    model_config = ConfigDict(frozen=True)

    suite: str
    n_cases: int
    registry_version: str
    n_routes: int
    seed: int | None = None
    mode: str = "live"
    """``"live"`` or ``"replay"``/``"record"``/``"auto"`` when a replay cache drove
    the run. Latency percentiles from a replayed run measure this machine, not a
    provider — the field is here so a report cannot claim otherwise (§9.2)."""
    candidate: ArmResult
    baselines: list[ArmResult] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    cache_stats: dict[str, int] | None = None
    elapsed_s: float = 0.0

    # -- the headline -------------------------------------------------------- #

    def arm(self, name: str) -> ArmResult | None:
        """Look up an arm by name (``"candidate"`` or a baseline name)."""
        if name in ("candidate", self.candidate.name):
            return self.candidate
        return next((arm for arm in self.baselines if arm.name == name), None)

    def delta(
        self, metric: str = "accuracy", baseline: str = NAIVE_FULL_CATALOG
    ) -> float | None:
        """Candidate minus baseline on ``metric`` — **the headline number** (§9.3).

        ``None`` when the baseline did not run or the metric is undefined on one
        of the arms. An absent baseline never reads as a zero delta: the report's
        ship rule is stated as a margin over naive, so "no baseline" means "no
        answer", not "tied".
        """
        other = self.arm(baseline)
        if other is None:
            return None
        mine = getattr(self.candidate.metrics, metric, None)
        theirs = getattr(other.metrics, metric, None)
        if not isinstance(mine, (int, float)) or not isinstance(theirs, (int, float)):
            return None
        return float(mine) - float(theirs)

    @property
    def headline(self) -> str:
        """One sentence: the delta the report asks to be shown first (§9.3)."""
        delta = self.delta()
        if delta is None:
            return (
                f"accuracy {self.candidate.metrics.accuracy:.1%} "
                f"(no {NAIVE_FULL_CATALOG} baseline ran; the report's bar is a "
                f"margin over naive, so there is no headline number)"
            )
        naive = self.arm(NAIVE_FULL_CATALOG)
        naive_accuracy = naive.metrics.accuracy if naive else 0.0
        return (
            f"accuracy {self.candidate.metrics.accuracy:.1%} vs "
            f"{naive_accuracy:.1%} naive-full-catalog "
            f"= {delta * 100:+.2f} pp on {self.n_cases} cases / "
            f"{self.n_routes} routes"
        )

    # -- output -------------------------------------------------------------- #

    def summary_table(self) -> str:
        """Render the run as a plain-text table (plan §9).

        Pure ``str.format`` — no ``rich``, because the fixture/runner half of the
        harness is core and the renderer must not drag in the [v0.2] ``[eval]``
        extra (§13 ruling #17).
        """
        arms = [self.candidate, *self.baselines]
        names = [arm.name for arm in arms]
        width = max(22, *(len(name) for name in names))

        lines: list[str] = []
        lines.append(
            f"suite: {self.suite}  |  cases: {self.n_cases}  |  routes: {self.n_routes}"
            f"  |  registry: {self.registry_version}  |  mode: {self.mode}"
            + (f"  |  seed: {self.seed}" if self.seed is not None else "")
        )
        lines.append("")
        header = f"{'metric':<30}" + "".join(f"{name:>{width}}" for name in names)
        if len(arms) > 1:
            header += f"{'delta':>14}"
        lines.append(header)
        lines.append("-" * len(header))

        for label, attribute, style in _TABLE_ROWS:
            values = [getattr(arm.metrics, attribute) for arm in arms]
            row = f"{label:<30}" + "".join(
                f"{_fmt(value, style):>{width}}" for value in values
            )
            if len(arms) > 1:
                row += f"{_fmt_delta(values[0], values[1], style):>14}"
            lines.append(row)

        lines.append("")
        lines.append(f"headline: {self.headline}")
        if self.gates:
            lines.append("")
            lines.append(f"proto-gates (plan §9.5; binding at n >= {_MIN_BINDING_CASES}):")
            lines.extend(gate.line() for gate in self.gates)
            if self.n_cases < _MIN_BINDING_CASES:
                lines.append(
                    f"  WARNING: {self.n_cases} cases is below the n >= "
                    f"{_MIN_BINDING_CASES} statistical floor (plan §9.2). These "
                    f"gates are advisory; do not ship on them."
                )
        confusion = self.candidate.metrics.confusion
        if confusion:
            lines.append("")
            lines.append(
                "confusion (gold->predicted): "
                + ", ".join(f"{label} {count}" for label, count in confusion.items())
            )
        if self.cache_stats:
            lines.append(
                "replay cache: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.cache_stats.items()))
            )
        return "\n".join(lines)

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialise the whole run, optionally writing it to ``path``.

        Returns the JSON text either way, so it composes with a pipe as well as
        with a file. Parent directories are created.
        """
        payload = self.model_dump_json(indent=indent)
        if path is not None:
            file = Path(path)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(payload + "\n", encoding="utf-8")
        return payload

    def __str__(self) -> str:
        return self.summary_table()


_TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("accuracy", "accuracy", "pct"),
    ("route accuracy", "route_accuracy", "pct"),
    ("schema validity", "schema_validity", "pct"),
    ("schema validity (1st pass)", "first_pass_schema_validity", "pct"),
    ("shortlist recall@K", "recall_at_k", "pct"),
    ("hallucinated routes", "hallucinated_routes", "int"),
    ("invalid route references", "invalid_route_references", "int"),
    ("harmful clarify", "harmful_clarify", "int"),
    ("missed clarify", "missed_clarify", "int"),
    ("over abstain", "over_abstain", "int"),
    ("missed abstain", "missed_abstain", "int"),
    ("clarify precision", "clarify_precision", "pct"),
    ("clarify recall", "clarify_recall", "pct"),
    ("errors", "n_errors", "int"),
    ("latency p50 (ms)", "latency_p50_ms", "ms"),
    ("latency p95 (ms)", "latency_p95_ms", "ms"),
)


def _fmt(value: Any, style: str) -> str:
    """Format one table cell, with ``--`` for an undefined metric."""
    if value is None:
        return "--"
    if style == "pct":
        return f"{value:.2%}"
    if style == "ms":
        return f"{value:.1f}"
    return f"{value:d}"


def _fmt_delta(mine: Any, theirs: Any, style: str) -> str:
    """Format the candidate-minus-baseline cell."""
    if mine is None or theirs is None:
        return "--"
    difference = float(mine) - float(theirs)
    if style == "pct":
        return f"{difference * 100:+.2f} pp"
    if style == "ms":
        return f"{difference:+.1f}"
    return f"{int(difference):+d}"


# --------------------------------------------------------------------------- #
# Scoring — everything below reads the Decision + AuditRecord and nothing else.
# --------------------------------------------------------------------------- #


def _score(
    case: EvalCase,
    decision: Decision | None,
    *,
    registry_names: frozenset[str],
    wall_ms: float,
    error: str | None,
) -> CaseResult:
    """Score one case against its gold label (plan §9.2)."""
    expected = case.expected
    tags = sorted(case.tags)

    if decision is None:
        return CaseResult(
            case_id=case.id,
            query=case.query,
            expected_kind=expected.kind,
            predicted_kind="error",
            correct=False,
            schema_valid=False,
            latency_ms=wall_ms,
            tags=tags,
            error=error,
        )

    audit = decision.audit
    predicted = list(audit.routes)
    kind = audit.kind

    if isinstance(expected, ExpectedRoute):
        correct = kind == "route" and bool(predicted) and predicted[0] in expected.any_of
    elif isinstance(expected, ExpectedMultiRoute):
        if kind != "multi_route":
            correct = False
        elif expected.order_sensitive:
            correct = predicted == list(expected.routes)
        else:
            correct = set(predicted) == set(expected.routes)
    elif isinstance(expected, ExpectedClarify):
        # A route that the labeller marked non-harmful counts (§9.1): scoring it
        # as a miss would push the router toward asking questions it need not ask.
        correct = kind == "clarify" or (
            kind == "route" and bool(predicted) and predicted[0] in expected.acceptable_routes
        )
    else:
        # A configured fallback turns a terminal abstain into kind="route" with
        # decision_path="fallback" (§13 ruling #5). That is the router doing
        # exactly what it was told to do on an out-of-scope query, so it is
        # scored as a hit; audit.abstain_reason preserves what really happened.
        correct = kind == "abstain" or decision.decision_path == "fallback"

    gold = case.gold_routes()
    shortlist_names = {candidate.route_name for candidate in audit.shortlist}
    gold_in_shortlist: bool | None = None
    if gold and not audit.shortlist_skipped and shortlist_names:
        gold_in_shortlist = bool(gold & shortlist_names)

    return CaseResult(
        case_id=case.id,
        query=case.query,
        expected_kind=expected.kind,
        predicted_kind=kind,
        predicted_routes=predicted,
        correct=correct,
        schema_valid=audit.abstain_reason not in _SCHEMA_FAILURE_REASONS,
        validation_retries=audit.validation_retries,
        hallucinated_route=any(name not in registry_names for name in predicted),
        invalid_route_reference=audit.abstain_reason == "invalid_route_reference",
        gold_in_shortlist=gold_in_shortlist,
        shortlist_size=len(audit.shortlist),
        shortlist_skipped=audit.shortlist_skipped,
        weak_retrieval=audit.weak_retrieval,
        decision_path=audit.decision_path,
        abstain_reason=audit.abstain_reason,
        confidence=audit.confidence.score if audit.confidence is not None else None,
        latency_ms=audit.latency.total_ms,
        decision_id=audit.decision_id,
        tags=tags,
    )


def _context_for(case: EvalCase) -> RequestContext | None:
    """Build the case's :class:`~switchboard.core.context.RequestContext`."""
    if not case.context:
        return None
    try:
        return RequestContext.model_validate(case.context)
    except Exception as exc:
        raise ConfigError(
            f"eval case {case.id!r} has a context that is not a RequestContext: {exc}"
        ) from exc


def _run_case(router: Router, case: EvalCase, registry_names: frozenset[str]) -> CaseResult:
    """Run one case through ``router.route()`` and score it."""
    context = _context_for(case)
    started = time.perf_counter()
    with sample_scope():
        try:
            decision: Decision | None = router.route(case.query, context=context)
            error: str | None = None
        except ConfigError:
            # Broken configuration — including a strict replay-cache miss — makes
            # the whole run meaningless, so it is never swallowed into a per-case
            # error column (plan §3.8, §9.6).
            raise
        except Exception as exc:
            decision, error = None, f"{type(exc).__name__}: {exc}"
    wall_ms = (time.perf_counter() - started) * 1000.0
    return _score(case, decision, registry_names=registry_names, wall_ms=wall_ms, error=error)


async def _arun_case(
    router: Router, case: EvalCase, registry_names: frozenset[str]
) -> CaseResult:
    """Async twin of :func:`_run_case` (plan §2.5)."""
    context = _context_for(case)
    started = time.perf_counter()
    with sample_scope():
        try:
            decision: Decision | None = await router.aroute(case.query, context=context)
            error: str | None = None
        except ConfigError:
            raise
        except Exception as exc:
            decision, error = None, f"{type(exc).__name__}: {exc}"
    wall_ms = (time.perf_counter() - started) * 1000.0
    return _score(case, decision, registry_names=registry_names, wall_ms=wall_ms, error=error)


# --------------------------------------------------------------------------- #
# Baselines (plan §9.3)
# --------------------------------------------------------------------------- #


def build_naive_baseline(router: Router, *, seed: int | None = None) -> Router:
    """Build the ``naive-full-catalog`` arm from ``router`` (plan §9.3).

    *"What a developer hand-rolls: one structured-output call, entire
    entitlement-filtered catalog in the prompt, no shortlist, no rationale-first
    layout, no confidence machinery. The report's mandated bar."*

    Held **identical** to the candidate — same client, same model, same enabled
    decision kinds, same wire-schema mode, same retry budget, same temperature and
    max_tokens — so the measured difference is architecture and not schema
    asymmetry (§9.3). Changed, deliberately:

    * ``shortlist=None`` with ``max_candidates=len(registry)`` — the whole catalog
      goes in, which at 100+ routes is the degradation the report documents;
    * ``candidate_order="registry"`` — no seeded shuffle, so position bias is left
      in place;
    * ``system_prompt=NAIVE_SYSTEM_PROMPT`` — no rationale-first block;
    * ``confidence="none"`` and ``fallback=None`` — no thresholds, no clarify
      downgrade, no fallback substitution.

    The oversized-``max_candidates`` warning §3.3 raises is suppressed here: it is
    warning about precisely the thing this baseline exists to demonstrate.
    """
    registry = router.registry
    config = router.config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Router(
            registry,
            client=router.client,
            shortlist=None,
            max_candidates=max(len(registry), 1),
            candidate_order="registry",
            multi_route=config.multi_route,
            allow_clarify=config.allow_clarify,
            fallback=None,
            confidence="none",
            retry=config.retry,
            wire_schema=config.wire_schema,
            content_mode=config.content_mode,
            on_provider_error=config.on_provider_error,
            system_prompt=NAIVE_SYSTEM_PROMPT,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=seed if seed is not None else config.seed,
            sink=None,
            otel=False,
        )


def _resolve_baseline(name: str, router: Router, seed: int | None) -> Router:
    """Build the named baseline arm, or explain why this build cannot (§9.3)."""
    if name == NAIVE_FULL_CATALOG:
        return build_naive_baseline(router, seed=seed)
    if name in _V02_BASELINES:
        raise ConfigError(
            f"baseline {name!r} is [v0.2] and is not implemented in v0.1 (plan §9.3). "
            f"v0.1 ships {NAIVE_FULL_CATALOG!r}, the report's mandated bar."
        )
    raise ConfigError(
        f"unknown baseline {name!r}; this build runs {list(BASELINES)} "
        f"({list(_V02_BASELINES)} are [v0.2])."
    )


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _as_suite(suite: EvalSuite | Iterable[EvalCase]) -> EvalSuite:
    """Accept an :class:`EvalSuite` or any iterable of cases."""
    if isinstance(suite, EvalSuite):
        return suite
    cases = list(suite)
    for case in cases:
        if not isinstance(case, EvalCase):
            raise ConfigError(
                f"run_suite(suite=...) takes an EvalSuite or an iterable of "
                f"EvalCase, found {type(case).__name__}"
            )
    return EvalSuite(name="suite", cases=cases)


def _with_client(router: Router, client: Any) -> Router:
    """A shallow clone of ``router`` driving ``client`` instead.

    The Router holds no per-request state (§2.5) — the registry is frozen, the
    ``LoopConfig`` is frozen — so swapping the client on a copy is safe and, more
    to the point, leaves the caller's Router untouched. Rebuilding it from
    ``LoopConfig`` instead would silently drop anything the config does not carry
    (the redactor, the sink, the emitter).
    """
    clone = copy.copy(router)
    clone._client = client
    clone._sync_capable = callable(getattr(client, "complete", None))
    clone._async_capable = callable(getattr(client, "acomplete", None))
    return clone


def _prepare(
    router: Router,
    suite: EvalSuite | Iterable[EvalCase],
    baselines: Sequence[str],
    cache: ReplayCache | str | Path | None,
    seed: int | None,
) -> tuple[EvalSuite, Router, list[tuple[str, Router]], ReplayCache | None, str]:
    """Resolve the suite, the cache wiring and every arm, before anything runs."""
    resolved = _as_suite(suite)
    if not isinstance(router, Router):
        raise ConfigError(
            f"run_suite(router=...) takes a Router, got {type(router).__name__}"
        )

    replay_cache: ReplayCache | None = None
    mode = "live"
    candidate = router
    if cache is not None:
        existing = router.client
        if isinstance(existing, ReplayClient):
            replay_cache = existing.cache
            mode = existing.mode
        else:
            replay_cache = cache if isinstance(cache, ReplayCache) else ReplayCache(cache)
            candidate = _with_client(
                router, ReplayClient(existing, cache=replay_cache, mode="auto")
            )
            mode = "auto"
    elif isinstance(router.client, ReplayClient):
        replay_cache = router.client.cache
        mode = router.client.mode

    arms = [(name, _resolve_baseline(name, candidate, seed)) for name in baselines]
    return resolved, candidate, arms, replay_cache, mode


def _assemble(
    suite: EvalSuite,
    router: Router,
    candidate_cases: list[CaseResult],
    candidate_elapsed: float,
    baseline_arms: list[tuple[str, list[CaseResult], float]],
    *,
    seed: int | None,
    mode: str,
    cache: ReplayCache | None,
    elapsed: float,
) -> SuiteResult:
    """Build the :class:`SuiteResult` from the scored arms."""
    model = router.config.request_model
    candidate = ArmResult(
        name="candidate",
        metrics=SuiteMetrics.from_cases(candidate_cases),
        cases=candidate_cases,
        model=model,
        elapsed_s=candidate_elapsed,
    )
    baselines = [
        ArmResult(
            name=name,
            metrics=SuiteMetrics.from_cases(cases),
            cases=cases,
            model=model,
            elapsed_s=arm_elapsed,
        )
        for name, cases, arm_elapsed in baseline_arms
    ]
    result = SuiteResult(
        suite=suite.name,
        n_cases=len(suite.cases),
        registry_version=router.registry.version,
        n_routes=len(router.registry),
        seed=seed,
        mode=mode,
        candidate=candidate,
        baselines=baselines,
        cache_stats=cache.stats() if cache is not None else None,
        elapsed_s=elapsed,
    )
    return result.model_copy(update={"gates": _proto_gates(result)})


def _proto_gates(result: SuiteResult) -> list[GateResult]:
    """Evaluate proto-G1 and proto-G4 (plan §9.5, run in the v0.1 dogfood)."""
    binding = result.n_cases >= _MIN_BINDING_CASES
    delta = result.delta("accuracy", NAIVE_FULL_CATALOG)
    recall = result.candidate.metrics.recall_at_k
    return [
        GateResult(
            id="G1",
            name=f"accuracy delta vs {NAIVE_FULL_CATALOG}",
            value=delta,
            threshold=_G1_MARGIN,
            passed=None if delta is None else delta >= _G1_MARGIN,
            binding=binding,
            signed=True,
            detail=(
                "" if delta is not None else f"— no {NAIVE_FULL_CATALOG} arm ran"
            ),
        ),
        GateResult(
            id="G4",
            name="shortlist recall@K",
            value=recall,
            threshold=_G4_RECALL,
            passed=None if recall is None else recall >= _G4_RECALL,
            binding=binding,
            detail=(
                ""
                if recall is not None
                else "— no case exercised retrieval (shortlist disabled or bypassed)"
            ),
        ),
    ]


def run_suite(
    router: Router,
    suite: EvalSuite | Iterable[EvalCase],
    *,
    baselines: Sequence[str] = BASELINES,
    seed: int | None = None,
    cache: ReplayCache | str | Path | None = None,
) -> SuiteResult:
    """Run ``suite`` through ``router`` and every baseline, and score it (plan §9).

    Example::

        suite  = dogfood_suite(n_routes=120)
        router = Router(suite.registry(), client=client, shortlist="auto")
        result = run_suite(router, suite, cache="tests/fixtures/replay")
        print(result.summary_table())
        result.to_json("eval-run.json")

    Args:
        router: the router under test. Left untouched — a ``cache`` is applied to
            a shallow clone.
        suite: an :class:`~switchboard.evals.fixtures.EvalSuite` or any iterable of
            :class:`~switchboard.evals.fixtures.EvalCase`.
        baselines: baseline names to run alongside. Defaults to
            :data:`BASELINES` — ``naive-full-catalog``, which the report mandates
            as the bar. Pass ``()`` to skip it and lose the headline number.
        seed: provider ``seed`` for the baseline arm, and stamped on the result.
            The candidate's seed is whatever its Router was built with; the
            candidate-order shuffle is seeded from a content digest either way
            (§13 ruling #8), so a run is reproducible without this.
        cache: a :class:`~switchboard.evals.cache.ReplayCache` or a path. Wraps
            the router's client in ``mode="auto"`` — hits replay, misses record.
            For a strict, keyless CI lane, build the Router with
            ``ReplayClient(..., mode="replay")`` yourself; that mode is detected
            and reported rather than overridden.

    Returns:
        A :class:`SuiteResult`. Its headline is the **delta versus the naive
        baseline**, not the candidate's absolute accuracy.

    Raises:
        ConfigError: an unknown/[v0.2] baseline, a malformed suite, a bad case
            context, or a strict replay-cache miss (which fails the run by design).
    """
    started = time.perf_counter()
    resolved, candidate, arms, replay_cache, mode = _prepare(
        router, suite, baselines, cache, seed
    )
    registry_names = frozenset(candidate.registry.names)

    arm_started = time.perf_counter()
    candidate_cases = [_run_case(candidate, case, registry_names) for case in resolved.cases]
    candidate_elapsed = time.perf_counter() - arm_started

    baseline_arms: list[tuple[str, list[CaseResult], float]] = []
    for name, arm_router in arms:
        arm_started = time.perf_counter()
        names = frozenset(arm_router.registry.names)
        baseline_arms.append(
            (
                name,
                [_run_case(arm_router, case, names) for case in resolved.cases],
                time.perf_counter() - arm_started,
            )
        )

    return _assemble(
        resolved,
        candidate,
        candidate_cases,
        candidate_elapsed,
        baseline_arms,
        seed=seed,
        mode=mode,
        cache=replay_cache,
        elapsed=time.perf_counter() - started,
    )


async def arun_suite(
    router: Router,
    suite: EvalSuite | Iterable[EvalCase],
    *,
    baselines: Sequence[str] = BASELINES,
    seed: int | None = None,
    cache: ReplayCache | str | Path | None = None,
    concurrency: int = 8,
) -> SuiteResult:
    """Async twin of :func:`run_suite`, concurrency-limited (plan §9, §2.5).

    Cases within an arm run concurrently under a semaphore; arms run in sequence,
    because two arms sharing one client would interleave their provider calls and
    make the per-arm latency percentiles meaningless.

    Concurrency does **not** perturb the replay cache: ``sample_index`` is scoped
    per case through a :class:`~contextvars.ContextVar` (see
    :func:`~switchboard.evals.cache.sample_scope`), so cache keys do not depend on
    how the scheduler interleaved the run.

    Args:
        concurrency: maximum in-flight cases per arm. Keep it under the provider's
            rate limit; a replayed run is CPU-bound and does not benefit much
            above a handful.
    """
    if concurrency < 1:
        raise ConfigError(f"arun_suite(concurrency={concurrency}) must be >= 1")

    started = time.perf_counter()
    resolved, candidate, arms, replay_cache, mode = _prepare(
        router, suite, baselines, cache, seed
    )

    async def run_arm(arm_router: Router) -> tuple[list[CaseResult], float]:
        names = frozenset(arm_router.registry.names)
        limiter = asyncio.Semaphore(concurrency)

        async def one(case: EvalCase) -> CaseResult:
            async with limiter:
                return await _arun_case(arm_router, case, names)

        arm_started = time.perf_counter()
        cases = await asyncio.gather(*(one(case) for case in resolved.cases))
        return list(cases), time.perf_counter() - arm_started

    candidate_cases, candidate_elapsed = await run_arm(candidate)
    baseline_arms: list[tuple[str, list[CaseResult], float]] = []
    for name, arm_router in arms:
        cases, arm_elapsed = await run_arm(arm_router)
        baseline_arms.append((name, cases, arm_elapsed))

    return _assemble(
        resolved,
        candidate,
        candidate_cases,
        candidate_elapsed,
        baseline_arms,
        seed=seed,
        mode=mode,
        cache=replay_cache,
        elapsed=time.perf_counter() - started,
    )
