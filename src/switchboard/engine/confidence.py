"""Confidence signals, fusion, and the no-signal rule (plan §6.2, §6.3).

The design stance is the report's Recommendation #4 and it is worth restating
because every function here follows from it: **logprobs discriminate correctness
better than verbalized scores**, verbalized confidence "tracks commitment more
than correctness", and models do not reliably convert a stated confidence into an
abstain decision. So confidence is computed **outside the model** wherever the
provider path allows it, every signal is advisory, fusion is conservative (a
``min``, never an average), and the resulting scalar is explicitly *not* a
calibrated probability — it exists to be compared against thresholds that only
ever *downgrade* a decision (§6.4).

Signals and their phases:

============== ===== ===========================================================
Signal         Phase Notes
============== ===== ===========================================================
``p_route``    v0.1  geometric-mean token probability of the committed route
``margin``     v0.2  first-divergent-token gap; :func:`compute_margin` ships now,
                     the loop does not call it in v0.1
``agreement``  v0.2  self-consistency vote share
``stated``     v0.1  verbalized; recorded, damped x0.85, never solely trusted
============== ===== ===========================================================

Token alignment (:func:`compute_p_route`) is a **normative core utility**, not a
per-adapter helper (§6.2): every provider that returns token logprobs returns
them in the same ``TokenLP`` shape (§4.1, §13 ruling #1), so the alignment is
written once, here, and every adapter inherits identical semantics. When
alignment fails it returns ``None``. It never guesses — a wrong ``p_route`` is
worse than no ``p_route``, because thresholds act on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from switchboard.core.audit import ConfidenceReport

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from switchboard.providers.base import TokenLP

__all__ = [
    "build_confidence_report",
    "compute_margin",
    "compute_p_route",
    "default_fusion",
    "signals_are_actionable",
]

_STATED_DAMPING = 0.85
"""Verbalized confidence is damped before it can influence anything (§6.3,
Appendix A). It is the signal the evidence distrusts most."""

_OVERTURN_PENALTY = 0.8
"""A self-consistency majority that overturned the greedy vote is a strong
clarify indicator by itself (§6.2), so the fused score is cut (§6.3). [v0.2]"""


# --------------------------------------------------------------------------- #
# Token alignment (plan §6.2) — shared by p_route and margin.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Alignment:
    """Where the committed route value sits in the sampled token stream."""

    text: str
    """The concatenated token strings — the model's raw output as it was tokenised."""
    starts: tuple[int, ...]
    """Character offset at which each token begins in :attr:`text`."""
    ends: tuple[int, ...]
    """Character offset one past each token's last character."""
    name_start: int
    name_end: int
    """Half-open char span of the route *value* (the name, excluding its quotes)."""
    token_indices: tuple[int, ...]
    """Indices of every token overlapping that span — the tokens that spell the
    committed route, including any token that straddles the opening quote or the
    closing one (BPE splits mid-name, and a merge like ``'"ref'`` is routine)."""


def _align_route_value(
    token_logprobs: Sequence[TokenLP] | None,
    route_name: str | None,
) -> _Alignment | None:
    """Locate ``route_name``'s committed value in the token stream (plan §6.2).

    "Search for the quoted route value in the concatenated token string, map char
    offsets to token indices." Quoting matters: the value the model committed to
    is a JSON *string*, so it is searched for in quoted form and, failing that, in
    escaped-quote form (the payload is itself a JSON string on some tool-calling
    paths) and single-quoted form (non-strict rungs).

    The occurrence is anchored on the ``"route"`` key wherever that key is present,
    because a route name routinely appears in the rationale as well — and the
    rationale is emitted *before* the commitment (§3.4), so an unanchored search
    would happily measure the model's musings instead of its decision. Without an
    anchor the **last** occurrence wins, for the same ordering reason.

    Returns ``None`` whenever the value cannot be located unambiguously.
    """
    if not token_logprobs or not route_name:
        return None

    tokens = [lp.token for lp in token_logprobs]
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token)
        ends.append(cursor)
    text = "".join(tokens)

    span = _locate_value(text, route_name)
    if span is None:
        return None
    name_start, name_end = span

    indices = tuple(
        index
        for index in range(len(tokens))
        if ends[index] > name_start and starts[index] < name_end
    )
    if not indices:
        return None
    return _Alignment(
        text=text,
        starts=tuple(starts),
        ends=tuple(ends),
        name_start=name_start,
        name_end=name_end,
        token_indices=indices,
    )


def _locate_value(text: str, name: str) -> tuple[int, int] | None:
    """Return the half-open char span of ``name`` as a committed JSON value."""
    forms = (f'"{name}"', f'\\"{name}\\"', f"'{name}'")

    for key in ('"route"', '\\"route\\"', "'route'"):
        anchor = text.rfind(key)
        if anchor == -1:
            continue
        for form in forms:
            found = text.find(form, anchor + len(key))
            if found != -1:
                offset = form.index(name)
                return found + offset, found + offset + len(name)

    best = -1
    best_form: str | None = None
    for form in forms:
        found = text.rfind(form)
        if found > best:
            best, best_form = found, form
    if best >= 0 and best_form is not None:
        offset = best_form.index(name)
        return best + offset, best + offset + len(name)
    return None


# --------------------------------------------------------------------------- #
# Signal 1 — p_route [v0.1]
# --------------------------------------------------------------------------- #


def compute_p_route(
    token_logprobs: Sequence[TokenLP] | None,
    route_name: str | None,
) -> float | None:
    """Geometric-mean token probability of the committed route (plan §6.2). [v0.1]

    ``p_route = exp(mean(logprob(t) for t in route_name_tokens))``.

    Two properties make this the primary v0.1 signal:

    * **It is measured after the reasoning.** The wire schema emits ``rationale``
      before ``route`` (§3.4, §4.4), so these tokens are post-reasoning
      commitment tokens, not the model thinking out loud.
    * **It is length-normalised.** A geometric mean, not a product, so
      ``escalate_to_human_agent`` is not punished for being longer than ``refund``.

    Handles JSON quoting and BPE splits mid-name by mapping char offsets to token
    indices: any token overlapping the value's character span counts, whether it
    covers one character of the name or all of it plus the surrounding quotes.

    Returns:
        The probability in ``[0, 1]``, or ``None`` when there are no logprobs or
        alignment failed. ``None`` is a first-class answer — it makes the
        threshold machinery inert (:func:`signals_are_actionable`) rather than
        feeding it a fabricated number.
    """
    alignment = _align_route_value(token_logprobs, route_name)
    if alignment is None or token_logprobs is None:
        return None
    logprobs = [token_logprobs[index].logprob for index in alignment.token_indices]
    if not logprobs:
        return None
    mean = math.fsum(logprobs) / len(logprobs)
    # min(mean, 0.0) guards against providers that report un-normalised scores;
    # exp() of a non-positive mean is always a valid probability.
    return math.exp(min(mean, 0.0))


# --------------------------------------------------------------------------- #
# Signal 2 — margin [v0.2]
# --------------------------------------------------------------------------- #


def compute_margin(
    token_logprobs: Sequence[TokenLP] | None,
    route_name: str | None,
    candidates: Sequence[str] = (),
) -> float | None:
    """Probability gap at the first divergent token (plan §6.2). **[v0.2]**

    Implemented now, deliberately **not called by the v0.1 loop** (§13 ruling #9);
    ``ThresholdPolicy.margin_clarify_below`` is the gate that will consume it.

    Algorithm, exactly as specified:

    1. build a trie over the surviving candidate names — walked here as a prefix
       filter, since materialising nodes buys nothing at K ≤ 20 and the walk is
       the trie;
    2. find the first position where ≥2 candidates diverge (including the case
       where one candidate *ends* where another continues — closing the string is
       a decision too);
    3. locate the token covering that position and renormalise its
       ``top_logprobs`` mass over only the trie-valid continuations;
    4. ``margin = p(top1) - p(top2)``.

    **Specified fallback (§6.2):** providers return only 5-20 alternatives. If the
    runner-up's continuation token is not among them, this returns ``None`` and
    fusion ignores the signal. There is no guessing and no "assume the rest of the
    mass belongs to the runner-up" — that assumption is how a margin signal turns
    into noise on exactly the ambiguous queries it was supposed to catch.
    """
    if not route_name:
        return None
    names = [name for name in dict.fromkeys(candidates) if name]
    if len(names) < 2 or route_name not in names:
        return None
    alignment = _align_route_value(token_logprobs, route_name)
    if alignment is None or token_logprobs is None:
        return None

    divergence = _first_divergence(route_name, names)
    if divergence is None:
        return None

    position = alignment.name_start + divergence
    token_index = next(
        (
            index
            for index in alignment.token_indices
            if alignment.starts[index] <= position < alignment.ends[index]
        ),
        None,
    )
    if token_index is None:
        return None

    token = token_logprobs[token_index]
    alternatives = list(token.top)
    if all(text != token.token for text, _ in alternatives):
        # Some providers omit the sampled token from top_logprobs; without it
        # there is no top1 to measure a gap from.
        alternatives.append((token.token, token.logprob))
    if len(alternatives) < 2:
        return None

    token_start = alignment.starts[token_index]
    prefix = alignment.text[alignment.name_start : max(token_start, alignment.name_start)]
    covers_open_quote = token_start < alignment.name_start

    valid: list[float] = []
    for text, logprob in alternatives:
        reconstructed = _reconstruct(prefix, text, covers_open_quote=covers_open_quote)
        if reconstructed and any(name.startswith(reconstructed) for name in names):
            valid.append(logprob)
    if len(valid) < 2:
        return None

    weights = [math.exp(min(logprob, 0.0)) for logprob in valid]
    total = math.fsum(weights)
    if total <= 0.0:
        return None
    weights.sort(reverse=True)
    return max(0.0, min(1.0, (weights[0] - weights[1]) / total))


def _first_divergence(name: str, candidates: Sequence[str]) -> int | None:
    """Index of the first character at which the candidate trie branches."""
    for index in range(len(name) + 1):
        prefix = name[:index]
        sharing = [candidate for candidate in candidates if candidate.startswith(prefix)]
        if len(sharing) < 2:
            return None
        next_chars = {candidate[index] for candidate in sharing if len(candidate) > index}
        terminates = any(len(candidate) == index for candidate in sharing)
        if len(next_chars) > 1 or (next_chars and terminates):
            return index
    return None


def _reconstruct(prefix: str, alternative: str, *, covers_open_quote: bool) -> str:
    """Rebuild the route value that ``alternative`` would have produced."""
    body = alternative
    if covers_open_quote:
        # The token also spans the opening quote (e.g. '"ref'); everything before
        # the last quote in the alternative belongs to the JSON scaffolding.
        quote = body.rfind('"')
        if quote != -1:
            body = body[quote + 1 :]
    candidate = prefix + body
    closing = candidate.find('"')
    return candidate[:closing] if closing != -1 else candidate


# --------------------------------------------------------------------------- #
# Fusion and the no-signal rule (plan §6.3)
# --------------------------------------------------------------------------- #


def default_fusion(report: ConfidenceReport) -> float:
    """Fuse the available signals into one scalar (plan §6.3, normative).

    ``min`` of whatever is available, verbalized damped x0.85 first, then x0.8 if a
    self-consistency majority overturned the greedy vote. Conservative *by
    construction*: a fused score can never exceed its weakest signal, so adding a
    signal can only lower confidence, never raise it — which is the only safe
    direction when the output feeds downgrade-only thresholds.

    Pluggable via ``Router(confidence_fusion=...)``; this is the default.

    Note on ``stated``: §6.2 says verbalized confidence is "recorded when nothing
    better exists" and "never used when a better signal exists", while this
    fusion — the plan's own normative code — takes it into the ``min``
    unconditionally. There is no conflict in practice, because the
    ``stated_confidence`` field is only appended to the wire schema on the
    verbalized rung (§3.4), so ``stated`` is ``None`` on paths that have logprobs.
    The ``min`` is kept verbatim rather than "fixed", so that a caller who *does*
    populate both gets the conservative reading.
    """
    available = [
        value
        for value in (
            report.p_route,
            report.agreement,
            _STATED_DAMPING * report.stated if report.stated is not None else None,
        )
        if value is not None
    ]
    confidence = min(available) if available else 0.0
    return confidence * _OVERTURN_PENALTY if report.vote_overturned else confidence


def signals_are_actionable(
    report: ConfidenceReport,
    *,
    thresholds_on_verbalized: bool = False,
) -> bool:
    """Whether the threshold machinery may act on ``report`` (plan §6.3, §13 #9).

    **The no-signal rule.** When the resolved capability set yields neither
    logprobs nor votes — a bare BYO callable, or an Anthropic path before v0.2's
    self-consistency vote — the threshold downgrade machinery is **inert by
    default**. Verbalized-only scores are still recorded (``method="verbalized"``),
    they simply do not trigger clarify or abstain, because acting on exactly the
    signal the evidence distrusts would invert Recommendation #4: the router would
    start abstaining based on the model's self-assessed commitment.

    Two cases return ``False``:

    * **No signal at all** — always, regardless of ``thresholds_on_verbalized``.
      A report with no signals fuses to ``0.0``, which is below every threshold; a
      router that acted on it would abstain on *every* request. That is not a
      conservative failure mode, it is a broken one.
    * **Verbalized only** — unless the caller opts in with
      ``thresholds_on_verbalized=True`` (the Router warns once at construction).

    Model-elected clarify/abstain always passes through regardless of this rule —
    that is the model making a decision, not the library inferring one (§6.4).
    """
    if report.p_route is not None or report.margin is not None or report.agreement is not None:
        return True
    if report.stated is None:
        return False
    return thresholds_on_verbalized


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_confidence_report(
    *,
    token_logprobs: Sequence[TokenLP] | None = None,
    route_name: str | None = None,
    stated: float | None = None,
    agreement: float | None = None,
    margin: float | None = None,
    vote_overturned: bool = False,
    vote_n: int | None = None,
    avg_logprob: float | None = None,
    thresholds: Mapping[str, float] | None = None,
    method: str | None = None,
    fusion: Callable[[ConfidenceReport], float] | None = None,
) -> ConfidenceReport:
    """Assemble the one :class:`ConfidenceReport` for a decision (plan §6.3).

    The same object is attached to ``Decision.confidence`` and stored verbatim in
    ``AuditRecord.confidence``, so recalibration (§6.4) and distillation (§8.5)
    see the **raw evidence** — every signal, plus the thresholds as they were
    actually applied — and not merely the fused scalar. That is what makes an
    offline threshold sweep possible from logs alone.

    Args:
        token_logprobs: from ``LLMResult.token_logprobs``; ``None`` on paths with
            no logprobs (§6.1).
        route_name: the committed route, needed to align the logprobs. ``None``
            for clarify/abstain outcomes, where ``p_route`` is undefined.
        stated: the model's verbalized ``stated_confidence``, raw and undamped —
            damping happens in fusion, and the raw value is what gets recorded.
        agreement: ``winner_votes / n``. [v0.2]
        margin: from :func:`compute_margin`. [v0.2]
        vote_overturned: majority overturned greedy. [v0.2]
        vote_n: number of self-consistency votes, for the ``method`` string. [v0.2]
        avg_logprob: a provider-level average logprob (Gemini's ``avg_logprobs``),
            used **only** as the §6.2 coarse fallback when token alignment fails.
            It is marked as such in ``method`` (``"logprobs:avg"``) so nothing
            downstream mistakes a whole-response average for a route-token
            measurement.
        thresholds: thresholds-as-applied, e.g.
            ``ThresholdPolicy.as_thresholds()``. Recorded, never enforced here.
        method: override the derived method string.
        fusion: override :func:`default_fusion`.

    Returns:
        A fully populated report whose ``score`` is the fused scalar.
    """
    p_route = compute_p_route(token_logprobs, route_name)
    used_avg = False
    if p_route is None and avg_logprob is not None:
        p_route = math.exp(min(avg_logprob, 0.0))
        used_avg = True

    if method is None:
        parts: list[str] = []
        if p_route is not None:
            parts.append("logprobs:avg" if used_avg else "logprobs")
        if agreement is not None:
            parts.append(f"vote:n={vote_n}" if vote_n else "vote")
        if not parts and stated is not None:
            parts.append("verbalized")
        method = "+".join(parts) if parts else "none"

    report = ConfidenceReport(
        score=0.0,
        method=method,
        p_route=p_route,
        margin=margin,
        agreement=agreement,
        vote_overturned=vote_overturned,
        stated=stated,
        thresholds=dict(thresholds) if thresholds else {},
    )
    score = (fusion or default_fusion)(report)
    return report.model_copy(update={"score": max(0.0, min(1.0, score))})
