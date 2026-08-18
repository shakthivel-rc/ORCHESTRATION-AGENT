"""Run the 126-route support catalog through a Router, offline (plan §10.3, §9.2).

::

    python examples/support_triage/demo.py            # from the repo root
    python -m examples.support_triage.demo            # same thing, as a module

Zero extras, no network, no API key, fully deterministic — this is the file CI
runs to prove the flagship example has not rotted (plan §10.4 stage 4).

What it prints:

1. **Catalog** — size, ``registry.version``, domains, entitlement-gated routes.
2. **Shortlist at 126 routes** — ``shortlist="auto"`` is *above* its bypass
   threshold here, so retrieval genuinely runs: K=10 from the plan §5.3 band
   table (``25 <= N < 150``), plus the pinned ``human_handoff``, in shuffled
   order with the replay seed recorded.
3. **Gold-set readout** — route accuracy, clarify/abstain accuracy, and
   recall@K decomposed into the two failure modes plan §9.2 separates: the
   *retrieval gap* (the right route never reached the prompt) and the
   *decision gap* (it did, and the decider still missed).
4. **Entitlements** — the same query with and without an entitlement, showing
   the candidate set shrink and the decision change (plan §7.1).
5. **Three narrated decisions** — a clean route with extracted args, a clarify,
   and an out-of-scope abstain resolved by the configured fallback, plus a
   sync/async parity check (plan §2.5).

See :data:`STUB_NOTICE` for what the client is and — more importantly — is not.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from switchboard import (
    InMemorySink,
    LLMRequest,
    Registry,
    RequestContext,
    Route,
    Router,
)

try:  # `python -m examples.support_triage.demo`
    from .catalog import ABSTAIN, CLARIFY, DOMAINS, ENTITLEMENTS, GOLD_CASES, registry
except ImportError:  # `python examples/support_triage/demo.py`
    from catalog import (  # type: ignore[no-redef]
        ABSTAIN,
        CLARIFY,
        DOMAINS,
        ENTITLEMENTS,
        GOLD_CASES,
        registry,
    )

if TYPE_CHECKING:
    from collections.abc import Callable

    from switchboard import AuditRecord, Decision

__all__ = ["STUB_NOTICE", "build_stub_client", "main"]

STUB_NOTICE = """\
WHAT THE CLIENT IS. build_stub_client() is NOT a model. It is a deterministic
keyword matcher: it reads the candidate list switchboard put in the prompt,
scores each candidate by the trigger phrases the catalog declares in
Route.metadata["triggers"], and answers with the wire schema switchboard asked
for. Ties become clarify; no match at all becomes abstain. It exists so this file
runs identically on every machine, forever, with no key and no network.

So the score below is NOT a benchmark result. The gold queries contain the very
phrases the catalog declares, so a perfect score is by construction and means
only "the pipeline is wired correctly end to end". A real number comes from
pointing Router(client=...) at a real provider and re-running the same
GOLD_CASES; plan §10.3 requires such a page to state its model and its date, and
a stub has neither. What IS meaningful below is everything the stub does not
control: the entitlement filter, which routes retrieval surfaced (recall@K), the
shuffle seed, the args validation, the fallback, the audit trail."""


# --------------------------------------------------------------------------- #
# The offline stand-in client (see the module docstring: this is not a model).
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9]+")
_CANDIDATE_HEADING = "# Candidates for this request"
_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)
_CARD_RE = re.compile(r"^- ([a-z][a-z0-9_\-.:]*):", re.MULTILINE)

#: Field-name -> extractor. The stub fills only the argument fields it knows how
#: to read out of the query; anything else it leaves absent, which is exactly the
#: condition plan §3.8 row 4 turns into a `clarify` with `missing=(...)`.
_EXTRACTORS: dict[str, re.Pattern[str]] = {
    "order_id": re.compile(r"\border\s+#?([a-z0-9][a-z0-9-]*\d[a-z0-9-]*)", re.IGNORECASE),
    "invoice_id": re.compile(r"\b(inv-[a-z0-9-]+)", re.IGNORECASE),
    "sku": re.compile(r"\bsku\s+([a-z0-9][a-z0-9-]*)", re.IGNORECASE),
    "new_email": re.compile(r"\b([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE),
    "seats": re.compile(r"\b(\d+)\s+(?:more\s+)?(?:seats?|licen[cs]es?|users?)", re.IGNORECASE),
}


def _norm(text: str) -> str:
    """Space-padded, punctuation-free lowercase — so ``in`` is word-boundary safe."""
    return " " + " ".join(_WORD_RE.findall(text.lower())) + " "


def _prompt_query(request: LLMRequest) -> str:
    """The user query switchboard placed in segment D (plan §4.6)."""
    for segment in reversed(request.segments):
        found = _QUERY_RE.search(segment.content)
        if found:
            return found.group(1)
    return ""


def _prompt_candidates(request: LLMRequest, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """The candidate route names switchboard offered in segment C (plan §4.6).

    Segment C is absent when the small-catalog bypass fired, in which case the
    whole entitled directory is the option set — hence ``fallback``.
    """
    for segment in request.segments:
        if segment.content.startswith(_CANDIDATE_HEADING):
            return tuple(_CARD_RE.findall(segment.content))
    return fallback


def _trigger_score(query: str, triggers: tuple[str, ...]) -> int:
    """Longest trigger phrase (in words) that occurs in ``query``; 0 for none."""
    padded = _norm(query)
    best = 0
    for trigger in triggers:
        phrase = _norm(trigger)
        if phrase in padded:
            best = max(best, len(phrase.split()))
    return best


def _extract_args(query: str, route: Route) -> dict[str, Any]:
    """Pull whatever argument values the query plainly states."""
    if route.args_model is None:
        return {}
    args: dict[str, Any] = {}
    for field, annotation in route.args_model.model_fields.items():
        pattern = _EXTRACTORS.get(field)
        if pattern is None:
            continue
        found = pattern.search(query)
        if found is None:
            continue
        value = found.group(1)
        args[field] = int(value) if annotation.annotation is int else value
    return args


def build_stub_client(catalog: Registry) -> Callable[[LLMRequest], dict[str, Any]]:
    """Build the deterministic offline stand-in for a model.

    The returned callable is a plain ``Callable[[LLMRequest], dict]``, which is
    the zero-extras BYO row of the plan §4.1 coercion table: switchboard wraps it,
    hands it the request, and validates whatever dict comes back against the wire
    schema it generated for that call.

    Decision rules, in order:

    1. score every *candidate* (never the whole catalog — the stub is bound by
       the shortlist exactly as a model would be);
    2. no candidate scores above zero -> ``abstain`` (``reason="model_elected"``);
    3. two or more candidates tie for the top score -> ``clarify``, naming them;
    4. otherwise -> ``route``, with whatever arguments the query plainly states.
    """
    all_names = catalog.names

    def stub(request: LLMRequest) -> dict[str, Any]:
        query = _prompt_query(request)
        candidates = _prompt_candidates(request, all_names)

        scored = sorted(
            (
                (_trigger_score(query, tuple(route.metadata.get("triggers", ()))), name)
                for name in candidates
                if (route := catalog.get(name)) is not None
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        if not scored or scored[0][0] == 0:
            return {
                "rationale": "No candidate route covers this request.",
                "kind": "abstain",
                "reason": "model_elected",
            }

        top = scored[0][0]
        tied = [name for score, name in scored if score == top]
        if len(tied) > 1:
            labels = [catalog[name].clarify_label or name for name in tied[:3]]
            return {
                "rationale": f"The request matches {len(tied)} routes equally well.",
                "kind": "clarify",
                "question": f"Did you mean {' or '.join(labels)}?",
                "candidates": tied[:3],
            }

        chosen = catalog[tied[0]]
        return {
            "rationale": f"The request states a phrase specific to {chosen.name}.",
            "kind": "route",
            "route": chosen.name,
            "args": _extract_args(query, chosen) or None,
        }

    return stub


# --------------------------------------------------------------------------- #
# Reporting helpers.
# --------------------------------------------------------------------------- #


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _outcome(decision: Decision) -> str:
    """The label a decision should be scored against (plan §6.6).

    A configured fallback arrives as ``kind="route"`` with
    ``decision_path="fallback"``; downstream code branches on ``kind`` alone, but
    an *evaluation* must not count the escalation route as a correct commitment,
    so the fallback is folded back to ``abstain`` here.
    """
    if decision.decision_path == "fallback":
        return ABSTAIN
    if decision.kind == "route":
        return decision.route
    if decision.kind == "clarify":
        return CLARIFY
    return ABSTAIN


def _describe(decision: Decision) -> str:
    if decision.kind == "route":
        args = decision.args.model_dump(exclude_none=True) if decision.args else {}
        return f"route={decision.route} args={args or '{}'}"
    if decision.kind == "clarify":
        return f'clarify question={decision.question!r} candidates={list(decision.candidates)}'
    if decision.kind == "abstain":
        return f"abstain reason={decision.reason}"
    return decision.kind


# --------------------------------------------------------------------------- #
# Sections.
# --------------------------------------------------------------------------- #


def show_catalog() -> None:
    _rule("1. The catalog")
    gated = [route for route in registry.routes if route.requires]
    with_args = [route for route in registry.routes if route.args_model is not None]
    pinned = [route.name for route in registry.routes if route.pinned]
    print(f"routes                {len(registry)}")
    print(f"registry.version      {registry.version}   (keys the prompt cache, the index, every audit row)")
    print(f"domains               {len(DOMAINS)}: {', '.join(DOMAINS)}")
    print(f"routes with args      {len(with_args)}")
    print(f"entitlement-gated     {len(gated)}: {', '.join(r.name for r in gated)}")
    print(f"entitlements in use   {', '.join(sorted(ENTITLEMENTS))}")
    print(f"pinned                {', '.join(pinned)}  (always a candidate; also Router(fallback=...))")


def show_shortlist(router: Router, ctx: RequestContext, sink: InMemorySink) -> None:
    _rule("2. Shortlist behaviour at 126 routes")
    query = "where is my package for order 4471"
    router.route(query, context=ctx)
    record = sink.latest
    assert record is not None
    print(f"query                 {query!r}")
    print(f"shortlister           {type(router.shortlister).__name__}  (Router(shortlist='auto'))")
    print(f"catalog / entitled    {record.candidates_total} / {record.candidates_entitled}")
    print(f"bypass fired          {record.shortlist_skipped}  (auto bypasses below 25 entitled routes)")
    print(f"candidates in prompt  {len(record.shortlist)}  = K + pinned (plan §5.3 band: N<150 -> K=10)")
    print(f"shuffle seed          {record.shuffle_seed}  (stable digest, so the prompt replays byte-exact)")
    print("\nretriever ranking (pre-shuffle rank, backend-native score, source):")
    for candidate in sorted(record.shortlist, key=lambda c: c.rank):
        print(f"  {candidate.rank:>2}  {candidate.score:8.3f}  {candidate.source:<6} {candidate.route_name}")


def score_gold(router: Router, ctx: RequestContext, sink: InMemorySink) -> None:
    _rule("3. Gold-set readout")
    route_total = route_hit = 0
    clarify_total = clarify_hit = 0
    abstain_total = abstain_hit = 0
    recall_total = recall_hit = 0
    retrieval_gap: list[str] = []
    decision_gap: list[tuple[str, str, str]] = []

    for query, expected in GOLD_CASES:
        decision = router.route(query, context=ctx)
        record: AuditRecord | None = sink.latest
        assert record is not None
        actual = _outcome(decision)

        if expected == CLARIFY:
            clarify_total += 1
            clarify_hit += actual == CLARIFY
        elif expected == ABSTAIN:
            abstain_total += 1
            abstain_hit += actual == ABSTAIN
        else:
            route_total += 1
            retrieved = expected in {c.route_name for c in record.shortlist}
            recall_total += 1
            recall_hit += retrieved
            if actual == expected:
                route_hit += 1
            elif not retrieved:
                retrieval_gap.append(f"{query!r} -> {expected} never reached the prompt")
            else:
                decision_gap.append((query, expected, actual))

    total = route_total + clarify_total + abstain_total
    correct = route_hit + clarify_hit + abstain_hit
    print(f"cases                 {total}")
    print(f"route accuracy        {route_hit}/{route_total}   ({route_hit / route_total:.1%})")
    print(f"clarify accuracy      {clarify_hit}/{clarify_total}   (ambiguous by construction)")
    print(f"abstain accuracy      {abstain_hit}/{abstain_total}   (out of scope by construction)")
    print(f"overall               {correct}/{total}   ({correct / total:.1%})")
    print(f"recall@K              {recall_hit}/{recall_total}   ({recall_hit / recall_total:.1%})"
          "   the gold route was in the shortlist")
    print("\n(Read STUB_NOTICE again: the decider is a keyword matcher over phrases this very")
    print(" catalog declares, so its score is a wiring check. recall@K is the honest number here")
    print(" — retrieval never sees the triggers, it only sees descriptions and examples.)")
    print("\nfailure decomposition (plan §9.2: a miss is a retrieval bug or a decision bug, never 'bad'):")
    print(f"  retrieval gap       {len(retrieval_gap)}")
    for line in retrieval_gap:
        print(f"                        {line}")
    print(f"  decision gap        {len(decision_gap)}")
    for query, expected, actual in decision_gap:
        print(f"                        {query!r}\n                          want {expected}, got {actual}")


def show_entitlements(router: Router, sink: InMemorySink) -> None:
    _rule("4. Entitlements (plan §7.1) — the same query, two contexts")
    query = "rotate my api key, it was compromised"
    for label, entitlements in (
        ("developer     ", frozenset({"developer"})),
        ("no entitlement", frozenset()),
    ):
        decision = router.route(query, context=RequestContext(tenant_id="acme", entitlements=entitlements))
        record = sink.latest
        assert record is not None
        print(f"{label}  entitled={record.candidates_entitled:>3}  ->  {_describe(decision)}")
    print("\nThe gated route is filtered out *before* the LLM call, so it cannot be leaked by a")
    print("prompt injection or a confused model: it is simply not in the candidate set.")


def show_narrated(router: Router, ctx: RequestContext) -> None:
    _rule("5. Three decisions, narrated")
    for query, note in (
        ("I want my money back for order 123", "clean commitment, arguments extracted in the same call"),
        ("can you cancel it", "ambiguous: order vs subscription -> a question, not a coin flip"),
        ("what is the capital of France", "out of scope -> abstain, resolved by Router(fallback=...)"),
    ):
        decision = router.route(query, context=ctx)
        print(f"\nquery      {query!r}")
        print(f"note       {note}")
        print(f"decision   {_describe(decision)}")
        print(f"path       kind={decision.kind} decision_path={decision.decision_path} "
              f"downgraded_from={decision.downgraded_from}")
        print(f"confidence score={decision.confidence.score} method={decision.confidence.method!r}")
        print(f"rationale  {decision.rationale}")

    # Plan §2.5: every public entry point has an `a`-prefixed twin, and the two
    # drivers share one pipeline — so they must agree, not merely both work.
    probe = "cancel my order 2210"
    sync_decision = router.route(probe, context=ctx)
    async_decision = asyncio.run(router.aroute(probe, context=ctx))
    print(f"\nsync/async parity   route() -> {_describe(sync_decision)}")
    print(f"                    aroute() -> {_describe(async_decision)}")
    print(f"                    identical: {_describe(sync_decision) == _describe(async_decision)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run every section against one Router. Deterministic and offline."""
    sink = InMemorySink(maxlen=256)
    router = Router(
        registry,
        client=build_stub_client(registry),
        shortlist="auto",          # bypass below 25 entitled routes, BM25 above -> BM25 here
        allow_clarify=True,
        fallback="human_handoff",  # terminal abstain arrives as kind="route", decision_path="fallback"
        sink=sink,
        otel=False,                # no tracer provider is configured in this demo
    )
    router.warm()  # build the index at startup instead of on the first user's latency budget

    # Full entitlements: this "user" can see every route in the catalog.
    ctx = RequestContext(tenant_id="acme", user_id="user-42", entitlements=ENTITLEMENTS)

    print(STUB_NOTICE)
    show_catalog()
    show_shortlist(router, ctx, sink)
    score_gold(router, ctx, sink)
    show_entitlements(router, sink)
    show_narrated(router, ctx)

    _rule("Swapping the stub for a real model")
    print("Everything above is one keyword argument away from a live provider:\n")
    print('    router = Router(registry, client="instructor:openai/gpt-5-nano",')
    print('                    shortlist="auto", allow_clarify=True, fallback="human_handoff")\n')
    print("pip install 'switchboard[instructor]' — the catalog, the gold set, the shortlist and")
    print("the audit trail are unchanged. With a client that returns logprobs, confidence.score")
    print("also stops reading 0.0/'none': the thresholds in plan §6.4 need a signal to act on,")
    print("and a bare BYO callable supplies none (the no-signal rule, plan §6.3).")
    router.close()


if __name__ == "__main__":
    main()
