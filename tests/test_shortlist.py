"""Shortlist & retrieval behaviour (plan §5).

The shortlister is the *retrieve-then-decide* stage of §2.1, and §5 hangs four
guarantees on it that this module exists to prove:

* **the ranking is real** — fielded BM25F, so a name hit beats a description hit
  and an obviously-matching route comes back first (§5.2);
* **the index is tenant-stable** — IDF is computed over the full registry, never
  over the entitled subset, which is exactly why one index is shareable and why
  the entitlement filter has to happen at retrieval time instead (§5.2, §5.9);
* **K is evidence-bound** — 10/15/20 by catalog size, clamped to ``[5, 20]``,
  because K=100 costs 10-16 accuracy points at ~10x the tokens (§5.3);
* **entitlement is never negotiable** — filter *before* truncate inside the
  backend, and re-intersect afterwards in the loop, because a shortlister is
  untrusted on entitlements (§5.1, §5.9).

Fixtures come from ``conftest.py``; anything specific to retrieval is built
locally here so no other module inherits a catalog shaped for BM25 asserts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pytest

from conftest import wire_output
from switchboard import Registry, RequestContext, Route, Router
from switchboard.core.candidates import Candidate, ShortlistResult
from switchboard.engine.shortlist import (
    DEFAULT_SHORTLIST_MIN_ROUTES,
    K_MAX,
    K_MIN,
    AutoShortlister,
    BM25Shortlister,
    EmbeddingShortlister,
    effective_k,
    resolve_shortlister,
    tokenize,
)
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from collections.abc import Set as AbstractSet

SHORTLIST_LOGGER = "switchboard.engine.shortlist"


# --------------------------------------------------------------------------- #
# Local catalogs — shaped for retrieval asserts, not for the router.
# --------------------------------------------------------------------------- #


def _names(result: ShortlistResult) -> list[str]:
    """Candidate names in result order."""
    return [candidate.route_name for candidate in result.candidates]


def _all(routes: Sequence[Route]) -> frozenset[str]:
    """The whole catalog as an ``allowed`` set."""
    return frozenset(route.name for route in routes)


@pytest.fixture
def order_routes() -> list[Route]:
    """Five routes where three genuinely talk about orders.

    Three matching routes matters: below three above-zero scores the backend
    enters §5.5's degenerate-retrieval handling, which is a different code path
    with its own tests below. This catalog keeps the ranking tests on the
    ordinary path.
    """
    return [
        Route(name="refund_order", description="Issue or check a refund for an order."),
        Route(name="track_order", description="Track shipment status for an existing order."),
        Route(name="cancel_order", description="Cancel an order before it ships."),
        Route(name="reset_password", description="Reset the account password."),
        Route(name="weather_report", description="Report tomorrow's weather forecast."),
    ]


def _generated(count: int, *, pinned: bool = False) -> list[Route]:
    """A synthetic catalog of ``count`` routes that all share vocabulary.

    Descriptions deliberately overlap ("widget", "workspace") so a single query
    scores well above zero on many routes — the ordinary retrieval path — while
    the numeric suffix keeps one route clearly best.
    """
    routes = [
        Route(
            name=f"widget_task_{index}",
            description=(
                f"Operate on widget alpha{index} inside workspace {index}. "
                f"Use for widget work in workspace {index}."
            ),
            examples=(f"do the widget alpha{index} job",),
            tags=frozenset({"widget"}),
        )
        for index in range(count)
    ]
    if pinned:
        routes.append(
            Route(
                name="human_handoff",
                description="Escalate to a human support agent.",
                pinned=True,
            )
        )
    return routes


# --------------------------------------------------------------------------- #
# BM25 ranking (plan §5.2)
# --------------------------------------------------------------------------- #


def test_bm25_ranks_the_obviously_matching_route_first(order_routes: list[Route]) -> None:
    """The whole premise of §5.2: lexical retrieval is a serviceable first stage."""
    backend = BM25Shortlister(min_routes=0)
    backend.build(order_routes, registry_version="v1")

    result = backend.shortlist(
        "I need a refund for my order", allowed=_all(order_routes), k=3
    )

    assert _names(result)[0] == "refund_order"
    assert result.skipped is False
    assert result.weak_retrieval is False
    assert {candidate.source for candidate in result.candidates} == {"bm25"}
    # Ranks are 0-based and pre-shuffle: the audit record replays the prompt from
    # them plus the seed (§5.6, §8.2).
    assert [candidate.rank for candidate in result.candidates] == [0, 1, 2]


def test_fielded_weights_a_name_hit_outranks_a_description_hit() -> None:
    """``name`` x3.0 vs ``description`` x1.0 (plan §5.2, Appendix A).

    Both routes contain the query term exactly once. If the field weights were
    not applied the two would tie and the tie-break would fall to catalog
    position — which would hand the win to the *description* match, since it is
    declared first here on purpose.
    """
    routes = [
        Route(name="alpha_widget", description="This handles a refund when the customer asks."),
        Route(name="refund", description="Completely unrelated widget maintenance duties."),
        Route(name="beta_widget", description="Something about a refund policy footnote."),
    ]
    backend = BM25Shortlister(min_routes=0)
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("refund", allowed=_all(routes), k=3)

    assert _names(result)[0] == "refund"
    by_name = {candidate.route_name: candidate.score for candidate in result.candidates}
    assert by_name["refund"] > by_name["alpha_widget"]


def test_fielded_weights_are_exactly_three_two_one() -> None:
    """``name`` x3.0, ``examples`` x2.0, ``description`` x1.0 (§5.2, Appendix A).

    Every field of every route here is the same token length, so BM25F's
    per-field length normaliser is identical across the corpus and cancels out.
    What is left is the weight ratio itself, which the three scores reproduce to
    the digit — the strongest available statement that the weights are applied
    per field rather than to one concatenated document.
    """
    routes = [
        Route(
            name="alpha_task",
            description="sprocket reconciliation quarterly ledger",
            examples=("widget assembly floor plan",),
        ),
        Route(
            name="beta_task",
            description="widget assembly floor plan",
            examples=("sprocket reconciliation quarterly ledger",),
        ),
        Route(
            name="sprocket_task",
            description="widget assembly floor plan",
            examples=("widget assembly floor plan",),
        ),
        Route(
            name="gamma_task",
            description="lantern polishing evening shift",
            examples=("lantern polishing evening shift",),
        ),
    ]
    backend = BM25Shortlister(min_routes=0)
    backend.build(routes, registry_version="v1")

    scores = {
        candidate.route_name: candidate.score
        for candidate in backend.shortlist("sprocket", allowed=_all(routes), k=4).candidates
    }

    unit = scores["alpha_task"]  # description-only hit
    assert unit > 0.0
    assert scores["beta_task"] == pytest.approx(2.0 * unit)  # examples
    assert scores["sprocket_task"] == pytest.approx(3.0 * unit)  # name


def test_tokenizer_splits_identifiers_and_drops_stopwords() -> None:
    """Route names are identifiers; queries are prose (plan §5.2)."""
    assert tokenize("Cancel the getOrderStatus_v2 request") == [
        "cancel",
        "order",
        "status",
        "v2",
        "request",
    ]
    # Filtering must never empty an otherwise non-empty document, or a route
    # named "how_to" would silently drop out of the index entirely.
    assert tokenize("how to") == ["how", "to"]


def test_idf_is_computed_over_the_full_registry_not_the_entitled_subset(
    order_routes: list[Route],
) -> None:
    """Plan §5.2: "IDF is computed over the **full registry**".

    The observable consequence — and the reason it is specified — is that a
    route's score does not move when a *different* tenant's routes are filtered
    out. Tenant-stable scores are what make one shared index legitimate (§5.4),
    and they are why the entitlement boundary has to be enforced at retrieval
    time rather than by rebuilding the corpus per tenant.
    """
    backend = BM25Shortlister(min_routes=0)
    backend.build(order_routes, registry_version="v1")

    everyone = backend.shortlist("order", allowed=_all(order_routes), k=5)
    narrowed = backend.shortlist(
        "order", allowed=frozenset({"track_order", "cancel_order"}), k=5
    )

    full_scores = {c.route_name: c.score for c in everyone.candidates}
    narrow_scores = {c.route_name: c.score for c in narrowed.candidates}
    for name, score in narrow_scores.items():
        assert score == pytest.approx(full_scores[name])


# --------------------------------------------------------------------------- #
# "auto": the small-catalog bypass (plan §5.1, §5.3)
# --------------------------------------------------------------------------- #


def test_auto_bypasses_below_the_bypass_threshold() -> None:
    """N < 25 entitled routes: no shortlist at all (plan §5.3, first row).

    ``skipped=True`` and ``source="all"`` are the observable contract: the prompt
    layer reads them and renders the full entitled registry in canonical order as
    a stable cached prefix instead of a per-query candidate block (§5.6).
    """
    routes = _generated(DEFAULT_SHORTLIST_MIN_ROUTES - 1)
    backend = AutoShortlister()
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("widget alpha3 in workspace 3", allowed=_all(routes), k=10)

    assert result.skipped is True
    assert {candidate.source for candidate in result.candidates} == {"all"}
    assert _names(result) == [route.name for route in routes]
    assert result.weak_retrieval is False


def test_auto_engages_the_backend_at_the_threshold() -> None:
    """N >= 25 entitled routes: the configured backend runs (plan §5.3)."""
    routes = _generated(DEFAULT_SHORTLIST_MIN_ROUTES)
    backend = AutoShortlister()
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("widget alpha3 in workspace 3", allowed=_all(routes), k=10)

    assert result.skipped is False
    assert {candidate.source for candidate in result.candidates} == {"bm25"}
    assert len(result.candidates) == 10
    assert _names(result)[0] == "widget_task_3"


def test_auto_bypass_keys_on_the_entitled_count_not_the_registry_size() -> None:
    """"A tenant entitled to 12 of 400 routes gets the bypass" (plan §5.1)."""
    routes = _generated(60)
    backend = AutoShortlister()
    backend.build(routes, registry_version="v1")

    allowed = frozenset(route.name for route in routes[:12])
    result = backend.shortlist("widget alpha3", allowed=allowed, k=10)

    assert result.skipped is True
    assert set(_names(result)) == set(allowed)


# --------------------------------------------------------------------------- #
# K sizing (plan §5.3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("entitled", "expected"),
    [(25, 10), (149, 10), (150, 15), (999, 15), (1000, 20), (5000, 20)],
)
def test_k_defaults_follow_the_evidence_band_table(entitled: int, expected: int) -> None:
    """§5.3's table, verbatim: 10 below 150, 15 to 999, 20 at 1000+."""
    assert effective_k(entitled) == expected


def test_k_is_clamped_up_to_the_floor() -> None:
    """A tiny explicit K is raised to 5 — below that the shortlist starves (§5.3)."""
    assert effective_k(30, 2) == K_MIN
    assert effective_k(30, 5) == K_MIN


def test_oversized_k_warns_and_clamps(caplog: pytest.LogCaptureFixture) -> None:
    """K=50 without opt-in clamps to 20 and says why (plan §5.3).

    "Never large K": the 99% Success Paradox measured K=100 at -10-16 accuracy
    points and ~10x the prompt tokens, so exceeding the window is opt-in and
    always logged.
    """
    with caplog.at_level(logging.WARNING, logger=SHORTLIST_LOGGER):
        clamped = effective_k(200, 50)

    assert clamped == K_MAX
    assert "99% Success Paradox" in caplog.text
    assert "allow_oversized_k=True" in caplog.text


def test_oversized_k_is_honoured_with_the_opt_in(caplog: pytest.LogCaptureFixture) -> None:
    """``allow_oversized_k=True`` passes K through — still with a warning (§5.3)."""
    with caplog.at_level(logging.WARNING, logger=SHORTLIST_LOGGER):
        assert effective_k(200, 50, allow_oversized_k=True) == 50

    assert "99% Success Paradox" in caplog.text


def test_k_below_one_is_a_config_error() -> None:
    """"K < 1" is in §3.8's ConfigError list: a shortlist of nothing cannot decide."""
    with pytest.raises(ConfigError, match="shortlist_k"):
        effective_k(30, 0)


def test_router_refuses_k_below_one(small_registry: Registry, fake_client: Any) -> None:
    """The same refusal, surfaced at ``Router(...)`` construction (plan §3.8)."""
    with pytest.raises(ConfigError, match="shortlist_k"):
        Router(small_registry, client=fake_client, shortlist_k=0, otel=False)


# --------------------------------------------------------------------------- #
# Pinned routes and degenerate retrieval (plan §5.5)
# --------------------------------------------------------------------------- #


def test_pinned_routes_are_appended_regardless_of_score() -> None:
    """"The fallback path must never be retrieved out of existence" (plan §5.5).

    ``human_handoff`` shares no vocabulary with the query, so it scores zero and
    would fall outside any top-K. It is appended anyway, tagged
    ``source="pinned"`` so the audit record shows it was not retrieved on merit.
    """
    routes = _generated(30, pinned=True)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("widget alpha3 in workspace 3", allowed=_all(routes), k=5)

    assert "human_handoff" in _names(result)
    pinned = next(c for c in result.candidates if c.route_name == "human_handoff")
    assert pinned.source == "pinned"
    assert pinned.score == 0.0
    # Appended, not substituted: the retrieved top-K is still K entries long.
    assert sum(1 for c in result.candidates if c.source == "bm25") == 5


def test_pinned_does_not_mean_visible() -> None:
    """A pinned route the tenant may not see stays invisible (plan §5.5, §7.4)."""
    routes = _generated(30, pinned=True)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="v1")

    allowed = _all(routes) - {"human_handoff"}
    result = backend.shortlist("widget alpha3", allowed=allowed, k=5)

    assert "human_handoff" not in _names(result)


def test_degenerate_retrieval_sets_weak_retrieval() -> None:
    """Fewer than 3 above-zero scores flags the result (plan §5.5).

    The policy stage reads this and widens the clarify band — "a query nothing
    matches is a clarification candidate, not a guessing opportunity".
    """
    routes = _generated(40)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("zzzz qqqq nothing matches here", allowed=_all(routes), k=5)

    assert result.weak_retrieval is True
    assert result.skipped is False
    # Still a non-empty block: a query that matched nothing must not silently
    # become "no eligible routes" (§5.5 branch (b)).
    assert len(result.candidates) == 5
    assert all(candidate.score == 0.0 for candidate in result.candidates)


def test_weak_retrieval_over_a_small_catalog_falls_back_to_the_full_registry() -> None:
    """§5.5 branch (a): at N <= min_routes, show everything rather than noise."""
    routes = _generated(10)
    backend = BM25Shortlister(min_routes=DEFAULT_SHORTLIST_MIN_ROUTES)
    backend.build(routes, registry_version="v1")

    result = backend.shortlist("zzzz qqqq nothing", allowed=_all(routes), k=5)

    assert result.skipped is True
    assert result.weak_retrieval is False
    assert {candidate.source for candidate in result.candidates} == {"all"}
    assert _names(result) == [route.name for route in routes]


# --------------------------------------------------------------------------- #
# ENTITLEMENT SAFETY (plan §5.1, §5.9, §7.4)
# --------------------------------------------------------------------------- #


def test_the_top_scoring_route_cannot_appear_when_it_is_not_entitled(
    order_routes: list[Route],
) -> None:
    """Filter **before** truncate (plan §5.9).

    ``refund_order`` is the runaway winner for this query. Remove it from
    ``allowed`` and it must be absent — not demoted, not truncated away by luck,
    absent. Truncate-then-filter would have kept it in the top-K and then dropped
    a legitimate route to make room, starving the tenant.
    """
    backend = BM25Shortlister(min_routes=0)
    backend.build(order_routes, registry_version="v1")

    allowed = _all(order_routes) - {"refund_order"}
    result = backend.shortlist("I need a refund for my order", allowed=allowed, k=3)

    assert "refund_order" not in _names(result)
    assert set(_names(result)) <= allowed
    # A slot freed by the filter goes to an entitled route, not to nothing.
    assert len(result.candidates) == 3


def test_entitled_set_is_never_starved_by_truncation() -> None:
    """A tenant entitled to few routes still gets all of them (plan §5.9)."""
    routes = _generated(60)
    backend = BM25Shortlister(min_routes=0)
    backend.build(routes, registry_version="v1")

    allowed = frozenset({"widget_task_41", "widget_task_42", "widget_task_43"})
    result = backend.shortlist("widget alpha0 workspace 0", allowed=allowed, k=10)

    assert set(_names(result)) == allowed


class _RogueShortlister:
    """A backend that ignores ``allowed`` entirely.

    Plan §5.1 says the Router treats every shortlister as **untrusted on
    entitlements** — "a buggy backend can degrade recall but never leak a route".
    This is the buggy backend.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self._names = tuple(names)

    def build(self, routes: Sequence[Route], *, registry_version: str) -> None:
        del routes, registry_version

    async def abuild(self, routes: Sequence[Route], *, registry_version: str) -> None:
        del routes, registry_version

    def shortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        del query, allowed, k, ctx
        return ShortlistResult(
            candidates=[
                Candidate(route_name=name, score=9.0 - index, rank=index, source="bm25")
                for index, name in enumerate(self._names)
            ]
        )

    async def ashortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        return self.shortlist(query, allowed=allowed, k=k, ctx=ctx)

    @property
    def fingerprint(self) -> str:
        return "rogue"


def test_router_re_intersects_a_rogue_shortlisters_candidates(
    router_factory: Callable[..., Router], small_registry: Registry
) -> None:
    """The Router's own re-intersection is the second line of defence (§5.1).

    ``billing_report`` carries ``requires={"billing"}``; this context does not.
    Even though the shortlister hands it back ranked first, it must never reach
    the prompt — an unauthorised name that never enters the prompt cannot be
    leaked, chosen, or hallucinated into reach (§7.4).
    """
    rogue = _RogueShortlister(["billing_report", "refund", "track_order"])
    router = router_factory([wire_output("refund", {"order_id": "A-1"})], shortlist=rogue)

    decision = router.route(
        "please issue a refund", context=RequestContext(tenant_id="acme")
    )

    assert "billing_report" not in router.client.last_prompt
    assert all(
        candidate.route_name != "billing_report"
        for candidate in decision.audit.shortlist
    )


# --------------------------------------------------------------------------- #
# Index lifecycle (plan §5.4)
# --------------------------------------------------------------------------- #


def test_index_rebuilds_when_the_registry_version_changes() -> None:
    """"Every request compares the live registry hash to the index's" (§5.4)."""
    routes = _generated(30)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="v1")
    first_key = backend.index_key

    grown = [*routes, Route(name="brand_new", description="A brand new widget capability.")]
    backend.build(grown, registry_version="v2")

    assert backend.registry_version == "v2"
    assert backend.index_key != first_key
    assert backend.index_key is not None
    assert backend.index_key.startswith("v2:")
    found = backend.shortlist("brand new widget capability", allowed=_all(grown), k=5)
    assert "brand_new" in _names(found)


def test_index_build_is_idempotent_for_one_version() -> None:
    """A no-op rebuild is what makes "attempt it on every request" affordable (§5.4)."""
    routes = _generated(30)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="v1")

    # Rebuilding with a *different* catalog under the same version must not take
    # effect: the version is the identity, and honouring the second call would
    # desynchronise the index from the prompt-cache prefix keyed on that version.
    backend.build([Route(name="ghost", description="Should not be indexed.")], registry_version="v1")

    assert "ghost" not in backend.shortlist(
        "ghost", allowed=frozenset({"ghost", *_all(routes)}), k=5
    ).names


def test_index_key_is_version_fingerprint_scope() -> None:
    """``f"{registry_version}:{fingerprint}:{scope}"`` (plan §5.4)."""
    routes = _generated(30)
    backend = BM25Shortlister()
    backend.build(routes, registry_version="abc123")

    assert backend.index_key == f"abc123:{backend.fingerprint}:*"


def test_fingerprint_changes_with_configuration() -> None:
    """A config change invalidates persisted indexes exactly like a route edit (§5.4)."""
    default = BM25Shortlister()
    reweighted = BM25Shortlister(field_weights={"name": 9.0})
    retuned = BM25Shortlister(k1=1.2)

    assert default.fingerprint != reweighted.fingerprint
    assert default.fingerprint != retuned.fingerprint


# --------------------------------------------------------------------------- #
# EmbeddingShortlister — BYO callable, zero extras (plan §5.2)
# --------------------------------------------------------------------------- #

_VOCAB = ("refund", "order", "password", "weather", "escalate")


class _CountingEmbedder:
    """A deterministic bag-of-words embedder that records every batch it is given.

    Zero extras, exactly as §5.2 promises for the BYO path: no model download, no
    network call, and the vectors are trivially predictable so cosine ranking is
    assertable rather than approximate.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            row = [float(lowered.count(word)) for word in _VOCAB]
            # A small constant keeps every vector non-zero, so normalisation is
            # always defined even for a route that shares no vocabulary.
            vectors.append([*row, 0.25])
        return vectors

    @property
    def embedded(self) -> list[str]:
        """Every text ever handed to the embedder, flattened."""
        return [text for batch in self.batches for text in batch]


@pytest.fixture
def embed_routes() -> list[Route]:
    """Three routes whose vocabulary is disjoint enough to rank unambiguously."""
    return [
        Route(name="refund_order", description="refund refund order money back"),
        Route(name="reset_password", description="password password credentials"),
        Route(name="weather_now", description="weather weather forecast"),
    ]


def test_embedding_shortlister_ranks_by_cosine(embed_routes: list[Route]) -> None:
    """Brute-force cosine over normalized vectors (plan §5.2)."""
    embedder = _CountingEmbedder()
    backend = EmbeddingShortlister(embedder, min_routes=0)
    backend.build(embed_routes, registry_version="e1")

    result = backend.shortlist(
        "refund my order please", allowed=_all(embed_routes), k=2
    )

    assert _names(result)[0] == "refund_order"
    assert {candidate.source for candidate in result.candidates} == {"embed"}
    assert backend.dim == len(_VOCAB) + 1


def test_embedding_shortlister_re_embeds_only_changed_routes(
    embed_routes: list[Route],
) -> None:
    """Per-route embedding cache keyed on ``sha256(embed_text)`` (plan §7.3).

    "Adding one route to a 5000-route catalog costs exactly one embedding call."
    Here: editing one description of three costs exactly one.
    """
    embedder = _CountingEmbedder()
    backend = EmbeddingShortlister(embedder, min_routes=0)
    backend.build(embed_routes, registry_version="e1")
    assert len(embedder.embedded) == 3
    assert backend.cached_vectors == 3

    embedder.batches.clear()
    edited = [
        *embed_routes[:2],
        Route(name="weather_now", description="weather weather forecast and sunshine"),
    ]
    backend.build(edited, registry_version="e2")

    assert len(embedder.batches) == 1
    assert embedder.batches[0] == [edited[2].embed_text]


def test_embedding_cache_ignores_an_args_model_only_change(
    embed_routes: list[Route],
) -> None:
    """An ``args_model`` edit bumps ``registry_version`` without re-embedding (§7.3).

    ``embed_text`` is ``name + description + examples`` precisely so the argument
    schema — which lives in the prompt, and therefore correctly invalidates the
    prompt cache — does not drag the embedding index along with it.
    """
    from pydantic import BaseModel

    class _Args(BaseModel):
        order_id: str

    embedder = _CountingEmbedder()
    backend = EmbeddingShortlister(embedder, min_routes=0)
    backend.build(embed_routes, registry_version="e1")
    embedder.batches.clear()

    re_typed = [
        Route(
            name=embed_routes[0].name,
            description=embed_routes[0].description,
            args_model=_Args,
        ),
        *embed_routes[1:],
    ]
    backend.build(re_typed, registry_version="e2")

    assert embedder.batches == []


def test_embedding_shortlister_async_twin_matches_the_sync_one(
    embed_routes: list[Route],
) -> None:
    """Full sync/async parity is a library-wide rule (plan §2.5)."""
    backend = EmbeddingShortlister(_CountingEmbedder(), min_routes=0)
    backend.build(embed_routes, registry_version="e1")

    sync = backend.shortlist("refund my order", allowed=_all(embed_routes), k=3)
    everything = _all(embed_routes)
    async def _run() -> ShortlistResult:
        return await backend.ashortlist("refund my order", allowed=everything, k=3)

    assert _names(asyncio.run(_run())) == _names(sync)


def test_embedding_backend_rejects_a_ragged_matrix(embed_routes: list[Route]) -> None:
    """A dimensionality change without a fingerprint change is a ConfigError (§5.2)."""

    def ragged(texts: list[str]) -> list[list[float]]:
        return [[1.0] * (index + 1) for index in range(len(texts))]

    backend = EmbeddingShortlister(ragged, min_routes=0)
    with pytest.raises(ConfigError, match="dimensionality"):
        backend.build(embed_routes, registry_version="e1")


# --------------------------------------------------------------------------- #
# Config resolution (plan §3.3)
# --------------------------------------------------------------------------- #


def test_resolve_shortlister_understands_the_dsl() -> None:
    """``"auto"`` / ``"bm25:top_k=12"`` / ``None`` (plan §3.3, §5.2)."""
    assert isinstance(resolve_shortlister("auto"), AutoShortlister)
    tuned = resolve_shortlister("bm25:top_k=12")
    assert isinstance(tuned, BM25Shortlister)
    assert tuned.top_k == 12
    assert resolve_shortlister(None) is None


def test_resolve_shortlister_refuses_unknown_and_deferred_variants() -> None:
    """Hybrid is [v0.2]; it raises rather than silently degrading (plan §5.2)."""
    with pytest.raises(ConfigError, match="hybrid"):
        resolve_shortlister("hybrid")
    with pytest.raises(ConfigError, match="unknown shortlist variant"):
        resolve_shortlister("bm52")


def test_router_refuses_disabled_shortlist_over_an_oversized_catalog(
    large_registry: Registry, fake_client: Any
) -> None:
    """``shortlist=None`` + N > ``max_candidates`` is a ConfigError (plan §3.3).

    Silent truncation would drop routes, so the Router refuses at construction.
    ``"auto"`` never errors — that is the documented escape.
    """
    with pytest.raises(ConfigError, match="max_candidates"):
        Router(large_registry, client=fake_client, shortlist=None, otel=False)

    Router(large_registry, client=fake_client, shortlist="auto", otel=False)
