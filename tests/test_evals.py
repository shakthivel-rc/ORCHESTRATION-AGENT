"""Tests for ``switchboard.evals`` — fixtures, replay cache and runner (plan §9).

The suite is organised around the four claims plan §9 makes that a user would be
entitled to rely on:

1. a frozen catalog fixture **round-trips without changing ``registry_version``**,
   so an eval run and a production run can be proven to be about the same catalog;
2. the JSONL container is **schema-versioned** and says so when it is not;
3. ``--replay`` is **strict** — a cache miss fails the run — and a recorded call
   replays byte-identically, which is what makes the CI lane deterministic,
   keyless and free;
4. the runner's numbers are derived from the ``Decision`` + ``AuditRecord``
   production already emits, and the headline is the **delta versus the naive
   baseline**.

The client here is local rather than ``conftest.FakeLLMClient`` because these
tests need an *oracle* — a model that answers correctly exactly when the gold
route is in front of it — and a way to handicap the naive arm. That is the whole
mechanism the delta measures, so it has to be controllable per arm.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from switchboard import (
    ClientCapabilities,
    ConfigError,
    LLMRequest,
    LLMResult,
    PromptSegment,
    Registry,
    Route,
    Router,
    Usage,
)
from switchboard.evals import (
    CACHE_SCHEMA,
    FIXTURE_SCHEMA,
    NAIVE_FULL_CATALOG,
    NAIVE_SYSTEM_PROMPT,
    EvalCase,
    EvalSuite,
    ExpectedAbstain,
    ExpectedClarify,
    ExpectedMultiRoute,
    ExpectedRoute,
    ReplayCache,
    ReplayClient,
    SuiteResult,
    arun_suite,
    dogfood_registry,
    dogfood_suite,
    load_cases,
    load_suite,
    registry_from_fixture,
    registry_to_fixture,
    run_suite,
    sample_scope,
    save_cases,
    save_suite,
)
from switchboard.evals.cache import CacheKey

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

# --------------------------------------------------------------------------- #
# A local oracle client.
# --------------------------------------------------------------------------- #

EVAL_CAPS = ClientCapabilities(
    structured="tool_strict", logprobs=False, caching="explicit", reasoning_toggle=True
)
"""The Anthropic-shaped rung: static wire schema, no logprobs.

Chosen deliberately. Without logprobs the confidence thresholds are inert (§6.3),
so no decision is silently downgraded to clarify mid-test and the metrics under
assertion are the runner's arithmetic rather than the policy stage's.
"""

_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)


class OracleClient:
    """A model that answers correctly iff the gold route is in the prompt.

    Implements both provider Protocols (plan §4.1), so one instance drives
    ``route()`` and ``aroute()``.

    Args:
        answers: query text -> the route name that query should reach. A query
            absent from the mapping is answered ``abstain``.
        clarify: queries that should be answered with a clarify question.
        naive_wrong: queries the model gets wrong **only** when it sees
            :data:`~switchboard.evals.NAIVE_SYSTEM_PROMPT`. This is the handicap
            that stands in for the degradation the report documents at 100+ tools
            in one prompt — the thing the ``naive-full-catalog`` delta exists to
            quantify.
    """

    def __init__(
        self,
        answers: Mapping[str, str],
        *,
        clarify: Iterable[str] = (),
        naive_wrong: Iterable[str] = (),
        capabilities: ClientCapabilities | None = None,
        model_id: str = "fake/eval-model",
    ) -> None:
        self.answers = dict(answers)
        self.clarify = set(clarify)
        self.naive_wrong = set(naive_wrong)
        self.capabilities = capabilities if capabilities is not None else EVAL_CAPS
        self.model_id = model_id
        self.provider = "fake"
        self.calls = 0
        self.prompts: list[str] = []

    # -- provider Protocols -------------------------------------------------- #

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        prompt = "\n\n".join(segment.content for segment in request.segments)
        self.prompts.append(prompt)
        self.calls += 1
        payload = self._answer(prompt)
        return LLMResult(
            parsed=None,
            raw_text=json.dumps(payload, ensure_ascii=False),
            token_logprobs=None,
            usage=Usage(input_tokens=900, output_tokens=40),
            model_id=self.model_id,
            provider_meta={"response_id": f"fake-{self.calls:04d}"},
        )

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        return self.complete(request)

    # -- internals ----------------------------------------------------------- #

    def _answer(self, prompt: str) -> dict[str, Any]:
        match = _QUERY_RE.search(prompt)
        query = match.group(1) if match else ""
        naive = NAIVE_SYSTEM_PROMPT.splitlines()[0] in prompt

        if query in self.clarify:
            return {
                "rationale": "Several routes fit this equally well.",
                "kind": "clarify",
                "question": "Which workspace and which action did you mean?",
                "candidates": None,
                "missing": ["action", "workspace"],
            }
        route = self.answers.get(query)
        if route is None or (naive and query in self.naive_wrong):
            return {
                "rationale": "Nothing in the catalog handles this.",
                "kind": "abstain",
                "reason": "model_elected",
            }
        if f"- {route}:" not in prompt:
            # The gold route was not retrieved. A model cannot pick what it was
            # never shown — this is the retrieval gap recall@K measures (§9.2).
            return {
                "rationale": "No listed route matches this request.",
                "kind": "abstain",
                "reason": "model_elected",
            }
        return {
            "rationale": f"The request names the {route} operation directly.",
            "kind": "route",
            "route": route,
            "args": None,
        }


class _Args(BaseModel):
    """A tiny args model for hand-built provider requests."""

    rationale: str = ""


def _request(text: str = "hello", *, seed: int | None = None) -> LLMRequest:
    """A minimal, hand-built :class:`~switchboard.LLMRequest` for cache tests."""
    return LLMRequest(
        segments=(
            PromptSegment(role="system", content="rules", cache="stable"),
            PromptSegment(role="user", content=text),
        ),
        output_schema=_Args,
        temperature=0.0,
        seed=seed,
    )


@pytest.fixture
def eval_registry() -> Registry:
    """A 30-route synthetic catalog — above the ``"auto"`` bypass line (§5.3)."""
    return dogfood_registry(n_routes=30)


@pytest.fixture
def eval_suite() -> EvalSuite:
    """The dogfood suite over that catalog: route, clarify and abstain golds."""
    return dogfood_suite(n_routes=30)


def _oracle_for(suite: EvalSuite, *, naive_wrong_every: int = 3) -> OracleClient:
    """Build an oracle that knows this suite's gold labels."""
    answers: dict[str, str] = {}
    clarify: list[str] = []
    route_queries: list[str] = []
    for case in suite.cases:
        if isinstance(case.expected, ExpectedRoute):
            answers[case.query] = case.expected.any_of[0]
            route_queries.append(case.query)
        elif isinstance(case.expected, ExpectedClarify):
            clarify.append(case.query)
    return OracleClient(
        answers,
        clarify=clarify,
        naive_wrong=route_queries[::naive_wrong_every],
    )


def _router_for(registry: Registry, client: Any, **kwargs: Any) -> Router:
    """A candidate Router with test-friendly defaults (no OTel, no sink)."""
    kwargs.setdefault("shortlist", "auto")
    kwargs.setdefault("otel", False)
    kwargs.setdefault(
        "retry",
        {"schema_attempts": 2, "provider_attempts": 2, "backoff": "expo", "base_delay": 0.0},
    )
    return Router(registry, client=client, **kwargs)


# --------------------------------------------------------------------------- #
# 1. Catalog fixtures — the version-preserving round trip (plan §9.1, §7.3).
# --------------------------------------------------------------------------- #


class TrackArgs(BaseModel):
    """An args model with a required and an optional field."""

    order_id: str
    reason: str | None = None
    quantity: int = 1


def _rich_registry() -> Registry:
    """A catalog exercising every field the fixture has to carry."""
    return Registry(
        [
            Route(
                name="refund",
                description="Issue or check a refund for an order.",
                args_model=TrackArgs,
                examples=("I want my money back", "refund order 12"),
                tags=frozenset({"billing", "orders"}),
                requires=frozenset({"billing"}),
                clarify_label="a refund",
                group="commerce",
                metadata={"owner": "payments", "handler": print},
            ),
            Route(
                name="human_handoff",
                description="Escalate to a human support agent.",
                pinned=True,
            ),
        ]
    )


def test_registry_fixture_round_trip_preserves_version(eval_registry: Registry) -> None:
    """registry -> fixture -> registry keeps ``registry_version`` (plan §9.1, §7.3).

    This is the load-bearing property of the whole fixture format. If the version
    moved, every audit record from a replayed run would name a catalog that never
    existed, and the run could not be compared to production at all.
    """
    fixture = registry_to_fixture(eval_registry)
    assert fixture["fixture"] == FIXTURE_SCHEMA
    assert fixture["registry_version"] == eval_registry.version
    assert fixture["content_hash"] == eval_registry.content_hash

    rebuilt = registry_from_fixture(fixture)
    assert rebuilt.version == eval_registry.version
    assert rebuilt.content_hash == eval_registry.content_hash
    assert rebuilt.names == eval_registry.names
    assert rebuilt == eval_registry


def test_registry_fixture_round_trip_with_args_models() -> None:
    """Args schemas survive as JSON Schema and rebuild as permissive models (§9.1)."""
    registry = _rich_registry()
    rebuilt = registry_from_fixture(registry_to_fixture(registry))

    assert rebuilt.version == registry.version

    original = registry["refund"]
    copy = rebuilt["refund"]
    assert copy.args_model is not None
    # The pinned schema is returned verbatim — that is what preserves the hash.
    assert copy.args_model.model_json_schema() == TrackArgs.model_json_schema()
    # ...and the rebuilt model is a usable, permissive model.
    instance = copy.args_model(order_id="A-1")
    assert instance.model_dump()["order_id"] == "A-1"
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        copy.args_model()

    # Behaviour-bearing, non-hashed fields survive too.
    assert copy.clarify_label == original.clarify_label
    assert copy.group == original.group
    assert copy.requires == original.requires
    assert rebuilt["human_handoff"].pinned is True
    # Non-JSON metadata is dropped, exactly as Route.content_hash drops it.
    assert copy.metadata == {"owner": "payments"}
    # One route card renders identically on both sides (§5.6).
    assert copy.card() == original.card()


def test_registry_fixture_strict_rejects_a_tampered_catalog() -> None:
    """A fixture whose content moved no longer reproduces its pinned version."""
    fixture = registry_to_fixture(_rich_registry())
    fixture["routes"][0]["description"] = "Something else entirely."

    with pytest.raises(ConfigError, match="pinned to registry_version"):
        registry_from_fixture(fixture)

    # ...and can still be inspected deliberately.
    inspected = registry_from_fixture(fixture, strict=False)
    assert inspected["refund"].description == "Something else entirely."


def test_registry_fixture_rejects_unknown_schema() -> None:
    """An unknown fixture family/version is refused, never parsed hopefully."""
    fixture = registry_to_fixture(_rich_registry())
    fixture["fixture"] = "sb-eval/99"
    with pytest.raises(ConfigError, match="version '99' is not supported"):
        registry_from_fixture(fixture)

    fixture["fixture"] = "some-other-tool/1"
    with pytest.raises(ConfigError, match="not a switchboard eval fixture"):
        registry_from_fixture(fixture)


# --------------------------------------------------------------------------- #
# 2. EvalCase parsing and the JSONL container (plan §9.1).
# --------------------------------------------------------------------------- #


def test_eval_case_expected_is_discriminated_on_kind() -> None:
    """All four gold labels parse from plain JSON, discriminated on ``kind`` (§9.1)."""
    route = EvalCase.model_validate(
        {"id": "a", "query": "q", "expected": {"kind": "route", "any_of": ["refund"]}}
    )
    multi = EvalCase.model_validate(
        {
            "id": "b",
            "query": "q",
            "expected": {"kind": "multi_route", "routes": ["a", "b"], "order_sensitive": True},
        }
    )
    clarify = EvalCase.model_validate(
        {
            "id": "c",
            "query": "q",
            "expected": {"kind": "clarify", "missing": ["order_id"], "acceptable_routes": ["r"]},
        }
    )
    abstain = EvalCase.model_validate({"id": "d", "query": "q", "expected": {"kind": "abstain"}})

    assert isinstance(route.expected, ExpectedRoute)
    assert isinstance(multi.expected, ExpectedMultiRoute)
    assert isinstance(clarify.expected, ExpectedClarify)
    assert isinstance(abstain.expected, ExpectedAbstain)

    # Defaults specified by the plan.
    assert route.expected.args_match == "subset"
    assert route.source == "hand"
    assert multi.expected.order_sensitive is True

    # Gold-route projection, used by recall@K.
    assert route.gold_routes() == frozenset({"refund"})
    assert multi.gold_routes() == frozenset({"a", "b"})
    assert clarify.gold_routes() == frozenset({"r"})
    assert abstain.gold_routes() == frozenset()

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        EvalCase.model_validate({"id": "e", "query": "q", "expected": {"kind": "nope"}})


def test_jsonl_round_trip_with_header(tmp_path: Path) -> None:
    """Cases save and load through the ``{"fixture": "sb-eval/1"}`` header (§9.1)."""
    cases = [
        EvalCase(
            id="one",
            query="refund order 12",
            expected=ExpectedRoute(any_of=["refund"]),
            tags={"billing", "smoke"},
            context={"tenant_id": "acme", "locale": "en-GB"},
        ),
        EvalCase(id="two", query="the weather", expected=ExpectedAbstain(), source="synthetic"),
    ]
    path = tmp_path / "nested" / "cases.jsonl"
    save_cases(cases, path, name="unit")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["fixture"] == FIXTURE_SCHEMA
    assert json.loads(lines[0])["suite"] == "unit"
    assert len(lines) == 3
    # Tags are serialised sorted, so the file is byte-stable across processes.
    assert json.loads(lines[1])["tags"] == ["billing", "smoke"]

    loaded = load_cases(path)
    assert loaded == cases
    suite = load_suite(path)
    assert suite.name == "unit"
    assert len(suite) == 2
    assert [case.id for case in suite] == ["one", "two"]


def test_jsonl_pins_the_catalog(tmp_path: Path, eval_registry: Registry) -> None:
    """A suite can carry its frozen catalog in the header line (plan §9.1)."""
    suite = dogfood_suite(n_routes=30)
    path = save_suite(suite, tmp_path / "dogfood.jsonl")

    reloaded = load_suite(path)
    assert reloaded.name == suite.name
    assert len(reloaded) == len(suite)
    rebuilt = reloaded.registry()
    assert rebuilt is not None
    assert rebuilt.version == eval_registry.version


def test_jsonl_header_is_validated(tmp_path: Path) -> None:
    """Missing, foreign and future headers all fail loudly (plan §9.1)."""
    missing = tmp_path / "no-header.jsonl"
    missing.write_text(
        json.dumps({"id": "a", "query": "q", "expected": {"kind": "abstain"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be the header"):
        load_cases(missing)

    future = tmp_path / "future.jsonl"
    future.write_text(json.dumps({"fixture": "sb-eval/7"}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="version '7' is not supported"):
        load_cases(future)

    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text(json.dumps({"fixture": "bfcl/1"}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a switchboard eval fixture"):
        load_cases(foreign)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="is empty"):
        load_cases(empty)

    with pytest.raises(ConfigError, match="not found"):
        load_cases(tmp_path / "nope.jsonl")


def test_jsonl_rejects_bad_lines_with_line_numbers(tmp_path: Path) -> None:
    """Errors name the line, because fixtures are hand-edited (plan §9.1)."""
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps({"fixture": FIXTURE_SCHEMA}) + "\n" + "{not json\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r":2: not valid JSON"):
        load_cases(path)

    duplicate = tmp_path / "dupe.jsonl"
    line = json.dumps({"id": "a", "query": "q", "expected": {"kind": "abstain"}})
    duplicate.write_text(
        json.dumps({"fixture": FIXTURE_SCHEMA}) + "\n" + line + "\n" + line + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate case id"):
        load_cases(duplicate)


def test_dogfood_suite_shape(eval_suite: EvalSuite, eval_registry: Registry) -> None:
    """The dogfood suite labels all three v0.1 gold kinds (plan §9.6)."""
    kinds = {case.expected.kind for case in eval_suite.cases}
    assert kinds == {"route", "clarify", "abstain"}
    assert len(eval_suite) == 40
    assert eval_suite.catalog is not None
    # Every gold route it names exists in the catalog it pins.
    names = set(eval_registry.names)
    for case in eval_suite.cases:
        assert case.gold_routes() <= names


# --------------------------------------------------------------------------- #
# 3. The record/replay cache (plan §9.6).
# --------------------------------------------------------------------------- #


def test_replay_strict_miss_raises(tmp_path: Path) -> None:
    """``mode="replay"`` fails the run on a miss — CI determinism (plan §9.6)."""
    client = ReplayClient(
        cache=tmp_path / "cache",
        mode="replay",
        model="fake/eval-model",
        capabilities=EVAL_CAPS,
    )
    with pytest.raises(ConfigError, match="replay cache miss in strict mode"):
        client.complete(_request())

    # It never falls back to a network call, and it never needed a client.
    assert client.wrapped is None


def test_replay_mode_requires_a_client_to_record(tmp_path: Path) -> None:
    """Only strict replay may run without a wrapped client (plan §9.6)."""
    with pytest.raises(ConfigError, match="needs a wrapped client"):
        ReplayClient(cache=tmp_path, mode="auto", model="m")
    with pytest.raises(ConfigError, match="must be 'record', 'replay' or 'auto'"):
        ReplayClient(cache=tmp_path, mode="sometimes", model="m")  # type: ignore[arg-type]


def test_record_then_replay_reproduces_byte_identically(tmp_path: Path) -> None:
    """A recorded call replays to the same bytes, from a fresh process-equivalent.

    This is the property the whole §9.6 CI story rests on: the replay lane must
    produce the *same decision* as the recorded live lane, which means the
    ``LLMResult`` must come back unchanged, not merely equivalent.
    """
    inner = OracleClient({}, model_id="fake/eval-model")
    directory = tmp_path / "cache"
    request = _request("record me", seed=11)

    recorder = ReplayClient(inner, cache=directory, mode="record")
    recorded = recorder.complete(request)
    recorder.cache.close()
    assert inner.calls == 1

    # The file is JSONL under a cache dir, with a schema header (plan §9.6).
    path = directory / "llm-calls.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"cache": CACHE_SCHEMA}
    assert len(lines) == 2

    # A brand-new cache object re-reads the file — no shared in-memory state.
    replayer = ReplayClient(
        cache=ReplayCache(directory),
        mode="replay",
        model="fake/eval-model",
        capabilities=inner.capabilities,
    )
    replayed = replayer.complete(request)

    assert replayed.model_dump(mode="json") == recorded.model_dump(mode="json")
    assert replayed.raw_text == recorded.raw_text
    assert inner.calls == 1, "replay must not reach the wrapped client"
    assert replayer.cache.stats()["hits"] == 1


def test_auto_mode_replays_hits_and_records_misses(tmp_path: Path) -> None:
    """``mode="auto"`` is the developer loop: hit -> replay, miss -> record (§9.6)."""
    inner = OracleClient({}, model_id="fake/eval-model")
    client = ReplayClient(inner, cache=tmp_path / "cache", mode="auto")

    first = client.complete(_request("q1"))
    assert inner.calls == 1
    second = client.complete(_request("q1"))
    assert inner.calls == 1
    assert second.raw_text == first.raw_text

    client.complete(_request("q2"))
    assert inner.calls == 2
    assert len(client.cache) == 2


def test_cache_key_covers_prompt_schema_and_sampling(tmp_path: Path) -> None:
    """The key is the plan §9.6 tuple, and every component actually moves it."""
    inner = OracleClient({}, model_id="fake/eval-model")
    client = ReplayClient(inner, cache=tmp_path / "cache", mode="auto")

    base = client.key_for(_request("same"))
    assert isinstance(base, CacheKey)
    assert base.model == "fake/eval-model"
    assert base.sample_index == 0

    assert client.key_for(_request("other")).prompt_hash != base.prompt_hash
    assert client.key_for(_request("same", seed=3)).seed == 3
    assert client.key_for(_request("same", seed=3)).digest != base.digest


def test_sample_scope_isolates_repeated_identical_calls(tmp_path: Path) -> None:
    """Identical calls within one scope get distinct ``sample_index`` values (§9.6)."""
    inner = OracleClient({}, model_id="fake/eval-model")
    client = ReplayClient(inner, cache=tmp_path / "cache", mode="auto")

    with sample_scope():
        first = client.key_for(_request("s"))
        second = client.key_for(_request("s"))
    assert (first.sample_index, second.sample_index) == (0, 1)

    # A fresh scope rewinds, so per-case keys do not depend on case ordering.
    with sample_scope():
        assert client.key_for(_request("s")).sample_index == 0


def test_cache_rejects_a_foreign_schema(tmp_path: Path) -> None:
    """A cache file from another build is refused rather than misread (§9.6)."""
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"cache": "sb-replay/99"}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="replay cache schema"):
        ReplayCache(path)


# --------------------------------------------------------------------------- #
# 4. The runner (plan §9.2, §9.3).
# --------------------------------------------------------------------------- #


def _recount(result: SuiteResult) -> float:
    """Recompute accuracy from the per-case rows, independently of the metrics."""
    cases = result.candidate.cases
    return sum(1 for case in cases if case.correct) / len(cases)


def test_run_suite_end_to_end(eval_registry: Registry, eval_suite: EvalSuite) -> None:
    """A full run over a 30-route catalog: metrics, baseline, delta (plan §9)."""
    oracle = _oracle_for(eval_suite)
    router = _router_for(eval_registry, oracle)

    result = run_suite(router, eval_suite)

    # -- shape -------------------------------------------------------------- #
    assert result.suite == eval_suite.name
    assert result.n_cases == len(eval_suite)
    assert result.n_routes == 30
    assert result.registry_version == eval_registry.version
    assert len(result.candidate.cases) == len(eval_suite)

    metrics = result.candidate.metrics
    assert metrics.n_cases == len(eval_suite)
    assert metrics.n_errors == 0

    # -- accuracy is the arithmetic it claims to be ------------------------- #
    assert metrics.accuracy == pytest.approx(_recount(result))
    route_cases = [c for c in result.candidate.cases if c.expected_kind == "route"]
    assert metrics.n_route_cases == len(route_cases)
    assert metrics.route_accuracy == pytest.approx(
        sum(1 for c in route_cases if c.correct) / len(route_cases)
    )

    # The oracle answers correctly whenever the gold route was retrieved, so
    # route accuracy is exactly shortlist recall — the retrieval-gap identity.
    assert metrics.route_accuracy == pytest.approx(metrics.recall_at_k)

    # -- clarify and abstain are gold labels, and were hit ------------------ #
    assert metrics.clarify_recall == pytest.approx(1.0)
    assert metrics.abstain_recall == pytest.approx(1.0)
    assert metrics.harmful_clarify == 0
    assert metrics.missed_clarify == 0

    # -- the invariants the plan states outright ---------------------------- #
    assert metrics.hallucinated_routes == 0, "a committed route is always registered"
    assert metrics.schema_validity == pytest.approx(1.0)
    assert metrics.latency_p50_ms is not None
    assert metrics.latency_p95_ms is not None

    # -- the baseline ran, and the headline is the delta -------------------- #
    naive = result.arm(NAIVE_FULL_CATALOG)
    assert naive is not None
    assert len(naive.cases) == len(eval_suite)
    assert naive.metrics.accuracy < metrics.accuracy
    delta = result.delta()
    assert delta is not None
    assert delta == pytest.approx(metrics.accuracy - naive.metrics.accuracy)
    assert delta > 0
    assert f"{delta * 100:+.2f} pp" in result.headline


def test_naive_baseline_puts_the_whole_catalog_in_the_prompt(
    eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """The baseline is naive in exactly the ways §9.3 specifies, and no others."""
    oracle = _oracle_for(eval_suite)
    router = _router_for(eval_registry, oracle)
    result = run_suite(router, eval_suite)

    naive = result.arm(NAIVE_FULL_CATALOG)
    assert naive is not None

    # No shortlist: every case saw the entire entitled catalog, unshuffled.
    assert all(case.shortlist_skipped for case in naive.cases)
    assert all(case.shortlist_size == len(eval_registry) for case in naive.cases)
    # ...so recall@K is undefined for it, not a free 100%.
    assert naive.metrics.recall_at_k is None

    # The candidate, by contrast, actually retrieved.
    assert not any(case.shortlist_skipped for case in result.candidate.cases)
    assert result.candidate.metrics.recall_at_k is not None

    # Same client and model on both arms (§9.3: differences are architectural).
    assert naive.model == result.candidate.model

    # The naive system block reached the provider.
    assert any(NAIVE_SYSTEM_PROMPT.splitlines()[0] in prompt for prompt in oracle.prompts)


def test_baseline_names_are_phase_gated(eval_registry: Registry, eval_suite: EvalSuite) -> None:
    """[v0.2] baselines are refused by name, never silently skipped (plan §9.3)."""
    router = _router_for(eval_registry, _oracle_for(eval_suite))
    with pytest.raises(ConfigError, match=r"embed-top1' is \[v0.2\]"):
        run_suite(router, eval_suite, baselines=["embed-top1"])
    with pytest.raises(ConfigError, match="unknown baseline"):
        run_suite(router, eval_suite, baselines=["nonsense"])


def test_run_suite_without_a_baseline_has_no_headline_number(
    eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """No baseline means no delta — never a zero delta (plan §9.3)."""
    router = _router_for(eval_registry, _oracle_for(eval_suite))
    result = run_suite(router, eval_suite, baselines=())

    assert result.baselines == []
    assert result.delta() is None
    assert "no naive-full-catalog baseline ran" in result.headline
    gate = next(g for g in result.gates if g.id == "G1")
    assert gate.passed is None


def test_proto_gates_are_advisory_below_the_statistical_floor(
    eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """G1/G4 run in the v0.1 dogfood but do not bind at n < 200 (plan §9.2, §9.5)."""
    router = _router_for(eval_registry, _oracle_for(eval_suite))
    result = run_suite(router, eval_suite)

    ids = [gate.id for gate in result.gates]
    assert ids == ["G1", "G4"]
    assert all(gate.binding is False for gate in result.gates)
    g1 = result.gates[0]
    assert g1.threshold == pytest.approx(0.10)
    assert g1.passed is (g1.value is not None and g1.value >= 0.10)

    table = result.summary_table()
    assert "advisory" in table
    assert "below the n >= 200 statistical floor" in table


def test_summary_table_and_to_json(
    tmp_path: Path, eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """The report renders without ``rich`` and serialises losslessly (plan §9)."""
    router = _router_for(eval_registry, _oracle_for(eval_suite))
    result = run_suite(router, eval_suite)

    table = result.summary_table()
    assert "accuracy" in table
    assert "shortlist recall@K" in table
    assert NAIVE_FULL_CATALOG in table
    assert "headline:" in table
    assert "confusion (gold->predicted)" in table
    assert str(result) == table

    path = tmp_path / "out" / "run.json"
    text = result.to_json(path)
    assert json.loads(text)["suite"] == eval_suite.name
    reloaded = SuiteResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.candidate.metrics.accuracy == pytest.approx(
        result.candidate.metrics.accuracy
    )
    assert len(reloaded.candidate.cases) == len(result.candidate.cases)


def test_run_suite_accepts_a_bare_iterable_of_cases(eval_registry: Registry) -> None:
    """``suite`` may be any iterable of cases, not only an ``EvalSuite``."""
    cases = [
        EvalCase(
            id="c1",
            query="create the billing entry in workspace 0",
            expected=ExpectedRoute(any_of=["billing_create_0"]),
        ),
        EvalCase(id="c2", query="what is the weather", expected=ExpectedAbstain()),
    ]
    oracle = OracleClient({cases[0].query: "billing_create_0"})
    router = _router_for(eval_registry, oracle)

    result = run_suite(router, cases, baselines=())
    assert result.n_cases == 2
    assert result.candidate.metrics.accuracy == pytest.approx(1.0)

    with pytest.raises(ConfigError, match="iterable of"):
        run_suite(router, ["not a case"], baselines=())  # type: ignore[list-item]


def test_errors_are_recorded_not_raised(eval_registry: Registry) -> None:
    """A provider failure loses a case, not the run (plan §9.2)."""

    class Broken(OracleClient):
        def complete(self, request: LLMRequest) -> LLMResult[Any]:
            from switchboard import ProviderTimeout

            raise ProviderTimeout("nope")

    router = _router_for(eval_registry, Broken({}))
    cases = [EvalCase(id="x", query="anything", expected=ExpectedAbstain())]

    result = run_suite(router, cases, baselines=())
    case = result.candidate.cases[0]
    assert case.predicted_kind == "error"
    assert case.error is not None and "ProviderTimeout" in case.error
    assert result.candidate.metrics.n_errors == 1
    assert result.candidate.metrics.accuracy == 0.0


# --------------------------------------------------------------------------- #
# 5. Runner + cache together — the §9.6 CI lane, end to end.
# --------------------------------------------------------------------------- #


def test_replay_run_reproduces_the_recorded_run(
    tmp_path: Path, eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """Record a whole suite, then replay it strictly with no client at all (§9.6)."""
    directory = tmp_path / "replay"
    oracle = _oracle_for(eval_suite)

    recorded = run_suite(
        _router_for(eval_registry, oracle), eval_suite, cache=directory, seed=7
    )
    assert recorded.mode == "auto"
    assert recorded.cache_stats is not None
    assert recorded.cache_stats["writes"] > 0
    live_calls = oracle.calls
    assert live_calls > 0

    strict = ReplayClient(
        cache=ReplayCache(directory),
        mode="replay",
        model=oracle.model_id,
        capabilities=oracle.capabilities,
    )
    replayed = run_suite(_router_for(eval_registry, strict), eval_suite, seed=7)

    assert replayed.mode == "replay"
    assert oracle.calls == live_calls, "the replay lane is keyless: no provider call"
    assert replayed.candidate.metrics.accuracy == pytest.approx(
        recorded.candidate.metrics.accuracy
    )
    assert replayed.candidate.metrics.recall_at_k == pytest.approx(
        recorded.candidate.metrics.recall_at_k
    )
    naive = replayed.arm(NAIVE_FULL_CATALOG)
    recorded_naive = recorded.arm(NAIVE_FULL_CATALOG)
    assert naive is not None and recorded_naive is not None
    assert naive.metrics.accuracy == pytest.approx(recorded_naive.metrics.accuracy)
    assert replayed.cache_stats is not None
    assert replayed.cache_stats["misses"] == 0


def test_strict_replay_miss_fails_the_run(
    tmp_path: Path, eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """An empty cache in strict mode fails the run rather than degrading (§9.6)."""
    strict = ReplayClient(
        cache=tmp_path / "empty",
        mode="replay",
        model="fake/eval-model",
        capabilities=EVAL_CAPS,
    )
    router = _router_for(eval_registry, strict)
    with pytest.raises(ConfigError, match="replay cache miss in strict mode"):
        run_suite(router, eval_suite.cases[:1], baselines=())


def test_run_suite_leaves_the_callers_router_alone(
    tmp_path: Path, eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """Wiring a cache clones the Router; the caller's client is untouched (§2.5)."""
    oracle = _oracle_for(eval_suite)
    router = _router_for(eval_registry, oracle)

    run_suite(router, eval_suite.cases[:2], baselines=(), cache=tmp_path / "c")
    assert router.client is oracle


# --------------------------------------------------------------------------- #
# 6. Sync/async parity (plan §2.5).
# --------------------------------------------------------------------------- #


async def test_arun_suite_matches_run_suite(
    eval_registry: Registry, eval_suite: EvalSuite
) -> None:
    """``arun_suite`` scores identically to ``run_suite`` (plan §2.5, §9)."""
    sync_result = run_suite(_router_for(eval_registry, _oracle_for(eval_suite)), eval_suite)
    async_result = await arun_suite(
        _router_for(eval_registry, _oracle_for(eval_suite)), eval_suite, concurrency=4
    )

    assert async_result.candidate.metrics.accuracy == pytest.approx(
        sync_result.candidate.metrics.accuracy
    )
    assert async_result.candidate.metrics.recall_at_k == pytest.approx(
        sync_result.candidate.metrics.recall_at_k
    )
    assert async_result.candidate.metrics.confusion == sync_result.candidate.metrics.confusion
    # Case order is preserved despite the concurrency limiter.
    assert [c.case_id for c in async_result.candidate.cases] == [
        c.case_id for c in sync_result.candidate.cases
    ]
    assert async_result.delta() == pytest.approx(sync_result.delta())

    with pytest.raises(ConfigError, match="concurrency"):
        await arun_suite(
            _router_for(eval_registry, _oracle_for(eval_suite)), eval_suite, concurrency=0
        )
