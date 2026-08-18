"""``Router`` end to end — the two drivers over one pipeline (plan §2.5, §3.6).

Everything below runs against a scripted :class:`~conftest.FakeLLMClient`, so the
suite is offline, deterministic and keyless. What is being proven is not that a
model can route — it is that the *contract around* the model holds:

* all four v0.1 decision kinds come back typed, with a full audit record (§3.4);
* ``route()`` and ``aroute()`` produce the **same Decision** for the same input,
  because they are two drivers over identical pure steps and not two
  implementations (§2.5);
* everything that can be wrong about a configuration is wrong at ``Router(...)``
  (§2.4, §3.8);
* schema exhaustion **degrades and never raises**, while transport failure
  **raises by default** — the raise-vs-degrade rule of §3.8, in both directions;
* an audit record is emitted on the success path *and* on the raise path,
  because the emit lives in a ``finally`` (§2.1);
* the redactor is one hook reaching every egress site (§7.4);
* one Router instance is safe across concurrent tasks (§2.5).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from conftest import GRAMMAR_LOGPROBS, NO_SIGNAL, FakeLLMClient, wire_output
from switchboard import (
    AbstainDecision,
    ClarifyDecision,
    ClientCapabilities,
    InMemorySink,
    MultiRouteDecision,
    Registry,
    RequestContext,
    Route,
    RouteDecision,
    Router,
)
from switchboard.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimit,
    ProviderTimeout,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from switchboard import Decision, LLMRequest

REFUND = wire_output("refund", {"order_id": "A-123", "reason": "damaged"})


def _volatile_audit_fields(decision: Decision) -> dict[str, Any]:
    """A decision's audit record minus the fields that cannot repeat.

    ``decision_id`` is a fresh ULID, the timestamps are wall clock, the latency
    block is a measurement and ``response_id`` is stamped per provider call.
    Everything else must be identical between two runs of the same input — that
    is what "identical Decision" means for an artifact that carries its own
    provenance.
    """
    record = decision.audit.model_dump(mode="json")
    for field in ("decision_id", "ts_start", "ts_end", "latency", "response_id"):
        record.pop(field, None)
    return record


# --------------------------------------------------------------------------- #
# The four decision kinds (plan §3.4)
# --------------------------------------------------------------------------- #


def test_route_decision(router_factory: Callable[..., Router], ctx: RequestContext) -> None:
    """The happy path: one committed route with validated args (plan §3.4)."""
    router = router_factory([REFUND])

    decision = router.route("I want my money back for order A-123", context=ctx)

    assert isinstance(decision, RouteDecision)
    assert decision.kind == "route"
    assert decision.route == "refund"
    assert decision.args is not None
    assert decision.args.order_id == "A-123"
    assert decision.decision_path == "llm"
    assert decision.downgraded_from is None
    assert decision.rationale
    assert decision.confidence.method == "logprobs"
    assert decision.audit.kind == "route"
    assert decision.audit.routes == ["refund"]
    assert decision.audit.registry_version == router.registry.version


def test_multi_route_decision(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``multi_route`` is off by default and typed when enabled (plan §3.3, §3.4)."""
    payload = wire_output(
        None,
        kind="multi_route",
        routes=[
            {"route": "refund", "args": {"order_id": "A-1"}},
            {"route": "track_order", "args": {"order_id": "A-1"}},
        ],
    )
    router = router_factory([payload], multi_route=True)

    decision = router.route("refund order A-1 and tell me where it is", context=ctx)

    assert isinstance(decision, MultiRouteDecision)
    assert [call.route for call in decision.routes] == ["refund", "track_order"]
    assert decision.routes[0].args is not None
    assert decision.routes[0].args.order_id == "A-1"
    assert decision.audit.routes == ["refund", "track_order"]


def test_a_kind_the_router_did_not_offer_is_not_decodable(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``multi_route`` is off by default, so it is absent from the ``kind`` enum.

    The model therefore cannot emit it at all (plan §4.4): the payload fails the
    schema, burns the repair budget, and degrades to ``unparseable_output`` — the
    router never has to decide what to do with a commitment it did not ask for.
    (The policy stage's defence-in-depth handling of a parsed-but-unconfigured
    ``multi_route`` is exercised directly in ``test_validate_confidence_policy``.)
    """
    payload = wire_output(
        None,
        kind="multi_route",
        routes=[{"route": "refund", "args": None}, {"route": "track_order", "args": None}],
    )
    router = router_factory([payload, payload, payload])

    decision = router.route("do both please", context=ctx)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "unparseable_output"


def test_clarify_decision(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """Model-elected clarify passes through untouched (plan §3.8, §6.5)."""
    payload = wire_output(
        None,
        kind="clarify",
        question="Do you want a refund or shipment tracking?",
        candidates=["refund", "track_order"],
        missing=["order_id"],
    )
    router = router_factory([payload])

    decision = router.route("my order", context=ctx)

    assert isinstance(decision, ClarifyDecision)
    assert decision.question == "Do you want a refund or shipment tracking?"
    assert decision.candidates == ("refund", "track_order")
    assert decision.missing == ("order_id",)
    assert decision.audit.abstain_reason == "model_elected"
    assert decision.audit.routes == []


def test_abstain_decision(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """Abstaining is a *result*, not an exception (plan §3, decision (c))."""
    router = router_factory([wire_output(None, kind="abstain", reason="model_elected")])

    decision = router.route("what is the weather in Lisbon?", context=ctx)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "model_elected"
    assert decision.audit.abstain_reason == "model_elected"


def test_no_eligible_routes_abstains_without_calling_the_model(
    fake_client: FakeLLMClient, sink: InMemorySink
) -> None:
    """An empty entitled set costs zero tokens (plan §3.8 row 1, §13 ruling #3)."""
    registry = Registry(
        [
            Route(
                name="secret_op",
                description="A privileged operation.",
                requires=frozenset({"admin"}),
            )
        ]
    )
    router = Router(registry, client=fake_client, sink=sink, otel=False)

    decision = router.route("do the thing", context=RequestContext(tenant_id="acme"))

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "no_eligible_routes"
    assert fake_client.calls == 0
    assert sink.latest is not None
    assert sink.latest.candidates_entitled == 0


def test_args_failure_downgrades_to_clarify(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """§3.8 row 4: the route was sound, the missing argument is a user question.

    The args-only failure gets its own single repair pass (§4.5) and then becomes
    a ``clarify`` naming exactly what is absent — a question the caller can put to
    the user verbatim.
    """
    missing_args = wire_output("refund", {"reason": "damaged"})
    router = router_factory([missing_args, missing_args, missing_args])

    decision = router.route("I want my money back", context=ctx)

    assert isinstance(decision, ClarifyDecision)
    assert decision.missing == ("order_id",)
    assert decision.downgraded_from == "route"
    assert decision.audit.abstain_reason == "invalid_args"
    # One repair pass, not the whole schema budget (§4.5).
    assert router.client.calls == 2


def test_args_failure_abstains_when_clarify_is_disabled(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """The other arm of §3.8 row 4."""
    missing_args = wire_output("refund", {"reason": "damaged"})
    router = router_factory([missing_args, missing_args], allow_clarify=False)

    decision = router.route("I want my money back", context=ctx)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "invalid_args"


def test_a_hallucinated_route_is_reported_as_such(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``invalid_route_reference``, not ``unparseable_output`` (plan §3.8, §9.2).

    The distinction is load-bearing: it is exactly the split between "the model
    cannot emit valid JSON" and "the model picked a route that does not exist",
    which the retrieval-gap vs confusion-gap decomposition is computed from.
    """
    ghost = wire_output("ghost_route", {"order_id": "A-1"})
    router = router_factory([ghost, ghost, ghost])

    decision = router.route("something", context=ctx)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "invalid_route_reference"


# --------------------------------------------------------------------------- #
# Sync/async parity (plan §2.5)
# --------------------------------------------------------------------------- #


def test_sync_and_async_produce_the_same_decision(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """"Parity is structural" (plan §2.5) — so assert it structurally.

    Same script, same query, same context: the two drivers must agree on the
    decision, on the confidence report, and on every audit field that is not a
    clock reading or a fresh identifier.
    """
    sync_router = router_factory([REFUND], fallback="human_handoff")
    async_router = router_factory([REFUND], fallback="human_handoff")

    sync_decision = sync_router.route("refund order A-123", context=ctx)
    async_decision = asyncio.run(async_router.aroute("refund order A-123", context=ctx))

    assert isinstance(sync_decision, RouteDecision)
    assert isinstance(async_decision, RouteDecision)
    assert sync_decision.kind == async_decision.kind
    assert sync_decision.route == async_decision.route
    assert sync_decision.args == async_decision.args
    assert sync_decision.rationale == async_decision.rationale
    assert sync_decision.confidence == async_decision.confidence
    assert _volatile_audit_fields(sync_decision) == _volatile_audit_fields(async_decision)
    # And the prompt itself is byte-identical, which is what makes replay work.
    assert sync_router.client.last_prompt == async_router.client.last_prompt


@pytest.mark.parametrize(
    "payload",
    [
        REFUND,
        wire_output(None, kind="clarify", question="Which?", candidates=["refund"]),
        wire_output(None, kind="abstain", reason="model_elected"),
    ],
    ids=["route", "clarify", "abstain"],
)
def test_parity_holds_for_every_decision_kind(
    payload: dict[str, Any], router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """Parity is not a happy-path-only property (plan §2.5)."""
    sync_decision = router_factory([payload]).route("q", context=ctx)
    async_decision = asyncio.run(router_factory([payload]).aroute("q", context=ctx))

    assert sync_decision.kind == async_decision.kind
    assert _volatile_audit_fields(sync_decision) == _volatile_audit_fields(async_decision)


def test_aroute_drives_a_sync_only_client_off_the_event_loop(
    small_registry: Registry, sink: InMemorySink
) -> None:
    """§2.5's mismatch rule, permissive direction: wrap it in ``to_thread``."""

    class SyncOnly:
        capabilities = GRAMMAR_LOGPROBS

        def __init__(self) -> None:
            self.inner = FakeLLMClient([REFUND])

        def complete(self, request: LLMRequest) -> Any:
            return self.inner.complete(request)

    router = Router(small_registry, client=SyncOnly(), sink=sink, otel=False)

    decision = asyncio.run(router.aroute("refund order A-123"))

    assert decision.kind == "route"


# --------------------------------------------------------------------------- #
# Construction-time failures (plan §2.4, §3.8)
# --------------------------------------------------------------------------- #


def test_unknown_fallback_route_is_refused(
    small_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """"A fallback must be a registered route" (plan §6.6)."""
    with pytest.raises(ConfigError, match="fallback"):
        Router(small_registry, client=fake_client, fallback="does_not_exist", otel=False)


def test_incoherent_thresholds_are_refused(
    small_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """Reordering the bands silently would abstain where the operator said clarify."""
    with pytest.raises(ConfigError, match="abstain_below"):
        Router(
            small_registry,
            client=fake_client,
            thresholds={"abstain_below": 0.9, "clarify_below": 0.2},
            otel=False,
        )


def test_k_below_one_is_refused(
    small_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """"K < 1" is on §3.8's ConfigError list."""
    with pytest.raises(ConfigError, match="shortlist_k"):
        Router(small_registry, client=fake_client, shortlist_k=0, otel=False)


def test_disabled_shortlist_over_max_candidates_is_refused(
    large_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """Silent truncation would drop routes, so it is refused (plan §3.3)."""
    with pytest.raises(ConfigError, match="max_candidates"):
        Router(large_registry, client=fake_client, shortlist=None, otel=False)


def test_a_sync_router_refuses_an_async_only_client(small_registry: Registry) -> None:
    """§2.5's mismatch rule, strict direction: sync ``route()`` + async-only client.

    The refusal lands on the ``route()`` call rather than on the constructor,
    because the same instance's ``aroute()`` is a perfectly valid use of an
    async-only client — refusing at construction would make the async path
    unreachable. The error type and the guidance are §3.8's.
    """

    class AsyncOnly:
        capabilities = ClientCapabilities()

        async def acomplete(self, request: LLMRequest) -> Any:
            return FakeLLMClient([REFUND]).complete(request)

    router = Router(small_registry, client=AsyncOnly(), otel=False)

    with pytest.raises(ConfigError, match="aroute"):
        router.route("refund order A-123")

    assert asyncio.run(router.aroute("refund order A-123")).kind == "route"


def test_an_unusable_client_is_refused_eagerly(small_registry: Registry) -> None:
    """"It cannot answer a routing call" (plan §4.1), refused at construction.

    Resolution is eager precisely so a typo'd spec or an object that is not a
    client fails now rather than on the first production request (plan §2.4).
    """

    class Useless:
        capabilities = ClientCapabilities()
        complete = "not callable"

    with pytest.raises(ConfigError, match="Cannot resolve client"):
        Router(small_registry, client=Useless(), otel=False)
    with pytest.raises(ConfigError, match="adapter:model"):
        Router(small_registry, client="not-a-spec", otel=False)


def test_a_registry_is_required(fake_client: FakeLLMClient) -> None:
    with pytest.raises(ConfigError, match="Registry"):
        Router([Route(name="a", description="d")], client=fake_client, otel=False)  # type: ignore[arg-type]


def test_deferred_configuration_is_refused_not_silently_downgraded(
    small_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """A promise the release cannot keep must raise (plan §13 ruling #9, §7.1)."""
    with pytest.raises(ConfigError, match=r"\[v0.2\]"):
        Router(small_registry, client=fake_client, per_tenant_index=True, otel=False)
    with pytest.raises(ConfigError, match=r"\[v0.2\]"):
        Router(small_registry, client=fake_client, default_visibility="deny", otel=False)
    with pytest.raises(ConfigError, match=r"\[v0.2\]"):
        Router(small_registry, client=fake_client, confidence="logprobs+vote:n=3", otel=False)


def test_an_oversized_candidate_budget_warns(
    small_registry: Registry, fake_client: FakeLLMClient
) -> None:
    """"Warn above 50 candidates" (plan §3.3), citing the degradation evidence."""
    with pytest.warns(UserWarning, match="max_candidates"):
        Router(small_registry, client=fake_client, max_candidates=80, otel=False)


# --------------------------------------------------------------------------- #
# The repair loop (plan §4.5) and transport failure (plan §3.8)
# --------------------------------------------------------------------------- #


def test_the_repair_loop_retries_exactly_twice_then_degrades(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``schema_attempts=2`` means at most three provider calls (plan §4.5).

    And then it **degrades** — schema exhaustion never raises, uniformly, whether
    or not a fallback is configured (§3.8, §13 ruling #2). A caller who wrapped
    ``route()`` in a try/except for this would be writing dead code.
    """
    router = router_factory(["not json", "still not json", "never json"])

    decision = router.route("something", context=ctx)

    assert router.client.calls == 3
    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "unparseable_output"
    assert decision.audit.validation_retries == 2
    assert decision.audit.error is None  # degraded, not errored


def test_the_repair_loop_stops_as_soon_as_it_succeeds(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    router = router_factory(["not json", REFUND, REFUND])

    decision = router.route("refund order A-123", context=ctx)

    assert router.client.calls == 2
    assert decision.kind == "route"
    assert decision.audit.validation_retries == 1


def test_schema_attempts_zero_makes_one_call(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """The budget is honoured exactly, including at its floor (plan §4.5)."""
    router = router_factory(
        ["not json"],
        retry={"schema_attempts": 0, "provider_attempts": 1, "backoff": "none"},
    )

    decision = router.route("q", context=ctx)

    assert router.client.calls == 1
    assert decision.kind == "abstain"


def test_provider_error_raises_by_default(
    router_factory: Callable[..., Router], sink: InMemorySink, ctx: RequestContext
) -> None:
    """"Provider transport failure after retries **raises** by default" (§3.8)."""
    router = router_factory([ProviderTimeout("upstream gone")])

    with pytest.raises(ProviderTimeout):
        router.route("q", context=ctx)

    # Retried up to provider_attempts=3 with the identical request.
    assert router.client.calls == 3


def test_provider_error_abstains_under_the_opt_in(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``on_provider_error="abstain"`` (plan §3.3, §3.8)."""
    router = router_factory(
        [ProviderTimeout("upstream gone")], on_provider_error="abstain"
    )

    decision = router.route("q", context=ctx)

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "provider_error"
    assert decision.audit.error == "ProviderTimeout"


def test_rate_limits_are_retried(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """``ProviderRateLimit`` is retryable and honours ``Retry-After`` (plan §3.8)."""
    router = router_factory([ProviderRateLimit("slow down", retry_after=0.0), REFUND])

    decision = router.route("refund order A-123", context=ctx)

    assert decision.kind == "route"
    assert router.client.calls == 2


def test_auth_errors_are_never_retried(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """"``ProviderAuthError`` is never retried" (plan §3.8).

    Retrying a rejected credential cannot succeed; it can only multiply the
    latency of a request that was doomed, and on some providers it counts toward
    a lockout.
    """
    router = router_factory([ProviderAuthError("bad key")])

    with pytest.raises(ProviderAuthError):
        router.route("q", context=ctx)

    assert router.client.calls == 1


def test_async_transport_ladder_matches_the_sync_one(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """The two drivers differ only at the await (plan §2.5)."""
    raising = router_factory([ProviderTimeout("gone")])
    with pytest.raises(ProviderError):
        asyncio.run(raising.aroute("q", context=ctx))
    assert raising.client.calls == 3

    never = router_factory([ProviderAuthError("bad key")])
    with pytest.raises(ProviderAuthError):
        asyncio.run(never.aroute("q", context=ctx))
    assert never.client.calls == 1


# --------------------------------------------------------------------------- #
# The audit tap (plan §2.1, §8.2, §8.3)
# --------------------------------------------------------------------------- #


def test_an_audit_record_is_emitted_on_the_success_path(
    router_factory: Callable[..., Router], sink: InMemorySink, ctx: RequestContext
) -> None:
    """Exactly one record, and it is the same object the caller holds (§8.2)."""
    router = router_factory([REFUND])

    decision = router.route("refund order A-123", context=ctx)

    assert len(sink.records) == 1
    assert sink.latest is decision.audit
    assert sink.latest.tenant_id == "acme"
    assert sink.latest.trace_id == "trace-abc"
    assert sink.latest.user_id_hash is not None
    assert sink.latest.user_id_hash != "user-42"


def test_an_audit_record_is_emitted_on_the_raise_path(
    router_factory: Callable[..., Router], sink: InMemorySink, ctx: RequestContext
) -> None:
    """THE ``finally`` RULE (plan §2.1).

    "The audit emitter is a tap, not a stage... emitted exactly once — in a
    ``finally``, so raised provider errors still produce a span with
    ``error.type`` set." A decision that failed is the one you most want a record
    of, and it is the one a naive implementation loses.
    """
    router = router_factory([ProviderTimeout("upstream gone")])

    with pytest.raises(ProviderTimeout):
        router.route("refund order A-123", context=ctx)

    assert len(sink.records) == 1
    record = sink.latest
    assert record is not None
    assert record.error == "ProviderTimeout"
    assert record.registry_version == router.registry.version
    assert record.tenant_id == "acme"
    # The span attribute projection carries it through as error.type (§8.1).
    assert record.as_otel_attributes()["error.type"] == "ProviderTimeout"


def test_the_raise_path_audit_is_emitted_from_aroute_too(
    router_factory: Callable[..., Router], sink: InMemorySink, ctx: RequestContext
) -> None:
    router = router_factory([ProviderTimeout("upstream gone")])

    with pytest.raises(ProviderTimeout):
        asyncio.run(router.aroute("q", context=ctx))

    assert len(sink.records) == 1
    assert sink.latest is not None
    assert sink.latest.error == "ProviderTimeout"


def test_the_audit_record_captures_the_retrieval_trace(
    router_factory: Callable[..., Router], large_registry: Registry, ctx: RequestContext
) -> None:
    """Seed + pre-shuffle ranks are what make a prompt reconstructible (§5.6, §8.2)."""
    router = router_factory(
        [wire_output(None, kind="abstain", reason="model_elected")],
        registry=large_registry,
    )

    decision = router.route("cancel the billing entry in workspace 0", context=ctx)
    record = decision.audit

    assert record.candidates_total == len(large_registry)
    assert record.candidates_entitled == len(large_registry)
    assert record.shortlist_skipped is False
    assert record.shuffle_seed is not None
    assert len(record.shortlist) > 0
    assert sorted(candidate.rank for candidate in record.shortlist) == list(
        range(len(record.shortlist))
    )


def test_a_broken_sink_never_breaks_routing(
    small_registry: Registry, ctx: RequestContext
) -> None:
    """"``route()`` never blocks or fails on sink I/O" (plan §8.3)."""

    class ExplodingSink:
        def emit(self, record: Any) -> None:
            raise RuntimeError("the warehouse is on fire")

        async def aemit(self, record: Any) -> None:
            raise RuntimeError("the warehouse is on fire")

        def flush(self) -> None: ...

        def close(self) -> None: ...

    router = Router(
        small_registry, client=FakeLLMClient([REFUND]), sink=ExplodingSink(), otel=False
    )

    assert router.route("refund order A-123", context=ctx).kind == "route"


# --------------------------------------------------------------------------- #
# Content modes and redaction (plan §7.4)
# --------------------------------------------------------------------------- #


def test_content_mode_none_stores_only_hashes(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """The default (plan §7.4, Appendix A): ``sha256(query)`` and length only.

    Everything that is *not* payload text — signals, candidates, the chosen
    route, tenant, registry version, latency — is captured in every mode; the
    hashes keep records joinable and dedupable without any text at all.
    """
    router = router_factory([REFUND], content_mode="none")

    record = router.route("refund order A-123 for jane@example.com", context=ctx).audit

    assert record.input_text is None
    assert record.args is None
    assert record.rationale is None
    assert record.inputs_hash
    assert record.args_hash
    assert record.args_schema_fingerprint
    assert record.routes == ["refund"]
    assert record.confidence is not None


def test_content_mode_full_stores_the_text(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """"``full`` — payload text verbatim" (plan §7.4)."""
    router = router_factory([REFUND], content_mode="full")

    record = router.route("refund order A-123", context=ctx).audit

    assert record.input_text == "refund order A-123"
    assert record.args == {"order_id": "A-123", "reason": "damaged"}
    assert record.rationale


def test_the_redactor_is_one_hook_at_every_egress_site(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """ONE HOOK, THREE SITES (plan §7.4).

    "No path where raw text leaks because a second hook was forgotten." The two
    observable sites here are the prompt the provider receives and the audit
    record the sink receives; the third — the shortlist/embedding input — is the
    same redacted string, which the seed and the candidate block are derived
    from.
    """
    calls: list[str] = []

    def redactor(text: str, context: RequestContext) -> str:
        calls.append(text)
        assert context.tenant_id == "acme"
        return text.replace("jane@example.com", "[EMAIL]")

    router = router_factory([REFUND], redactor=redactor, content_mode="redacted")

    decision = router.route(
        "refund order A-123 for jane@example.com", context=ctx
    )

    prompt = router.client.last_prompt
    assert "jane@example.com" not in prompt
    assert "[EMAIL]" in prompt
    assert decision.audit.input_text == "refund order A-123 for [EMAIL]"
    assert calls  # the hook actually ran


def test_redacted_mode_without_a_redactor_fails_closed(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """A PII control degrades toward *less* capture, never more (plan §7.4)."""
    router = router_factory([REFUND], content_mode="redacted")

    record = router.route("refund order A-123", context=ctx).audit

    assert record.input_text is None
    assert record.inputs_hash


def test_a_raising_redactor_is_a_config_error(
    router_factory: Callable[..., Router], ctx: RequestContext
) -> None:
    """Failing open would send the raw query to the provider (plan §3.8, §7.4)."""

    def broken(text: str, context: RequestContext) -> str:
        raise ValueError("regex exploded")

    router = router_factory([REFUND], redactor=broken)

    with pytest.raises(ConfigError, match="redactor"):
        router.route("refund order A-123", context=ctx)


# --------------------------------------------------------------------------- #
# Thread/task safety (plan §2.5)
# --------------------------------------------------------------------------- #


def _dispatch(request: LLMRequest) -> dict[str, Any]:
    """Answer according to the query in the last segment — no per-call state."""
    text = request.segments[-1].content
    if "refund" in text:
        order = text.split("order ")[-1].split()[0].strip()
        return wire_output("refund", {"order_id": order})
    if "parcel" in text:
        return wire_output("track_order", {"order_id": "unknown"})
    return wire_output(None, kind="abstain", reason="model_elected")


def test_one_router_serves_concurrent_tasks_without_cross_talk(
    small_registry: Registry, sink: InMemorySink
) -> None:
    """"One instance is safe across threads and tasks" (plan §2.5).

    The Router holds no per-request mutable state: the resolved configuration is
    a frozen ``LoopConfig``, the registry is frozen, and everything else lives in
    a per-call ``LoopState``. Twenty interleaved tasks with three different
    intents and twenty different tenants must each get their own answer and their
    own audit record.
    """
    router = Router(
        small_registry,
        client=FakeLLMClient(default=_dispatch),
        sink=sink,
        otel=False,
    )
    queries = (
        [f"refund order A-{index}" for index in range(10)]
        + ["where is my parcel"] * 5
        + ["what is the weather"] * 5
    )

    async def run_all() -> list[Decision]:
        return await asyncio.gather(
            *(
                router.aroute(query, context=RequestContext(tenant_id=f"tenant-{index}"))
                for index, query in enumerate(queries)
            )
        )

    decisions = asyncio.run(run_all())

    assert len(sink.records) == len(queries)
    for index, (query, decision) in enumerate(zip(queries, decisions, strict=True)):
        assert decision.audit.tenant_id == f"tenant-{index}"
        if query.startswith("refund"):
            assert isinstance(decision, RouteDecision)
            assert decision.route == "refund"
            assert decision.args is not None
            assert decision.args.order_id == f"A-{index}"
        elif query == "where is my parcel":
            assert decision.kind == "route"
            assert decision.route == "track_order"  # type: ignore[union-attr]
        else:
            assert decision.kind == "abstain"


def test_warm_is_idempotent_and_optional(
    router_factory: Callable[..., Router], large_registry: Registry
) -> None:
    """Eager index build moves cost off the first user's latency budget (§5.4)."""
    router = router_factory([REFUND], registry=large_registry)

    router.warm()
    router.warm()
    asyncio.run(router.awarm())

    assert router.shortlister is not None
    assert router.shortlister.registry_version == large_registry.version  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Entitlement is a pre-LLM boundary (plan §7.1, §7.4)
# --------------------------------------------------------------------------- #


def test_an_unentitled_route_never_enters_the_prompt(
    router_factory: Callable[..., Router]
) -> None:
    """"Unauthorized names never enter the prompt" (plan §7.4).

    They cannot then be leaked, chosen, or hallucinated into reach. The same
    request *with* the entitlement sees the route, which is what makes this a
    filter rather than a permanently missing route.
    """
    without = router_factory([REFUND])
    without.route("billing please", context=RequestContext(tenant_id="acme"))
    assert "billing_report" not in without.client.last_prompt

    with_entitlement = router_factory([REFUND])
    with_entitlement.route(
        "billing please",
        context=RequestContext(tenant_id="acme", entitlements=frozenset({"billing"})),
    )
    assert "billing_report" in with_entitlement.client.last_prompt


def test_a_fallback_the_tenant_cannot_see_leaves_the_abstain_standing(
    sink: InMemorySink,
) -> None:
    """"Entitlement is never bypassed" (plan §6.6)."""
    registry = Registry(
        [
            Route(name="refund", description="Issue or check a refund for an order."),
            Route(
                name="human_handoff",
                description="Escalate to a human support agent.",
                pinned=True,
                requires=frozenset({"support"}),
            ),
        ]
    )
    router = Router(
        registry,
        client=FakeLLMClient(["not json", "still not json", "never json"]),
        fallback="human_handoff",
        sink=sink,
        otel=False,
    )

    decision = router.route("gibberish", context=RequestContext(tenant_id="acme"))

    assert isinstance(decision, AbstainDecision)
    assert decision.reason == "unparseable_output"


# --------------------------------------------------------------------------- #
# Introspection surface (plan §3.6)
# --------------------------------------------------------------------------- #


def test_deferred_call_surfaces_raise_rather_than_pretending(
    router_factory: Callable[..., Router]
) -> None:
    """Streaming is [v0.2]; a silent fallback to ``route()`` would stream nothing
    and explain nothing (plan §13 ruling #10)."""
    router = router_factory([REFUND])

    with pytest.raises(ConfigError, match=r"\[v0.2\]"):
        router.stream_route("q")
    with pytest.raises(ConfigError, match=r"\[v0.2\]"):
        asyncio.run(router.astream_route("q"))


def test_the_router_exposes_its_wiring(
    router_factory: Callable[..., Router], small_registry: Registry, sink: InMemorySink
) -> None:
    """Read-only introspection, so a test or a dashboard needs no private access."""
    router = router_factory([REFUND], capabilities=NO_SIGNAL)

    assert router.registry is small_registry
    assert router.sink is sink
    assert router.config.registry is small_registry
    assert router.config.capabilities == NO_SIGNAL
    assert "Router(routes=5" in repr(router)
