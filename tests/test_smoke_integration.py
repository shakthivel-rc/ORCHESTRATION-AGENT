"""Independent end-to-end smoke checks written outside the implementation agents.

These are deliberately *not* unit tests of any one module: each one drives the real
public API the way a user would, and asserts a guarantee the plan makes by name.
They exist to catch integration drift that per-module suites can miss.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import switchboard as sb
from switchboard import (
    ConfigError,
    Registry,
    RequestContext,
    Route,
    Router,
)
from switchboard.telemetry.emitter import InMemorySink

REPO = Path(__file__).resolve().parents[1]


class RefundArgs(BaseModel):
    order_id: str
    reason: str | None = None


def _catalog(n_extra: int = 0) -> Registry:
    routes = [
        Route(
            name="refund",
            description="Issue or check a refund for an order.",
            args_model=RefundArgs,
            examples=("I want my money back for order 123",),
            tags=frozenset({"billing"}),
        ),
        Route(
            name="track_order",
            description="Track shipment status for an existing order.",
            examples=("where is my package",),
        ),
        Route(
            name="human_handoff",
            description="Escalate to a human support agent.",
            pinned=True,
        ),
    ]
    routes += [
        Route(name=f"filler_{i:03d}", description=f"Handle filler topic number {i}.")
        for i in range(n_extra)
    ]
    return Registry(routes)


def _stub(payload: dict[str, Any]):
    """A BYO plain callable: the zero-extras client path (plan §4.1)."""

    def client(prompt: str) -> dict[str, Any]:
        return payload

    return client


# --------------------------------------------------------------------------- #
# 1. The quickstart contract
# --------------------------------------------------------------------------- #


def test_quickstart_shape_end_to_end() -> None:
    """plan §3.7: 10-line quickstart returns a typed RouteDecision with parsed args."""
    router = Router(
        registry=_catalog(),
        client=_stub(
            {
                "rationale": "The user asks about a package for a known order.",
                "kind": "route",
                "route": "refund",
                "args": {"order_id": "123", "reason": "damaged"},
            }
        ),
        shortlist="auto",
        allow_clarify=True,
        fallback="human_handoff",
    )
    d = router.route("refund my order 123", context=RequestContext(tenant_id="acme"))

    assert d.kind == "route"
    assert d.route == "refund"
    assert isinstance(d.args, RefundArgs)  # validated instance, not a dict
    assert d.args.order_id == "123"
    assert d.decision_path == "llm"
    assert d.audit.registry_version == router.registry.version
    assert d.audit.decision_id  # ULID present


def test_public_api_surface_is_importable() -> None:
    """plan §2.3: everything the docs reference is reachable from the top level."""
    for name in (
        "Route",
        "Registry",
        "Router",
        "RequestContext",
        "RouteDecision",
        "ClarifyDecision",
        "AbstainDecision",
        "MultiRouteDecision",
        "ConfidenceReport",
        "AuditRecord",
        "LLMClient",
        "ConfigError",
    ):
        assert hasattr(sb, name), f"{name} missing from switchboard.__all__ surface"


# --------------------------------------------------------------------------- #
# 2. All four decision kinds through one call surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "expected_kind"),
    [
        (
            {
                "rationale": "clear refund request",
                "kind": "route",
                "route": "refund",
                "args": {"order_id": "1"},
            },
            "route",
        ),
        (
            {
                "rationale": "ambiguous",
                "kind": "clarify",
                "question": "Which order?",
                "candidates": ["refund", "track_order"],
                "missing": ["order_id"],
            },
            "clarify",
        ),
    ],
)
def test_decision_kinds_round_trip(payload: dict[str, Any], expected_kind: str) -> None:
    router = Router(registry=_catalog(), client=_stub(payload), allow_clarify=True)
    d = router.route("something")
    assert d.kind == expected_kind
    # every kind carries the full contract
    assert d.rationale
    assert d.confidence is not None
    assert d.audit is not None


def test_abstain_with_fallback_becomes_a_route_decision() -> None:
    """plan §13 ruling #5: fallback yields kind='route', decision_path='fallback'."""
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "no match", "kind": "abstain", "reason": "model_elected"}),
        fallback="human_handoff",
    )
    d = router.route("what is the capital of France")
    assert d.kind == "route"
    assert d.route == "human_handoff"
    assert d.decision_path == "fallback"
    assert d.args is None
    # the pre-fallback truth survives in the audit
    assert d.audit.abstain_reason == "model_elected"


def test_abstain_without_fallback_stays_abstain() -> None:
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "no match", "kind": "abstain", "reason": "model_elected"}),
        fallback=None,
    )
    d = router.route("what is the capital of France")
    assert d.kind == "abstain"
    assert d.reason == "model_elected"


# --------------------------------------------------------------------------- #
# 3. Degradation, not explosion (plan §3.8)
# --------------------------------------------------------------------------- #


def test_garbage_output_degrades_to_abstain_without_raising() -> None:
    """plan §13 ruling #2: schema exhaustion degrades; it never raises."""
    calls: list[int] = []

    def broken(prompt: str) -> str:
        calls.append(1)
        return "I am not JSON at all."

    router = Router(registry=_catalog(), client=broken, fallback=None)
    d = router.route("refund please")

    assert d.kind == "abstain"
    assert d.reason in {"unparseable_output", "invalid_route_reference"}
    # 1 initial attempt + 2 repair retries (plan Appendix A: schema_attempts=2)
    assert len(calls) == 3, f"expected 3 provider calls, got {len(calls)}"
    assert d.audit.validation_retries == 2


def test_hallucinated_route_name_is_rejected() -> None:
    """plan §4.5: route must be in the candidate set even under a Literal enum."""
    router = Router(
        registry=_catalog(),
        client=_stub(
            {"rationale": "made this up", "kind": "route", "route": "definitely_not_a_route"}
        ),
        fallback=None,
    )
    d = router.route("do the thing")
    assert d.kind == "abstain"
    assert d.reason in {"invalid_route_reference", "unparseable_output"}


def test_unknown_fallback_route_fails_fast_at_construction() -> None:
    """plan §3.3: eager validation — config errors surface before any request."""
    with pytest.raises(ConfigError):
        Router(registry=_catalog(), client=_stub({}), fallback="no_such_route")


def test_missing_extra_names_the_extra() -> None:
    """plan §2.4: never a bare ImportError."""
    with pytest.raises(ConfigError) as exc:
        Router(registry=_catalog(), client="instructor:openai/gpt-5-nano")
    assert "instructor" in str(exc.value)


# --------------------------------------------------------------------------- #
# 4. Retrieve-then-decide (plan §5)
# --------------------------------------------------------------------------- #


def test_shortlist_bypasses_below_threshold_and_engages_above() -> None:
    payload = {"rationale": "r", "kind": "route", "route": "track_order"}

    small = Router(registry=_catalog(), client=_stub(payload), shortlist="auto")
    d_small = small.route("where is my package")
    assert d_small.audit.shortlist_skipped is True, "auto must bypass below 25 routes"

    big = Router(registry=_catalog(n_extra=200), client=_stub(payload), shortlist="auto")
    d_big = big.route("where is my package")
    assert d_big.audit.shortlist_skipped is False, "auto must engage at 203 routes"
    # K band for 150 <= N < 1000 is 15, plus the pinned route
    assert 5 <= len(d_big.audit.shortlist) <= 21
    names = {c.route_name for c in d_big.audit.shortlist}
    assert "track_order" in names, "gold route must survive retrieval"
    assert "human_handoff" in names, "pinned routes are always candidates"


def test_shortlist_seed_is_stable_across_processes() -> None:
    """plan §5.6: the seed is a sha256 digest, NOT Python's randomized hash()."""
    script = (
        "import sys; sys.path.insert(0, 'src');"
        "from switchboard.engine.prompt import order_candidates;"
        "from switchboard.core.candidates import Candidate;"
        "cands=[Candidate(route_name=n, score=1.0, rank=i, source='bm25')"
        " for i,n in enumerate(['a','b','c','d','e'])];"
        "ordered, seed = order_candidates(cands, 'shuffle', 'the query', 'ver123');"
        "print(seed, [c.route_name for c in ordered])"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
        )
        for seed in ("0", "1", "12345")
    ]
    outs = {r.stdout.strip() for r in runs}
    assert all(r.returncode == 0 for r in runs), [r.stderr for r in runs]
    assert len(outs) == 1, f"seed/order varied under PYTHONHASHSEED: {outs}"


def test_entitlement_filtering_happens_before_the_prompt() -> None:
    """plan §7.4: an unentitled route never enters the candidate set."""
    reg = Registry(
        [
            Route(name="public_route", description="Anyone can use this one."),
            Route(
                name="secret_route",
                description="Only billing admins may use this one.",
                requires=frozenset({"billing_admin"}),
            ),
        ]
    )
    seen: list[str] = []

    def spy(prompt: str) -> dict[str, Any]:
        seen.append(prompt)
        return {"rationale": "r", "kind": "route", "route": "public_route"}

    router = Router(registry=reg, client=spy, shortlist="auto")
    router.route("do something", context=RequestContext(tenant_id="t", entitlements=frozenset()))

    assert "secret_route" not in seen[0], "unentitled route leaked into the prompt"
    assert "public_route" in seen[0]


# --------------------------------------------------------------------------- #
# 5. Confidence honesty (plan §6.3)
# --------------------------------------------------------------------------- #


def test_no_signal_rule_keeps_thresholds_inert() -> None:
    """A bare callable yields no logprobs -> thresholds must NOT downgrade."""
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "r", "kind": "route", "route": "refund", "args": {"order_id": "9"}}),
        allow_clarify=True,
        fallback=None,
    )
    d = router.route("refund order 9")
    assert d.kind == "route", "verbalized/absent signal must not trigger a downgrade"
    assert d.confidence.method in {"none", "verbalized"}
    assert d.downgraded_from is None


# --------------------------------------------------------------------------- #
# 6. Audit / observability (plan §8)
# --------------------------------------------------------------------------- #


def test_audit_record_is_emitted_and_json_serialisable() -> None:
    sink = InMemorySink()
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "r", "kind": "route", "route": "track_order"}),
        sink=sink,
    )
    router.route("where is my package", context=RequestContext(tenant_id="acme"))

    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.kind == "route"
    assert rec.routes == ["track_order"]
    # round-trips as JSON: it is the compliance row and the distillation example
    blob = json.loads(rec.model_dump_json())
    assert blob["schema_version"] == "1"
    assert blob["registry_version"] == router.registry.version


def test_content_mode_none_stores_hashes_not_text() -> None:
    """plan §7.4: default content_mode='none' keeps payload text out of the record."""
    sink = InMemorySink()
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "r", "kind": "route", "route": "track_order"}),
        sink=sink,
        content_mode="none",
    )
    secret = "my card number is 4111111111111111"
    router.route(secret)

    rec = sink.records[0]
    assert rec.input_text is None
    assert rec.inputs_hash
    assert secret not in rec.model_dump_json()


def test_otel_attributes_use_only_sanctioned_namespaces() -> None:
    """plan §8.1: only gen_ai.* / switchboard.* — no invented pseudo-standard keys."""
    sink = InMemorySink()
    router = Router(
        registry=_catalog(),
        client=_stub({"rationale": "r", "kind": "route", "route": "track_order"}),
        sink=sink,
    )
    router.route("where is my package")

    attrs = sink.records[0].as_otel_attributes()
    assert attrs, "no attributes emitted"
    for key in attrs:
        assert key.startswith(("gen_ai.", "switchboard.", "error.")), key


# --------------------------------------------------------------------------- #
# 7. Sync/async parity and concurrency (plan §2.5)
# --------------------------------------------------------------------------- #


def test_sync_and_async_agree() -> None:
    payload = {"rationale": "r", "kind": "route", "route": "refund", "args": {"order_id": "77"}}
    router = Router(registry=_catalog(), client=_stub(payload))

    sync_d = router.route("refund 77")
    async_d = asyncio.run(router.aroute("refund 77"))

    assert sync_d.kind == async_d.kind
    assert sync_d.route == async_d.route
    assert sync_d.args == async_d.args


def test_one_router_serves_concurrent_tasks_without_crosstalk() -> None:
    """plan §2.5: no per-request mutable state on the Router."""

    def by_query(request: Any) -> dict[str, Any]:
        # Segment D (last) is the user query; earlier segments carry the route
        # directory, which mentions every route name (plan §4.6 layout).
        query_segment = request.segments[-1].content
        route = "refund" if "refund" in query_segment else "track_order"
        args = {"order_id": "1"} if route == "refund" else None
        return {"rationale": "r", "kind": "route", "route": route, "args": args}

    router = Router(registry=_catalog(), client=by_query)

    async def main() -> list[str]:
        queries = ["refund me"] * 20 + ["where is my package"] * 20
        out = await asyncio.gather(*(router.aroute(q) for q in queries))
        return [d.route for d in out]

    routes = asyncio.run(main())
    assert routes[:20] == ["refund"] * 20
    assert routes[20:] == ["track_order"] * 20


# --------------------------------------------------------------------------- #
# 8. The zero-dependency guarantee (plan §2.4)
# --------------------------------------------------------------------------- #


def test_importing_switchboard_pulls_in_no_optional_sdk() -> None:
    script = (
        "import sys; sys.path.insert(0, 'src');"
        "import switchboard, switchboard.evals;"
        "banned={'openai','litellm','instructor','anthropic','boto3',"
        "'opentelemetry','numpy','fastembed','yaml','rich','sklearn','torch'};"
        "leaked=sorted(banned & {m.split('.')[0] for m in sys.modules});"
        "print(leaked)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", f"optional deps leaked into core: {proc.stdout}"
