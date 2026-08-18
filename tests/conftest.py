"""Shared test fixtures for the whole suite (plan §9.1, §4.1).

Everything a test needs to drive a real ``Router`` **without a network call**
lives here, so no test file has to hand-roll a client or a catalog and the suite
cannot drift into testing three different fakes.

Contents:

* :class:`FakeLLMClient` — a scripted :class:`~switchboard.LLMClient` /
  :class:`~switchboard.AsyncLLMClient` that records every
  :class:`~switchboard.LLMRequest` it receives, can be handed dicts, raw strings,
  ``LLMResult`` s or exceptions, and synthesises token logprobs so the confidence
  ladder (§6.2) is exercisable offline.
* :func:`wire_output` — build a well-formed wire-schema payload for a route.
* ``small_registry`` — the 5-route catalog from the §3.7 quickstart shape:
  one pinned ``human_handoff``, one route with an ``args_model``, one with
  ``requires``.
* ``large_registry`` — a generated 120-route catalog, the size plan §9.6's
  dogfood suite runs against, so shortlist and bypass behaviour are testable at
  the scale the evidence is about.
* ``ctx`` — a populated :class:`~switchboard.RequestContext`.
* ``router_factory`` — builds a ``Router`` wired to a ``FakeLLMClient``, with
  every ``Router(...)`` knob overridable per test.

``asyncio_mode = "auto"`` is set in ``pyproject.toml``, so a plain
``async def test_...`` runs without a marker.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from switchboard import (
    ClientCapabilities,
    InMemorySink,
    LLMRequest,
    LLMResult,
    Registry,
    RequestContext,
    Route,
    Router,
    TokenLP,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

__all__ = [
    "GRAMMAR_LOGPROBS",
    "NO_SIGNAL",
    "TOOL_STRICT_NO_LOGPROBS",
    "FakeLLMClient",
    "wire_output",
]


# --------------------------------------------------------------------------- #
# Capability presets — the rungs of the plan §4.3 ladder, named.
# --------------------------------------------------------------------------- #

GRAMMAR_LOGPROBS = ClientCapabilities(
    structured="grammar", logprobs=True, caching="implicit", reasoning_toggle=True
)
"""The strongest rung: OpenAI strict / Gemini responseSchema. ``wire_schema="auto"``
resolves to ``dynamic`` here, and the confidence thresholds are live because
``p_route`` is computable (plan §4.4, §6.3)."""

TOOL_STRICT_NO_LOGPROBS = ClientCapabilities(
    structured="tool_strict", logprobs=False, caching="explicit", reasoning_toggle=True
)
"""The Anthropic shape: forced tool use, explicit caching, **no logprobs**.
``wire_schema="auto"`` resolves to ``static`` (a per-query enum would void the
A+B cache, §4.4) and the threshold machinery is inert (§6.3)."""

NO_SIGNAL = ClientCapabilities()
"""All-conservative defaults — what a bare BYO callable reports (plan §4.1).
Arms the full validate-and-retry loop and leaves confidence inert."""


# --------------------------------------------------------------------------- #
# Wire-output helper (plan §4.4 field order)
# --------------------------------------------------------------------------- #


def wire_output(
    route: str | None = None,
    args: Mapping[str, Any] | None = None,
    *,
    kind: str = "route",
    rationale: str = "The request names an order and asks about it.",
    routes: Sequence[Mapping[str, Any]] | None = None,
    question: str | None = None,
    candidates: Sequence[str] | None = None,
    missing: Sequence[str] | None = None,
    reason: str | None = None,
    stated_confidence: float | None = None,
) -> dict[str, Any]:
    """Build a well-formed wire-schema payload (plan §4.4).

    Key order matches the generated schema's field order — ``rationale`` first,
    then ``kind``, then ``route``/``args`` — which matters for more than looks:
    :func:`FakeLLMClient` serialises this dict to JSON and derives token logprobs
    from the resulting text, and ``engine/confidence.py`` anchors its token
    alignment on the ``"route"`` key appearing *after* the rationale (§6.2). A
    reordered payload would silently stop exercising that path.

    Args:
        route: the committed route name (``kind="route"``).
        args: extracted arguments for that route.
        kind: ``"route"``, ``"multi_route"``, ``"clarify"`` or ``"abstain"``.
        rationale: free-text reasoning, emitted before the commitment.
        routes: ``[{"route": ..., "args": ...}, ...]`` for ``kind="multi_route"``.
        question / candidates / missing: the clarify arm.
        reason: the abstain arm; the wire enum admits only ``"model_elected"``.
        stated_confidence: verbalized score, present only on the verbalized rung.
    """
    payload: dict[str, Any] = {"rationale": rationale, "kind": kind}
    payload["route"] = route
    payload["args"] = dict(args) if args is not None else None
    if routes is not None:
        payload["routes"] = [dict(item) for item in routes]
    if question is not None or candidates is not None or missing is not None:
        payload["question"] = question
        payload["candidates"] = list(candidates) if candidates is not None else None
        payload["missing"] = list(missing) if missing is not None else None
    payload["reason"] = reason if reason is not None else ("model_elected" if kind == "abstain" else None)
    if stated_confidence is not None:
        payload["stated_confidence"] = stated_confidence
    return payload


# --------------------------------------------------------------------------- #
# FakeLLMClient
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9_]+|.", re.DOTALL)
"""Splits text into whitespace runs, identifier runs and single characters. The
concatenation is exactly the input, which is the only property the token
alignment in ``engine/confidence.py`` depends on (§6.2)."""


class FakeLLMClient:
    """A scripted, offline provider client implementing both provider Protocols.

    Satisfies :class:`~switchboard.LLMClient` **and**
    :class:`~switchboard.AsyncLLMClient`, so one instance drives ``route()`` and
    ``aroute()`` alike and a test can assert sync/async parity by running the same
    script through both (plan §2.5).

    The script is consumed one entry per provider call, so the §4.5 repair loop is
    driven by simply listing what the model "says" on each attempt::

        client = FakeLLMClient([
            "not json at all",                       # attempt 1: schema failure
            wire_output("refund", {"order_id": "1"}),  # attempt 2: repaired
        ])

    Accepted script entries:

    ==========================  ================================================
    Entry                       Behaviour
    ==========================  ================================================
    ``dict``                    serialised to JSON as ``raw_text``; ``parsed`` is
                                left ``None`` so the real
                                ``engine/validate.py`` parse path runs
    ``str``                     used verbatim as ``raw_text``
    ``BaseException``           raised (use ``ProviderTimeout()`` etc. to drive
                                the §3.8 transport-retry ladder)
    :class:`~switchboard.LLMResult`  returned unchanged
    ``callable``                called with the ``LLMRequest``; its return value
                                is then coerced by these same rules
    ==========================  ================================================

    When the script runs out, ``default`` is returned forever (``None`` means
    "repeat the last entry"), so a test that only cares about one response does
    not have to predict how many attempts the loop will make.

    Args:
        responses: the script. A single non-list entry is wrapped for you.
        capabilities: what the client claims it can do. Defaults to
            :data:`GRAMMAR_LOGPROBS`; pass :data:`NO_SIGNAL` to exercise the
            no-signal rule (§6.3) or :data:`TOOL_STRICT_NO_LOGPROBS` for the
            static-wire-schema path (§4.4).
        model_id: reported in ``LLMResult.model_id`` and the audit record.
        provider: reported as ``gen_ai.provider.name`` (§8.1).
        route_prob: per-token probability assigned to the committed route's
            tokens when synthesising logprobs — this is what ``p_route`` comes
            out as (§6.2), so ``route_prob=0.2`` reliably lands a decision in the
            abstain band and ``0.5`` in the clarify band under the default
            thresholds.
        other_prob: per-token probability for every other token.
        logprobs: force logprob synthesis on/off independently of
            ``capabilities.logprobs``. ``None`` follows the capability.
        usage: token usage reported on every call.

    Attributes:
        requests: every :class:`~switchboard.LLMRequest` received, in order — the
            repair loop's growing segment list is visible here.
        calls: number of provider calls made.
    """

    def __init__(
        self,
        responses: Any = None,
        *,
        capabilities: ClientCapabilities | None = None,
        model_id: str = "fake/router-model",
        provider: str = "fake",
        route_prob: float = 0.95,
        other_prob: float = 0.98,
        logprobs: bool | None = None,
        usage: Usage | None = None,
        default: Any = None,
    ) -> None:
        if responses is None:
            script: list[Any] = []
        elif isinstance(responses, (list, tuple)):
            script = list(responses)
        else:
            script = [responses]

        self.script = script
        self.capabilities = capabilities if capabilities is not None else GRAMMAR_LOGPROBS
        self.model_id = model_id
        self.provider = provider
        self.route_prob = route_prob
        self.other_prob = other_prob
        self._logprobs = logprobs
        self._usage = usage if usage is not None else Usage(
            input_tokens=800, cached_input_tokens=0, output_tokens=60
        )
        self.default = default

        self.requests: list[LLMRequest] = []
        self.calls = 0

    # -- introspection ------------------------------------------------------- #

    @property
    def emits_logprobs(self) -> bool:
        """Whether this client synthesises ``token_logprobs`` (plan §6.1)."""
        return self.capabilities.logprobs if self._logprobs is None else self._logprobs

    @property
    def last_request(self) -> LLMRequest | None:
        """The most recent request, or ``None``."""
        return self.requests[-1] if self.requests else None

    @property
    def last_prompt(self) -> str:
        """The most recent request's segments flattened, for substring asserts."""
        request = self.last_request
        return "\n\n".join(s.content for s in request.segments) if request else ""

    def push(self, *responses: Any) -> FakeLLMClient:
        """Append entries to the script. Returns ``self`` so it chains."""
        self.script.extend(responses)
        return self

    def reset(self) -> None:
        """Forget recorded requests and rewind the call counter."""
        self.requests.clear()
        self.calls = 0

    # -- LLMClient / AsyncLLMClient ------------------------------------------ #

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """Answer one routing call synchronously (plan §4.1)."""
        return self._respond(request)

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """Answer one routing call asynchronously — identical script, no thread hop."""
        return self._respond(request)

    # -- internals ----------------------------------------------------------- #

    def _respond(self, request: LLMRequest) -> LLMResult[Any]:
        self.requests.append(request)
        index = self.calls
        self.calls += 1

        if index < len(self.script):
            entry = self.script[index]
        elif self.default is not None:
            entry = self.default
        elif self.script:
            entry = self.script[-1]
        else:
            entry = wire_output(
                route=None,
                kind="abstain",
                reason="model_elected",
                rationale="__unscripted__: FakeLLMClient ran out of script",
            )

        return self._coerce(entry, request)

    def _coerce(self, entry: Any, request: LLMRequest) -> LLMResult[Any]:
        if callable(entry) and not isinstance(entry, type):
            entry = entry(request)
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, type) and issubclass(entry, BaseException):
            raise entry()
        if isinstance(entry, LLMResult):
            return entry
        if isinstance(entry, BaseModel):
            entry = entry.model_dump(mode="json")

        if isinstance(entry, dict):
            raw_text = json.dumps(entry, ensure_ascii=False)
            route_name = entry.get("route")
        else:
            raw_text = str(entry)
            route_name = None

        return LLMResult(
            parsed=None,  # the real engine/validate.py parse path stays exercised
            raw_text=raw_text,
            token_logprobs=self._synthesise_logprobs(raw_text, route_name),
            usage=self._usage,
            model_id=self.model_id,
            provider_meta={"response_id": f"fake-{self.calls:04d}", "provider": self.provider},
        )

    def _synthesise_logprobs(
        self, raw_text: str, route_name: str | None
    ) -> list[TokenLP] | None:
        """Build a token stream whose concatenation is exactly ``raw_text``.

        Tokens overlapping the committed route's **value** — located the same way
        ``engine/confidence.py`` locates it, anchored on the ``"route"`` key — get
        ``log(route_prob)``; everything else gets ``log(other_prob)``. Because
        ``p_route`` is the geometric mean over exactly those tokens, the resulting
        ``confidence.score`` is ``route_prob`` to within floating-point noise,
        which is what makes threshold tests readable (§6.2, §6.4).
        """
        if not self.emits_logprobs:
            return None

        span: tuple[int, int] | None = None
        if route_name:
            needle = f'"{route_name}"'
            anchor = raw_text.rfind('"route"')
            found = raw_text.find(needle, anchor + 7) if anchor != -1 else raw_text.rfind(needle)
            if found != -1:
                span = (found + 1, found + 1 + len(route_name))

        route_lp = math.log(max(self.route_prob, 1e-9))
        other_lp = math.log(max(self.other_prob, 1e-9))

        tokens: list[TokenLP] = []
        cursor = 0
        for match in _TOKEN_RE.finditer(raw_text):
            text = match.group(0)
            start, end = cursor, cursor + len(text)
            cursor = end
            in_route = span is not None and end > span[0] and start < span[1]
            logprob = route_lp if in_route else other_lp
            tokens.append(
                TokenLP(token=text, logprob=logprob, top=[(text, logprob)])
            )
        return tokens

    def __repr__(self) -> str:
        return (
            f"FakeLLMClient(model_id={self.model_id!r}, calls={self.calls}, "
            f"scripted={len(self.script)}, logprobs={self.emits_logprobs})"
        )


# --------------------------------------------------------------------------- #
# Catalogs
# --------------------------------------------------------------------------- #


class RefundArgs(BaseModel):
    """Args for the ``refund`` route — one required field, one optional.

    The required/optional split is the point: a missing ``order_id`` is what
    drives §3.8's row-4 downgrade from ``route`` to ``clarify``.
    """

    order_id: str
    reason: str | None = None


class TrackArgs(BaseModel):
    """Args for ``track_order`` — one required field."""

    order_id: str


@pytest.fixture
def refund_args() -> type[BaseModel]:
    """The ``refund`` route's args model, for tests that construct one directly."""
    return RefundArgs


@pytest.fixture
def small_registry() -> Registry:
    """A 5-route catalog below the ``"auto"`` bypass threshold (plan §5.3, §3.7).

    Deliberately covers every route feature the loop branches on:

    * ``refund`` — has an ``args_model`` with a required field (§4.5 layer 3);
    * ``track_order`` — args model, and lexically confusable with ``refund``
      because both talk about orders (the "confusion gap" of §5.7);
    * ``billing_report`` — carries ``requires={"billing"}``, so the entitlement
      filter has something to filter (§7.1);
    * ``account_settings`` — plain, no args;
    * ``human_handoff`` — ``pinned=True``, so it survives retrieval regardless of
      score and can serve as ``Router(fallback=...)`` (§5.5, §6.6).

    At five routes ``shortlist="auto"`` bypasses retrieval entirely, which is the
    §5.3 ``N < 25`` row: full entitled registry, canonical order, stable prefix.
    """
    return Registry(
        [
            Route(
                name="refund",
                description=(
                    "Issue or check a refund for an order. Use when the customer "
                    "wants money back. Do not use for delivery questions."
                ),
                args_model=RefundArgs,
                examples=(
                    "I want my money back for order 123",
                    "please refund my last purchase",
                ),
                tags=frozenset({"billing", "orders"}),
                clarify_label="a refund",
            ),
            Route(
                name="track_order",
                description=(
                    "Track shipment status for an existing order. Use for "
                    "'where is my package' questions. Do not use for refunds."
                ),
                args_model=TrackArgs,
                examples=("where is my package for order 123", "has my order shipped"),
                tags=frozenset({"orders", "shipping"}),
                clarify_label="tracking an order",
            ),
            Route(
                name="billing_report",
                description="Generate a billing statement for the current account.",
                examples=("send me this month's invoice",),
                tags=frozenset({"billing"}),
                requires=frozenset({"billing"}),
            ),
            Route(
                name="account_settings",
                description="Change account preferences such as email or password.",
                examples=("I want to change my password",),
                tags=frozenset({"account"}),
            ),
            Route(
                name="human_handoff",
                description="Escalate to a human support agent.",
                examples=("let me talk to a person",),
                tags=frozenset({"support"}),
                pinned=True,
            ),
        ]
    )


_DOMAINS: tuple[tuple[str, str], ...] = (
    ("billing", "invoices, charges and payment methods"),
    ("shipping", "parcels, carriers and delivery windows"),
    ("account", "profile, credentials and preferences"),
    ("catalog", "product listings, stock and pricing"),
    ("support", "tickets, escalations and service levels"),
    ("reporting", "exports, dashboards and scheduled summaries"),
)
_ACTIONS: tuple[tuple[str, str], ...] = (
    ("create", "Create a new"),
    ("update", "Update an existing"),
    ("cancel", "Cancel a pending"),
    ("list", "List every"),
    ("inspect", "Show the details of one"),
)


@pytest.fixture
def large_registry() -> Registry:
    """A generated 120-route catalog (plan §9.6 dogfood scale, §5.3 shortlist band).

    120 routes is not arbitrary: it is the size of the flagship catalog the v0.2
    kill-gate is measured on (§11), it sits well above the ``"auto"`` bypass
    threshold of 25 so retrieval genuinely runs, and it lands in the §5.3
    ``25 <= N < 150`` band where K=10.

    Routes are ``{domain}_{action}_{n}`` across six domains and five actions, with
    descriptions and examples that are *similar but distinguishable* — the point
    is to exercise retrieval and the confusion gap, not to make selection trivial.
    ``human_handoff`` is appended pinned, so fallback and the §5.5 pinned rule are
    testable at scale too.
    """
    routes: list[Route] = []
    for index in range(119):
        domain, subject = _DOMAINS[index % len(_DOMAINS)]
        action, verb = _ACTIONS[(index // len(_DOMAINS)) % len(_ACTIONS)]
        serial = index // (len(_DOMAINS) * len(_ACTIONS))
        name = f"{domain}_{action}_{serial}"
        routes.append(
            Route(
                name=name,
                description=(
                    f"{verb} {domain} record in workspace {serial} covering "
                    f"{subject}. Use when the request is about {domain} and asks "
                    f"to {action}. Do not use for other domains."
                ),
                examples=(
                    f"{action} the {domain} entry in workspace {serial}",
                    f"can you {action} my {domain} item",
                ),
                tags=frozenset({domain, action}),
            )
        )
    routes.append(
        Route(
            name="human_handoff",
            description="Escalate to a human support agent.",
            examples=("let me talk to a person",),
            tags=frozenset({"support"}),
            pinned=True,
        )
    )
    return Registry(routes)


@pytest.fixture
def ctx() -> RequestContext:
    """A populated :class:`~switchboard.RequestContext` (plan §3.5).

    Carries the ``billing`` entitlement, so ``small_registry``'s
    ``billing_report`` route survives the filter; drop it to exercise the
    entitlement boundary.
    """
    return RequestContext(
        tenant_id="acme",
        user_id="user-42",
        entitlements=frozenset({"billing", "pro"}),
        conversation_id="conv-7",
        locale="en-GB",
        trace_id="trace-abc",
    )


# --------------------------------------------------------------------------- #
# Router wiring
# --------------------------------------------------------------------------- #


@pytest.fixture
def sink() -> InMemorySink:
    """A ring-buffer sink; ``sink.latest`` is the last emitted audit record."""
    return InMemorySink(maxlen=100)


@pytest.fixture
def fake_client() -> FakeLLMClient:
    """An unscripted :class:`FakeLLMClient`; ``push()`` or reassign ``script``."""
    return FakeLLMClient()


@pytest.fixture
def router_factory(
    small_registry: Registry, sink: InMemorySink
) -> Callable[..., Router]:
    """Build a ``Router`` wired to a :class:`FakeLLMClient` (plan §3.3).

    Every ``Router(...)`` keyword passes straight through, so a test overrides
    exactly what it is about and inherits sensible defaults for the rest::

        router = router_factory(responses=[wire_output("refund", {"order_id": "1"})])
        router = router_factory(registry=large_registry, shortlist="bm25")
        router = router_factory(capabilities=NO_SIGNAL, fallback="human_handoff")

    Defaults chosen for tests rather than for production: OTel is off (no tracer
    provider is configured in the suite), the sink is the ``sink`` fixture so
    audit records are inspectable, and ``retry`` uses deterministic ``"expo"``
    backoff with a zero base delay so the transport-retry ladder can be exercised
    without sleeping.

    Args:
        responses: the ``FakeLLMClient`` script (ignored when ``client`` is given).
        capabilities: the client's declared capabilities.
        client: a pre-built client, bypassing ``FakeLLMClient`` entirely.
        registry: catalog override; defaults to ``small_registry``.
        client_kwargs: extra keyword arguments for ``FakeLLMClient``.
        **kwargs: forwarded verbatim to ``Router``.

    Returns:
        The constructed ``Router``. Its client is reachable as ``router.client``,
        so a test can read ``router.client.requests`` afterwards.
    """

    def build(
        responses: Any = None,
        *,
        capabilities: ClientCapabilities | None = None,
        client: Any = None,
        registry: Registry | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Router:
        if client is None:
            client = FakeLLMClient(
                responses,
                capabilities=capabilities,
                **dict(client_kwargs or {}),
            )
        kwargs.setdefault("sink", sink)
        kwargs.setdefault("otel", False)
        kwargs.setdefault(
            "retry", {"schema_attempts": 2, "provider_attempts": 3, "backoff": "expo", "base_delay": 0.0}
        )
        return Router(registry if registry is not None else small_registry, client=client, **kwargs)

    return build


@pytest.fixture
def router(router_factory: Callable[..., Router]) -> Router:
    """A ready-to-use router that always answers ``route -> refund``.

    The single most common setup: a happy-path decision with valid args, which
    every test about audit records, telemetry or the Decision contract can use
    without scripting a client of its own.
    """
    return router_factory(
        [wire_output("refund", {"order_id": "A-123", "reason": "damaged"})],
        fallback="human_handoff",
    )


def pytest_collection_modifyitems(items: Iterable[Any]) -> None:
    """Keep collection stable across runs (no reordering, no implicit markers).

    Declared explicitly so a future plugin cannot start shuffling the suite: the
    replay-cache tests of plan §9.6 depend on deterministic ordering to assert
    cache-hit behaviour across cases.
    """
    del items
