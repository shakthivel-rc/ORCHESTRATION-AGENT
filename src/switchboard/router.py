"""``Router`` — config wiring plus the ``route()`` / ``aroute()`` drivers (plan §3.3, §3.6).

The Router is deliberately thin. Every decision stage lives in
``engine/loop.py`` as a pure step; this module resolves configuration **once, at
construction**, and then runs those steps in order. That split is what plan §2.5
means by *"one pipeline, two drivers"*: ``route()`` and ``aroute()`` differ only
where they must — the provider call and the shortlist build — and nowhere else,
so sync/async parity is structural rather than maintained by hand.

**Eager validation (plan §2.4, §3.8).** Construction resolves the client (so a
missing extra raises :class:`~switchboard.errors.MissingDependencyError`
immediately, not on the first production request), checks that the fallback route
exists, that the thresholds are ordered and in ``[0, 1]``, that ``K >= 1``, and
that ``shortlist=None`` is not paired with a catalog larger than
``max_candidates`` — silent truncation would drop routes. It warns on a
reasoning-default model and on an oversized candidate budget. Everything that can
be wrong about a configuration is wrong at ``Router(...)``, not at request time.

**Thread safety (plan §2.5).** The Router holds no per-request mutable state: the
resolved configuration is a frozen :class:`~switchboard.engine.loop.LoopConfig`,
the registry is frozen, and everything else lives in a per-call
:class:`~switchboard.engine.loop.LoopState`. One instance is safe across threads
and tasks.

**Redaction (plan §7.4).** ``Router(redactor=...)`` is applied from exactly one
place — :meth:`Router._new_state` — and reaches all three egress sites from
there: the shortlist/embedding input, the routing prompt, and the audit/OTel
capture. There is no second hook to forget.

This module is Pydantic + stdlib only and must never import an adapter (§2.4):
``providers.resolve_client`` does that lazily, inside a function.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from switchboard._models import is_reasoning_default
from switchboard.core.audit import AuditDraft
from switchboard.core.registry import Registry

# NOTE: ConfidenceSource is imported at runtime, not under TYPE_CHECKING: it is a
# Pydantic field annotation on ConfidenceSpec, which Pydantic resolves at class
# construction time. Deferring it breaks the model build at import.
from switchboard.engine.loop import (
    ConfidenceSource,
    LoopConfig,
    LoopState,
    RetrySpec,
    finalize_audit,
    run_pipeline_async,
    run_pipeline_sync,
)
from switchboard.engine.policy import ThresholdPolicy
from switchboard.engine.shortlist import (
    DEFAULT_SHORTLIST_MIN_ROUTES,
    Shortlister,
    resolve_shortlister,
)
from switchboard.errors import ConfigError
from switchboard.providers import capabilities_of, resolve_client
from switchboard.telemetry.emitter import (
    resolve_sink,
    safe_aemit,
    safe_close,
    safe_emit,
    safe_flush,
)
from switchboard.telemetry.otel import OTelEmitter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from switchboard.core.audit import ConfidenceReport
    from switchboard.core.context import RequestContext
    from switchboard.core.decision import Decision
    from switchboard.engine.loop import OnProviderError
    from switchboard.engine.prompt import CandidateOrder
    from switchboard.engine.schema import SchemaModeSetting
    from switchboard.engine.shortlist import AsyncEmbedFn, EmbedFn, IndexStore
    from switchboard.telemetry.emitter import ContentMode, DecisionSink, Redactor

__all__ = [
    "ClientSpec",
    "ConfidenceSpec",
    "RetrySpec",
    "Router",
    "ShortlistSpec",
]

logger = logging.getLogger("switchboard.router")

_MAX_CANDIDATES_WARN = 50
"""Plan §3.3: warn above 50 candidates. Anthropic documents degradation past
~30-50 tools and OpenAI advises fewer than 20 per turn; past this the prompt is
the problem, not the model."""


# --------------------------------------------------------------------------- #
# Object forms of the string DSL (plan §3.3: "Object forms are the stable API;
# strings are sugar").
# --------------------------------------------------------------------------- #


class ClientSpec(BaseModel):
    """Object form of ``client="adapter:model"`` (plan §3.3).

    ``providers.resolve_client`` accepts this directly — it reads ``.adapter`` and
    ``.model`` — so a caller can build one, keep it in configuration, and hand it
    to any number of Routers.

        >>> ClientSpec.parse("instructor:openai/gpt-5-nano")
        ClientSpec(adapter='instructor', model='openai/gpt-5-nano', temperature=0.0, want_logprobs=True)
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    adapter: str
    model: str
    temperature: float = 0.0
    """Routing is a decision, not a composition: sampling above 0 buys nothing
    and costs reproducibility (plan §4.7)."""
    want_logprobs: bool = True
    """True at spec level; adapters map it down when the model cannot supply them."""

    @classmethod
    def parse(cls, spec: str, **overrides: Any) -> ClientSpec:
        """Parse ``"adapter:model"``. Only the first colon separates (§3.3)."""
        from switchboard.providers import parse_client_spec

        adapter, model = parse_client_spec(spec)
        return cls(adapter=adapter, model=model, **overrides)

    def __str__(self) -> str:
        return f"{self.adapter}:{self.model}"


class ShortlistSpec(BaseModel):
    """Object form of ``shortlist="bm25:top_k=10"`` (plan §3.3, §5.2).

    Rendered back to the DSL string that
    :func:`~switchboard.engine.shortlist.resolve_shortlister` consumes, so there
    is exactly one parser for the grammar and the object form cannot drift from
    the string form.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    variant: Literal["auto", "bm25", "embed", "hybrid"] = "auto"
    top_k: int | None = None
    min_routes: int | None = None
    model: str | None = None
    """Embedding model id, for ``variant="embed"``."""
    backend: str | None = None
    """Packaged embedding backend name (``"fastembed"``, ``"openai"``). [v0.2]"""

    @classmethod
    def parse(cls, spec: str) -> ShortlistSpec:
        """Parse a ``"variant[:key=value,...]"`` shortlist spec string."""
        head, _, tail = spec.strip().partition(":")
        variant = head.strip().lower()
        if variant not in ("auto", "bm25", "embed", "hybrid"):
            raise ConfigError(
                f"unknown shortlist variant {variant!r}; supported: "
                f"['auto', 'bm25', 'embed', 'hybrid'] (hybrid is [v0.2])"
            )
        values: dict[str, Any] = {"variant": variant}
        for item in tail.split(","):
            chunk = item.strip()
            if not chunk:
                continue
            key, sep, value = chunk.partition("=")
            if not sep:
                raise ConfigError(
                    f"malformed shortlist spec parameter {chunk!r}; expected key=value"
                )
            key = key.strip()
            if key not in ("top_k", "min_routes", "model", "backend"):
                raise ConfigError(f"unknown shortlist parameter {key!r} in {spec!r}")
            values[key] = value.strip()
        return cls.model_validate(values)

    def to_dsl(self) -> str:
        """Render back to the canonical DSL string."""
        params = [
            f"{name}={value}"
            for name, value in (
                ("top_k", self.top_k),
                ("min_routes", self.min_routes),
                ("model", self.model),
                ("backend", self.backend),
            )
            if value is not None
        ]
        return f"{self.variant}:{','.join(params)}" if params else self.variant

    def __str__(self) -> str:
        return self.to_dsl()


class ConfidenceSpec(BaseModel):
    """Object form of ``confidence="logprobs"`` (plan §3.3, §6.2).

    ``vote_n`` exists so the [v0.2] self-consistency surface is stable now, but a
    value above 1 is refused rather than silently ignored: a caller who asked for
    3 votes and got 1 would be reading a confidence number that does not mean
    what they think it means (§13 ruling #9).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    source: ConfidenceSource = "logprobs"
    vote_n: int = 1
    """Self-consistency votes. [v0.2]; must be 1 in v0.1."""

    @model_validator(mode="after")
    def _check(self) -> ConfidenceSpec:
        if self.vote_n != 1:
            raise ConfigError(
                f"confidence vote_n={self.vote_n} (self-consistency voting) is [v0.2] "
                f"and is not implemented in v0.1; use confidence='logprobs' or 'none' "
                f"(plan §13 ruling #9)."
            )
        return self

    @classmethod
    def parse(cls, spec: str) -> ConfidenceSpec:
        """Parse ``"logprobs"`` / ``"verbalized"`` / ``"none"``.

        ``"logprobs+vote:n=3"`` and the margin gate are [v0.2]; they raise a
        phase-tagged :class:`~switchboard.errors.ConfigError` rather than
        degrading silently to the logprob-only rung.
        """
        raw = spec.strip().lower()
        if not raw:
            raise ConfigError("confidence spec string is empty; use 'logprobs' or 'none'")
        if "vote" in raw or "margin" in raw:
            raise ConfigError(
                f"confidence={spec!r} requests a [v0.2] signal (self-consistency vote / "
                f"margin gate) that is not implemented in v0.1. Use 'logprobs' (the "
                f"default), 'verbalized' or 'none' (plan §13 ruling #9)."
            )
        if raw not in ("logprobs", "verbalized", "none"):
            raise ConfigError(
                f"unknown confidence source {spec!r}; supported: 'logprobs', "
                f"'verbalized', 'none'"
            )
        return cls(source=raw)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.source


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class Router:
    """The decision layer: one typed ``route()`` call over your catalog (plan §3).

    Example (plan §3.7)::

        registry = Registry([
            Route(name="refund", description="Issue or check a refund for an order",
                  args_model=RefundArgs),
            Route(name="track_order", description="Track shipment status"),
            Route(name="human_handoff", description="Escalate to a human", pinned=True),
        ])
        router = Router(registry=registry, client="instructor:openai/gpt-5-nano",
                        shortlist="auto", allow_clarify=True, fallback="human_handoff")
        decision = router.route("where is my package for order 123?",
                                context=RequestContext(tenant_id="acme"))
        if decision.kind == "route":
            print(decision.route, decision.args, decision.confidence.score)

    Args:
        registry: the frozen route catalog (plan §3.2).
        client: ``"adapter:model"``, a :class:`ClientSpec`, an object implementing
            ``complete``/``acomplete``, or a plain callable (the zero-extras BYO
            path, §4.1). Resolved eagerly.
        shortlist: ``"auto"`` (default — bypass below ``shortlist_min_routes``,
            else BM25), ``"bm25[:...]"``, ``"embed[:...]"``, a
            :class:`ShortlistSpec`, a :class:`~switchboard.engine.shortlist.Shortlister`
            instance, or ``None`` to disable retrieval entirely (§5.1).
        shortlist_k: K handed to the backend. ``None`` (default) sizes it from
            §5.3's evidence band table — 10 below 150 entitled routes, 15 to 999,
            20 above. Any value is clamped to ``[5, 20]`` unless
            ``allow_oversized_k`` is set.
        shortlist_min_routes: bypass threshold for ``"auto"`` (default 25, §5.3).
        multi_route: offer the ``multi_route`` arm (default False).
        allow_clarify: offer the ``clarify`` arm (default True).
        allow_plan: [v0.5]. Accepted and inert — ``plan`` is never emitted in v0.1.
        fallback: a **registered** route name substituted on terminal abstain
            paths (§6.6). Entitlement is never bypassed.
        confidence: ``"logprobs"`` (default), ``"verbalized"``, ``"none"``, or a
            :class:`ConfidenceSpec`. Inert when the client has no logprobs (§6.3).
        thresholds: a :class:`~switchboard.engine.policy.ThresholdPolicy` or a
            mapping of its fields. Downgrade-only (§6.4).
        max_candidates: hard cap on what reaches the prompt (default 25; warns
            above 50).
        candidate_order: ``"shuffle"`` (default, seeded), ``"score"`` or
            ``"registry"`` (§5.6).
        retry: a :class:`~switchboard.engine.loop.RetrySpec` or a mapping —
            ``schema_attempts=2``, ``provider_attempts=3``, ``expo_jitter``.
        redactor: ``Callable[[str, RequestContext], str]``, applied at all three
            egress sites from one place (§7.4).
        content_mode: ``"none"`` (default, hash-only), ``"redacted"`` or
            ``"full"`` audit payload capture (§7.4).
        wire_schema: ``"auto"`` (default), ``"dynamic"`` or ``"static"`` (§4.4).
        per_tenant_index: [v0.2]. Accepted; ``True`` raises because a shared index
            is not a per-tenant one and pretending otherwise would distort IDF
            silently (§5.4).
        on_provider_error: ``"raise"`` (default) or ``"abstain"`` (§3.8).
        sink: a :class:`~switchboard.telemetry.emitter.DecisionSink`, an iterable
            of them, or ``None`` for the no-op default (§8.3).
        system_prompt: replaces the shipped system block in prompt segment A. The
            segment layout, the untrusted-input delimiters and the ordering are
            structural and stay put (§4.6).
        embed: a BYO embedding callable for ``shortlist="embed"`` — works with
            zero extras (§5.2).
        index_store: where built indexes are persisted (§5.4).
        allow_oversized_k: permit ``shortlist_k > 20``, with a warning (§5.3).
        thresholds_on_verbalized: act on verbalized-only confidence. Off by
            default and warned about, because it inverts the report's
            Recommendation #4 (§6.3).
        confidence_fusion: replace
            :func:`~switchboard.engine.confidence.default_fusion` (§6.3).
        default_visibility: ``"allow"`` (v0.1). ``"deny"`` is [v0.2] and raises.
        temperature / max_tokens / seed: per-call provider knobs.
        otel: emit ``gen_ai.*`` spans when the ``[otel]`` extra is installed
            (§8.1). Spans are independent of ``sink``.
        tracer_provider: an explicit OTel provider; by default the globally
            configured one is used. switchboard never installs one.
        agent_name / agent_id: ``gen_ai.agent.name`` / ``gen_ai.agent.id``.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        client: Any,
        shortlist: str | ShortlistSpec | Shortlister | None = "auto",
        shortlist_k: int | None = None,
        shortlist_min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
        multi_route: bool = False,
        allow_clarify: bool = True,
        allow_plan: bool = False,
        fallback: str | None = None,
        confidence: str | ConfidenceSpec = "logprobs",
        thresholds: ThresholdPolicy | Mapping[str, Any] | None = None,
        max_candidates: int = 25,
        candidate_order: CandidateOrder = "shuffle",
        retry: RetrySpec | Mapping[str, Any] | None = None,
        redactor: Callable[[str, RequestContext], str] | None = None,
        content_mode: ContentMode = "none",
        wire_schema: SchemaModeSetting = "auto",
        per_tenant_index: bool = False,
        on_provider_error: OnProviderError = "raise",
        sink: DecisionSink | Iterable[DecisionSink] | None = None,
        system_prompt: str | None = None,
        embed: EmbedFn | AsyncEmbedFn | None = None,
        index_store: IndexStore | None = None,
        allow_oversized_k: bool = False,
        thresholds_on_verbalized: bool = False,
        confidence_fusion: Callable[[ConfidenceReport], float] | None = None,
        default_visibility: Literal["allow", "deny"] = "allow",
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        otel: bool = True,
        tracer_provider: Any | None = None,
        agent_name: str = "switchboard.router",
        agent_id: str | None = None,
    ) -> None:
        if not isinstance(registry, Registry):
            raise ConfigError(
                f"Router(registry=...) must be a Registry, got "
                f"{type(registry).__name__}. Build one with Registry([Route(...), ...])."
            )
        self._registry = registry

        # -- client: resolved eagerly so a missing extra fails now (§2.4) ----- #
        self._client = resolve_client(client)
        capabilities = capabilities_of(self._client)
        self._sync_capable = callable(getattr(self._client, "complete", None))
        self._async_capable = callable(getattr(self._client, "acomplete", None))
        if not (self._sync_capable or self._async_capable):
            raise ConfigError(
                f"Resolved client {type(self._client).__name__!r} implements neither "
                f"complete() nor acomplete(); it cannot answer a routing call "
                f"(plan §4.1)."
            )
        request_model = _client_model(self._client)
        provider = _client_provider(self._client)

        # -- thresholds / retry / confidence: object forms -------------------- #
        policy = _as_model(ThresholdPolicy, thresholds, "thresholds")
        retry_spec = _as_model(RetrySpec, retry, "retry")
        confidence_spec = (
            confidence
            if isinstance(confidence, ConfidenceSpec)
            else ConfidenceSpec.parse(confidence)
        )

        # -- shortlist -------------------------------------------------------- #
        if shortlist_k is not None and shortlist_k < 1:
            raise ConfigError(
                f"Router(shortlist_k={shortlist_k}) must be >= 1; a shortlist of "
                f"zero routes cannot produce a decision (plan §3.8)."
            )
        if shortlist_min_routes < 0:
            raise ConfigError(
                f"Router(shortlist_min_routes={shortlist_min_routes}) must be >= 0"
            )
        if max_candidates < 1:
            raise ConfigError(
                f"Router(max_candidates={max_candidates}) must be >= 1"
            )
        if default_visibility not in ("allow", "deny"):
            raise ConfigError(
                f"Router(default_visibility={default_visibility!r}) must be "
                f"'allow' or 'deny' (plan §7.1)."
            )
        if default_visibility == "deny":
            # Checked here as well as in engine/entitlements.py so the refusal
            # lands at construction rather than on the first production request:
            # a deployment that asked for deny-by-default and silently got
            # allow-by-default is a security regression (plan §2.4, §7.1).
            raise ConfigError(
                "Router(default_visibility='deny') (deny-by-default entitlements) "
                "is [v0.2] and is not implemented in v0.1. It is refused rather "
                "than silently downgraded to 'allow'. Express the same posture "
                "explicitly with Route.requires on every route (plan §7.1)."
            )
        if per_tenant_index:
            raise ConfigError(
                "Router(per_tenant_index=True) is [v0.2] and is not implemented in "
                "v0.1. The v0.1 index is shared over the full registry with a "
                "per-request entitlement filter (plan §5.4); silently serving that "
                "as a per-tenant index would distort IDF without saying so."
            )
        self._shortlister = _resolve_shortlist(
            shortlist,
            embed=embed,
            min_routes=shortlist_min_routes,
            store=index_store,
        )
        if self._shortlister is None and len(registry) > max_candidates:
            raise ConfigError(
                f"Router(shortlist=None) with {len(registry)} routes exceeds "
                f"max_candidates={max_candidates}. Silent truncation would drop "
                f"routes, so this is refused (plan §3.3). Use shortlist='auto' "
                f"(which never errors), or raise max_candidates."
            )

        # -- ordering / fallback ---------------------------------------------- #
        if candidate_order not in ("shuffle", "score", "registry"):
            raise ConfigError(
                f"Router(candidate_order={candidate_order!r}) must be 'shuffle', "
                f"'score' or 'registry' (plan §5.6)."
            )
        if fallback is not None and fallback not in registry:
            raise ConfigError(
                f"Router(fallback={fallback!r}) names a route that is not in the "
                f"registry. A fallback must be a registered route (plan §6.6); "
                f"available: {', '.join(sorted(registry.names)[:20])}"
                f"{'...' if len(registry) > 20 else ''}"
            )

        # -- warnings (plan §3.3, §4.3, §6.3) --------------------------------- #
        if max_candidates > _MAX_CANDIDATES_WARN:
            warnings.warn(
                f"Router(max_candidates={max_candidates}) exceeds "
                f"{_MAX_CANDIDATES_WARN}. Tool-selection accuracy degrades past "
                f"~30-50 candidates and collapses at 100+ (plan §3.3, §5.3); "
                f"prefer a shortlist over a bigger prompt.",
                stacklevel=2,
            )
        if request_model and is_reasoning_default(request_model):
            warnings.warn(
                f"Router(client=...) is configured with {request_model!r}, which "
                f"engages extended reasoning by default and cannot be forced off. "
                f"Routing on a reasoning mode is a misconfiguration (~13s vs ~0.33s "
                f"TTFT, plan §4.3); use a small non-reasoning model such as "
                f"gemini-2.5-flash-lite or gpt-5-nano.",
                stacklevel=2,
            )
        if thresholds_on_verbalized:
            warnings.warn(
                "Router(thresholds_on_verbalized=True) lets clarify/abstain "
                "downgrades act on the model's own stated confidence. The evidence "
                "is that verbalized confidence tracks commitment rather than "
                "correctness (plan §6.3), so this inverts the library's default "
                "posture. Prefer a client that returns logprobs.",
                stacklevel=2,
            )
        stale = policy.validate_provenance(
            model_id=request_model, registry_version=registry.version
        )
        if stale:
            warnings.warn(
                f"ThresholdPolicy was calibrated against registry version "
                f"{policy.registry_version!r} but this registry is "
                f"{registry.version!r}. Route descriptions changed underneath the "
                f"calibration, so the thresholds are approximate (plan §6.4).",
                stacklevel=2,
            )

        # -- telemetry -------------------------------------------------------- #
        self._sink = resolve_sink(sink)
        self._emitter = OTelEmitter(
            content_mode=content_mode,
            tracer_provider=tracer_provider,
            enabled=otel,
            agent_name=agent_name,
            agent_id=agent_id,
        )
        self._redactor = redactor
        self._content_mode: ContentMode = content_mode

        # -- the frozen, shared loop configuration (plan §2.5) ---------------- #
        self._config = LoopConfig(
            registry=registry,
            capabilities=capabilities,
            emitter=self._emitter,
            shortlister=self._shortlister,
            shortlist_k=shortlist_k,
            allow_oversized_k=allow_oversized_k,
            max_candidates=max_candidates,
            candidate_order=candidate_order,
            multi_route=multi_route,
            allow_clarify=allow_clarify,
            allow_plan=allow_plan,
            fallback=fallback,
            thresholds=policy,
            confidence_source=confidence_spec.source,
            confidence_fusion=confidence_fusion,
            thresholds_on_verbalized=thresholds_on_verbalized,
            retry=retry_spec,
            wire_schema=wire_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            on_provider_error=on_provider_error,
            provider=provider,
            request_model=request_model,
            system_prompt=system_prompt,
            content_mode=content_mode,
            default_visibility=default_visibility,
        )

    # -- introspection ------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"Router(routes={len(self._registry)}, "
            f"version={self._registry.version!r}, "
            f"model={self._config.request_model!r}, "
            f"shortlist={type(self._shortlister).__name__ if self._shortlister else None})"
        )

    @property
    def registry(self) -> Registry:
        """The frozen catalog this router decides over."""
        return self._registry

    @property
    def client(self) -> Any:
        """The resolved provider client."""
        return self._client

    @property
    def config(self) -> LoopConfig:
        """The frozen, shared loop configuration (plan §2.5)."""
        return self._config

    @property
    def shortlister(self) -> Shortlister | None:
        """The resolved retrieval backend, or ``None`` when disabled."""
        return self._shortlister

    @property
    def sink(self) -> DecisionSink:
        """The resolved decision sink (a ``NoopSink`` when none was configured)."""
        return self._sink

    @property
    def emitter(self) -> OTelEmitter:
        """The OTel span emitter. Also carries ``execute_tool_span()`` (§8.1)."""
        return self._emitter

    # -- call surface (plan §3.6) -------------------------------------------- #

    def route(self, query: str, *, context: RequestContext | None = None) -> Decision:
        """Decide which route handles ``query`` (plan §3.6).

        Returns a :data:`~switchboard.core.decision.Decision` — ``route``,
        ``multi_route``, ``clarify`` or ``abstain``. Clarify and abstain are
        *results*, not exceptions (§3, decision (c)), and a configured fallback
        arrives as ``kind="route"`` with ``decision_path="fallback"``, so a caller
        branching on ``kind`` alone needs no special cases (§13 ruling #5).

        Raises:
            ConfigError: the resolved client is async-only. Plan §2.5 makes the
                sync-driver/async-client pair a configuration error; use
                :meth:`aroute` instead.
            ProviderError: transport failure that survived the retry budget,
                unless ``on_provider_error="abstain"`` (§3.8). An audit record and
                a span are still emitted — the emit is in a ``finally``.
        """
        if not self._sync_capable:
            raise ConfigError(
                f"Router.route() is synchronous but the configured client "
                f"({type(self._client).__name__}) only implements acomplete(). "
                f"Use `await router.aroute(...)`, or pass a synchronous client "
                f"(plan §2.5)."
            )
        state = self._new_state(query, context)
        with self._emitter.decision_span(
            conversation_id=context.conversation_id if context else None
        ) as span:
            _stamp_span_ids(state, span)
            try:
                state = run_pipeline_sync(state, self._client)
            except BaseException as exc:
                state.draft.set_error(exc)
                raise
            finally:
                state = finalize_audit(state)
                if state.record is not None:
                    span.set_record(state.record)
                    self._emitter.record_metrics(state.record)
                    safe_emit(self._sink, state.record)
        return _require_decision(state)

    async def aroute(
        self, query: str, *, context: RequestContext | None = None
    ) -> Decision:
        """Async twin of :meth:`route` (plan §3.6, §2.5).

        Identical pipeline; only the shortlist build and the provider call are
        awaited. A **sync-only** client is driven through ``asyncio.to_thread``
        so it cannot block the event loop — the documented direction of §2.5's
        mismatch rule.
        """
        state = self._new_state(query, context)
        async with self._emitter.adecision_span(
            conversation_id=context.conversation_id if context else None
        ) as span:
            _stamp_span_ids(state, span)
            try:
                state = await run_pipeline_async(state, self._client)
            except BaseException as exc:
                state.draft.set_error(exc)
                raise
            finally:
                state = finalize_audit(state)
                if state.record is not None:
                    span.set_record(state.record)
                    self._emitter.record_metrics(state.record)
                    await safe_aemit(self._sink, state.record)
        return _require_decision(state)

    # -- index lifecycle (plan §5.4) ----------------------------------------- #

    def warm(self) -> None:
        """Build the shortlist index eagerly (plan §3.6, §5.4).

        Optional: the index is built lazily on the first request after a registry
        version change anyway. Calling this at startup moves that cost off the
        first user's latency budget. Idempotent.
        """
        if self._shortlister is not None:
            self._shortlister.build(
                self._registry.routes, registry_version=self._registry.version
            )

    async def awarm(self) -> None:
        """Async twin of :meth:`warm`, single-flighted against stampedes (§5.4)."""
        if self._shortlister is not None:
            await self._shortlister.abuild(
                self._registry.routes, registry_version=self._registry.version
            )

    # -- streaming (plan §3.6) [v0.2] ---------------------------------------- #

    def stream_route(
        self, query: str, *, context: RequestContext | None = None
    ) -> Iterator[Any]:
        """Stream rationale deltas and commit the decision atomically. **[v0.2]**

        Defined so the call surface of §3.6 exists and type-checks from v0.1; it
        raises rather than silently falling back to :meth:`route`, because a
        caller who wrote a streaming UI against it would otherwise see nothing
        stream and no error explaining why (§13 ruling #10).
        """
        del query, context
        raise ConfigError(
            "[v0.2] Router.stream_route() (rationale deltas with an atomic "
            "decision commit) is not implemented in v0.1; it was deliberately "
            "removed from the MVP (plan §13 ruling #10). Use route() / aroute()."
        )

    async def astream_route(
        self, query: str, *, context: RequestContext | None = None
    ) -> Any:
        """Async twin of :meth:`stream_route`. **[v0.2]**"""
        del query, context
        raise ConfigError(
            "[v0.2] Router.astream_route() is not implemented in v0.1 (plan §13 "
            "ruling #10). Use aroute()."
        )

    # -- outcome join (plan §8.2, §8.5) [v0.5] -------------------------------- #

    def record_outcome(self, decision_id: str, outcome: str) -> None:
        """Attach a downstream outcome to a past decision (plan §8.2). **[v0.5] seam**

        ``AuditRecord.outcome`` is joined *later*, once the caller knows whether
        executing the chosen route actually worked; the distillation dataset
        builder then treats ``outcome == "success"`` as a training-eligibility
        signal independent of confidence (§8.5 step 1).

        v0.1 forwards to the sink when it implements ``record_outcome`` — that is
        the seam a JSONL-tailing store or a database sink hooks — and is otherwise
        a no-op. It deliberately never raises and never mutates an emitted record:
        an :class:`~switchboard.core.audit.AuditRecord` is frozen, and rewriting
        history in place would break the [v1.0] tamper-evident hash chain.
        """
        forward = getattr(self._sink, "record_outcome", None)
        if callable(forward):
            try:
                forward(decision_id, outcome)
            except Exception:  # sinks never break the caller (plan §8.3)
                logger.warning(
                    "switchboard: sink.record_outcome() failed for decision %s",
                    decision_id,
                    exc_info=True,
                )

    async def arecord_outcome(self, decision_id: str, outcome: str) -> None:
        """Async twin of :meth:`record_outcome`. **[v0.5] seam**"""
        forward = getattr(self._sink, "arecord_outcome", None)
        if callable(forward):
            try:
                await forward(decision_id, outcome)
            except Exception:
                logger.warning(
                    "switchboard: sink.arecord_outcome() failed for decision %s",
                    decision_id,
                    exc_info=True,
                )
            return
        self.record_outcome(decision_id, outcome)

    # -- lifecycle ------------------------------------------------------------ #

    def close(self) -> None:
        """Flush and close the decision sink. Idempotent; never raises (§8.3)."""
        safe_flush(self._sink)
        safe_close(self._sink)

    async def aclose(self) -> None:
        """Async twin of :meth:`close`."""
        await asyncio.to_thread(self.close)

    # -- internals ------------------------------------------------------------ #

    def _new_state(self, query: str, context: RequestContext | None) -> LoopState:
        """Build the per-request state, applying the redactor **once** (plan §7.4).

        This is the single place ``Router(redactor=...)`` is invoked on the query.
        The redacted string then reaches every egress site from here: the
        shortlist/embedding input, the prompt (segment D), and — via the bound,
        context-free form handed to ``apply_content_mode`` — the audit record and
        the OTel span. There is no second hook to forget.

        A redactor that *raises* is treated as broken configuration and
        propagated, not swallowed: failing open would send the raw query to the
        provider, which is precisely what the hook exists to prevent (§3.8's
        "raise iff the caller's configuration is broken").
        """
        redacted = query
        bound: Redactor | None = None
        if self._redactor is not None:
            bound = _bind_redactor(
                self._redactor, context if context is not None else _empty_context()
            )
            try:
                redacted = bound(query)
            except Exception as exc:
                raise ConfigError(
                    f"Router(redactor=...) raised {type(exc).__name__}: {exc}. The "
                    f"redactor is a security control applied before the query "
                    f"reaches the provider (plan §7.4), so a failing redactor is "
                    f"refused rather than bypassed."
                ) from exc

        return LoopState(
            query=query,
            config=self._config,
            draft=AuditDraft(),
            ctx=context,
            redacted_query=redacted,
            bound_redactor=bound,
        )


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #

_EMPTY_CONTEXT: RequestContext | None = None


def _empty_context() -> RequestContext:
    """The stand-in context handed to a redactor when the caller supplied none.

    ``RequestContext`` is frozen, so one shared instance is safe and avoids
    rebuilding a Pydantic model on every context-less request.
    """
    global _EMPTY_CONTEXT
    if _EMPTY_CONTEXT is None:
        from switchboard.core.context import RequestContext as _RequestContext

        _EMPTY_CONTEXT = _RequestContext()
    return _EMPTY_CONTEXT


def _bind_redactor(
    redactor: Callable[[str, RequestContext], str], ctx: RequestContext
) -> Redactor:
    """Narrow the two-argument public hook to the one-argument audit-site form.

    ``Router(redactor=...)`` is documented as ``Callable[[str, RequestContext],
    str]`` (plan §3.3) while
    :func:`~switchboard.telemetry.emitter.apply_content_mode` has no request
    context of its own; binding once per request is what keeps a single hook
    serving all three egress sites (§7.4).
    """

    def bound(text: str) -> str:
        return redactor(text, ctx)

    return bound


def _stamp_span_ids(state: LoopState, span: Any) -> None:
    """Copy the decision span's ids into the audit draft (plan §8.2).

    Both are ``None`` when the ``[otel]`` extra is absent, in which case a
    caller-supplied ``RequestContext.trace_id`` is used instead — a record must be
    correlatable whether or not tracing is installed.
    """
    trace_id = span.trace_id
    if trace_id is not None:
        state.draft.trace_id = trace_id
    state.draft.span_id = span.span_id


def _require_decision(state: LoopState) -> Decision:
    """Return the decision, or fail loudly rather than returning ``None``."""
    if state.decision is None:  # pragma: no cover - every path sets one
        raise ConfigError(
            "the decision loop finished without producing a Decision; this is a "
            "switchboard bug, please report it with the audit record"
        )
    return state.decision


def _as_model(model: type[Any], value: Any, name: str) -> Any:
    """Accept an instance, a mapping of fields, or ``None`` for the defaults."""
    if value is None:
        return model()
    if isinstance(value, model):
        return value
    if isinstance(value, dict):
        return model(**value)
    raise ConfigError(
        f"Router({name}=...) must be a {model.__name__}, a mapping of its fields, "
        f"or None; got {type(value).__name__}."
    )


def _resolve_shortlist(
    spec: str | ShortlistSpec | Shortlister | None,
    *,
    embed: EmbedFn | AsyncEmbedFn | None,
    min_routes: int,
    store: IndexStore | None,
) -> Shortlister | None:
    """Normalize ``Router(shortlist=...)`` to a backend or ``None`` (plan §3.3)."""
    if isinstance(spec, ShortlistSpec):
        spec = spec.to_dsl()
    return resolve_shortlister(
        spec, embed=embed, min_routes=min_routes, store=store
    )


def _client_model(client: Any) -> str | None:
    """The model id to stamp on requests, spans and audit records (§8.2).

    Adapters spell it differently — ``LiteLLMAdapter.model`` /
    ``InstructorAdapter.model`` carry the full ``provider/model`` spec, while
    ``CallableAdapter.model_id`` carries ``"byo:<name>"`` — so this reads them in
    the order that yields the most specific answer.
    """
    for attribute in ("model", "model_id", "model_name"):
        value = getattr(client, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _client_provider(client: Any) -> str | None:
    """The vendor name for ``gen_ai.provider.name`` (§8.1), when the adapter knows."""
    for attribute in ("provider", "route_prefix", "vendor"):
        value = getattr(client, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None
