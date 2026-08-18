"""Pre-LLM entitlement filtering (plan §7.1; §2.2 row "Entitlement filter").

**Phase: [v0.2]; this is the documented passthrough stub [v0.1]** (§13 ruling #20,
§11 week 2). Read that literally, because the word "stub" is doing less work than
it looks like:

* The **predicates themselves are real and evaluated here in v0.1.**
  :attr:`Route.requires` compiles to ``ctx.entitlements >= route.requires`` and
  :attr:`Route.visibility` is called — both are cheap, both are correct, and
  implementing them was never the expensive part.
* The **default registry — no ``requires``, no ``visibility`` — passes through
  untouched**, which is why this is honestly labelled a passthrough: for the
  quickstart catalog in §3.7 this function returns exactly what it was given,
  with the shared cohort key, and no per-route work worth measuring.
* What is genuinely deferred to [v0.2] is *not* the predicate: it is **cohort
  view caching** (§7.2 — LRU-cached ``RegistryView`` keyed by
  ``(registry_hash, entitlement_key)``, so a cohort shares one rendered prompt
  block and one provider-side cache prefix) and **deny-by-default**
  (``Router(default_visibility="deny")`` for regulated deployments).

Until the [v0.2] filter is wired into ``Registry.view()``, callers must not treat
``Registry.view(ctx)`` as a security boundary — *this* function is where the
boundary is enforced today. Everything it drops is dropped **before** the
shortlister runs and therefore before any route name can enter a prompt, which is
what makes pre-LLM filtering simultaneously a security boundary and the report's
prompt-shrinking accuracy win (§7.1, §7.4).

Predicates are sync, pure and O(µs) by contract: they run over every route on
every request. Slow lookups (DB / IdP / policy service) are resolved by the
caller *before* ``route()`` into ``RequestContext.entitlements`` (§3.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from switchboard.core.audit import canonical_json, sha256_hex
from switchboard.core.context import RequestContext
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from switchboard.core.route import Route

__all__ = ["SHARED_COHORT_KEY", "EntitlementResult", "filter_routes"]

SHARED_COHORT_KEY = ""
"""The ``entitlement_key`` for a registry that declares no entitlements at all.

Deliberately the empty string, matching the constant ``Registry.view()`` uses for
its v0.1 passthrough view: when no route carries ``requires`` or ``visibility``
there is no entitlement dimension, every context is in one cohort, and the
rendered route directory (prompt segment B, §4.6) is byte-identical for everyone.
Deriving a per-registry key there would fragment the provider prompt cache for
no reason — ``view_hash`` already mixes in ``registry_hash`` (§7.2).
"""

_KEY_PREFIX_LEN = 12
"""Width of the derived cohort key, matching ``registry_version`` and
``view_hash`` (§7.2, §7.3). 48 bits of digest is ample for cohort identity and
keeps prompt-cache keys short (§4.6)."""


@dataclass(frozen=True, slots=True)
class EntitlementResult:
    """What the entitlement filter produced for one request (plan §2.2, §7.1)."""

    routes: tuple[Route, ...]
    """The surviving routes, in the input catalog's order.

    May be **empty**: an empty entitled set is a normal outcome that degrades to
    ``abstain(reason="no_eligible_routes")`` with no LLM call (§13 ruling #3), not
    an error.
    """

    entitlement_key: str | None
    """Deterministic cohort key, or ``None`` when this request is uncacheable.

    ``None`` means "a ``visibility`` lambda participated and the caller supplied no
    ``RequestContext.entitlement_key``" — the view is correct, it simply cannot be
    keyed for the [v0.2] cohort cache (§7.1).
    """

    filtered_count: int
    """How many routes the filter removed (``len(input) - len(routes)``)."""

    predicate_errors: tuple[tuple[str, str], ...] = ()
    """``(route_name, exception_type_name)`` for every ``visibility`` predicate that
    raised. A raising predicate is treated as ``False`` — fail closed — and recorded
    here rather than propagated: a user-supplied lambda must never be able to crash
    ``route()`` (§7.1: predicates are pure and O(µs); anything else is a bug in the
    predicate, not a routing failure)."""

    @property
    def names(self) -> tuple[str, ...]:
        """Surviving route names, in catalog order."""
        return tuple(route.name for route in self.routes)

    @property
    def allowed(self) -> frozenset[str]:
        """Surviving route names as a set.

        This is the ``allowed`` set the validator enforces membership against
        (§4.5 step 2) and the enum a dynamic wire schema is built from (§4.4).
        """
        return frozenset(route.name for route in self.routes)

    @property
    def cacheable(self) -> bool:
        """Whether :attr:`entitlement_key` can key a [v0.2] cohort view cache."""
        return self.entitlement_key is not None

    def __len__(self) -> int:
        return len(self.routes)


def filter_routes(
    registry: Iterable[Route],
    ctx: RequestContext | None = None,
    default_visibility: Literal["allow", "deny"] = "allow",
) -> EntitlementResult:
    """Filter ``registry`` down to the routes ``ctx`` is entitled to see (plan §7.1).

    Two predicates, evaluated in this order per route:

    1. ``ctx.entitlements >= route.requires`` — declarative claim-set sugar.
       Hashable, side-effect free, and the only kind of predicate that contributes
       to a deterministic :attr:`~EntitlementResult.entitlement_key`.
    2. ``route.visibility(ctx)`` — the escape hatch, an arbitrary sync predicate.
       Called defensively: an exception is recorded in
       :attr:`~EntitlementResult.predicate_errors` and treated as ``False``.

    A route with neither is visible to everyone, so the default catalog passes
    through untouched (see the module docstring on what "passthrough stub" means
    at v0.1).

    Args:
        registry: any iterable of routes — a :class:`~switchboard.core.registry.Registry`,
            a :class:`~switchboard.core.registry.RegistryView`, or a plain sequence.
            Consumed exactly once.
        ctx: the request context. ``None`` is treated as an empty
            :class:`~switchboard.core.context.RequestContext`, so routes that declare
            ``requires`` are filtered out — declaring a requirement is opting into
            enforcement, and "no context" cannot satisfy it.
        default_visibility: ``"allow"`` (v0.1). ``"deny"`` is the [v0.2]
            deny-by-default posture for regulated deployments and raises
            :class:`~switchboard.errors.ConfigError` here rather than silently
            behaving as ``"allow"`` — a config that promises deny-by-default and
            quietly grants allow-by-default is exactly the kind of broken
            configuration §3.8 says must raise.

    Returns:
        An :class:`EntitlementResult`. The caller decides what an empty result
        means; the router degrades it to ``abstain("no_eligible_routes")`` with no
        LLM call (§13 ruling #3).

    Raises:
        ConfigError: ``default_visibility`` is not ``"allow"`` (see above).
    """
    if default_visibility not in ("allow", "deny"):
        raise ConfigError(
            f"default_visibility must be 'allow' or 'deny', got {default_visibility!r}"
        )
    if default_visibility == "deny":
        raise ConfigError(
            "default_visibility='deny' (deny-by-default) is [v0.2] and is not "
            "implemented in v0.1; it is refused rather than silently downgraded to "
            "'allow', because a deployment that asked for deny-by-default and got "
            "allow-by-default is a security regression. Until then, express the same "
            "posture explicitly with Route.requires on every route."
        )

    routes = tuple(registry)
    context = ctx if ctx is not None else RequestContext()

    surviving: list[Route] = []
    predicate_errors: list[tuple[str, str]] = []
    declares_entitlements = False
    predicate_participated = False

    for route in routes:
        if route.requires:
            declares_entitlements = True
            if not context.entitlements >= route.requires:
                continue
        predicate = route.visibility
        if predicate is not None:
            declares_entitlements = True
            predicate_participated = True
            try:
                visible = bool(predicate(context))
            except Exception as exc:  # fail closed; a user lambda never crashes route()
                predicate_errors.append((route.name, type(exc).__name__))
                visible = False
            if not visible:
                continue
        surviving.append(route)

    return EntitlementResult(
        routes=tuple(surviving),
        entitlement_key=_entitlement_key(
            surviving,
            context,
            declares_entitlements=declares_entitlements,
            predicate_participated=predicate_participated,
        ),
        filtered_count=len(routes) - len(surviving),
        predicate_errors=tuple(predicate_errors),
    )


def _entitlement_key(
    surviving: list[Route],
    ctx: RequestContext,
    *,
    declares_entitlements: bool,
    predicate_participated: bool,
) -> str | None:
    """Derive the cohort key for this filter pass (plan §7.1, §7.2, §13 ruling #15).

    Three cases, in priority order:

    1. **A ``visibility`` lambda was actually invoked** → the outcome is not a pure
       function of ``ctx.entitlements`` (a lambda may read ``user_id``, ``extra``,
       or worse, the wall clock), so no key can be *derived*. The caller's
       ``RequestContext.entitlement_key`` is used when supplied; otherwise ``None``
       — this request opts out of view caching, exactly as §7.1 specifies.
    2. **No route declares any entitlement** → :data:`SHARED_COHORT_KEY`: one
       cohort, one rendered route directory, one provider cache prefix.
    3. **Only ``requires`` participated** → a deterministic sha256 over the sorted
       surviving names. Identical entitlement profiles produce identical keys
       across processes and machines (canonical JSON + sha256, §8.2), which is what
       makes the cohort prompt-cache discount cohort-wide rather than per-tenant.
    """
    if predicate_participated:
        return ctx.entitlement_key
    if not declares_entitlements:
        return SHARED_COHORT_KEY
    names = sorted(route.name for route in surviving)
    return sha256_hex(canonical_json(names))[:_KEY_PREFIX_LEN]
