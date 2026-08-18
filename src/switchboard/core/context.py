"""``RequestContext`` — per-request caller state (plan §3.5).

Everything the routing decision may depend on that is *not* the query itself
lives here. The hard rule from plan §7.1: predicates only **read** this object.
Slow lookups (DB / IdP / policy service) are resolved by the caller *before*
``route()`` and handed in as ``entitlements`` — entitlement predicates run over
every route on every request, pre-LLM, and must stay O(microseconds).

Fields tagged [v0.2] in the plan exist here in v0.1 as inert, accepted-and-stored
fields: the contract is stable now, the behaviour lands later.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RequestContext"]


class RequestContext(BaseModel):
    """Caller-supplied context for one routing decision (plan §3.5).

    Frozen: the loop treats it as an immutable input so a decision is exactly
    reconstructible from its audit record.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str | None = None
    """Stored raw in the audit record; hashed by default on OTel spans (§7.4)."""
    user_id: str | None = None
    entitlements: frozenset[str] = frozenset()
    """Consumed by ``Route.requires`` and ``Route.visibility`` predicates (§7.1)."""
    entitlement_key: str | None = None
    """Caller-supplied cohort key, needed for view caching when lambda
    ``visibility`` predicates opt out of the derived key. [v0.2]"""
    fallback_route: str | None = None
    """Per-request fallback override; must still pass entitlement (§6.6). [v0.2]"""
    conversation_id: str | None = None
    """Maps to the ``gen_ai.conversation.id`` span attribute (§8.1)."""
    locale: str | None = None
    history: tuple[dict[str, str], ...] = ()
    """Optional prior turns, appended after the cached prefix (§4.6)."""
    trace_id: str | None = None
    """Propagated into the audit record and the OTel span (§8.2)."""
    extra: dict[str, Any] = Field(default_factory=dict)

    def has(self, entitlement: str) -> bool:
        """Return whether ``entitlement`` is present in :attr:`entitlements`.

        The readable spelling of ``entitlement in ctx.entitlements`` used by
        ``visibility`` predicates (plan §3.5, §7.1).
        """
        return entitlement in self.entitlements
