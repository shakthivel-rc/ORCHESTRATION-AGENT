"""``Registry`` / ``RegistryView`` — the immutable route catalog (plan §3.2, §7.2).

Frozen after construction (plan §3, decision (d)): composition returns a *new*
object. This is not stylistic — ``registry.version`` is a content hash that keys
the provider prompt cache, the shortlist index and every audit record, so a
mutable catalog would silently poison all three (§7.3). It is also what makes a
single ``Router`` instance safe to share across threads and tasks (§2.5).

Nothing here is invalidated by wall-clock or by deploys — only by content.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, overload

from switchboard.core.audit import canonical_json, sha256_hex
from switchboard.core.route import Route
from switchboard.errors import ConfigError, RegistryError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from switchboard.core.context import RequestContext

__all__ = ["Registry", "RegistryView"]

_VERSION_PREFIX_LEN = 12
"""Plan §7.3: ``registry_version = registry_hash[:12]``. 48 bits of digest is
ample for cache keying and keeps prompt cache keys short (§4.6)."""

_V01_ENTITLEMENT_KEY = ""
"""The entitlement key used by :meth:`Registry.view` in v0.1.

Entitlement filtering ships as a documented passthrough stub in v0.1 (§13 ruling
#20, §11 week 2), so *every* context yields the same view. Deriving ``view_hash``
from a constant empty key rather than from ``ctx.entitlement_key`` is therefore
the honest encoding: identical views must produce identical hashes, otherwise
the prompt-cache prefix would be needlessly fragmented per tenant while the
rendered block is byte-identical. [v0.2] swaps this for the real cohort key.
"""


def _view_hash(registry_hash: str, entitlement_key: str | None) -> str:
    """``sha256(registry_hash + canonical(entitlement_key))[:12]`` (plan §7.2)."""
    return sha256_hex(registry_hash, canonical_json(entitlement_key))[
        :_VERSION_PREFIX_LEN
    ]


class _RouteSequence:
    """Shared read-only sequence surface over an ordered set of routes.

    Backs both :class:`Registry` and :class:`RegistryView` so callers can treat
    a tenant view exactly like a registry: iterate it, ``len()`` it, test
    membership, and index it by name.
    """

    __slots__ = ()

    _routes: tuple[Route, ...]
    _index: Mapping[str, Route]

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[Route]:
        return iter(self._routes)

    def __contains__(self, item: object) -> bool:
        """Membership by route name or by Route instance."""
        if isinstance(item, Route):
            return bool(self._index.get(item.name) == item)
        if isinstance(item, str):
            return item in self._index
        return False

    @overload
    def __getitem__(self, key: str) -> Route: ...

    @overload
    def __getitem__(self, key: int) -> Route: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[Route, ...]: ...

    def __getitem__(self, key: str | int | slice) -> Route | tuple[Route, ...]:
        """Look up by route name (primary) or by position (sequence sugar)."""
        if isinstance(key, str):
            try:
                return self._index[key]
            except KeyError:
                raise KeyError(f"no route named {key!r}") from None
        return self._routes[key]

    @property
    def routes(self) -> tuple[Route, ...]:
        """The routes, in catalog order."""
        return self._routes

    @property
    def names(self) -> tuple[str, ...]:
        """Route names, in catalog order."""
        return tuple(route.name for route in self._routes)

    def get(self, name: str) -> Route | None:
        """Return the route called ``name``, or ``None``."""
        return self._index.get(name)


class RegistryView(_RouteSequence):
    """A frozen, entitlement-scoped slice of a registry (plan §7.2).

    Tenants with identical entitlement profiles form a **cohort** sharing one
    view — and therefore one rendered candidate block and one provider-side
    prompt-cache prefix, so the ~90% caching discount is earned cohort-wide
    rather than per tenant. :attr:`view_hash`, never ``tenant_id``, is the single
    tenancy key used by the prompt cache (§4.6); real tenant ids appear only in
    audit records and opt-in per-tenant index keys (§13 ruling #15).

    Unlike :class:`Registry`, a view **may be empty**: an empty entitled set is a
    normal outcome that degrades to ``abstain(reason="no_eligible_routes")``
    with no LLM call (§13 ruling #3), not an error.
    """

    __slots__ = ("_index", "_registry_version", "_routes", "_view_hash")

    _view_hash: str
    _registry_version: str

    def __init__(
        self,
        routes: Iterable[Route],
        view_hash: str,
        registry_version: str,
    ) -> None:
        materialised = tuple(routes)
        object.__setattr__(self, "_routes", materialised)
        object.__setattr__(
            self,
            "_index",
            MappingProxyType({route.name: route for route in materialised}),
        )
        object.__setattr__(self, "_view_hash", view_hash)
        object.__setattr__(self, "_registry_version", registry_version)

    @property
    def view_hash(self) -> str:
        """12-hex cohort key; keys prompt-cache segment B (§4.6)."""
        return self._view_hash

    @property
    def registry_version(self) -> str:
        """The ``registry_version`` this view was derived from."""
        return self._registry_version

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RegistryView is frozen after construction")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RegistryView is frozen after construction")

    def __repr__(self) -> str:
        return (
            f"RegistryView(routes={len(self._routes)}, "
            f"view_hash={self._view_hash!r}, "
            f"registry_version={self._registry_version!r})"
        )


class Registry(_RouteSequence):
    """An immutable, content-addressed catalog of routes (plan §3.2).

    Validates that names are unique and that the catalog is non-empty, then
    freezes. Every composition operation (:meth:`merge`, :meth:`filter`,
    :meth:`__or__`) returns a new ``Registry``.

    Example::

        reg  = Registry([Route(name="refund", description="..."), ...])
        reg2 = reg | other                       # duplicate name -> RegistryError
        reg3 = reg.merge(other, on_conflict="override")
        sub  = reg.filter(lambda r: "billing" in r.tags)
        reg.version                              # -> "3fa9c21b04d1"
    """

    __slots__ = ("_content_hash", "_index", "_routes")

    _content_hash: str | None

    def __init__(self, routes: Iterable[Route]) -> None:
        materialised = tuple(routes)
        if not materialised:
            raise RegistryError(
                "Registry must contain at least one route; an empty catalog "
                "cannot produce a decision"
            )
        index: dict[str, Route] = {}
        for route in materialised:
            if not isinstance(route, Route):
                raise RegistryError(
                    f"Registry accepts Route instances, got {type(route).__name__}"
                )
            if route.name in index:
                raise RegistryError(f"duplicate route name {route.name!r} in registry")
            index[route.name] = route
        object.__setattr__(self, "_routes", materialised)
        object.__setattr__(self, "_index", MappingProxyType(index))
        object.__setattr__(self, "_content_hash", None)

    # -- immutability ------------------------------------------------------- #

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "Registry is frozen after construction; use merge()/filter()/| to "
            "derive a new registry (plan §3, decision (d))"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Registry is frozen after construction")

    # -- identity ----------------------------------------------------------- #

    @property
    def content_hash(self) -> str:
        """Full SHA-256 digest over the sorted route content hashes (plan §7.3).

        ``registry_hash = sha256(sorted(route_hashes))`` — sorted, so declaration
        order does not change catalog identity. Computed once and cached.
        """
        cached = self._content_hash
        if cached is None:
            cached = sha256_hex(
                canonical_json(sorted(route.content_hash for route in self._routes))
            )
            object.__setattr__(self, "_content_hash", cached)
        return cached

    @property
    def version(self) -> str:
        """``registry_version``: the 12-hex prefix of :attr:`content_hash` (§7.3).

        Stamped into every Decision, audit record and OTel span, and used as the
        key for the prompt-cache prefix (§4.6) and the shortlist index (§5.4) —
        so a route edit invalidates embeddings and cached prefixes *together*.
        """
        return self.content_hash[:_VERSION_PREFIX_LEN]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Registry):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        return hash(self.content_hash)

    def __repr__(self) -> str:
        return f"Registry(routes={len(self._routes)}, version={self.version!r})"

    # -- composition -------------------------------------------------------- #

    def filter(self, predicate: Callable[[Route], bool]) -> Registry:
        """Return a new registry of the routes satisfying ``predicate``.

        Raises :class:`~switchboard.errors.RegistryError` if nothing matches — a
        registry is non-empty by definition. To narrow a catalog per request
        (where "nothing matches" is a legitimate abstain, not an error), use
        :meth:`view` instead.
        """
        return Registry([route for route in self._routes if predicate(route)])

    def merge(
        self,
        other: Registry | Iterable[Route],
        on_conflict: Literal["error", "override", "keep"] = "error",
    ) -> Registry:
        """Union this registry with ``other``, returning a new registry.

        Args:
            other: a registry or any iterable of routes.
            on_conflict: what to do when a name exists in both — ``"error"``
                raises :class:`~switchboard.errors.RegistryError`, ``"override"``
                takes ``other``'s route, ``"keep"`` keeps this one.

        Overridden routes retain their original position, so merge order is
        deterministic and the content hash is stable across equivalent merges.
        """
        if on_conflict not in ("error", "override", "keep"):
            raise ConfigError(
                f"on_conflict must be 'error', 'override' or 'keep', "
                f"got {on_conflict!r}"
            )
        merged: dict[str, Route] = {route.name: route for route in self._routes}
        for route in other:
            if route.name in merged:
                if on_conflict == "error":
                    raise RegistryError(
                        f"duplicate route name {route.name!r} while merging "
                        f"registries; pass on_conflict='override' or 'keep' to "
                        f"resolve it"
                    )
                if on_conflict == "keep":
                    continue
            merged[route.name] = route
        return Registry(tuple(merged.values()))

    def __or__(self, other: Registry) -> Registry:
        """``reg | other`` — union; a duplicate name raises ``RegistryError`` (§3.2)."""
        if not isinstance(other, Registry):
            return NotImplemented
        return self.merge(other, "error")

    # -- tenancy ------------------------------------------------------------ #

    def view(self, ctx: RequestContext | None = None) -> RegistryView:
        """Return the entitlement-scoped :class:`RegistryView` for ``ctx``.

        **v0.1 behaviour is a documented passthrough** (§13 ruling #20, §11 week
        2): every route is returned regardless of ``Route.requires`` /
        ``Route.visibility`` / ``ctx.entitlements``, and :attr:`~RegistryView.view_hash`
        is derived from the registry hash plus an empty entitlement key, so all
        contexts share one cohort. The call site, the returned type and the hash
        derivation are final now; [v0.2] slots the real filter in behind this
        signature without touching any caller.

        Callers must **not** treat v0.1 ``view()`` as a security boundary. When
        the filter lands it enforces ``ctx.entitlements >= route.requires`` plus
        the ``visibility`` predicate, pre-LLM, so unauthorised names never enter
        the prompt (§7.1, §7.4).
        """
        del ctx  # [v0.2] entitlement filter — inert in v0.1 by design.
        registry_hash = self.content_hash
        return RegistryView(
            self._routes,
            _view_hash(registry_hash, _V01_ENTITLEMENT_KEY),
            registry_hash[:_VERSION_PREFIX_LEN],
        )
