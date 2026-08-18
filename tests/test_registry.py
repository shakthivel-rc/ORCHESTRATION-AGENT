"""Contract tests for :class:`switchboard.Registry` / ``RegistryView`` (plan §3.2, §7.2).

Three properties carry the whole design:

1. **Frozen** (§3, decision (d)) — composition returns a *new* object, so one
   ``Router`` is safe to share across threads and tasks (§2.5) and no cache key
   can move under a caller's feet.
2. **Content-addressed** — ``version`` is a pure function of route content and of
   nothing else (not declaration order, not construction path, not wall clock).
   It keys the prompt cache (§4.6), the shortlist index (§5.4) and every audit
   record (§8.2), so both directions matter: equal content must produce an equal
   version, and any content change must produce a different one (§7.3).
3. **Sequence-shaped** — ``Registry`` and ``RegistryView`` expose the same
   read-only surface, so a tenant view is a drop-in for the catalog.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from switchboard import Registry, RegistryView, RequestContext, Route
from switchboard.errors import ConfigError, RegistryError


class RefundArgs(BaseModel):
    order_id: str


def _r(name: str, description: str = "A route.", **kwargs: Any) -> Route:
    return Route(name=name, description=description, **kwargs)


@pytest.fixture
def reg() -> Registry:
    """Three routes, deliberately not in alphabetical order."""
    return Registry([_r("refund"), _r("account"), _r("track_order")])


# --------------------------------------------------------------------------- #
# Construction validation (plan §3.2)
# --------------------------------------------------------------------------- #


def test_duplicate_route_names_raise_registry_error() -> None:
    with pytest.raises(RegistryError, match="duplicate route name 'refund'"):
        Registry([_r("refund"), _r("track_order"), _r("refund", "A different one.")])


def test_duplicate_detection_is_by_name_not_by_content() -> None:
    """Names are wire-enum members (§4.4): two routes cannot share one enum slot

    even when everything else about them differs.
    """
    with pytest.raises(RegistryError, match="duplicate route name"):
        Registry([_r("refund", "Issue a refund."), _r("refund", "Check a refund.")])


def test_empty_registry_raises_registry_error() -> None:
    with pytest.raises(RegistryError, match="at least one route"):
        Registry([])


def test_empty_generator_also_raises() -> None:
    with pytest.raises(RegistryError, match="at least one route"):
        Registry(route for route in ())


def test_non_route_members_raise_registry_error() -> None:
    with pytest.raises(RegistryError, match="Route instances"):
        Registry([_r("refund"), {"name": "track", "description": "d"}])  # type: ignore[list-item]


def test_registry_accepts_any_iterable_and_materialises_it() -> None:
    reg = Registry(_r(name) for name in ("a", "b", "c"))
    assert reg.names == ("a", "b", "c")
    assert len(reg) == 3


# --------------------------------------------------------------------------- #
# Immutability (plan §3, decision (d))
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("attribute", ["_routes", "_index", "version", "anything_new"])
def test_registry_rejects_attribute_assignment(reg: Registry, attribute: str) -> None:
    with pytest.raises(AttributeError, match="frozen"):
        setattr(reg, attribute, "nope")


def test_registry_rejects_attribute_deletion(reg: Registry) -> None:
    with pytest.raises(AttributeError, match="frozen"):
        del reg._routes


def test_registry_has_no_instance_dict_to_smuggle_state_into(reg: Registry) -> None:
    """``__slots__`` is what makes the freeze real rather than advisory."""
    assert not hasattr(reg, "__dict__")


def test_composition_never_mutates_the_receiver(reg: Registry) -> None:
    before_names, before_version = reg.names, reg.version
    reg.merge(Registry([_r("extra")]))
    reg.filter(lambda route: route.name == "refund")
    reg | Registry([_r("other")])
    reg.view()
    assert reg.names == before_names
    assert reg.version == before_version


def test_internal_index_is_a_read_only_mapping(reg: Registry) -> None:
    with pytest.raises(TypeError):
        reg._index["injected"] = _r("injected")  # type: ignore[index]


# --------------------------------------------------------------------------- #
# version / content_hash determinism (plan §7.3)
# --------------------------------------------------------------------------- #


def test_version_is_twelve_lowercase_hex_chars(reg: Registry) -> None:
    assert len(reg.version) == 12
    assert all(char in "0123456789abcdef" for char in reg.version)
    assert reg.content_hash.startswith(reg.version)
    assert len(reg.content_hash) == 64


def test_version_ignores_declaration_order() -> None:
    """The same catalog typed in a different order is the *same* catalog — else

    a harmless source reshuffle would void every prompt cache and every index.
    """
    routes = [_r("refund"), _r("account"), _r("track_order"), _r("human_handoff")]
    forward = Registry(routes)
    backward = Registry(list(reversed(routes)))
    shuffled = Registry([routes[2], routes[0], routes[3], routes[1]])
    assert forward.version == backward.version == shuffled.version
    assert forward.content_hash == backward.content_hash == shuffled.content_hash


def test_version_is_stable_across_separately_constructed_equal_routes() -> None:
    """Content addressing must survive a process boundary, so it may depend only

    on field values — never on object identity.
    """
    a = Registry([_r("refund", "Issue a refund.", tags=frozenset({"billing"}))])
    b = Registry([_r("refund", "Issue a refund.", tags=frozenset({"billing"}))])
    assert a.version == b.version


def test_version_is_stable_across_repeated_access(reg: Registry) -> None:
    assert reg.version == reg.version == reg.version


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("description", "A completely different description."),
        ("args_model", RefundArgs),
        ("examples", ("give me my money back",)),
        ("tags", frozenset({"billing"})),
        ("requires", frozenset({"billing"})),
        ("metadata", {"tier": "gold"}),
    ],
)
def test_version_changes_when_any_route_changes(reg: Registry, field: str, value: Any) -> None:
    edited = Registry([r.model_copy(update={field: value}) if r.name == "refund" else r for r in reg])
    assert edited.version != reg.version


def test_version_changes_when_a_route_is_added_or_removed(reg: Registry) -> None:
    assert Registry([*reg, _r("extra")]).version != reg.version
    assert Registry([r for r in reg if r.name != "account"]).version != reg.version


def test_version_changes_when_a_route_is_renamed(reg: Registry) -> None:
    renamed = Registry([r.model_copy(update={"name": "refunds"}) if r.name == "refund" else r for r in reg])
    assert renamed.version != reg.version


def test_equality_and_hashing_follow_content(reg: Registry) -> None:
    same = Registry(list(reversed(list(reg))))
    other = Registry([*reg, _r("extra")])
    assert reg == same
    assert hash(reg) == hash(same)
    assert reg != other
    assert len({reg, same, other}) == 2


def test_equality_against_a_foreign_type_is_not_an_error(reg: Registry) -> None:
    assert reg != "not a registry"
    assert (reg == 42) is False


# --------------------------------------------------------------------------- #
# merge (plan §3.2)
# --------------------------------------------------------------------------- #


@pytest.fixture
def left() -> Registry:
    return Registry([_r("a", "Left A."), _r("b", "Left B.")])


@pytest.fixture
def right() -> Registry:
    return Registry([_r("b", "Right B."), _r("c", "Right C.")])


def test_merge_on_conflict_error_is_the_default(left: Registry, right: Registry) -> None:
    with pytest.raises(RegistryError, match="duplicate route name 'b' while merging"):
        left.merge(right)


def test_merge_error_message_points_at_the_resolution(left: Registry, right: Registry) -> None:
    with pytest.raises(RegistryError, match="on_conflict='override' or 'keep'"):
        left.merge(right, "error")


def test_merge_on_conflict_override_takes_the_other_route(left: Registry, right: Registry) -> None:
    merged = left.merge(right, on_conflict="override")
    assert merged.names == ("a", "b", "c")
    assert merged["b"].description == "Right B."


def test_merge_on_conflict_keep_retains_the_receivers_route(left: Registry, right: Registry) -> None:
    merged = left.merge(right, on_conflict="keep")
    assert merged.names == ("a", "b", "c")
    assert merged["b"].description == "Left B."


def test_merge_accepts_on_conflict_positionally_and_by_keyword(
    left: Registry, right: Registry
) -> None:
    """§3.2 documents the keyword spelling; the positional one is the natural

    call. Both must work, and must agree.
    """
    assert left.merge(right, "override").version == left.merge(right, on_conflict="override").version


def test_merge_overridden_routes_keep_their_original_position(
    left: Registry, right: Registry
) -> None:
    """Position determines the rendered card order (§5.6) and therefore the

    cached prefix bytes, so merge order has to be deterministic.
    """
    assert left.merge(right, "override").names == ("a", "b", "c")
    assert right.merge(left, "override").names == ("b", "c", "a")


def test_merge_accepts_a_bare_iterable_of_routes(left: Registry) -> None:
    merged = left.merge([_r("c"), _r("d")])
    assert merged.names == ("a", "b", "c", "d")


def test_merge_returns_a_new_registry_leaving_both_operands_alone(
    left: Registry, right: Registry
) -> None:
    left_version, right_version = left.version, right.version
    merged = left.merge(right, "override")
    assert merged is not left and merged is not right
    assert (left.version, right.version) == (left_version, right_version)


def test_merge_of_disjoint_registries_is_commutative_in_content(left: Registry) -> None:
    """Order changes the *sequence* but never the content hash (§7.3)."""
    other = Registry([_r("y"), _r("z")])
    assert left.merge(other).version == other.merge(left).version
    assert left.merge(other).names != other.merge(left).names


def test_merge_with_an_identical_registry_is_a_no_op_on_the_version(left: Registry) -> None:
    assert left.merge(Registry(list(left)), "override").version == left.version


@pytest.mark.parametrize("bad", ["ERROR", "replace", "", None, True])
def test_merge_rejects_an_unknown_on_conflict_strategy(
    left: Registry, right: Registry, bad: Any
) -> None:
    with pytest.raises(ConfigError, match="on_conflict must be"):
        left.merge(right, bad)


# --------------------------------------------------------------------------- #
# | operator (plan §3.2)
# --------------------------------------------------------------------------- #


def test_or_operator_unions_disjoint_registries(left: Registry) -> None:
    union = left | Registry([_r("c")])
    assert isinstance(union, Registry)
    assert union.names == ("a", "b", "c")


def test_or_operator_raises_on_a_duplicate_name(left: Registry, right: Registry) -> None:
    with pytest.raises(RegistryError, match="duplicate route name 'b'"):
        left | right


def test_or_operator_matches_merge_with_the_default_strategy(left: Registry) -> None:
    other = Registry([_r("c")])
    assert (left | other).version == left.merge(other).version


def test_or_operator_returns_not_implemented_for_foreign_operands(left: Registry) -> None:
    assert left.__or__([_r("c")]) is NotImplemented  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        left | [_r("c")]  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# filter (plan §3.2)
# --------------------------------------------------------------------------- #


def test_filter_returns_the_matching_subset() -> None:
    reg = Registry(
        [
            _r("refund", tags=frozenset({"billing"})),
            _r("invoice", tags=frozenset({"billing"})),
            _r("track", tags=frozenset({"shipping"})),
        ]
    )
    billing = reg.filter(lambda route: "billing" in route.tags)
    assert billing.names == ("refund", "invoice")
    assert billing.version != reg.version


def test_filter_preserves_order(reg: Registry) -> None:
    assert reg.filter(lambda route: True).names == reg.names


def test_filter_matching_everything_reproduces_the_version(reg: Registry) -> None:
    assert reg.filter(lambda route: True).version == reg.version


def test_filter_to_nothing_raises_because_a_registry_is_non_empty_by_definition(
    reg: Registry,
) -> None:
    """The docstring's distinction: an empty *filter* is a caller mistake, while

    an empty *view* is a normal request-time outcome that abstains (§13 ruling #3).
    """
    with pytest.raises(RegistryError, match="at least one route"):
        reg.filter(lambda route: False)
    assert len(reg.view(RequestContext())) == 3


def test_filter_returns_a_new_object(reg: Registry) -> None:
    assert reg.filter(lambda route: True) is not reg


# --------------------------------------------------------------------------- #
# Sequence surface — shared by Registry and RegistryView
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["registry", "view"])
def surface(request: pytest.FixtureRequest, reg: Registry) -> Any:
    """Run the sequence contract against both implementations of it."""
    return reg if request.param == "registry" else reg.view()


def test_len_reports_the_route_count(surface: Any) -> None:
    assert len(surface) == 3


def test_iteration_yields_routes_in_catalog_order(surface: Any) -> None:
    assert [route.name for route in surface] == ["refund", "account", "track_order"]


def test_iteration_is_repeatable(surface: Any) -> None:
    assert list(surface) == list(surface)


def test_names_and_routes_properties(surface: Any) -> None:
    assert surface.names == ("refund", "account", "track_order")
    assert isinstance(surface.routes, tuple)
    assert surface.routes[0].name == "refund"


def test_membership_by_name(surface: Any) -> None:
    assert "refund" in surface
    assert "nonexistent" not in surface


def test_membership_by_route_instance(surface: Any) -> None:
    assert _r("refund") in surface
    assert _r("refund", "A different description.") not in surface
    assert _r("nonexistent") not in surface


def test_membership_of_a_foreign_type_is_false_not_an_error(surface: Any) -> None:
    assert 42 not in surface
    assert None not in surface
    assert object() not in surface


def test_getitem_by_name(surface: Any) -> None:
    assert surface["refund"].description == "A route."


def test_getitem_by_unknown_name_raises_key_error(surface: Any) -> None:
    with pytest.raises(KeyError, match="no route named 'nope'"):
        surface["nope"]


def test_getitem_by_index_and_slice(surface: Any) -> None:
    assert surface[0].name == "refund"
    assert surface[-1].name == "track_order"
    assert [route.name for route in surface[0:2]] == ["refund", "account"]


def test_getitem_by_out_of_range_index_raises_index_error(surface: Any) -> None:
    with pytest.raises(IndexError):
        surface[99]


def test_get_returns_none_for_a_missing_name(surface: Any) -> None:
    assert surface.get("refund") is not None
    assert surface.get("nope") is None


def test_repr_names_the_type_and_the_size(reg: Registry) -> None:
    assert repr(reg) == f"Registry(routes=3, version={reg.version!r})"
    assert repr(reg.view()).startswith("RegistryView(routes=3, view_hash=")


# --------------------------------------------------------------------------- #
# view() — v0.1 passthrough with a real, deterministic hash (plan §7.2)
# --------------------------------------------------------------------------- #


def test_view_returns_a_registry_view_of_every_route(reg: Registry) -> None:
    view = reg.view(RequestContext(tenant_id="acme"))
    assert isinstance(view, RegistryView)
    assert view.names == reg.names


def test_view_is_a_documented_passthrough_in_v01(reg: Registry) -> None:
    """§13 ruling #20: entitlement filtering is [v0.2]. In v0.1 ``requires`` and

    ``visibility`` are accepted and stored but do NOT filter, so a test asserting
    the opposite would be testing a security boundary that does not exist yet.
    """
    gated = Registry(
        [
            _r("open"),
            _r("gated", requires=frozenset({"billing"})),
            _r("hidden", visibility=lambda ctx: False),
        ]
    )
    view = gated.view(RequestContext(entitlements=frozenset()))
    assert view.names == ("open", "gated", "hidden")


def test_view_accepts_no_context_at_all(reg: Registry) -> None:
    assert reg.view().names == reg.names
    assert reg.view(None).view_hash == reg.view(RequestContext()).view_hash


def test_view_hash_is_twelve_hex_chars(reg: Registry) -> None:
    view_hash = reg.view().view_hash
    assert len(view_hash) == 12
    assert all(char in "0123456789abcdef" for char in view_hash)


def test_view_hash_is_deterministic_for_one_registry(reg: Registry) -> None:
    assert reg.view().view_hash == reg.view().view_hash


def test_view_hash_is_one_cohort_across_contexts_in_v01(reg: Registry) -> None:
    """v0.1 renders a byte-identical candidate block for every tenant, so every

    tenant must share one prompt-cache prefix key — fragmenting it per tenant
    would forfeit the caching discount for nothing (§4.6, §7.2).
    """
    hashes = {
        reg.view(ctx).view_hash
        for ctx in (
            RequestContext(),
            RequestContext(tenant_id="acme"),
            RequestContext(tenant_id="globex", entitlements=frozenset({"billing"})),
            RequestContext(entitlement_key="cohort-7"),
        )
    }
    assert len(hashes) == 1


def test_view_hash_tracks_registry_content(reg: Registry) -> None:
    changed = Registry([*reg, _r("extra")])
    assert changed.view().view_hash != reg.view().view_hash


def test_view_hash_is_not_the_registry_version(reg: Registry) -> None:
    """They key different things — the candidate block versus the whole catalog —

    so they must be distinguishable strings, not accidental aliases.
    """
    assert reg.view().view_hash != reg.version


def test_view_carries_the_registry_version_it_came_from(reg: Registry) -> None:
    assert reg.view().registry_version == reg.version


def test_view_is_frozen(reg: Registry) -> None:
    view = reg.view()
    with pytest.raises(AttributeError, match="frozen"):
        view._routes = ()
    with pytest.raises(AttributeError, match="frozen"):
        del view._view_hash
    assert not hasattr(view, "__dict__")


def test_a_view_may_legitimately_be_empty() -> None:
    """Unlike a Registry: an empty entitled set degrades to abstain, not an error

    (§13 ruling #3). Constructed directly because v0.1 ``view()`` never filters.
    """
    reg = Registry([_r("only")])
    empty = RegistryView([], view_hash="0" * 12, registry_version=reg.version)
    assert len(empty) == 0
    assert empty.names == ()
    assert list(empty) == []
    assert "only" not in empty
