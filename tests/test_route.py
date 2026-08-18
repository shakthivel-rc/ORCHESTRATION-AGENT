"""Contract tests for :class:`switchboard.Route` (plan §3.1, §5.6, §7.3).

Route is the atom every cache key in the system is built from: its
``content_hash`` rolls up into ``registry_version``, which keys the provider
prompt cache (§4.6), the shortlist index (§5.4) and every audit record (§8.2).
So the properties under test here are not cosmetic — a hash that moves when it
shouldn't costs money (cache misses, re-embeds), and a hash that *doesn't* move
when it should serves a stale prompt against a changed catalog (§7.3).

The other half is rendering: :meth:`Route.card` is the ONE function that renders
a route for a prompt (§5.6), so its exact bytes are part of the contract —
distillation examples harvested from audit records must be format-identical to
what the live loop sent.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from switchboard import Route
from switchboard.core.route import ROUTE_NAME_PATTERN
from switchboard.errors import ConfigError, RegistryError


class RefundArgs(BaseModel):
    """Two fields, one required and one optional — drives ``args_summary``."""

    order_id: str
    reason: str | None = None


class OtherArgs(BaseModel):
    """A genuinely different schema, for hash-instability assertions."""

    ticket_id: int
    priority: str = "normal"


def _route(**overrides: Any) -> Route:
    """A canonical route, with per-test field overrides."""
    base: dict[str, Any] = {
        "name": "refund",
        "description": "Issue or check a refund for an order.",
        "args_model": RefundArgs,
        "examples": ("I want my money back for order 123",),
        "tags": frozenset({"billing", "orders"}),
        "requires": frozenset({"billing"}),
        "metadata": {"owner": "payments-team"},
    }
    base.update(overrides)
    return Route(**base)


# --------------------------------------------------------------------------- #
# Name grammar (plan §3.1): ^[a-z][a-z0-9_\-.:]{0,63}\Z
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "refund",
        "track_order",
        "billing.report",
        "orders:create",
        "human-handoff",
        "a0",
        "a" + "b9_-.:" * 10 + "xyz",  # 64 chars exactly
    ],
)
def test_name_accepts_the_documented_alphabet(name: str) -> None:
    assert len(name) <= 64
    assert Route(name=name, description="d").name == name


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("", "empty"),
        ("Refund", "uppercase start"),
        ("reFund", "uppercase inside"),
        ("1refund", "digit start"),
        ("_refund", "underscore start"),
        ("-refund", "hyphen start"),
        (".refund", "dot start"),
        (":refund", "colon start"),
        ("refund order", "space"),
        ("refund!", "punctuation outside the alphabet"),
        ("refund/order", "slash"),
        ("réfund", "non-ascii"),
        ("refund\n", "trailing newline would split the route card"),
        ("\nrefund", "leading newline"),
        ("refund\ntrack", "embedded newline"),
        ("a" * 65, "65 chars — one over the limit"),
    ],
)
def test_name_rejects_everything_else(name: str, why: str) -> None:
    with pytest.raises(RegistryError) as excinfo:
        Route(name=name, description="d")
    assert repr(name) in str(excinfo.value), why


def test_name_error_is_a_config_error_not_a_pydantic_error() -> None:
    """§3.8: a malformed name is caller *configuration*, so it raises the

    library's own ``RegistryError`` (a ``ConfigError``) rather than surfacing as
    a generic ``ValidationError`` the caller would have to introspect.
    """
    with pytest.raises(RegistryError) as excinfo:
        Route(name="BAD", description="d")
    assert isinstance(excinfo.value, ConfigError)
    assert not isinstance(excinfo.value, ValidationError)


def test_name_pattern_is_anchored_at_both_ends() -> None:
    """The exported pattern must behave the same as the validator that uses it."""
    assert ROUTE_NAME_PATTERN.match("refund")
    assert not ROUTE_NAME_PATTERN.match("refund\n")
    assert not ROUTE_NAME_PATTERN.match("Refund")


# --------------------------------------------------------------------------- #
# args_model type rejection (plan §3.8: "bad args_model")
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [
        dict,
        list,
        str,
        42,
        "RefundArgs",
        RefundArgs(order_id="1"),  # an *instance*, not the class
        object,
        lambda: None,
    ],
)
def test_args_model_rejects_non_basemodel_subclasses(bad: Any) -> None:
    with pytest.raises(RegistryError, match="args_model must be a pydantic BaseModel"):
        Route(name="refund", description="d", args_model=bad)


def test_args_model_accepts_none_and_basemodel_subclasses() -> None:
    assert Route(name="a", description="d").args_model is None
    assert Route(name="a", description="d", args_model=None).args_model is None
    assert Route(name="a", description="d", args_model=RefundArgs).args_model is RefundArgs


# --------------------------------------------------------------------------- #
# Immutability (plan §3.1: frozen)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "other"),
        ("description", "other"),
        ("args_model", OtherArgs),
        ("examples", ()),
        ("tags", frozenset()),
        ("pinned", True),
        ("metadata", {}),
    ],
)
def test_route_is_frozen(field: str, value: Any) -> None:
    route = _route()
    with pytest.raises(ValidationError):
        setattr(route, field, value)


def test_route_compares_by_value_not_identity() -> None:
    """``Registry.__contains__`` tests ``self._index[name] == route``, so value

    equality — not identity — is the membership contract.
    """
    assert _route() == _route()
    assert _route() is not _route()
    assert _route() != _route(description="different")


def test_route_is_not_hashable_so_identity_flows_through_content_hash() -> None:
    """§3.1 gives Route a mutable ``metadata: dict`` field, which makes Pydantic's

    generated ``__hash__`` unusable. That is fine and deliberate: set-membership
    and cache keying go through :attr:`Route.content_hash`, never through
    ``hash(route)``. Asserted so nobody "fixes" hashability by quietly dropping
    ``metadata`` out of the model.
    """
    with pytest.raises(TypeError):
        hash(_route())
    assert isinstance(_route().content_hash, str)


def test_model_copy_produces_an_independent_edited_route() -> None:
    route = _route()
    edited = route.model_copy(update={"description": "Something else entirely."})
    assert route.description == "Issue or check a refund for an order."
    assert edited.description == "Something else entirely."


# --------------------------------------------------------------------------- #
# content_hash stability (plan §3.1, §7.3)
# --------------------------------------------------------------------------- #


def test_content_hash_is_a_full_sha256_hex_digest() -> None:
    digest = _route().content_hash
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_content_hash_is_stable_across_repeated_access() -> None:
    route = _route()
    assert route.content_hash == route.content_hash


def test_content_hash_ignores_keyword_construction_order() -> None:
    """Two routes with identical content built with kwargs in a different order

    must be indistinguishable — otherwise ``registry_version`` would depend on
    how the caller happened to type the constructor call.
    """
    a = Route(
        name="refund",
        description="Issue or check a refund.",
        tags=frozenset({"billing"}),
        examples=("give me my money back",),
        args_model=RefundArgs,
        metadata={"owner": "payments", "cost_class": "low"},
    )
    b = Route(
        metadata={"cost_class": "low", "owner": "payments"},
        args_model=RefundArgs,
        examples=("give me my money back",),
        tags=frozenset({"billing"}),
        description="Issue or check a refund.",
        name="refund",
    )
    assert a.content_hash == b.content_hash


def test_content_hash_ignores_frozenset_iteration_order() -> None:
    """``tags``/``requires`` are unordered sets; two spellings of the same set

    (whose Python iteration order may differ) must hash identically.
    """
    a = _route(tags=frozenset({"billing", "orders", "payments"}))
    b = _route(tags=frozenset(["payments", "billing", "orders"]))
    assert list(a.tags) == list(a.tags)  # sanity: frozensets are what we think
    assert a.content_hash == b.content_hash


def test_content_hash_ignores_metadata_key_order() -> None:
    a = _route(metadata={"owner": "payments", "tier": "gold"})
    b = _route(metadata={"tier": "gold", "owner": "payments"})
    assert a.content_hash == b.content_hash


# --------------------------------------------------------------------------- #
# content_hash instability — the half that prevents stale caches (§7.3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "refunds"),
        ("description", "Issue or check a refund for an order. Not for tracking."),
        ("args_model", OtherArgs),
        ("args_model", None),
        ("examples", ("I want my money back for order 123", "refund please")),
        ("examples", ()),
        ("tags", frozenset({"billing"})),
        ("tags", frozenset({"billing", "orders", "payments"})),
        ("requires", frozenset()),
        ("metadata", {"owner": "billing-team"}),
    ],
)
def test_content_hash_changes_when_hashed_content_changes(field: str, value: Any) -> None:
    assert _route().content_hash != _route(**{field: value}).content_hash


def test_content_hash_tracks_example_order_because_rendering_does() -> None:
    """``examples`` is an ordered tuple: ``card()`` and ``embed_text`` render it

    in declaration order, so a reorder genuinely changes the prompt bytes and the
    embedded text and MUST bump the hash (§7.3). Sorting it before hashing would
    hide the change behind an unchanged ``registry_version``.
    """
    a = _route(examples=("first", "second"))
    b = _route(examples=("second", "first"))
    assert a.card() != b.card()
    assert a.embed_text != b.embed_text
    assert a.content_hash != b.content_hash


def test_content_hash_is_recomputed_after_model_copy() -> None:
    """The documented reason the hash is never memoised on the instance:

    ``model_copy(update=...)`` is the idiomatic way to edit a frozen model, and a
    cached hash would survive it, silently pinning the prompt cache and the
    shortlist index to a registry that no longer exists.
    """
    route = _route()
    before = route.content_hash
    edited = route.model_copy(update={"description": "A completely new description."})
    assert edited.content_hash != before
    assert route.content_hash == before  # the original is untouched


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pinned", True),
        ("clarify_label", "a refund"),
        ("group", "billing"),
    ],
)
def test_routing_behaviour_fields_do_not_move_the_hash(field: str, value: Any) -> None:
    """``pinned``/``clarify_label``/``group`` are not in the plan's hashed tuple:

    they change *how the router uses* a route, not the prompt-facing content the
    cache and the embeddings are keyed on.
    """
    assert _route().content_hash == _route(**{field: value}).content_hash


# --------------------------------------------------------------------------- #
# visibility is excluded from the hash (plan §3.1, explicitly)
# --------------------------------------------------------------------------- #


def test_visibility_is_excluded_from_content_hash() -> None:
    """A lambda is not serialisable, and two deployments sharing a registry must

    still agree on the hash — so ``visibility`` cannot participate.
    """
    plain = _route()
    with_pred = _route(visibility=lambda ctx: ctx.has("billing"))
    other_pred = _route(visibility=lambda ctx: False)

    assert with_pred.visibility is not None
    assert other_pred.visibility is not None
    assert with_pred.visibility is not other_pred.visibility
    assert plain.content_hash == with_pred.content_hash == other_pred.content_hash


def test_visibility_predicate_is_callable_with_a_request_context() -> None:
    from switchboard import RequestContext

    route = _route(visibility=lambda ctx: ctx.has("billing"))
    assert route.visibility is not None
    assert route.visibility(RequestContext(entitlements=frozenset({"billing"}))) is True
    assert route.visibility(RequestContext()) is False


# --------------------------------------------------------------------------- #
# metadata: only its JSON-serialisable subset is hashed
# --------------------------------------------------------------------------- #


def handler_a() -> str:
    return "a"


def handler_b() -> str:
    return "b"


def test_unserialisable_metadata_is_dropped_from_the_hash() -> None:
    """A handler reference lives in ``metadata`` (§3.1 decision (a)). Its repr

    embeds a memory address, so hashing it would make ``registry_version``
    process-dependent — and swapping the handler must not look like a content
    change.
    """
    bare = Route(name="refund", description="d")
    with_handler = Route(name="refund", description="d", metadata={"handler": handler_a})
    swapped = Route(name="refund", description="d", metadata={"handler": handler_b})

    assert with_handler.metadata["handler"] is handler_a
    assert bare.content_hash == with_handler.content_hash == swapped.content_hash


def test_serialisable_metadata_still_participates_in_the_hash() -> None:
    a = Route(name="refund", description="d", metadata={"handler": handler_a, "tier": 1})
    b = Route(name="refund", description="d", metadata={"handler": handler_b, "tier": 2})
    assert a.content_hash != b.content_hash


def test_metadata_hashing_canonicalises_nested_structures() -> None:
    """Nested mappings are canonicalised (key order irrelevant) but nested lists

    keep their order, because a list is ordered data.
    """
    a = Route(
        name="refund",
        description="d",
        metadata={"limits": {"max": 5, "min": 1}, "regions": ["eu", "us"]},
    )
    same = Route(
        name="refund",
        description="d",
        metadata={"regions": ["eu", "us"], "limits": {"min": 1, "max": 5}},
    )
    reordered_list = Route(
        name="refund",
        description="d",
        metadata={"limits": {"max": 5, "min": 1}, "regions": ["us", "eu"]},
    )
    assert a.content_hash == same.content_hash
    assert a.content_hash != reordered_list.content_hash


# --------------------------------------------------------------------------- #
# embed_text (plan §7.3)
# --------------------------------------------------------------------------- #


def test_embed_text_is_name_plus_description_plus_examples() -> None:
    route = Route(
        name="refund",
        description="Issue a refund.",
        examples=("give me my money back", "refund order 12"),
    )
    assert route.embed_text == "refund\nIssue a refund.\ngive me my money back\nrefund order 12"


def test_embed_text_normalises_whitespace() -> None:
    """Newlines and runs of spaces in a description must not vary the embedded

    bytes for semantically identical content.
    """
    messy = Route(
        name="refund",
        description="Issue   a refund.\n  Use for money-back requests.",
        examples=("give me\tmy money   back",),
    )
    clean = Route(
        name="refund",
        description="Issue a refund. Use for money-back requests.",
        examples=("give me my money back",),
    )
    assert messy.embed_text == clean.embed_text


def test_embed_text_excludes_args_model_so_schema_edits_do_not_force_a_reembed() -> None:
    """§7.3: an args-schema-only change bumps ``content_hash`` (the prompt cache

    does contain the schema) without invalidating any embedding.
    """
    a = _route(args_model=RefundArgs)
    b = _route(args_model=OtherArgs)
    assert a.embed_text == b.embed_text
    assert a.content_hash != b.content_hash


def test_embed_text_excludes_tags_and_metadata() -> None:
    assert _route().embed_text == _route(tags=frozenset(), metadata={}).embed_text


# --------------------------------------------------------------------------- #
# args_summary + card() rendering (plan §5.6)
# --------------------------------------------------------------------------- #


def test_args_summary_reports_name_type_and_requiredness_in_declaration_order() -> None:
    route = Route(name="refund", description="d", args_model=RefundArgs)
    assert route.args_summary() == "order_id (string, required), reason (string, optional)"


def test_args_summary_unwraps_optionals_and_names_container_types() -> None:
    class Complex(BaseModel):
        ids: list[str]
        count: int
        ratio: float
        flag: bool
        note: str | None = None

    route = Route(name="c", description="d", args_model=Complex)
    assert route.args_summary() == (
        "ids (array[string], required), count (integer, required), "
        "ratio (number, required), flag (boolean, required), note (string, optional)"
    )


def test_args_summary_is_none_without_an_args_model_or_with_an_empty_one() -> None:
    class Empty(BaseModel):
        pass

    assert Route(name="a", description="d").args_summary() == "none"
    assert Route(name="a", description="d", args_model=Empty).args_summary() == "none"


def test_card_renders_the_canonical_block() -> None:
    route = Route(
        name="refund",
        description="Issue or check a refund for an order.",
        args_model=RefundArgs,
        examples=("I want my money back for order 123",),
        tags=frozenset({"payments", "billing"}),
    )
    assert route.card() == (
        "- refund: Issue or check a refund for an order.\n"
        "  tags: billing, payments\n"
        "  args: order_id (string, required), reason (string, optional)\n"
        '  e.g. "I want my money back for order 123"'
    )


def test_card_sorts_tags_so_the_cached_prefix_is_byte_stable() -> None:
    a = Route(name="r", description="d", tags=frozenset({"b", "a", "c"}))
    b = Route(name="r", description="d", tags=frozenset(["c", "b", "a"]))
    assert a.card() == b.card()
    assert "  tags: a, b, c" in a.card()


def test_card_omits_lines_for_absent_content() -> None:
    assert Route(name="r", description="Plain route.").card() == "- r: Plain route."


def test_card_include_args_false_drops_only_the_args_line() -> None:
    """§4.6 segment C: the shortlist pointer block wants names and one-liners

    only — tokens are the scarce resource there.
    """
    route = Route(
        name="refund",
        description="Issue a refund.",
        args_model=RefundArgs,
        tags=frozenset({"billing"}),
        examples=("refund me",),
    )
    lines = route.card(include_args=False).splitlines()
    assert lines == [
        "- refund: Issue a refund.",
        "  tags: billing",
        '  e.g. "refund me"',
    ]
    assert "args:" in route.card(include_args=True)


def test_card_normalises_whitespace_so_a_route_never_spans_extra_lines() -> None:
    """The candidate block is line-oriented: one route, one card, no stray

    newlines from a multi-line description.
    """
    route = Route(
        name="refund",
        description="Issue a refund.\n\nUse when   the customer wants money back.",
        examples=("give me\nmy money back",),
    )
    card = route.card()
    assert card.splitlines() == [
        "- refund: Issue a refund. Use when the customer wants money back.",
        '  e.g. "give me my money back"',
    ]


def test_card_keeps_examples_in_declaration_order() -> None:
    route = Route(name="r", description="d", examples=("zebra", "apple"))
    assert route.card().splitlines()[1:] == ['  e.g. "zebra"', '  e.g. "apple"']


def test_card_is_deterministic_across_equal_routes() -> None:
    assert _route().card() == _route().card()
