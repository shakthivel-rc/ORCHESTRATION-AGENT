"""``Route`` — one addressable destination in the catalog (plan §3.1).

A Route carries **no handler** (plan §3, decision (a)): switchboard decides, it
never executes. A handler reference belongs in :attr:`Route.metadata`. This is
also the security posture in §7.4 — "decides, never executes" means a routing
compromise cannot become code execution.

The description is prompt-facing and is the single biggest lever on quality:
write it as "description-as-prompt" — what it does, when to use it, and when
*not* to. The not-clause is what separates confusable pairs, which is the error
class shortlisting cannot fix (plan §5.7).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable  # noqa: TC003 - runtime: Pydantic field annotation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from switchboard.core.audit import canonical_json, sha256_hex
from switchboard.core.context import RequestContext  # noqa: TC001 - runtime: field annotation
from switchboard.errors import RegistryError

# NOTE on the two noqa'd imports above: `Callable` and `RequestContext` both appear
# in the `visibility` field annotation, which Pydantic resolves at class-construction
# time. Deferring either into a TYPE_CHECKING block breaks Route's schema build at
# import time, so ruff's type-only-import suggestion must not be applied here.

__all__ = ["ROUTE_NAME_PATTERN", "Route"]

ROUTE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_\-.:]{0,63}\Z")
"""Plan §3.1. Route names are wire-enum members (§4.4) and OTel attribute values,
so they are constrained to a lowercase, tokenizer-friendly, JSON-safe alphabet.

The grammar is the plan's ``^[a-z][a-z0-9_\\-.:]{0,63}$`` with one correction:
Python's ``$`` also matches *before* a single trailing newline, so ``"refund\\n"``
would satisfy the plan's literal regex. A newline inside a route name splits the
one-line route card in two (breaking the line-oriented candidate block of §4.6)
and is not a legal OTel attribute value, so the anchor is ``\\Z`` — end of string,
no exceptions."""


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-serialisable subset of ``metadata``.

    ``metadata`` is documented as opaque passthrough and routinely holds handler
    references and other live objects. Those cannot participate in a stable
    content hash (their repr embeds a memory address), so they are dropped
    rather than stringified — a handler swap must not look like a content change.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        try:
            json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            continue
        safe[key] = value
    return safe


def _json_type_name(schema: dict[str, Any]) -> str:
    """Best-effort human-readable type name for one JSON-schema property."""
    if "type" in schema:
        type_name = schema["type"]
        if type_name == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                return f"array[{_json_type_name(items)}]"
            return "array"
        return str(type_name)
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "enum" in schema:
        return "enum"
    for key in ("anyOf", "oneOf", "allOf"):
        options = schema.get(key)
        if isinstance(options, list):
            names = [
                _json_type_name(opt)
                for opt in options
                if isinstance(opt, dict) and opt.get("type") != "null"
            ]
            if names:
                return "|".join(dict.fromkeys(names))
    return "any"


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Route cards are line-oriented and go into a cached prompt prefix; a stray
    newline in a description would corrupt the layout and, worse, vary the
    rendered bytes for semantically identical content.
    """
    return " ".join(text.split())


class Route(BaseModel):
    """One route in a :class:`~switchboard.core.registry.Registry` (plan §3.1).

    Frozen and hash-stable: :attr:`content_hash` feeds ``registry_version``,
    which keys the prompt cache, the shortlist index and every audit record
    (plan §7.3), so a Route must not change identity under the caller's feet.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    """Unique per Registry; a wire-enum member (§4.4). Must match
    :data:`ROUTE_NAME_PATTERN`."""
    description: str
    """1-3 sentences, prompt-facing. Write as "description-as-prompt" (§5.7)."""
    args_model: type[BaseModel] | None = None
    """Args extracted in the *same* LLM call; ``None`` means the route takes none."""
    examples: tuple[str, ...] = ()
    """Few-shot utterances. Also indexed by the shortlister at x2.0 weight —
    utterance-shaped text matches utterance-shaped queries (§5.2)."""
    tags: frozenset[str] = frozenset()
    """Shortlist field boost, grouping, coarse classes."""
    pinned: bool = False
    """Always appended to candidates regardless of score (§5.5), so the fallback
    path can never be retrieved out of existence."""
    requires: frozenset[str] = frozenset()
    """Declarative entitlement: ``ctx.entitlements >= requires``. Hashable, so it
    contributes to a deterministic ``entitlement_key`` for view caching. [v0.2]"""
    visibility: Callable[[RequestContext], bool] | None = None
    """Escape-hatch predicate; sync, pure, O(microseconds) — it runs over every
    route on every request. Opts the route out of view caching unless the caller
    supplies ``RequestContext.entitlement_key`` (§7.1). [v0.2]"""
    clarify_label: str | None = None
    """Human phrase for templated clarify questions; defaults to the
    description's first clause (§6.5). [v0.2]"""
    group: str | None = None
    """Hierarchical routing group (§5.8). [v0.5]"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Opaque passthrough — handler ref, owner, cost class. Only its
    JSON-serialisable subset participates in :attr:`content_hash`."""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Enforce plan §3.1's name grammar, as a ``RegistryError`` (§3.8)."""
        if not ROUTE_NAME_PATTERN.match(value):
            raise RegistryError(
                f"invalid route name {value!r}: names must match "
                f"{ROUTE_NAME_PATTERN.pattern} (lowercase start, then "
                f"a-z 0-9 _ - . : up to 64 chars)"
            )
        return value

    @field_validator("args_model", mode="before")
    @classmethod
    def _validate_args_model(cls, value: Any) -> Any:
        """Reject anything that is not a ``BaseModel`` subclass (§3.8 "bad args_model").

        Caught here rather than at first use because the args model is what the
        wire schema is built from (§4.4); a dataclass or a bare dict would fail
        deep inside the provider call instead of at construction.
        """
        if value is None:
            return None
        if isinstance(value, type) and issubclass(value, BaseModel):
            return value
        raise RegistryError(
            f"args_model must be a pydantic BaseModel subclass, got {value!r}"
        )

    # -- derived views ------------------------------------------------------ #

    @property
    def content_hash(self) -> str:
        """SHA-256 over the route's semantic content (plan §3.1, §7.3).

        Hashed: ``name``, ``description``, the ``args_model`` JSON schema,
        ``examples``, ``tags``, ``requires``, and the JSON-serialisable subset of
        ``metadata``. ``visibility`` is excluded — a lambda is not serialisable,
        and two deployments sharing a registry must agree on the hash.

        The unordered collections (``tags``, ``requires`` — both frozensets) are
        sorted before hashing so iteration order cannot change identity.
        ``examples`` is a **tuple and is hashed in declaration order**: it is
        rendered in that order by :meth:`card` and by :attr:`embed_text`, so
        sorting it here would let a reorder change the prompt bytes and the embed
        text without bumping ``registry_version`` — the exact stale-cache failure
        §7.3 exists to prevent.

        Deliberately **recomputed on every access, never memoised on the
        instance**. A cached private attribute survives ``model_copy(update=...)``
        — the idiomatic way to edit a frozen model — so an edited Route would
        keep reporting its pre-edit hash. That stale hash would propagate into
        ``registry_version`` and silently pin the prompt cache, the shortlist
        index and the calibration artifact to a registry that no longer exists,
        which is precisely the invalidation failure §7.3 exists to prevent.
        The cost is a small hash over a handful of fields; ``Registry`` caches the
        aggregate, so the hot path pays it once per registry, not per request.
        """
        schema: Any = None
        if self.args_model is not None:
            try:
                schema = self.args_model.model_json_schema()
            except Exception:  # pragma: no cover - exotic/unschematisable model
                schema = f"{self.args_model.__module__}.{self.args_model.__qualname__}"
        payload = {
            "name": self.name,
            "description": self.description,
            "args_schema": schema,
            "examples": list(self.examples),
            "tags": sorted(self.tags),
            "requires": sorted(self.requires),
            "metadata": _json_safe_metadata(self.metadata),
        }
        return sha256_hex(canonical_json(payload))

    @property
    def embed_text(self) -> str:
        """The text the embedding cache is keyed on (plan §7.3).

        ``embed_text = name + description + examples``, deliberately excluding
        ``args_model``: an args-schema-only change bumps ``registry_hash``
        (invalidating the prompt cache, which does contain the schema) without
        forcing a re-embed of anything.
        """
        parts = [self.name, _normalize_ws(self.description)]
        parts.extend(_normalize_ws(example) for example in self.examples)
        return "\n".join(parts)

    def args_summary(self) -> str:
        """One-line argument summary for :meth:`card`, e.g. ``"order_id (string,
        required), reason (string, optional)"``.

        Field order follows the model's declaration order, which Pydantic
        preserves in ``model_json_schema()`` — deterministic rendering matters
        because this string lands in a cached prompt prefix (§4.6).
        """
        if self.args_model is None:
            return "none"
        try:
            schema = self.args_model.model_json_schema()
        except Exception:  # pragma: no cover - exotic/unschematisable model
            return "unknown"
        properties = schema.get("properties") or {}
        if not properties:
            return "none"
        required = set(schema.get("required") or ())
        parts = []
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                field_schema = {}
            requiredness = "required" if field_name in required else "optional"
            parts.append(
                f"{field_name} ({_json_type_name(field_schema)}, {requiredness})"
            )
        return ", ".join(parts)

    def card(self, include_args: bool = True) -> str:
        """Render this route's canonical prompt card (plan §5.6).

        "One formatting function renders a route card identically in shortlist
        mode, full-registry mode, and eval fixtures" — this is that function, so
        distillation examples harvested from audit records are format-consistent
        regardless of which mode produced them. Nothing else may render a route
        for a prompt.

        Format (whitespace-normalised, tags sorted, examples in declaration
        order)::

            - refund: Issue or check a refund for an order.
              tags: billing, payments
              args: order_id (string, required), reason (string, optional)
              e.g. "I want my money back for order 123"

        Args:
            include_args: emit the ``args:`` line. False for the shortlist
                pointer block (§4.6 segment C), where only names and one-line
                descriptions are needed and tokens are the scarce resource.
        """
        lines = [f"- {self.name}: {_normalize_ws(self.description)}"]
        if self.tags:
            lines.append(f"  tags: {', '.join(sorted(self.tags))}")
        if include_args and self.args_model is not None:
            lines.append(f"  args: {self.args_summary()}")
        for example in self.examples:
            lines.append(f'  e.g. "{_normalize_ws(example)}"')
        return "\n".join(lines)
