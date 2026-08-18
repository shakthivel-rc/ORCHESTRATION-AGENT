"""Per-call wire-schema construction (plan §4.4; field order §3.4; strict §4.2).

The **wire schema** is not the :data:`~switchboard.core.decision.Decision` union.
It is a small, flat model built per call with :func:`pydantic.create_model` that
describes exactly what the LLM is allowed to emit; ``confidence`` and ``audit``
are library-attached afterwards (plan §3, decision (b)).

Two things are load-bearing here.

**Field order.** The emitted model declares ``rationale`` first and ``route``
after ``kind``, so free-text reasoning is decoded *before* the committed route
token — the reason-then-commit pattern the evidence supports (§3.4). Pydantic
preserves declaration order in ``model_json_schema()["properties"]``, and
``create_model(**fields)`` preserves the insertion order of its keyword
arguments, so the JSON Schema the provider sees carries the same order. Adapters
that need it explicitly (Gemini ``propertyOrdering``) can read
``list(model.model_fields)``.

**Dynamic vs static ``route`` (§4.4, §13 ruling #14).** ``dynamic`` makes
hallucinated names *unrepresentable* on grammar rungs (``route:
Literal[<candidates>]``); ``static`` keeps the schema stable per router config
(``route: str``) because on ``tool_strict`` the tool definition precedes the
system/user content in the cached prefix, and a per-query enum there would
invalidate the A+B prompt cache on every request and silently void the ~90%
caching economics. The validator enforces ``route in candidates`` in **both**
modes, so ``static`` loses nothing but the decoding-time guarantee.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, create_model

from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from switchboard.core.decision import AbstainReason, DecisionKind
    from switchboard.providers.base import ClientCapabilities

__all__ = [
    "MODEL_ELECTABLE_ABSTAIN_REASONS",
    "SchemaModeSetting",
    "WireSchemaMode",
    "build_wire_schema",
    "enabled_kinds",
    "resolve_schema_mode",
    "strict_compatible_json_schema",
]

WireSchemaMode = Literal["dynamic", "static"]
"""A *resolved* wire-schema mode (plan §4.4)."""

SchemaModeSetting = Literal["auto", "dynamic", "static"]
"""The ``Router(wire_schema=...)`` config value, ``"auto"`` included (plan §3.3)."""

MODEL_ELECTABLE_ABSTAIN_REASONS: tuple[AbstainReason, ...] = ("model_elected",)
"""The subset of :data:`~switchboard.core.decision.AbstainReason` the *model* may
emit (plan §3.4, §6.4).

Every other member of that closed vocabulary is a **library** verdict:
``low_confidence`` comes from threshold fusion, ``unparseable_output`` /
``invalid_route_reference`` / ``invalid_args`` from the validator,
``no_eligible_routes`` from the entitlement filter (with no LLM call at all), and
``provider_error`` from the transport. Offering those to the model would let it
mislabel a decision that the audit, the distillation filters and the eval
fixtures all key on, so the wire enum is narrowed to the one reason a model can
truthfully claim: it elected to abstain.
"""

_SCHEMA_CACHE_SIZE = 256
"""Distinct wire schemas kept alive. One entry per ``(candidate set, mode, kinds,
verbalized)``; ``static`` mode collapses every candidate set onto one entry."""

_MAX_MODEL_NAME_LEN = 64


def _literal_of(values: tuple[str, ...]) -> Any:
    """Build ``Literal[*values]`` from a runtime tuple.

    ``Literal[("a", "b")]`` is spelled with a tuple subscript, which typing
    unpacks into ``Literal["a", "b"]``. The enum members are only known at call
    time (they are this request's candidate names), so this cannot be written as
    a static annotation.
    """
    return Literal[values]


def enabled_kinds(
    *,
    allow_clarify: bool = True,
    multi_route: bool = False,
    allow_plan: bool = False,
) -> tuple[DecisionKind, ...]:
    """Return the decision kinds this router configuration permits (plan §3.3).

    The single source of truth shared by the wire schema's ``kind`` enum and the
    prompt's closing instruction (``engine/prompt.py``), so the two can never
    disagree about what the model is allowed to answer.

    ``route`` and ``abstain`` are always present: a router that cannot route is
    pointless, and model-elected abstain is honored on every path regardless of
    configuration (§6.3, "Model-elected clarify/abstain always pass through").

    Args:
        allow_clarify: ``Router(allow_clarify=...)``, default True.
        multi_route: ``Router(multi_route=...)``, default False.
        allow_plan: ``Router(allow_plan=...)``. **Inert in v0.1** — ``plan`` is a
            [v0.5] decision kind, so it is never offered on the wire even when
            this is True. The parameter exists so callers written against the
            final config surface keep working.
    """
    del allow_plan  # [v0.5] PlanDecision — accepted, deliberately not emitted.
    kinds: list[DecisionKind] = ["route"]
    if multi_route:
        kinds.append("multi_route")
    if allow_clarify:
        kinds.append("clarify")
    kinds.append("abstain")
    return tuple(kinds)


def resolve_schema_mode(
    capabilities: ClientCapabilities,
    configured: SchemaModeSetting = "auto",
) -> WireSchemaMode:
    """Resolve ``Router(wire_schema=...)`` against the client's rung (plan §4.4).

    ``"auto"`` picks ``dynamic`` only on the ``grammar`` rung, where the output
    schema does **not** participate in the cached prompt prefix (OpenAI
    ``response_format``, Gemini ``responseSchema``, vLLM+XGrammar) and a per-query
    enum is therefore free. On ``tool_strict`` the tool definition sits *ahead* of
    the cached content, so a per-query enum would bust the A+B cache on every
    request; on ``json_mode`` and ``none`` the schema is rendered into the prompt
    itself, with the same consequence. Those three resolve to ``static``, where
    the candidate enum is carried by the prompt's candidate list and enforced by
    the validator.

    An explicit ``"dynamic"``/``"static"`` is always honored — the escape hatch
    for a provider whose caching behaviour differs from its rung's default.
    """
    if configured == "auto":
        return "dynamic" if capabilities.structured == "grammar" else "static"
    if configured in ("dynamic", "static"):
        return configured
    raise ConfigError(
        f"wire_schema must be 'auto', 'dynamic' or 'static', got {configured!r}"
    )


def build_wire_schema(
    candidates: Sequence[str],
    *,
    mode: WireSchemaMode = "static",
    allow_clarify: bool = True,
    multi_route: bool = False,
    allow_plan: bool = False,
    verbalized: bool = False,
    model_name: str = "RouterOutput",
) -> type[BaseModel]:
    """Build this call's flat ``RouterOutput`` model (plan §4.4, §3.4).

    Field order is normative and is exactly::

        rationale -> kind -> route -> args
                  -> routes                     (only when multi_route)
                  -> question, candidates, missing   (only when allow_clarify)
                  -> reason
                  -> stated_confidence          (only when verbalized)

    ``rationale`` precedes ``kind``/``route`` so reasoning is decoded before
    commitment (§3.4); ``stated_confidence`` is appended **last**, after the
    commitment, so a verbalized score cannot steer the decoding of the route it
    is supposed to be scoring (§6.2).

    **Why ``args`` is an untyped object.** Each route has its own ``args_model``,
    so the "correct" wire type is a discriminated union of per-route arg models
    keyed on ``route``. v0.1 deliberately does not do that: with K=10 candidates
    the union multiplies the emitted schema by the number of candidates (blowing
    past strict-mode schema-size and nesting limits, and past the ~90% of the
    prompt budget the route directory already owns), it re-introduces a per-query
    schema on exactly the ``tool_strict`` rung where §4.4 needs a *stable* one,
    and several providers reject ``oneOf``/discriminated unions in strict mode
    outright. So the wire carries ``args: dict[str, Any] | None`` — a free-form
    object — and ``engine/validate.py`` re-validates it against the chosen
    route's ``args_model`` (§4.5 step 3), with one repair pass and a downgrade to
    ``clarify`` on failure. The type guarantee is preserved; it just moves from
    the decoder to the validator. Revisit if grammar-rung union support and
    schema budgets both improve.

    Args:
        candidates: the entitlement-filtered candidate route names. Used as the
            ``route`` enum in ``dynamic`` mode and ignored in ``static`` mode.
            Sorted and de-duplicated internally so an identical candidate *set*
            yields a byte-identical schema no matter how the prompt shuffled it
            (§5.6) — the schema participates in the cached prefix on some rungs.
        mode: resolved by :func:`resolve_schema_mode`.
        allow_clarify: emit the clarify arm and offer ``"clarify"`` as a kind.
        multi_route: emit the ``routes`` array and offer ``"multi_route"``.
        allow_plan: [v0.5], inert — see :func:`enabled_kinds`.
        verbalized: append ``stated_confidence``. Set only when the verbalized
            rung is active (no logprobs, no vote — §6.2); it is recorded, damped
            x0.85 at fusion, and never solely trusted.
        model_name: class name of the generated model, which becomes the schema
            title (and, on ``tool_strict``, often the tool name).

    Returns:
        A fresh-or-memoized ``BaseModel`` subclass. Identical arguments return the
        **same class object**, so repeated identical candidate sets never rebuild.

    Raises:
        ConfigError: ``mode="dynamic"`` with an empty candidate set (an empty
            ``Literal`` is not constructible — and that state should already have
            degraded to ``abstain("no_eligible_routes")`` with no LLM call, §13
            ruling #3), or an out-of-range ``mode``/``model_name``.
    """
    if mode not in ("dynamic", "static"):
        raise ConfigError(
            f"wire schema mode must be 'dynamic' or 'static', got {mode!r}; "
            f"call resolve_schema_mode() to turn 'auto' into one of them"
        )
    names = tuple(sorted({name for name in candidates if name}))
    if mode == "dynamic" and not names:
        raise ConfigError(
            "cannot build a dynamic wire schema with no candidates: an empty "
            "route enum is not representable. An empty candidate set must "
            "degrade to abstain(reason='no_eligible_routes') with no LLM call "
            "(plan §13 ruling #3)."
        )
    if not model_name.isidentifier() or len(model_name) > _MAX_MODEL_NAME_LEN:
        raise ConfigError(
            f"model_name must be a Python identifier of at most "
            f"{_MAX_MODEL_NAME_LEN} characters, got {model_name!r}"
        )
    # In static mode the candidate names do not appear anywhere in the schema, so
    # normalising them away makes the memo key what §4.4 promises the schema is:
    # "stable per router config". Without this every distinct shortlist would
    # rebuild a byte-identical model and evict the cache on every request.
    key_names = names if mode == "dynamic" else ()
    return _build_wire_schema_cached(
        key_names,
        mode,
        allow_clarify,
        multi_route,
        allow_plan,
        verbalized,
        model_name,
    )


@lru_cache(maxsize=_SCHEMA_CACHE_SIZE)
def _build_wire_schema_cached(
    candidates: tuple[str, ...],
    mode: WireSchemaMode,
    allow_clarify: bool,
    multi_route: bool,
    allow_plan: bool,
    verbalized: bool,
    model_name: str,
) -> type[BaseModel]:
    """Memoized core of :func:`build_wire_schema`; arguments are pre-normalised.

    Building a Pydantic model is not free (it compiles a core schema), and the
    candidate set repeats constantly in production — same tenant, same registry,
    similar queries. Keyed on the frozen tuple of everything that can change the
    emitted schema.
    """
    kinds = enabled_kinds(
        allow_clarify=allow_clarify, multi_route=multi_route, allow_plan=allow_plan
    )
    route_type: Any = _literal_of(candidates) if mode == "dynamic" else str

    # Insertion order below IS the wire order (plan §3.4). Pydantic keeps the
    # declaration order of `create_model(**fields)` in `model_fields` and in
    # `model_json_schema()["properties"]`, so nothing downstream has to re-sort.
    fields: dict[str, Any] = {}

    fields["rationale"] = (
        str,
        Field(
            description=(
                "Fill this FIRST, before any other field. One or two sentences: "
                "the evidence in the request that decides the outcome, and the "
                "closest route you rejected. Commit only after writing it."
            )
        ),
    )
    fields["kind"] = (
        _literal_of(kinds),
        Field(
            description=(
                "The decision. 'route' = exactly one listed route matches. "
                + ("'multi_route' = several independent asks. " if multi_route else "")
                + (
                    "'clarify' = in scope but underdetermined. "
                    if allow_clarify
                    else ""
                )
                + "'abstain' = nothing in the catalog handles this request."
            )
        ),
    )
    fields["route"] = (
        route_type | None,
        Field(
            default=None,
            description=(
                "Required when kind='route'; null otherwise. Must be one of the "
                "candidate route names listed in the prompt, copied exactly."
            ),
        ),
    )
    fields["args"] = (
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "Arguments for the chosen route, as a JSON object keyed by the "
                "argument names shown on that route's card. Null when the route "
                "takes no arguments. Never invent a value that is not present or "
                "unambiguously implied in the request."
            ),
        ),
    )

    if multi_route:
        item_model = create_model(
            f"{model_name}Call",
            __doc__="One (route, args) pair inside a multi_route decision.",
            route=(
                route_type,
                Field(description="A candidate route name, copied exactly."),
            ),
            args=(
                dict[str, Any] | None,
                Field(default=None, description="Arguments for this route, or null."),
            ),
        )
        fields["routes"] = (
            list[item_model] | None,  # type: ignore[valid-type]
            Field(
                default=None,
                description=(
                    "Required when kind='multi_route'; null otherwise. One entry "
                    "per independent ask in the request. Never use this to hedge "
                    "between two candidates for a single ask - that is a clarify."
                ),
            ),
        )

    if allow_clarify:
        fields["question"] = (
            str | None,
            Field(
                default=None,
                description=(
                    "Required when kind='clarify'; null otherwise. One short "
                    "question that would resolve exactly this ambiguity."
                ),
            ),
        )
        # Enum-constrained to the entitled candidate set on the dynamic rung, so
        # the model cannot offer a route this tenant may not see (plan §6.5).
        fields["candidates"] = (
            list[route_type] | None,
            Field(
                default=None,
                description=(
                    "The candidate routes you were torn between; null when not "
                    "clarifying."
                ),
            ),
        )
        fields["missing"] = (
            list[str] | None,
            Field(
                default=None,
                description=(
                    "Names of the arguments you could not extract; null when not "
                    "clarifying or when nothing is missing."
                ),
            ),
        )

    fields["reason"] = (
        _literal_of(MODEL_ELECTABLE_ABSTAIN_REASONS) | None,
        Field(
            default=None,
            description=(
                "Required when kind='abstain'; null otherwise. Always "
                "'model_elected' - every other abstain reason is decided by the "
                "library, not by you."
            ),
        ),
    )

    if verbalized:
        # Appended LAST, after the commitment, so it cannot steer the decoding of
        # the route it scores (plan §6.2). Damped x0.85 at fusion (§6.3).
        fields["stated_confidence"] = (
            float | None,
            Field(
                default=None,
                ge=0.0,
                le=1.0,
                description=(
                    "How confident you are in the decision above, from 0.0 to "
                    "1.0. Answer after you have committed; do not revise the "
                    "fields above to match it."
                ),
            ),
        )

    return create_model(
        model_name,
        __doc__=(
            "One routing decision. Write `rationale` first, then commit to a "
            "`kind` and fill only the fields that kind requires; leave the rest "
            "null."
        ),
        **fields,
    )


# --------------------------------------------------------------------------- #
# Strict-mode JSON Schema (plan §4.2: OpenAI strict Structured Outputs).
# --------------------------------------------------------------------------- #


def strict_compatible_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Project ``model`` to an OpenAI-strict-safe JSON Schema (plan §4.2).

    Strict Structured Outputs (and Gemini's ``responseSchema``, and forced tool
    use on several providers) reject the schema Pydantic emits by default. Four
    transformations, applied through the whole document including ``$defs``:

    1. **every** property is listed in ``required`` — strict mode has no notion
       of an omittable key;
    2. optionality is therefore re-expressed as ``anyOf: [T, {"type": "null"}]``,
       which is how strict mode spells "may be absent". Pydantic already emits
       that shape for ``T | None``; a field that had a default without being
       nullable is wrapped here so the projection is total;
    3. ``default`` is stripped everywhere — strict mode forbids it, and a default
       is meaningless once the key is mandatory;
    4. every object that declares ``properties`` gets ``additionalProperties:
       false``.

    Objects that declare **no** ``properties`` are left untouched: that is the
    free-form ``args`` object (see :func:`build_wire_schema`), and clamping it to
    ``additionalProperties: false`` would make it permanently empty rather than
    merely unconstrained. Providers that reject free-form objects on the grammar
    rung must have their adapter drop to a lower rung for that field — the
    validate-and-retry loop (§4.5) stays armed on every rung precisely for this.

    The returned mapping is a fresh deep copy on every call; callers may mutate it
    (adapters routinely bolt on ``name``/``strict`` wrappers).

    Args:
        model: any Pydantic model. Assumes optionality is expressed as ``T | None``
            with a ``None`` default, which is how :func:`build_wire_schema`
            writes it.
    """
    return copy.deepcopy(_strict_schema_cached(model))


@lru_cache(maxsize=_SCHEMA_CACHE_SIZE)
def _strict_schema_cached(model: type[BaseModel]) -> dict[str, Any]:
    """Memoized strict projection. Keyed on the model class, which
    :func:`build_wire_schema` already memoizes — so a steady-state router
    generates its schema exactly once."""
    # A model's own schema is always an object, so the recursive walk returns a
    # dict here; the annotation pins that down for callers.
    strict: dict[str, Any] = _strictify(model.model_json_schema())
    return strict


def _is_nullable(node: dict[str, Any]) -> bool:
    """Whether ``node`` already admits ``null``."""
    node_type = node.get("type")
    if node_type == "null" or (isinstance(node_type, list) and "null" in node_type):
        return True
    for combinator in ("anyOf", "oneOf"):
        options = node.get(combinator)
        if isinstance(options, list) and any(
            isinstance(option, dict) and option.get("type") == "null"
            for option in options
        ):
            return True
    return False


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``node`` so it also admits ``null``, keeping annotations outside.

    ``{"type": "string", "description": "d"}`` becomes
    ``{"description": "d", "anyOf": [{"type": "string"}, {"type": "null"}]}`` —
    descriptions must stay at the property level to survive the provider's own
    schema normalisation.
    """
    out: dict[str, Any] = {}
    inner = dict(node)
    for key in ("title", "description"):
        if key in inner:
            out[key] = inner.pop(key)
    existing = inner.get("anyOf")
    if isinstance(existing, list) and len(inner) == 1:
        out["anyOf"] = [*existing, {"type": "null"}]
    else:
        out["anyOf"] = [inner, {"type": "null"}]
    return out


def _strictify_property(node: Any) -> Any:
    """Strictify one property schema, converting "had a default" into nullable."""
    if not isinstance(node, dict):
        return _strictify(node)
    optional = "default" in node
    out = _strictify(node)
    if optional and not _is_nullable(out):
        out = _nullable(out)
    return out


def _strictify(node: Any) -> Any:
    """Recursively apply the four strict-mode transformations to ``node``."""
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "default":
            continue  # (3) strict mode forbids defaults.
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _strictify_property(sub) for name, sub in value.items()}
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: _strictify(sub) for name, sub in value.items()}
        else:
            out[key] = _strictify(value)

    properties = out.get("properties")
    if isinstance(properties, dict):
        out["required"] = list(properties)  # (1) everything is required.
        out["additionalProperties"] = False  # (4) closed objects only.
    return out
