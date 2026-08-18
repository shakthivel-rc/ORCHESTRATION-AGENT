"""Eval fixtures — cases, catalogs and their on-disk forms (plan §9.1).

Three things live here, and nothing else:

1. **The expected-label models.** ``ExpectedRoute`` / ``ExpectedMultiRoute`` /
   ``ExpectedClarify`` / ``ExpectedAbstain`` mirror the
   :data:`~switchboard.core.decision.Decision` union *exactly*, discriminated on
   ``kind``. Clarify and abstain are first-class gold labels, not error buckets —
   that is the whole point of §9.1, because "clarifying is cheaper than
   mis-routing" (§9.2) is only measurable if clarify can be the right answer.

2. **The frozen-catalog round trip.** :func:`registry_to_fixture` freezes a
   :class:`~switchboard.core.registry.Registry` to a plain JSON dict pinned to its
   ``registry_version``; :func:`registry_from_fixture` rebuilds it. The round trip
   is **version-preserving**: rebuilding a fixture yields a registry whose
   ``version`` is byte-identical to the one that was frozen, so an eval run and a
   production run can be proven to have decided over the same catalog (§7.3).

3. **The JSONL container.** Cases are one ``EvalCase`` per line under a
   ``{"fixture": "sb-eval/1"}`` header line, so the file is schema-versioned and a
   future format change fails loudly instead of silently mis-parsing.

Plus :func:`dogfood_suite`, the repo's own synthetic catalog + cases that plan
§9.6 runs on every PR in replay mode.

**Packaging (§9, §13 ruling #17).** This module is Pydantic + stdlib only. The
CLI (``switchboard eval ...``), ``gates.yaml`` and rendered reports are the
[v0.2] ``[eval]`` extra; fixtures and the replay cache are core so that a user can
write and load an eval suite with nothing installed but switchboard itself.
"""

from __future__ import annotations

import json
import keyword
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model, field_serializer

from switchboard.core.registry import Registry
from switchboard.core.route import Route
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "FIXTURE_SCHEMA",
    "EvalCase",
    "EvalSuite",
    "Expected",
    "ExpectedAbstain",
    "ExpectedClarify",
    "ExpectedMultiRoute",
    "ExpectedRoute",
    "dogfood_registry",
    "dogfood_suite",
    "load_cases",
    "load_suite",
    "registry_from_fixture",
    "registry_to_fixture",
    "save_cases",
    "save_suite",
]

FIXTURE_SCHEMA = "sb-eval/1"
"""The value of the JSONL header line's ``"fixture"`` key (plan §9.1).

``<family>/<version>``. A file whose family differs is not a switchboard fixture;
a file whose version differs was written by a different switchboard major and is
refused rather than parsed on a hope.
"""

_FIXTURE_FAMILY = FIXTURE_SCHEMA.partition("/")[0]
"""``"sb-eval"`` — the family half of :data:`FIXTURE_SCHEMA`, so a foreign file
and a future switchboard get different, actionable error messages."""


# --------------------------------------------------------------------------- #
# Expected labels — one per Decision kind (plan §9.1, §3.4).
# --------------------------------------------------------------------------- #


class ExpectedRoute(BaseModel):
    """Gold label: exactly one route should be committed (plan §9.1)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["route"] = "route"
    any_of: list[str]
    """Acceptable route names. A list, not a scalar, because overlapping catalogs
    are real: two routes may both be defensible answers and scoring the run
    against a single "right" name would manufacture errors that no user has."""
    args: dict[str, Any] | None = None
    """Gold arguments. **Scored at [v0.2]** (§9.2 tags arg exact match as full-metrics
    work); the field exists now so labelled data written today is not thrown away."""
    args_match: Literal["exact", "subset"] = "subset"
    """How :attr:`args` is compared once arg scoring lands. [v0.2]"""


class ExpectedMultiRoute(BaseModel):
    """Gold label: several independent routes (plan §9.1)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["multi_route"] = "multi_route"
    routes: list[str]
    order_sensitive: bool = False
    """switchboard's ``multi_route`` is explicitly order-independent (§3.4), so
    this defaults False; set it when the labelled data really does encode a
    sequence, and scoring compares sequences instead of sets."""


class ExpectedClarify(BaseModel):
    """Gold label: the router should have asked a question (plan §9.1).

    Clarify is a first-class gold label. ``acceptable_routes`` is what keeps the
    metric honest: some ambiguous queries have a route that is *not harmful* to
    pick, and counting that as a hard error would push a router toward guessing.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["clarify"] = "clarify"
    missing: list[str] = Field(default_factory=list)
    """Facts the question ought to ask for. Soft credit, scored at [v0.2]."""
    acceptable_routes: list[str] = Field(default_factory=list)
    """Routes that count as non-harmful instead of a clarify."""


class ExpectedAbstain(BaseModel):
    """Gold label: nothing in the catalog handles this query (plan §9.1)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["abstain"] = "abstain"


Expected = Annotated[
    ExpectedRoute | ExpectedMultiRoute | ExpectedClarify | ExpectedAbstain,
    Field(discriminator="kind"),
]
"""The gold-label union, discriminated on ``kind`` exactly like ``Decision`` (§9.1)."""


class EvalCase(BaseModel):
    """One labelled query (plan §9.1)."""

    model_config = ConfigDict(frozen=True)

    id: str
    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    """:class:`~switchboard.core.context.RequestContext` fields, as a mapping. Kept
    loose so a fixture file survives context fields being added to the model."""
    expected: Expected
    tags: set[str] = Field(default_factory=set)
    """Free-form; the harness reports per-tag breakdowns from these."""
    source: str = "hand"
    """``hand`` | ``audit_log`` | ``bfcl`` | ``metatool`` | ``synthetic``. A plain
    ``str`` rather than a Literal so a fixture from a [v0.2] adapter that this
    build does not know about still loads."""

    @field_serializer("tags")
    def _sort_tags(self, value: set[str]) -> list[str]:
        """Emit tags sorted so a saved fixture file is byte-stable.

        ``set`` iteration order is not reproducible across processes; a fixture
        that re-serialises differently on every run makes ``git diff`` useless and
        breaks content-addressed caching of the file itself.
        """
        return sorted(value)

    def gold_routes(self) -> frozenset[str]:
        """Route names that would satisfy this case, for shortlist recall (§9.2).

        Empty when the gold label names no route — an ``abstain`` case has no
        route to retrieve, so it contributes nothing to recall@K rather than
        counting as a miss.
        """
        expected = self.expected
        if isinstance(expected, ExpectedRoute):
            return frozenset(expected.any_of)
        if isinstance(expected, ExpectedMultiRoute):
            return frozenset(expected.routes)
        if isinstance(expected, ExpectedClarify):
            return frozenset(expected.acceptable_routes)
        return frozenset()


_CASE_ADAPTER: TypeAdapter[EvalCase] = TypeAdapter(EvalCase)


class EvalSuite(BaseModel):
    """A named set of cases, optionally pinned to a frozen catalog (plan §9.1)."""

    model_config = ConfigDict(frozen=True)

    name: str = "suite"
    cases: list[EvalCase] = Field(default_factory=list)
    catalog: dict[str, Any] | None = None
    """A frozen :func:`registry_to_fixture` dict, or ``None`` when the suite is run
    against a live registry (``app.routes:registry``). Pinning the catalog is what
    makes a historical run reproducible after the route descriptions move on."""

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:  # type: ignore[override]
        """Iterate the cases, so ``for case in suite`` reads the obvious way.

        Overrides ``BaseModel.__iter__`` (which yields ``(field, value)`` pairs)
        deliberately: a suite is a collection of cases first and a model second.
        Use ``suite.model_dump()`` when the field mapping is what you want.
        """
        return iter(self.cases)

    def registry(self) -> Registry | None:
        """Rebuild the pinned catalog, or ``None`` when the suite has none."""
        return None if self.catalog is None else registry_from_fixture(self.catalog)


# --------------------------------------------------------------------------- #
# Catalog fixtures — registry.to_fixture() and back (plan §9.1, §7.3).
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> bool:
    """Whether ``value`` survives a JSON round trip.

    Mirrors ``Route``'s own metadata filter: only the JSON-serialisable subset of
    ``metadata`` participates in ``Route.content_hash``, so only that subset may
    be written to a fixture — writing more would change nothing about the hash but
    would make the file lie about what was hashed.
    """
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _route_to_fixture(route: Route) -> dict[str, Any]:
    """Freeze one route. ``visibility`` is dropped — a callable is not data."""
    schema: Any = None
    unavailable = False
    if route.args_model is not None:
        try:
            schema = route.args_model.model_json_schema()
        except Exception:  # pragma: no cover - exotic/unschematisable args model
            schema = None
            unavailable = True
    entry: dict[str, Any] = {
        "name": route.name,
        "description": route.description,
        "args_schema": schema,
        "examples": list(route.examples),
        "tags": sorted(route.tags),
        "requires": sorted(route.requires),
        "pinned": route.pinned,
        "clarify_label": route.clarify_label,
        "group": route.group,
        "metadata": {k: v for k, v in route.metadata.items() if _json_safe(v)},
    }
    if unavailable:
        entry["args_schema_unavailable"] = True
    return entry


def registry_to_fixture(registry: Registry) -> dict[str, Any]:
    """Freeze ``registry`` to a JSON-serialisable fixture dict (plan §9.1).

    This is ``registry.to_fixture()`` from the plan, spelled as a function so the
    core :class:`~switchboard.core.registry.Registry` keeps no dependency on the
    eval package. The dict is *content-hashed and pinned*: it carries both the
    full ``content_hash`` and the 12-hex ``registry_version``, and
    :func:`registry_from_fixture` refuses to hand back a registry whose version
    does not match.

    What survives the freeze: every field ``Route.content_hash`` covers (name,
    description, the args-model JSON Schema, examples, tags, requires, the
    JSON-safe metadata subset) plus ``pinned``, ``clarify_label`` and ``group``,
    which change behaviour without changing identity.

    What does not: ``Route.visibility``, which is a Python callable. It is
    excluded from ``content_hash`` for exactly the same reason (§3.1) — two
    deployments sharing a registry must agree on the hash — and entitlement
    predicates are [v0.2] regardless.
    """
    if not isinstance(registry, Registry):
        raise ConfigError(
            f"registry_to_fixture() takes a Registry, got {type(registry).__name__}"
        )
    return {
        "fixture": FIXTURE_SCHEMA,
        "kind": "catalog",
        "registry_version": registry.version,
        "content_hash": registry.content_hash,
        "routes": [_route_to_fixture(route) for route in registry],
    }


_JSON_TO_PY: Mapping[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _annotation_for(node: Mapping[str, Any]) -> Any:
    """Loosest annotation that still carries the property's declared JSON type.

    Deliberately shallow: item types, ``$ref`` targets, enums and every string /
    numeric constraint collapse to the container type or to ``Any``. The rebuilt
    model exists to give the harness field *names* and *requiredness*, not to
    re-litigate the original model's validation — see :func:`_permissive_args_model`.
    """
    declared = node.get("type")
    if isinstance(declared, str):
        return _JSON_TO_PY.get(declared, Any)
    return Any


def _model_name_for(schema: Mapping[str, Any], route_name: str) -> str:
    """A valid Python identifier for the generated class.

    Cosmetic only — the class's own ``__name__`` never reaches a hash or a prompt,
    because :func:`_permissive_args_model` pins ``model_json_schema()`` to the
    stored schema (title included).
    """
    title = schema.get("title")
    candidate = title if isinstance(title, str) and title.isidentifier() else None
    if candidate is None:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in route_name)
        candidate = f"{cleaned.title().replace('_', '')}Args"
    return candidate if candidate.isidentifier() else "FixtureArgs"


def _permissive_args_model(
    schema: Mapping[str, Any], *, route_name: str
) -> type[BaseModel]:
    """Rebuild an args model from stored JSON Schema, permissively (plan §9.1).

    Two properties matter, and only these two:

    * **Hash fidelity.** ``model_json_schema()`` is pinned to the stored schema
      verbatim, so ``Route.content_hash`` — which hashes exactly that dict —
      reproduces to the byte. Without this, a fixture round trip would change
      ``registry_version`` and every audit record from the replayed run would
      claim a catalog that never existed (§7.3).
    * **Permissiveness.** Field types are the loosest annotation carrying the
      declared JSON type, optional fields default to ``None``, and no constraint,
      validator or nested ``$defs`` model is reconstructed. A fixture is a
      *record of a catalog*, not the catalog's source code; re-deriving strict
      validation from a lossy JSON Schema would reject args the original model
      accepted and score the run wrong.

    Properties whose names are not usable as Python identifiers are dropped from
    the rebuilt model — they cannot be a field — but remain in the pinned schema,
    so identity is unaffected.
    """
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required = set(required) if isinstance(required, (list, tuple, set)) else set()

    definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        if (
            not isinstance(field_name, str)
            or not field_name.isidentifier()
            or keyword.iskeyword(field_name)
            or field_name.startswith("_")
        ):
            continue
        node = field_schema if isinstance(field_schema, dict) else {}
        annotation = _annotation_for(node)
        if field_name in required:
            definitions[field_name] = (annotation, ...)
        else:
            definitions[field_name] = (annotation | None, None)

    model: type[BaseModel] = create_model(_model_name_for(schema, route_name), **definitions)
    pinned: dict[str, Any] = json.loads(json.dumps(schema))

    def _pinned_schema(cls: type[BaseModel], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return the stored schema verbatim, whatever the caller asks for.

        Signature-compatible with ``BaseModel.model_json_schema`` by accepting and
        ignoring everything: a fixture has exactly one recorded schema, so
        ``mode="serialization"`` and a custom ``ref_template`` cannot be honoured
        and must not silently return a *different* dict than the one hashed.
        Returns a fresh copy each call, so a caller that mutates the result cannot
        change what the next ``content_hash`` sees.
        """
        del cls, args, kwargs
        copied: dict[str, Any] = json.loads(json.dumps(pinned))
        return copied

    model.model_json_schema = classmethod(_pinned_schema)  # type: ignore[assignment]
    return model


def _route_from_fixture(entry: Mapping[str, Any]) -> Route:
    """Rebuild one :class:`~switchboard.core.route.Route` from a fixture entry."""
    name = entry.get("name")
    if not isinstance(name, str):
        raise ConfigError(
            f"catalog fixture route entry has no 'name' string: {entry!r}"
        )
    if entry.get("args_schema_unavailable"):
        raise ConfigError(
            f"catalog fixture route {name!r} was frozen from an args_model whose "
            f"JSON Schema could not be generated, so it cannot be rebuilt without "
            f"changing registry_version. Re-freeze the registry with an args_model "
            f"that schematises (plan §9.1)."
        )
    schema = entry.get("args_schema")
    args_model = (
        _permissive_args_model(schema, route_name=name)
        if isinstance(schema, dict)
        else None
    )
    return Route(
        name=name,
        description=str(entry.get("description", "")),
        args_model=args_model,
        examples=tuple(entry.get("examples") or ()),
        tags=frozenset(entry.get("tags") or ()),
        requires=frozenset(entry.get("requires") or ()),
        pinned=bool(entry.get("pinned", False)),
        clarify_label=entry.get("clarify_label"),
        group=entry.get("group"),
        metadata=dict(entry.get("metadata") or {}),
    )


def registry_from_fixture(
    fixture: Mapping[str, Any], *, strict: bool = True
) -> Registry:
    """Rebuild the registry frozen by :func:`registry_to_fixture` (plan §9.1).

    Args:
        fixture: the dict :func:`registry_to_fixture` produced.
        strict: verify that the rebuilt registry's ``version`` still equals the
            pinned one. Leave it on. Turning it off is for inspecting a fixture
            written by an incompatible build, and the resulting registry must not
            be used to make claims about a historical run.

    Raises:
        ConfigError: unknown fixture schema version, a malformed payload, or —
            under ``strict`` — a ``registry_version`` that no longer reproduces.
    """
    _check_schema(fixture.get("fixture"), where="catalog fixture")
    routes = fixture.get("routes")
    if not isinstance(routes, (list, tuple)) or not routes:
        raise ConfigError(
            "catalog fixture has no 'routes' array; a registry needs at least one "
            "route (plan §3.2)"
        )
    registry = Registry([_route_from_fixture(entry) for entry in routes])
    pinned = fixture.get("registry_version")
    if strict and isinstance(pinned, str) and pinned and registry.version != pinned:
        raise ConfigError(
            f"catalog fixture is pinned to registry_version {pinned!r} but rebuilt "
            f"as {registry.version!r}. The fixture and this switchboard build "
            f"disagree about catalog identity, so every audit record from a run "
            f"over it would claim a catalog that does not exist (plan §7.3). "
            f"Re-freeze the registry, or pass strict=False to inspect it anyway."
        )
    return registry


# --------------------------------------------------------------------------- #
# JSONL container (plan §9.1: schema-versioned `{"fixture": "sb-eval/1"}` header).
# --------------------------------------------------------------------------- #


def _check_schema(value: Any, *, where: str) -> None:
    """Validate a ``"fixture"`` tag, with an error a human can act on."""
    if value is None:
        raise ConfigError(
            f"{where} has no 'fixture' schema tag. Every switchboard eval fixture "
            f'starts with {{"fixture": "{FIXTURE_SCHEMA}"}} — a JSONL file needs it '
            f"as its first line (plan §9.1)."
        )
    if not isinstance(value, str):
        raise ConfigError(f"{where}: 'fixture' must be a string, got {value!r}")
    if value == FIXTURE_SCHEMA:
        return
    family, _, version = value.partition("/")
    if family != _FIXTURE_FAMILY:
        raise ConfigError(
            f"{where}: {value!r} is not a switchboard eval fixture (expected the "
            f"{_FIXTURE_FAMILY!r} family, e.g. {FIXTURE_SCHEMA!r})."
        )
    raise ConfigError(
        f"{where}: fixture schema version {version!r} is not supported by this "
        f"switchboard build, which reads {FIXTURE_SCHEMA!r}. Fixture schema changes "
        f"are breaking (plan §10.1); convert the file or install a matching "
        f"switchboard."
    )


def _iter_json_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for every non-blank line of a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"{path}:{number}: not valid JSON ({exc.msg}). Eval fixtures are "
                    f"JSONL — exactly one JSON object per line (plan §9.1)."
                ) from exc
            if not isinstance(payload, dict):
                raise ConfigError(
                    f"{path}:{number}: expected a JSON object, got "
                    f"{type(payload).__name__}"
                )
            yield number, payload


def load_suite(path: str | Path, *, name: str | None = None) -> EvalSuite:
    """Load a JSONL fixture file into an :class:`EvalSuite` (plan §9.1).

    Line 1 is the header — ``{"fixture": "sb-eval/1"}``, optionally carrying
    ``"suite"`` (a name) and ``"catalog"`` (a frozen
    :func:`registry_to_fixture` dict). Every subsequent line is one
    :class:`EvalCase`.

    Args:
        path: the ``.jsonl`` file.
        name: override the suite name; defaults to the header's ``"suite"``, then
            to the file stem.

    Raises:
        ConfigError: missing/unknown header, malformed JSON, an unparseable case,
            or a duplicate case id.
    """
    file = Path(path)
    if not file.exists():
        raise ConfigError(f"eval fixture file not found: {file}")

    header: dict[str, Any] | None = None
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for number, payload in _iter_json_lines(file):
        if header is None:
            if "fixture" not in payload:
                raise ConfigError(
                    f"{file}:{number}: the first line of an eval fixture must be the "
                    f'header {{"fixture": "{FIXTURE_SCHEMA}"}}, not a case (plan §9.1).'
                )
            _check_schema(payload.get("fixture"), where=f"{file}:{number}")
            header = payload
            continue
        try:
            case = _CASE_ADAPTER.validate_python(payload)
        except Exception as exc:
            raise ConfigError(f"{file}:{number}: invalid EvalCase — {exc}") from exc
        if case.id in seen:
            raise ConfigError(
                f"{file}:{number}: duplicate case id {case.id!r}; ids key per-case "
                f"reporting and must be unique within a suite"
            )
        seen.add(case.id)
        cases.append(case)

    if header is None:
        raise ConfigError(
            f"{file} is empty; an eval fixture needs at least the header line "
            f'{{"fixture": "{FIXTURE_SCHEMA}"}} (plan §9.1).'
        )
    catalog = header.get("catalog")
    return EvalSuite(
        name=name or str(header.get("suite") or file.stem),
        cases=cases,
        catalog=catalog if isinstance(catalog, dict) else None,
    )


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load just the cases from a JSONL fixture file (plan §9.1)."""
    return load_suite(path).cases


def save_suite(suite: EvalSuite, path: str | Path) -> Path:
    """Write ``suite`` as JSONL with the schema header line (plan §9.1).

    The header carries the suite name and, when the suite pins one, the frozen
    catalog — so a single file is a complete, reproducible eval artifact.
    Parent directories are created. Returns the written path.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {"fixture": FIXTURE_SCHEMA, "suite": suite.name}
    if suite.catalog is not None:
        header["catalog"] = suite.catalog
    with file.open("w", encoding="utf-8") as handle:
        handle.write(_dump_line(header))
        for case in suite.cases:
            handle.write(_dump_line(case.model_dump(mode="json")))
    return file


def save_cases(
    cases: Iterable[EvalCase],
    path: str | Path,
    *,
    name: str = "suite",
    catalog: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``cases`` as a JSONL fixture file (plan §9.1).

    Sugar over :func:`save_suite` for the common case of "I have a list of cases".
    """
    return save_suite(
        EvalSuite(
            name=name,
            cases=list(cases),
            catalog=dict(catalog) if catalog is not None else None,
        ),
        path,
    )


def _dump_line(payload: Mapping[str, Any]) -> str:
    """One JSONL line: compact, sorted keys, UTF-8, newline-terminated.

    Sorted keys because a fixture file lives in git — an unstable key order turns
    every re-save into a whole-file diff.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


# --------------------------------------------------------------------------- #
# The dogfood catalog and suite (plan §9.6, §9.7).
# --------------------------------------------------------------------------- #

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

_OUT_OF_SCOPE: tuple[str, ...] = (
    "what is the weather in Paris tomorrow",
    "write me a haiku about the sea",
    "who won the 1998 world cup",
    "translate 'good morning' into Portuguese",
    "book me a table for two at eight",
)


def dogfood_registry(*, n_routes: int = 120) -> Registry:
    """The repo's own synthetic catalog (plan §9.6 dogfood).

    ``{domain}_{action}_{serial}`` across six domains and five actions, plus a
    pinned ``human_handoff``. Descriptions are *similar but distinguishable* on
    purpose: a catalog whose routes are trivially separable would make the
    shortlist look free and the naive baseline look fine, which is precisely the
    illusion §9.3's mandated bar exists to puncture.

    120 is the default because it is above the 100-route line the report's
    ship/no-ship rule is stated on (§9.5 G1) and well above the ``"auto"``
    shortlist bypass threshold of 25 (§5.3), so retrieval genuinely runs.

    Args:
        n_routes: total catalog size, including ``human_handoff``. Must be >= 2.
    """
    if n_routes < 2:
        raise ConfigError(
            f"dogfood_registry(n_routes={n_routes}) must be >= 2 (one generated "
            f"route plus human_handoff)"
        )
    routes: list[Route] = []
    for index in range(n_routes - 1):
        domain, subject = _DOMAINS[index % len(_DOMAINS)]
        action, verb = _ACTIONS[(index // len(_DOMAINS)) % len(_ACTIONS)]
        serial = index // (len(_DOMAINS) * len(_ACTIONS))
        routes.append(
            Route(
                name=f"{domain}_{action}_{serial}",
                description=(
                    f"{verb} {domain} record in workspace {serial} covering "
                    f"{subject}. Use when the request is about {domain} and asks "
                    f"to {action}. Do not use for other domains."
                ),
                examples=(
                    f"{action} the {domain} entry in workspace {serial}",
                    f"can you {action} my {domain} item in workspace {serial}",
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


def dogfood_suite(
    *, n_routes: int = 120, n_route_cases: int | None = None, pin_catalog: bool = True
) -> EvalSuite:
    """The v0.1 dogfood suite: cases over :func:`dogfood_registry` (plan §9.6).

    Deliberately code, not a data file. It is generated from the same catalog
    generator the registry uses, so the two can never drift, and it ships inside
    the wheel with no packaged data and no ``[eval]`` extra — which is what lets
    the PR lane run it in replay mode with nothing installed (§9.6).

    Three case families, matching the three gold labels v0.1 scores:

    * **route** — the route's own second example, verbatim. A router that cannot
      hit a route from the text indexed for that route has a retrieval problem,
      and recall@K will say so (proto-G4).
    * **clarify** — a domain named with no action and no workspace, so several
      routes fit equally well. Guessing is the failure mode being measured.
    * **abstain** — plainly out of catalog.

    Args:
        n_routes: catalog size handed to :func:`dogfood_registry`.
        n_route_cases: how many route cases to emit (default: one per domain and
            action pair present, capped at the catalog size).
        pin_catalog: freeze the catalog into the suite (see :attr:`EvalSuite.catalog`).
    """
    registry = dogfood_registry(n_routes=n_routes)
    generated = [route for route in registry if route.name != "human_handoff"]
    limit = len(generated) if n_route_cases is None else max(0, min(n_route_cases, len(generated)))

    cases: list[EvalCase] = []
    step = max(1, len(generated) // limit) if limit else 1
    for position, route in enumerate(generated[:: step][:limit]):
        cases.append(
            EvalCase(
                id=f"route-{position:03d}-{route.name}",
                query=route.examples[1],
                expected=ExpectedRoute(any_of=[route.name]),
                tags={"dogfood", "route", *sorted(route.tags)},
                source="synthetic",
            )
        )

    for position, (domain, _subject) in enumerate(_DOMAINS):
        if not any(route.name.startswith(f"{domain}_") for route in generated):
            continue
        cases.append(
            EvalCase(
                id=f"clarify-{position:03d}-{domain}",
                query=f"something needs sorting out with my {domain}",
                expected=ExpectedClarify(missing=["action", "workspace"]),
                tags={"dogfood", "clarify", domain},
                source="synthetic",
            )
        )

    for position, query in enumerate(_OUT_OF_SCOPE):
        cases.append(
            EvalCase(
                id=f"abstain-{position:03d}",
                query=query,
                expected=ExpectedAbstain(),
                tags={"dogfood", "abstain"},
                source="synthetic",
            )
        )

    return EvalSuite(
        name=f"dogfood-{n_routes}",
        cases=cases,
        catalog=registry_to_fixture(registry) if pin_catalog else None,
    )


def dogfood_gold(case: EvalCase) -> Sequence[str]:
    """The gold route names for ``case``, in a stable order.

    Exposed because the harness's recall@K and the [v0.2] ``shortlist-oracle``
    baseline both need "what should have been retrieved" and must agree on it.
    """
    return sorted(case.gold_routes())
