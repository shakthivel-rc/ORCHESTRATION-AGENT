"""Layered semantic validation of one model output (plan §4.5).

Three layers, applied in order, each with its own failure mode in the degradation
table (§3.8):

1. :func:`parse_wire_output` — schema parse. Primary enforcement on rungs 3-4 of
   the capability ladder (§4.3), a safety net on rungs 1-2 where constrained
   decoding already guarantees shape. Failure after ``schema_attempts`` repairs →
   ``abstain("unparseable_output")``.
2. :func:`check_route_reference` — ``route ∈`` the entitlement-filtered candidate
   set. Applied **even under a ``Literal`` wire enum** (§4.4): defence in depth is
   the whole point, since the enum is absent in ``wire_schema="static"`` mode and
   the router treats every shortlister as untrusted on entitlements (§5.1).
   Failure after repair → ``abstain("invalid_route_reference")``.
3. :func:`validate_args` — the chosen route's ``args_model``. An args-only failure
   is *not* a routing failure: the route was sound and a missing argument is a
   user question, so after one repair pass it downgrades to ``clarify`` carrying
   the missing field names (§3.8 row 4), or ``abstain("invalid_args")`` when
   ``allow_clarify=False``.

Everything here is a **pure function**. The retry loop that calls them — attempt
counting, the repair segment appended *after* the query segment so the A+B cache
still hits, temperature dropped to 0 on retry — lives in ``engine/loop.py``. This
module supplies the predicates and the repair text, nothing else.

:class:`ValidationError` is a plain dataclass, deliberately **not** an exception:
a model that emitted unusable output is a healthy system producing a degraded
*decision*, not a broken configuration, and §3.8's raise-vs-degrade rule puts it
firmly on the degrade side (§13 ruling #2).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from switchboard.core.route import Route

__all__ = [
    "ValidationError",
    "ValidationStage",
    "check_route_reference",
    "extract_json_object",
    "format_error_for_repair",
    "parse_wire_output",
    "validate_args",
]

_WireT = TypeVar("_WireT", bound=BaseModel)

ValidationStage = Literal["schema", "route_ref", "args"]
"""Which of §4.5's three layers rejected the output. The stage decides the
degradation: ``schema`` → ``unparseable_output``, ``route_ref`` →
``invalid_route_reference``, ``args`` → ``clarify`` (or ``invalid_args``)."""

_MAX_REPORTED_ERRORS = 5
"""Repair prompts are cheap only while they stay short; five field errors is
plenty for the model to fix the output, and the truncation is announced."""

_MAX_LISTED_ROUTES = 40
"""Upper bound on route names inlined into a repair message. ``max_candidates``
defaults to 25 (§3.3), so this only bites on deliberately oversized candidate
sets."""


@dataclass(frozen=True, slots=True)
class ValidationError:
    """One validation failure — a value, not an exception (plan §4.5, §3.8).

    Carried through the repair loop, rendered into the repair segment by
    :func:`format_error_for_repair`, and finally mapped to an
    :data:`~switchboard.core.decision.AbstainReason` (or a ``clarify``) by
    ``engine/policy.py``.
    """

    stage: ValidationStage
    message: str
    """Human-ish, single-line-ish summary. Safe to log; may quote field names and
    schema constraints, never the user's payload."""
    detail: dict[str, Any] = field(default_factory=dict)
    """Structured context. Documented keys:

    * ``errors``: ``list[str]`` of ``"field: message"`` entries from Pydantic.
    * ``missing``: ``list[str]`` of required fields the output omitted — this is
      what becomes ``ClarifyDecision.missing`` (§3.8 row 4).
    * ``route``: the route whose ``args_model`` rejected the args (``args`` stage).
    * ``unknown``: route names that were not in the candidate set (``route_ref``).
    * ``allowed``: the candidate names, for the repair prompt (``route_ref``).
    """

    @property
    def missing(self) -> tuple[str, ...]:
        """Required field names the model failed to supply.

        Populated for the ``args`` stage (and for a ``schema`` failure that was
        purely a missing top-level field). ``ClarifyDecision.missing`` is built
        from this, so the caller can ask the user for exactly what is absent.
        """
        raw = self.detail.get("missing")
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(str(item) for item in raw)


# --------------------------------------------------------------------------- #
# Layer 1 — schema parse
# --------------------------------------------------------------------------- #


def parse_wire_output(
    raw_text_or_obj: str | bytes | Mapping[str, Any] | BaseModel | None,
    wire_model: type[_WireT],
) -> tuple[_WireT | None, ValidationError | None]:
    """Parse a provider response into the per-call wire model (plan §4.5 layer 1).

    Tolerant by design, because the input ranges from "already a validated model
    instance" (instructor / strict Structured Outputs) to "prose with a fenced
    JSON block in the middle" (a bare BYO callable on the ``structured="none"``
    rung, §4.3). Tolerance here is not laxity: whatever is extracted is still
    validated by Pydantic against ``wire_model``, so the *contract* is identical
    on every rung — only the effort to find the payload differs.

    Extraction order for text:

    1. strip Markdown code fences (```` ```json ... ``` ````),
    2. take the **outermost balanced** ``{...}`` object — brace-counting that
       respects JSON string literals and escapes, so a ``}`` inside the rationale
       cannot truncate the object,
    3. ``json.loads`` it, then ``wire_model.model_validate``.

    Args:
        raw_text_or_obj: the provider's output — a model instance, a mapping, raw
            text, or ``None``.
        wire_model: the per-call schema built by ``engine/schema.py``.

    Returns:
        ``(instance, None)`` on success, ``(None, ValidationError(stage="schema"))``
        on failure. Never raises: a malformed model output is a degradation, not
        an exception (§3.8).
    """
    if raw_text_or_obj is None:
        return None, ValidationError("schema", "the model returned no output")

    if isinstance(raw_text_or_obj, wire_model):
        return raw_text_or_obj, None

    payload: Any
    if isinstance(raw_text_or_obj, BaseModel):
        # A different model class (e.g. an adapter's own parsed type): round-trip
        # through a dict so field aliases and nested models re-validate cleanly.
        payload = raw_text_or_obj.model_dump()
    elif isinstance(raw_text_or_obj, Mapping):
        payload = dict(raw_text_or_obj)
    else:
        text = (
            raw_text_or_obj.decode("utf-8", "replace")
            if isinstance(raw_text_or_obj, bytes)
            else raw_text_or_obj
        )
        blob = _extract_json_object(text)
        if blob is None:
            return None, ValidationError(
                "schema",
                "no JSON object found in the model output",
                {"errors": ["output did not contain a '{...}' object"]},
            )
        try:
            payload = json.loads(blob)
        except (ValueError, TypeError) as exc:
            return None, ValidationError(
                "schema",
                f"model output is not valid JSON: {exc}",
                {"errors": [str(exc)]},
            )

    try:
        return wire_model.model_validate(payload), None
    except PydanticValidationError as exc:
        errors = _describe(exc)
        return None, ValidationError(
            "schema",
            "model output did not match the required schema: " + "; ".join(errors),
            {"errors": errors, "missing": _missing_fields(exc)},
        )


def extract_json_object(text: str) -> str | None:
    """Public alias of the tolerant JSON extractor used by :func:`parse_wire_output`.

    Exposed because the loop needs the *same* extraction when a schema failure has
    to be re-classified: under ``wire_schema="dynamic"`` a hallucinated route name
    is rejected by the ``Literal`` enum during parsing, so plan §3.8's
    ``invalid_route_reference`` row can only be recognised by re-reading the raw
    payload. Sharing one extractor keeps that second read byte-identical to the
    first, instead of a near-copy that drifts.
    """
    return _extract_json_object(text)


def _extract_json_object(text: str) -> str | None:
    """Return the outermost balanced ``{...}`` object in ``text``, or ``None``."""
    stripped = _strip_code_fences(text).strip()
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    # Unbalanced (truncated by max_tokens, most likely). Fall back to the widest
    # plausible span so a single missing brace does not cost a whole repair round
    # trip; json.loads is the arbiter either way.
    end = stripped.rfind("}")
    return stripped[start : end + 1] if end > start else None


def _strip_code_fences(text: str) -> str:
    """Return the contents of the first Markdown code fence, or ``text`` unchanged."""
    opening = text.find("```")
    if opening == -1:
        return text
    body_start = text.find("\n", opening)
    if body_start == -1:
        return text
    closing = text.find("```", body_start)
    return text[body_start + 1 : closing] if closing != -1 else text[body_start + 1 :]


# --------------------------------------------------------------------------- #
# Layer 2 — route reference
# --------------------------------------------------------------------------- #


def check_route_reference(
    parsed: BaseModel,
    allowed: AbstractSet[str],
) -> ValidationError | None:
    """Check every route name the output references against ``allowed`` (§4.5 layer 2).

    Runs **regardless of wire-schema mode** (§4.4): under ``"dynamic"`` the
    ``Literal`` enum already makes a hallucinated name unrepresentable, but under
    ``"static"`` — mandatory on the ``tool_strict`` rung, where a per-query enum
    would void the prompt cache — the prompt is the only thing pointing at the
    candidate list, and a prompt is not an enforcement mechanism. This check is
    the enforcement mechanism, and it is also the last line of defence for the
    entitlement boundary: a name outside ``allowed`` is a name this tenant may not
    see (§7.4).

    Inspects, when present: ``route`` (single commit), ``routes`` (multi-route
    members, as names or as objects with a ``route`` attribute) and ``candidates``
    (the clarify arm's offered routes — enum-constrained in-schema, re-checked
    here so a clarify question can never surface an unentitled route name, §6.5).

    Args:
        parsed: the schema-valid wire output.
        allowed: the entitlement-filtered candidate names.

    Returns:
        ``None`` when every reference is legal, otherwise a
        ``ValidationError(stage="route_ref")``.
    """
    kind = _str_attr(parsed, "kind") or "route"
    route = _str_attr(parsed, "route")

    referenced: list[str] = []
    if route:
        referenced.append(route)
    members = _member_names(parsed)
    referenced.extend(members)
    referenced.extend(name for name in _str_sequence(parsed, "candidates"))

    unknown = [name for name in dict.fromkeys(referenced) if name not in allowed]
    if unknown:
        return ValidationError(
            "route_ref",
            "model named route(s) that are not candidates for this request: "
            + ", ".join(repr(name) for name in unknown),
            {"unknown": unknown, "allowed": sorted(allowed)},
        )

    if kind == "route" and not route:
        return ValidationError(
            "route_ref",
            "model selected kind='route' but named no route",
            {"missing": ["route"], "allowed": sorted(allowed)},
        )
    if kind == "multi_route" and not members:
        return ValidationError(
            "route_ref",
            "model selected kind='multi_route' but named no routes",
            {"missing": ["routes"], "allowed": sorted(allowed)},
        )
    return None


# --------------------------------------------------------------------------- #
# Layer 3 — args
# --------------------------------------------------------------------------- #


def validate_args(
    parsed_args: dict[str, Any] | BaseModel | None,
    route: Route,
) -> tuple[BaseModel | None, ValidationError | None]:
    """Validate extracted args against ``route.args_model`` (plan §4.5 layer 3).

    ``None`` args are validated as ``{}`` rather than short-circuited, precisely so
    that a route whose model has required fields reports them as **missing**
    instead of silently returning "no args" — that missing list is what §3.8's
    row 4 turns into a ``clarify`` the caller can put to the user verbatim.

    Args:
        parsed_args: the ``args`` object from the wire output — a mapping, an
            already-typed model instance, or ``None``.
        route: the committed route. ``route.args_model is None`` means the route
            takes no arguments; any args the model volunteered are dropped rather
            than treated as an error, since a spurious argument on an argument-less
            route is noise, not a reason to re-ask the user.

    Returns:
        ``(instance, None)`` on success (``(None, None)`` for an args-less route),
        otherwise ``(None, ValidationError(stage="args"))`` whose
        :attr:`~ValidationError.missing` names the absent required fields.
    """
    args_model = route.args_model
    if args_model is None:
        return None, None
    if isinstance(parsed_args, args_model):
        return parsed_args, None

    payload: dict[str, Any]
    if isinstance(parsed_args, BaseModel):
        payload = parsed_args.model_dump()
    elif isinstance(parsed_args, Mapping):
        payload = dict(parsed_args)
    else:
        payload = {}

    try:
        return args_model.model_validate(payload), None
    except PydanticValidationError as exc:
        errors = _describe(exc)
        missing = _missing_fields(exc)
        summary = (
            f"missing required argument(s) for route {route.name!r}: " + ", ".join(missing)
            if missing
            else f"invalid arguments for route {route.name!r}: " + "; ".join(errors)
        )
        return None, ValidationError(
            "args",
            summary,
            {"route": route.name, "missing": missing, "errors": errors},
        )


# --------------------------------------------------------------------------- #
# Repair text
# --------------------------------------------------------------------------- #


def format_error_for_repair(err: ValidationError, *, max_chars: int = 800) -> str:
    """Render ``err`` as the corrective instruction for the repair segment (§4.5).

    The repair segment is appended **after** the query segment so the stable A+B
    prefix still hits the provider cache (§4.6); this function produces only the
    error-and-re-ask text. Including the model's prior output is the prompt
    builder's job — it owns truncation and redaction of anything the model echoed
    back (§7.4).

    The wording is deliberately corrective rather than conversational: it names
    what was wrong, states the legal alternatives, and re-asks. It never apologises
    to the model and never restates the whole schema (the schema is already in the
    cached prefix; repeating it would both cost tokens and invite the model to
    re-derive rather than repair).
    """
    if err.stage == "schema":
        errors = _string_list(err.detail.get("errors")) or [err.message]
        body = (
            "Your previous output could not be parsed into the required schema. "
            "Problems: " + "; ".join(errors[:_MAX_REPORTED_ERRORS]) + ". "
            "Return exactly one JSON object matching the schema, with no prose and "
            "no code fences."
        )
    elif err.stage == "route_ref":
        allowed = _string_list(err.detail.get("allowed"))
        listed = allowed[:_MAX_LISTED_ROUTES]
        suffix = ", ..." if len(allowed) > len(listed) else ""
        body = (
            f"{err.message}. You may only choose from these routes: "
            f"{', '.join(listed)}{suffix}. Pick one of them, or set kind to "
            f"'clarify' or 'abstain' if none of them fits."
        )
    else:
        missing = list(err.missing)
        errors = _string_list(err.detail.get("errors")) or [err.message]
        detail = (
            "The following required fields were absent: " + ", ".join(missing) + ". "
            if missing
            else "; ".join(errors[:_MAX_REPORTED_ERRORS]) + ". "
        )
        body = (
            f"The arguments you extracted did not validate. {detail}"
            "Re-read the user's message and fill them in from what it actually says. "
            "Do not invent values: if the message does not contain one, set kind to "
            "'clarify' and ask for it."
        )
    return _truncate(body, max_chars)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _describe(exc: PydanticValidationError) -> list[str]:
    """Flatten a Pydantic error into ``"field: message"`` strings, bounded."""
    described: list[str] = []
    for error in exc.errors()[:_MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        described.append(f"{location}: {error.get('msg', 'invalid')}")
    remaining = exc.error_count() - len(described)
    if remaining > 0:
        described.append(f"(+{remaining} more)")
    return described


def _missing_fields(exc: PydanticValidationError) -> list[str]:
    """Extract the field paths Pydantic reported as absent.

    Pydantic v2 reports an omitted required field as ``type == "missing"``; a
    ``None`` supplied for a non-optional field surfaces separately as a type error
    and is deliberately *not* counted as missing — the model did answer, it just
    answered wrongly, and asking the user to supply a field they were never asked
    about would be the wrong clarify question.
    """
    missing: list[str] = []
    for error in exc.errors():
        if error.get("type") != "missing":
            continue
        location = ".".join(str(part) for part in error.get("loc", ()))
        if location and location not in missing:
            missing.append(location)
    return missing


def _member_names(parsed: BaseModel) -> list[str]:
    """Route names referenced by a multi-route output's ``routes`` field."""
    names: list[str] = []
    for item in _sequence_attr(parsed, "routes"):
        if isinstance(item, str):
            if item:
                names.append(item)
            continue
        name = item.get("route") if isinstance(item, Mapping) else getattr(item, "route", None)
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _str_attr(obj: object, name: str) -> str | None:
    """Read ``obj.name`` (or ``obj[name]``) as a non-empty string, else ``None``."""
    value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    return value if isinstance(value, str) and value else None


def _sequence_attr(obj: object, name: str) -> Sequence[Any]:
    """Read ``obj.name`` as a sequence, tolerating absence and scalars."""
    value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _str_sequence(obj: object, name: str) -> list[str]:
    """Read ``obj.name`` as a list of non-empty strings."""
    return [item for item in _sequence_attr(obj, name) if isinstance(item, str) and item]


def _string_list(value: object) -> list[str]:
    """Coerce a ``detail`` entry to a list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _truncate(text: str, limit: int) -> str:
    """Hard-cap ``text``, marking the cut so the model knows it is not the whole story."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."
