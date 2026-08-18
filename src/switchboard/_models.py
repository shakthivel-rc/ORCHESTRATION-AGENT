"""Model-ID data tables: static capabilities, prices and deprecation dates.

This module is **pure data plus lookup logic**. It imports nothing but the stdlib and Pydantic —
no provider SDK, ever (plan §2.4). Adapters read from it; it never reads from adapters.

Why a table and not probing: plan §4.3 ships a *static* capability table keyed by
``(provider, model-family)`` at v0.1, because the report flags that per-provider support matrices
were not independently verified. A one-time runtime canary probe lands at v0.2 and will *override*
these rows; until then this file is the single place to edit when a provider ships a new model.

**Prices are a dated snapshot**, stamped by :data:`PRICE_TABLE_VERSION` (currently
``2026-08-07``), expressed in **USD per 1,000,000 tokens**. They drift; re-verify against provider
pricing pages before relying on them for billing. The discipline that makes the snapshot safe:

* An unknown model resolves to ``None`` from :func:`lookup`, and cost accounting therefore reports
  ``cost=None`` — **never a guessed number**. A missing price is a missing price.
* An individual price field is ``None`` whenever that specific rate is not published or not
  confidently known (e.g. an unannounced cache-read rate), even when the sibling fields are known.
* Downstream cost code must treat ``None`` as "not computable" and propagate it, not coerce to 0.0.

Capability semantics mirror ``providers.base.ClientCapabilities`` exactly (plan §4.1):

* ``structured`` — highest rung of the enforcement ladder the model supports:
  ``grammar`` > ``tool_strict`` > ``json_mode`` > ``none`` (plan §4.3).
* ``logprobs`` — token logprobs are retrievable, which is what arms the ``p_route`` signal (§6.2).
* ``caching`` — ``explicit`` (caller places breakpoints), ``implicit`` (automatic prefix cache), or
  ``none``.
* ``reasoning_toggle`` — the adapter can force extended reasoning **off**, which routing requires
  (13s vs 0.33s TTFT, plan §4.3). Models that reason by default and cannot be forced off are listed
  in :data:`REASONING_DEFAULT_MODELS` and warrant a Router construction warning.
"""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy, inside the function
    from switchboard.providers.base import ClientCapabilities

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_TABLE",
    "PRICE_TABLE_VERSION",
    "REASONING_DEFAULT_MODELS",
    "CachingMode",
    "ModelInfo",
    "StructuredMode",
    "capabilities_for",
    "is_reasoning_default",
    "lookup",
]

#: Date stamp of the price/capability snapshot below. Bump whenever a price row changes; audit
#: records and cost reports carry it so an old log can be re-priced against the right table.
PRICE_TABLE_VERSION = "2026-08-07"

StructuredMode = Literal["grammar", "tool_strict", "json_mode", "none"]
CachingMode = Literal["explicit", "implicit", "none"]


class ModelInfo(BaseModel):
    """One row of the static model table. Frozen: the table is read-only shared state."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: str
    """Canonical, provider-prefix-free model id (the table key), e.g. ``gemini-2.5-flash-lite``."""

    provider: str
    """Vendor that publishes the model: ``google`` | ``openai`` | ``anthropic`` | ..."""

    structured: StructuredMode = "none"
    """Highest structure-enforcement rung the model supports (plan §4.3 ladder)."""

    logprobs: bool = False
    """Whether token logprobs can be requested (arms ``p_route``, plan §6.2)."""

    caching: CachingMode = "none"
    """Prompt-cache style: caller-placed breakpoints, automatic prefix cache, or none."""

    reasoning_toggle: bool = False
    """Whether extended reasoning can be forced OFF (required for routing, plan §4.3)."""

    price_in: float | None = None
    """USD per 1M uncached input tokens. ``None`` = unknown; never guess."""

    price_cached: float | None = None
    """USD per 1M cached-read input tokens. ``None`` = unknown or unpublished."""

    price_out: float | None = None
    """USD per 1M output tokens (reasoning tokens included where the provider bills them so)."""

    deprecated_after: date | None = None
    """Retirement date announced by the provider; Router warns past it [v0.2]. ``None`` = none announced."""

    notes: str = ""
    """Free-text operator guidance: migration target, confidence caveats, tier positioning."""


def _row(
    model_id: str,
    provider: str,
    *,
    structured: StructuredMode,
    logprobs: bool,
    caching: CachingMode,
    reasoning_toggle: bool,
    price_in: float | None = None,
    price_cached: float | None = None,
    price_out: float | None = None,
    deprecated_after: date | None = None,
    notes: str = "",
) -> ModelInfo:
    """Build a :class:`ModelInfo`; keeps the table below readable at one row per line-group."""
    return ModelInfo(
        model_id=model_id,
        provider=provider,
        structured=structured,
        logprobs=logprobs,
        caching=caching,
        reasoning_toggle=reasoning_toggle,
        price_in=price_in,
        price_cached=price_cached,
        price_out=price_out,
        deprecated_after=deprecated_after,
        notes=notes,
    )


#: Static capability + price table, keyed by canonical model id (plan §4.3 data, §4.7 defaults).
#: Keys are family-prefix friendly: :func:`lookup` resolves ``gpt-5-nano-2026-01-01`` and
#: ``openai/gpt-5-nano`` to the ``gpt-5-nano`` row via longest-prefix match.
MODEL_TABLE: dict[str, ModelInfo] = {
    # --- Google / Gemini -------------------------------------------------------------------
    "gemini-2.5-flash-lite": _row(
        "gemini-2.5-flash-lite",
        "google",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=0.10,
        price_cached=0.025,
        price_out=0.40,
        deprecated_after=date(2026, 10, 16),
        notes=(
            "Primary default (plan §4.7): ~0.33s TTFT, responseSchema grammar rung. "
            "Retires 2026-10-16; documented migration target is gemini-3.1-flash-lite. "
            "Force thinkingBudget=0."
        ),
    ),
    "gemini-2.5-flash": _row(
        "gemini-2.5-flash",
        "google",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=0.30,
        price_cached=0.075,
        price_out=2.50,
        notes="Step up from flash-lite when routing accuracy needs it; non-reasoning at thinkingBudget=0.",
    ),
    "gemini-2.5-pro": _row(
        "gemini-2.5-pro",
        "google",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=False,
        price_in=1.25,
        price_cached=0.31,
        price_out=10.00,
        notes=(
            "Thinking cannot be disabled -> reasoning-default; misconfiguration for routing "
            "(seconds of TTFT). Listed in REASONING_DEFAULT_MODELS."
        ),
    ),
    "gemini-3.1-flash-lite": _row(
        "gemini-3.1-flash-lite",
        "google",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=0.25,
        price_cached=None,  # cache-read rate not published at snapshot time -> None, not a guess
        price_out=1.50,
        notes="Successor tier and documented migration target for gemini-2.5-flash-lite (plan §4.7).",
    ),
    # --- OpenAI ----------------------------------------------------------------------------
    "gpt-5-nano": _row(
        "gpt-5-nano",
        "openai",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=0.05,
        price_cached=0.005,
        price_out=0.40,
        notes=(
            "Alternate default (plan §4.7): cheapest input, sub-second TTFT, strict Structured "
            "Outputs + logprobs. Adapter forces reasoning_effort='minimal'."
        ),
    ),
    "gpt-5-mini": _row(
        "gpt-5-mini",
        "openai",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=0.25,
        price_cached=0.025,
        price_out=2.00,
        notes="Accuracy step up from gpt-5-nano at the same strict/logprobs capability set.",
    ),
    "gpt-5": _row(
        "gpt-5",
        "openai",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=True,
        price_in=1.25,
        price_cached=0.125,
        price_out=10.00,
        notes="Oversized for routing; listed for cost accounting and eval baselines, not as a default.",
    ),
    "gpt-4.1-mini": _row(
        "gpt-4.1-mini",
        "openai",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=False,
        price_in=0.40,
        price_cached=0.10,
        price_out=1.60,
        notes="Non-reasoning by construction; useful when a gpt-5-family reasoning toggle is unavailable.",
    ),
    "gpt-4.1-nano": _row(
        "gpt-4.1-nano",
        "openai",
        structured="grammar",
        logprobs=True,
        caching="implicit",
        reasoning_toggle=False,
        price_in=0.10,
        price_cached=0.025,
        price_out=0.40,
        notes="Cheapest non-reasoning OpenAI routing tier.",
    ),
    # --- Anthropic -------------------------------------------------------------------------
    "claude-haiku-4-5": _row(
        "claude-haiku-4-5",
        "anthropic",
        structured="tool_strict",
        logprobs=False,
        caching="explicit",
        reasoning_toggle=True,
        price_in=1.00,
        price_cached=0.10,
        price_out=5.00,
        notes=(
            "Anthropic-shop default (plan §4.7): <600ms TTFT, forced-tool structured output, strong "
            "explicit cache_control. No logprobs -> confidence falls to the vote rung [v0.2]."
        ),
    ),
    "claude-sonnet-4-5": _row(
        "claude-sonnet-4-5",
        "anthropic",
        structured="tool_strict",
        logprobs=False,
        caching="explicit",
        reasoning_toggle=True,
        price_in=3.00,
        price_cached=0.30,
        price_out=15.00,
        notes="Accuracy tier for hard catalogs; same no-logprobs confidence caveat as Haiku.",
    ),
    "claude-opus-4-5": _row(
        "claude-opus-4-5",
        "anthropic",
        structured="tool_strict",
        logprobs=False,
        caching="explicit",
        reasoning_toggle=True,
        price_in=5.00,
        price_cached=0.50,
        price_out=25.00,
        notes="Oversized for routing; listed for cost accounting and eval baselines.",
    ),
}

#: The primary default model (plan §4.7 / Appendix A). Small, non-reasoning, grammar rung.
DEFAULT_MODEL = "gemini-2.5-flash-lite"

#: Models that engage extended reasoning **by default and cannot be forced off**. Configuring one
#: for routing is a misconfiguration (plan §4.3: 13s vs 0.33s TTFT), so the Router warns at
#: construction. Models that reason by default but expose an off switch (the gpt-5 family via
#: ``reasoning_effort="minimal"``, Anthropic via omitting ``thinking``, Gemini via
#: ``thinkingBudget=0``) carry ``reasoning_toggle=True`` instead and are handled silently by their
#: adapter. Entries are matched by family prefix — use :func:`is_reasoning_default`, which
#: normalizes provider prefixes and dated suffixes, rather than a bare ``in`` test.
REASONING_DEFAULT_MODELS: frozenset[str] = frozenset(
    {
        "gemini-2.5-pro",
        "gpt-5-pro",
        "o1",
        "o1-pro",
        "o3",
        "o3-pro",
        "o4-mini",
        "deepseek-r1",
    }
)

# Characters that may follow a family prefix for it to count as a family match: version dashes,
# date suffixes, vendor revision markers. Prevents "gpt-5" from matching a hypothetical "gpt-50".
_BOUNDARY = frozenset("-_.@:/ ")

# Leading namespace segment, e.g. "instructor:", "openai/", "us.", "vertex_ai/".
_NAMESPACE_RE = re.compile(r"^[a-z0-9_\-]+[:/.](.+)$")


def _normalized_forms(model_id: str) -> list[str]:
    """Candidate spellings of ``model_id``, most specific first.

    Handles the shapes the router actually receives: bare ids (``gpt-5-nano``), client specs
    (``instructor:openai/gpt-5-nano``), vendor-namespaced ids (``vertex_ai/gemini-2.5-flash-lite``)
    and Bedrock-style ids (``us.anthropic.claude-haiku-4-5-20251001-v1:0``) by progressively
    stripping leading namespace segments.
    """
    forms: list[str] = []

    def add(value: str) -> None:
        cleaned = value.strip().strip("/:. ")
        if cleaned and cleaned not in forms:
            forms.append(cleaned)

    current = model_id.strip().lower()
    add(current)
    for _ in range(6):  # bounded: no pathological input can spin here
        match = _NAMESPACE_RE.match(current)
        if match is None:
            break
        current = match.group(1)
        add(current)
    return forms


def _prefix_match(form: str, keys: tuple[str, ...]) -> str | None:
    """Longest key that ``form`` extends at a legal family boundary, or ``None``."""
    best: str | None = None
    for key in keys:
        if not form.startswith(key):
            continue
        if len(form) > len(key) and form[len(key)] not in _BOUNDARY:
            continue  # "gpt-50" must not match family "gpt-5"
        if best is None or len(key) > len(best):
            best = key
    return best


def lookup(model_id: str) -> ModelInfo | None:
    """Resolve a model id to its :class:`ModelInfo`, or ``None`` if it is unknown.

    Resolution is two-pass: **exact** match on any normalized spelling first, then **longest-prefix
    (family)** match. So ``"openai/gpt-5-nano"``, ``"gpt-5-nano"`` and ``"gpt-5-nano-2026-01-01"``
    all resolve to the ``gpt-5-nano`` row, while ``"gpt-5-mini-2026-01-01"`` resolves to
    ``gpt-5-mini`` rather than the shorter ``gpt-5``.

    Returning ``None`` is a first-class outcome: unknown models get conservative capabilities and
    ``cost=None`` downstream, never a guessed price or an assumed strict-schema rung.
    """
    if not model_id:
        return None
    forms = _normalized_forms(model_id)

    for form in forms:  # pass 1: exact
        row = MODEL_TABLE.get(form)
        if row is not None:
            return row

    keys = tuple(MODEL_TABLE)
    for form in forms:  # pass 2: longest-prefix / family
        key = _prefix_match(form, keys)
        if key is not None:
            return MODEL_TABLE[key]
    return None


def capabilities_for(model_id: str) -> ClientCapabilities | None:
    """Static :class:`~switchboard.providers.base.ClientCapabilities` for ``model_id``.

    Returns ``None`` for unknown models so the caller can decide the conservative default (adapters
    fall back to an all-defaults ``ClientCapabilities()``, which arms the full validate-and-retry
    loop and leaves the confidence engine inert — plan §4.1, §6.3).

    ``ClientCapabilities`` is imported **lazily, inside this function**: ``providers.base`` is
    allowed to depend on core data, so a module-level import here would close an import cycle.
    """
    # Deliberately function-local: breaks the providers.base <-> _models import cycle.
    from switchboard.providers.base import ClientCapabilities

    info = lookup(model_id)
    if info is None:
        return None
    return ClientCapabilities(
        structured=info.structured,
        logprobs=info.logprobs,
        caching=info.caching,
        reasoning_toggle=info.reasoning_toggle,
    )


def is_reasoning_default(model_id: str) -> bool:
    """Whether ``model_id`` reasons by default and cannot be forced off (plan §4.3 warning).

    Applies the same normalization/family matching as :func:`lookup`, so ``"openai/o3-mini"`` and
    ``"gemini-2.5-pro-preview-06-05"`` are both recognized.
    """
    if not model_id:
        return False
    keys = tuple(REASONING_DEFAULT_MODELS)
    for form in _normalized_forms(model_id):
        if form in REASONING_DEFAULT_MODELS or _prefix_match(form, keys) is not None:
            return True
    return False
