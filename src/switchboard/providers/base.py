"""Normative provider I/O contract (plan §4.1; §13 ruling #1).

``LLMRequest`` / ``LLMResult`` / ``ClientCapabilities`` / ``TokenLP`` are the
**only** provider types in the codebase — the confidence engine, every adapter,
the audit record and BYO-callable coercion all consume exactly these. Three
competing provider contracts were reconciled into this one (§13 ruling #1), and
capabilities are enum-valued because the graceful-degradation ladder (§4.3)
needs to compare rungs, not booleans.

This module is import-safe with Pydantic alone: it declares Protocols and models
only, and never imports a provider SDK. The SDK imports live in the sibling
``*_adapter.py`` modules (plan §2.4).
"""

from __future__ import annotations

from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AsyncLLMClient",
    "ClientCapabilities",
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "PromptSegment",
    "TokenLP",
    "Usage",
    "capabilities_of",
]

T = TypeVar("T", bound=BaseModel)


class PromptSegment(BaseModel):
    """One cache-addressable slice of the routing prompt (plan §4.1, §4.6).

    Segments are ordered stable→variable so provider prefix caches hit on every
    request after the first per ``(registry_version, view)``: A = system rules,
    B = the entitlement-filtered route directory, C = the shortlist pointer,
    D = the redacted user query, always last.
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user"] = "user"
    content: str
    cache: Literal["stable", "variable"] = "variable"
    cache_key: str | None = None
    """Provider cache key, e.g. ``"sb1:reg=3fa9c21b04d1:view=7be2d90a11ef"`` (§4.6)."""


class LLMRequest(BaseModel):
    """One routing call as handed to an adapter (plan §4.1).

    ``output_schema`` is built per call by ``engine/schema.py`` with ``rationale``
    ordered first so free-text reasoning is emitted *before* the committed route
    token (plan §3.4, §4.4).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    segments: tuple[PromptSegment, ...]
    """Ordered; stable segments always first (plan §4.6)."""
    output_schema: type[BaseModel]
    temperature: float = 0.0
    max_tokens: int = 512
    want_logprobs: bool = True
    """Adapters map this down silently when ``capabilities.logprobs`` is False."""
    seed: int | None = None

    def with_segment(self, segment: PromptSegment) -> LLMRequest:
        """Return a copy with ``segment`` appended after the existing segments.

        Used by the validate-and-retry loop (plan §4.5): the repair segment
        (Pydantic error + truncated prior output + re-ask) is appended **after**
        the query segment so the cached A+B prefix still hits.
        """
        return self.model_copy(update={"segments": (*self.segments, segment)})


class TokenLP(BaseModel):
    """One sampled token with its logprob and alternatives (plan §4.1).

    ``top`` carries the provider's ``top_logprobs`` alternatives; the margin
    signal (§6.2) needs them and returns ``None`` rather than guessing when the
    runner-up continuation is absent.
    """

    token: str
    logprob: float
    top: list[tuple[str, float]] = Field(default_factory=list)


class Usage(BaseModel):
    """Token accounting for one or more LLM calls (plan §4.1, §8.4).

    ``cached_input_tokens`` is priced at the cache-read rate, which is the whole
    point of the cache-shaped prompt layout in §4.6.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across self-consistency votes and repair retries.

        ``AuditRecord.usage`` is the sum over *every* LLM call a decision made
        (plan §8.2: "llm_calls > 1 under vote/retry"), so this is the accumulator
        the loop uses. ``sum(usages, Usage())`` works.
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMResult(BaseModel, Generic[T]):
    """What every adapter returns (plan §4.1).

    ``raw_text`` is **always** kept even when ``parsed`` succeeded: it feeds the
    audit record, the repair prompt (§4.5) and distillation (§8.5).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed: T | None
    """``None`` means the adapter could not yield schema-valid output."""
    raw_text: str
    token_logprobs: list[TokenLP] | None = None
    """``None`` when the provider path has no logprobs (plan §6.1)."""
    usage: Usage
    model_id: str
    attempts: int = 1
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class ClientCapabilities(BaseModel):
    """What a client can actually do — drives the degradation ladder (plan §4.3).

    The engine branches on these rather than flattening to the weakest provider
    (report risk #3): no strict schema → validate-and-retry; no logprobs → the
    confidence ladder drops a rung; no explicit caching → the prompt is still
    cache-shaped for implicit caching.

    All defaults are the conservative rung, so a bare BYO callable with no
    ``capabilities`` attribute satisfies the contract safely.
    """

    structured: Literal["grammar", "tool_strict", "json_mode", "none"] = "none"
    logprobs: bool = False
    caching: Literal["explicit", "implicit", "none"] = "none"
    parallel_tool_calls: bool = False
    reasoning_toggle: bool = False
    """True when the adapter can force reasoning OFF — required for routing (§4.3)."""


@runtime_checkable
class LLMClient(Protocol):
    """Synchronous provider contract (plan §4.1). One required method."""

    capabilities: ClientCapabilities

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """Issue one routing call and return its result."""
        ...


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Asynchronous mirror of :class:`LLMClient` (plan §4.1, §2.5)."""

    capabilities: ClientCapabilities

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """Issue one routing call and return its result."""
        ...


def capabilities_of(client: object) -> ClientCapabilities:
    """Probe a client's capabilities, defaulting to the conservative rung.

    Plan §4.1 specifies capabilities are read via
    ``getattr(client, "capabilities", ClientCapabilities())`` so that a bare BYO
    callable satisfies the contract with all-conservative defaults (which in turn
    arms the full validate-and-retry loop and marks the confidence engine inert,
    §6.3). This is that probe, in one place.
    """
    caps = getattr(client, "capabilities", None)
    return caps if isinstance(caps, ClientCapabilities) else ClientCapabilities()
