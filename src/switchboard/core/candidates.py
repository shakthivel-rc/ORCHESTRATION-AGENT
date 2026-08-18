"""``Candidate`` / ``ShortlistResult`` — the shared retrieval models (plan §5.1).

These are the ONE candidate representation in the codebase: the ``Shortlister``
Protocol returns them, the [v0.2] streaming ``shortlisted`` event carries them,
and ``AuditRecord.shortlist`` stores them verbatim. Nothing re-models a
candidate anywhere else.

The Router treats every shortlister as **untrusted on entitlements**: candidates
are re-intersected with the entitled set after shortlisting, so a buggy backend
can degrade recall but can never leak a route (plan §5.1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = ["Candidate", "ShortlistResult"]

CandidateSource = Literal["bm25", "embed", "hybrid", "pinned", "all"]
"""Which stage produced a candidate. ``"all"`` = small-catalog bypass emitted the
whole entitled registry (§5.3); ``"pinned"`` = ``Route(pinned=True)`` appended
regardless of score, so the fallback path can never be retrieved out of
existence (§5.5)."""


class Candidate(BaseModel):
    """One shortlisted route (plan §5.1)."""

    route_name: str
    score: float
    """Backend-native; comparable only *within* one backend — hybrid fusion is
    therefore rank-based (RRF), never score-normalised (§5.2)."""
    rank: int
    """0-based rank **pre-shuffle**. The seeded shuffle (§5.6) reorders the
    prompt; this preserves what the retriever actually thought, for audit and
    for the retrieval-gap vs decision-gap decomposition (§9.2)."""
    source: CandidateSource


class ShortlistResult(BaseModel):
    """The outcome of one shortlist pass (plan §5.1)."""

    candidates: list[Candidate]
    skipped: bool = False
    """True when the small-catalog bypass fired (< ``shortlist_min_routes``, §5.3):
    the full entitled registry goes into the prompt as a stable cached prefix."""
    weak_retrieval: bool = False
    """True when fewer than 3 candidates scored above zero (§5.5). The policy
    stage reads this and biases toward ``clarify`` — a query nothing matches is a
    clarification candidate, not a guessing opportunity."""
    index_key: str | None = None
    """``f"{registry_version}:{shortlister.fingerprint}:{scope}"`` (§5.4)."""

    @property
    def names(self) -> tuple[str, ...]:
        """Candidate route names in result order.

        The candidate set the validator enforces membership against (§4.5 step 2)
        and the enum the dynamic wire schema is built from (§4.4).
        """
        return tuple(c.route_name for c in self.candidates)
