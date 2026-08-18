"""Shortlist & retrieval subsystem (plan §5).

The shortlister is the *retrieve-then-decide* stage — an optimization **inside**
an LLM-first design. The LLM always makes the final call; the shortlister only
controls what it sees. Two invariants govern everything in this module:

1. **Filter before truncate (§5.9).** ``shortlist()`` receives the entitled set
   and restricts scoring to it *before* taking the top-K. Truncate-then-filter
   can starve a tenant's routes down to nothing.
2. **Untrusted on entitlements (§5.1).** Even so, the Router re-intersects the
   returned candidates with the entitled set, so a bug here can degrade recall
   but can never leak a route.

Everything in this file is **pure Python + Pydantic** (§2.4, §13 ruling #7): no
``rank-bm25``, no ``numpy``. BM25 shortlisting is core functionality, not an
extra — the fielded BM25F variant we need isn't what ``rank-bm25`` implements
anyway, and a compact hand-rolled index is cheaper than a dependency.

Contents:

* :func:`tokenize` — the shared lexical tokenizer (§5.2).
* :class:`Shortlister` — the Protocol every backend implements (§5.1).
* :class:`BM25Shortlister` — the default, fielded BM25F (§5.2).
* :class:`EmbeddingShortlister` — BYO ``embed`` callable, brute-force cosine (§5.2).
* :class:`AutoShortlister` — the ``"auto"`` policy: bypass below 25 routes (§5.3).
* :func:`effective_k` — the evidence-bound K table and its clamp (§5.3).
* :class:`IndexStore` / :class:`MemoryIndexStore` — index lifecycle (§5.4).
* :func:`resolve_shortlister` — the ``shortlist=...`` config DSL (§3.3).

The seeded shuffle of §5.6 is deliberately **not** here: candidate *ordering* is
a prompt-assembly concern and lives in ``engine/prompt.py``. This module always
returns candidates in score order with ``Candidate.rank`` set pre-shuffle.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from switchboard.core.audit import sha256_hex
from switchboard.core.candidates import Candidate, ShortlistResult
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from switchboard.core.candidates import CandidateSource
    from switchboard.core.context import RequestContext
    from switchboard.core.route import Route

__all__ = [
    "BM25_B",
    "BM25_K1",
    "DEFAULT_FIELD_WEIGHTS",
    "DEFAULT_SCOPE",
    "DEFAULT_SHORTLIST_K",
    "DEFAULT_SHORTLIST_MIN_ROUTES",
    "FIELD_NAMES",
    "K_MAX",
    "K_MIN",
    "MIN_POSITIVE_CANDIDATES",
    "STOPWORDS",
    "TOKENIZER_VERSION",
    "AsyncEmbedFn",
    "AutoShortlister",
    "BM25Shortlister",
    "EmbedFn",
    "EmbeddingShortlister",
    "IndexStore",
    "MemoryIndexStore",
    "Shortlister",
    "effective_k",
    "resolve_shortlister",
    "tokenize",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Authoritative defaults (plan Appendix A, §5.3).
# --------------------------------------------------------------------------- #

DEFAULT_SHORTLIST_K = 10
"""``shortlist_k`` default (Appendix A). Also the §5.3 band value for N < 150."""

DEFAULT_SHORTLIST_MIN_ROUTES = 25
"""``shortlist_min_routes``: below this many entitled routes ``"auto"`` bypasses
retrieval entirely (§5.3). OpenAI advises <20 functions/turn and Anthropic
documents degradation past ~30-50, so below this band retrieval adds miss-risk
for no win — and the full registry stays a stable, cacheable prompt prefix."""

K_MIN = 5
"""Lower bound of the ``shortlist_k`` clamp (§5.3)."""

K_MAX = 20
"""Upper bound of the ``shortlist_k`` clamp (§5.3). Above 20 needs
``allow_oversized_k=True`` — see :func:`effective_k`."""

MIN_POSITIVE_CANDIDATES = 3
"""Fewer than this many above-zero scores triggers the degenerate-retrieval
handling of §5.5."""

DEFAULT_SCOPE = "*"
"""Index scope: ``"*"`` is the shared index over the full registry. A real tenant
id appears here only under opt-in ``per_tenant_index`` [v0.2] (§5.4, §13 #15)."""

FIELD_NAMES: tuple[str, ...] = ("name", "examples", "description", "tags")
"""The fielded document layout, in a fixed order so serialized indexes and
embedded documents are byte-deterministic (§5.2)."""

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "name": 3.0,
    "examples": 2.0,
    "description": 1.0,
    "tags": 1.0,
}
"""BM25F field weights (Appendix A). ``examples`` outranks ``description``
because utterance-shaped text matches utterance-shaped queries (§5.2)."""

BM25_K1 = 1.5
"""Okapi term-frequency saturation (Appendix A)."""

BM25_B = 0.75
"""Okapi length-normalisation strength (Appendix A)."""

TOKENIZER_VERSION = "1"
"""Bumped whenever :func:`tokenize` changes semantics. It feeds
:attr:`BM25Shortlister.fingerprint`, so a tokenizer change invalidates every
persisted index (§5.4)."""

_BLOB_VERSION = 1
"""Serialisation version stamped into every :class:`IndexStore` blob."""


# --------------------------------------------------------------------------- #
# Tokenizer (plan §5.2): lowercase / alnum, identifier-aware, no stemming.
# --------------------------------------------------------------------------- #

STOPWORDS: frozenset[str] = frozenset(
    (
        "a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at",
        "be", "been", "being", "but", "by", "can", "cannot", "could", "did", "do",
        "does", "doing", "done", "for", "from", "get", "got", "had", "has", "have",
        "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "just", "me", "my", "no", "nor", "not", "of", "on",
        "once", "only", "or", "other", "our", "out", "over", "own", "please", "s",
        "so", "some", "such", "t", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "too", "up", "us", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your",
    )
)
"""A deliberately small English stoplist. No stemming and no language detection:
on short route corpora the marginal gain does not justify a dependency (§5.2),
and multilingual / paraphrase-heavy traffic is what
:class:`EmbeddingShortlister` exists for."""

_CHUNK_RE = re.compile(r"[^\W_]+")
"""Alphanumeric runs — splits on whitespace, punctuation and ``_`` in one pass,
so ``snake_case`` falls apart for free."""

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
"""``camelCase`` / ``PascalCase`` / acronym splitting, applied only to ASCII
chunks. ``getOrderStatus`` -> ``get order status``; ``HTTPServer`` -> ``http
server``."""


def _split_chunk(chunk: str) -> list[str]:
    """Split one alphanumeric chunk into lowercased sub-tokens.

    Identifier splitting is applied only to *mixed-case ASCII* chunks. A
    non-ASCII chunk (``"café"``, ``"東京"``) is emitted whole: the camel regex
    would shear it at the first non-ASCII character, which is worse than not
    splitting at all.
    """
    if chunk.isascii() and not (chunk.islower() or chunk.isupper() or chunk.isdigit()):
        return [part.lower() for part in _CAMEL_RE.findall(chunk)]
    return [chunk.lower()]


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` for lexical retrieval (plan §5.2).

    Lowercase, alphanumeric, identifier-aware (``snake_case`` and ``camelCase``
    are split), stopword-filtered, **not** stemmed.

    One deliberate exception to stopword filtering: if filtering would empty an
    otherwise non-empty token list, the unfiltered tokens are returned instead.
    A route literally named ``"how_to"``, or a query of ``"what is it"``, must
    still produce *something* to match on — an empty document would silently
    drop out of the index, and an empty query would look like weak retrieval for
    entirely the wrong reason.

    Example::

        >>> tokenize("Cancel the getOrderStatus_v2 request")
        ['cancel', 'order', 'status', 'v2', 'request']
    """
    tokens: list[str] = []
    for chunk in _CHUNK_RE.findall(text):
        tokens.extend(_split_chunk(chunk))
    filtered = [token for token in tokens if token not in STOPWORDS]
    if not filtered and tokens:
        return tokens
    return filtered


def _field_texts(route: Route) -> tuple[str, ...]:
    """Render one route as the fielded document, in :data:`FIELD_NAMES` order."""
    return (
        route.name,
        " ".join(route.examples),
        route.description,
        " ".join(sorted(route.tags)),
    )


# --------------------------------------------------------------------------- #
# Protocols (plan §5.1, §5.4).
# --------------------------------------------------------------------------- #

EmbedFn = Callable[[list[str]], "Sequence[Sequence[float]]"]
"""BYO embedding callable — works with **zero extras** (§10.2). Anything
sequence-of-sequence-of-number is accepted, so a numpy-returning embedder needs
no conversion on the caller's side and no numpy import on ours."""

AsyncEmbedFn = Callable[[list[str]], Awaitable["Sequence[Sequence[float]]"]]
"""Async BYO embedding callable; awaited by ``abuild`` / ``ashortlist``."""


@runtime_checkable
class Shortlister(Protocol):
    """The retrieval backend contract (plan §5.1).

    Implementations must be safe to share across threads and tasks: one Router
    instance holds one shortlister and calls :meth:`shortlist` concurrently.
    Only :meth:`build` mutates state, and it is idempotent per
    ``registry_version``.
    """

    def build(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """(Re)build the index over ``routes``. Idempotent for a given version."""
        ...

    def shortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Return at most ``k`` candidates drawn **only** from ``allowed``."""
        ...

    async def abuild(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Async twin of :meth:`build`, single-flighted against stampedes (§5.4)."""
        ...

    async def ashortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Async twin of :meth:`shortlist`."""
        ...

    @property
    def fingerprint(self) -> str:
        """Hash of the backend's configuration — variant, weights, tokenizer or
        embedding-model version. Half of the index key, so a config change
        invalidates persisted indexes exactly like a registry change (§5.4)."""
        ...


@runtime_checkable
class IndexStore(Protocol):
    """Where a built index is persisted between processes (plan §5.4).

    ``FileIndexStore`` is [v0.2]; Redis / S3 are two user methods. Blobs are
    JSON — **never pickle** (§5.4), because an index blob is data that crosses a
    trust boundary.
    """

    def load(self, key: str) -> bytes | None:
        """Return the blob stored under ``key``, or ``None``."""
        ...

    def save(self, key: str, blob: bytes) -> None:
        """Store ``blob`` under ``key``."""
        ...


class MemoryIndexStore:
    """Process-local :class:`IndexStore` [v0.1] (plan §5.4).

    Useful for sharing one built index between several Router instances in the
    same process, and for asserting rebuild behaviour in tests. Thread-safe.
    """

    __slots__ = ("_blobs", "_lock")

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def load(self, key: str) -> bytes | None:
        with self._lock:
            return self._blobs.get(key)

    def save(self, key: str, blob: bytes) -> None:
        with self._lock:
            self._blobs[key] = bytes(blob)

    def clear(self) -> None:
        """Drop every stored blob (forces a rebuild on the next ``build``)."""
        with self._lock:
            self._blobs.clear()

    def keys(self) -> tuple[str, ...]:
        """The index keys currently held, sorted."""
        with self._lock:
            return tuple(sorted(self._blobs))

    def __len__(self) -> int:
        with self._lock:
            return len(self._blobs)

    def __repr__(self) -> str:
        return f"MemoryIndexStore(entries={len(self)})"


# --------------------------------------------------------------------------- #
# K sizing (plan §5.3).
# --------------------------------------------------------------------------- #


def _band_k(n_entitled: int) -> int:
    """The §5.3 evidence-bound K for an entitled-set size."""
    if n_entitled < 150:
        return 10  # RAG-MCP: 43.13% vs 13.62%, >50% token cut.
    if n_entitled < 1000:
        return 15  # 110-agent enterprise-routing study: +10-11 F1 recovery.
    return 20  # Hierarchical routing is [v0.5]; until then K=20 + lint pressure.


def effective_k(
    n_entitled: int,
    configured_k: int | None = None,
    *,
    allow_oversized_k: bool = False,
) -> int:
    """Resolve the K to pass to :meth:`Shortlister.shortlist` (plan §5.3).

    Args:
        n_entitled: size of the post-entitlement route set.
        configured_k: the caller's ``shortlist_k``. ``None`` means "size it from
            the §5.3 band table" — 10 below 150 entitled routes, 15 for
            150-999, 20 at 1000+. An explicit int overrides the table (the
            caller asked for a specific K) and is only clamped.
        allow_oversized_k: permit K > 20. Off by default.

    Returns:
        K, clamped to ``[5, 20]`` unless ``allow_oversized_k`` is set.

    Raises:
        ConfigError: ``configured_k < 1`` — a shortlist of zero routes cannot
            produce a decision (§3.8).

    **Never large K.** K=100 costs 10-16 accuracy points at ~10x the tokens (the
    "99% Success Paradox"), which is why exceeding the clamp is opt-in and
    always logged.
    """
    requested = _band_k(n_entitled) if configured_k is None else int(configured_k)
    if requested < 1:
        raise ConfigError(f"shortlist_k must be >= 1, got {requested}")
    if requested > K_MAX:
        if allow_oversized_k:
            logger.warning(
                f"shortlist_k={requested} exceeds the [{K_MIN}, {K_MAX}] clamp and was allowed "
                f"via allow_oversized_k=True. Recall the 99% Success Paradox: K=100 costs 10-16 "
                f"accuracy points at ~10x the prompt tokens. Prefer better route descriptions "
                f"(switchboard enrich) over a bigger K."
            )
            return requested
        logger.warning(
            f"shortlist_k={requested} exceeds the [{K_MIN}, {K_MAX}] clamp and was reduced to "
            f"{K_MAX} (99% Success Paradox: K=100 costs 10-16 accuracy points at ~10x the prompt "
            f"tokens). Pass allow_oversized_k=True to override."
        )
        return K_MAX
    return max(requested, K_MIN)


# --------------------------------------------------------------------------- #
# Shared index bookkeeping.
# --------------------------------------------------------------------------- #


class _BaseShortlister(ABC):
    """Route bookkeeping, locking and result assembly shared by the backends.

    Not public API and not part of the :class:`Shortlister` Protocol — it just
    keeps the degenerate-retrieval, pinned-route and entitlement rules in
    exactly one place, so BM25 and embeddings cannot drift apart.

    ``top_k`` is the K preference parsed from the config DSL (``"bm25:top_k=12"``)
    or ``None``; the Router reads it to seed ``shortlist_k``. :meth:`shortlist`
    itself always takes K from its argument.
    """

    __slots__ = (
        "_alock",
        "_aloop",
        "_lock",
        "_min_routes",
        "_names",
        "_pinned",
        "_registry_version",
        "_scope",
        "_store",
        "top_k",
    )

    def __init__(
        self,
        *,
        min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
        store: IndexStore | None = None,
        scope: str = DEFAULT_SCOPE,
        top_k: int | None = None,
    ) -> None:
        if min_routes < 0:
            raise ConfigError(f"shortlist_min_routes must be >= 0, got {min_routes}")
        self._min_routes = min_routes
        self._store = store
        self._scope = scope
        self.top_k = top_k
        self._lock = threading.Lock()
        self._alock: asyncio.Lock | None = None
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._registry_version: str | None = None
        self._names: tuple[str, ...] = ()
        self._pinned: tuple[int, ...] = ()

    # -- identity / lifecycle ----------------------------------------------- #

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Hash of this backend's configuration (§5.4)."""

    @property
    def registry_version(self) -> str | None:
        """The ``registry_version`` this index was built for, or ``None``."""
        return self._registry_version

    @property
    def is_built(self) -> bool:
        """Whether an index is present."""
        return self._registry_version is not None

    @property
    def min_routes(self) -> int:
        """The bypass / degenerate-fallback threshold (``shortlist_min_routes``)."""
        return self._min_routes

    @property
    def scope(self) -> str:
        """Index scope — ``"*"`` shared, or a tenant id under ``per_tenant_index`` [v0.2]."""
        return self._scope

    @property
    def index_key(self) -> str | None:
        """``f"{registry_version}:{fingerprint}:{scope}"`` (§5.4), or ``None``."""
        if self._registry_version is None:
            return None
        return self._index_key_for(self._registry_version)

    def _index_key_for(self, registry_version: str) -> str:
        return f"{registry_version}:{self.fingerprint}:{self._scope}"

    def _async_lock(self) -> asyncio.Lock:
        """Single-flight lock for async rebuilds (§5.4: "no rebuild stampedes").

        Recreated if the shortlister is driven from a different event loop,
        since an ``asyncio.Lock`` binds to the loop that first awaits it.
        """
        loop = asyncio.get_running_loop()
        if self._alock is None or self._aloop is not loop:
            self._alock = asyncio.Lock()
            self._aloop = loop
        return self._alock

    def _record_routes(self, routes: Sequence[Route], registry_version: str) -> None:
        """Store canonical order and pinned positions. Callers hold ``_lock``."""
        self._names = tuple(route.name for route in routes)
        self._pinned = tuple(i for i, route in enumerate(routes) if route.pinned)
        self._registry_version = registry_version

    def _require_built(self) -> None:
        if self._registry_version is None:
            raise ConfigError(
                f"{type(self).__name__} has no index; call build(routes, "
                f"registry_version=...) before shortlist(). The Router does this lazily on "
                f"every request, so seeing this means the shortlister was driven directly."
            )

    # -- entitlement + assembly --------------------------------------------- #

    def _allowed_docs(self, allowed: AbstractSet[str]) -> list[int]:
        """Entitled doc ids in **canonical registry order** (§5.9).

        This is the filter-before-truncate step: every scoring path below
        consumes this list, so a route outside ``allowed`` is never scored, let
        alone ranked or truncated against.
        """
        return [i for i, name in enumerate(self._names) if name in allowed]

    def _full_result(self, allowed_docs: Sequence[int]) -> ShortlistResult:
        """The whole entitled set in canonical order, ``source="all"`` (§5.3, §5.5).

        ``skipped=True``: for the prompt layer this is byte-identical to the
        small-catalog bypass — full registry, canonical order, never shuffled,
        stable cached prefix (§5.6). Pinned routes need no special handling
        here because everything entitled is already present.
        """
        return ShortlistResult(
            candidates=[
                Candidate(route_name=self._names[doc], score=0.0, rank=rank, source="all")
                for rank, doc in enumerate(allowed_docs)
            ],
            skipped=True,
            weak_retrieval=False,
            index_key=self.index_key,
        )

    def _assemble(
        self,
        scored: list[tuple[int, float]],
        *,
        allowed_docs: Sequence[int],
        k: int,
        source: CandidateSource,
    ) -> ShortlistResult:
        """Turn ranked ``(doc, score)`` pairs into a :class:`ShortlistResult`.

        Implements degenerate retrieval and the pinned-route rule (both §5.5).
        ``scored`` must already be restricted to ``allowed_docs`` and sorted by
        descending score with a deterministic tie-break.
        """
        if k < 1:
            raise ConfigError(f"shortlist k must be >= 1, got {k}")

        positives = sum(1 for _, score in scored if score > 0.0)
        weak = positives < MIN_POSITIVE_CANDIDATES
        if weak and len(allowed_docs) <= self._min_routes:
            # (a) Small entitled set and nothing matched: showing the LLM
            # everything strictly dominates showing it noise.
            return self._full_result(allowed_docs)

        top = list(scored[:k])
        if weak:
            # (b) Pass the low-score top-K and let the policy stage bias toward
            # clarify. Pad from canonical order so the decision stage always has
            # a non-empty candidate block to reason (or abstain) over — a query
            # that matched nothing must not silently become "no eligible routes".
            chosen = {doc for doc, _ in top}
            for doc in allowed_docs:
                if len(top) >= k:
                    break
                if doc not in chosen:
                    top.append((doc, 0.0))
                    chosen.add(doc)

        candidates = [
            Candidate(route_name=self._names[doc], score=float(score), rank=rank, source=source)
            for rank, (doc, score) in enumerate(top)
        ]

        # Pinned routes are appended regardless of score — the fallback path
        # (e.g. human_handoff) must never be retrieved out of existence (§5.5).
        # They are still entitlement-checked: pinned does not mean visible.
        if self._pinned:
            present = {doc for doc, _ in top}
            allowed_set = set(allowed_docs)
            for doc in self._pinned:
                if doc in present or doc not in allowed_set:
                    continue
                candidates.append(
                    Candidate(
                        route_name=self._names[doc],
                        score=0.0,
                        rank=len(candidates),
                        source="pinned",
                    )
                )

        return ShortlistResult(
            candidates=candidates,
            skipped=False,
            weak_retrieval=weak,
            index_key=self.index_key,
        )


# --------------------------------------------------------------------------- #
# BM25 (plan §5.2) — the default backend.
# --------------------------------------------------------------------------- #


def _okapi_idf(n_docs: int, df: int) -> float:
    """``ln(1 + (N - df + 0.5) / (df + 0.5))`` — smoothed and non-negative (§5.2)."""
    if n_docs <= 0:
        return 0.0
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Shortlister(_BaseShortlister):
    """Fielded BM25F over the route catalog — pure Python (plan §5.2).

    Scoring, for query terms ``t`` and fields ``f`` (name x3.0, examples x2.0,
    description x1.0, tags x1.0)::

        score(q, d) = SUM_t idf(t) * SUM_f w_f *  tf(t,f,d) * (k1 + 1)
                                                 ---------------------------------------
                                                 tf + k1 * (1 - b + b*len_f/avg_f)

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    Three details that are decisions, not accidents:

    * **IDF is computed over the full registry**, never over the entitled
      subset (§5.2). Scores are therefore tenant-stable and one index is
      shareable across every tenant — which is exactly why the entitlement
      filter has to happen at *retrieval* time instead (§5.9).
    * **Per-field lengths and per-field averages**, so a three-word ``name``
      field is not penalised against a sixty-word ``description``.
    * The ``+1`` inside the idf log (the Lucene form of the ``+0.5``-smoothed
      Okapi idf) keeps idf non-negative, so "scored above zero" remains a
      meaningful test for degenerate retrieval (§5.5).

    Query-term frequency is ignored (the standard Okapi simplification for
    short queries): repeating a word in the query does not double its weight.
    """

    __slots__ = (
        "_avg_field_len",
        "_b",
        "_df",
        "_field_len",
        "_idf",
        "_k1",
        "_postings",
        "_weights",
    )

    def __init__(
        self,
        *,
        field_weights: Mapping[str, float] | None = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
        min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
        store: IndexStore | None = None,
        scope: str = DEFAULT_SCOPE,
        top_k: int | None = None,
    ) -> None:
        super().__init__(min_routes=min_routes, store=store, scope=scope, top_k=top_k)
        weights = dict(DEFAULT_FIELD_WEIGHTS)
        if field_weights:
            unknown = sorted(set(field_weights) - set(FIELD_NAMES))
            if unknown:
                raise ConfigError(
                    f"unknown BM25 field weight(s) {unknown}; fields are {list(FIELD_NAMES)}"
                )
            weights.update(field_weights)
        if any(value < 0 for value in weights.values()):
            raise ConfigError(f"BM25 field weights must be >= 0, got {weights}")
        if k1 < 0:
            raise ConfigError(f"BM25 k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ConfigError(f"BM25 b must be in [0, 1], got {b}")
        self._weights: tuple[float, ...] = tuple(float(weights[name]) for name in FIELD_NAMES)
        self._k1 = float(k1)
        self._b = float(b)
        self._postings: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._field_len: tuple[tuple[int, ...], ...] = ()
        self._avg_field_len: tuple[float, ...] = ()

    def __repr__(self) -> str:
        return (
            f"BM25Shortlister(routes={len(self._names)}, terms={len(self._postings)}, "
            f"registry_version={self._registry_version!r})"
        )

    @property
    def fingerprint(self) -> str:
        """Covers variant, field weights, ``k1`` / ``b`` and the tokenizer version."""
        return sha256_hex(
            "bm25",
            {
                "fields": list(FIELD_NAMES),
                "weights": list(self._weights),
                "k1": self._k1,
                "b": self._b,
                "tokenizer": TOKENIZER_VERSION,
            },
        )[:16]

    # -- build -------------------------------------------------------------- #

    def build(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Build (or reload) the postings for ``routes`` (plan §5.4).

        Idempotent: a second call with the same ``registry_version`` returns
        immediately, which is what makes "lazy build on the first request after
        a version change" cheap enough to attempt on *every* request.
        """
        if self._registry_version == registry_version:
            return
        with self._lock:
            if self._registry_version == registry_version:
                return
            materialised = list(routes)
            key = self._index_key_for(registry_version)
            if self._store is not None:
                blob = self._store.load(key)
                if blob is not None and self._load_blob(blob, registry_version):
                    return
            self._build_locked(materialised, registry_version)
            if self._store is not None:
                self._store.save(key, self._dump_blob())

    async def abuild(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Async twin of :meth:`build`, single-flighted (§5.4).

        Indexing is pure CPU work in the sub-millisecond range for realistic
        catalogs, so it runs inline rather than on a thread; the lock is what
        matters, since it stops N concurrent requests from each rebuilding
        after one version bump.
        """
        if self._registry_version == registry_version:
            return
        async with self._async_lock():
            self.build(routes, registry_version=registry_version)

    def _build_locked(self, routes: Sequence[Route], registry_version: str) -> None:
        n_fields = len(FIELD_NAMES)
        postings: dict[str, dict[int, list[int]]] = {}
        field_len: list[tuple[int, ...]] = []
        totals = [0] * n_fields

        for doc, route in enumerate(routes):
            lengths: list[int] = []
            for field_index, text in enumerate(_field_texts(route)):
                tokens = tokenize(text)
                lengths.append(len(tokens))
                totals[field_index] += len(tokens)
                for term, count in Counter(tokens).items():
                    per_doc = postings.setdefault(term, {})
                    tfs = per_doc.get(doc)
                    if tfs is None:
                        tfs = [0] * n_fields
                        per_doc[doc] = tfs
                    tfs[field_index] = count
            field_len.append(tuple(lengths))

        n_docs = len(routes)
        self._postings = {
            term: [(doc, tuple(tfs)) for doc, tfs in sorted(per_doc.items())]
            for term, per_doc in sorted(postings.items())
        }
        # df over the FULL registry corpus — never the entitled subset (§5.2).
        self._df = {term: len(entries) for term, entries in self._postings.items()}
        self._idf = {term: _okapi_idf(n_docs, df) for term, df in self._df.items()}
        self._field_len = tuple(field_len)
        self._avg_field_len = tuple((total / n_docs) if n_docs else 0.0 for total in totals)
        self._record_routes(routes, registry_version)

    # -- persistence (JSON; never pickle — §5.4) ---------------------------- #

    def _dump_blob(self) -> bytes:
        payload: dict[str, Any] = {
            "v": _BLOB_VERSION,
            "variant": "bm25",
            "fingerprint": self.fingerprint,
            "registry_version": self._registry_version,
            "names": list(self._names),
            "pinned": list(self._pinned),
            "field_len": [list(lengths) for lengths in self._field_len],
            "avg_field_len": list(self._avg_field_len),
            "postings": {
                term: [[doc, list(tfs)] for doc, tfs in entries]
                for term, entries in self._postings.items()
            },
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _load_blob(self, blob: bytes, registry_version: str) -> bool:
        """Restore from ``blob``.

        Returns False — leaving state untouched — on any mismatch or
        corruption, so the caller simply rebuilds. A stale or truncated cache
        entry must degrade to a rebuild, never to a wrong index.
        """
        try:
            payload = json.loads(blob.decode("utf-8"))
            if (
                payload.get("v") != _BLOB_VERSION
                or payload.get("variant") != "bm25"
                or payload.get("fingerprint") != self.fingerprint
                or payload.get("registry_version") != registry_version
            ):
                return False
            names = tuple(str(name) for name in payload["names"])
            postings = {
                str(term): [(int(doc), tuple(int(tf) for tf in tfs)) for doc, tfs in entries]
                for term, entries in payload["postings"].items()
            }
            field_len = tuple(tuple(int(x) for x in row) for row in payload["field_len"])
            avg_field_len = tuple(float(x) for x in payload["avg_field_len"])
            pinned = tuple(int(doc) for doc in payload["pinned"])
        except (AttributeError, KeyError, TypeError, ValueError, UnicodeDecodeError):
            return False
        if len(field_len) != len(names) or len(avg_field_len) != len(FIELD_NAMES):
            return False
        self._names = names
        self._pinned = pinned
        self._postings = postings
        self._df = {term: len(entries) for term, entries in postings.items()}
        self._idf = {term: _okapi_idf(len(names), df) for term, df in self._df.items()}
        self._field_len = field_len
        self._avg_field_len = avg_field_len
        self._registry_version = registry_version
        return True

    # -- query -------------------------------------------------------------- #

    def shortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Score ``query`` against the entitled routes and return the top-K (§5.2)."""
        del ctx  # [v0.2] per-tenant index scope; inert in v0.1.
        self._require_built()
        allowed_docs = self._allowed_docs(allowed)
        scored = self._score(tokenize(query), allowed_docs)
        return self._assemble(scored, allowed_docs=allowed_docs, k=k, source="bm25")

    async def ashortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Async twin of :meth:`shortlist`.

        BM25 performs no I/O, so this delegates directly rather than paying for
        a thread hop — parity of *contract*, not of mechanism (§2.5).
        """
        return self.shortlist(query, allowed=allowed, k=k, ctx=ctx)

    def _score(
        self, query_tokens: Sequence[str], allowed_docs: Sequence[int]
    ) -> list[tuple[int, float]]:
        if not query_tokens or not allowed_docs:
            return []
        # A bytearray mask rather than a set: postings for a common term run to
        # thousands of docs, and this membership test is the innermost thing
        # BM25 does. Indexing a bytearray beats hashing an int.
        mask = bytearray(len(self._names))
        for doc in allowed_docs:
            mask[doc] = 1
        k1 = self._k1
        b = self._b
        weights = self._weights
        # An average of 0 only occurs for a field that is empty across the whole
        # corpus, where tf is 0 anyway; 1.0 keeps the normaliser finite.
        averages = tuple(avg if avg > 0.0 else 1.0 for avg in self._avg_field_len)
        scores: dict[int, float] = {}

        for term in dict.fromkeys(query_tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc, tfs in self._postings[term]:
                if not mask[doc]:
                    continue
                lengths = self._field_len[doc]
                total = 0.0
                for field_index, tf in enumerate(tfs):
                    if not tf:
                        continue
                    norm = 1.0 - b + b * (lengths[field_index] / averages[field_index])
                    total += weights[field_index] * (tf * (k1 + 1.0)) / (tf + k1 * norm)
                if total:
                    scores[doc] = scores.get(doc, 0.0) + idf * total

        # Ties break on canonical registry position: deterministic across
        # processes, which replay-cached evals depend on.
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


# --------------------------------------------------------------------------- #
# Embeddings (plan §5.2) — BYO callable, zero extras.
# --------------------------------------------------------------------------- #


def _is_async_callable(fn: object) -> bool:
    """True for coroutine functions *and* objects with an async ``__call__``.

    The MRO walk is what catches the second case: a class instance whose
    ``__call__`` is ``async def`` is a perfectly ordinary embedder (a client
    object with a connection pool, say), and treating it as sync would call it
    and silently index coroutine objects.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    for klass in type(fn).__mro__:
        call = klass.__dict__.get("__call__")
        if call is not None:
            return inspect.iscoroutinefunction(call)
    return False


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize, so cosine similarity is a plain dot product."""
    values = tuple(float(x) for x in vector)
    norm = math.sqrt(math.fsum(x * x for x in values))
    if norm == 0.0:
        return values
    return tuple(x / norm for x in values)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two **already normalized** vectors of equal length."""
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _as_matrix(result: Any, expected: int) -> list[list[float]]:
    """Coerce an embed callable's return value into a list of float rows.

    Accepts anything iterable-of-iterable-of-number — a numpy array included —
    without importing numpy: the values are pulled through ``float()``.
    """
    try:
        rows = [[float(x) for x in row] for row in result]
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"embed callable must return a sequence of float vectors, got "
            f"{type(result).__name__}"
        ) from exc
    if len(rows) != expected:
        raise ConfigError(
            f"embed callable returned {len(rows)} vectors for {expected} input texts"
        )
    return rows


def _load_backend(name: str, model: str | None) -> EmbedFn:
    """Resolve a packaged embedding backend, importing it lazily (§2.4).

    ``shortlisters.embedding_backends`` is the only module allowed to touch an
    embedding SDK, and the import happens *inside this function* so that merely
    importing ``switchboard.engine.shortlist`` never reaches for one.
    """
    from switchboard.shortlisters.embedding_backends import get_backend

    return get_backend(name, model=model)


def _callable_id(fn: object, model: str | None) -> str:
    """Best-effort stable identity for a BYO embed callable."""
    module = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    base = f"{module}.{qualname}"
    return f"{base}:{model}" if model else base


class EmbeddingShortlister(_BaseShortlister):
    """Dense retrieval over route embeddings (plan §5.2).

    Bring your own ``embed`` callable and this works with **zero extras**
    (§10.2); pass a backend name (``"fastembed"``, ``"openai"``) and the
    ``[embed]`` extra supplies one. Vectors are L2-normalized and compared by
    brute-force cosine in pure Python: 5k routes x 768 dims is ~4 MB and
    single-digit milliseconds, and an ANN index below that scale is pure
    dependency risk (§5.2).

    Recommended for multilingual or paraphrase-heavy traffic, where a lexical
    tokenizer with no stemmer runs out of road.

    **Embedding cache (§7.3).** Route vectors are cached by
    ``sha256(Route.embed_text)`` where ``embed_text = name + description +
    examples``. An ``args_model``-only edit bumps ``registry_version`` — which
    correctly invalidates the prompt cache, since the prompt contains the schema
    — without re-embedding anything. The cache survives rebuilds, so adding one
    route to a 5000-route catalog costs exactly one embedding call.

    Args:
        embed: a callable taking ``list[str]`` and returning one vector per
            input, **or** the name of a packaged backend. Async callables are
            supported and require the ``a``-prefixed methods.
        model: model id passed to a packaged backend; also recorded in the
            fingerprint.
        embed_id: stable identity for a BYO callable. Defaults to
            ``module.qualname``, which is fine within one deployment; set it
            explicitly if you persist indexes across differently-structured
            processes.
    """

    __slots__ = ("_dim", "_embed", "_embed_id", "_hashes", "_is_async", "_vec_cache", "_vectors")

    def __init__(
        self,
        embed: EmbedFn | AsyncEmbedFn | str | None = None,
        *,
        model: str | None = None,
        embed_id: str | None = None,
        min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
        store: IndexStore | None = None,
        scope: str = DEFAULT_SCOPE,
        top_k: int | None = None,
    ) -> None:
        super().__init__(min_routes=min_routes, store=store, scope=scope, top_k=top_k)
        if isinstance(embed, str):
            backend = embed.strip().lower()
            # Resolved eagerly so a missing extra raises MissingDependencyError at
            # Router construction, not on the first request (§2.4).
            self._embed: Callable[[list[str]], Any] = _load_backend(backend, model)
            resolved_id = embed_id or f"{backend}:{model or 'default'}"
        elif callable(embed):
            self._embed = cast("Callable[[list[str]], Any]", embed)
            resolved_id = embed_id or _callable_id(embed, model)
        else:
            raise ConfigError(
                "EmbeddingShortlister needs an embed callable "
                "(Callable[[list[str]], list[list[float]]]) or a backend name such as "
                "'fastembed' or 'openai'"
            )
        self._embed_id = resolved_id
        self._is_async = _is_async_callable(self._embed)
        self._vec_cache: dict[str, tuple[float, ...]] = {}
        self._vectors: tuple[tuple[float, ...], ...] = ()
        self._hashes: tuple[str, ...] = ()
        self._dim: int | None = None

    def __repr__(self) -> str:
        return (
            f"EmbeddingShortlister(embed={self._embed_id!r}, routes={len(self._names)}, "
            f"dim={self._dim}, registry_version={self._registry_version!r})"
        )

    @property
    def fingerprint(self) -> str:
        """Covers the variant and the embedding identity (backend + model).

        Swapping embedding models must invalidate the index — cosine scores from
        two different models are not comparable, and a stale matrix would
        silently rank garbage.
        """
        return sha256_hex("embed", {"embed_id": self._embed_id})[:16]

    @property
    def embed_id(self) -> str:
        """The embedding identity string that feeds :attr:`fingerprint`."""
        return self._embed_id

    @property
    def dim(self) -> int | None:
        """Vector dimensionality, once built."""
        return self._dim

    @property
    def cached_vectors(self) -> int:
        """How many route vectors are memoised by ``sha256(embed_text)`` (§7.3)."""
        return len(self._vec_cache)

    # -- build -------------------------------------------------------------- #

    def build(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Embed any routes whose ``embed_text`` is not already cached (§7.3)."""
        if self._registry_version == registry_version:
            return
        with self._lock:
            if self._registry_version == registry_version:
                return
            materialised = list(routes)
            key = self._index_key_for(registry_version)
            if self._store is not None:
                blob = self._store.load(key)
                if blob is not None and self._load_blob(blob, registry_version):
                    return
            hashes, todo_hashes, todo_texts = self._plan_build(materialised)
            if todo_texts:
                if self._is_async:
                    raise ConfigError(
                        "EmbeddingShortlister was given an async embed callable; use "
                        "abuild() / ashortlist() (or Router.aroute()) instead of the sync API"
                    )
                self._absorb(todo_hashes, _as_matrix(self._embed(todo_texts), len(todo_texts)))
            self._finish_build(materialised, hashes, registry_version)
            if self._store is not None:
                self._store.save(key, self._dump_blob())

    async def abuild(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Async twin of :meth:`build`, single-flighted against stampedes (§5.4)."""
        if self._registry_version == registry_version:
            return
        async with self._async_lock():
            if self._registry_version == registry_version:
                return
            materialised = list(routes)
            key = self._index_key_for(registry_version)
            if self._store is not None:
                blob = self._store.load(key)
                if blob is not None and self._load_blob(blob, registry_version):
                    return
            hashes, todo_hashes, todo_texts = self._plan_build(materialised)
            if todo_texts:
                self._absorb(todo_hashes, await self._aembed(todo_texts))
            self._finish_build(materialised, hashes, registry_version)
            if self._store is not None:
                self._store.save(key, self._dump_blob())

    def _plan_build(self, routes: Sequence[Route]) -> tuple[list[str], list[str], list[str]]:
        """Return ``(per-route hashes, hashes to embed, texts to embed)``.

        De-duplicated by content hash, so two routes with identical embed text
        cost one embedding call, and an unchanged route costs none.
        """
        hashes = [sha256_hex(route.embed_text) for route in routes]
        todo: dict[str, str] = {}
        for route, digest in zip(routes, hashes, strict=True):
            if digest not in self._vec_cache:
                todo.setdefault(digest, route.embed_text)
        return hashes, list(todo), list(todo.values())

    def _absorb(self, hashes: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        for digest, vector in zip(hashes, vectors, strict=True):
            self._vec_cache[digest] = _normalize(vector)

    def _finish_build(
        self, routes: Sequence[Route], hashes: Sequence[str], registry_version: str
    ) -> None:
        vectors = tuple(self._vec_cache[digest] for digest in hashes)
        dims = {len(vector) for vector in vectors}
        if len(dims) > 1:
            raise ConfigError(
                f"embed callable returned vectors of inconsistent dimensionality: {sorted(dims)}"
            )
        self._vectors = vectors
        self._hashes = tuple(hashes)
        self._dim = next(iter(dims)) if dims else None
        self._record_routes(routes, registry_version)

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        """Await an async embed callable, or run a blocking one off the loop."""
        if self._is_async:
            return _as_matrix(await self._embed(texts), len(texts))
        return _as_matrix(await asyncio.to_thread(self._embed, texts), len(texts))

    # -- persistence -------------------------------------------------------- #

    def _dump_blob(self) -> bytes:
        payload: dict[str, Any] = {
            "v": _BLOB_VERSION,
            "variant": "embed",
            "fingerprint": self.fingerprint,
            "registry_version": self._registry_version,
            "names": list(self._names),
            "pinned": list(self._pinned),
            "hashes": list(self._hashes),
            "vectors": [list(vector) for vector in self._vectors],
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _load_blob(self, blob: bytes, registry_version: str) -> bool:
        try:
            payload = json.loads(blob.decode("utf-8"))
            if (
                payload.get("v") != _BLOB_VERSION
                or payload.get("variant") != "embed"
                or payload.get("fingerprint") != self.fingerprint
                or payload.get("registry_version") != registry_version
            ):
                return False
            names = tuple(str(name) for name in payload["names"])
            pinned = tuple(int(doc) for doc in payload["pinned"])
            vectors = tuple(tuple(float(x) for x in row) for row in payload["vectors"])
            hashes = tuple(str(digest) for digest in payload["hashes"])
        except (AttributeError, KeyError, TypeError, ValueError, UnicodeDecodeError):
            return False
        if len(vectors) != len(names) or len(hashes) != len(names):
            return False
        if len({len(vector) for vector in vectors}) > 1:
            return False
        for digest, vector in zip(hashes, vectors, strict=True):
            self._vec_cache[digest] = vector
        self._names = names
        self._pinned = pinned
        self._vectors = vectors
        self._hashes = hashes
        self._dim = len(vectors[0]) if vectors else None
        self._registry_version = registry_version
        return True

    # -- query -------------------------------------------------------------- #

    def shortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Embed ``query`` and rank the entitled routes by cosine similarity."""
        del ctx  # [v0.2] per-tenant index scope; inert in v0.1.
        self._require_built()
        if self._is_async:
            raise ConfigError(
                "EmbeddingShortlister was given an async embed callable; use ashortlist() "
                "(or Router.aroute()) instead of the sync API"
            )
        query_vector = _normalize(_as_matrix(self._embed([query]), 1)[0])
        return self._rank(query_vector, allowed, k)

    async def ashortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Async twin of :meth:`shortlist`.

        A coroutine ``embed`` is awaited directly; a blocking one is pushed to
        ``asyncio.to_thread`` so a network-backed embedder cannot stall the loop.
        """
        del ctx  # [v0.2] per-tenant index scope; inert in v0.1.
        self._require_built()
        query_vector = _normalize((await self._aembed([query]))[0])
        return self._rank(query_vector, allowed, k)

    def _rank(
        self, query_vector: Sequence[float], allowed: AbstractSet[str], k: int
    ) -> ShortlistResult:
        if self._dim is not None and len(query_vector) != self._dim:
            raise ConfigError(
                f"embed callable returned a {len(query_vector)}-dim query vector but the "
                f"index holds {self._dim}-dim route vectors; the embedding model changed "
                f"without the shortlister fingerprint changing (pass embed_id=...)"
            )
        allowed_docs = self._allowed_docs(allowed)
        vectors = self._vectors
        scored = sorted(
            ((doc, _cosine(query_vector, vectors[doc])) for doc in allowed_docs),
            key=lambda item: (-item[1], item[0]),
        )
        return self._assemble(scored, allowed_docs=allowed_docs, k=k, source="embed")


# --------------------------------------------------------------------------- #
# "auto" (plan §5.1, §5.3).
# --------------------------------------------------------------------------- #


class AutoShortlister(_BaseShortlister):
    """The ``shortlist="auto"`` policy — the v0.1 default (plan §5.1, §13 #7).

    Resolves **per request** against the entitled set size:

    * ``N < shortlist_min_routes`` (25): bypass retrieval entirely. The result
      is the full entitled registry in canonical order with ``skipped=True`` and
      ``source="all"``, which the prompt layer renders as a stable cached prefix
      rather than shuffling (§5.6). Below this band retrieval adds miss-risk for
      no win.
    * otherwise: delegate to the configured backend (BM25 by default).

    The bypass check runs on the *entitled* count, not the registry count: a
    tenant entitled to 12 of 400 routes gets the bypass, which is both cheaper
    and more accurate for them.
    """

    __slots__ = ("_delegate",)

    def __init__(
        self,
        delegate: Shortlister | None = None,
        *,
        min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
        store: IndexStore | None = None,
        scope: str = DEFAULT_SCOPE,
        top_k: int | None = None,
    ) -> None:
        super().__init__(min_routes=min_routes, store=store, scope=scope, top_k=top_k)
        self._delegate: Shortlister = delegate or BM25Shortlister(
            min_routes=min_routes, store=store, scope=scope, top_k=top_k
        )

    def __repr__(self) -> str:
        return (
            f"AutoShortlister(min_routes={self._min_routes}, delegate={self._delegate!r}, "
            f"registry_version={self._registry_version!r})"
        )

    @property
    def delegate(self) -> Shortlister:
        """The backend used above the bypass threshold."""
        return self._delegate

    @property
    def fingerprint(self) -> str:
        return sha256_hex(
            "auto",
            {"min_routes": self._min_routes, "delegate": self._delegate.fingerprint},
        )[:16]

    def build(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Record canonical order for the bypass, then build the delegate.

        The version check precedes any materialisation: the Router calls this on
        every request, and copying a 5000-route sequence just to discover
        nothing changed is pure waste.
        """
        if self._registry_version != registry_version:
            with self._lock:
                if self._registry_version != registry_version:
                    self._record_routes(list(routes), registry_version)
        self._delegate.build(routes, registry_version=registry_version)

    async def abuild(self, routes: Sequence[Route], *, registry_version: str) -> None:
        """Async twin of :meth:`build`."""
        if self._registry_version != registry_version:
            async with self._async_lock():
                if self._registry_version != registry_version:
                    self._record_routes(list(routes), registry_version)
        await self._delegate.abuild(routes, registry_version=registry_version)

    def shortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Bypass below the threshold, otherwise delegate (§5.1)."""
        self._require_built()
        allowed_docs = self._allowed_docs(allowed)
        if len(allowed_docs) < self._min_routes:
            return self._full_result(allowed_docs)
        return self._delegate.shortlist(query, allowed=allowed, k=k, ctx=ctx)

    async def ashortlist(
        self,
        query: str,
        *,
        allowed: AbstractSet[str],
        k: int,
        ctx: RequestContext | None = None,
    ) -> ShortlistResult:
        """Async twin of :meth:`shortlist`."""
        self._require_built()
        allowed_docs = self._allowed_docs(allowed)
        if len(allowed_docs) < self._min_routes:
            return self._full_result(allowed_docs)
        return await self._delegate.ashortlist(query, allowed=allowed, k=k, ctx=ctx)


# --------------------------------------------------------------------------- #
# Config resolution (plan §3.3 DSL).
# --------------------------------------------------------------------------- #

_ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "auto": frozenset({"top_k", "min_routes"}),
    "bm25": frozenset({"top_k", "min_routes", "k1", "b"}),
    "embed": frozenset({"top_k", "min_routes", "model", "backend"}),
    "hybrid": frozenset({"top_k", "min_routes"}),
}


def _parse_params(tail: str, variant: str) -> dict[str, str]:
    """Parse the ``key=value,key=value`` half of a shortlist spec (§3.3)."""
    params: dict[str, str] = {}
    for item in tail.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        key, sep, value = chunk.partition("=")
        if not sep:
            raise ConfigError(
                f"malformed shortlist spec parameter {chunk!r}; expected key=value "
                f"(e.g. 'bm25:top_k=10')"
            )
        key = key.strip()
        if key not in _ALLOWED_PARAMS[variant]:
            raise ConfigError(
                f"unknown parameter {key!r} for shortlist variant {variant!r}; "
                f"supported: {sorted(_ALLOWED_PARAMS[variant])}"
            )
        params[key] = value.strip()
    return params


def _int_param(params: Mapping[str, str], key: str) -> int | None:
    if key not in params:
        return None
    try:
        return int(params[key])
    except ValueError as exc:
        raise ConfigError(
            f"shortlist parameter {key}={params[key]!r} must be an integer"
        ) from exc


def _float_param(params: Mapping[str, str], key: str, default: float) -> float:
    if key not in params:
        return default
    try:
        return float(params[key])
    except ValueError as exc:
        raise ConfigError(f"shortlist parameter {key}={params[key]!r} must be a number") from exc


def _default_backend_for(model: str | None) -> str:
    """Pick a packaged backend when only a model name was given.

    ``text-embedding-*`` is OpenAI's naming, so it routes there; everything else
    defaults to the local ``fastembed`` backend (no network call, no API key).
    Pass ``backend=`` explicitly to override.
    """
    if model and model.startswith("text-embedding-"):
        return "openai"
    return "fastembed"


def resolve_shortlister(
    spec: str | Shortlister | None = "auto",
    *,
    embed: EmbedFn | AsyncEmbedFn | None = None,
    min_routes: int = DEFAULT_SHORTLIST_MIN_ROUTES,
    store: IndexStore | None = None,
    scope: str = DEFAULT_SCOPE,
) -> Shortlister | None:
    """Resolve the ``Router(shortlist=...)`` value into a backend (plan §3.3).

    Accepts:

    * ``"auto"`` — the default: bypass below ``min_routes``, else BM25 (§5.1).
    * ``"bm25"`` / ``"bm25:top_k=10"`` / ``"bm25:k1=1.2,b=0.75"``.
    * ``"embed"`` / ``"embed:model=..."`` / ``"embed:backend=fastembed"`` — uses
      ``embed`` if a BYO callable was supplied, otherwise resolves a packaged
      backend eagerly so a missing extra fails at construction (§2.4).
    * ``"hybrid"`` — [v0.2]; raises :class:`~switchboard.errors.ConfigError`.
    * any object satisfying :class:`Shortlister`, returned unchanged.
    * ``None`` — shortlisting disabled. The Router then raises ``ConfigError`` if
      the entitled set exceeds ``max_candidates``, since silently truncating
      would drop routes (§3.3).

    Raises:
        ConfigError: unknown variant, malformed parameters, or ``"hybrid"``.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        if isinstance(spec, Shortlister):
            return spec
        raise ConfigError(
            f"shortlist must be a spec string, a Shortlister, or None; got {type(spec).__name__}"
        )

    raw = spec.strip()
    if not raw:
        raise ConfigError("shortlist spec string is empty; use 'auto', 'bm25', 'embed' or None")
    head, _, tail = raw.partition(":")
    variant = head.strip().lower()
    if variant not in _ALLOWED_PARAMS:
        raise ConfigError(
            f"unknown shortlist variant {variant!r}; supported: "
            f"{sorted(_ALLOWED_PARAMS)} (hybrid is [v0.2])"
        )
    if variant == "hybrid":
        raise ConfigError(
            "shortlist='hybrid' (RRF fusion over BM25 + embeddings) is [v0.2] and not "
            "implemented in v0.1; use 'auto', 'bm25' or 'embed'"
        )

    params = _parse_params(tail, variant)
    top_k = _int_param(params, "top_k")
    parsed_min = _int_param(params, "min_routes")
    resolved_min = min_routes if parsed_min is None else parsed_min

    if variant == "bm25":
        return BM25Shortlister(
            k1=_float_param(params, "k1", BM25_K1),
            b=_float_param(params, "b", BM25_B),
            min_routes=resolved_min,
            store=store,
            scope=scope,
            top_k=top_k,
        )
    if variant == "embed":
        model = params.get("model")
        source: EmbedFn | AsyncEmbedFn | str = (
            embed if embed is not None else params.get("backend") or _default_backend_for(model)
        )
        return EmbeddingShortlister(
            source,
            model=model,
            min_routes=resolved_min,
            store=store,
            scope=scope,
            top_k=top_k,
        )
    return AutoShortlister(min_routes=resolved_min, store=store, scope=scope, top_k=top_k)
