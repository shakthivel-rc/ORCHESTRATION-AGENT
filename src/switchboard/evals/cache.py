"""Record/replay cache for provider calls (plan §9.6).

*"Every LLM call content-addressed by ``(model, prompt_hash, schema_hash,
temperature, seed, sample_index)``, stored as JSONL. ``--record`` populates;
``--replay`` is strict (cache miss fails the run) — CI is deterministic, keyless,
free."*

That last clause is the requirement, and strictness is what delivers it. A cache
that silently fell through to the network on a miss would turn a green PR into a
bill and a flake, and would let a prompt change slip in unnoticed — the miss *is*
the signal that the prompt, the schema or the model moved. So
:class:`ReplayClient` in ``mode="replay"`` raises on a miss and never holds a
credential.

The key is derived from the :class:`~switchboard.providers.base.LLMRequest` the
loop actually built, so it covers everything that can change an answer: the
rendered segments (catalog, shortlist, candidate order, query), the wire schema
(which encodes the dynamic route enum, §4.4), and the sampling knobs. Nothing is
hashed that a provider would not see.

**Packaging.** Pydantic + stdlib only — the replay cache is core, not part of the
[v0.2] ``[eval]`` extra (§13 ruling #17), because a keyless deterministic test
lane must work in a bare install.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from switchboard.core.audit import canonical_json, sha256_hex
from switchboard.errors import ConfigError
from switchboard.providers.base import (
    ClientCapabilities,
    LLMResult,
    TokenLP,
    Usage,
    capabilities_of,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from switchboard.providers.base import LLMRequest

__all__ = [
    "CACHE_SCHEMA",
    "DEFAULT_CACHE_FILENAME",
    "CacheEntry",
    "CacheKey",
    "ReplayCache",
    "ReplayClient",
    "ReplayMode",
    "sample_scope",
]

CACHE_SCHEMA = "sb-replay/1"
"""Header tag of a cache file. Bumped when the entry layout changes."""

DEFAULT_CACHE_FILENAME = "llm-calls.jsonl"
"""File written inside a cache *directory* (plan §9.6: "stored as JSONL")."""

ReplayMode = Literal["record", "replay", "auto"]
"""``record`` populates (always calls the provider), ``replay`` is strict (a miss
raises, no provider needed), ``auto`` replays hits and records misses."""


# --------------------------------------------------------------------------- #
# Sample scoping (the `sample_index` component of the plan §9.6 key).
# --------------------------------------------------------------------------- #

_SAMPLE_COUNTS: ContextVar[dict[str, int] | None] = ContextVar(
    "switchboard_replay_sample_counts", default=None
)


@contextlib.contextmanager
def sample_scope() -> Iterator[None]:
    """Scope ``sample_index`` to one decision (plan §9.6).

    Inside the scope, the *n*-th genuinely identical request — same model, prompt,
    schema, temperature and seed — gets ``sample_index = n``. That is what makes
    [v0.2] self-consistency votes cacheable as distinct samples instead of
    collapsing onto one entry.

    The counter lives in a :class:`~contextvars.ContextVar`, so it is per-task
    under ``asyncio`` and per-thread otherwise: a concurrency-limited
    :func:`~switchboard.evals.harness.arun_suite` gives every case its own
    counter, and cache keys therefore do not depend on how the scheduler
    interleaved the run. Without that, replay would be reproducible only at
    concurrency 1.

    Outside any scope ``sample_index`` is always 0, so identical calls collapse
    onto one entry and the cache behaves like a cache. That is the right default:
    a client cannot see decision boundaries on its own, and a counter that ran
    across a whole process would make the second identical call in a REPL miss.
    Multiple samples of one call only arise under [v0.2] self-consistency voting,
    which is exactly where a scope is available to declare them.
    """
    token = _SAMPLE_COUNTS.set({})
    try:
        yield
    finally:
        _SAMPLE_COUNTS.reset(token)


# --------------------------------------------------------------------------- #
# Key and entry.
# --------------------------------------------------------------------------- #


class CacheKey(BaseModel):
    """The content address of one provider call (plan §9.6, normative tuple)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str
    """The request model id, e.g. ``"openai/gpt-5-nano"``. Answers are not
    portable across models, so the model is part of the address, not metadata."""
    prompt_hash: str
    """sha256 over the ordered ``(role, content)`` of every prompt segment."""
    schema_hash: str
    """sha256 over the wire schema's JSON Schema. Under ``wire_schema="dynamic"``
    the candidate enum lives in there (§4.4), so a changed shortlist changes this
    even when the rendered prompt did not."""
    temperature: float = 0.0
    seed: int | None = None
    sample_index: int = 0
    """Which sample of an otherwise identical call this is — see :func:`sample_scope`."""

    @property
    def digest(self) -> str:
        """The 64-hex address this entry is stored under."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def describe(self) -> str:
        """One-line human form, for a strict-miss error message."""
        return (
            f"model={self.model!r} prompt={self.prompt_hash[:12]} "
            f"schema={self.schema_hash[:12]} temperature={self.temperature} "
            f"seed={self.seed} sample={self.sample_index}"
        )


def prompt_hash(request: LLMRequest) -> str:
    """Hash the rendered prompt of ``request`` (plan §9.6).

    Hashes role + content and nothing else: ``cache_key`` and ``cache`` are
    derived from that same content (§4.6), so including them would only add a
    second way for the address to change without the model's input changing.
    """
    return sha256_hex(
        canonical_json(
            [{"role": segment.role, "content": segment.content} for segment in request.segments]
        )
    )


def schema_hash(request: LLMRequest) -> str:
    """Hash the wire schema of ``request`` (plan §9.6)."""
    try:
        schema: Any = request.output_schema.model_json_schema()
    except Exception:  # pragma: no cover - exotic/unschematisable output model
        schema = f"{request.output_schema.__module__}.{request.output_schema.__qualname__}"
    return sha256_hex(canonical_json(schema))


class CacheEntry(BaseModel):
    """One recorded provider response (plan §9.6).

    Stores what an adapter returns *on the wire*, never ``LLMResult.parsed``: the
    parsed object is an instance of a per-call generated wire-schema model, which
    is not reconstructible from JSON. Replay therefore hands back
    ``parsed=None`` and the real ``engine/validate.py`` parse path runs again —
    which is the honest replay, because parsing is part of what the harness is
    measuring (§9.2's schema-validity rate).
    """

    model_config = ConfigDict(protected_namespaces=())

    key: CacheKey
    raw_text: str
    model_id: str
    usage: Usage = Field(default_factory=Usage)
    token_logprobs: list[TokenLP] | None = None
    attempts: int = 1
    provider_meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, key: CacheKey, result: LLMResult[Any]) -> CacheEntry:
        """Capture ``result`` under ``key``."""
        return cls(
            key=key,
            raw_text=result.raw_text,
            model_id=result.model_id,
            usage=result.usage,
            token_logprobs=result.token_logprobs,
            attempts=result.attempts,
            provider_meta=dict(result.provider_meta),
        )

    def to_result(self) -> LLMResult[Any]:
        """Rebuild the :class:`~switchboard.providers.base.LLMResult`."""
        return LLMResult(
            parsed=None,
            raw_text=self.raw_text,
            token_logprobs=self.token_logprobs,
            usage=self.usage,
            model_id=self.model_id,
            attempts=self.attempts,
            provider_meta=dict(self.provider_meta),
        )


# --------------------------------------------------------------------------- #
# The store.
# --------------------------------------------------------------------------- #


class ReplayCache:
    """A JSONL, content-addressed store of provider responses (plan §9.6).

    Append-only on disk, dict in memory, last-write-wins on load — so re-recording
    a call appends a fresh line and the stale one is simply shadowed. That makes
    the file replayable *and* diffable in a PR, which is the point of the weekly
    re-record lane (§9.6).

    Thread-safe: the in-memory index and the append handle are guarded by one
    lock, because a concurrency-limited async suite records from several tasks.

    Args:
        path: a ``.jsonl`` file, or a directory (then
            ``<dir>/llm-calls.jsonl``). Parents are created on first write.
        autoflush: fsync-free flush after every write, so a crashed run keeps
            what it had already recorded. Turn it off for bulk re-recording.
    """

    def __init__(self, path: str | Path, *, autoflush: bool = True) -> None:
        self._path = _resolve_cache_path(path)
        self._autoflush = autoflush
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._handle: Any = None
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.load()

    # -- introspection ------------------------------------------------------- #

    @property
    def path(self) -> Path:
        """The JSONL file this cache reads and appends to."""
        return self._path

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        digest = key.digest if isinstance(key, CacheKey) else key
        return isinstance(digest, str) and digest in self._entries

    def __repr__(self) -> str:
        return (
            f"ReplayCache(path={str(self._path)!r}, entries={len(self._entries)}, "
            f"hits={self.hits}, misses={self.misses}, writes={self.writes})"
        )

    def stats(self) -> dict[str, int]:
        """Hit/miss/write counters for the report line (plan §9.6)."""
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    # -- persistence --------------------------------------------------------- #

    def load(self) -> ReplayCache:
        """(Re)read the file into memory. Missing file = empty cache."""
        with self._lock:
            self._entries.clear()
            if not self._path.exists():
                return self
            with self._path.open("r", encoding="utf-8") as handle:
                for number, raw in enumerate(handle, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ConfigError(
                            f"{self._path}:{number}: replay cache is not valid JSONL "
                            f"({exc.msg}). Delete the file and re-record (plan §9.6)."
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ConfigError(
                            f"{self._path}:{number}: expected a JSON object"
                        )
                    if "cache" in payload and "key" not in payload:
                        _check_cache_schema(payload["cache"], self._path, number)
                        continue
                    try:
                        entry = CacheEntry.model_validate(payload)
                    except Exception as exc:
                        raise ConfigError(
                            f"{self._path}:{number}: invalid cache entry — {exc}"
                        ) from exc
                    self._entries[entry.key.digest] = entry
        return self

    def _open(self) -> Any:
        """Open the append handle, writing the schema header on a fresh file."""
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self._path.exists() or self._path.stat().st_size == 0
            self._handle = self._path.open("a", encoding="utf-8")
            if fresh:
                self._handle.write(_dump_line({"cache": CACHE_SCHEMA}))
        return self._handle

    def flush(self) -> None:
        """Flush buffered appends. Never raises on an already-closed cache."""
        with self._lock:
            if self._handle is not None:
                self._handle.flush()

    def close(self) -> None:
        """Flush and close the append handle. Idempotent."""
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                self._handle.close()
                self._handle = None

    def __enter__(self) -> ReplayCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the cache itself ---------------------------------------------------- #

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Look ``key`` up, counting the hit or the miss."""
        with self._lock:
            entry = self._entries.get(key.digest)
            if entry is None:
                self.misses += 1
            else:
                self.hits += 1
            return entry

    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        """Record ``entry`` under ``key``, in memory and on disk."""
        stored = entry if entry.key == key else entry.model_copy(update={"key": key})
        with self._lock:
            self._entries[key.digest] = stored
            handle = self._open()
            handle.write(_dump_line(stored.model_dump(mode="json")))
            if self._autoflush:
                handle.flush()
            self.writes += 1


def _resolve_cache_path(path: str | Path) -> Path:
    """A ``.jsonl`` path stays a file; anything else is treated as a cache dir."""
    resolved = Path(path)
    if resolved.suffix == ".jsonl":
        return resolved
    return resolved / DEFAULT_CACHE_FILENAME


def _check_cache_schema(value: Any, path: Path, number: int) -> None:
    """Refuse a cache file written by an incompatible build."""
    if value == CACHE_SCHEMA:
        return
    raise ConfigError(
        f"{path}:{number}: replay cache schema {value!r} is not supported by this "
        f"switchboard build, which reads {CACHE_SCHEMA!r}. Re-record the cache "
        f"(plan §9.6)."
    )


def _dump_line(payload: Any) -> str:
    """One compact, key-sorted JSONL line."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


# --------------------------------------------------------------------------- #
# The client wrapper.
# --------------------------------------------------------------------------- #


class ReplayClient:
    """Wrap any :class:`~switchboard.providers.base.LLMClient` in the cache (§9.6).

    Satisfies both provider Protocols — ``complete`` and ``acomplete`` — so one
    instance serves ``route()`` and ``aroute()``, and forwards ``model`` /
    ``provider`` so audit records and spans still name the real model rather than
    the wrapper (§8.2).

    Example::

        cache = ReplayCache("tests/fixtures/replay")
        client = ReplayClient(real_client, cache=cache, mode="auto")   # dev
        client = ReplayClient(cache=cache, mode="replay",              # CI
                              capabilities=GRAMMAR_LOGPROBS, model="openai/gpt-5-nano")
        router = Router(registry, client=client, shortlist="auto")

    Args:
        client: the client to record from. May be ``None`` in ``mode="replay"``,
            which is the point of strict replay: CI holds no credential.
        cache: the :class:`ReplayCache`, or a path to one.
        mode: ``"record"``, ``"replay"`` or ``"auto"``.
        capabilities: what to report to the Router. Defaults to the wrapped
            client's. **Must be supplied when replaying without a client**, and
            must match what was recorded — capabilities decide the wire-schema
            mode and whether logprobs are requested (§4.3, §4.4), so a mismatch
            changes the schema hash and every lookup misses.
        model: the model id used in the cache key. Defaults to the wrapped
            client's; required when replaying without one.

    Raises:
        ConfigError: a strict-replay miss, or a configuration that cannot serve a
            call (replay without a model, record without a client).
    """

    def __init__(
        self,
        client: Any = None,
        *,
        cache: ReplayCache | str | Path,
        mode: ReplayMode = "auto",
        capabilities: ClientCapabilities | None = None,
        model: str | None = None,
    ) -> None:
        if mode not in ("record", "replay", "auto"):
            raise ConfigError(
                f"ReplayClient(mode={mode!r}) must be 'record', 'replay' or 'auto' "
                f"(plan §9.6)."
            )
        if client is None and mode != "replay":
            raise ConfigError(
                f"ReplayClient(mode={mode!r}) needs a wrapped client to call on a "
                f"cache miss. Only mode='replay' runs without one — that is what "
                f"makes the CI lane keyless (plan §9.6)."
            )
        self._client = client
        self.cache = cache if isinstance(cache, ReplayCache) else ReplayCache(cache)
        self.mode: ReplayMode = mode
        self.capabilities = (
            capabilities if capabilities is not None else capabilities_of(client)
        )
        resolved_model = model or _client_model(client)
        if not resolved_model:
            raise ConfigError(
                "ReplayClient could not determine a model id, and the cache key is "
                "content-addressed on it (plan §9.6). Pass model='provider/name'."
            )
        self.model = resolved_model
        self.model_id = resolved_model
        self.provider = _client_provider(client)

    # -- introspection ------------------------------------------------------- #

    @property
    def wrapped(self) -> Any:
        """The client being recorded from, or ``None`` under strict replay."""
        return self._client

    def __repr__(self) -> str:
        return (
            f"ReplayClient(mode={self.mode!r}, model={self.model!r}, "
            f"cache={len(self.cache)} entries)"
        )

    # -- LLMClient / AsyncLLMClient ------------------------------------------ #

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """Serve one routing call from the cache, or record it (plan §9.6)."""
        key = self.key_for(request)
        entry = self.cache.get(key) if self.mode != "record" else None
        if entry is not None:
            return entry.to_result()
        self._require_live(key)
        if not callable(getattr(self._client, "complete", None)):
            raise ConfigError(
                f"ReplayClient(mode={self.mode!r}) must record through a synchronous "
                f"client for route(), but {type(self._client).__name__} implements "
                f"only acomplete(). Use aroute(), or wrap a sync client (plan §2.5)."
            )
        result: LLMResult[Any] = self._client.complete(request)
        self.cache.put(key, CacheEntry.from_result(key, result))
        return result

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """Async twin of :meth:`complete` (plan §2.5).

        A sync-only wrapped client is driven through ``asyncio.to_thread`` so a
        re-record lane never blocks the event loop — the same direction §2.5 gives
        the Router for a sync/async mismatch.
        """
        key = self.key_for(request)
        entry = self.cache.get(key) if self.mode != "record" else None
        if entry is not None:
            return entry.to_result()
        self._require_live(key)
        acomplete = getattr(self._client, "acomplete", None)
        if callable(acomplete):
            result: LLMResult[Any] = await acomplete(request)
        else:
            result = await asyncio.to_thread(self._client.complete, request)
        self.cache.put(key, CacheEntry.from_result(key, result))
        return result

    # -- keys ---------------------------------------------------------------- #

    def key_for(self, request: LLMRequest) -> CacheKey:
        """Build this request's :class:`CacheKey` (plan §9.6)."""
        base = CacheKey(
            model=self.model,
            prompt_hash=prompt_hash(request),
            schema_hash=schema_hash(request),
            temperature=request.temperature,
            seed=request.seed,
            sample_index=0,
        )
        return base.model_copy(update={"sample_index": self._next_sample(base.digest)})

    def _next_sample(self, digest: str) -> int:
        """Next ``sample_index`` for an identical call — see :func:`sample_scope`.

        0 outside a scope: without decision boundaries there is no such thing as
        "the second sample", and counting anyway would turn every repeat call in
        a process into a cache miss.
        """
        scoped = _SAMPLE_COUNTS.get()
        if scoped is None:
            return 0
        index = scoped.get(digest, 0)
        scoped[digest] = index + 1
        return index

    def _require_live(self, key: CacheKey) -> None:
        """Fail the run on a strict-replay miss (plan §9.6: "cache miss fails")."""
        if self.mode == "replay":
            raise ConfigError(
                f"replay cache miss in strict mode: {key.describe()}\n"
                f"  cache: {self.cache.path} ({len(self.cache)} entries)\n"
                f"A miss means the prompt, the wire schema, the model or the "
                f"sampling knobs changed since the cache was recorded. Re-record "
                f"with mode='record' (or 'auto') against a real provider and commit "
                f"the refreshed cache — the replay lane is deterministic, keyless "
                f"and free precisely because it never falls back to the network "
                f"(plan §9.6)."
            )


def _client_model(client: Any) -> str | None:
    """Read a client's model id the way ``Router`` does (plan §8.2)."""
    for attribute in ("model", "model_id", "model_name"):
        value = getattr(client, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _client_provider(client: Any) -> str | None:
    """Read a client's vendor name the way ``Router`` does (plan §8.1)."""
    for attribute in ("provider", "route_prefix", "vendor"):
        value = getattr(client, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None
