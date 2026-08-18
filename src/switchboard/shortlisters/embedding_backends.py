"""Packaged embedding backends for :class:`EmbeddingShortlister` (plan §5.2, §10.2).

This module is one of exactly three places in the codebase permitted to import a
third-party SDK (plan §2.4 — the others are ``providers/*_adapter.py`` and
``telemetry/otel.py``), and even here **nothing is imported at module import
time**: every SDK import lives inside :func:`get_backend`'s factory functions, so
``import switchboard.shortlisters.embedding_backends`` is free of side effects
and costs nothing in a bare-Pydantic install. That is what the CI bare-venv
guard asserts (§2.4).

An ``ImportError`` is never allowed to escape: it is translated into
:class:`~switchboard.errors.MissingDependencyError`, which names the extra to
install (§3.8, §13 ruling #2).

Nothing here is required to use dense retrieval. A BYO callable works with
**zero extras** (§10.2)::

    Router(shortlist=EmbeddingShortlister(my_embed_fn))

These backends exist only so that users who *want* a batteries-included path do
not have to write the batching themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from switchboard.errors import ConfigError, MissingDependencyError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from switchboard.engine.shortlist import EmbedFn

__all__ = ["BACKENDS", "DEFAULT_MODELS", "get_backend"]

BACKENDS: tuple[str, ...] = ("fastembed", "openai")
"""Backend names accepted by :func:`get_backend`."""

DEFAULT_MODELS: dict[str, str] = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}
"""Default model per backend. ``bge-small-en-v1.5`` is 384-dim and runs locally
with no API key; ``text-embedding-3-small`` is the plan's named OpenAI model."""

_OPENAI_BATCH = 512
"""Texts per OpenAI embeddings request. Route catalogs are small, but a 5k-route
registry still needs batching on the first build."""


def get_backend(name: str, *, model: str | None = None, **options: Any) -> EmbedFn:
    """Return an embedding callable for the named backend (plan §5.2).

    Args:
        name: ``"fastembed"`` (local, no API key) or ``"openai"``
            (``text-embedding-3-small`` by default).
        model: model id; falls back to :data:`DEFAULT_MODELS`.
        **options: forwarded to the backend — ``fastembed`` accepts any
            ``TextEmbedding`` kwarg (``cache_dir``, ``threads``, ...);
            ``openai`` accepts ``client`` (a pre-configured SDK client),
            ``dimensions`` and ``batch_size``.

    Returns:
        A ``Callable[[list[str]], Sequence[Sequence[float]]]`` — exactly the BYO
        embed contract, so a packaged backend and a hand-written one are
        interchangeable. Vectors are **not** normalized here;
        :class:`~switchboard.engine.shortlist.EmbeddingShortlister` normalizes
        everything it receives, so both paths behave identically.

    Raises:
        MissingDependencyError: the backend's SDK is not installed.
        ConfigError: unknown backend name.

    The returned callable is constructed eagerly (the model is loaded / the
    client is built now), so a missing extra or a bad model id surfaces at
    ``Router(...)`` construction rather than on the first request (§2.4).
    """
    key = name.strip().lower()
    if key == "fastembed":
        return _fastembed_backend(model or DEFAULT_MODELS["fastembed"], options)
    if key == "openai":
        return _openai_backend(model or DEFAULT_MODELS["openai"], options)
    raise ConfigError(
        f"unknown embedding backend {name!r}; supported: {list(BACKENDS)}. "
        f"Any Callable[[list[str]], list[list[float]]] also works with zero extras — "
        f"pass it to EmbeddingShortlister directly."
    )


def _fastembed_backend(model: str, options: dict[str, Any]) -> EmbedFn:
    """Local ONNX embeddings via ``fastembed`` — no API key, no network at query time."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - depends on install profile
        raise MissingDependencyError("embed", "fastembed") from exc

    embedder = TextEmbedding(model_name=model, **options)

    def embed(texts: list[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        # fastembed yields numpy arrays; float() them here so nothing downstream
        # of this module ever needs numpy imported (§2.4).
        return [[float(x) for x in vector] for vector in embedder.embed(list(texts))]

    return embed


def _openai_backend(model: str, options: dict[str, Any]) -> EmbedFn:
    """Hosted embeddings via the ``openai`` SDK (``text-embedding-3-small``)."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on install profile
        raise MissingDependencyError("openai", "openai") from exc

    client = options.pop("client", None) or OpenAI()
    batch_size = int(options.pop("batch_size", _OPENAI_BATCH))
    if batch_size < 1:
        raise ConfigError(f"openai embedding batch_size must be >= 1, got {batch_size}")
    extra = dict(options)

    def embed(texts: list[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = client.embeddings.create(model=model, input=batch, **extra)
            # Sort by index: the API documents order but does not guarantee it
            # across retries, and a mis-ordered matrix is a silent recall bug.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append([float(x) for x in item.embedding])
        return vectors

    return embed
