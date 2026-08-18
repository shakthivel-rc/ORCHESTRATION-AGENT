"""Optional shortlist backends (plan §5.2).

The default BM25 shortlister is pure Python and lives in ``engine/shortlist.py``
— in core, no extra (§13 ruling #7). This package holds only what needs an
optional dependency: :mod:`switchboard.shortlisters.embedding_backends`, behind
the ``[embed]`` extra, one of the three modules permitted to import third-party
SDKs (§2.4).

Re-exports here are **lazy by module-level ``__getattr__``**. Importing
``switchboard.shortlisters`` — which happens implicitly whenever anything
touches ``switchboard.shortlisters.embedding_backends`` — must not pull an
embedding SDK into ``sys.modules``, because the CI bare-venv guard asserts a
routing call leaves the deny-list untouched (§2.4). ``embedding_backends``
itself is already side-effect free, so the laziness here is belt-and-braces:
it keeps the package importable even if that module ever grows a top-level
import it should not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BACKENDS", "DEFAULT_MODELS", "get_backend"]

if TYPE_CHECKING:
    from switchboard.shortlisters.embedding_backends import (
        BACKENDS,
        DEFAULT_MODELS,
        get_backend,
    )


def __getattr__(name: str) -> Any:
    """Resolve the re-exported names on first access (PEP 562)."""
    if name in __all__:
        from switchboard.shortlisters import embedding_backends

        return getattr(embedding_backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
