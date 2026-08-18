"""Package version resolution.

The git tag is the single source of truth (hatch-vcs writes it into the installed distribution's
metadata at build time; see plan §10.1). At runtime we read it back out of the installed
distribution rather than hard-coding a literal, so a source checkout, an editable install and a
wheel all agree.

Stdlib only — this module is imported by ``switchboard/__init__.py`` and must never pull in a
third-party package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

__all__ = ["__version__"]

#: Version used when the package is not installed as a distribution (e.g. a bare source checkout
#: run with ``PYTHONPATH=src``, or a not-yet-tagged repository). Mirrors the ``fallback-version``
#: configured for hatch-vcs in ``pyproject.toml``.
_FALLBACK_VERSION = "0.1.0.dev0"

try:
    __version__: str = _dist_version("switchboard")
except PackageNotFoundError:  # pragma: no cover - depends on install state, not on logic
    __version__ = _FALLBACK_VERSION
