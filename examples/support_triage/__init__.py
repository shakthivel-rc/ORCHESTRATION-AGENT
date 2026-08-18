"""The flagship ``support_triage`` example (plan §10.3).

A ~120-route customer-support catalog with entitlements, ``args_model`` s, a
pinned escalation route and a labelled gold set, plus a runnable demo that scores
a ``Router`` over it offline.

* ``catalog.py`` — the routes, the entitlements and ``GOLD_CASES``. Also the
  benchmark fixture the [v0.2] eval CLI reads.
* ``demo.py`` — runs the catalog through a ``Router`` with a deterministic
  offline stub client and prints the shortlist behaviour and an accuracy
  readout.

Run it with zero extras installed::

    python examples/support_triage/demo.py

Re-exports are deliberately thin: the interesting names live in ``catalog``.
"""

from __future__ import annotations

from .catalog import (
    ABSTAIN,
    CLARIFY,
    DOMAINS,
    ENTITLEMENTS,
    FALLBACK_ROUTE,
    GOLD_CASES,
    GoldCase,
    registry,
    routes,
)

__all__ = [
    "ABSTAIN",
    "CLARIFY",
    "DOMAINS",
    "ENTITLEMENTS",
    "FALLBACK_ROUTE",
    "GOLD_CASES",
    "GoldCase",
    "registry",
    "routes",
]
