"""Evaluation harness (plan §9; §13 ruling #17).

Named ``evals``, not ``eval``, to avoid shadowing the builtin.

The harness is a **product surface, not test infrastructure** (§9). The report's
adoption wedge is accuracy credibility on the user's own catalog, and its ship
rule is a margin over what a developer would hand-roll — so the headline number
this package produces is a *delta against the ``naive-full-catalog`` baseline*,
never a standalone accuracy.

What ships in v0.1 (§9.7):

* **Fixtures** (:mod:`switchboard.evals.fixtures`) — the ``EvalCase`` schema whose
  expected labels mirror the ``Decision`` union exactly (clarify and abstain are
  gold labels, not error buckets), the version-preserving
  ``registry_to_fixture`` / ``registry_from_fixture`` catalog freeze, the
  ``{"fixture": "sb-eval/1"}`` JSONL container, and the dogfood suite §9.6 runs on
  every PR.
* **Record/replay** (:mod:`switchboard.evals.cache`) — every provider call
  content-addressed by ``(model, prompt_hash, schema_hash, temperature, seed,
  sample_index)``. ``mode="replay"`` is strict, so the CI lane is deterministic,
  keyless and free.
* **The runner** (:mod:`switchboard.evals.harness`) — route accuracy, schema
  validity, hallucinated route names, shortlist recall@K, clarify/abstain
  confusion and latency percentiles, all derived from the ``Decision`` and
  ``AuditRecord`` production already emits, plus the naive baseline and the
  proto-G1/G4 gates.

**Packaging (§13 ruling #17).** Everything here is Pydantic + stdlib. The CLI
(``switchboard eval ...``), ``gates.yaml`` and rendered reports are the [v0.2]
``[eval]`` extra — importing this package must never require it, because a
keyless replay lane has to work in a bare install.
"""

from __future__ import annotations

from switchboard.evals.cache import (
    CACHE_SCHEMA,
    CacheEntry,
    CacheKey,
    ReplayCache,
    ReplayClient,
    ReplayMode,
    sample_scope,
)
from switchboard.evals.fixtures import (
    FIXTURE_SCHEMA,
    EvalCase,
    EvalSuite,
    Expected,
    ExpectedAbstain,
    ExpectedClarify,
    ExpectedMultiRoute,
    ExpectedRoute,
    dogfood_registry,
    dogfood_suite,
    load_cases,
    load_suite,
    registry_from_fixture,
    registry_to_fixture,
    save_cases,
    save_suite,
)
from switchboard.evals.harness import (
    BASELINES,
    NAIVE_FULL_CATALOG,
    NAIVE_SYSTEM_PROMPT,
    ArmResult,
    CaseResult,
    GateResult,
    SuiteMetrics,
    SuiteResult,
    arun_suite,
    build_naive_baseline,
    run_suite,
)

__all__ = [
    "BASELINES",
    "CACHE_SCHEMA",
    "FIXTURE_SCHEMA",
    "NAIVE_FULL_CATALOG",
    "NAIVE_SYSTEM_PROMPT",
    "ArmResult",
    "CacheEntry",
    "CacheKey",
    "CaseResult",
    "EvalCase",
    "EvalSuite",
    "Expected",
    "ExpectedAbstain",
    "ExpectedClarify",
    "ExpectedMultiRoute",
    "ExpectedRoute",
    "GateResult",
    "ReplayCache",
    "ReplayClient",
    "ReplayMode",
    "SuiteMetrics",
    "SuiteResult",
    "arun_suite",
    "build_naive_baseline",
    "dogfood_registry",
    "dogfood_suite",
    "load_cases",
    "load_suite",
    "registry_from_fixture",
    "registry_to_fixture",
    "run_suite",
    "sample_scope",
    "save_cases",
    "save_suite",
]
