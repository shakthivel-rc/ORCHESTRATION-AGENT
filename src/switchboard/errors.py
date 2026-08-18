"""Unified exception taxonomy for switchboard (plan §3.8).

The governing rule, quoted from the plan:

    *raise* iff the caller's configuration or infrastructure is broken such that
    no valid decision could exist; *degrade to a Decision kind* iff the system is
    healthy but the model is uncertain or its output unusable.

Consequently there is no ``NoEligibleRoutesError`` (§13 ruling #3) and no
"unparseable output" exception (§13 ruling #2): both of those are
:class:`~switchboard.core.decision.AbstainDecision` outcomes, not exceptions.

The tree (§3.8)::

    SwitchboardError
    ├── ConfigError
    │   ├── RegistryError
    │   └── MissingDependencyError
    └── ProviderError
        ├── ProviderTimeout        # retryable
        ├── ProviderRateLimit      # retryable, honors Retry-After
        └── ProviderAuthError      # never retried

This module is **stdlib-only** and is imported by ``core/`` and ``router.py``;
it must never grow a third-party import (plan §2.4).
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "ConfigError",
    "MissingDependencyError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimit",
    "ProviderTimeout",
    "RegistryError",
    "SwitchboardError",
]


class SwitchboardError(Exception):
    """Base class for every exception raised by switchboard.

    Catching this catches everything the library raises on its own behalf;
    it never wraps errors raised by user-supplied callables verbatim.
    """


# --------------------------------------------------------------------------- #
# Configuration / caller-error branch — always raised, never degraded.
# --------------------------------------------------------------------------- #


class ConfigError(SwitchboardError):
    """The caller's configuration cannot produce a valid decision (plan §3.8).

    Raised for: bad DSL strings, an unknown fallback route name, invalid
    thresholds, ``K < 1``, a sync ``Router`` wired to an async-only client, and
    ``shortlist=None`` with more entitled routes than ``max_candidates``
    (silent truncation would drop routes).
    """


class RegistryError(ConfigError):
    """The route catalog itself is invalid (plan §3.8).

    Raised for duplicate or malformed route names, an empty registry, and an
    ``args_model`` that is not a :class:`pydantic.BaseModel` subclass.
    """


class MissingDependencyError(ConfigError):
    """A spec named an optional extra that is not installed (plan §3.8, §4.2).

    Raised **eagerly at ``Router(...)`` construction** so a missing extra fails
    fast rather than on the first request (plan §2.4). Adapters never raise a
    bare :class:`ImportError` (§13 ruling #2) — they translate it into this.

    Args:
        extra: the ``pip install switchboard[<extra>]`` extra name, e.g. ``"openai"``.
        package: the importable distribution that was missing, e.g. ``"openai"``.
    """

    def __init__(self, extra: str, package: str) -> None:
        self.extra = extra
        self.package = package
        super().__init__(
            f"Optional dependency {package!r} is required for this feature but is "
            f"not installed. Install it with: pip install switchboard[{extra}]"
        )


# --------------------------------------------------------------------------- #
# Provider / transport branch.
# --------------------------------------------------------------------------- #


class ProviderError(SwitchboardError):
    """Transport-layer failure talking to the model provider (plan §3.8).

    By default a provider failure that survives the retry budget propagates to
    the caller; ``Router(on_provider_error="abstain")`` converts it into
    ``abstain(reason="provider_error")`` instead [v0.2].

    Attributes:
        retryable: whether the retry driver (§4.5, ``RetrySpec.provider_attempts``)
            may re-issue the request. Consulted as a class attribute so bare
            ``ProviderError`` instances are treated conservatively (not retried).
    """

    retryable: ClassVar[bool] = False


class ProviderTimeout(ProviderError):
    """The provider did not answer in time — retryable (3 attempts, expo+jitter)."""

    retryable: ClassVar[bool] = True


class ProviderRateLimit(ProviderError):
    """The provider rate-limited the request — retryable, honors ``Retry-After``.

    Args:
        retry_after: seconds parsed from the provider's ``Retry-After`` header,
            when the adapter could recover it. ``None`` means "use the configured
            exponential backoff".
    """

    retryable: ClassVar[bool] = True

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Credentials were rejected — **never** retried (plan §3.8)."""

    retryable: ClassVar[bool] = False
