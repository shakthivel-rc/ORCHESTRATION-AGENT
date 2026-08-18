"""Contract tests for the exception taxonomy (plan §3.8).

The tree is API, not an implementation detail: callers write ``except
ConfigError`` around startup and ``except ProviderError`` around the call, and
the retry driver (§4.5) dispatches on ``retryable``. Reparenting any node here
silently changes which ``except`` clause fires in someone's production handler.

The governing rule, from §3.8:

    *raise* iff the caller's configuration or infrastructure is broken such that
    no valid decision could exist; *degrade to a Decision kind* iff the system is
    healthy but the model is uncertain or its output unusable.

Which is why the negative assertions below matter as much as the positive ones:
there is deliberately **no** exception for "no eligible routes" (§13 ruling #3)
and none for "unparseable output" (§13 ruling #2) — those are
``AbstainDecision`` outcomes.
"""

from __future__ import annotations

import switchboard
from switchboard.errors import (
    ConfigError,
    MissingDependencyError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimit,
    ProviderTimeout,
    RegistryError,
    SwitchboardError,
)

ALL_ERRORS = (
    SwitchboardError,
    ConfigError,
    RegistryError,
    MissingDependencyError,
    ProviderError,
    ProviderTimeout,
    ProviderRateLimit,
    ProviderAuthError,
)


# --------------------------------------------------------------------------- #
# The exact tree of plan §3.8
# --------------------------------------------------------------------------- #


def test_switchboard_error_is_the_single_root() -> None:
    assert issubclass(SwitchboardError, Exception)
    assert SwitchboardError.__bases__ == (Exception,)
    for error in ALL_ERRORS:
        assert issubclass(error, SwitchboardError)


def test_the_root_has_exactly_two_branches() -> None:
    """§3.8's tree has two arms and only two: configuration and transport."""
    assert set(SwitchboardError.__subclasses__()) == {ConfigError, ProviderError}


def test_config_error_branch() -> None:
    assert ConfigError.__bases__ == (SwitchboardError,)
    assert set(ConfigError.__subclasses__()) == {RegistryError, MissingDependencyError}


def test_registry_error_is_a_config_error() -> None:
    """A bad catalog is bad *configuration*, so one ``except ConfigError`` at

    startup catches duplicate names, an empty registry and a bad ``args_model``
    alongside bad DSL and invalid thresholds.
    """
    assert issubclass(RegistryError, ConfigError)
    assert issubclass(RegistryError, SwitchboardError)
    assert not issubclass(RegistryError, ProviderError)
    assert RegistryError.__bases__ == (ConfigError,)


def test_missing_dependency_error_is_a_config_error() -> None:
    """§4.2: a missing extra is caught at ``Router(...)`` construction, which is

    configuration time — not a transport failure.
    """
    assert issubclass(MissingDependencyError, ConfigError)
    assert not issubclass(MissingDependencyError, ProviderError)
    assert MissingDependencyError.__bases__ == (ConfigError,)


def test_provider_error_branch() -> None:
    assert ProviderError.__bases__ == (SwitchboardError,)
    assert set(ProviderError.__subclasses__()) == {
        ProviderTimeout,
        ProviderRateLimit,
        ProviderAuthError,
    }
    for leaf in (ProviderTimeout, ProviderRateLimit, ProviderAuthError):
        assert leaf.__bases__ == (ProviderError,)
        assert not issubclass(leaf, ConfigError)


def test_the_two_branches_never_overlap() -> None:
    """Nothing may be catchable as both — the branch determines whether the

    failure is the caller's config or the network.
    """
    assert not issubclass(ConfigError, ProviderError)
    assert not issubclass(ProviderError, ConfigError)


def test_catching_the_root_catches_everything_the_library_raises() -> None:
    for error in ALL_ERRORS:
        try:
            raise error("boom") if error is not MissingDependencyError else error("otel", "pkg")
        except SwitchboardError as caught:
            assert isinstance(caught, error)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"{error.__name__} was not caught")


def test_the_module_exports_exactly_the_documented_tree() -> None:
    assert set(switchboard.errors.__all__) == {error.__name__ for error in ALL_ERRORS}


def test_every_error_is_re_exported_from_the_package_root() -> None:
    """§3 promises the tree from the top level; framework glue catches these by

    importing ``switchboard``, never ``switchboard.errors``.
    """
    for error in ALL_ERRORS:
        assert getattr(switchboard, error.__name__) is error
        assert error.__name__ in switchboard.__all__


def test_degraded_outcomes_are_deliberately_not_exceptions() -> None:
    """§13 rulings #2 and #3: "no eligible routes" and "unparseable output" are

    ``AbstainDecision`` results. If either ever acquires an exception class, the
    §3.8 rule has been broken and this test should fail loudly.
    """
    for absent in (
        "NoEligibleRoutesError",
        "UnparseableOutputError",
        "InvalidArgsError",
        "LowConfidenceError",
        "AbstainError",
        "ClarifyError",
    ):
        assert not hasattr(switchboard.errors, absent)
        assert not hasattr(switchboard, absent)


# --------------------------------------------------------------------------- #
# Retry semantics (plan §3.8, §4.5)
# --------------------------------------------------------------------------- #


def test_retryable_is_a_class_attribute_so_bare_instances_are_conservative() -> None:
    """The retry driver reads it off the class, so an adapter that raises a bare

    ``ProviderError`` gets the safe answer (no retry) rather than an AttributeError.
    """
    assert ProviderError.retryable is False
    assert ProviderError("boom").retryable is False


def test_transient_failures_are_retryable() -> None:
    assert ProviderTimeout.retryable is True
    assert ProviderRateLimit.retryable is True


def test_auth_failures_are_never_retried() -> None:
    """§3.8: retrying rejected credentials burns the budget and can trip lockouts."""
    assert ProviderAuthError.retryable is False
    assert ProviderAuthError("401").retryable is False


def test_rate_limit_carries_an_optional_retry_after() -> None:
    """§3.8: "honors Retry-After". ``None`` means fall back to expo backoff, so

    the attribute must always exist rather than being conditionally set.
    """
    assert ProviderRateLimit("slow down").retry_after is None
    limited = ProviderRateLimit("slow down", retry_after=12.5)
    assert limited.retry_after == 12.5
    assert str(limited) == "slow down"


def test_rate_limit_retry_after_is_keyword_only() -> None:
    """A positional second argument is an exception *message* part, never a delay —

    otherwise ``ProviderRateLimit("rate limited", 30)`` would silently mean two
    different things depending on the adapter.
    """
    positional = ProviderRateLimit("rate limited", 30)
    assert positional.retry_after is None
    assert positional.args == ("rate limited", 30)


# --------------------------------------------------------------------------- #
# MissingDependencyError names the extra (plan §3.8, §4.2, §10.2)
# --------------------------------------------------------------------------- #


def test_missing_dependency_message_names_the_pip_extra_to_install() -> None:
    """The message is the fix: an operator hitting this at startup must be able to

    copy-paste their way out without reading §10.2's extras matrix.
    """
    error = MissingDependencyError("instructor", "instructor")
    message = str(error)
    assert "pip install switchboard[instructor]" in message
    assert "'instructor'" in message
    assert "not installed" in message


def test_missing_dependency_distinguishes_the_extra_from_the_package() -> None:
    """They differ in practice: the ``gemini`` extra installs ``google-genai``,

    and telling the user to ``pip install switchboard[google-genai]`` would be
    actively wrong.
    """
    error = MissingDependencyError("gemini", "google-genai")
    assert error.extra == "gemini"
    assert error.package == "google-genai"
    assert "pip install switchboard[gemini]" in str(error)
    assert "'google-genai'" in str(error)
    assert "switchboard[google-genai]" not in str(error)


def test_missing_dependency_is_catchable_as_a_config_error() -> None:
    try:
        raise MissingDependencyError("otel", "opentelemetry-api")
    except ConfigError as caught:
        assert isinstance(caught, MissingDependencyError)
        assert caught.extra == "otel"


def test_error_instances_keep_their_message() -> None:
    for error in (SwitchboardError, ConfigError, RegistryError, ProviderError, ProviderTimeout):
        assert str(error("something went wrong")) == "something went wrong"
