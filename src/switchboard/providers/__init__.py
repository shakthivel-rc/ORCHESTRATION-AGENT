"""Provider layer: the normative I/O contract plus the spec resolver (plan §4.1, §4.2).

``base.py`` holds the dependency-free contract. The sibling ``*_adapter.py``
modules are the only place an optional SDK may be imported, and
:func:`resolve_client` imports them **lazily via importlib** so that:

* importing ``switchboard`` never drags in a provider SDK (§2.4 import topology,
  enforced in CI by the bare-venv guard), and
* a missing extra becomes a :class:`~switchboard.errors.MissingDependencyError`
  naming the exact extra, never a bare ``ImportError`` (§13 ruling #2).

Resolution is **eager at ``Router(...)`` construction**: the Router calls
:func:`resolve_client` in ``__init__``, so a typo'd spec or an uninstalled extra
fails immediately rather than on the first production request (§2.4).

v0.1 ships ``openai``, ``instructor``, ``litellm`` and bring-your-own callables.
Naming a native ``anthropic`` / ``gemini`` / ``bedrock`` adapter raises a
:class:`~switchboard.errors.ConfigError` that says which phase it lands in and
which LiteLLM spec reaches the same model today.

    >>> resolve_client("litellm:gemini/gemini-2.5-flash-lite")   # doctest: +SKIP
    LiteLLMAdapter('gemini/gemini-2.5-flash-lite', structured='grammar')
    >>> resolve_client(lambda prompt: '{"route": "billing"}')     # doctest: +SKIP
    CallableAdapter('byo:<lambda>', mode='prompt', sync)
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple, cast

from switchboard.errors import ConfigError, MissingDependencyError
from switchboard.providers.base import (
    AsyncLLMClient,
    ClientCapabilities,
    LLMClient,
    LLMRequest,
    LLMResult,
    PromptSegment,
    TokenLP,
    Usage,
    capabilities_of,
)

__all__ = [
    "SHIPPED_ADAPTERS",
    "AsyncLLMClient",
    "ClientCapabilities",
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "PromptSegment",
    "TokenLP",
    "Usage",
    "capabilities_of",
    "is_llm_client",
    "parse_client_spec",
    "resolve",
    "resolve_client",
]


class _AdapterEntry(NamedTuple):
    """Where an adapter lives and what it needs installed."""

    module: str
    class_name: str
    extra: str
    package: str


_ADAPTERS: dict[str, _AdapterEntry] = {
    "instructor": _AdapterEntry("instructor_adapter", "InstructorAdapter", "instructor", "instructor"),
    "litellm": _AdapterEntry("litellm_adapter", "LiteLLMAdapter", "litellm", "litellm"),
    "openai": _AdapterEntry("openai_adapter", "OpenAIAdapter", "openai", "openai"),
}

#: Adapter names ``resolve_client`` can build in this release (plan §13 ruling #13).
SHIPPED_ADAPTERS: tuple[str, ...] = tuple(sorted(_ADAPTERS))

#: Adapters the plan schedules for a later phase → (phase, LiteLLM spec to use meanwhile).
_DEFERRED_ADAPTERS: dict[str, tuple[str, str]] = {
    "anthropic": ("v0.2", "litellm:anthropic/claude-haiku-4-5"),
    "gemini": ("v0.2", "litellm:gemini/gemini-2.5-flash-lite"),
    "google": ("v0.2", "litellm:gemini/gemini-2.5-flash-lite"),
    "bedrock": ("v0.5", "litellm:bedrock/anthropic.claude-haiku-4-5-v1:0"),
    "vertex": ("v1.0", "litellm:vertex_ai/gemini-2.5-flash-lite"),
    "databricks": ("v1.0", "litellm:databricks/<model>"),
    "vllm": ("v1.0", "litellm:hosted_vllm/<model>"),
}


def parse_client_spec(spec: str) -> tuple[str, str]:
    """Split ``"adapter:model"`` into its two halves (plan §3.3 client DSL).

    Only the **first** colon separates: model ids legitimately contain colons
    (``"litellm:bedrock/anthropic.claude-haiku-4-5-v1:0"``), so a naive
    ``split(":")`` would corrupt them.

    Raises:
        ConfigError: the spec is empty, has no adapter prefix, or names no model.
    """
    raw = spec.strip()
    if not raw:
        raise ConfigError("Empty client spec. Use 'adapter:model', e.g. 'instructor:openai/gpt-5-nano'.")

    adapter, separator, model = raw.partition(":")
    adapter = adapter.strip().lower()
    model = model.strip()
    if not separator or not adapter or not model:
        raise ConfigError(
            f"Client spec {spec!r} is not in 'adapter:model' form. Examples: "
            f"'instructor:openai/gpt-5-nano', 'litellm:gemini/gemini-2.5-flash-lite'. "
            f"Adapters available in this release: {', '.join(SHIPPED_ADAPTERS)}."
        )
    return adapter, model


def is_llm_client(obj: object) -> bool:
    """Whether ``obj`` already satisfies ``LLMClient`` or ``AsyncLLMClient``.

    Duck-typed on purpose. ``isinstance`` against a ``runtime_checkable``
    Protocol carrying the non-method member ``capabilities`` raises
    ``TypeError`` on some runtimes and silently ignores member *types* on the
    rest, so it would be both fragile and misleading here. Presence of a callable
    ``complete``/``acomplete`` is the real contract; ``capabilities`` is optional
    by design and read through ``capabilities_of`` (§4.1).
    """
    return any(callable(getattr(obj, name, None)) for name in ("complete", "acomplete"))


def _import_adapter_module(entry: _AdapterEntry) -> Any:
    """Import an adapter module, translating any import failure into a typed error.

    The adapter modules themselves import cleanly without their SDK — the SDK
    import is lazy inside them — so an ``ImportError`` here means the *module*
    is unusable, which is still a missing-dependency condition from the caller's
    point of view and must never surface as a bare ``ImportError`` (§13 #2).
    """
    try:
        return importlib.import_module(f"{__name__}.{entry.module}")
    except ImportError as exc:  # pragma: no cover - only reachable on a broken install
        raise MissingDependencyError(entry.extra, entry.package) from exc


def _unknown_adapter_error(adapter: str, spec: str) -> ConfigError:
    deferred = _DEFERRED_ADAPTERS.get(adapter)
    if deferred is not None:
        phase, suggestion = deferred
        return ConfigError(
            f"The native {adapter!r} adapter is scheduled for {phase} and is not part of this "
            f"release: v0.1 ships {', '.join(SHIPPED_ADAPTERS)} plus bring-your-own callables "
            f"(plan §4.2, §13 ruling #13). Use {suggestion!r} to reach the same model through "
            f"LiteLLM in the meantime."
        )
    return ConfigError(
        f"Unknown client adapter {adapter!r} in spec {spec!r}. Available: "
        f"{', '.join(SHIPPED_ADAPTERS)}; or pass an LLMClient instance or a plain callable."
    )


def _build_adapter(adapter: str, model: str, spec: str, kwargs: dict[str, Any]) -> LLMClient | AsyncLLMClient:
    entry = _ADAPTERS.get(adapter)
    if entry is None:
        raise _unknown_adapter_error(adapter, spec)

    module = _import_adapter_module(entry)
    factory = getattr(module, entry.class_name, None)
    if factory is None:  # pragma: no cover - guards against a bad refactor of the table above
        raise ConfigError(
            f"Adapter module {entry.module!r} does not define {entry.class_name!r}; "
            f"this is a switchboard packaging bug."
        )
    # MissingDependencyError from the adapter's own lazy SDK import propagates unchanged —
    # it already names the right extra.
    return cast("LLMClient | AsyncLLMClient", factory(model, **kwargs))


def resolve_client(spec: Any, **adapter_kwargs: Any) -> LLMClient | AsyncLLMClient:
    """Turn a ``client=`` value into something implementing the provider protocol.

    Accepted forms (plan §3.3, §4.1):

    * ``"adapter:model"`` — e.g. ``"instructor:openai/gpt-5-nano"``,
      ``"litellm:gemini/gemini-2.5-flash-lite"``. The adapter module is imported
      lazily and constructed here.
    * an object that already has ``complete`` and/or ``acomplete`` — returned
      unchanged, including a pre-built adapter or a user's own class.
    * a spec object exposing ``.adapter`` and ``.model`` (the ``ClientSpec``
      object form) — treated exactly like the string it stands for.
    * any other callable — wrapped in
      :class:`~switchboard.providers.callable_adapter.CallableAdapter` with
      all-conservative capabilities.

    Args:
        spec: one of the forms above.
        **adapter_kwargs: forwarded to the adapter constructor
            (``capabilities=``, ``extra_kwargs=``, ``require_sync=`` …). Ignored
            when ``spec`` is already a client.

    Returns:
        A client satisfying ``LLMClient``, ``AsyncLLMClient`` or both.

    Raises:
        ConfigError: unusable spec, unknown adapter, or an adapter deferred to a
            later phase.
        MissingDependencyError: the named adapter's extra is not installed
            (a ``ConfigError`` subclass, so callers may catch either).
    """
    if spec is None:
        raise ConfigError(
            "Router(client=...) is required. Pass a spec string "
            "('litellm:gemini/gemini-2.5-flash-lite'), an LLMClient, or a plain callable."
        )

    if isinstance(spec, str):
        adapter, model = parse_client_spec(spec)
        return _build_adapter(adapter, model, spec, adapter_kwargs)

    # Already a client: checked before `callable()` so an adapter whose class also defines
    # __call__ is never re-wrapped as a BYO callable.
    if is_llm_client(spec):
        return cast("LLMClient | AsyncLLMClient", spec)

    adapter_attr = getattr(spec, "adapter", None)
    model_attr = getattr(spec, "model", None)
    if isinstance(adapter_attr, str) and isinstance(model_attr, str):
        rendered = f"{adapter_attr}:{model_attr}"
        return _build_adapter(adapter_attr.strip().lower(), model_attr.strip(), rendered, adapter_kwargs)

    if callable(spec):
        # Lazy by §2.4 policy: providers/__init__ imports no adapter at module scope.
        from switchboard.providers.callable_adapter import CallableAdapter

        return CallableAdapter(spec, **adapter_kwargs)

    raise ConfigError(
        f"Cannot resolve client from {type(spec).__name__!r}. Pass a spec string "
        f"('adapter:model'), an object with complete()/acomplete(), or a plain callable."
    )


#: Spelling used in plan §2.4 (``providers.resolve(spec)``); identical behavior.
resolve = resolve_client
