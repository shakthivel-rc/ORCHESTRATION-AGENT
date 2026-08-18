"""Bring-your-own plain-callable adapter (plan §4.1 coercion table).

This is the **zero-extras path**: it depends on nothing but the stdlib and
Pydantic, so ``pip install switchboard`` alone is enough to route. Every other
adapter is optional; this one must always work.

What it accepts (signature-inspected, per the plan's coercion table)::

    Callable[[str], ...]          # the rendered prompt
    Callable[[LLMRequest], ...]   # the full request object
    async variants of either      # auto-detected

What it accepts back::

    str          -> LLMResult(parsed=None, raw_text=s)     # the repair loop parses it
    dict         -> validated against request.output_schema
    LLMResult    -> passed straight through

Anything else is a caller mistake and raises :class:`~switchboard.errors.ConfigError`.

The wrapper declares an all-conservative :class:`ClientCapabilities`, which is
load-bearing rather than lazy: ``structured="none"`` arms the full
validate-and-retry loop (§4.5) and ``logprobs=False`` leaves the confidence
engine inert under the no-signal rule (§6.3). A caller whose callable *does*
populate ``token_logprobs`` can say so by passing ``capabilities=``.

Sync/async mismatch follows plan §2.5: an async-only callable driven from
:meth:`CallableAdapter.complete` is a ``ConfigError`` (construct with
``require_sync=True`` to get that error at construction time; the Router reads
:attr:`CallableAdapter.is_async` to check early). A sync callable driven from
:meth:`CallableAdapter.acomplete` is wrapped in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_type_hints

from pydantic import BaseModel, ValidationError

from switchboard.errors import ConfigError
from switchboard.providers.base import ClientCapabilities, LLMRequest, LLMResult, Usage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

__all__ = ["CallMode", "CallableAdapter", "render_prompt"]

CallMode = Literal["prompt", "request"]
"""How the wrapped callable wants its single positional argument."""

#: Parameter names that mean "the whole LLMRequest" when the annotation is missing.
_REQUEST_NAMES = frozenset({"request", "req", "llm_request", "llmrequest", "r"})

#: Parameter names that mean "the rendered prompt string" when the annotation is missing.
_PROMPT_NAMES = frozenset({"prompt", "text", "query", "message", "content", "s", "q", "input"})

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

#: Reached through a constant, not a literal: we need the ``__call__`` *function object* to
#: test whether it is ``async def``, which is a different question from "is this callable"
#: (the check ruff B004 is about).
_CALL = "__call__"


def render_prompt(request: LLMRequest, separator: str = "\n\n") -> str:
    """Flatten a request's segments into the single string a BYO callable receives.

    Segment order is preserved exactly (stable → variable, plan §4.6), so the
    rendered prefix stays byte-stable across requests and any implicit prefix
    caching downstream of the callable still hits. Roles are *not* stamped into
    the text: the segments assembled by ``engine/prompt.py`` already read as
    instructions, and injecting ``"system:"`` markers would change the bytes a
    caller sees for no gain.
    """
    return separator.join(segment.content for segment in request.segments)


def _unwrap(fn: object) -> object:
    """Peel ``functools.partial`` layers and ``functools.wraps`` indirection."""
    target = fn
    for _ in range(8):  # bounded: no pathological chain can spin here
        if isinstance(target, functools.partial):
            target = target.func
            continue
        wrapped = getattr(target, "__wrapped__", None)
        if wrapped is not None and wrapped is not target:
            target = wrapped
            continue
        break
    return target


def _detect_async(fn: object) -> bool:
    """Whether ``fn`` returns an awaitable, across partials, wrappers and ``__call__``.

    ``inspect.iscoroutinefunction`` already follows ``functools.partial`` on
    supported runtimes, but not every wrapper shape, and never a callable object
    whose ``__call__`` is ``async def`` — both of which the plan requires us to
    auto-detect. So the checks are layered, cheapest first.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    target = _unwrap(fn)
    if inspect.iscoroutinefunction(target):
        return True
    dunder_call = getattr(type(target), _CALL, None)
    if dunder_call is not None and inspect.iscoroutinefunction(dunder_call):
        return True
    bound_call = getattr(target, _CALL, None)
    return bool(bound_call is not None and inspect.iscoroutinefunction(_unwrap(bound_call)))


def _hints_for(fn: object) -> dict[str, Any]:
    """Best-effort resolved type hints; ``{}` when they cannot be evaluated.

    ``get_type_hints`` evaluates string annotations, which can fail for any
    number of legitimate reasons (a ``TYPE_CHECKING``-only import in the
    caller's module, a forward reference to a local class). A failure here is
    never fatal — the parameter-name heuristic below covers it.
    """
    target = _unwrap(fn)
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        target = _unwrap(getattr(target, _CALL, target))
    try:
        return dict(get_type_hints(target))
    except Exception:  # annotations are advisory here, never load-bearing
        return {}


def _detect_mode(fn: Callable[..., Any], describe: str) -> CallMode:
    """Infer whether ``fn`` wants the rendered prompt or the whole ``LLMRequest``.

    Order of evidence: resolved annotation, then raw (string) annotation, then
    parameter name, then the documented default of ``"prompt"``. When the
    signature cannot be read at all (C builtins, some callable objects) we fall
    back to the positional-arg-count path and assume the prompt form.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return "prompt"  # unreadable signature -> assume the common `(str) -> ...` shape

    params = [p for p in signature.parameters.values() if p.kind in _POSITIONAL]
    has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values())

    if not params:
        if has_varargs:
            return "prompt"
        raise ConfigError(
            f"BYO client {describe} takes no positional argument. switchboard calls it with exactly "
            f"one: either the rendered prompt (str) or the LLMRequest."
        )

    required = [p for p in params if p.default is inspect.Parameter.empty]
    if len(required) > 1:
        names = ", ".join(p.name for p in required)
        raise ConfigError(
            f"BYO client {describe} requires {len(required)} positional arguments ({names}); "
            f"switchboard calls it with exactly one. Bind the rest with functools.partial."
        )

    first = params[0]
    hints = _hints_for(fn)
    annotation: Any = hints.get(first.name, first.annotation)

    if annotation is LLMRequest:
        return "request"
    if annotation is str:
        return "prompt"
    if isinstance(annotation, str):
        text = annotation.strip().strip("'\"")
        if "LLMRequest" in text:
            return "request"
        if text in {"str", "builtins.str"}:
            return "prompt"

    name = first.name.lower()
    if name in _REQUEST_NAMES:
        return "request"
    if name in _PROMPT_NAMES:
        return "prompt"
    return "prompt"


def _close_awaitable(value: object) -> None:
    """Close a stray coroutine so Python does not warn about it never being awaited."""
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):  # best-effort cleanup on an already-failing path
            close()


class CallableAdapter:
    """Wrap a plain callable so it satisfies both ``LLMClient`` and ``AsyncLLMClient``.

    Args:
        fn: the user's callable. Sync or async; ``(str) -> ...`` or
            ``(LLMRequest) -> ...``; both auto-detected.
        model_id: what lands in ``LLMResult.model_id`` and the audit record.
            Defaults to ``"byo:<callable name>"``.
        capabilities: override the all-conservative default. Only pass this if
            the callable genuinely delivers the capability — declaring
            ``logprobs=True`` without populating ``LLMResult.token_logprobs``
            makes the confidence engine read a signal that is not there.
        call_mode: force ``"prompt"`` or ``"request"`` when inspection guesses
            wrong (e.g. a ``*args`` wrapper).
        require_sync: raise ``ConfigError`` at construction if ``fn`` is async.
            The sync ``Router`` passes this to get the plan §2.5 mismatch error
            at construction rather than on the first call.
        prompt_separator: joiner used by :func:`render_prompt`.

    Attributes:
        is_async: True when the callable returns an awaitable. The Router reads
            this to detect a sync-driver/async-client mismatch early.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        model_id: str | None = None,
        capabilities: ClientCapabilities | None = None,
        call_mode: CallMode | None = None,
        require_sync: bool = False,
        prompt_separator: str = "\n\n",
    ) -> None:
        if not callable(fn):
            raise ConfigError(
                f"BYO client must be callable; got {type(fn).__name__!r}. Pass a function, an "
                f"object with __call__, or an LLMClient implementation."
            )
        self._fn = fn
        self._separator = prompt_separator
        self.model_id = model_id or f"byo:{self._name_of(fn)}"
        # Per-instance, never a shared class attribute: capabilities are read (and in v0.2 probed)
        # per client, and a shared mutable model would leak edits across routers.
        self.capabilities = capabilities if capabilities is not None else ClientCapabilities()
        self.is_async = _detect_async(fn)
        self.call_mode: CallMode = call_mode or _detect_mode(fn, self._describe())

        if require_sync and self.is_async:
            raise ConfigError(
                f"BYO client {self._describe()} is async but this Router is synchronous "
                f"(plan §2.5). Use `await router.aroute(...)`, or pass a synchronous callable."
            )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _name_of(fn: object) -> str:
        target = _unwrap(fn)
        name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
        return str(name) if name else type(target).__name__

    def _describe(self) -> str:
        return f"{self._name_of(self._fn)}()"

    def _invoke(self, request: LLMRequest) -> Any:
        if self.call_mode == "request":
            return self._fn(request)
        return self._fn(render_prompt(request, self._separator))

    def _result(
        self,
        *,
        parsed: BaseModel | None,
        raw_text: str,
    ) -> LLMResult[Any]:
        return LLMResult(parsed=parsed, raw_text=raw_text, usage=Usage(), model_id=self.model_id)

    def _coerce(self, out: Any, request: LLMRequest) -> LLMResult[Any]:
        """Apply the plan §4.1 coercion table to whatever the callable returned."""
        if isinstance(out, LLMResult):
            return out

        if isinstance(out, str):
            # Unparsed on purpose: engine/validate.py owns parsing + the repair loop, so a
            # plain-text callable gets the same treatment as a model that returned loose JSON.
            return self._result(parsed=None, raw_text=out)

        payload: Any
        if isinstance(out, BaseModel):
            if isinstance(out, request.output_schema):
                return self._result(parsed=out, raw_text=out.model_dump_json())
            payload = out.model_dump(mode="json")
        elif isinstance(out, Mapping):
            payload = dict(out)
        else:
            raise ConfigError(
                f"BYO client {self._describe()} returned {type(out).__name__!r}. switchboard "
                f"accepts str (raw model text), dict (validated against the wire schema), or "
                f"LLMResult (passed through) — see plan §4.1."
            )

        raw_text = json.dumps(payload, default=str, ensure_ascii=False)
        try:
            parsed = request.output_schema.model_validate(payload)
        except ValidationError:
            # Not a config error: the callable did its job, the *content* was wrong. Degrading to
            # parsed=None hands it to the validate-and-retry loop (§4.5), which is armed because
            # this adapter reports structured="none".
            return self._result(parsed=None, raw_text=raw_text)
        return self._result(parsed=parsed, raw_text=raw_text)

    # ------------------------------------------------------------ LLMClient

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        """Call the wrapped callable synchronously (plan §4.1)."""
        if self.is_async:
            raise ConfigError(
                f"BYO client {self._describe()} is async and cannot be driven by the synchronous "
                f"route(); use `await router.aroute(...)` (plan §2.5)."
            )
        out = self._invoke(request)
        if inspect.isawaitable(out):
            _close_awaitable(out)
            raise ConfigError(
                f"BYO client {self._describe()} returned an awaitable from a synchronous call; "
                f"use `await router.aroute(...)` (plan §2.5)."
            )
        return self._coerce(out, request)

    # ------------------------------------------------------- AsyncLLMClient

    async def acomplete(self, request: LLMRequest) -> LLMResult[Any]:
        """Call the wrapped callable from async code (plan §4.1, §2.5).

        A synchronous callable is offloaded with ``asyncio.to_thread`` so it
        cannot block the event loop — the documented direction of the §2.5
        mismatch rule.
        """
        out = self._invoke(request) if self.is_async else await asyncio.to_thread(self._invoke, request)
        if inspect.isawaitable(out):
            out = await out
        return self._coerce(out, request)

    def __repr__(self) -> str:
        kind = "async" if self.is_async else "sync"
        return f"CallableAdapter({self.model_id!r}, mode={self.call_mode!r}, {kind})"
