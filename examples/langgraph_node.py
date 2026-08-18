"""LangGraph conditional-edge node (plan §3.7, §10.3 item 2).

::

    python examples/langgraph_node.py

The node below is the ``.with_structured_output()`` boilerplate switchboard
replaces: one ``await router.aroute(...)`` returns a typed decision, and the node
turns it into a ``Command``. There is no coupling in either direction —
switchboard does not import LangGraph, and LangGraph does not know switchboard
exists.

**LangGraph is optional here.** ``pip install langgraph`` and the real
``Command`` / ``StateGraph`` are used; without it a five-line stand-in ``Command``
keeps the module importable and the ``__main__`` block below still exercises the
half that matters — :func:`route_node` against a real ``Router`` with an offline
client. switchboard itself needs no extras at all.

The three cases the node handles are the whole Decision contract (plan §3.4):

* ``route``    -> go to the handler node named by the decision, carrying the args;
* ``clarify``  -> go ask the user the question the router wrote;
* everything else -> ``unroutable``. With ``Router(fallback=...)`` configured a
  terminal abstain would arrive as ``kind="route"`` instead and never reach this
  branch — that is the point of the one-shape fallback rule (plan §6.6). This
  example deliberately leaves the fallback off so the abstain arm is visible.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, TypedDict

from pydantic import BaseModel

from switchboard import Registry, RequestContext, Route, Router

try:  # pip install langgraph
    from langgraph.types import Command

    HAVE_LANGGRAPH = True
except ImportError:  # pragma: no cover - the offline path this example ships with
    from dataclasses import dataclass, field

    HAVE_LANGGRAPH = False

    @dataclass(frozen=True)
    class Command:  # type: ignore[no-redef]
        """Stand-in for ``langgraph.types.Command`` so this file runs bare."""

        goto: str
        update: dict[str, Any] = field(default_factory=dict)


__all__ = ["State", "build_graph", "route_node", "router"]


# --------------------------------------------------------------------------- #
# Graph state.
# --------------------------------------------------------------------------- #


class State(TypedDict, total=False):
    """The slice of graph state this node reads and writes."""

    input: str
    user_id: str
    args: dict[str, Any]
    question: str


# --------------------------------------------------------------------------- #
# The catalog, and an offline stand-in client (see examples/quickstart_byo.py).
# --------------------------------------------------------------------------- #


class RefundArgs(BaseModel):
    order_id: str
    reason: str | None = None


registry = Registry([
    Route(name="refund", description="Issue or check a refund for an order",
          args_model=RefundArgs, examples=("I want my money back for order 123",),
          tags=frozenset({"billing"})),
    Route(name="track_order", description="Track shipment status for an existing order",
          examples=("where is my package",), tags=frozenset({"shipping"})),
    Route(name="human_handoff", description="Escalate to a human support agent", pinned=True),
])

_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)
_ORDER_RE = re.compile(r"\border\s+#?([a-z0-9-]*\d[a-z0-9-]*)", re.IGNORECASE)


async def offline_client(prompt: str) -> dict[str, Any]:
    """An async BYO callable: switchboard auto-detects the coroutine (plan §4.1)."""
    found = _QUERY_RE.search(prompt)
    query = (found.group(1) if found else "").lower()
    if "package" in query or "track" in query:
        return {"rationale": "A shipment status question.", "kind": "route", "route": "track_order"}
    if "refund" in query or "money back" in query:
        order = _ORDER_RE.search(query)
        return {
            "rationale": "A request for money back.",
            "kind": "route",
            "route": "refund",
            "args": {"order_id": order.group(1)} if order else {},
        }
    if "order" in query:
        return {
            "rationale": "About an order, but which action is unclear.",
            "kind": "clarify",
            "question": "Do you want a refund, or to track the shipment?",
            "candidates": ["refund", "track_order"],
        }
    return {"rationale": "Out of scope.", "kind": "abstain", "reason": "model_elected"}


# One Router per process, built at import time: it is thread- and task-safe, holds
# no per-request state, and resolves all its configuration once (plan §2.5).
# `fallback` is deliberately NOT set here — see the module docstring.
router = Router(registry=registry, client=offline_client, shortlist="auto",
                allow_clarify=True, otel=False)


# --------------------------------------------------------------------------- #
# ===== plan §3.7 LangGraph node — verbatim ================================= #
# --------------------------------------------------------------------------- #


async def route_node(state: State) -> Command:
    d = await router.aroute(state["input"], context=RequestContext(user_id=state["user_id"]))
    # The one-line `case` arms below are the plan §3.7 snippet, kept verbatim; E701 is
    # suppressed rather than the snippet reformatted.
    match d.kind:
        case "route":   return Command(goto=d.route, update={"args": d.args.model_dump() if d.args else {}})  # noqa: E701
        case "clarify": return Command(goto="ask_user", update={"question": d.question})  # noqa: E701
        case _:         return Command(goto="unroutable")     # noqa: E701 - abstain: no fallback configured


# --------------------------------------------------------------------------- #
# ===== end of the snippet ================================================== #
# --------------------------------------------------------------------------- #


def build_graph() -> Any:
    """Wire :func:`route_node` into a real graph. Requires ``langgraph``.

    ``route_node`` returns a ``Command``, so it *is* the conditional edge — no
    separate ``add_conditional_edges`` router function, and no string-matching on
    an LLM's free text. The ``goto`` targets are route names, which is why plan
    §3.1 constrains them to a lowercase identifier alphabet.
    """
    if not HAVE_LANGGRAPH:  # pragma: no cover - offline path
        raise RuntimeError(
            "build_graph() needs LangGraph: pip install langgraph. The routing half of this "
            "example (route_node) runs without it — see __main__ below."
        )
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(State)
    graph.add_node("route", route_node)
    for name in ("refund", "track_order", "human_handoff", "ask_user", "unroutable"):
        graph.add_node(name, _terminal_node(name))
        graph.add_edge(name, END)
    graph.add_edge(START, "route")
    return graph.compile()


def _terminal_node(name: str) -> Any:
    """A placeholder handler node. Your real ones go here."""

    async def node(state: State) -> State:
        del state
        return {}

    node.__name__ = f"{name}_node"
    return node


# --------------------------------------------------------------------------- #
# Runnable demonstration.
# --------------------------------------------------------------------------- #


async def _demo() -> None:
    print(f"langgraph installed: {HAVE_LANGGRAPH}"
          f"{'' if HAVE_LANGGRAPH else '  (using the stand-in Command; route_node is real)'}\n")
    for query in (
        "where is my package",
        "I want my money back for order 4471",
        "something about my order",
        "what is the capital of France",
    ):
        command = await route_node({"input": query, "user_id": "user-42"})
        update = getattr(command, "update", None) or {}
        print(f"{query!r:<42} -> Command(goto={command.goto!r}, update={update})")

    print("\nNote 'what is the capital of France' landed on 'unroutable'. Configure")
    print("Router(fallback='human_handoff') and the same query arrives as kind='route'")
    print("with decision_path='fallback', so the `case \"route\"` arm handles it and the")
    print("graph needs no unroutable node at all (plan §6.6, §13 ruling #5).")


if __name__ == "__main__":
    asyncio.run(_demo())
