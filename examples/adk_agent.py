"""Google ADK agent — the router as a plain tool (plan §3.7, §10.3 item 2).

::

    python examples/adk_agent.py

No coupling in either direction: :func:`pick_route` is an ordinary function that
takes a query and returns a dict, which is exactly what ADK (and OpenAI Agents,
and every other tool-calling framework) already knows how to call. switchboard
does not import ADK; ADK does not import switchboard.

**Why route inside a tool at all?** Because delegation is where multi-agent
systems misfire: the outer agent has the conversation, and asking it to *also*
pick correctly among 100+ destinations is the exact prompt-dilution failure a
dedicated decision layer exists to fix (plan §1). ``pick_route`` gives the agent
one small, audited, typed decision instead.

Two details in the snippet earn their place:

* ``exclude={"audit"}`` — the audit record belongs in your telemetry pipeline,
  not in an agent's context window. It carries hashes, latency, token usage and
  the shortlist; feeding it back to the model wastes tokens and leaks internals.
  The library ships :meth:`Decision.model_dump_public` for exactly this shape.
* ``mode="json"`` — ``args`` is an instance of the chosen route's ``args_model``,
  an arbitrary Pydantic model. switchboard annotates it with ``SerializeAsAny`` so
  it serialises with its *runtime* type; without that this dump would silently
  emit ``{}``.

**ADK is optional here.** ``pip install google-adk`` and the real ``Agent`` /
``ToolContext`` are used; without it, small stand-ins keep the module importable
and the ``__main__`` block still calls :func:`pick_route` against a real Router.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from switchboard import Registry, RequestContext, Route, Router

try:  # pip install google-adk
    from google.adk.agents import Agent
    from google.adk.tools import ToolContext

    HAVE_ADK = True
except ImportError:  # pragma: no cover - the offline path this example ships with
    HAVE_ADK = False

    @dataclass
    class ToolContext:  # type: ignore[no-redef]
        """Stand-in for ``google.adk.tools.ToolContext``."""

        state: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Agent:  # type: ignore[no-redef]
        """Stand-in for ``google.adk.agents.Agent``."""

        model: str = ""
        name: str = ""
        instruction: str = ""
        tools: tuple[Any, ...] = ()


__all__ = ["adk_agent", "pick_route", "router"]


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


def offline_client(prompt: str) -> dict[str, Any]:
    """A synchronous BYO callable standing in for a provider (plan §4.1).

    Synchronous on purpose: :func:`pick_route` is a *sync* tool, so it calls
    ``router.route()``. Pairing a sync driver with an async-only client is a
    configuration error raised at ``Router(...)``, not a surprise at request time
    (plan §2.5).
    """
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
            "rationale": "About an order, but the action is unclear.",
            "kind": "clarify",
            "question": "Do you want a refund, or to track the shipment?",
            "candidates": ["refund", "track_order"],
        }
    return {"rationale": "Out of scope.", "kind": "abstain", "reason": "model_elected"}


router = Router(registry=registry, client=offline_client, shortlist="auto",
                allow_clarify=True, fallback="human_handoff", otel=False)


# --------------------------------------------------------------------------- #
# ===== plan §3.7 Google ADK snippet — verbatim ============================= #
# --------------------------------------------------------------------------- #


def pick_route(query: str, tool_context: ToolContext) -> dict:
    d = router.route(query, context=RequestContext(user_id=tool_context.state.get("user_id")))
    return d.model_dump(mode="json", exclude={"audit"})


adk_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="support_triage",
    instruction=(
        "You are a support assistant. Call pick_route with the customer's message to decide "
        "what to do. If it returns kind='clarify', ask the user its question verbatim. If it "
        "returns kind='route', hand off to that route. Never invent a route name yourself."
    ),
    tools=[pick_route],
)


# --------------------------------------------------------------------------- #
# ===== end of the snippet ================================================== #
# --------------------------------------------------------------------------- #


def _demo() -> None:
    print(f"google-adk installed: {HAVE_ADK}"
          f"{'' if HAVE_ADK else '  (using stand-ins; pick_route and the Router are real)'}\n")
    tool_context = ToolContext(state={"user_id": "user-42"})

    for query in (
        "I want my money back for order 4471",
        "something about my order",
        "what is the capital of France",
    ):
        payload = pick_route(query, tool_context)
        print(f"query   {query!r}")
        print(f"tool -> {json.dumps(payload, indent=2, sort_keys=False)}\n")

    print("Everything the agent needs is in that payload and nothing else is: `kind` to branch")
    print("on, `route`/`args` or `question`, `confidence`, and `decision_path` (note the last")
    print("query: 'fallback', so the agent sees a route, not an error). The audit record went")
    print("to Router(sink=...) instead — d.model_dump_public() is the same projection, named.")


if __name__ == "__main__":
    _demo()
