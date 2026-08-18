"""FastAPI endpoint (plan §3.7, §10.3 item 2).

::

    python examples/fastapi_app.py              # offline demo of the handler
    uvicorn examples.fastapi_app:app --reload   # a real server (needs fastapi)

One async handler, three lines of branching, no framework coupling. The whole
integration is that ``router.aroute()`` returns a discriminated union and the
handler branches on ``kind``:

* ``route``   -> dispatch to your handler for that route. **A configured fallback
  already arrives here** as ``kind="route"`` with ``decision_path="fallback"``,
  so there is no fourth branch to forget (plan §6.6, §13 ruling #5);
* ``clarify`` -> 200 with the question the router wrote;
* anything else -> 422 with a machine-readable ``AbstainReason``. Abstain is a
  *result*, not an exception (plan §3, decision (c)): nothing here is wrapped in
  a ``try``.

**Multi-tenancy in one line.** ``entitlements=frozenset(user.scopes)`` is the
whole integration: routes declaring ``requires`` are filtered out *before* the
LLM call, so a tenant's prompt never contains a route they may not use, and an
empty candidate set degrades to ``abstain("no_eligible_routes")`` with no
provider call at all (plan §7.1, §3.8).

**FastAPI is optional here.** Without it, tiny stand-ins for ``FastAPI``,
``Depends`` and ``JSONResponse`` keep the module importable and the ``__main__``
block still calls :func:`chat` directly. switchboard itself needs no extras.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from switchboard import Registry, RequestContext, Route, Router

try:  # pip install fastapi
    from fastapi import Depends, FastAPI
    from fastapi.responses import JSONResponse

    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover - the offline path this example ships with
    HAVE_FASTAPI = False

    @dataclass
    class JSONResponse:  # type: ignore[no-redef]
        """Stand-in for ``fastapi.responses.JSONResponse``."""

        content: dict[str, Any]
        status_code: int = 200

    class FastAPI:  # type: ignore[no-redef]
        """Stand-in exposing just enough of the decorator surface to import."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.routes: dict[str, Any] = {}

        def post(self, path: str) -> Any:
            def decorate(fn: Any) -> Any:
                self.routes[path] = fn
                return fn

            return decorate

    def Depends(dependency: Any) -> Any:  # mirrors FastAPI's capitalised spelling
        """Stand-in that resolves the dependency eagerly, at import time."""
        return dependency()


__all__ = ["HANDLERS", "ChatIn", "User", "app", "auth", "chat", "router"]


# --------------------------------------------------------------------------- #
# App-side types: your auth, your request body, your handlers.
# --------------------------------------------------------------------------- #


class ChatIn(BaseModel):
    """The request body."""

    message: str


@dataclass(frozen=True)
class User:
    """Whatever your auth dependency already returns."""

    tenant: str
    scopes: tuple[str, ...] = ()


def auth() -> User:
    """Your real dependency. Resolve slow lookups (DB/IdP) *here*, not in a
    ``Route.visibility`` predicate: predicates run over every route on every
    request and must stay O(microseconds) (plan §7.1)."""
    return User(tenant="acme", scopes=("billing",))


@dataclass
class Reply:
    """Whatever your handlers return."""

    handled_by: str
    args: dict[str, Any] = field(default_factory=dict)


async def _handle(decision: Any) -> Reply:
    args = decision.args.model_dump(exclude_none=True) if decision.args else {}
    return Reply(handled_by=decision.route, args=args)


#: Your dispatch table, keyed by route name. `Route` carries no handler on
#: purpose (plan §3, decision (a)): switchboard decides, your code executes.
HANDLERS: dict[str, Any] = {
    "refund": _handle,
    "track_order": _handle,
    "billing_report": _handle,
    "human_handoff": _handle,
}


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
    Route(name="billing_report", description="Generate a billing statement for the account",
          examples=("send me this month's invoice",), tags=frozenset({"billing"}),
          requires=frozenset({"billing"})),
    Route(name="human_handoff", description="Escalate to a human support agent", pinned=True),
])

_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)
_ORDER_RE = re.compile(r"\border\s+#?([a-z0-9-]*\d[a-z0-9-]*)", re.IGNORECASE)


async def offline_client(prompt: str) -> dict[str, Any]:
    """An async BYO callable standing in for a provider (plan §4.1)."""
    found = _QUERY_RE.search(prompt)
    query = (found.group(1) if found else "").lower()
    listed = set(re.findall(r"^- ([a-z][a-z0-9_.:-]*):", prompt, re.MULTILINE))
    if ("invoice" in query or "statement" in query) and "billing_report" in listed:
        return {"rationale": "Asks for a billing statement.", "kind": "route", "route": "billing_report"}
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


app = FastAPI(title="switchboard + FastAPI")

# One Router for the process lifetime. It holds no per-request state, resolves its
# configuration once at construction, and is safe across tasks (plan §2.5). Build
# it at import time or in a lifespan hook — never per request.
router = Router(registry=registry, client=offline_client, shortlist="auto",
                allow_clarify=True, fallback="human_handoff", otel=False)


# --------------------------------------------------------------------------- #
# ===== plan §3.7 FastAPI handler — verbatim ================================ #
# --------------------------------------------------------------------------- #


@app.post("/chat")
async def chat(body: ChatIn, user: User = Depends(auth)):  # noqa: B008 - FastAPI's own idiom
    d = await router.aroute(body.message, context=RequestContext(
        tenant_id=user.tenant, entitlements=frozenset(user.scopes)))
    if d.kind == "route":
        return await HANDLERS[d.route](d)          # fallback already arrives as kind="route"
    if d.kind == "clarify":
        return {"reply": d.question}
    return JSONResponse({"error": "cannot_route", "reason": d.reason}, status_code=422)


# --------------------------------------------------------------------------- #
# ===== end of the snippet ================================================== #
# --------------------------------------------------------------------------- #


async def _demo() -> None:
    print(f"fastapi installed: {HAVE_FASTAPI}"
          f"{'' if HAVE_FASTAPI else '  (using stand-ins; chat() and the Router are real)'}\n")

    entitled = User(tenant="acme", scopes=("billing",))
    unentitled = User(tenant="beta", scopes=())

    print("POST /chat as a tenant WITH the 'billing' entitlement")
    for message in (
        "where is my package",
        "I want my money back for order 4471",
        "send me this month's invoice",
        "something about my order",
        "what is the capital of France",
    ):
        result = await chat(ChatIn(message=message), user=entitled)
        print(f"  {message!r:<42} -> {result}")

    print("\nPOST /chat as a tenant WITHOUT it — billing_report is filtered out pre-LLM")
    gated = "send me this month's invoice"
    result = await chat(ChatIn(message=gated), user=unentitled)
    print(f"  {gated!r:<42} -> {result}")

    # The 422 branch never fired above, and that is the fallback rule working: with
    # `fallback="human_handoff"` every terminal abstain is resolved into a route.
    # Drop the fallback and the same query reaches the third branch.
    no_fallback = Router(registry=registry, client=offline_client, shortlist="auto",
                         allow_clarify=True, otel=False)
    d = await no_fallback.aroute("what is the capital of France")
    print("\nThe same query through a Router with NO fallback configured:")
    print(f"  kind={d.kind} reason={d.reason}  -> "
          f"{JSONResponse({'error': 'cannot_route', 'reason': d.reason}, status_code=422)}")
    await no_fallback.aclose()

    print("\nNothing above raised. clarify and abstain are results; the 422 carries a closed")
    print("AbstainReason your client can switch on, and the audit record explains it. The one")
    print("thing that DOES raise is broken configuration — an unknown fallback route, a bad")
    print("threshold, a missing extra — and it raises at Router(...), not on request #1.")


if __name__ == "__main__":
    asyncio.run(_demo())
