"""The plan §3.7 quickstart, runnable with **zero extras** (plan §10.3 item 1).

::

    python examples/quickstart_byo.py

No API key. No network. No optional dependency. ``pip install switchboard`` pulls
in Pydantic and the standard library and nothing else, and this file is the proof:
it is the §3.7 quickstart character-for-character, with exactly one substitution —
``client="instructor:openai/gpt-5-nano"`` becomes ``client=keyword_stub``, a
twenty-line keyword matcher standing in for a model.

That substitution is the whole point. Plan §4.1's coercion table says a BYO client
is *any callable*: hand it a rendered prompt, hand back a ``str`` or a ``dict``,
and switchboard does the rest — schema construction, validate-and-retry, argument
typing, clarify/abstain arms, confidence, the audit record. Nothing about the
library assumes an SDK. Swap the callable for ``"instructor:openai/gpt-5-nano"``
and everything below is unchanged; that three-line upgrade is at the bottom.

The stub is not a model and does not pretend to be: it looks for a handful of
keywords in the query and answers with the wire schema switchboard generated for
this call. It exists so this file runs identically on every machine, forever.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# --------------------------------------------------------------------------- #
# The stand-in "model": a plain callable, the plan §4.1 BYO row.
#
# switchboard hands a `Callable[[str], ...]` the rendered prompt and accepts back
# a str (parsed by the repair loop), a dict (validated against the wire schema),
# or an LLMResult. This one returns a dict.
#
# Everything it needs is in the prompt switchboard built: the untrusted user
# query is delimited by <user_request> tags and placed last (plan §4.6 segment D,
# §7.4), so pulling it back out is one regex.
# --------------------------------------------------------------------------- #

_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)
_ORDER_RE = re.compile(r"\border\s+#?([a-z0-9-]*\d[a-z0-9-]*)", re.IGNORECASE)


def keyword_stub(prompt: str) -> dict[str, Any]:
    """Answer one routing call by keyword. Deterministic; stands in for a model.

    The returned dict is the **wire schema** of plan §4.4 — ``rationale`` first
    (reason before you commit), then ``kind``, then the fields that kind needs.
    switchboard validates it, re-validates ``args`` against the chosen route's
    ``args_model``, and only then hands back a typed ``Decision``.
    """
    found = _QUERY_RE.search(prompt)
    query = (found.group(1) if found else "").lower()

    if "package" in query or "track" in query or "shipped" in query:
        return {
            "rationale": "The request asks where a parcel is, which is a shipment status question.",
            "kind": "route",
            "route": "track_order",
            "args": None,  # track_order declares no args_model
        }
    if "refund" in query or "money back" in query:
        order = _ORDER_RE.search(query)
        return {
            "rationale": "The request asks for money back, which is the refund route.",
            "kind": "route",
            "route": "refund",
            # A missing order_id here would be a plan §3.8 row-4 downgrade to
            # clarify: the route was right, the argument is a question for the user.
            "args": {"order_id": order.group(1)} if order else {},
        }
    if "human" in query or "agent" in query or "person" in query:
        return {
            "rationale": "The request explicitly asks for a human.",
            "kind": "route",
            "route": "human_handoff",
            "args": None,
        }
    return {
        "rationale": "Nothing in this catalog handles the request.",
        "kind": "abstain",
        "reason": "model_elected",
    }


# --------------------------------------------------------------------------- #
# ===== plan §3.7 quickstart — verbatim, except `client=` =================== #
# --------------------------------------------------------------------------- #

# (`noqa` below suppresses import-order and line-length lint so the snippet can stay
#  byte-for-byte what plan §3.7 prints; nothing here is reformatted.)
from switchboard import Route, Registry, Router, RequestContext  # noqa: E402, I001
from pydantic import BaseModel  # noqa: E402

class RefundArgs(BaseModel):
    order_id: str
    reason: str | None = None

registry = Registry([
    Route(name="refund", description="Issue or check a refund for an order",
          args_model=RefundArgs, examples=("I want my money back for order 123",), tags=frozenset({"billing"})),  # noqa: E501
    Route(name="track_order", description="Track shipment status for an existing order"),
    Route(name="human_handoff", description="Escalate to a human support agent", pinned=True),
])
router = Router(registry=registry, client=keyword_stub,  # §3.7: "instructor:openai/gpt-5-nano"
                shortlist="auto", allow_clarify=True, fallback="human_handoff")
decision = router.route("where is my package for order 123?", context=RequestContext(tenant_id="acme"))
if decision.kind == "route":
    print(decision.route, decision.args, decision.confidence.score)

# --------------------------------------------------------------------------- #
# ===== end of the quickstart =============================================== #
# --------------------------------------------------------------------------- #


def _show(title: str, body: str) -> None:
    print(f"\n{'-' * 76}\n{title}\n{'-' * 76}\n{body}")


def main() -> None:
    """Everything past the quickstart: what those three printed values mean."""
    print(f"\n{'=' * 76}\nswitchboard quickstart — BYO callable, zero extras\n{'=' * 76}")
    print(f"\nThe three values printed above came from the §3.7 snippet verbatim:\n"
          f"  decision.route             {decision.route!r}\n"
          f"  decision.args              {decision.args!r}   (track_order declares no args_model)\n"
          f"  decision.confidence.score  {decision.confidence.score}"
          f"   method={decision.confidence.method!r}")

    _show(
        "Why confidence reads 0.0 / 'none' (plan §6.3, the no-signal rule)",
        "Confidence is computed OUTSIDE the model, from token logprobs — the evidence is that a\n"
        "model's stated confidence tracks commitment rather than correctness. A bare callable\n"
        "returns no logprobs, so there is no signal, so the score is 0.0 and the clarify/abstain\n"
        "threshold machinery is deliberately INERT rather than acting on a number it does not\n"
        "trust. A client that returns logprobs (instructor/OpenAI/Gemini) lights it up with no\n"
        "other change. Model-elected clarify and abstain always pass through regardless.",
    )

    _show(
        "The catalog went into the prompt whole (plan §5.3)",
        f"registry.version    {registry.version}   (keys the prompt cache, the index, the audit)\n"
        f"routes              {len(registry)}\n"
        f"shortlist='auto'    below 25 routes retrieval is BYPASSED — the full catalog is a\n"
        f"                    stable, cacheable prompt prefix and retrieval could only lose\n"
        f"                    recall. examples/support_triage/ shows the other side of that\n"
        f"                    threshold, at 126 routes.\n"
        f"bypass fired        {decision.audit.shortlist_skipped}",
    )

    # Every decision kind, from the same router and the same stub.
    ctx = RequestContext(tenant_id="acme")
    lines = []
    for query in (
        "I want my money back for order 998",   # route + typed args
        "please refund me",                     # route was right, order_id missing -> clarify
        "can I speak to a person",              # route
        "what is the capital of France",        # out of scope -> abstain -> fallback
    ):
        d = router.route(query, context=ctx)
        detail = {
            "route": lambda d: f"route={d.route} args={d.args!r} path={d.decision_path}",
            "clarify": lambda d: f"question={d.question!r} missing={list(d.missing)}",
            "abstain": lambda d: f"reason={d.reason}",
        }[d.kind](d)
        lines.append(f"{query!r:<40} -> kind={d.kind:<8} {detail}")
    _show(
        "One call surface, four outcomes (plan §3.4)",
        "\n".join(lines)
        + "\n\nNote the last one: `fallback='human_handoff'` turns a terminal abstain into a\n"
          "kind='route' with decision_path='fallback'. Downstream code branches on `kind`\n"
          "alone and needs no special case — the pre-fallback reason is kept in the audit.",
    )

    # Distillation eligibility is a content-mode decision, so show both sides of
    # it: the default router captures hashes, an opted-in one captures text.
    verbose = Router(registry=registry, client=keyword_stub, fallback="human_handoff",
                     content_mode="full")
    training_row = verbose.route("I want my money back for order 998", context=ctx).audit
    verbose.close()

    record = decision.audit
    _show(
        "The audit record is the product (plan §8.2)",
        f"decision_id         {record.decision_id}\n"
        f"registry_version    {record.registry_version}\n"
        f"inputs_hash         {record.inputs_hash[:24]}...   (content_mode='none': hashes, not text)\n"
        f"input_text          {record.input_text!r}   <- nothing captured, by default\n"
        f"kind / routes       {record.kind} / {record.routes}\n"
        f"candidates          total={record.candidates_total} entitled={record.candidates_entitled}\n"
        f"validation_retries  {record.validation_retries}\n"
        f"latency             total={record.latency.total_ms:.2f}ms\n"
        f"\nrecord.as_training_example() -> {record.as_training_example()}\n"
        f"Empty on purpose: hashes do not train models, so a record captured under the default\n"
        f"content_mode='none' is simply not distillation-eligible (plan §8.5 step 1). Opt in per\n"
        f"router with content_mode='redacted' (redactor applied) or 'full', and the same record\n"
        f"projects to a training row [v0.5]:\n"
        + json.dumps(training_row.as_training_example(), indent=2, default=str),
    )

    _show(
        "Three more lines for a real model (plan §10.3 item 1)",
        "    pip install 'switchboard[instructor]'\n\n"
        "    router = Router(registry=registry, client='instructor:openai/gpt-5-nano',\n"
        "                    shortlist='auto', allow_clarify=True, fallback='human_handoff')\n\n"
        "Nothing else changes: same Registry, same route() call, same Decision union, same\n"
        "audit schema. The adapter is imported lazily at Router(...) construction, so a\n"
        "missing extra fails there with MissingDependencyError rather than on your first\n"
        "production request (plan §2.4, §3.8).",
    )
    router.close()


if __name__ == "__main__":
    main()
