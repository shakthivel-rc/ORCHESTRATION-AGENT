# switchboard

**The bring-your-own-model decision layer:** one typed `route()` call that picks among your
tools/routes, extracts args, asks for clarification, or abstains — in any framework, with any
provider, with an audit trail you can distill.

`switchboard` is a framework-agnostic, LLM-first, registry-driven route/tool selector. It is
**not** a gateway, **not** an agent framework, and **not** embedding-only. It **decides, never
executes** — execution authority stays with your code.

For a code-derived engineering guide, see the documentation set at
[docs/README.md](docs/README.md).

The moat is the decision contract, not the model call:

- A typed, provider-agnostic `Decision` discriminated union — `route | multi_route | clarify |
  abstain` — with validated args, a confidence report, and a full audit record on **every** outcome.
- Constrained-decoding enforcement with a graceful-degradation ladder (`grammar` → `tool_strict` →
  `json_mode` → `none`), never a flat lowest-common-denominator abstraction.
- Multi-tenant entitlement filtering as a pre-LLM security boundary that also shrinks the prompt.
- Decision-log capture from day one: the audit record **is** the OTel span payload **is** the
  distillation training example.
- An eval harness you run on **your own** catalog, with go/no-go gates encoded as CI thresholds.

**Status: v0.1 in development.** The API below is the target contract; see
[ORCHESTRATION_AGENT_PLAN.md](ORCHESTRATION_AGENT_PLAN.md) for the authoritative specification,
phase tags (`[v0.1]` / `[v0.2]` / `[v0.5]` / `[v1.0]`), and the evidence base behind every default.

## Quickstart

```python
from switchboard import Route, Registry, Router, RequestContext
from pydantic import BaseModel

class RefundArgs(BaseModel):
    order_id: str
    reason: str | None = None

registry = Registry([
    Route(name="refund", description="Issue or check a refund for an order",
          args_model=RefundArgs, examples=("I want my money back for order 123",), tags=frozenset({"billing"})),
    Route(name="track_order", description="Track shipment status for an existing order"),
    Route(name="human_handoff", description="Escalate to a human support agent", pinned=True),
])
router = Router(registry=registry, client="instructor:openai/gpt-5-nano",
                shortlist="auto", allow_clarify=True, fallback="human_handoff")
decision = router.route("where is my package for order 123?", context=RequestContext(tenant_id="acme"))
if decision.kind == "route":
    print(decision.route, decision.args, decision.confidence.score)
```

Every public entry point has an `a`-prefixed async twin: `router.aroute(...)`.

## Install

```bash
pip install switchboard                 # core: Pydantic only
pip install "switchboard[instructor]"   # recommended default provider path
```

The core has exactly one dependency — `pydantic>=2.7,<3`. A bring-your-own plain callable works
with **zero extras**; every extra is an optimization, never a requirement. BM25 shortlisting is
core (pure Python). Every extra is upper-bounded on purpose (an unbounded transitive pin is how
CVE-2026-42208 shipped a credential-exfiltrating wheel), and SDK imports are lazy, so a compromised
optional dependency cannot execute for users who never installed it.

| Extra | Deps (pinned ranges) | Unlocks | Phase |
|---|---|---|---|
| *(none)* | `pydantic>=2.7,<3` | full core: primitives, `Decision` union, BYO client, BM25 auto-shortlist, BYO-embed shortlist, validate-and-retry, audit records, eval fixtures/replay | v0.1 |
| `instructor` | `instructor>=1.7,<2` | recommended default: structured output across 15+ providers | v0.1 |
| `litellm` | `litellm>=1.61,<2` | broadest provider matrix; logprobs where upstream supports | v0.1 |
| `otel` | `opentelemetry-api>=1.25,<2` | `gen_ai.*` span emission (API-only dep; SDK stays user-side) | v0.1 |
| `openai` | `openai>=1.60,<3` | native strict Structured Outputs + logprobs + `prompt_cache_key` | v0.2 |
| `anthropic` | `anthropic>=0.40,<1` | forced-tool structured output + explicit `cache_control` | v0.2 |
| `gemini` | `google-genai>=1.0,<2` | `responseSchema` + `propertyOrdering`; cheapest routing tier | v0.2 |
| `embed` | `fastembed>=0.3,<1`, `numpy>=1.24,<3` | packaged embedding backends (BYO embed-callable needs nothing) | v0.2 |
| `eval` | `pyyaml>=6,<7`, `rich>=13,<15` | eval CLI, `gates.yaml`, report rendering | v0.2 |
| `bedrock` | `boto3>=1.34,<2` | Bedrock Converse | v0.5 |
| `distill` | `pyarrow>=15,<22` | decision-log → training-set exporter | v0.5 |
| `distill-train` | `sentence-transformers`, `scikit-learn` | the default classifier trainer | v0.5 |
| `all` | union of the above | demos / CI kitchen sink | — |

## Defaults worth knowing

Small **non-reasoning** models only — reasoning modes inflate TTFT from ~0.33s to seconds, so
adapters force reasoning off and the Router warns if a reasoning-default model is configured.
Model IDs, prices and `deprecated_after` dates are **data**, not code: see
`src/switchboard/_models.py`. The default is `gemini-2.5-flash-lite` (retires 2026-10-16 →
`gemini-3.1-flash-lite`); alternates are `gpt-5-nano` and `claude-haiku-4-5`. Prices there are a
dated snapshot; an unknown model yields `cost=None`, never a guess.

## Development

```bash
python -m pip install -e ".[all]"
ruff check src tests
mypy
pytest
```

## License

MIT.
# ORCHESTRATION-AGENT
