# Building a Framework-Agnostic, LLM-First Routing/Orchestration Library: Landscape, Evidence, and a Concrete Build Plan

## TL;DR
- **Build it, but position it precisely.** No existing library occupies the exact niche you describe — a *framework-agnostic, LLM-first (not embedding-first), registry-driven route/tool selector* that drops into LangGraph nodes, Google ADK agents, or plain FastAPI handlers. The closest incumbents are either embedding-first (Aurelio `semantic-router`, vLLM Semantic Router), model-cost routers (RouteLLM, LiteLLM, Portkey), or framework-coupled selectors (LlamaIndex `RouterQueryEngine`, OpenAI Agents SDK handoffs, ADK transfer). The gap is real but narrow, and it is closing fast — ship an MVP in weeks, not quarters.
- **The research is unambiguous about the hard problem you must solve.** LLM tool/route selection degrades non-linearly as the catalog grows — accuracy collapses to a **13.62% baseline** at large tool counts, per Gan & Sun, RAG-MCP (arXiv 2505.03275), which found retrieval-augmented tool discovery "more than triples tool selection accuracy (43.13% vs 13.62% baseline)." It also suffers position bias ("lost in the middle") and hallucinated tool names. The winning architecture is *retrieve-then-decide*: embedding/BM25 shortlist to ~5–20 candidates, then an LLM makes the final structured decision. This makes embedding shortlisting a first-class optimization inside your LLM-first design, not a competitor to it.
- **Differentiate on the "decision contract," not the model call.** Your moat is a typed, provider-agnostic decision object (route + args + confidence + clarify/abstain/fallback + audit record), constrained-decoding enforcement, multi-tenant entitlement filtering, and decision-log capture for distillation. Abstract providers via an optional-extras protocol layer (default to `instructor`/LiteLLM, allow BYO-callable) so you take zero hard SDK dependencies.

## Key Findings

### 1. There is a genuine but narrow gap
- **Embedding-first incumbents don't do what you want.** Aurelio `semantic-router` (MIT; 3.1k★ on GitHub, v0.1.12 released 18 Nov 2025 by jamescalam per the aurelio-labs releases page; Aurelio's own site currently displays "★ 3,745"; healthy release cadence) explicitly uses vector similarity over utterances and *does not use an LLM for routing* — it is the philosophical opposite of an LLM-first design. vLLM Semantic Router (vllm-project, ~2.6k stars, v0.1 "Iris" Jan 2026) is a ModernBERT/Envoy ExtProc gateway for model selection, not an in-process decision library.
- **Cost/model routers solve a different problem.** RouteLLM (LMSYS, ICLR 2025) routes between a strong and weak model for cost, not among tools/routes. LiteLLM, Portkey, OpenRouter, Martian are gateways optimizing provider/model choice — orthogonal to your "which of N app routes/tools" decision.
- **Framework-coupled selectors exist but are locked in.** LlamaIndex `RouterQueryEngine` with `LLMSingleSelector`/`PydanticSingleSelector` is genuinely LLM-first (function-calling to pick 1..N tools), but coupled to LlamaIndex objects. OpenAI Agents SDK handoffs and Google ADK agent-transfer/`AgentTool` are LLM-first but bound to their runtimes and (in practice) their preferred models. LangGraph routing is just `.with_structured_output()` inside a node — a pattern you re-write every time, which is exactly the boilerplate you want to eliminate.
- **The nearest structural analog is ToolRegistry** (Peng Ding, Univ. of Chicago; arXiv 2507.10593; MIT) — a protocol-agnostic tool-management library with multi-provider schema generation (OpenAI/Anthropic/Gemini), tag-based permissions, and BM25F progressive disclosure. It manages/executes tools but is not primarily a *decision agent* with confidence/clarify/abstain semantics. It's both a competitor and a potential dependency/interop target. LLMRouter (ulab-uiuc, released Dec 2025) is academic and query→model oriented.

### 2. Research on LLM tool/route selection (the evidence base)
- **Non-linear accuracy collapse with scale.** Vendor-documented thresholds are concrete: Anthropic's tool-search documentation states Claude's ability to pick the right tool degrades once you exceed roughly 30–50 available tools, and OpenAI's function-calling guide advises aiming for fewer than 20 functions available at the start of a turn (per the Nerd Level Tech synthesis, "How Many Tools Can an AI Agent Handle? (2026 Data)"). Reported degradation curves: with ~50 tools most models hold 84–95%; ~200 tools drops to 41–83%; ~740 tools collapses to 0–20% (MachineLearningMastery, vLLM SR, webscraft — 2025–2026 syntheses of primary papers).
- **There is a hard ceiling of 128 tools per request in the OpenAI API**, confirmed by the API error string "Invalid 'tools': array too long. Expected an array with maximum length 128" (GitHub issue oh-my-opencode #2848) and the Assistants docs ("give the Assistant access to up to 128 tools"). Real degradation appears far below this limit.
- **RAG-MCP is the load-bearing primary result.** Gan & Sun (Beijing Univ. of Posts and Telecommunications; Queen Mary Univ. of London), "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation," arXiv 2505.03275 (submitted 6 May 2025; CC BY-NC-ND 4.0). Abstract, verbatim: retrieval-augmented tool discovery "significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks." Their MCP stress test varied N from 1 to 11,100 tools in 26 intervals; the pipeline encodes the query (with Qwen-max), retrieves the most relevant MCP, and injects only that description into the main LLM's prompt. This is the empirical foundation for retrieve-then-decide.
- **Position bias is architectural.** BiasBusters (arXiv 2510.00307) and the tool-learning robustness study (arXiv 2407.03007) show LLMs prefer tools by position/superficial metadata; a shuffled toolset dropped DeepSeek success from 41% to 27%. With 741 tools, middle positions show materially lower selection accuracy than head/tail — a RoPE "lost in the middle" effect. Shortlisting to a handful mitigates this automatically.
- **The "99% Success Paradox."** arXiv 2605.18857 shows near-perfect retrieval (Success@100 ≈ 100%) can still yield near-random LLM selection: at K=100, LLM accuracy drops 10–16 points while token cost rises 10×. Implication: retrieve *tightly* (small K), don't dump 100 candidates on the model.
- **Enterprise-scale routing degradation, quantified.** Gillespie & Perry (Superhuman/Grammarly), "Scaling Enterprise Agent Routing," arXiv 2606.17519: on a 110-agent/584-tool production catalog, routing F1 drops 16–23 points scaling 10→110 agents; embedding-based shortlisting recovers +10–11 F1 across models/providers. Decomposes error into a "retrieval gap" and a "confusion gap" (10pp oracle ceiling drop even with perfect retrieval).
- **Benchmarks to evaluate against:** BFCL V4 (Berkeley/Gorilla; AST-based, scales to thousands of functions; native FC vs prompt modes; publishes cost+latency); τ-bench / τ²/τ³-bench (Sierra; tool-agent-user, pass^k reliability — GPT-4o <50% success, pass^8 <25% retail); MetaTool (199 tools, tool-usage awareness); ToolBench/ToolLLM (2,413 tools); ToolACE, NESTFUL, API-Bank, Gorilla, ToolSandbox (distractor tools), MTU-Bench. Tool-DE (arXiv Oct 2025) shows LLM-enriched tool descriptions add +6–7 pts NDCG@10/Recall@10.
- **Structured output vs free-text classification.** Constrained decoding (OpenAI Structured Outputs, Outlines, XGrammar, llguidance) gives near-100% schema fidelity vs best-effort function-calling. But there's a quality caveat: BAML reports gpt-4o on BFCL scored 93.63% (native FC) vs 91.37% (strict constrained decoding); the "Let Me Speak Freely?" line of work shows over-constraining can hurt reasoning. Best practice: allow an in-schema free-text `reasoning`/`rationale` field before the committed `route` field, or a two-pass reason-then-constrain pattern for hard decisions.
- **Confidence & abstention — treat with skepticism.** Verbalized confidence is stable under paraphrase and roughly calibrated, but multiple 2026 papers (arXiv 2601.07767 RiskEval; arXiv 2606.29490) find models do not convert stated confidence into correct abstain decisions and are insensitive to error penalties — "reported confidence tracks commitment more than correctness." Logprob-based signals discriminate correctness better than verbalized confidence. Self-consistency (2+ samples) improves reliability but costs extra calls. Practical stance: expose confidence as a *signal*, default abstention/clarification thresholds conservatively, and prefer logprobs + a small self-consistency vote for high-stakes routes rather than trusting a verbalized 0–1 score.

### 3. Design guidance for an LLM-first router
- **Core loop = optional shortlist → structured LLM decision → validate → fallback.** Keep the LLM as the decider; use embeddings/BM25 only to cut candidates to a small K (research says small K, ~5–20).
- **Prompt structure:** stable system + full registry as a *prefix* (for prompt caching), tenant-filtered candidate list, few-shot exemplars per route (descriptions-as-prompts, not docs), user query last. Randomize/curate order or shortlist to blunt position bias.
- **Enforce output via constrained decoding** where the provider supports it; fall back to instructor-style validate-and-retry elsewhere. Use a discriminated union (tagged `kind`) to model outcomes: `route` | `multi_route` | `clarify` | `abstain`/`fallback`.
- **First-class outcomes:** single route, parallel multi-route, sequential plan, argument/parameter extraction alongside the route, ask-user clarification, and a default/fallback route. Clarification and abstention must be selectable outputs, not exceptions.
- **Model choice & economics:** use a *small, non-reasoning* model for routing. Cheapest options with low TTFT: Gemini 2.5 Flash-Lite (confirmed exact — **$0.10/M input, $0.40/M output** per Google AI for Developers pricing / OpenRouter; ~0.33s TTFT via AI Studio) and GPT-5 nano (~$0.05/M input, ~$0.40/M output). Claude Haiku 4.5 ($1/$5, per Anthropic) and Gemini Flash both deliver sub-600ms TTFT. **Avoid reasoning modes for routing** — they inflate TTFT to many seconds (GPT-5-mini medium-reasoning TTFT ~13s vs Flash-Lite non-reasoning ~0.33s). Note: Google has set Gemini 2.5 Flash-Lite retirement for 16 Oct 2026, after which the cheapest tier is Gemini 3.1 Flash-Lite at $0.25/$1.50 (Curlscape, verified 12 Jul 2026) — so keep the model pluggable.
- **Cost reduction levers:** (1) prompt caching — a static registry prefix qualifies for ~90% cached-input discounts on OpenAI, Anthropic, and Gemini (cache reads ≈10% of base input rate; Anthropic official docs: "a cache hit costs 10% of the standard input price"; Haiku 4.5 page: "up to 90% cost savings with prompt caching"); (2) embedding shortlist to shrink the prompt; (3) distill decision logs into a fine-tuned small classifier for hot paths; (4) batch API (50% off) for offline eval. RAG-MCP's >50% token cut plus caching stacks multiplicatively.

### 4. Library design & packaging for adoption
- **Provider abstraction — take zero hard SDK deps.** Define a narrow `LLMClient` Protocol (or ABC) with one method returning a validated structured object; ship thin adapters behind optional extras (`pip install yourlib[openai]`, `[litellm]`, `[instructor]`, `[bedrock]`, `[vertex]`, `[databricks]`). Default recommended path: `instructor` (per its README/PyPI: "3M+ monthly downloads · 10K+ GitHub stars · 1000+ community contributors," supporting "15+ providers including OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Ollama, DeepSeek") and/or LiteLLM for the broadest provider matrix; allow a plain `Callable`/BYO-client so power users bypass everything. Note the CVE-2026-42208 supply-chain incident on semantic-router's unbounded LiteLLM pin (which could resolve to a compromised wheel that exfiltrates credentials) — pin transitive deps with ranges and avoid mandatory heavy deps.
- **API:** sync + async parity, Pydantic v2 models, full type hints, `Route`/`Registry`/`Router` primitives, streaming-aware (stream the rationale, commit the route atomically).
- **Observability:** emit OpenTelemetry GenAI semantic-convention spans (`gen_ai.*`, `execute_tool`, `gen_ai.tool.name`) — the CNCF standard now supported by Datadog, MLflow, Google ADK, Langfuse, Arize/Phoenix. Don't invent private attributes.
- **Packaging (2026 golden path):** `pyproject.toml` + hatchling (or uv), `src/` layout, `uv build`/`uv publish` with PyPI Trusted Publishing (OIDC, no tokens), dynamic versioning from git tags, SemVer, flexible dependency ranges for a library, ruff + mypy, excellent docs + copy-paste quickstart. Adoption of small AI-infra libs is driven by: a 10-line quickstart, framework-agnostic proof (LangGraph + ADK + FastAPI examples), honest benchmarks on the user's own catalog, and strong docs.

### 5. Multi-tenant / enterprise concerns
- **Entitlement filtering is a first-class, pre-LLM step:** filter the registry to the tenant's/user's visible+authorized routes before shortlisting — this both enforces security and shrinks the prompt (accuracy win). Model it as a `visibility`/`entitlement` predicate on each `Route`.
- **Registry versioning & cache invalidation:** content-hash the registry; version it; key prompt-cache and embedding-index entries by (registry_version, tenant). Invalidate embeddings and cached prefixes on registry change.
- **Decision audit logging:** persist the full decision contract (inputs hash, candidates, chosen route, args, confidence, model, latency, tokens, cost, tenant, registry_version) — this is both your compliance audit trail and your distillation training set.
- **PII:** support redaction hooks on the routing prompt; prefer routing on intent/metadata over raw payload where possible; keep OTel content capture opt-in.

### 6. Concrete recommendation
- **Verdict: build it.** The precise gap — framework-agnostic + LLM-first + registry-driven + typed decision contract + multi-tenant + distillation-ready — is unoccupied. But the window is narrowing (Anthropic Tool Search, ADK/OpenAI handoffs, ToolRegistry, semantic-router all encroaching), so speed and sharp positioning matter.
- **Positioning:** "The bring-your-own-model decision layer: one typed `route()` call that picks among your tools/routes, extracts args, asks for clarification, or abstains — in any framework, with any provider, with an audit trail you can distill." Explicitly *not* a gateway, *not* an agent framework, *not* embedding-only.

## Details

### Competitive landscape comparison

| Project | LLM-first? | Framework-coupled? | Scope | License | Maturity |
|---|---|---|---|---|---|
| Aurelio `semantic-router` | No (embedding-first) | No | intent/route select | MIT | 3.1k★ (site shows 3,745), v0.1.12 (18 Nov 2025), healthy |
| vLLM Semantic Router | No (BERT classifier) | Gateway (Envoy) | model/reasoning routing | Apache-2.0 | ~2.6k★, v0.1 Jan 2026 |
| RouteLLM (LMSYS) | Classifier | No | strong/weak model cost | Apache-2.0 | ICLR 2025, research-grade |
| LiteLLM / Portkey / OpenRouter | No | Gateway | provider/model routing | MIT / Apache-2.0 | mature, widely adopted |
| LlamaIndex RouterQueryEngine | Yes | Yes (LlamaIndex) | select 1..N query engines/tools | MIT | mature |
| OpenAI Agents SDK handoffs | Yes | Yes (OpenAI SDK) | agent handoff (tool-call) | MIT | GA 2025 |
| Google ADK transfer / AgentTool | Yes | Yes (ADK) | sub-agent transfer | Apache-2.0 | GA 2025 |
| Anthropic Tool Search (MCP) | Hybrid (BM25/regex + LLM) | Claude platform | on-demand tool discovery | proprietary | shipped Nov 2025–Jan 2026 |
| ToolRegistry | Provider-agnostic exec | No | tool registration/exec + disclosure | MIT | arXiv 2507.10593, active |
| **Your proposed library** | **Yes (LLM decides)** | **No (agnostic)** | **route/tool decision contract** | **(MIT)** | **greenfield** |

- Anthropic Tool Search caveat: independent tests are unflattering — Arcade reported 56% (regex)/64% (BM25) retrieval accuracy over 4,027 tools; Stacklok reported 34% for Tool Search vs 94% for its own optimizer over 2,792 tools. Token savings are real (~85%) but retrieval accuracy is the bottleneck — reinforcing that a good *decision* layer on top of retrieval is where value accrues.

### Recommended core API sketch
```python
from yourlib import Route, Registry, Router
from pydantic import BaseModel

class RefundArgs(BaseModel):
    order_id: str
    reason: str | None = None

registry = Registry([
    Route(name="refund", description="Issue or check a refund for an order",
          args_model=RefundArgs, examples=["I want my money back for order 123"],
          tags={"billing"}, visibility=lambda ctx: ctx.tenant.has("billing")),
    Route(name="track_order", description="Track shipment status", ...),
    # ... hundreds more
])

router = Router(
    registry=registry,
    client="instructor:openai/gpt-5-nano",   # or a BYO Callable / LiteLLM / Bedrock
    shortlist="embed:top_k=12",               # optional optimization, LLM still decides
    multi_route=True, allow_clarify=True, fallback="human_handoff",
    confidence="logprobs+self_consistency:n=3",
)

decision = router.route("where is my package for order 123?",
                        context=RequestContext(tenant=t, user=u))
# decision.kind in {"route","multi_route","clarify","abstain"}
# decision.route, decision.args (validated), decision.confidence, decision.audit
```
- Async: `await router.aroute(...)`. Streaming: `router.stream_route(...)` yields rationale tokens then a final committed decision.
- The `audit` object is the OTel span payload and the distillation record.

### Phased roadmap
- **MVP (v0.1):** single + multi route; Pydantic v2 decision contract; constrained-decoding via instructor/LiteLLM; one provider path proven + BYO-callable; clarify/abstain/fallback; embedding shortlist optional; OTel spans; LangGraph + FastAPI + ADK examples.
- **v0.2:** entitlement/visibility predicates; registry versioning + prompt-cache/embedding-index keying; confidence signals (logprobs, self-consistency); eval harness against BFCL/MetaTool-style fixtures + user's own catalog.
- **v0.5:** decision-log capture + distillation pipeline to a fine-tuned small classifier for hot routes; sequential plans; hierarchical routing for very large catalogs.
- **v1.0:** stable API + SemVer guarantees; provider matrix (OpenAI, Anthropic, Gemini/Vertex, Bedrock, Databricks Model Serving, vLLM); benchmark report; hardened multi-tenant audit.

### Biggest risks (what would make it fail)
1. **Commoditization by frameworks/providers** — ADK/OpenAI handoffs, Anthropic Tool Search, and semantic-router keep absorbing this. Mitigation: be the *neutral* cross-framework layer they can't be, and own the decision-contract + distillation story.
2. **"Just write `.with_structured_output()`"** — the boilerplate you replace is small, so value must come from the hard parts (shortlisting, confidence, multi-tenant, audit, distillation, eval). If the API isn't dramatically simpler, adoption stalls.
3. **Provider abstraction leakage** — constrained decoding, logprobs, and caching differ per provider; a lowest-common-denominator API loses the features that matter. Mitigation: capability-detection + graceful degradation, not a flat abstraction.
4. **Accuracy credibility** — if you can't show wins on the user's own catalog (not just cherry-picked benchmarks), architects won't trust it. Ship the eval harness as a headline feature.
5. **Supply-chain/dependency risk** — heavy mandatory deps (the semantic-router/LiteLLM CVE-2026-42208 lesson). Keep the core near-zero-dependency (Pydantic only), everything else optional extras with pinned ranges.

## Recommendations
1. **Commit to build, scope tightly to the decision contract.** Ship an MVP (v0.1) in weeks with the API above; do not build a gateway or agent framework.
2. **Provider strategy:** default to `instructor` + LiteLLM behind a one-method `LLMClient` Protocol; support BYO-callable; put every SDK behind optional extras; keep the core dependency footprint to Pydantic v2 only.
3. **Make retrieve-then-decide the default, LLM-first the principle.** Embedding/BM25 shortlist to K≈5–20 (never dump 100 tools), then constrained-decoding LLM decision with an in-schema rationale field.
4. **Treat confidence honestly.** Prefer logprobs + optional 2–3-sample self-consistency over verbalized scores; default clarify/abstain thresholds conservatively; document the calibration caveats.
5. **Lead with observability + eval + multi-tenant.** OTel GenAI spans, an eval harness runnable on the user's own catalog, and entitlement filtering are the enterprise wedge that framework-coupled tools lack.
6. **Capture decision logs from day one** for audit and distillation — this is the compounding asset and the long-term differentiator.

**Benchmarks/thresholds that would change the plan:**
- If Anthropic Tool Search (or an OSS clone) crosses ~85–90% retrieval accuracy at 1k+ tools *and* goes cross-provider, the retrieval half of your value erodes — pivot harder to the decision-contract/distillation/multi-tenant layer.
- If your own eval can't beat naive `.with_structured_output()` by a meaningful margin on a 100+ route catalog, don't ship — the boilerplate-elimination value alone won't sustain adoption.
- If prompt-caching + small-model routing doesn't get per-decision latency under your app's budget (target sub-second TTFT), add the distilled-classifier fast path earlier (pull v0.5 forward).

## Caveats
- Several quantitative degradation figures (e.g., "84–95% at 50 tools," "13% at 100+") come from 2025–2026 secondary syntheses (MachineLearningMastery, vLLM SR, webscraft) that aggregate primary papers; the RAG-MCP (arXiv 2505.03275), BiasBusters (2510.00307), and enterprise-routing (2606.17519) primary sources are solid, but exact percentages vary by model and setup — re-prove on your catalog.
- Pricing and latency figures are 2026 snapshots; small-model list prices move (GPT-5-mini's input price reportedly doubled over ~90 days), Gemini 2.5 Flash-Lite retires 16 Oct 2026, and some newer model tiers cited by aggregators are unverified. Verify against official pricing pages before relying on them.
- Constrained-decoding-hurts-quality (BAML, "Let Me Speak Freely?") and confidence-unreliability findings are contested/nuanced; the safe engineering choice (in-schema rationale, logprobs, conservative abstention) holds regardless.
- GitHub stars/downloads are momentum proxies, not quality; the landscape is shifting monthly (Anthropic Tool Search Jan 2026, vLLM SR v0.1 Jan 2026), so re-scan incumbents right before launch.
- This report did not independently verify every provider's current constrained-decoding/logprob support matrix — validate per-provider capabilities during v0.1.