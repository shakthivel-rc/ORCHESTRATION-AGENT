# Glossary
Purpose: Define domain terms and internal jargon used across the codebase.
Audience: New engineers, operators, contributors, and AI agents reading code/docs.
Last verified against commit not-a-git-repository on 2026-08-07.

| Term | Meaning | Code pointer |
|---|---|---|
| Abstain | A decision that declines to route. It is a normal result, not an exception. | [`AbstainDecision`](../src/switchboard/core/decision.py), [`AbstainReason`](../src/switchboard/core/audit.py) |
| Args model | Pydantic model used to validate arguments extracted for a route. | [`Route.args_model`](../src/switchboard/core/route.py), [`validate_args`](../src/switchboard/engine/validate.py) |
| Audit record | Frozen canonical artifact attached to every decision and projected to telemetry/training. | [`AuditRecord`](../src/switchboard/core/audit.py) |
| BYO callable | User-supplied function used as a provider client without optional SDKs. | [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py) |
| Candidate | One route surfaced by shortlisting for the LLM to consider. | [`Candidate`](../src/switchboard/core/candidates.py) |
| Clarify | A decision asking the user for more information instead of guessing. | [`ClarifyDecision`](../src/switchboard/core/decision.py) |
| Content mode | Audit payload capture policy: `none`, `redacted`, or `full`. | [`ContentMode`](../src/switchboard/telemetry/emitter.py) |
| Decision | Discriminated union returned by routing APIs. | [`Decision`](../src/switchboard/core/decision.py) |
| Decision path | Whether a decision came from the LLM loop, a future distilled path, or fallback. | [`DecisionPath`](../src/switchboard/core/audit.py) |
| Entitlement | Caller-supplied capability string used to filter routes before LLM exposure. | [`RequestContext.entitlements`](../src/switchboard/core/context.py), [`filter_routes`](../src/switchboard/engine/entitlements.py) |
| Fallback | Registered route substituted for terminal abstain when entitled. | [`apply_fallback`](../src/switchboard/engine/policy.py) |
| Grammar rung | Highest structured-output capability rung, allowing dynamic route enums by default. | [`ClientCapabilities`](../src/switchboard/providers/base.py), [`resolve_schema_mode`](../src/switchboard/engine/schema.py) |
| Loop state | Per-request mutable accumulator used by the engine pipeline. | [`LoopState`](../src/switchboard/engine/loop.py) |
| No-signal rule | Thresholds are inert when confidence lacks actionable logprob/vote signals. | [`signals_are_actionable`](../src/switchboard/engine/confidence.py) |
| Pinned route | Route appended to shortlist candidates when entitled, regardless of score. | [`Route.pinned`](../src/switchboard/core/route.py), [`_BaseShortlister._assemble`](../src/switchboard/engine/shortlist.py) |
| Provider adapter | Object normalizing a model SDK or callable into `LLMClient`/`AsyncLLMClient`. | [`providers`](../src/switchboard/providers/__init__.py) |
| Registry | Immutable catalog of routes with content-addressed version. | [`Registry`](../src/switchboard/core/registry.py) |
| Registry version | 12-hex content-hash prefix stamped into prompts, indexes, audits, and eval fixtures. | [`Registry.version`](../src/switchboard/core/registry.py) |
| Replay cache | JSONL cache of provider calls keyed by model, prompt, schema, sampling, and sample index. | [`ReplayCache`](../src/switchboard/evals/cache.py) |
| Route | One addressable destination in the catalog. | [`Route`](../src/switchboard/core/route.py) |
| Shortlist | Candidate reduction step before the LLM decision. | [`ShortlistResult`](../src/switchboard/core/candidates.py), [`Shortlister`](../src/switchboard/engine/shortlist.py) |
| Threshold policy | Downgrade thresholds and calibration metadata. | [`ThresholdPolicy`](../src/switchboard/engine/policy.py) |
| Wire schema | Per-call Pydantic model that constrains model output; distinct from the public `Decision` union. | [`build_wire_schema`](../src/switchboard/engine/schema.py) |
| Weak retrieval | Shortlister found fewer than three positive-score candidates. | [`ShortlistResult.weak_retrieval`](../src/switchboard/core/candidates.py), [`_BaseShortlister._assemble`](../src/switchboard/engine/shortlist.py) |

## Related documents

- [Overview](01-OVERVIEW.md)
- [Data model](04-DATA-MODEL.md)
- [Business logic](05-BUSINESS-LOGIC.md)
- [API contracts](07-API-CONTRACTS.md)
