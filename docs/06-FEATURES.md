# Features
Purpose: List implemented and stubbed features with entry points, logic locations, dependencies, and test status.
Audience: Contributors planning work, reviewers checking coverage, AI agents choosing extension points.
Last verified against commit not-a-git-repository on 2026-08-07.

## Feature inventory

| Feature | Entry point | Core logic location | Dependencies | Test coverage status |
|---|---|---|---|---|
| Route catalog definition | `Route(...)`, `Registry([...])` | [`Route`](../src/switchboard/core/route.py), [`Registry`](../src/switchboard/core/registry.py) | Pydantic only | Covered by [`tests/test_route.py`](../tests/test_route.py) and [`tests/test_registry.py`](../tests/test_registry.py). |
| Sync decision routing | `Router.route(query, context=...)` | [`Router.route`](../src/switchboard/router.py), [`engine.loop`](../src/switchboard/engine/loop.py) | Provider client or callable | Covered by [`tests/test_smoke_integration.py`](../tests/test_smoke_integration.py), [`tests/test_router.py`](../tests/test_router.py). |
| Async decision routing | `await Router.aroute(...)` | [`Router.aroute`](../src/switchboard/router.py), [`run_decision_async`](../src/switchboard/engine/loop.py) | Async or sync provider client | Covered by async tests in [`tests/test_router.py`](../tests/test_router.py) and [`tests/test_smoke_integration.py`](../tests/test_smoke_integration.py). |
| Entitlement filtering | `RequestContext.entitlements`, `Route.requires`, `Route.visibility` | [`filter_routes`](../src/switchboard/engine/entitlements.py), [`filter_entitlements`](../src/switchboard/engine/loop.py) | Pydantic only | Covered by smoke and registry tests, including pre-prompt filtering. |
| Auto/BM25 shortlisting | `Router(shortlist="auto" | "bm25")` | [`AutoShortlister`](../src/switchboard/engine/shortlist.py), [`BM25Shortlister`](../src/switchboard/engine/shortlist.py) | Pydantic + stdlib | Covered by [`tests/test_shortlist.py`](../tests/test_shortlist.py) and prompt schema tests. |
| Embedding shortlisting | `Router(shortlist="embed", embed=...)` | [`EmbeddingShortlister`](../src/switchboard/engine/shortlist.py) | BYO callable; optional `[embed]` for packaged backends | Covered for BYO behavior and error cases in [`tests/test_shortlist.py`](../tests/test_shortlist.py). |
| Hybrid shortlisting | `shortlist="hybrid"` | [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) | Planned v0.2 | Stubbed: raises `ConfigError`; covered by tests. |
| Prompt segment assembly | Internal from router | [`build_segments`](../src/switchboard/engine/prompt.py) | Pydantic + stdlib | Covered by [`tests/test_prompt_schema.py`](../tests/test_prompt_schema.py). |
| Wire schema generation | Internal from router | [`build_wire_schema`](../src/switchboard/engine/schema.py) | Pydantic | Covered by [`tests/test_prompt_schema.py`](../tests/test_prompt_schema.py). |
| Validate-and-repair loop | Internal provider call stage | [`run_decision_sync`](../src/switchboard/engine/loop.py), [`validate.py`](../src/switchboard/engine/validate.py) | Provider client | Covered by [`tests/test_validate_confidence_policy.py`](../tests/test_validate_confidence_policy.py) and smoke tests. |
| Confidence scoring | `Router(confidence=...)` | [`build_confidence_report`](../src/switchboard/engine/confidence.py), [`score_confidence`](../src/switchboard/engine/loop.py) | Token logprobs if available | Covered by [`tests/test_validate_confidence_policy.py`](../tests/test_validate_confidence_policy.py). |
| Threshold policy | `Router(thresholds=...)` | [`ThresholdPolicy`](../src/switchboard/engine/policy.py), [`resolve_decision`](../src/switchboard/engine/policy.py) | Pydantic | Covered by policy tests. |
| Fallback substitution | `Router(fallback="route_name")` | [`apply_fallback`](../src/switchboard/engine/policy.py), [`apply_fallback`](../src/switchboard/engine/loop.py) | Registered, entitled route | Covered by smoke and policy tests. |
| BYO callable provider | `Router(client=callable)` | [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py) | Pydantic + stdlib | Covered by zero-dependency and router tests. |
| Instructor adapter | `Router(client="instructor:provider/model")` | [`InstructorAdapter`](../src/switchboard/providers/instructor_adapter.py) | `[instructor]` and provider SDK | Missing live-provider tests in repo; lazy import / missing extra behavior covered. |
| LiteLLM adapter | `Router(client="litellm:model")` | [`LiteLLMAdapter`](../src/switchboard/providers/litellm_adapter.py) | `[litellm]` | Missing live-provider tests in repo; lazy import behavior covered. |
| Native OpenAI adapter | `Router(client="openai:gpt-5-nano")` | [`OpenAIAdapter`](../src/switchboard/providers/openai_adapter.py), [`providers.resolve_client`](../src/switchboard/providers/__init__.py) | `[openai]` optional | Covered with fake-client contract tests; no live-provider test. |
| Native Anthropic/Gemini/Bedrock adapters | Deferred adapter specs | [`providers.resolve_client`](../src/switchboard/providers/__init__.py) | Declared extras | Stubbed: deferred adapter errors; adapter modules not present. |
| Audit record | `decision.audit` | [`AuditRecord`](../src/switchboard/core/audit.py), [`AuditDraft`](../src/switchboard/core/audit.py) | Pydantic | Covered by [`tests/test_audit.py`](../tests/test_audit.py). |
| Sink emission | `Router(sink=...)`, `Router.close()` | [`DecisionSink`](../src/switchboard/telemetry/emitter.py), sink classes | Pydantic + stdlib | Covered by audit/smoke/router tests. |
| Queued sink delivery | `QueuedSink(inner_sink, maxlen=...)` | [`QueuedSink`](../src/switchboard/telemetry/emitter.py) | Pydantic + stdlib | Covered by [`tests/test_queued_sink.py`](../tests/test_queued_sink.py). |
| OpenTelemetry spans and metrics | `Router(otel=True)`, `router.emitter` | [`OTelEmitter`](../src/switchboard/telemetry/otel.py) | `[otel]` optional | No-op behavior and metric recording are covered; SDK/exporter setup remains host-owned. |
| Eval fixtures | `EvalCase`, `EvalSuite`, load/save helpers | [`fixtures.py`](../src/switchboard/evals/fixtures.py) | Pydantic + stdlib | Covered by [`tests/test_evals.py`](../tests/test_evals.py). |
| Replay cache | `ReplayCache`, `ReplayClient` | [`cache.py`](../src/switchboard/evals/cache.py) | Pydantic + stdlib | Covered by [`tests/test_evals.py`](../tests/test_evals.py). |
| Eval runner | `run_suite`, `arun_suite` | [`harness.py`](../src/switchboard/evals/harness.py) | Pydantic + stdlib | Covered by [`tests/test_evals.py`](../tests/test_evals.py). |
| CLI | `switchboard version`, `switchboard eval dogfood`, `switchboard eval inspect` | [`cli.py`](../src/switchboard/cli.py) | Pydantic + stdlib | Covered by [`tests/test_cli.py`](../tests/test_cli.py). |
| Distillation JSONL exporter | `collect_training_examples`, `write_jsonl`, `iter_audit_jsonl` | [`switchboard.distill`](../src/switchboard/distill/__init__.py) | Pydantic + stdlib | Covered by [`tests/test_distill.py`](../tests/test_distill.py). |
| Stream route | `Router.stream_route`, `Router.astream_route` | Methods in [`Router`](../src/switchboard/router.py) | Planned v0.2 | Stubbed: raises `ConfigError`; tests cover refusal. |
| Outcome join | `Router.record_outcome`, `Router.arecord_outcome` | [`Router.record_outcome`](../src/switchboard/router.py) | Sink may implement hook | Seam implemented; no persistent store in repo. |
| Framework examples | FastAPI, LangGraph, ADK files | [`examples/`](../examples) | Optional framework packages | Examples run offline with stand-ins; referenced in examples README. |
| Support triage flagship example | `python examples/support_triage/demo.py` | [`catalog.py`](../examples/support_triage/catalog.py), [`demo.py`](../examples/support_triage/demo.py) | Pydantic + stdlib | Demonstration fixture; no dedicated test file, but examples are described as CI-run in comments/README. |

## Related documents

- [Architecture](02-ARCHITECTURE.md)
- [API contracts](07-API-CONTRACTS.md)
- [Development](09-DEVELOPMENT.md)
- [Technical debt](12-TECH-DEBT.md)
