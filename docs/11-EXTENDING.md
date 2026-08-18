# Extending
Purpose: Explain extension points, recipes, conventions, and fragile areas for safe changes.
Audience: Contributors adding providers, shortlisters, routes, integrations, or telemetry sinks.
Last verified against commit not-a-git-repository on 2026-08-07.

## Designed extension points

| Extension point | How to plug in | Evidence |
|---|---|---|
| New route | Add `Route` to a `Registry`; optionally supply `args_model`, `requires`, `visibility`, `metadata`. | [`Route`](../src/switchboard/core/route.py), [`Registry`](../src/switchboard/core/registry.py) |
| New provider client | Pass object implementing `complete()` and/or `acomplete()`, or add adapter behind `resolve_client`. | [`LLMClient`](../src/switchboard/providers/base.py), [`resolve_client`](../src/switchboard/providers/__init__.py) |
| BYO provider callable | Pass a sync/async callable returning `str`, `dict`, Pydantic model, or `LLMResult`. | [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py) |
| New shortlister | Implement `Shortlister` protocol or subclass internal `_BaseShortlister` with care. | [`Shortlister`](../src/switchboard/engine/shortlist.py) |
| Embedding backend | Add backend resolution in `shortlisters.embedding_backends`. | [`get_backend`](../src/switchboard/shortlisters/embedding_backends.py), [`_load_backend`](../src/switchboard/engine/shortlist.py) |
| New telemetry sink | Implement `DecisionSink` or subclass `BaseSink`; pass to `Router(sink=...)`. | [`DecisionSink`](../src/switchboard/telemetry/emitter.py), [`BaseSink`](../src/switchboard/telemetry/emitter.py) |
| Eval fixtures | Create `EvalCase` / `EvalSuite` and run through `run_suite`. | [`EvalCase`](../src/switchboard/evals/fixtures.py), [`run_suite`](../src/switchboard/evals/harness.py) |
| Framework integration | Call `route()` / `aroute()` from application/framework code and branch on `decision.kind`. | [`examples/fastapi_app.py`](../examples/fastapi_app.py), [`examples/langgraph_node.py`](../examples/langgraph_node.py), [`examples/adk_agent.py`](../examples/adk_agent.py) |

## Recipe: add a new endpoint in a host app

1. Build or reuse a process-lifetime `Router`.
2. Resolve auth/scopes in the host framework.
3. Create `RequestContext(tenant_id=..., user_id=..., entitlements=...)`.
4. Call `await router.aroute(body.message, context=context)` for async frameworks.
5. Branch:
   - `route`: dispatch through the host app's handler table.
   - `clarify`: return the question as a normal user-facing response.
   - `abstain`: return a machine-readable failure or let fallback resolve it.
6. Keep execution outside Switchboard; route metadata may hold handler references if the app wants.

Evidence: [`chat`](../examples/fastapi_app.py) and [`HANDLERS`](../examples/fastapi_app.py).

## Recipe: add a new module/agent route

1. Choose a stable lowercase route name that matches [`ROUTE_NAME_PATTERN`](../src/switchboard/core/route.py).
2. Write a prompt-facing description: what it does, when to use it, and when not to use it.
3. Add examples shaped like real user utterances.
4. Define a Pydantic args model for required extraction.
5. Gate the route with `requires` or `visibility` if tenant-sensitive.
6. Consider `pinned=True` only for fallback/escalation routes.
7. Add tests for route construction and an integration test with a BYO stub.

## Recipe: add a new provider adapter

1. Put SDK imports inside functions or adapter constructors, not module scope.
2. Implement `complete()` and `acomplete()` where possible.
3. Produce `LLMResult` with `raw_text` always set.
4. Set `ClientCapabilities` conservatively.
5. Translate transport errors into `ProviderTimeout`, `ProviderRateLimit`, `ProviderAuthError`, or `ProviderError`.
6. Treat schema/model-output validation failures as `LLMResult(parsed=None, ...)`, not provider errors.
7. Register the adapter name in [`providers.__init__`](../src/switchboard/providers/__init__.py).
8. Add lazy-import tests in [`tests/test_zero_deps.py`](../tests/test_zero_deps.py).

## Recipe: add a new integration framework

1. Do not add framework imports to `switchboard` core.
2. Create an example or external integration that imports the framework optionally.
3. Keep the integration as a thin branch on `Decision.kind`.
4. Use `Decision.model_dump_public()` or `model_dump(..., exclude={"audit"})` when returning data to an agent context.
5. Leave execution and side effects in the host framework.

Evidence: optional import stand-ins in [`examples/langgraph_node.py`](../examples/langgraph_node.py), [`examples/fastapi_app.py`](../examples/fastapi_app.py), and [`examples/adk_agent.py`](../examples/adk_agent.py).

## Recipe: add a new tenant/customer

1. Model customer-specific capabilities as entitlement strings.
2. Set `Route.requires` for routes that need those capabilities.
3. Resolve the user's scopes outside Switchboard.
4. Pass `RequestContext(tenant_id=..., entitlements=frozenset(scopes))`.
5. Avoid slow network/database work inside `Route.visibility`.

Evidence: [`RequestContext`](../src/switchboard/core/context.py), [`filter_routes`](../src/switchboard/engine/entitlements.py), and [`examples/fastapi_app.py`](../examples/fastapi_app.py).

## Conventions to follow

- Keep core, engine, router, provider protocol, and telemetry emitter importable with Pydantic + stdlib only.
- Prefer degradation to `Decision` outcomes for model uncertainty; raise only for broken configuration or infrastructure.
- Maintain sync/async parity by changing loop stages, not duplicating logic.
- Keep prompt trust order: stable trusted config first, untrusted query last.
- Do not put handler execution into `Route` or `Router`.
- Preserve audit finalization and sink emission in `finally`.
- Add tests near the existing focused test module for the surface changed.

## Do not touch without care

| Area | Why fragile | Evidence |
|---|---|---|
| `Route.content_hash` / `Registry.version` | Drives prompt cache, shortlist index, audit records, eval fixtures. | [`Route.content_hash`](../src/switchboard/core/route.py), [`Registry.version`](../src/switchboard/core/registry.py) |
| Prompt segment ordering and cache keys | Affects provider cache economics and prompt-injection boundary. | [`build_segments`](../src/switchboard/engine/prompt.py) |
| Wire schema field order | `rationale` before route commitment is a deliberate behavior. | [`build_wire_schema`](../src/switchboard/engine/schema.py) |
| Validation degradation table | Consumer-visible abstain/clarify reasons depend on it. | [`resolve_policy`](../src/switchboard/engine/loop.py), [`_degrade`](../src/switchboard/engine/loop.py) |
| Fallback shape | Consumers expect fallback to arrive as `kind="route"`. | [`apply_fallback`](../src/switchboard/engine/policy.py) |
| Content-mode gate | Prevents payload leakage to sinks/spans. | [`apply_content_mode`](../src/switchboard/telemetry/emitter.py) |
| Optional dependency imports | Zero-dependency contract is tested. | [`tests/test_zero_deps.py`](../tests/test_zero_deps.py) |

## Related documents

- [Architecture](02-ARCHITECTURE.md)
- [Business logic](05-BUSINESS-LOGIC.md)
- [API contracts](07-API-CONTRACTS.md)
- [Development](09-DEVELOPMENT.md)
