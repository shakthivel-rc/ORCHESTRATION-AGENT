# Overview
Purpose: Explain what Switchboard is, who uses it, and where its system boundary starts and ends.
Audience: New engineers, AI coding agents, technical leads evaluating the project.
Last verified against commit not-a-git-repository on 2026-08-07.

## One-paragraph system summary

Switchboard is a Python package that provides a framework-agnostic, bring-your-own-model decision layer: callers register a typed catalog of routes, pass a user query plus optional request context to `Router.route()` or `Router.aroute()`, and receive a typed `Decision` telling them to call one route, call multiple routes, ask for clarification, or abstain. It does not execute handlers; route execution remains caller-owned. This boundary is explicit in [`Route`](../src/switchboard/core/route.py), whose docstring says a route carries no handler, and in [`Router.route`](../src/switchboard/router.py), which returns a `Decision` union rather than invoking application code.

## Primary users and goals

| User | Goal | Evidence |
|---|---|---|
| Application developers | Route user intent among their own tools/routes with typed args and no framework lock-in. | Public exports in [`switchboard.__init__`](../src/switchboard/__init__.py) include `Route`, `Registry`, `Router`, `Decision`, provider protocols, and sinks. |
| Agent/framework integrators | Put a stable decision step in front of LangGraph, FastAPI, Google ADK, or a custom tool runner. | Examples implement `route_node`, `chat`, and `pick_route` in [`examples/langgraph_node.py`](../examples/langgraph_node.py), [`examples/fastapi_app.py`](../examples/fastapi_app.py), and [`examples/adk_agent.py`](../examples/adk_agent.py). |
| Operators / platform engineers | Capture audit records, JSONL logs, and optional OpenTelemetry spans without letting telemetry break routing. | [`AuditRecord`](../src/switchboard/core/audit.py), [`DecisionSink`](../src/switchboard/telemetry/emitter.py), and [`OTelEmitter`](../src/switchboard/telemetry/otel.py). |
| Evaluation owners | Run labeled catalog evals with replayable provider calls and compare against a naive full-catalog baseline. | [`run_suite`](../src/switchboard/evals/harness.py), [`ReplayClient`](../src/switchboard/evals/cache.py), and [`EvalCase`](../src/switchboard/evals/fixtures.py). |

## Upstream systems

Switchboard depends on the caller to supply these inputs:

- A route catalog: [`Registry`](../src/switchboard/core/registry.py) built from [`Route`](../src/switchboard/core/route.py).
- A provider client: an adapter spec, a prebuilt client implementing [`LLMClient`](../src/switchboard/providers/base.py) / [`AsyncLLMClient`](../src/switchboard/providers/base.py), or a plain callable wrapped by [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py).
- Optional per-request state: [`RequestContext`](../src/switchboard/core/context.py), including tenant, user, entitlement, locale, history, and trace metadata.
- Optional telemetry sinks: implementations of [`DecisionSink`](../src/switchboard/telemetry/emitter.py).

## Downstream systems

Switchboard produces:

- A typed [`Decision`](../src/switchboard/core/decision.py) for caller code to branch on.
- A frozen [`AuditRecord`](../src/switchboard/core/audit.py) attached to every decision.
- Optional JSONL / callback / in-memory sink outputs through [`JSONLSink`](../src/switchboard/telemetry/emitter.py), [`CallbackSink`](../src/switchboard/telemetry/emitter.py), and [`InMemorySink`](../src/switchboard/telemetry/emitter.py).
- Optional OpenTelemetry GenAI spans through [`OTelEmitter`](../src/switchboard/telemetry/otel.py).
- Eval fixture and replay artifacts through [`switchboard.evals`](../src/switchboard/evals/__init__.py).

## Scope boundaries

Switchboard explicitly does not do the following:

- It does not execute tools, handlers, HTTP calls, graph nodes, or side effects; callers execute after inspecting the decision. Evidence: [`Route`](../src/switchboard/core/route.py) stores no handler field, and [`OTelEmitter.execute_tool_span`](../src/switchboard/telemetry/otel.py) is opened by the caller after a decision.
- It does not provide a running web service. The FastAPI route in [`examples/fastapi_app.py`](../examples/fastapi_app.py) is an example integration, not package runtime.
- It does not install provider SDKs in core. Provider SDK imports are lazy in [`providers.resolve_client`](../src/switchboard/providers/__init__.py), [`InstructorAdapter`](../src/switchboard/providers/instructor_adapter.py), and [`LiteLLMAdapter`](../src/switchboard/providers/litellm_adapter.py).
- It does not run a CLI today. [`pyproject.toml`](../pyproject.toml) comments out `[project.scripts]` because `switchboard/cli.py` is not present.
- It does not ship migrations or IaC. No database migration or infrastructure directories are present in the repository survey.

## README and comment contradictions

- [`README.md`](../README.md) says the root plan is the authoritative specification and describes several target-contract features. The executable code implements many of them, but CLI entry points, native OpenAI/Anthropic/Gemini/Bedrock adapters, and several v0.2/v0.5 features are deferred or stubbed. See [technical debt](12-TECH-DEBT.md).
- [`ORCHESTRATION_AGENT_PLAN.md`](../ORCHESTRATION_AGENT_PLAN.md) lists files such as `streaming.py`, `cli.py`, `hierarchy.py`, and native provider adapters that are not present in the current source tree. The code exposes `Router.stream_route()` / `Router.astream_route()` methods that raise `ConfigError` instead.

## Related documents

- [Architecture](02-ARCHITECTURE.md)
- [Data model](04-DATA-MODEL.md)
- [API contracts](07-API-CONTRACTS.md)
- [Technical debt](12-TECH-DEBT.md)
