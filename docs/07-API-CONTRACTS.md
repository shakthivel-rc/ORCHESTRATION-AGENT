# API Contracts
Purpose: Document public interfaces, shapes, errors, and breaking-change surfaces.
Audience: Application developers, integration authors, and maintainers changing public APIs.
Last verified against commit not-a-git-repository on 2026-08-07.

## Python package API

The supported import surface is `switchboard.__all__`; names outside it, including `switchboard.engine`, are internal. Evidence: [`switchboard.__init__`](../src/switchboard/__init__.py).

### Catalog construction

| Interface | Auth | Request shape | Response shape | Errors | Breaking-change warning |
|---|---|---|---|---|---|
| `Route(...)` | None | Fields are defined in [data model](04-DATA-MODEL.md). | Frozen `Route`. | `RegistryError` for invalid name or `args_model`. | Route name grammar and content-hash inputs affect prompt/schema/cache identity. |
| `Registry(routes)` | None | Iterable of `Route`. | Frozen `Registry`. | `RegistryError` for empty registry, non-Route item, duplicate name. | Registry version semantics affect audit, prompt cache, shortlist indexes, eval fixtures. |
| `Registry.merge(other, on_conflict=...)` | None | Registry or iterable of `Route`; conflict mode `error`, `override`, `keep`. | New `Registry`. | `ConfigError` for bad conflict mode; `RegistryError` for duplicate under `error`. | Merge ordering is deterministic; changing it changes registry versions. |
| `Registry.view(ctx=None)` | None | Optional `RequestContext`. | `RegistryView`. | None in normal use. | Current view is not the v0.1 security boundary; do not move enforcement out of `filter_routes()` without tests. |

### Routing

| Interface | Auth | Request shape | Response shape | Errors | Breaking-change warning |
|---|---|---|---|---|---|
| `Router(registry, *, client, ...)` | None | Config keys are owned by [configuration](08-CONFIGURATION.md). | `Router` instance. | `ConfigError`, `MissingDependencyError` for invalid config/client/extra; warnings for risky model/candidate settings. | Constructor eagerly validates config; moving failures to first request weakens production safety. |
| `Router.route(query: str, *, context: RequestContext | None = None)` | Caller-owned | Query string and optional context. | `Decision` union. | `ConfigError` for sync driver with async-only client; `ProviderError` by default after retry exhaustion. | Return union discriminates on `kind`; consumers branch on that field. |
| `Router.aroute(query: str, *, context: RequestContext | None = None)` | Caller-owned | Same as `route`. | `Decision` union. | Provider/config errors as above. | Sync/async parity is an explicit design invariant. |
| `Router.warm()` / `Router.awarm()` | None | No args. | `None`. | Shortlister build/config errors. | Used to move index build off first request. |
| `Router.close()` / `Router.aclose()` | None | No args. | `None`; sink failures swallowed/logged. | None propagated by shipped sinks. | Closing JSONL sinks drops later records with warning. |
| `Router.stream_route()` / `Router.astream_route()` | None | Query + context. | None; raises. | `ConfigError` because streaming is v0.2. | Do not document as implemented until behavior exists. |
| `Router.record_outcome(decision_id, outcome)` | Caller-owned | Decision ULID and outcome string. | `None`. | Sink hook errors logged, not raised. | v0.5 seam only; no built-in persistent outcome store. |

### Decision union

Entity fields are canonical in [data model](04-DATA-MODEL.md). Wire consumers must branch on `decision.kind`:

- `route`: call `decision.route` with `decision.args`.
- `multi_route`: iterate `decision.routes`.
- `clarify`: ask `decision.question`; optional `candidates` and `missing` aid UI/state.
- `abstain`: inspect closed `decision.reason`.
- `plan`: type is defined but not emitted in v0.1.

Evidence: [`Decision`](../src/switchboard/core/decision.py), [`RouteDecision`](../src/switchboard/core/decision.py), [`ClarifyDecision`](../src/switchboard/core/decision.py), [`AbstainDecision`](../src/switchboard/core/decision.py).

### Provider protocol

| Interface | Request | Response | Errors | Evidence |
|---|---|---|---|---|
| `LLMClient.complete(request)` | `LLMRequest` with prompt segments, output schema, sampling knobs. | `LLMResult`. | Adapter maps transport failures to `ProviderError` subclasses; schema failures should return `LLMResult(parsed=None, ...)`. | [`LLMClient`](../src/switchboard/providers/base.py) |
| `AsyncLLMClient.acomplete(request)` | Same. | `LLMResult`. | Same. | [`AsyncLLMClient`](../src/switchboard/providers/base.py) |
| Plain callable client | Either rendered prompt string or `LLMRequest`, inferred by signature/name/type hints. | `str`, `dict`, Pydantic model, or `LLMResult`. | `ConfigError` for unusable signature/return type; async mismatch errors. | [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py) |
| Adapter spec string | `"adapter:model"`; only first colon separates adapter. | Built adapter instance. | `ConfigError` for unknown/deferred adapter; `MissingDependencyError` for missing extra. | [`parse_client_spec`](../src/switchboard/providers/__init__.py), [`resolve_client`](../src/switchboard/providers/__init__.py) |
| Native OpenAI spec | `"openai:model"` | `OpenAIAdapter`. | `MissingDependencyError` if `[openai]` is not installed; `ProviderError` subclasses for SDK transport failures. | [`OpenAIAdapter`](../src/switchboard/providers/openai_adapter.py) |

Changing `LLMRequest`, `LLMResult`, `ClientCapabilities`, `TokenLP`, or `Usage` breaks adapters, confidence, audit, and replay cache.

### Shortlister and index contracts

| Interface | Request | Response | Errors | Evidence |
|---|---|---|---|---|
| `Shortlister.build(routes, registry_version=...)` | Full registry route sequence and registry version. | `None`. | `ConfigError` for invalid backend state. | [`Shortlister`](../src/switchboard/engine/shortlist.py) |
| `Shortlister.shortlist(query, allowed, k, ctx=None)` | Redacted query, entitled route-name set, K. | `ShortlistResult`. | `ConfigError` for invalid K or unbuilt index. | [`Shortlister`](../src/switchboard/engine/shortlist.py) |
| `IndexStore.load/save` | String key and JSON bytes. | Bytes or `None`. | Store-defined; shortlisters rebuild on invalid blobs. | [`IndexStore`](../src/switchboard/engine/shortlist.py) |

### Telemetry contracts

| Interface | Request | Response | Errors | Evidence |
|---|---|---|---|---|
| `DecisionSink.emit(record)` | Final `AuditRecord`. | `None`. | Must not raise; router wraps with `safe_emit`. | [`DecisionSink`](../src/switchboard/telemetry/emitter.py), [`safe_emit`](../src/switchboard/telemetry/emitter.py) |
| `JSONLSink(path)` | Path to JSONL file. | Sink object. | Write/flush/close failures swallowed and logged. | [`JSONLSink`](../src/switchboard/telemetry/emitter.py) |
| `QueuedSink(sink, maxlen=..., drop_oldest=...)` | Wrapped sink and queue policy. | Sink object with `dropped`, `queued`, `closed` counters/properties. | Constructor raises `ConfigError` for invalid max length; downstream failures are isolated. | [`QueuedSink`](../src/switchboard/telemetry/emitter.py) |
| `OTelEmitter` span methods | Optional request/decision/result data. | Context-manager span handles. | OTel failures swallowed/logged. | [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |
| `OTelEmitter.record_metrics(record)` | Final `AuditRecord`. | OTel API metric recordings when available; no-op otherwise. | OTel failures swallowed/logged. | [`OTelEmitter.record_metrics`](../src/switchboard/telemetry/otel.py) |

### Eval API

| Interface | Request | Response | Errors | Evidence |
|---|---|---|---|---|
| `load_suite(path)` / `save_suite(suite, path)` | JSONL fixture path or suite. | `EvalSuite` / `Path`. | `ConfigError` for malformed/unknown schema. | [`load_suite`](../src/switchboard/evals/fixtures.py), [`save_suite`](../src/switchboard/evals/fixtures.py) |
| `ReplayCache(path)` | File or directory path. | Cache object. | `ConfigError` for malformed cache lines/schema. | [`ReplayCache`](../src/switchboard/evals/cache.py) |
| `ReplayClient(client, cache, mode, ...)` | Provider client or replay-only metadata. | Provider protocol client. | Strict replay miss raises `ConfigError`. | [`ReplayClient`](../src/switchboard/evals/cache.py) |
| `run_suite(router, suite, ...)` / `arun_suite(...)` | Router and eval suite/cases. | `SuiteResult`. | `ConfigError` for malformed suite, unknown baseline, replay miss, bad concurrency. | [`run_suite`](../src/switchboard/evals/harness.py), [`arun_suite`](../src/switchboard/evals/harness.py) |
| `collect_training_examples(records, ...)` | Iterable of `AuditRecord`. | `(list[TrainingExample], DistillStats)`. | `ConfigError` for invalid thresholds/caps. | [`switchboard.distill`](../src/switchboard/distill/__init__.py) |
| `write_jsonl(records, path, ...)` | Iterable of `AuditRecord` and output path. | `DistillStats`; writes JSONL rows. | `ConfigError` for invalid thresholds/caps. | [`write_jsonl`](../src/switchboard/distill/__init__.py) |
| `iter_audit_jsonl(path)` | Audit JSONL path. | Iterator of `AuditRecord`. | `ConfigError` for invalid JSON or non-audit rows. | [`iter_audit_jsonl`](../src/switchboard/distill/__init__.py) |

## REST / HTTP interfaces

Switchboard ships no service routes. The only REST endpoint in the repository is an example:

| Method | Path | Auth | Request | Response | Errors | Evidence |
|---|---|---|---|---|---|---|
| `POST` | `/chat` | Example `Depends(auth)` returns `User`; not Switchboard auth. | `{"message": str}` via `ChatIn`. | `200` route handler reply for route/fallback, `200 {"reply": question}` for clarify. | `422 {"error": "cannot_route", "reason": AbstainReason}` for abstain when no fallback resolves it. | [`chat`](../examples/fastapi_app.py) |

This is illustrative. Changing it does not change package API, but it can break example consumers.

## Events, webhooks, and message schemas

- No webhooks are implemented.
- No runtime event bus is implemented.
- The replay cache and eval fixture JSONL formats are file schemas, not message bus contracts; see [data model](04-DATA-MODEL.md).
- OTel spans are emitted through host-configured OpenTelemetry providers, not a Switchboard-owned exporter.

## CLI commands

The package installs a `switchboard` script through [`pyproject.toml`](../pyproject.toml).

| Command | Auth | Input | Output | Errors | Evidence |
|---|---|---|---|---|---|
| `switchboard version` | None | No args. | Version string. | Process exits nonzero only on unexpected runtime error. | [`cli.py`](../src/switchboard/cli.py) |
| `switchboard eval dogfood [--routes N] [--seed N] [--no-baseline] [--json]` | None | Generated dogfood suite. | Summary table or `SuiteResult` JSON. | Eval/router config errors. | [`cli.py`](../src/switchboard/cli.py) |
| `switchboard eval inspect PATH` | None | Eval suite JSONL path. | Suite metadata JSON. | Fixture loading errors. | [`cli.py`](../src/switchboard/cli.py), [`load_suite`](../src/switchboard/evals/fixtures.py) |

Full `lint`, `enrich`, `distill`, `costs`, YAML gates, and rich reporting commands remain roadmap work.

## Error codes and exception tree

Switchboard's exception tree is:

- `SwitchboardError`
  - `ConfigError`
    - `RegistryError`
    - `MissingDependencyError`
  - `ProviderError`
    - `ProviderTimeout`
    - `ProviderRateLimit`
    - `ProviderAuthError`

Evidence: [`errors.py`](../src/switchboard/errors.py). Decision-level refusal reasons use the `AbstainReason` literal in [data model](04-DATA-MODEL.md), not exceptions.

## Related documents

- [Data model](04-DATA-MODEL.md)
- [Configuration](08-CONFIGURATION.md)
- [Features](06-FEATURES.md)
- [Extending](11-EXTENDING.md)
