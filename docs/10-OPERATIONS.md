# Operations
Purpose: Summarize deployment, rollback, observability, runbook guidance, and known failure modes from the code.
Audience: Operators, SREs, platform engineers, and application owners embedding Switchboard.
Last verified against commit not-a-git-repository on 2026-08-07.

## Deployment model

Switchboard is deployed as a Python library inside a host application. The repository does not define containers, Kubernetes manifests, systemd units, Terraform, database migrations, or a standalone server. Packaging metadata is in [`pyproject.toml`](../pyproject.toml); the wheel package is `src/switchboard`.

Typical production deployment steps:

1. Install the package plus the minimum required extras, for example `switchboard[instructor]`, `switchboard[litellm]`, or `switchboard[otel]`.
2. Configure provider credentials in the host environment or provider SDK configuration.
3. Build one process-lifetime `Registry`.
4. Build one process-lifetime `Router`.
5. Optionally call `router.warm()` / `router.awarm()` at startup to prebuild shortlisting indexes.
6. Route each request through `route()` / `aroute()`.
7. Execute selected application handlers outside Switchboard.
8. Call `router.close()` / `router.aclose()` during application shutdown.

## Release and package procedure

This repository now has a CI workflow at [`ci.yml`](../.github/workflows/ci.yml)
that runs lint, type check, tests, and offline examples. A safe release procedure
for this package is:

1. Run the CI workflow on the release branch.
2. Confirm [`pyproject.toml`](../pyproject.toml) dependency ranges and public API
   exports are intentional.
3. Build artifacts with the configured Hatch backend:

   ```bash
   python -m pip install --upgrade build
   python -m build
   ```

4. Smoke-test the wheel in a fresh virtual environment with `python -m pip install
   dist/*.whl` and `switchboard version`.
5. Publish through the repository owner's package publishing process. The code
   does not include credentials or publishing automation.

## Rollback

Rollback is host-owned but follows a predictable sequence. Because route catalog
identity is content-addressed, reverting route definitions changes
`registry.version`, which in turn invalidates prompt-cache segment B and
shortlist indexes. Evidence: [`Registry.version`](../src/switchboard/core/registry.py), [`segment_b_cache_key`](../src/switchboard/engine/prompt.py), and [`_BaseShortlister.index_key`](../src/switchboard/engine/shortlist.py).

Operationally:

- Roll back the host application package/container to the previous Switchboard and catalog version.
- Clear or allow rebuild of any external `IndexStore` entries if a custom store is used.
- Re-record eval replay cache entries if prompt/schema/model changed and strict replay starts missing.

## Health checks

Switchboard does not ship a network health endpoint because it is a library. Host
applications should expose their own readiness endpoint that runs these startup
checks before accepting traffic:

| Check | How | Failure symptom |
|---|---|---|
| Import/core health | Import `switchboard` and instantiate a small BYO-callable router. | Optional SDK imported too early or package install broken. |
| Catalog validity | Build `Registry([...])` at startup. | `RegistryError` for empty/duplicate/malformed catalog. |
| Provider config | Build `Router(client=...)` eagerly at startup. | `MissingDependencyError`, `ConfigError`, or provider SDK config error. |
| Shortlist warmup | Call `router.warm()` / `router.awarm()`. | Index-store/config errors before first user request. |
| Sink write | Configure `InMemorySink`/`JSONLSink` in staging and inspect latest record. | Missing audit records or sink warnings. |
| Eval replay lane | Run `run_suite()` with replay cache. | Cache miss means prompt/schema/model/sampling changed. |

Readiness should fail on catalog/provider construction errors. Liveness should be
host-process liveness only; do not call a real provider on every liveness probe,
or a provider outage will restart otherwise healthy workers and amplify the
incident.

## Audit and replay retention

Switchboard's JSONL files are append-only by design, so retention must be handled
by the host deployment:

| Artifact | Default retention recommendation | Rotation recommendation | Deletion / archival notes |
|---|---|---|---|
| Audit JSONL from `JSONLSink` | 30 days for operational debugging; longer only if compliance requires it. | Rotate daily or at 100 MB, whichever comes first, using the host log manager. | Use `content_mode="none"` or `"redacted"` unless a documented compliance basis allows payload retention. |
| Replay cache JSONL | Keep only committed deterministic fixtures needed by CI. | Re-record into a new file when prompt/schema/model changes; review diffs. | Delete stale cache files when their registry/model is no longer supported. |
| Distillation JSONL | Treat as training data; retain according to model-training governance. | Write immutable dated exports. | Include only records accepted by `switchboard.distill` filters; avoid raw payload exports unless approved. |

For high-throughput audit delivery, wrap slow sinks with `QueuedSink`; see
[`QueuedSink`](../src/switchboard/telemetry/emitter.py).

## Logs, metrics, and traces

| Signal | What exists | Where to look | Evidence |
|---|---|---|---|
| Warnings | Config warnings for oversized candidates, reasoning-default models, threshold provenance, sink/OTel failure warnings. | Python logging / warnings in host process. | [`Router.__init__`](../src/switchboard/router.py), [`RateLimitedLogger`](../src/switchboard/telemetry/emitter.py) |
| Audit records | One `AuditRecord` per routing call, including errors when possible. | `decision.audit`, configured sink, JSONL file. | [`AuditRecord`](../src/switchboard/core/audit.py), [`Router.route`](../src/switchboard/router.py) |
| JSONL logs | Append-only audit records. | `JSONLSink.path`. | [`JSONLSink`](../src/switchboard/telemetry/emitter.py) |
| OTel traces | Parent `invoke_agent`, child `chat`, optional `embeddings`, caller-opened `execute_tool`. | Host-configured OTel provider/exporter. | [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |
| Metrics | API-only OTel instruments record operation duration, token usage, decision count, and resolved cost when `[otel]` is installed. | Host-configured OTel meter provider/exporter. | [`OTelEmitter.record_metrics`](../src/switchboard/telemetry/otel.py) |

## Failure modes and runbook notes

| Failure mode | Symptom | Likely cause | Operator response | Evidence |
|---|---|---|---|---|
| Missing optional extra | `MissingDependencyError` during `Router(...)`. | Adapter spec named uninstalled SDK. | Install correct extra or use BYO callable/LiteLLM fallback. | [`MissingDependencyError`](../src/switchboard/errors.py), [`resolve_client`](../src/switchboard/providers/__init__.py) |
| Bad route catalog | `RegistryError` at startup. | Empty registry, duplicate route, bad name, invalid args model. | Fix catalog before deploy. | [`Registry.__init__`](../src/switchboard/core/registry.py), [`Route._validate_name`](../src/switchboard/core/route.py) |
| No eligible routes | `AbstainDecision(reason="no_eligible_routes")`; no provider usage. | User lacks required entitlements or visibility predicate failed closed. | Check `RequestContext.entitlements`, `Route.requires`, `predicate_errors`. | [`filter_routes`](../src/switchboard/engine/entitlements.py) |
| Provider outage/rate limit | Raised `ProviderError` by default, or `abstain(provider_error)` if configured. | SDK transport error after retry budget. | Inspect host provider credentials/network/rate limits; consider `on_provider_error`. | [`_call_provider_sync`](../src/switchboard/engine/loop.py) |
| Invalid provider output | `abstain(unparseable_output)` or `abstain(invalid_route_reference)` after repairs. | Model returned malformed JSON or route outside candidates. | Inspect `audit.validation_retries`, attempts in state during tests, provider capabilities/schema mode. | [`validate_output`](../src/switchboard/engine/loop.py) |
| Missing route args | `ClarifyDecision` with `missing` fields or `abstain(invalid_args)`. | Required args absent or unparseable. | Ask user for missing fields; improve route descriptions/examples. | [`validate_args`](../src/switchboard/engine/validate.py), [`_missing_args_question`](../src/switchboard/engine/loop.py) |
| Weak retrieval | Clarify band widened; `audit.weak_retrieval=True`. | Few candidates scored above zero. | Improve descriptions/examples/tags or use embedding shortlister. | [`_BaseShortlister._assemble`](../src/switchboard/engine/shortlist.py), [`WEAK_RETRIEVAL_CLARIFY_MARGIN`](../src/switchboard/engine/policy.py) |
| Sink failure | Routing succeeds; warning logged and record dropped for that sink. | JSONL path closed/unwritable or callback raises. | Fix sink path/callback; monitor warnings. | [`safe_emit`](../src/switchboard/telemetry/emitter.py), [`JSONLSink`](../src/switchboard/telemetry/emitter.py) |
| OTel unavailable | `OTelEmitter.available=False`, no spans. | `[otel]` extra missing or disabled. | Install/configure OTel API/SDK in host app if traces required. | [`otel_available`](../src/switchboard/telemetry/otel.py), [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |

## Escalation context

When escalating an incident, include:

- `decision.audit.decision_id`
- `decision.audit.registry_version`
- `decision.kind`, `decision.decision_path`, and `decision.audit.abstain_reason`
- `decision.audit.trace_id` / `span_id` if available
- `decision.audit.shortlist`, `weak_retrieval`, `shortlist_skipped`
- Provider, request model, response model, usage, and latency from audit
- Whether content capture was `none`, `redacted`, or `full`

## Related documents

- [Configuration](08-CONFIGURATION.md)
- [Data model](04-DATA-MODEL.md)
- [Business logic](05-BUSINESS-LOGIC.md)
- [Technical debt](12-TECH-DEBT.md)
