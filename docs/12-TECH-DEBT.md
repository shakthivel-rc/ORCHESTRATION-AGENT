# Technical Debt
Purpose: Rank incomplete features, gaps, risks, dead/stubbed code, and remediation paths.
Audience: Maintainers planning roadmap work, reviewers assessing risk, AI agents choosing safe tasks.
Last verified against commit not-a-git-repository on 2026-08-07.

## Ranked risks and gaps

| Rank | Risk | Evidence | Impact | Remediation sketch |
|---|---|---|---|---|
| 1 | Remaining native provider adapters are declared but not implemented. | OpenAI is implemented in [`openai_adapter.py`](../src/switchboard/providers/openai_adapter.py), but `anthropic_adapter.py`, `gemini_adapter.py`, and `bedrock_adapter.py` are still deferred in [`providers.resolve_client`](../src/switchboard/providers/__init__.py). | Users may expect those native extras to resolve directly but get `ConfigError`; they can use `instructor:` or `litellm:` meanwhile. | Implement one adapter at a time with lazy imports, capability tests, and live/replay tests. |
| 2 | `Registry.view()` and `filter_routes()` split can confuse security reasoning. | [`Registry.view`](../src/switchboard/core/registry.py) docstring calls v0.1 view passthrough; [`filter_routes`](../src/switchboard/engine/entitlements.py) enforces predicates. | A contributor may incorrectly rely on `Registry.view(ctx)` alone. | Either wire real view filtering/caching or strengthen tests around the current split. |
| 3 | Live provider behavior is not fully tested in repo. | Tests cover adapters mostly via import/missing-extra and fakes; no provider cassettes or live matrix are present. | Adapter behavior may drift with SDK/provider changes. | Add replay-backed adapter fixtures and opt-in live tests. |
| 4 | `allow_plan`, `record_outcome`, margin, self-consistency, and multi-route member thresholds are future seams. | Stub/inert fields across [`core/decision.py`](../src/switchboard/core/decision.py), [`router.py`](../src/switchboard/router.py), [`confidence.py`](../src/switchboard/engine/confidence.py), and [`policy.py`](../src/switchboard/engine/policy.py). | Public surface exists but behavior is partial. | Implement or clearly label phase-gated APIs in generated docs/API docs. |
| 5 | No lockfile despite release-plan discussion. | [`ORCHESTRATION_AGENT_PLAN.md`](../ORCHESTRATION_AGENT_PLAN.md) discusses `uv.lock`; no lockfile in file survey. | Reproducibility depends on package ranges and current resolver state. | Add lockfile if the project wants reproducible dev/test envs. |

## Recently implemented from this list

| Item | Implementation |
|---|---|
| Minimal CLI and package entry point | [`cli.py`](../src/switchboard/cli.py) implements `switchboard version`, `switchboard eval dogfood`, and `switchboard eval inspect`; [`pyproject.toml`](../pyproject.toml) enables the `switchboard` script. |
| Native OpenAI adapter | [`OpenAIAdapter`](../src/switchboard/providers/openai_adapter.py) implements direct OpenAI Chat Completions with JSON schema response format, usage, logprobs, and lazy SDK import. |
| Deployment, rollback, readiness, and retention guidance | [Operations](10-OPERATIONS.md) now includes a release procedure, readiness/liveness guidance, and JSONL retention/rotation recommendations. |
| Queue-based sink hardening | [`QueuedSink`](../src/switchboard/telemetry/emitter.py) adds a bounded daemon-drained queue with drop counters, flush, and deterministic close. |
| OTel metrics | [`OTelEmitter.record_metrics`](../src/switchboard/telemetry/otel.py) records API-only operation duration, token usage, decision count, and cost metrics from finalized audit records. |
| Plan/source drift marker | [`ORCHESTRATION_AGENT_PLAN.md`](../ORCHESTRATION_AGENT_PLAN.md) now starts with an implementation-status table distinguishing current files from roadmap items. |
| Distillation exporter | [`switchboard.distill`](../src/switchboard/distill/__init__.py) now filters `AuditRecord` streams into training examples and writes JSONL with counters. |
| CI workflow for tests and examples | [`ci.yml`](../.github/workflows/ci.yml) runs lint, type check, pytest, and offline examples on Python 3.10/3.12. |

## Dead code / stubs / phase-tag clusters

| Area | Current state | Evidence |
|---|---|---|
| Streaming route APIs | Methods exist and raise `ConfigError`. | [`Router.stream_route`](../src/switchboard/router.py), [`Router.astream_route`](../src/switchboard/router.py) |
| Hybrid shortlist | Config parser recognizes name and raises `ConfigError`. | [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) |
| Plan decisions | Types exist; schema generation ignores `allow_plan` in v0.1. | [`PlanDecision`](../src/switchboard/core/decision.py), [`enabled_kinds`](../src/switchboard/engine/schema.py) |
| Deny-by-default visibility | Config accepted but raises when set to `"deny"`. | [`Router.__init__`](../src/switchboard/router.py), [`filter_routes`](../src/switchboard/engine/entitlements.py) |
| Per-tenant index | Config accepted but `True` raises. | [`Router.__init__`](../src/switchboard/router.py) |
| Eval CLI and full gates | Minimal CLI exists; rich reports, YAML gates, lint/enrich/cost commands remain deferred. | [`cli.py`](../src/switchboard/cli.py), [`switchboard.evals`](../src/switchboard/evals/__init__.py), [`pyproject.toml`](../pyproject.toml) |
| Native adapter names | Deferred adapters generate instructive errors. | [`_DEFERRED_ADAPTERS`](../src/switchboard/providers/__init__.py) |

## Security concerns

| Concern | Existing mitigation | Residual risk | Evidence |
|---|---|---|---|
| Prompt injection from user input | Query is last, tagged, and closing tag is neutralized. | Bad custom `system_prompt` can weaken instructions. | [`_render_query`](../src/switchboard/engine/prompt.py), [`_neutralize`](../src/switchboard/engine/prompt.py) |
| Route leakage across tenants | Entitlement filter runs before shortlisting/prompt; shortlister results are re-intersected. | `Registry.view()` passthrough may mislead direct users. | [`filter_routes`](../src/switchboard/engine/entitlements.py), [`_absorb_shortlist`](../src/switchboard/engine/loop.py) |
| Payload leakage in audit/traces | `content_mode` gate and OTel env double-key. | `content_mode="full"` plus env-enabled OTel can emit content. | [`apply_content_mode`](../src/switchboard/telemetry/emitter.py), [`OTelEmitter.capture_content`](../src/switchboard/telemetry/otel.py) |
| Optional dependency supply chain | Core has one dependency; optional imports are lazy and upper-bounded. | `all` extra still installs many optional packages in dev/CI. | [`pyproject.toml`](../pyproject.toml), [`tests/test_zero_deps.py`](../tests/test_zero_deps.py) |
| Visibility predicate behavior | Predicate exceptions fail closed and are recorded. | Slow predicates can hurt latency; code cannot enforce O(microseconds). | [`filter_routes`](../src/switchboard/engine/entitlements.py) |

## Missing tests or validation gaps

- No live-provider or recorded-cassette adapter suites are present for Instructor/LiteLLM/native provider behavior.
- The CI workflow exists, but it has not been run by GitHub in this checkout.
- No docs link checker or Mermaid parser configuration is committed as a reusable tool; verification was run ad hoc during documentation generation.
- `QueuedSink` covers bounded background delivery, but no durable fsync-per-record compliance sink exists.

## Related documents

- [Overview](01-OVERVIEW.md)
- [Operations](10-OPERATIONS.md)
- [Development](09-DEVELOPMENT.md)
- [ADR index](adr/README.md)
