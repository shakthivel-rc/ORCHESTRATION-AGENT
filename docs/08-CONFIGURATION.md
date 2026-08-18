# Configuration
Purpose: Canonically document every environment variable, Router/config key, and project config key present in code.
Audience: Application integrators, operators, and maintainers adjusting runtime behavior.
Last verified against commit not-a-git-repository on 2026-08-07.

## Environment variables

| Name | Purpose | Default | Required | Secret | Blast radius if wrong | Evidence |
|---|---|---|---|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Enables OTel message content capture only when `Router(content_mode!="none")` also permits payload content. Truthy values are `1`, `true`, `yes`, `on`, `y`, `t`. | Empty / false | Optional | No | If enabled with permissive `content_mode`, redacted/full prompt and output content can reach tracing backend; if disabled, span content is omitted. | [`CAPTURE_CONTENT_ENV`](../src/switchboard/telemetry/otel.py), [`OTelEmitter.capture_content`](../src/switchboard/telemetry/otel.py) |

No other environment variables are read directly by Switchboard source. Provider API keys, endpoints, and SDK-specific settings are delegated to the provider SDK or host application.

OPEN QUESTION: The repository does not document provider-specific credential environment variables or secret-source conventions for Instructor, LiteLLM, OpenAI, Anthropic, Gemini, or Bedrock deployments.

## `Router(...)` configuration keys

| Key | Purpose | Default | Required | Secret | Blast radius if wrong | Evidence |
|---|---|---|---|---|---|---|
| `registry` | Route catalog. | None | Required | No | Bad type raises; wrong catalog routes to wrong domain. | [`Router.__init__`](../src/switchboard/router.py) |
| `client` | Provider spec/client/callable. | None | Required | May wrap secrets externally | Missing/unknown extra or wrong adapter prevents routing. | [`Router.__init__`](../src/switchboard/router.py), [`resolve_client`](../src/switchboard/providers/__init__.py) |
| `shortlist` | Retrieval backend spec/object/disabled. | `"auto"` | Optional | No | Disabled over large catalog is refused; wrong backend can hurt recall. | [`Router.__init__`](../src/switchboard/router.py), [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) |
| `shortlist_k` | Candidate K override. | `None` -> band table | Optional | No | Too low misses routes; too high costs tokens and may reduce accuracy. | [`effective_k`](../src/switchboard/engine/shortlist.py) |
| `shortlist_min_routes` | Auto bypass threshold. | `25` | Optional | No | Too high bypasses retrieval on large catalogs; too low adds retrieval miss-risk. | [`DEFAULT_SHORTLIST_MIN_ROUTES`](../src/switchboard/engine/shortlist.py) |
| `multi_route` | Enables `multi_route` arm. | `False` | Optional | No | If false, multi-route model output becomes ambiguity; if true, consumers must handle multi-route. | [`build_wire_schema`](../src/switchboard/engine/schema.py), [`resolve_decision`](../src/switchboard/engine/policy.py) |
| `allow_clarify` | Enables clarify arm. | `True` | Optional | No | If false, underdetermined/low-confidence cases abstain instead of asking. | [`enabled_kinds`](../src/switchboard/engine/schema.py), [`_degrade`](../src/switchboard/engine/loop.py) |
| `allow_plan` | Future plan decision toggle. | `False` | Optional | No | Accepted but inert in v0.1. | [`enabled_kinds`](../src/switchboard/engine/schema.py) |
| `fallback` | Registered route to substitute on terminal abstain. | `None` | Optional | No | Unknown route raises; bad fallback can route too many failures to one handler. | [`Router.__init__`](../src/switchboard/router.py), [`apply_fallback`](../src/switchboard/engine/policy.py) |
| `confidence` | Confidence source spec. | `"logprobs"` | Optional | No | `"none"` disables thresholds; unsupported v0.2 specs raise. | [`ConfidenceSpec`](../src/switchboard/router.py) |
| `thresholds` | Downgrade policy. | `ThresholdPolicy()` | Optional | No | Bad thresholds raise; overly aggressive thresholds cause clarify/abstain. | [`ThresholdPolicy`](../src/switchboard/engine/policy.py) |
| `max_candidates` | Prompt candidate cap. | `25` | Optional | No | Too low may omit candidates; above 50 warns. | [`Router.__init__`](../src/switchboard/router.py) |
| `candidate_order` | Candidate presentation order. | `"shuffle"` | Optional | No | Invalid value raises; non-shuffle can reintroduce position bias. | [`order_candidates`](../src/switchboard/engine/prompt.py) |
| `retry` | Schema/provider retry budgets. | `RetrySpec()` | Optional | No | Too low reduces recovery; too high increases latency/cost. | [`RetrySpec`](../src/switchboard/engine/loop.py) |
| `redactor` | Query/content redaction hook. | `None` | Optional | Can protect secrets | Raising redactor is a `ConfigError` before provider call. Missing redactor with `content_mode="redacted"` drops payloads. | [`Router._new_state`](../src/switchboard/router.py), [`apply_content_mode`](../src/switchboard/telemetry/emitter.py) |
| `content_mode` | Audit payload capture mode. | `"none"` | Optional | Controls secret exposure | `"full"` can persist raw payloads to sinks/spans if OTel env also allows. | [`ContentMode`](../src/switchboard/telemetry/emitter.py) |
| `wire_schema` | Schema mode setting. | `"auto"` | Optional | No | Wrong mode can trade strict decoding for cache behavior. | [`resolve_schema_mode`](../src/switchboard/engine/schema.py) |
| `per_tenant_index` | Future per-tenant index scope. | `False` | Optional | No | `True` raises in v0.1. | [`Router.__init__`](../src/switchboard/router.py) |
| `on_provider_error` | Provider failure policy. | `"raise"` | Optional | No | `"abstain"` hides transport failures as decisions; `"raise"` surfaces outages. | [`_handle_provider_error`](../src/switchboard/engine/loop.py) |
| `sink` | Audit sink or iterable of sinks. | `None` -> `NoopSink` | Optional | Sink may hold secrets | Broken sinks drop records with warnings, not request failures. | [`resolve_sink`](../src/switchboard/telemetry/emitter.py) |
| `system_prompt` | Override segment A text. | Built-in `SYSTEM_PROMPT` | Optional | Maybe | Bad prompt can reduce accuracy; cache key changes for custom text. | [`build_segments`](../src/switchboard/engine/prompt.py), [`segment_a_cache_key`](../src/switchboard/engine/prompt.py) |
| `embed` | BYO embedding callable. | `None` | Optional | Maybe | Required for BYO embed shortlisting; async callable requires async APIs. | [`EmbeddingShortlister`](../src/switchboard/engine/shortlist.py) |
| `index_store` | Shortlist index persistence. | `None` | Optional | Depends on store | Bad/stale blobs rebuild; bad store can increase latency. | [`IndexStore`](../src/switchboard/engine/shortlist.py) |
| `allow_oversized_k` | Permit K above 20. | `False` | Optional | No | Can increase token cost and reduce selection quality. | [`effective_k`](../src/switchboard/engine/shortlist.py) |
| `thresholds_on_verbalized` | Allow verbalized-only confidence to trigger thresholds. | `False` | Optional | No | Warns; can over-abstain/clarify based on weak signal. | [`Router.__init__`](../src/switchboard/router.py), [`signals_are_actionable`](../src/switchboard/engine/confidence.py) |
| `confidence_fusion` | Custom fusion callable. | `None` -> `default_fusion` | Optional | No | Bad fusion changes downgrade behavior. | [`build_confidence_report`](../src/switchboard/engine/confidence.py) |
| `default_visibility` | Entitlement default posture. | `"allow"` | Optional | No | `"deny"` raises in v0.1. | [`Router.__init__`](../src/switchboard/router.py), [`filter_routes`](../src/switchboard/engine/entitlements.py) |
| `temperature` | Provider sampling temperature. | `0.0` | Optional | No | Higher temperature can reduce determinism. | [`LLMRequest`](../src/switchboard/providers/base.py) |
| `max_tokens` | Provider output token cap. | `512` | Optional | No | Too low can truncate JSON; too high costs more. | [`LLMRequest`](../src/switchboard/providers/base.py) |
| `seed` | Provider seed when supported. | `None` | Optional | No | Affects reproducibility where provider honors it. | [`LLMRequest`](../src/switchboard/providers/base.py) |
| `otel` | Enable OTel emitter. | `True` | Optional | No | If extra absent, no-op; if configured host tracer, emits spans. | [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |
| `tracer_provider` | Explicit OTel tracer provider. | `None` | Optional | Maybe | Wrong provider can drop/misdirect spans. | [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |
| `agent_name` | OTel GenAI agent name. | `"switchboard.router"` | Optional | No | Affects span naming/attribution. | [`OTelEmitter.decision_span`](../src/switchboard/telemetry/otel.py) |
| `agent_id` | OTel GenAI agent ID. | `None` | Optional | No | Affects span attribution. | [`OTelEmitter.decision_span`](../src/switchboard/telemetry/otel.py) |

## Object-form config

| Type | Keys and defaults | Evidence |
|---|---|---|
| `ClientSpec` | `adapter`, `model`, `temperature=0.0`, `want_logprobs=True` | [`ClientSpec`](../src/switchboard/router.py) |
| `ShortlistSpec` | `variant="auto"`, `top_k=None`, `min_routes=None`, `model=None`, `backend=None` | [`ShortlistSpec`](../src/switchboard/router.py) |
| `ConfidenceSpec` | `source="logprobs"`, `vote_n=1`; `vote_n != 1` raises in v0.1 | [`ConfidenceSpec`](../src/switchboard/router.py) |
| `RetrySpec` | `schema_attempts=2`, `provider_attempts=3`, `backoff="expo_jitter"`, `base_delay=0.5`, `max_delay=8.0` | [`RetrySpec`](../src/switchboard/engine/loop.py) |
| `ThresholdPolicy` | `abstain_below=0.30`, `clarify_below=0.55`, `margin_clarify_below=0.10`, `multi_route_member_min=0.35`, provenance fields default `None` | [`ThresholdPolicy`](../src/switchboard/engine/policy.py) |

## Shortlist DSL config

| Spec | Meaning | Evidence |
|---|---|---|
| `auto` | Bypass below threshold, otherwise BM25. | [`AutoShortlister`](../src/switchboard/engine/shortlist.py) |
| `bm25[:top_k=...,min_routes=...,k1=...,b=...]` | Pure-Python BM25F. | [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) |
| `embed[:top_k=...,min_routes=...,model=...,backend=...]` | Embedding shortlister using BYO callable or packaged backend. | [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) |
| `hybrid` | v0.2, raises in v0.1. | [`resolve_shortlister`](../src/switchboard/engine/shortlist.py) |
| `None` | Disable shortlisting; Router refuses oversized catalogs that would truncate. | [`Router.__init__`](../src/switchboard/router.py) |

## Project/tool config keys

| Section | Key | Value | Evidence |
|---|---|---|---|
| `[project]` | `name` | `switchboard` | [`pyproject.toml`](../pyproject.toml) |
| `[project]` | `requires-python` | `>=3.10` | [`pyproject.toml`](../pyproject.toml) |
| `[project]` | `dependencies` | `pydantic>=2.7,<3` | [`pyproject.toml`](../pyproject.toml) |
| `[project.optional-dependencies]` | extras | See [tech stack](03-TECH-STACK.md) | [`pyproject.toml`](../pyproject.toml) |
| `[tool.hatch.version]` | `source`, `fallback-version` | `vcs`, `0.1.0.dev0` | [`pyproject.toml`](../pyproject.toml) |
| `[tool.ruff]` | `target-version`, `line-length`, `src` | `py310`, `110`, `["src", "tests"]` | [`pyproject.toml`](../pyproject.toml) |
| `[tool.mypy]` | `strict`, `files`, `python_version` | `true`, `["src/switchboard"]`, `3.10` | [`pyproject.toml`](../pyproject.toml) |
| `[tool.pytest.ini_options]` | `testpaths`, `asyncio_mode`, `pythonpath`, `addopts` | `["tests"]`, `auto`, `["src"]`, `-q` | [`pyproject.toml`](../pyproject.toml) |

## Environment differences

The repository does not define separate local/dev/staging/prod config files. Differences are created by host application configuration:

- Local/offline examples use BYO deterministic callables and `otel=False`.
- Production-like deployments should configure real provider clients, credentials in the host environment, redaction, sinks, and OTel provider/exporter outside Switchboard.
- Eval/replay workflows can wrap a client with `ReplayClient` to avoid provider credentials in CI.

## Related documents

- [Tech stack](03-TECH-STACK.md)
- [Data model](04-DATA-MODEL.md)
- [Operations](10-OPERATIONS.md)
- [Extending](11-EXTENDING.md)
