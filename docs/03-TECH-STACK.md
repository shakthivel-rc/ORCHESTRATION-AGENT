# Tech Stack
Purpose: Inventory languages, frameworks, libraries, optional services, and version constraints.
Audience: Engineers setting up the project, reviewing dependencies, or changing adapters.
Last verified against commit not-a-git-repository on 2026-08-07.

## Languages and runtimes

| Item | Version / constraint | Role | Evidence |
|---|---|---|---|
| Python | `>=3.10` | Only implementation language. | [`pyproject.toml`](../pyproject.toml) |
| PEP 561 typing | `src/switchboard/py.typed` | Marks package as typed. | [`py.typed`](../src/switchboard/py.typed), [`pyproject.toml`](../pyproject.toml) |
| Package versioning | Dynamic via `hatch-vcs`, fallback `0.1.0.dev0` | Uses Git tags when repository is initialized; fallback exists because checkout has no Git metadata. | [`pyproject.toml`](../pyproject.toml), [`_version.py`](../src/switchboard/_version.py) |

## Build and packaging

| Tool | Purpose | Evidence |
|---|---|---|
| `hatchling` | Build backend. | [`pyproject.toml`](../pyproject.toml) |
| `hatch-vcs` | Dynamic version source with fallback. | [`pyproject.toml`](../pyproject.toml) |
| `src/` layout | Wheel package path is `src/switchboard`. | [`pyproject.toml`](../pyproject.toml) |
| Project script | Installs `switchboard = switchboard.cli:main`. | [`pyproject.toml`](../pyproject.toml), [`cli.py`](../src/switchboard/cli.py) |

## Required dependency

| Dependency | Constraint | Why it is load-bearing | Evidence |
|---|---|---|---|
| `pydantic` | `>=2.7,<3` | Every public data object, decision union, generated wire schema, provider request/result, and eval fixture is a Pydantic model. | [`pyproject.toml`](../pyproject.toml), [`Route`](../src/switchboard/core/route.py), [`build_wire_schema`](../src/switchboard/engine/schema.py) |

## Optional extras

| Extra | Dependencies | Implemented code path today | Evidence |
|---|---|---|---|
| `instructor` | `instructor>=1.7,<2` | `InstructorAdapter`, recommended v0.1 structured-output path. | [`pyproject.toml`](../pyproject.toml), [`InstructorAdapter`](../src/switchboard/providers/instructor_adapter.py) |
| `litellm` | `litellm>=1.61,<2` | `LiteLLMAdapter`, broad provider matrix. | [`pyproject.toml`](../pyproject.toml), [`LiteLLMAdapter`](../src/switchboard/providers/litellm_adapter.py) |
| `otel` | `opentelemetry-api>=1.25,<2` | OTel span emission only; no SDK/exporter installed by Switchboard. | [`pyproject.toml`](../pyproject.toml), [`OTelEmitter`](../src/switchboard/telemetry/otel.py) |
| `embed` | `fastembed>=0.3,<1`, `numpy>=1.24,<3` | Packaged embedding backend loader exists; BYO embedding callable needs no extra. | [`pyproject.toml`](../pyproject.toml), [`EmbeddingShortlister`](../src/switchboard/engine/shortlist.py), [`embedding_backends.py`](../src/switchboard/shortlisters/embedding_backends.py) |
| `eval` | `pyyaml>=6,<7`, `rich>=13,<15` | CLI/reporting extra is declared, but eval modules themselves use stdlib + Pydantic. | [`pyproject.toml`](../pyproject.toml), [`switchboard.evals`](../src/switchboard/evals/__init__.py) |
| `openai` | `openai>=1.60,<3` | Native OpenAI chat-completions adapter. | [`pyproject.toml`](../pyproject.toml), [`OpenAIAdapter`](../src/switchboard/providers/openai_adapter.py) |
| `anthropic`, `gemini`, `bedrock` | Declared in `pyproject.toml` | Native adapters are still deferred; use Instructor or LiteLLM paths meanwhile. | [`pyproject.toml`](../pyproject.toml), [`providers.resolve_client`](../src/switchboard/providers/__init__.py) |
| `distill`, `distill-train` | Declared in `pyproject.toml` | JSONL distillation helpers are implemented without extras; heavier Parquet/training extras remain future work. | [`pyproject.toml`](../pyproject.toml), [`distill/__init__.py`](../src/switchboard/distill/__init__.py) |
| `all` | Union of extras | Development / demos / broad test environment. | [`pyproject.toml`](../pyproject.toml) |

Every optional dependency range is upper-bounded in [`pyproject.toml`](../pyproject.toml). Imports are lazy in adapter and telemetry modules, and zero-dependency behavior is guarded by [`tests/test_zero_deps.py`](../tests/test_zero_deps.py).

## Frameworks and integrations

Switchboard itself does not depend on FastAPI, LangGraph, or Google ADK. Example integrations guard those imports and provide stand-ins when absent:

- [`examples/fastapi_app.py`](../examples/fastapi_app.py): FastAPI `POST /chat` example.
- [`examples/langgraph_node.py`](../examples/langgraph_node.py): LangGraph conditional-edge node.
- [`examples/adk_agent.py`](../examples/adk_agent.py): Google ADK tool wrapper.

## Managed services

Switchboard directly manages no cloud service, database, queue, scheduler, exporter, or hosted runtime. It can call model providers through optional SDK adapters:

- Instructor-backed providers through [`InstructorAdapter`](../src/switchboard/providers/instructor_adapter.py).
- LiteLLM-supported providers through [`LiteLLMAdapter`](../src/switchboard/providers/litellm_adapter.py).
- Bring-your-own callables through [`CallableAdapter`](../src/switchboard/providers/callable_adapter.py).

Provider credentials and endpoints are delegated to the host app or the provider SDKs; Switchboard does not read API-key environment variables itself.

## Development tools

| Tool | Config | Evidence |
|---|---|---|
| Ruff | Target Python 3.10, line length 110, selected lint sets. | [`pyproject.toml`](../pyproject.toml) |
| mypy | Strict over `src/switchboard`, targeted relaxations for optional SDK adapters. | [`pyproject.toml`](../pyproject.toml) |
| pytest | Test path `tests`, `asyncio_mode=auto`, `pythonpath=src`, quiet addopts. | [`pyproject.toml`](../pyproject.toml) |
| GitHub Actions | CI runs lint, mypy, pytest, and offline examples on Python 3.10/3.12. | [`ci.yml`](../.github/workflows/ci.yml) |

## Deliberate vs incidental version constraints

Deliberate constraints:

- Python `>=3.10` is declared in project metadata and test tooling.
- `pydantic>=2.7,<3` is the hard dependency and foundational model layer.
- Optional extras are upper-bounded and lazily imported, matching comments in [`pyproject.toml`](../pyproject.toml) and tests in [`tests/test_zero_deps.py`](../tests/test_zero_deps.py).
- `opentelemetry-api` is API-only; SDK/exporter setup is user-owned per [`OTelEmitter`](../src/switchboard/telemetry/otel.py).

Incidental or currently unverified constraints:

- The repository includes `.venv/` artifacts in the working directory, but they are ignored by `.gitignore` and are not source of truth.
- No lockfile is present in the repository survey, despite [`ORCHESTRATION_AGENT_PLAN.md`](../ORCHESTRATION_AGENT_PLAN.md) discussing `uv.lock` as a future/release guard.

## Related documents

- [Configuration](08-CONFIGURATION.md)
- [Development](09-DEVELOPMENT.md)
- [Technical debt](12-TECH-DEBT.md)
- [ADR-0002](adr/0002-pydantic-only-core-lazy-extras.md)
