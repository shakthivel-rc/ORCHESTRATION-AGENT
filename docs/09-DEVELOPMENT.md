# Development
Purpose: Explain clean-machine setup, local run commands, tests, fixtures, and repository-specific debugging workflows.
Audience: Contributors and AI agents preparing to modify code safely.
Last verified against commit not-a-git-repository on 2026-08-07.

## Clean-machine setup

1. Install Python 3.10 or newer.
2. From the repository root, install editable development dependencies:

   ```bash
   python -m pip install -e ".[all]"
   ```

   The root [`README.md`](../README.md) lists this command, and [`pyproject.toml`](../pyproject.toml) defines the `all` extra.

3. For core-only work, a lighter install is enough:

   ```bash
   python -m pip install -e .
   ```

   Core imports should require only Pydantic plus the standard library; [`tests/test_zero_deps.py`](../tests/test_zero_deps.py) enforces this.

4. If working from source without installation, use `PYTHONPATH=src`, as described in [`examples/README.md`](../examples/README.md).

## Run commands

The commands declared by the repository are in [`README.md`](../README.md) and [`pyproject.toml`](../pyproject.toml):

```bash
ruff check src tests
mypy
pytest
```

After installing the package, the local CLI can be smoke-tested with:

```bash
switchboard version
switchboard eval dogfood --routes 20 --no-baseline
```

Use targeted pytest modules while iterating:

- Core catalog/decision/audit: `pytest tests/test_route.py tests/test_registry.py tests/test_decision.py tests/test_audit.py`
- Prompt/schema/validation/policy: `pytest tests/test_prompt_schema.py tests/test_validate_confidence_policy.py`
- Shortlisting: `pytest tests/test_shortlist.py`
- Router/e2e: `pytest tests/test_router.py tests/test_smoke_integration.py`
- Eval harness: `pytest tests/test_evals.py`
- Optional-dependency hygiene: `pytest tests/test_zero_deps.py`

## Running examples

All examples are designed to run offline with no optional dependencies:

```bash
PYTHONPATH=src python examples/quickstart_byo.py
PYTHONPATH=src python examples/fastapi_app.py
PYTHONPATH=src python examples/langgraph_node.py
PYTHONPATH=src python examples/adk_agent.py
PYTHONPATH=src python examples/support_triage/demo.py
```

Evidence: [`examples/README.md`](../examples/README.md), [`quickstart_byo.py`](../examples/quickstart_byo.py), [`support_triage/demo.py`](../examples/support_triage/demo.py).

## Seeding and fixture workflows

There is no database seed workflow. Instead:

- Use `dogfood_registry()` and `dogfood_suite()` from [`fixtures.py`](../src/switchboard/evals/fixtures.py) for synthetic eval data.
- Use `registry_to_fixture()` / `registry_from_fixture()` to freeze and reload route catalogs.
- Use `ReplayCache` and `ReplayClient` from [`cache.py`](../src/switchboard/evals/cache.py) for deterministic provider-call replay.

## Common development workflows

### Add a new route

1. Define a Pydantic `args_model` if arguments are required.
2. Add a `Route` with a route name matching the grammar in [`ROUTE_NAME_PATTERN`](../src/switchboard/core/route.py).
3. Add clear `description`, `examples`, and `tags`; examples and descriptions influence prompt cards and retrieval.
4. If tenant-gated, use `requires` or `visibility`.
5. Add route/registry and router behavior tests.

### Add a provider adapter

1. Keep SDK imports lazy inside the adapter module.
2. Return `LLMResult` and `ClientCapabilities` exactly as defined in [`providers.base`](../src/switchboard/providers/base.py).
3. Map transport failures to `ProviderError` subclasses from [`errors.py`](../src/switchboard/errors.py).
4. Register the adapter in [`providers.__init__`](../src/switchboard/providers/__init__.py).
5. Extend zero-dependency import tests in [`tests/test_zero_deps.py`](../tests/test_zero_deps.py).

### Add a shortlister

1. Implement the [`Shortlister`](../src/switchboard/engine/shortlist.py) protocol.
2. Build over the full registry but score only `allowed` route names.
3. Return `ShortlistResult` and preserve pinned/weak retrieval semantics if using `_BaseShortlister`.
4. Add tests in [`tests/test_shortlist.py`](../tests/test_shortlist.py).

### Debug a routing miss

1. Inspect `decision.audit.shortlist`, `shortlist_skipped`, `weak_retrieval`, and `shuffle_seed`.
2. Check whether the gold route survived entitlement filtering.
3. Check `audit.abstain_reason`, `validation_retries`, and `confidence`.
4. Reconstruct prompt behavior using the same `registry_version` and candidate seed.
5. Use eval `CaseResult` fields from [`harness.py`](../src/switchboard/evals/harness.py) for retrieval gap vs decision gap.

## Repository-specific debugging tips

- A confidence score of `0.0` with `method="none"` is expected for BYO callables without logprobs; thresholds are inert unless signals are actionable. Evidence: [`signals_are_actionable`](../src/switchboard/engine/confidence.py).
- `Registry.view(ctx)` alone does not enforce v0.1 route security; the router's `filter_entitlements()` stage calls [`filter_routes`](../src/switchboard/engine/entitlements.py).
- A fallback decision is still `kind="route"`; inspect `decision.decision_path` and `decision.audit.abstain_reason` to see that fallback happened.
- A missing optional provider package should raise `MissingDependencyError` at `Router(...)`, not first request.
- If an async callable is used with sync `route()`, expect `ConfigError`; use `aroute()`.

## Related documents

- [Configuration](08-CONFIGURATION.md)
- [Features](06-FEATURES.md)
- [Extending](11-EXTENDING.md)
- [Technical debt](12-TECH-DEBT.md)
