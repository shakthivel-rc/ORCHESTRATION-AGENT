# ADR-0002: Pydantic-Only Core and Lazy Optional Extras
Purpose: Record the inferred dependency strategy.
Audience: Maintainers adding dependencies or provider integrations.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

The core library defines typed models and protocols, while provider SDKs and OTel are optional. Tests assert optional dependencies are not imported from core paths.

## Decision

Keep the hard dependency set to Pydantic and the standard library; expose provider/telemetry functionality through optional extras imported lazily.

## Consequences

- `pip install switchboard` can route with a BYO callable.
- Missing extras fail at router/adaptor construction with typed `MissingDependencyError`.
- Contributors must not add SDK imports to core, engine, router, or package root.

## Evidence in code

- [`pyproject.toml`](../../pyproject.toml) declares only `pydantic>=2.7,<3` as a dependency.
- [`CallableAdapter`](../../src/switchboard/providers/callable_adapter.py) is the zero-extra provider path.
- [`resolve_client`](../../src/switchboard/providers/__init__.py) lazily imports adapters.
- [`OTelEmitter`](../../src/switchboard/telemetry/otel.py) lazily imports OpenTelemetry.
- [`tests/test_zero_deps.py`](../../tests/test_zero_deps.py) enforces import hygiene.

## Related documents

- [Tech stack](../03-TECH-STACK.md)
- [Configuration](../08-CONFIGURATION.md)
- [Technical debt](../12-TECH-DEBT.md)
