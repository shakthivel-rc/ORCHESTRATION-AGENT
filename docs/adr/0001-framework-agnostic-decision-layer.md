# ADR-0001: Framework-Agnostic Decision Layer
Purpose: Record the inferred decision to keep Switchboard as a decision layer rather than an executor or framework.
Audience: Maintainers and contributors evaluating boundary changes.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

The package exposes route selection primitives and examples for several frameworks, but no framework dependency is imported by core package code. `Route` carries metadata but no handler field, and integrations dispatch in caller code.

## Decision

Switchboard decides which route should handle a request and returns a typed `Decision`; it never executes the selected handler.

## Consequences

- The package can be embedded in FastAPI, LangGraph, ADK, or custom runtimes.
- A routing compromise is separated from code execution authority.
- Applications must maintain their own dispatch tables and handler authorization.

## Evidence in code

- [`Route`](../../src/switchboard/core/route.py) documents that a route carries no handler.
- [`Router.route`](../../src/switchboard/router.py) returns a `Decision`.
- [`examples/fastapi_app.py`](../../examples/fastapi_app.py) dispatches through `HANDLERS`.
- [`OTelEmitter.execute_tool_span`](../../src/switchboard/telemetry/otel.py) is caller-opened around execution.

## Related documents

- [Overview](../01-OVERVIEW.md)
- [Architecture](../02-ARCHITECTURE.md)
- [API contracts](../07-API-CONTRACTS.md)
