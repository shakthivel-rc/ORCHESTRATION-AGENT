# ADR-0006: Segmented Prompt and Dynamic Wire Schema
Purpose: Record the inferred decision to separate prompt cache segments from per-call output schemas.
Audience: Engineers changing prompts, schema generation, or provider adapters.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

Routing quality depends on stable instructions, route descriptions, candidate lists, and untrusted user input. Provider cache behavior and route-name enforcement differ by structured-output capability rung.

## Decision

Assemble prompts as stable-to-variable segments with user input last, and build a per-call wire schema whose route enum is dynamic only when the provider capability rung supports it without breaking cache assumptions.

## Consequences

- Prompt cache keys are stable for system and route-directory segments.
- User input is isolated in the last `user` segment.
- The wire schema is not the public `Decision` model.
- Static schema mode relies on validator route-reference checks.

## Evidence in code

- [`build_segments`](../../src/switchboard/engine/prompt.py) assembles A/B/C/D segments.
- [`_render_query`](../../src/switchboard/engine/prompt.py) tags untrusted input.
- [`build_wire_schema`](../../src/switchboard/engine/schema.py) creates per-call schema.
- [`resolve_schema_mode`](../../src/switchboard/engine/schema.py) chooses dynamic vs static mode.
- [`check_route_reference`](../../src/switchboard/engine/validate.py) enforces membership after parsing.

## Related documents

- [Architecture](../02-ARCHITECTURE.md)
- [Business logic](../05-BUSINESS-LOGIC.md)
- [API contracts](../07-API-CONTRACTS.md)
