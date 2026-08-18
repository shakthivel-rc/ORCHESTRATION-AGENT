# Architecture Decision Records
Purpose: Index reconstructed ADRs inferred from the current codebase.
Audience: Maintainers, reviewers, and contributors who need design rationale.
Last verified against commit not-a-git-repository on 2026-08-07.

## ADR index

| ADR | Status | Decision |
|---|---|---|
| [ADR-0001](0001-framework-agnostic-decision-layer.md) | Reconstructed | Keep Switchboard as a framework-agnostic decision layer that never executes handlers. |
| [ADR-0002](0002-pydantic-only-core-lazy-extras.md) | Reconstructed | Keep core Pydantic-only and import optional extras lazily. |
| [ADR-0003](0003-content-addressed-immutable-registry.md) | Reconstructed | Use immutable route catalogs and content-addressed registry versions. |
| [ADR-0004](0004-single-loop-sync-async-parity.md) | Reconstructed | Share one loop pipeline across sync and async routing APIs. |
| [ADR-0005](0005-llm-first-shortlist-before-decide.md) | Reconstructed | Retrieve candidates before the LLM, but keep the LLM/policy as final decider. |
| [ADR-0006](0006-segmented-prompt-and-dynamic-wire-schema.md) | Reconstructed | Use segmented prompts and per-call wire schemas. |
| [ADR-0007](0007-degrade-model-failures-raise-infrastructure-failures.md) | Reconstructed | Degrade model-output failures to decisions and raise infrastructure/config failures. |
| [ADR-0008](0008-audit-record-as-canonical-artifact.md) | Reconstructed | Use `AuditRecord` as the single source for audit, telemetry, and training projections. |

## Related documents

- [Architecture](../02-ARCHITECTURE.md)
- [Technical debt](../12-TECH-DEBT.md)
- [Extending](../11-EXTENDING.md)
