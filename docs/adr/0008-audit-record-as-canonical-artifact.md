# ADR-0008: Audit Record as Canonical Artifact
Purpose: Record the inferred decision to use one audit model for logs, spans, and training projections.
Audience: Engineers changing telemetry, auditing, evals, or distillation.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

Routing decisions need observability, compliance/audit records, and future distillation data. Multiple schemas for the same decision would drift.

## Decision

Use `AuditRecord` as the canonical frozen artifact, then project it to OTel attributes, JSONL sink rows, and training examples.

## Consequences

- Every decision has one decision ID and registry version.
- Span attributes and audit logs share source data.
- Content capture can be enforced once through `apply_content_mode`.
- Future distillation can read the same records operators already collect.

## Evidence in code

- [`AuditRecord`](../../src/switchboard/core/audit.py) defines the canonical model.
- [`AuditDraft.finalize`](../../src/switchboard/core/audit.py) freezes it once.
- [`AuditRecord.as_otel_attributes`](../../src/switchboard/core/audit.py) projects to spans.
- [`AuditRecord.as_training_example`](../../src/switchboard/core/audit.py) projects to training rows.
- [`apply_content_mode`](../../src/switchboard/telemetry/emitter.py) gates payload content before sinks/spans consume the record.

## Related documents

- [Data model](../04-DATA-MODEL.md)
- [Operations](../10-OPERATIONS.md)
- [Technical debt](../12-TECH-DEBT.md)
