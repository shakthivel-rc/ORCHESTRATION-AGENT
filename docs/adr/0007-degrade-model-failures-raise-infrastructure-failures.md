# ADR-0007: Degrade Model Failures, Raise Infrastructure Failures
Purpose: Record the inferred error-handling decision.
Audience: Engineers changing validation, retries, providers, or API error behavior.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

Routing systems must distinguish an uncertain or malformed model answer from broken caller configuration or provider infrastructure. Treating all failures as exceptions makes ordinary "cannot route" outcomes hard to handle; treating all failures as decisions can hide outages.

## Decision

Degrade healthy-system model failures to typed `Decision` outcomes, but raise broken configuration and provider transport failures by default.

## Consequences

- Clarify and abstain are first-class outcomes.
- Bad JSON and invalid route references can be audited and repaired.
- Provider outages are visible unless explicitly configured to abstain.
- Tests and consumers can switch on closed `AbstainReason` values.

## Evidence in code

- [`errors.py`](../../src/switchboard/errors.py) states the raise-vs-degrade rule.
- [`parse_wire_output`](../../src/switchboard/engine/validate.py) returns `ValidationError` values, not exceptions.
- [`resolve_policy`](../../src/switchboard/engine/loop.py) maps validation failures to decisions.
- [`_handle_provider_error`](../../src/switchboard/engine/loop.py) raises by default.
- [`AbstainReason`](../../src/switchboard/core/audit.py) defines the closed reason vocabulary.

## Related documents

- [Business logic](../05-BUSINESS-LOGIC.md)
- [API contracts](../07-API-CONTRACTS.md)
- [Operations](../10-OPERATIONS.md)
