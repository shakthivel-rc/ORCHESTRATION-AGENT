# Switchboard Documentation
Purpose: Index the documentation set, recommended reading paths, and all open questions.
Audience: New engineers, operators, contributors, reviewers, and AI agents.
Last verified against commit not-a-git-repository on 2026-08-07.

## Documentation index

| File | Purpose | Intended audience |
|---|---|---|
| [01-OVERVIEW.md](01-OVERVIEW.md) | Purpose, users, context, boundaries, and README/plan contradictions. | New engineers, technical leads. |
| [02-ARCHITECTURE.md](02-ARCHITECTURE.md) | Components, runtime topology, request flows, and cross-cutting concerns. | Engineers changing core behavior. |
| [03-TECH-STACK.md](03-TECH-STACK.md) | Languages, dependencies, extras, tools, and managed-service posture. | Setup owners, dependency reviewers. |
| [04-DATA-MODEL.md](04-DATA-MODEL.md) | Canonical entities, fields, relationships, storage, schemas, and lifecycle. | Model/schema contributors. |
| [05-BUSINESS-LOGIC.md](05-BUSINESS-LOGIC.md) | Enforced rules, validations, state diagrams, and defaults. | Routing/policy contributors. |
| [06-FEATURES.md](06-FEATURES.md) | Feature inventory with entry points, logic locations, dependencies, and coverage. | Product/engineering planners. |
| [07-API-CONTRACTS.md](07-API-CONTRACTS.md) | Public interfaces, request/response shapes, errors, and breaking surfaces. | Integrators and maintainers. |
| [08-CONFIGURATION.md](08-CONFIGURATION.md) | Environment variables, Router keys, DSLs, and project config. | Operators and app integrators. |
| [09-DEVELOPMENT.md](09-DEVELOPMENT.md) | Local setup, run/test commands, workflows, and debugging tips. | Contributors and AI agents. |
| [10-OPERATIONS.md](10-OPERATIONS.md) | Deploy/rollback posture, health checks, observability, and runbook notes. | Operators and SREs. |
| [11-EXTENDING.md](11-EXTENDING.md) | Extension points, recipes, conventions, and fragile areas. | Contributors adding functionality. |
| [12-TECH-DEBT.md](12-TECH-DEBT.md) | Ranked risks, stubs, missing tests, and remediation sketches. | Maintainers and roadmap owners. |
| [13-GLOSSARY.md](13-GLOSSARY.md) | Plain-language definitions for domain terms and internal jargon. | Everyone new to the project. |
| [adr/README.md](adr/README.md) | ADR index. | Maintainers and reviewers. |
| [adr/0001-framework-agnostic-decision-layer.md](adr/0001-framework-agnostic-decision-layer.md) | Reconstructed decision: decide but never execute. | Maintainers. |
| [adr/0002-pydantic-only-core-lazy-extras.md](adr/0002-pydantic-only-core-lazy-extras.md) | Reconstructed decision: Pydantic-only core and lazy extras. | Dependency contributors. |
| [adr/0003-content-addressed-immutable-registry.md](adr/0003-content-addressed-immutable-registry.md) | Reconstructed decision: immutable content-addressed registry. | Core contributors. |
| [adr/0004-single-loop-sync-async-parity.md](adr/0004-single-loop-sync-async-parity.md) | Reconstructed decision: one sync/async loop. | Router contributors. |
| [adr/0005-llm-first-shortlist-before-decide.md](adr/0005-llm-first-shortlist-before-decide.md) | Reconstructed decision: shortlist before LLM decision. | Retrieval contributors. |
| [adr/0006-segmented-prompt-and-dynamic-wire-schema.md](adr/0006-segmented-prompt-and-dynamic-wire-schema.md) | Reconstructed decision: segmented prompts and wire schema. | Prompt/schema contributors. |
| [adr/0007-degrade-model-failures-raise-infrastructure-failures.md](adr/0007-degrade-model-failures-raise-infrastructure-failures.md) | Reconstructed decision: degrade model failures, raise infra failures. | Error-handling contributors. |
| [adr/0008-audit-record-as-canonical-artifact.md](adr/0008-audit-record-as-canonical-artifact.md) | Reconstructed decision: one audit artifact for logs/spans/training. | Telemetry/eval contributors. |

## Reading paths

### New engineer, day one

1. [Overview](01-OVERVIEW.md)
2. [Architecture](02-ARCHITECTURE.md)
3. [Data model](04-DATA-MODEL.md)
4. [Business logic](05-BUSINESS-LOGIC.md)
5. [Glossary](13-GLOSSARY.md)

### Operator on call

1. [Operations](10-OPERATIONS.md)
2. [Configuration](08-CONFIGURATION.md)
3. [API contracts](07-API-CONTRACTS.md)
4. [Data model](04-DATA-MODEL.md) for `AuditRecord`
5. [Technical debt](12-TECH-DEBT.md)

### Contributor adding a feature

1. [Development](09-DEVELOPMENT.md)
2. [Extending](11-EXTENDING.md)
3. [Features](06-FEATURES.md)
4. [Business logic](05-BUSINESS-LOGIC.md)
5. [ADR index](adr/README.md)

## Open questions register

| Source | OPEN QUESTION |
|---|---|
| [04-DATA-MODEL.md](04-DATA-MODEL.md) | OPEN QUESTION: No repository file defines production retention, rotation, archival, or deletion policies for JSONL audit logs and replay caches. |
| [08-CONFIGURATION.md](08-CONFIGURATION.md) | OPEN QUESTION: The repository does not document provider-specific credential environment variables or secret-source conventions for Instructor, LiteLLM, OpenAI, Anthropic, Gemini, or Bedrock deployments. |

## Verification note

This checkout is not a Git repository, so the required verification line uses `not-a-git-repository` instead of a commit SHA. This was verified with `git rev-parse` during documentation generation.

## Related documents

- [Overview](01-OVERVIEW.md)
- [Architecture](02-ARCHITECTURE.md)
- [ADR index](adr/README.md)
