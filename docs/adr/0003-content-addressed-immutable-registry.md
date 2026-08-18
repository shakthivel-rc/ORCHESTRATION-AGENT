# ADR-0003: Content-Addressed Immutable Registry
Purpose: Record the inferred decision to make catalogs immutable and hash-addressed.
Audience: Engineers changing route, registry, cache, or eval behavior.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

Prompt caches, shortlist indexes, audit records, and eval fixtures all need a stable catalog identity. Mutable route catalogs would make cached prompts and indexes stale without a reliable invalidation point.

## Decision

Make `Route` and `Registry` immutable and derive a registry version from route content hashes.

## Consequences

- Route edits automatically change `registry.version`.
- Prompt cache keys, shortlist index keys, and audit records can refer to the same catalog identity.
- Composition returns new registries rather than mutating existing ones.

## Evidence in code

- [`Route.content_hash`](../../src/switchboard/core/route.py) hashes semantic route fields.
- [`Registry.content_hash`](../../src/switchboard/core/registry.py) hashes sorted route hashes.
- [`Registry.version`](../../src/switchboard/core/registry.py) truncates the digest to 12 hex chars.
- [`segment_b_cache_key`](../../src/switchboard/engine/prompt.py) uses registry version.
- [`_BaseShortlister.index_key`](../../src/switchboard/engine/shortlist.py) uses registry version.

## Related documents

- [Data model](../04-DATA-MODEL.md)
- [Business logic](../05-BUSINESS-LOGIC.md)
- [Extending](../11-EXTENDING.md)
