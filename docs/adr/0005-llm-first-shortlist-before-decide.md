# ADR-0005: LLM-First Shortlist-Before-Decide
Purpose: Record the inferred decision to use retrieval as a candidate reducer, not as the final router.
Audience: Engineers changing retrieval, prompt construction, or policy behavior.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

Large catalogs are expensive and degrade tool selection, but pure retrieval can miss intent and should not be the final authority.

## Decision

Run entitlement filtering and optional shortlisting before the LLM, but let the LLM make the final typed decision from the candidate set.

## Consequences

- Small catalogs bypass retrieval and keep a stable prompt prefix.
- Large catalogs use BM25 or embeddings to reduce prompt size.
- Router re-checks entitlements and route references after retrieval.
- Weak retrieval biases toward clarification rather than forced commitment.

## Evidence in code

- [`AutoShortlister`](../../src/switchboard/engine/shortlist.py) bypasses below threshold and delegates above it.
- [`BM25Shortlister`](../../src/switchboard/engine/shortlist.py) is the default lexical backend.
- [`EmbeddingShortlister`](../../src/switchboard/engine/shortlist.py) supports dense retrieval.
- [`_absorb_shortlist`](../../src/switchboard/engine/loop.py) re-intersects with entitlements.
- [`resolve_decision`](../../src/switchboard/engine/policy.py) remains the final policy stage.

## Related documents

- [Architecture](../02-ARCHITECTURE.md)
- [Business logic](../05-BUSINESS-LOGIC.md)
- [Features](../06-FEATURES.md)
