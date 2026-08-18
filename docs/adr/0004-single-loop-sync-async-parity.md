# ADR-0004: Single Loop with Sync/Async Parity
Purpose: Record the inferred decision to share one decision pipeline across synchronous and asynchronous APIs.
Audience: Contributors changing router flow or async behavior.
Last verified against commit not-a-git-repository on 2026-08-07.

## Status

Reconstructed.

## Context

The library offers `route()` and `aroute()` and must keep behavior consistent across both. Duplicating the pipeline would let sync and async semantics drift.

## Decision

Implement the pipeline as shared `LoopState` stages in `engine.loop`; sync and async drivers differ only where I/O requires it.

## Consequences

- Routing behavior stays structurally aligned.
- Sync-only clients can be offloaded from async code.
- Async-only clients under sync `route()` are configuration errors.

## Evidence in code

- [`Router.route`](../../src/switchboard/router.py) and [`Router.aroute`](../../src/switchboard/router.py) delegate to shared pipeline helpers.
- [`run_pipeline_sync`](../../src/switchboard/engine/loop.py) and [`run_pipeline_async`](../../src/switchboard/engine/loop.py) own the orchestration sequence.
- [`run_decision_sync`](../../src/switchboard/engine/loop.py) and [`run_decision_async`](../../src/switchboard/engine/loop.py) mirror provider call/repair logic.
- [`_timed_call_async`](../../src/switchboard/engine/loop.py) uses `asyncio.to_thread` for sync-only clients.
- [`tests/test_smoke_integration.py`](../../tests/test_smoke_integration.py) includes sync/async agreement coverage.

## Related documents

- [Architecture](../02-ARCHITECTURE.md)
- [Business logic](../05-BUSINESS-LOGIC.md)
- [Development](../09-DEVELOPMENT.md)
