from __future__ import annotations

import threading
from datetime import datetime, timezone

from switchboard import AuditRecord, InMemorySink, LatencyBlock, QueuedSink
from switchboard.telemetry.emitter import BaseSink


def _record(i: int) -> AuditRecord:
    now = datetime.now(timezone.utc)
    return AuditRecord(
        decision_id=f"decision-{i}",
        ts_start=now,
        ts_end=now,
        inputs_hash=f"input-{i}",
        registry_version="abc123def456",
        kind="abstain",
        latency=LatencyBlock(total_ms=0.0),
    )


class _BlockingSink(BaseSink):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.records: list[AuditRecord] = []

    def emit(self, record: AuditRecord) -> None:
        self.started.set()
        self.release.wait(timeout=5)
        self.records.append(record)


def test_queued_sink_drains_on_close() -> None:
    inner = InMemorySink(maxlen=None)
    sink = QueuedSink(inner, maxlen=10)

    sink.emit(_record(1))
    sink.emit(_record(2))
    sink.close()

    assert [record.decision_id for record in inner.records] == ["decision-1", "decision-2"]
    assert sink.closed
    assert sink.queued == 0
    assert sink.dropped == 0


def test_queued_sink_drop_oldest_counts_drops() -> None:
    inner = _BlockingSink()
    sink = QueuedSink(inner, maxlen=2, drop_oldest=True)

    sink.emit(_record(1))
    assert inner.started.wait(timeout=5)

    sink.emit(_record(2))
    sink.emit(_record(3))
    sink.emit(_record(4))

    inner.release.set()
    sink.close()

    assert [record.decision_id for record in inner.records] == [
        "decision-1",
        "decision-3",
        "decision-4",
    ]
    assert sink.dropped == 1

