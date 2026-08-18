from __future__ import annotations

from datetime import datetime, timezone

from switchboard import AuditRecord, CostBlock, LatencyBlock, Usage
from switchboard.telemetry.otel import OTelEmitter


class _Instrument:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | int, dict[str, object]]] = []

    def record(self, value: float | int, attrs: dict[str, object]) -> None:
        self.calls.append(("record", value, attrs))

    def add(self, value: int, attrs: dict[str, object]) -> None:
        self.calls.append(("add", value, attrs))


class _Meter:
    def __init__(self) -> None:
        self.histograms: dict[str, _Instrument] = {}
        self.counters: dict[str, _Instrument] = {}

    def create_histogram(self, name: str, **kwargs: object) -> _Instrument:
        del kwargs
        instrument = _Instrument()
        self.histograms[name] = instrument
        return instrument

    def create_counter(self, name: str, **kwargs: object) -> _Instrument:
        del kwargs
        instrument = _Instrument()
        self.counters[name] = instrument
        return instrument


def test_otel_emitter_records_metrics_from_audit_record() -> None:
    meter = _Meter()
    emitter = OTelEmitter(enabled=False)
    emitter.available = True
    emitter._meter = meter
    now = datetime.now(timezone.utc)
    record = AuditRecord(
        decision_id="decision-1",
        ts_start=now,
        ts_end=now,
        inputs_hash="input",
        registry_version="abc123def456",
        kind="route",
        routes=["refund"],
        provider="openai",
        request_model="gpt-5-nano",
        usage=Usage(input_tokens=10, cached_input_tokens=3, output_tokens=4),
        cost=CostBlock(usd=0.01, price_table_version="test"),
        latency=LatencyBlock(total_ms=250.0),
    )

    emitter.record_metrics(record)

    assert meter.counters["switchboard.decision.count"].calls[0][1] == 1
    assert meter.histograms["gen_ai.client.operation.duration"].calls[0][1] == 0.25
    assert [call[1] for call in meter.histograms["gen_ai.client.token.usage"].calls] == [10, 3, 4]
    assert meter.histograms["switchboard.cost.usd"].calls[0][1] == 0.01

