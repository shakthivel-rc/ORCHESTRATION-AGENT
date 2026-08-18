from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from switchboard import AuditRecord, ConfidenceReport, LatencyBlock
from switchboard.distill import collect_training_examples, iter_audit_jsonl, write_jsonl
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    i: int,
    *,
    text: str | None = "refund order 123",
    route: str = "refund",
    score: float = 0.9,
    outcome: str | None = None,
    error: str | None = None,
    decision_path: str = "llm",
) -> AuditRecord:
    now = datetime.now(timezone.utc)
    return AuditRecord(
        decision_id=f"decision-{i}",
        ts_start=now,
        ts_end=now,
        tenant_id="acme",
        inputs_hash=f"input-{i}",
        input_text=text,
        registry_version="abc123def456",
        kind="route",
        routes=[route],
        confidence=ConfidenceReport(score=score, method="logprobs"),
        decision_path=decision_path,  # type: ignore[arg-type]
        latency=LatencyBlock(total_ms=1.0),
        outcome=outcome,
        error=error,
    )


def test_collect_training_examples_filters_and_dedupes() -> None:
    records = [
        _record(1, score=0.9),
        _record(2, text=None),
        _record(3, score=0.2),
        _record(4, score=0.2, outcome="success"),
        _record(5, error="ProviderTimeout"),
        _record(6, decision_path="fallback"),
        _record(7, route="refund"),
    ]
    duplicate = _record(8)
    duplicate = duplicate.model_copy(update={"inputs_hash": "input-1"})

    examples, stats = collect_training_examples([*records, duplicate], per_label_cap=2)

    assert [example.decision_id for example in examples] == ["decision-1", "decision-4"]
    assert stats.seen == 8
    assert stats.emitted == 2
    assert stats.skipped_no_content == 1
    assert stats.skipped_low_quality == 1
    assert stats.skipped_error == 1
    assert stats.skipped_not_route == 1
    assert stats.skipped_duplicate == 1
    assert stats.skipped_cap == 1


def test_write_jsonl_and_iter_audit_jsonl(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    record = _record(1)
    audit_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    loaded = list(iter_audit_jsonl(audit_path))
    assert loaded == [record]

    out = tmp_path / "training.jsonl"
    stats = write_jsonl(loaded, out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert stats.emitted == 1
    assert rows[0]["label"] == "refund"
    assert rows[0]["decision_id"] == "decision-1"


def test_iter_audit_jsonl_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid JSON"):
        list(iter_audit_jsonl(path))
