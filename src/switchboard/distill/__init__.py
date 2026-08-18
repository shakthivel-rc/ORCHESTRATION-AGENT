"""Audit-log distillation helpers (plan §8.5).

The training row still comes from :meth:`AuditRecord.as_training_example`; this
module implements the dataset-builder rules that method deliberately does not
own: only successful/high-confidence LLM route decisions, no provider errors,
dedupe by ``inputs_hash``, optional per-label caps, and JSONL export.

The implementation is Pydantic + stdlib only. The heavier ``[distill]`` /
``[distill-train]`` extras remain available for future Parquet export and model
training, but producing a safe JSONL dataset from audit records does not require
them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from switchboard.core.audit import AuditRecord
from switchboard.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "DistillStats",
    "TrainingExample",
    "collect_training_examples",
    "iter_audit_jsonl",
    "iter_training_examples",
    "write_jsonl",
]


class TrainingExample(BaseModel):
    """One distillation row derived from an :class:`AuditRecord`."""

    model_config = ConfigDict(frozen=True)

    text: str
    label: str
    registry_version: str
    tenant: str | None = None
    weight: float = 1.0
    decision_id: str
    inputs_hash: str


class DistillStats(BaseModel):
    """Counters from one dataset build."""

    model_config = ConfigDict(frozen=True)

    seen: int = 0
    eligible: int = 0
    emitted: int = 0
    skipped_no_content: int = 0
    skipped_not_route: int = 0
    skipped_low_quality: int = 0
    skipped_error: int = 0
    skipped_duplicate: int = 0
    skipped_cap: int = 0


class _Builder:
    def __init__(
        self,
        *,
        min_confidence: float,
        require_success_or_confidence: bool,
        per_label_cap: int | None,
        dedupe: bool,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ConfigError(f"min_confidence must be in [0, 1], got {min_confidence!r}")
        if per_label_cap is not None and per_label_cap < 1:
            raise ConfigError(f"per_label_cap must be >= 1, got {per_label_cap!r}")
        self.min_confidence = min_confidence
        self.require_success_or_confidence = require_success_or_confidence
        self.per_label_cap = per_label_cap
        self.dedupe = dedupe
        self.seen_hashes: set[str] = set()
        self.label_counts: Counter[str] = Counter()
        self.stats = Counter[str]()

    def consider(self, record: AuditRecord) -> TrainingExample | None:
        self.stats["seen"] += 1
        row = record.as_training_example()
        if not row:
            self.stats["skipped_no_content"] += 1
            return None
        if record.error is not None:
            self.stats["skipped_error"] += 1
            return None
        if record.kind != "route" or not record.routes:
            self.stats["skipped_not_route"] += 1
            return None
        if record.decision_path != "llm":
            self.stats["skipped_not_route"] += 1
            return None
        if self.require_success_or_confidence and not _quality_passes(record, self.min_confidence):
            self.stats["skipped_low_quality"] += 1
            return None
        if self.dedupe and record.inputs_hash in self.seen_hashes:
            self.stats["skipped_duplicate"] += 1
            return None
        label = str(row["label"])
        if self.per_label_cap is not None and self.label_counts[label] >= self.per_label_cap:
            self.stats["skipped_cap"] += 1
            return None

        self.stats["eligible"] += 1
        self.stats["emitted"] += 1
        self.seen_hashes.add(record.inputs_hash)
        self.label_counts[label] += 1
        return TrainingExample(
            text=str(row["text"]),
            label=label,
            registry_version=str(row["registry_version"]),
            tenant=row.get("tenant") if isinstance(row.get("tenant"), str) else None,
            weight=float(row.get("weight", 1.0)),
            decision_id=record.decision_id,
            inputs_hash=record.inputs_hash,
        )

    def snapshot(self) -> DistillStats:
        return DistillStats(**{field: self.stats[field] for field in DistillStats.model_fields})


def _quality_passes(record: AuditRecord, min_confidence: float) -> bool:
    """Dataset eligibility: downstream success or high-confidence LLM route."""
    if record.outcome == "success":
        return True
    return record.confidence is not None and record.confidence.score >= min_confidence


def iter_training_examples(
    records: Iterable[AuditRecord],
    *,
    min_confidence: float = 0.8,
    require_success_or_confidence: bool = True,
    per_label_cap: int | None = None,
    dedupe: bool = True,
) -> Iterator[TrainingExample]:
    """Yield eligible distillation rows from ``records``.

    This streaming helper returns examples only. Use
    :func:`collect_training_examples` or :func:`write_jsonl` when counters are
    needed.
    """
    builder = _Builder(
        min_confidence=min_confidence,
        require_success_or_confidence=require_success_or_confidence,
        per_label_cap=per_label_cap,
        dedupe=dedupe,
    )
    for record in records:
        example = builder.consider(record)
        if example is not None:
            yield example


def collect_training_examples(
    records: Iterable[AuditRecord],
    *,
    min_confidence: float = 0.8,
    require_success_or_confidence: bool = True,
    per_label_cap: int | None = None,
    dedupe: bool = True,
) -> tuple[list[TrainingExample], DistillStats]:
    """Return ``(examples, stats)`` for an in-memory build."""
    builder = _Builder(
        min_confidence=min_confidence,
        require_success_or_confidence=require_success_or_confidence,
        per_label_cap=per_label_cap,
        dedupe=dedupe,
    )
    examples: list[TrainingExample] = []
    for record in records:
        example = builder.consider(record)
        if example is not None:
            examples.append(example)
    return examples, builder.snapshot()


def write_jsonl(
    records: Iterable[AuditRecord],
    path: str | Path,
    *,
    min_confidence: float = 0.8,
    require_success_or_confidence: bool = True,
    per_label_cap: int | None = None,
    dedupe: bool = True,
) -> DistillStats:
    """Write eligible training examples as JSONL and return build counters."""
    builder = _Builder(
        min_confidence=min_confidence,
        require_success_or_confidence=require_success_or_confidence,
        per_label_cap=per_label_cap,
        dedupe=dedupe,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            example = builder.consider(record)
            if example is not None:
                fh.write(example.model_dump_json() + "\n")
    return builder.snapshot()


def iter_audit_jsonl(path: str | Path) -> Iterator[AuditRecord]:
    """Load :class:`AuditRecord` values from a JSONL sink file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload: Any = json.loads(stripped)
            except ValueError as exc:
                raise ConfigError(f"{source}:{number}: invalid JSON: {exc}") from exc
            try:
                yield AuditRecord.model_validate(payload)
            except Exception as exc:
                raise ConfigError(
                    f"{source}:{number}: line is not an AuditRecord: {type(exc).__name__}: {exc}"
                ) from exc
