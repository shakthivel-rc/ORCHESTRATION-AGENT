"""Contract tests for the audit layer (plan §8.2, §8.1, §8.5, §7.3).

``AuditRecord`` is one artifact doing three jobs — OTel span source, compliance
row, distillation example — so everything here is load-bearing somewhere:

* :func:`canonical_json` / :func:`sha256_hex` are the codebase's **only**
  hashing pair. Route content hashes, ``registry_version``, ``view_hash``,
  ``inputs_hash``, ``args_hash`` and the seeded shuffle all bottom out here, so
  "identical inputs dedupe across processes" (§8.2) is a property of these two
  functions and of nothing else.
* :func:`new_decision_id` must produce real ULIDs, because §8.2 specifies a
  sortable id purely so JSONL sinks are range-scannable without an index.
* :class:`AuditDraft` is the "tap, not a stage" accumulator (§2.1): stages write
  the slice they own in any order, and :meth:`AuditDraft.finalize` freezes it
  exactly once — a router that finalizes in a ``finally`` block must never emit
  two different records for one decision.
* :meth:`AuditRecord.as_otel_attributes` must not invent ``gen_ai.*`` keys
  (§8.1) and must not leak content unless explicitly told to (§7.4).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel

from switchboard import Candidate, ConfidenceReport, CostBlock, LatencyBlock, Usage
from switchboard.core.audit import (
    AuditDraft,
    AuditRecord,
    canonical_json,
    new_decision_id,
    sha256_hex,
)

CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


# --------------------------------------------------------------------------- #
# canonical_json (plan §8.2)
# --------------------------------------------------------------------------- #


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json([1, 2, 3]) == "[1,2,3]"


def test_canonical_json_is_key_order_independent_at_every_depth() -> None:
    """This is the whole dedup guarantee: two processes that build the same

    mapping in a different insertion order must produce identical bytes.
    """
    first = {"outer": {"z": 1, "a": {"n": 2, "m": 3}}, "list": [{"b": 1, "a": 2}]}
    second = {"list": [{"a": 2, "b": 1}], "outer": {"a": {"m": 3, "n": 2}, "z": 1}}
    assert canonical_json(first) == canonical_json(second)


def test_canonical_json_preserves_list_order_because_a_list_is_ordered_data() -> None:
    assert canonical_json([1, 2]) != canonical_json([2, 1])


def test_canonical_json_keeps_non_ascii_intact() -> None:
    """``ensure_ascii=False``: a redacted query in any language hashes as itself."""
    assert canonical_json({"q": "où est mon colis"}) == '{"q":"où est mon colis"}'


def test_canonical_json_sorts_sets_so_iteration_order_cannot_leak_in() -> None:
    assert canonical_json(frozenset({"b", "a", "c"})) == '["a","b","c"]'
    assert canonical_json({"b", "a"}) == canonical_json({"a", "b"})


def test_canonical_json_encodes_pydantic_models_in_json_mode() -> None:
    class Args(BaseModel):
        order_id: str
        when: datetime

    when = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    assert canonical_json(Args(order_id="A-1", when=when)) == (
        '{"order_id":"A-1","when":"2026-08-07T12:00:00Z"}'
    )


def test_canonical_json_encodes_datetimes_and_bytes() -> None:
    when = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    assert canonical_json(when) == '"2026-08-07T12:00:00+00:00"'
    assert canonical_json(b"hello") == '"hello"'


def test_canonical_json_encodes_a_class_by_its_qualified_name() -> None:
    class Args(BaseModel):
        pass

    assert canonical_json(Args) == f'"{Args.__module__}.{Args.__qualname__}"'


def test_canonical_json_never_falls_back_to_repr_for_unknown_objects() -> None:
    """A ``repr`` embeds a memory address, which would make every derived hash

    process-dependent and silently defeat cross-process dedup (§7.3).
    """

    class Opaque:
        pass

    encoded = canonical_json(Opaque())
    assert "0x" not in encoded
    assert encoded == canonical_json(Opaque())
    assert "Opaque" in encoded


def test_canonical_json_is_total_and_never_raises_inside_the_loop() -> None:
    payload = {"handler": lambda: None, "conn": object(), "ok": 1}
    assert json.loads(canonical_json(payload))["ok"] == 1


# --------------------------------------------------------------------------- #
# sha256_hex (plan §7.3, §8.2)
# --------------------------------------------------------------------------- #


def test_sha256_hex_matches_the_stdlib_for_a_single_string() -> None:
    assert sha256_hex("hello") == hashlib.sha256(b"hello").hexdigest()
    assert len(sha256_hex("hello")) == 64


def test_sha256_hex_hashes_bytes_verbatim() -> None:
    assert sha256_hex(b"hello") == hashlib.sha256(b"hello").hexdigest()
    assert sha256_hex("héllo") == hashlib.sha256("héllo".encode()).hexdigest()


def test_sha256_hex_separates_parts_unambiguously() -> None:
    """Without a delimiter, ``("ab", "c")`` and ``("a", "bc")`` would collide —

    which is exactly how a two-part cache key (registry hash + entitlement key)
    would start aliasing across tenants.
    """
    assert sha256_hex("ab", "c") != sha256_hex("a", "bc")
    assert sha256_hex("ab", "c") != sha256_hex("abc")


def test_sha256_hex_canonicalises_non_string_parts() -> None:
    assert sha256_hex({"a": 1, "b": 2}) == sha256_hex(canonical_json({"a": 1, "b": 2}))
    assert sha256_hex({"a": 1, "b": 2}) == sha256_hex({"b": 2, "a": 1})


def test_sha256_hex_is_deterministic_and_order_sensitive() -> None:
    assert sha256_hex("a", "b") == sha256_hex("a", "b")
    assert sha256_hex("a", "b") != sha256_hex("b", "a")


def test_sha256_hex_of_no_parts_is_the_empty_digest() -> None:
    assert sha256_hex() == hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------------------- #
# new_decision_id — ULID (plan §8.2)
# --------------------------------------------------------------------------- #


def test_decision_id_is_26_crockford_base32_chars() -> None:
    for _ in range(200):
        ulid = new_decision_id()
        assert len(ulid) == 26
        assert set(ulid) <= set(CROCKFORD32)


def test_decision_ids_are_unique_across_a_tight_loop() -> None:
    """80 bits of randomness per millisecond: a tight loop is the worst case,

    and it still must not collide.
    """
    ids = [new_decision_id() for _ in range(5000)]
    assert len(set(ids)) == 5000


def test_decision_ids_generated_in_order_never_sort_backwards() -> None:
    """The timestamp occupies the high bits, so the 10-char prefix is

    non-decreasing for ids generated in sequence — that is what makes a JSONL
    sink range-scannable by time (§8.2).
    """
    prefixes = [new_decision_id()[:10] for _ in range(2000)]
    assert prefixes == sorted(prefixes)


def test_decision_ids_sort_strictly_by_generation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the clock pinned, sortability is exact rather than probabilistic."""
    clock = {"ms": 1_700_000_000_000}
    monkeypatch.setattr(time, "time", lambda: clock["ms"] / 1000.0)

    ids = []
    for _ in range(50):
        ids.append(new_decision_id())
        clock["ms"] += 1

    assert ids == sorted(ids)
    assert len(set(ids)) == 50


def test_decision_id_encodes_the_millisecond_timestamp_in_its_first_ten_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned_ms = 1_754_500_000_123
    monkeypatch.setattr(time, "time", lambda: pinned_ms / 1000.0)

    ulid = new_decision_id()
    decoded = 0
    for char in ulid[:10]:
        decoded = decoded * 32 + CROCKFORD32.index(char)
    assert decoded == pinned_ms


def test_decision_ids_from_the_same_millisecond_still_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    ids = {new_decision_id() for _ in range(500)}
    assert len(ids) == 500
    assert len({ulid[:10] for ulid in ids}) == 1  # identical timestamp component


# --------------------------------------------------------------------------- #
# AuditDraft accumulation -> finalize (plan §2.1, §8.2)
# --------------------------------------------------------------------------- #


def test_a_bare_draft_finalizes_into_a_valid_record() -> None:
    """Every field is optional and order-independent: a draft must never reject a

    partial state, because the router finalizes in a ``finally`` block.
    """
    record = AuditDraft().finalize()
    assert isinstance(record, AuditRecord)
    assert record.schema_version == "1"
    assert len(record.decision_id) == 26
    assert record.kind == "abstain"
    assert record.decision_path == "llm"
    assert record.inputs_hash == sha256_hex("")
    assert record.usage == Usage()
    assert isinstance(record.latency, LatencyBlock)
    assert record.latency.total_ms >= 0.0


def test_draft_timestamps_are_timezone_aware_utc() -> None:
    record = AuditDraft().finalize()
    assert record.ts_start.tzinfo is not None
    assert record.ts_end.tzinfo is not None
    assert record.ts_end >= record.ts_start


def test_set_inputs_hashes_query_plus_context_deterministically() -> None:
    a, b = AuditDraft(), AuditDraft()
    a.set_inputs("where is my package", {"locale": "en-GB", "tenant": "acme"})
    b.set_inputs("where is my package", {"tenant": "acme", "locale": "en-GB"})
    assert a.inputs_hash == b.inputs_hash
    assert a.inputs_hash is not None
    assert len(a.inputs_hash) == 64


def test_set_inputs_hash_separates_different_queries_and_contexts() -> None:
    base, other_query, other_ctx, no_ctx = (AuditDraft() for _ in range(4))
    base.set_inputs("q", {"locale": "en"})
    other_query.set_inputs("q2", {"locale": "en"})
    other_ctx.set_inputs("q", {"locale": "fr"})
    no_ctx.set_inputs("q")
    hashes = {base.inputs_hash, other_query.inputs_hash, other_ctx.inputs_hash, no_ctx.inputs_hash}
    assert len(hashes) == 4


def test_set_inputs_only_captures_content_when_explicitly_given_it() -> None:
    """§7.4: ``input_text`` is populated only when ``content_mode`` permits and the

    redactor has already run. The hash is unconditional; the text is not.
    """
    hashed_only = AuditDraft()
    hashed_only.set_inputs("refund order A-123")
    assert hashed_only.input_text is None

    captured = AuditDraft()
    captured.set_inputs("refund order A-123", input_text="refund order [REDACTED]")
    assert captured.input_text == "refund order [REDACTED]"
    assert captured.inputs_hash == hashed_only.inputs_hash  # still joinable


def test_add_usage_accumulates_across_votes_and_repair_retries() -> None:
    draft = AuditDraft()
    draft.add_usage(Usage(input_tokens=800, cached_input_tokens=700, output_tokens=40))
    draft.add_usage(Usage(input_tokens=850, cached_input_tokens=800, output_tokens=55))
    record = draft.finalize()
    assert record.usage == Usage(input_tokens=1650, cached_input_tokens=1500, output_tokens=95)


def test_add_candidates_appends_and_may_be_called_more_than_once() -> None:
    draft = AuditDraft()
    draft.add_candidates([Candidate(route_name="refund", score=3.2, rank=0, source="bm25")])
    draft.add_candidates([Candidate(route_name="human_handoff", score=0.0, rank=1, source="pinned")])
    record = draft.finalize()
    assert [c.route_name for c in record.shortlist] == ["refund", "human_handoff"]
    assert record.shortlist[1].source == "pinned"


def test_add_latency_accumulates_while_set_latency_replaces() -> None:
    draft = AuditDraft()
    draft.add_latency("llm_total", 120.0)
    draft.add_latency("llm_total", 80.0)
    draft.set_latency("llm_ttft", 42.0)
    draft.set_latency("llm_ttft", 37.5)
    record = draft.finalize()
    assert record.latency.llm_total_ms == pytest.approx(200.0)
    assert record.latency.llm_ttft_ms == pytest.approx(37.5)


def test_timing_context_manager_records_even_when_the_block_raises() -> None:
    draft = AuditDraft()
    with pytest.raises(RuntimeError), draft.timing("shortlist"):
        raise RuntimeError("backend exploded")
    assert draft.stage_ms["shortlist"] >= 0.0
    assert draft.finalize().latency.shortlist_ms is not None


def test_unknown_stage_timings_are_ignored_rather_than_rejected() -> None:
    """Stages may record freely; ``LatencyBlock`` only projects the ones it names."""
    draft = AuditDraft()
    draft.add_latency("entitlement_filter", 1.5)
    record = draft.finalize()
    assert record.latency.validation_ms is None
    assert record.latency.total_ms >= 0.0


def test_total_latency_falls_back_to_the_wall_clock() -> None:
    start = datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc)
    draft = AuditDraft(ts_start=start)
    record = draft.finalize(ts_end=start + timedelta(milliseconds=250))
    assert record.latency.total_ms == pytest.approx(250.0)


def test_an_explicit_total_stage_timing_wins_over_the_wall_clock() -> None:
    start = datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc)
    draft = AuditDraft(ts_start=start)
    draft.set_latency("total", 17.0)
    record = draft.finalize(ts_end=start + timedelta(seconds=9))
    assert record.latency.total_ms == pytest.approx(17.0)


def test_set_error_stores_the_exception_type_not_its_message() -> None:
    """The message may carry payload text; ``error.type`` is a span attribute (§8.1)."""
    draft = AuditDraft()
    draft.set_error(ValueError("order A-123 for alice@example.com"))
    assert draft.error == "ValueError"
    assert "alice@example.com" not in (draft.error or "")

    draft.set_error("ProviderTimeout")
    assert draft.error == "ProviderTimeout"
    draft.set_error(None)
    assert draft.error is None


def test_finalize_is_idempotent_and_returns_the_same_record() -> None:
    """The router finalizes in a ``finally`` block *and* on the happy path — one

    decision must never produce two different records.
    """
    draft = AuditDraft(registry_version="a1b2c3d4e5f6")
    assert draft.finalized is False
    first = draft.finalize()
    assert draft.finalized is True
    second = draft.finalize()
    assert first is second
    assert first.ts_end == second.ts_end


def test_finalize_ignores_later_mutations_of_the_draft() -> None:
    draft = AuditDraft(kind="route", routes=["refund"])
    record = draft.finalize()
    draft.kind = "abstain"
    draft.routes.append("track_order")
    assert record.kind == "route"
    assert record.routes == ["refund"]
    assert draft.finalize().routes == ["refund"]


def test_finalize_snapshots_lists_rather_than_aliasing_them() -> None:
    draft = AuditDraft()
    draft.add_candidates([Candidate(route_name="refund", score=1.0, rank=0, source="all")])
    record = draft.finalize()
    draft.shortlist.clear()
    assert len(record.shortlist) == 1


def test_finalize_stamps_ts_end_back_onto_the_draft() -> None:
    draft = AuditDraft()
    assert draft.ts_end is None
    record = draft.finalize()
    assert draft.ts_end == record.ts_end


def test_a_fully_populated_draft_carries_every_field_through() -> None:
    """The end-to-end accumulation the loop actually performs, in arbitrary order."""
    confidence = ConfidenceReport(score=0.88, method="logprobs", p_route=0.88)
    draft = AuditDraft(
        registry_version="a1b2c3d4e5f6",
        tenant_id="acme",
        user_id_hash="u-hash",
        trace_id="trace-abc",
        span_id="span-1",
    )
    draft.set_inputs("refund order A-123", {"locale": "en-GB"}, input_text="refund order A-123")
    draft.candidates_total = 120
    draft.candidates_entitled = 118
    draft.add_candidates([Candidate(route_name="refund", score=9.1, rank=0, source="bm25")])
    draft.shortlist_skipped = False
    draft.weak_retrieval = False
    draft.shuffle_seed = 42
    draft.kind = "route"
    draft.routes = ["refund"]
    draft.args = {"order_id": "A-123"}
    draft.args_hash = sha256_hex({"order_id": "A-123"})
    draft.args_schema_fingerprint = "fp-1"
    draft.rationale = "The query names an order and asks for money back."
    draft.validation_retries = 1
    draft.confidence = confidence
    draft.decision_path = "llm"
    draft.provider = "openai"
    draft.request_model = "gpt-5-nano"
    draft.response_model = "gpt-5-nano-2026-01"
    draft.response_id = "resp-9"
    draft.add_usage(Usage(input_tokens=900, cached_input_tokens=850, output_tokens=48))
    draft.cost = CostBlock(usd=0.00012, price_table_version="2026-08-01")
    draft.add_latency("shortlist", 4.2)
    draft.add_latency("llm_total", 310.0)
    draft.set_latency("validation", 0.9)

    record = draft.finalize()
    assert record.tenant_id == "acme"
    assert record.registry_version == "a1b2c3d4e5f6"
    assert (record.candidates_total, record.candidates_entitled) == (120, 118)
    assert record.shuffle_seed == 42
    assert record.routes == ["refund"]
    assert record.args == {"order_id": "A-123"}
    assert record.args_schema_fingerprint == "fp-1"
    assert record.validation_retries == 1
    assert record.confidence is confidence
    assert record.provider == "openai"
    assert record.response_id == "resp-9"
    assert record.usage.cached_input_tokens == 850
    assert record.cost is not None and record.cost.usd == pytest.approx(0.00012)
    assert record.latency.shortlist_ms == pytest.approx(4.2)
    assert record.latency.llm_total_ms == pytest.approx(310.0)
    assert record.latency.validation_ms == pytest.approx(0.9)
    assert record.error is None


def test_the_finalized_record_is_frozen() -> None:
    from pydantic import ValidationError

    record = AuditDraft().finalize()
    with pytest.raises(ValidationError):
        record.kind = "route"


def test_records_serialise_to_one_json_line_for_a_jsonl_sink() -> None:
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", kind="route", routes=["refund"])
    draft.set_inputs("q")
    line = draft.finalize().model_dump_json()
    assert "\n" not in line
    assert json.loads(line)["schema_version"] == "1"


# --------------------------------------------------------------------------- #
# as_otel_attributes (plan §8.1, §7.4)
# --------------------------------------------------------------------------- #


def _populated_record(**overrides: Any) -> AuditRecord:
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", tenant_id="acme")
    draft.set_inputs("refund order A-123", input_text="refund order A-123")
    draft.kind = "route"
    draft.routes = ["refund"]
    draft.args = {"order_id": "A-123"}
    draft.rationale = "Names an order, asks for money back."
    draft.candidates_entitled = 5
    draft.add_candidates([Candidate(route_name="refund", score=1.0, rank=0, source="all")])
    draft.confidence = ConfidenceReport(score=0.88, method="logprobs")
    draft.provider = "openai"
    draft.request_model = "gpt-5-nano"
    draft.response_model = "gpt-5-nano-2026-01"
    draft.response_id = "resp-9"
    draft.validation_retries = 2
    draft.cost = CostBlock(usd=0.00012, price_table_version="2026-08-01")
    draft.add_usage(Usage(input_tokens=900, cached_input_tokens=850, output_tokens=48))
    for key, value in overrides.items():
        setattr(draft, key, value)
    return draft.finalize()


def test_otel_attributes_use_only_the_two_sanctioned_namespaces() -> None:
    """§8.1 is explicit: only *standard* ``gen_ai.*`` semconv attributes are

    emitted, and everything switchboard-specific rides under ``switchboard.*`` —
    never grafted onto ``gen_ai.*``. Nothing else may appear.
    """
    attrs = _populated_record().as_otel_attributes()
    assert attrs
    offenders = [k for k in attrs if not k.startswith(("gen_ai.", "switchboard."))]
    assert offenders == []


def test_the_only_non_namespaced_attribute_is_the_standard_error_type() -> None:
    """``error.type`` is itself an OTel semconv attribute and is named in §8.1's

    key-attribute list, so it is the single sanctioned exception — asserted
    explicitly so a stray application key can never sneak in beside it.
    """
    draft = AuditDraft(registry_version="a1b2c3d4e5f6")
    draft.set_error(TimeoutError("upstream"))
    attrs = draft.finalize().as_otel_attributes()
    extras = {k for k in attrs if not k.startswith(("gen_ai.", "switchboard."))}
    assert extras == {"error.type"}
    assert attrs["error.type"] == "TimeoutError"


def test_otel_attributes_carry_the_documented_gen_ai_keys() -> None:
    attrs = _populated_record().as_otel_attributes()
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "switchboard.router"
    assert attrs["gen_ai.output.type"] == "json"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-5-nano"
    assert attrs["gen_ai.response.model"] == "gpt-5-nano-2026-01"
    assert attrs["gen_ai.response.id"] == "resp-9"
    assert attrs["gen_ai.usage.input_tokens"] == 900
    assert attrs["gen_ai.usage.output_tokens"] == 48


def test_otel_attributes_carry_the_documented_switchboard_keys() -> None:
    record = _populated_record()
    attrs = record.as_otel_attributes()
    assert attrs["switchboard.decision.kind"] == "route"
    assert attrs["switchboard.decision.route"] == "refund"
    assert attrs["switchboard.decision.path"] == "llm"
    assert attrs["switchboard.registry.version"] == "a1b2c3d4e5f6"
    assert attrs["switchboard.candidates.count"] == 5
    assert attrs["switchboard.shortlist.size"] == 1
    assert attrs["switchboard.confidence.score"] == pytest.approx(0.88)
    assert attrs["switchboard.audit.id"] == record.decision_id
    assert attrs["switchboard.validation.retries"] == 2
    assert attrs["switchboard.cost.usd"] == pytest.approx(0.00012)


def test_unknown_values_are_omitted_rather_than_emitted_as_none() -> None:
    """OTel attribute values may not be ``None``; an absent field means no key."""
    attrs = AuditDraft().finalize().as_otel_attributes()
    assert None not in attrs.values()
    assert "gen_ai.provider.name" not in attrs
    assert "gen_ai.response.id" not in attrs
    assert "switchboard.tenant.id" not in attrs
    assert "switchboard.confidence.score" not in attrs
    assert "error.type" not in attrs


def test_attribute_values_are_otel_safe_primitives() -> None:
    for value in _populated_record().as_otel_attributes().values():
        assert isinstance(value, (str, int, float, bool, tuple))


def test_cost_is_never_emitted_when_it_could_not_be_resolved() -> None:
    """§8.4: ``usd is None`` for unknown models — cost is never guessed, and a

    missing price must not become a zero on a dashboard.
    """
    record = _populated_record(cost=CostBlock(usd=None, price_table_version="2026-08-01"))
    assert "switchboard.cost.usd" not in record.as_otel_attributes()


def test_multiple_routes_emit_both_the_primary_and_the_full_tuple() -> None:
    record = _populated_record(kind="multi_route", routes=["refund", "track_order"])
    attrs = record.as_otel_attributes()
    assert attrs["switchboard.decision.route"] == "refund"
    assert attrs["switchboard.decision.routes"] == ("refund", "track_order")


def test_a_single_route_does_not_emit_the_plural_attribute() -> None:
    assert "switchboard.decision.routes" not in _populated_record().as_otel_attributes()


def test_abstain_reason_rides_under_the_switchboard_namespace() -> None:
    record = _populated_record(kind="abstain", routes=[], abstain_reason="low_confidence")
    attrs = record.as_otel_attributes()
    assert attrs["switchboard.abstain.reason"] == "low_confidence"
    assert "switchboard.decision.route" not in attrs


def test_tenant_id_is_hashed_on_the_span_by_default() -> None:
    """§7.4: the raw tenant id stays in the record; the telemetry backend gets a

    hash unless the operator explicitly opts out.
    """
    record = _populated_record()
    hashed = record.as_otel_attributes()["switchboard.tenant.id"]
    assert hashed != "acme"
    assert hashed == sha256_hex("acme")[:16]
    assert record.tenant_id == "acme"


def test_tenant_id_can_be_emitted_raw_on_request() -> None:
    attrs = _populated_record().as_otel_attributes(hash_tenant=False)
    assert attrs["switchboard.tenant.id"] == "acme"


def test_content_attributes_are_omitted_unless_capture_content_is_true() -> None:
    """Double-keyed on ``content_mode`` *and* the OTel env var by the caller; this

    method never reads the environment, it just refuses by default (§8.1, §7.4).
    """
    attrs = _populated_record().as_otel_attributes()
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    serialised = canonical_json(attrs)
    assert "refund order A-123" not in serialised
    assert "Names an order" not in serialised


def test_content_attributes_appear_when_capture_content_is_true() -> None:
    attrs = _populated_record().as_otel_attributes(capture_content=True)
    inbound = json.loads(attrs["gen_ai.input.messages"])
    outbound = json.loads(attrs["gen_ai.output.messages"])
    assert inbound == [{"role": "user", "content": "refund order A-123"}]
    assert outbound[0]["role"] == "assistant"
    assert outbound[0]["content"]["kind"] == "route"
    assert outbound[0]["content"]["routes"] == ["refund"]
    assert outbound[0]["content"]["args"] == {"order_id": "A-123"}
    assert outbound[0]["content"]["rationale"].startswith("Names an order")


def test_capture_content_still_emits_nothing_when_no_content_was_captured() -> None:
    """A record built under ``content_mode="none"`` has nothing to leak, so the

    flag is inert rather than fabricating empty message payloads.
    """
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", kind="route", routes=["refund"])
    attrs = draft.finalize().as_otel_attributes(capture_content=True)
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs


def test_capture_content_does_not_widen_the_namespace_rule() -> None:
    attrs = _populated_record().as_otel_attributes(capture_content=True)
    assert [k for k in attrs if not k.startswith(("gen_ai.", "switchboard."))] == []


# --------------------------------------------------------------------------- #
# as_training_example (plan §8.5)
# --------------------------------------------------------------------------- #


def test_training_example_is_empty_without_captured_input_text() -> None:
    """§8.5 step 1: "hashes don't train models". A record captured under

    ``content_mode="none"`` is simply not eligible — and returning ``{}`` rather
    than raising lets the dataset builder filter a stream without try/except.
    """
    draft = AuditDraft(registry_version="a1b2c3d4e5f6", kind="route", routes=["refund"])
    draft.set_inputs("refund order A-123")
    record = draft.finalize()
    assert record.input_text is None
    assert record.as_training_example() == {}


def test_training_example_projects_the_record_when_content_was_captured() -> None:
    example = _populated_record().as_training_example()
    assert example == {
        "text": "refund order A-123",
        "label": "refund",
        "registry_version": "a1b2c3d4e5f6",
        "tenant": "acme",
        "weight": pytest.approx(0.88),
    }


def test_training_example_labels_non_route_kinds_by_kind() -> None:
    """clarify/abstain are never fast-pathed (§8.5 step 3), but they are still a

    label space the builder can reason about — and must not be mislabelled as a
    route.
    """
    assert _populated_record(kind="clarify", routes=[]).as_training_example()["label"] == "clarify"
    assert _populated_record(kind="abstain", routes=[]).as_training_example()["label"] == "abstain"


def test_training_example_weight_prefers_a_confirmed_outcome_over_confidence() -> None:
    """``outcome="success"`` is ground truth joined later via ``record_outcome()``;

    it outranks the model's own confidence (§8.5 step 1).
    """
    assert _populated_record(outcome="success").as_training_example()["weight"] == pytest.approx(1.0)
    assert _populated_record(confidence=None).as_training_example()["weight"] == pytest.approx(1.0)
    assert _populated_record().as_training_example()["weight"] == pytest.approx(0.88)


def test_training_example_survives_a_record_with_no_tenant() -> None:
    example = _populated_record(tenant_id=None).as_training_example()
    assert example["tenant"] is None
    assert example["text"] == "refund order A-123"
