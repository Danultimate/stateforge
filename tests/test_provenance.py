from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stateforge import units
from stateforge.exceptions import MemoryUnitNotFoundError
from stateforge.models import ProvenanceHop, ProvenanceRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mk_record(unit_id: str, *, source: str = "summarizer", hops: list[ProvenanceHop] | None = None) -> ProvenanceRecord:
    return ProvenanceRecord(
        id=str(uuid4()),
        memory_unit_id=unit_id,
        source=source,
        source_ref=None,
        ingested_at=_now(),
        trace=hops or [],
    )


class TestProvenanceWriteRead:
    async def test_write_and_read_basic(self, sf):
        s = await sf.create_session()
        u = units.kv(s.id, "x", 1)
        await sf.snapshot(s.id, units=[u])

        rec = _mk_record(u.id, source="agent")
        await sf.write_provenance(rec)
        got = await sf.get_provenance(u.id)
        assert got.id == rec.id
        assert got.memory_unit_id == u.id
        assert got.source == "agent"
        assert got.trace == []

    async def test_write_and_read_with_hops(self, sf):
        s = await sf.create_session()
        u = units.summary(s.id, "sum:1", "compressed", source="summarizer")
        await sf.snapshot(s.id, units=[u])

        rec = _mk_record(
            u.id,
            source="summarizer",
            hops=[
                ProvenanceHop(index=0, source="langchain", source_ref="ai"),
                ProvenanceHop(index=1, source="tool", source_ref="web_search"),
                ProvenanceHop(index=2, source="summarizer", source_ref=None),
            ],
        )
        await sf.write_provenance(rec)

        got = await sf.get_provenance(u.id)
        assert [h.index for h in got.trace] == [0, 1, 2]
        assert [h.source for h in got.trace] == ["langchain", "tool", "summarizer"]
        # Hops returned in hop_index ascending order.

    async def test_hop_order_preserved_under_unordered_write(self, sf):
        s = await sf.create_session()
        u = units.kv(s.id, "x", 1)
        await sf.snapshot(s.id, units=[u])

        rec = _mk_record(
            u.id,
            hops=[
                # Pass hops in non-monotonic insertion order; storage indexes by hop_index.
                ProvenanceHop(index=2, source="c", source_ref=None),
                ProvenanceHop(index=0, source="a", source_ref=None),
                ProvenanceHop(index=1, source="b", source_ref=None),
            ],
        )
        await sf.write_provenance(rec)
        got = await sf.get_provenance(u.id)
        assert [h.index for h in got.trace] == [0, 1, 2]
        assert [h.source for h in got.trace] == ["a", "b", "c"]


class TestProvenanceErrors:
    async def test_write_for_missing_unit_raises(self, sf):
        rec = _mk_record("ghost-unit")
        with pytest.raises(MemoryUnitNotFoundError):
            await sf.write_provenance(rec)

    async def test_read_for_missing_unit_raises(self, sf):
        with pytest.raises(MemoryUnitNotFoundError):
            await sf.get_provenance("ghost-unit")


class TestProvenanceLatestWins:
    async def test_multiple_records_returns_most_recent(self, sf):
        # If the same unit has multiple provenance records (e.g., re-ingested),
        # get_provenance returns the latest by ingested_at.
        s = await sf.create_session()
        u = units.kv(s.id, "x", 1)
        await sf.snapshot(s.id, units=[u])

        # First record
        rec1 = _mk_record(u.id, source="first")
        await sf.write_provenance(rec1)

        # Second, later record (manual ingested_at ahead of rec1)
        from datetime import timedelta
        rec2 = ProvenanceRecord(
            id=str(uuid4()),
            memory_unit_id=u.id,
            source="second",
            source_ref=None,
            ingested_at=rec1.ingested_at + timedelta(seconds=10),
            trace=[],
        )
        await sf.write_provenance(rec2)

        got = await sf.get_provenance(u.id)
        assert got.source == "second"
