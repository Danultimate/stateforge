from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from stateforge.enums import MemoryUnitType
from stateforge.models import (
    DiffEntry,
    MemoryDiff,
    MemoryUnit,
    ProvenanceHop,
    ProvenanceRecord,
    Session,
    Snapshot,
)


def _ts() -> datetime:
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


def _unit(uid: str, key: str = "k", value="v") -> MemoryUnit:
    return MemoryUnit(
        id=uid,
        session_id="s",
        type=MemoryUnitType.KV,
        key=key,
        value=value,
        embedding=None,
        metadata={},
        source="agent",
        source_ref=None,
        created_at=_ts(),
    )


class TestFrozen:
    def test_memory_unit_frozen(self):
        u = _unit("u1")
        with pytest.raises(FrozenInstanceError):
            u.key = "other"  # type: ignore[misc]

    def test_snapshot_frozen(self):
        s = Snapshot(
            id="snap1",
            session_id="s",
            label=None,
            parent_id=None,
            created_at=_ts(),
        )
        with pytest.raises(FrozenInstanceError):
            s.label = "x"  # type: ignore[misc]

    def test_session_frozen(self):
        sess = Session(
            id="s",
            label=None,
            head_snapshot_id=None,
            created_at=_ts(),
        )
        with pytest.raises(FrozenInstanceError):
            sess.label = "x"  # type: ignore[misc]

    def test_memory_diff_frozen(self):
        d = MemoryDiff(
            from_snapshot_id="a",
            to_snapshot_id="b",
            added=[],
            removed=[],
            modified=[],
            created_at=_ts(),
        )
        with pytest.raises(FrozenInstanceError):
            d.added = [_unit("x")]  # type: ignore[misc]

    def test_diff_entry_frozen(self):
        e = DiffEntry(change="added", before=None, after=_unit("u1"))
        with pytest.raises(FrozenInstanceError):
            e.change = "removed"  # type: ignore[misc]

    def test_provenance_record_frozen(self):
        p = ProvenanceRecord(
            id="p1",
            memory_unit_id="u1",
            source="x",
            source_ref=None,
            ingested_at=_ts(),
            trace=[],
        )
        with pytest.raises(FrozenInstanceError):
            p.source = "y"  # type: ignore[misc]


class TestDefaults:
    def test_session_metadata_default(self):
        s = Session(id="s", label=None, head_snapshot_id=None, created_at=_ts())
        assert s.metadata == {}

    def test_snapshot_metadata_default(self):
        s = Snapshot(
            id="s1", session_id="s", label=None, parent_id=None, created_at=_ts()
        )
        assert s.metadata == {}

    def test_session_metadata_factory_not_shared(self):
        s1 = Session(id="a", label=None, head_snapshot_id=None, created_at=_ts())
        s2 = Session(id="b", label=None, head_snapshot_id=None, created_at=_ts())
        assert s1.metadata is not s2.metadata


class TestDiffEntry:
    def test_added_has_only_after(self):
        e = DiffEntry(change="added", before=None, after=_unit("u1"))
        assert e.before is None
        assert e.after is not None

    def test_removed_has_only_before(self):
        e = DiffEntry(change="removed", before=_unit("u1"), after=None)
        assert e.before is not None
        assert e.after is None

    def test_modified_has_both(self):
        e = DiffEntry(
            change="modified", before=_unit("u1", value="old"), after=_unit("u1", value="new")
        )
        assert e.before is not None
        assert e.after is not None


class TestProvenance:
    def test_hop_order_preserved(self):
        hops = [
            ProvenanceHop(index=0, source="langchain", source_ref="ai"),
            ProvenanceHop(index=1, source="tool", source_ref="web_search"),
            ProvenanceHop(index=2, source="summarizer", source_ref=None),
        ]
        p = ProvenanceRecord(
            id="p1",
            memory_unit_id="u1",
            source="summarizer",
            source_ref=None,
            ingested_at=_ts(),
            trace=hops,
        )
        assert [h.index for h in p.trace] == [0, 1, 2]
        assert [h.source for h in p.trace] == ["langchain", "tool", "summarizer"]
