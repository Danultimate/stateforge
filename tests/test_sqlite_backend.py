from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from stateforge import units
from stateforge.exceptions import (
    HistoryTooDeepError,
    MemoryUnitNotFoundError,
    SessionNotFoundError,
    SnapshotNotFoundError,
)
from stateforge.models import (
    ProvenanceHop,
    ProvenanceRecord,
    Session,
    Snapshot,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mk_session(label: str = "s") -> Session:
    return Session(
        id=str(uuid4()),
        label=label,
        head_snapshot_id=None,
        created_at=_now(),
        metadata={},
    )


def _mk_snapshot(session_id: str, parent_id: str | None = None, label: str | None = None) -> Snapshot:
    return Snapshot(
        id=str(uuid4()),
        session_id=session_id,
        label=label,
        parent_id=parent_id,
        created_at=_now(),
        metadata={},
    )


# ────────────────────────────────────────────────────────────────────────────
# Sessions
# ────────────────────────────────────────────────────────────────────────────


class TestSessions:
    async def test_create_and_get(self, backend):
        sess = _mk_session()
        await backend.create_session(sess)
        got = await backend.get_session(sess.id)
        assert got.id == sess.id
        assert got.label == sess.label
        assert got.head_snapshot_id is None

    async def test_get_missing_raises(self, backend):
        with pytest.raises(SessionNotFoundError):
            await backend.get_session("nonexistent")

    async def test_list_returns_newest_first(self, backend):
        sessions = []
        for i in range(3):
            sess = Session(
                id=str(uuid4()),
                label=f"s{i}",
                head_snapshot_id=None,
                created_at=_now() + timedelta(seconds=i),
                metadata={},
            )
            await backend.create_session(sess)
            sessions.append(sess)
        listed = await backend.list_sessions(limit=10, before_id=None)
        assert [s.id for s in listed] == [sessions[2].id, sessions[1].id, sessions[0].id]

    async def test_list_pagination(self, backend):
        sessions = []
        for i in range(5):
            sess = Session(
                id=str(uuid4()),
                label=f"s{i}",
                head_snapshot_id=None,
                created_at=_now() + timedelta(seconds=i),
                metadata={},
            )
            await backend.create_session(sess)
            sessions.append(sess)
        first_page = await backend.list_sessions(limit=2, before_id=None)
        assert len(first_page) == 2
        next_page = await backend.list_sessions(limit=2, before_id=first_page[-1].id)
        assert len(next_page) == 2
        assert next_page[0].id != first_page[-1].id

    async def test_set_head_advances(self, backend, session_id):
        sess = Session(
            id=session_id, label=None, head_snapshot_id=None, created_at=_now(), metadata={}
        )
        await backend.create_session(sess)
        snap = _mk_snapshot(session_id)
        await backend.write_snapshot(snap, new_units=[], unit_ids_to_link=[])
        got = await backend.get_session(session_id)
        assert got.head_snapshot_id == snap.id


# ────────────────────────────────────────────────────────────────────────────
# Snapshot write (atomicity, head advance, junction)
# ────────────────────────────────────────────────────────────────────────────


class TestWriteSnapshot:
    async def test_empty_snapshot(self, backend_with_session):
        be, sess = backend_with_session
        snap = _mk_snapshot(sess.id, label="empty")
        await be.write_snapshot(snap, new_units=[], unit_ids_to_link=[])
        read = await be.read_snapshot(snap.id)
        assert read.id == snap.id
        assert read.label == "empty"
        # head advanced
        assert (await be.get_session(sess.id)).head_snapshot_id == snap.id
        # membership empty
        assert await be.read_snapshot_unit_ids(snap.id) == []

    async def test_snapshot_with_units(self, backend_with_session):
        be, sess = backend_with_session
        u1 = units.message(sess.id, "msg:0", "hello")
        u2 = units.kv(sess.id, "goal", {"task": "summarize"})
        snap = _mk_snapshot(sess.id)
        await be.write_snapshot(snap, new_units=[u1, u2], unit_ids_to_link=[u1.id, u2.id])

        ids = await be.read_snapshot_unit_ids(snap.id)
        assert sorted(ids) == sorted([u1.id, u2.id])
        loaded = await be.read_units(ids)
        loaded_by_id = {u.id: u for u in loaded}
        assert loaded_by_id[u1.id].value == "hello"
        assert loaded_by_id[u2.id].value == {"task": "summarize"}

    async def test_snapshot_for_missing_session_raises(self, backend):
        snap = _mk_snapshot("ghost-session")
        with pytest.raises(SessionNotFoundError):
            await backend.write_snapshot(snap, new_units=[], unit_ids_to_link=[])

    async def test_unit_ids_carry_forward_across_snapshots(self, backend_with_session):
        # snap1 has u1; snap2 (child) references u1 plus a new u2.
        be, sess = backend_with_session
        u1 = units.kv(sess.id, "x", 1)
        snap1 = _mk_snapshot(sess.id, label="s1")
        await be.write_snapshot(snap1, new_units=[u1], unit_ids_to_link=[u1.id])

        u2 = units.kv(sess.id, "y", 2)
        snap2 = _mk_snapshot(sess.id, parent_id=snap1.id, label="s2")
        await be.write_snapshot(snap2, new_units=[u2], unit_ids_to_link=[u1.id, u2.id])

        snap2_ids = await be.read_snapshot_unit_ids(snap2.id)
        assert sorted(snap2_ids) == sorted([u1.id, u2.id])
        # u1 was not re-inserted; INSERT OR IGNORE is idempotent.
        u1_read = await be.read_unit(u1.id)
        assert u1_read.value == 1

    async def test_atomicity_on_failure(self, backend_with_session):
        # Trying to link a non-existent unit id should rollback the snapshot row.
        be, sess = backend_with_session
        snap = _mk_snapshot(sess.id)
        from stateforge.exceptions import StorageError
        with pytest.raises(StorageError):
            await be.write_snapshot(
                snap, new_units=[], unit_ids_to_link=["does-not-exist"]
            )
        with pytest.raises(SnapshotNotFoundError):
            await be.read_snapshot(snap.id)
        # Head was not advanced.
        assert (await be.get_session(sess.id)).head_snapshot_id is None


# ────────────────────────────────────────────────────────────────────────────
# Read paths and pagination
# ────────────────────────────────────────────────────────────────────────────


class TestReads:
    async def test_read_units_empty_list(self, backend):
        assert await backend.read_units([]) == []

    async def test_read_unit_missing_raises(self, backend):
        with pytest.raises(MemoryUnitNotFoundError):
            await backend.read_unit("nope")

    async def test_read_snapshot_missing_raises(self, backend):
        with pytest.raises(SnapshotNotFoundError):
            await backend.read_snapshot("nope")

    async def test_read_snapshot_unit_ids_missing_raises(self, backend):
        with pytest.raises(SnapshotNotFoundError):
            await backend.read_snapshot_unit_ids("nope")

    async def test_list_snapshots_pagination(self, backend_with_session):
        be, sess = backend_with_session
        snaps = []
        parent = None
        for i in range(5):
            s = Snapshot(
                id=str(uuid4()),
                session_id=sess.id,
                label=f"s{i}",
                parent_id=parent,
                created_at=_now() + timedelta(seconds=i),
                metadata={},
            )
            await be.write_snapshot(s, new_units=[], unit_ids_to_link=[])
            snaps.append(s)
            parent = s.id
        # Newest first
        page1 = await be.list_snapshots(sess.id, limit=2, before_id=None)
        assert [s.label for s in page1] == ["s4", "s3"]
        page2 = await be.list_snapshots(sess.id, limit=2, before_id=page1[-1].id)
        assert [s.label for s in page2] == ["s2", "s1"]

    async def test_list_snapshots_missing_session(self, backend):
        with pytest.raises(SessionNotFoundError):
            await backend.list_snapshots("nope", limit=10, before_id=None)


# ────────────────────────────────────────────────────────────────────────────
# Rollback (atomic, non-destructive, head advances)
# ────────────────────────────────────────────────────────────────────────────


class TestRollback:
    async def test_rollback_creates_new_snapshot_referencing_same_units(
        self, backend_with_session
    ):
        be, sess = backend_with_session
        u1 = units.kv(sess.id, "x", 1)
        snap1 = _mk_snapshot(sess.id, label="initial")
        await be.write_snapshot(snap1, new_units=[u1], unit_ids_to_link=[u1.id])

        # Move forward
        u2 = units.kv(sess.id, "y", 2)
        snap2 = _mk_snapshot(sess.id, parent_id=snap1.id, label="forward")
        await be.write_snapshot(snap2, new_units=[u2], unit_ids_to_link=[u1.id, u2.id])

        # Now roll back to snap1: a new snapshot whose membership = snap1's.
        snap3 = Snapshot(
            id=str(uuid4()),
            session_id=sess.id,
            label="undo",
            parent_id=snap2.id,
            created_at=_now(),
            metadata={},
        )
        await be.write_rollback(snap3, copy_from_id=snap1.id)

        # head moved to snap3
        assert (await be.get_session(sess.id)).head_snapshot_id == snap3.id
        # snap3 references only u1 (same as snap1)
        assert await be.read_snapshot_unit_ids(snap3.id) == [u1.id]
        # snap1 + snap2 are still intact (non-destructive)
        assert await be.read_snapshot_unit_ids(snap1.id) == [u1.id]
        assert sorted(await be.read_snapshot_unit_ids(snap2.id)) == sorted([u1.id, u2.id])
        # all original units still readable
        assert (await be.read_unit(u1.id)).value == 1
        assert (await be.read_unit(u2.id)).value == 2

    async def test_rollback_to_missing_raises(self, backend_with_session):
        be, sess = backend_with_session
        snap = _mk_snapshot(sess.id)
        with pytest.raises(SnapshotNotFoundError):
            await be.write_rollback(snap, copy_from_id="nope")


# ────────────────────────────────────────────────────────────────────────────
# Parent walk
# ────────────────────────────────────────────────────────────────────────────


class TestWalkParents:
    async def test_root_only(self, backend_with_session):
        be, sess = backend_with_session
        snap = _mk_snapshot(sess.id)
        await be.write_snapshot(snap, new_units=[], unit_ids_to_link=[])
        chain = await be.walk_parents(snap.id, max_depth=10)
        assert [s.id for s in chain] == [snap.id]
        assert chain[0].parent_id is None

    async def test_multi_level_chain(self, backend_with_session):
        be, sess = backend_with_session
        ids = []
        parent = None
        for i in range(4):
            s = Snapshot(
                id=str(uuid4()),
                session_id=sess.id,
                label=f"s{i}",
                parent_id=parent,
                created_at=_now() + timedelta(seconds=i),
                metadata={},
            )
            await be.write_snapshot(s, new_units=[], unit_ids_to_link=[])
            ids.append(s.id)
            parent = s.id
        chain = await be.walk_parents(ids[-1], max_depth=100)
        # head → root order
        assert [s.id for s in chain] == list(reversed(ids))

    async def test_max_depth_exceeded(self, backend_with_session):
        be, sess = backend_with_session
        parent = None
        last = None
        for i in range(5):
            s = Snapshot(
                id=str(uuid4()),
                session_id=sess.id,
                label=f"s{i}",
                parent_id=parent,
                created_at=_now() + timedelta(seconds=i),
                metadata={},
            )
            await be.write_snapshot(s, new_units=[], unit_ids_to_link=[])
            parent = s.id
            last = s.id
        with pytest.raises(HistoryTooDeepError):
            await be.walk_parents(last, max_depth=2)

    async def test_walk_missing_raises(self, backend):
        with pytest.raises(SnapshotNotFoundError):
            await backend.walk_parents("nope", max_depth=10)


# ────────────────────────────────────────────────────────────────────────────
# Provenance
# ────────────────────────────────────────────────────────────────────────────


class TestProvenance:
    async def test_write_and_read_with_trace(self, backend_with_session):
        be, sess = backend_with_session
        u = units.kv(sess.id, "x", 1)
        snap = _mk_snapshot(sess.id)
        await be.write_snapshot(snap, new_units=[u], unit_ids_to_link=[u.id])

        rec = ProvenanceRecord(
            id=str(uuid4()),
            memory_unit_id=u.id,
            source="summarizer",
            source_ref=None,
            ingested_at=_now(),
            trace=[
                ProvenanceHop(index=0, source="langchain", source_ref="ai"),
                ProvenanceHop(index=1, source="tool", source_ref="web_search"),
                ProvenanceHop(index=2, source="summarizer", source_ref=None),
            ],
        )
        await be.write_provenance(rec)

        got = await be.read_provenance(u.id)
        assert got.id == rec.id
        assert got.memory_unit_id == u.id
        assert [h.index for h in got.trace] == [0, 1, 2]
        assert [h.source for h in got.trace] == ["langchain", "tool", "summarizer"]

    async def test_write_to_missing_unit_raises(self, backend):
        rec = ProvenanceRecord(
            id=str(uuid4()),
            memory_unit_id="ghost",
            source="x",
            source_ref=None,
            ingested_at=_now(),
            trace=[],
        )
        with pytest.raises(MemoryUnitNotFoundError):
            await backend.write_provenance(rec)

    async def test_read_for_missing_unit_raises(self, backend):
        with pytest.raises(MemoryUnitNotFoundError):
            await backend.read_provenance("ghost")
