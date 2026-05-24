from __future__ import annotations

import pytest

from stateforge import units
from stateforge.exceptions import (
    CrossSessionRollbackError,
    SnapshotNotFoundError,
)


class TestRollbackBasics:
    async def test_rollback_creates_new_snapshot(self, sf):
        s = await sf.create_session()
        u = units.kv(s.id, "x", 1)
        snap1 = await sf.snapshot(s.id, units=[u], label="initial")
        snap2 = await sf.snapshot(s.id, units=[], label="forward")

        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)
        assert snap3.id not in {snap1.id, snap2.id}

    async def test_rollback_advances_head(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)

        head = await sf.head(s.id)
        assert head is not None
        assert head.id == snap3.id

    async def test_rollback_is_non_destructive(self, sf):
        s = await sf.create_session()
        u1 = units.kv(s.id, "x", 1)
        u2 = units.kv(s.id, "y", 2)
        snap1 = await sf.snapshot(s.id, units=[u1])
        snap2 = await sf.snapshot(s.id, units=[u1, u2])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)

        # All pre-rollback snapshots still readable.
        await sf.get_snapshot(snap1.id)
        await sf.get_snapshot(snap2.id)
        # All units still readable.
        assert (await sf.get_unit(u1.id)).value == 1
        assert (await sf.get_unit(u2.id)).value == 2
        # Membership of snap3 matches snap1.
        snap1_keys = {u.key for u in await sf.get_units(snap1.id)}
        snap3_keys = {u.key for u in await sf.get_units(snap3.id)}
        assert snap1_keys == snap3_keys

    async def test_rollback_parent_is_previous_head(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)
        assert snap3.parent_id == snap2.id

    async def test_rollback_label_default(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)
        assert snap3.label is not None
        assert snap3.label.startswith("rollback to ")
        # Default label includes first 8 chars of target id.
        assert snap1.id[:8] in snap3.label

    async def test_rollback_label_custom(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id, label="undo step")
        assert snap3.label == "undo step"


class TestRollbackErrors:
    async def test_rollback_to_missing_snapshot_raises(self, sf):
        s = await sf.create_session()
        with pytest.raises(SnapshotNotFoundError):
            await sf.rollback(s.id, to_snapshot_id="nope")

    async def test_cross_session_rollback_raises(self, sf):
        s1 = await sf.create_session()
        s2 = await sf.create_session()
        snap_in_s1 = await sf.snapshot(s1.id, units=[])
        with pytest.raises(CrossSessionRollbackError):
            await sf.rollback(s2.id, to_snapshot_id=snap_in_s1.id)


class TestRollbackComposes:
    async def test_diff_after_rollback(self, sf):
        s = await sf.create_session()
        u_v1 = units.kv(s.id, "x", 1)
        u_v2 = units.kv(s.id, "x", 2)
        snap1 = await sf.snapshot(s.id, units=[u_v1])
        snap2 = await sf.snapshot(s.id, units=[u_v2])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)

        # Diff snap2 → snap3: x's value goes from 2 back to 1 (modified).
        d = await sf.diff(snap2.id, snap3.id)
        assert len(d.modified) == 1
        bef, aft = d.modified[0]
        assert bef.value == 2
        assert aft.value == 1

    async def test_history_includes_rollback_snapshot(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id)

        h = await sf.history(s.id)
        assert [x.id for x in h] == [snap3.id, snap2.id, snap1.id]
