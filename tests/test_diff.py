from __future__ import annotations

from datetime import timedelta

import pytest

from stateforge import units
from stateforge.exceptions import SnapshotNotFoundError


class TestDiffBasics:
    async def test_two_empty_snapshots(self, sf):
        s = await sf.create_session()
        a = await sf.snapshot(s.id, units=[])
        b = await sf.snapshot(s.id, units=[])
        d = await sf.diff(a.id, b.id)
        assert d.added == []
        assert d.removed == []
        assert d.modified == []

    async def test_added(self, sf):
        s = await sf.create_session()
        a = await sf.snapshot(s.id, units=[])
        u = units.kv(s.id, "k", 1)
        b = await sf.snapshot(s.id, units=[u])
        d = await sf.diff(a.id, b.id)
        assert len(d.added) == 1
        assert d.added[0].id == u.id
        assert d.removed == []
        assert d.modified == []

    async def test_removed(self, sf):
        s = await sf.create_session()
        u = units.kv(s.id, "k", 1)
        a = await sf.snapshot(s.id, units=[u])
        b = await sf.snapshot(s.id, units=[])
        d = await sf.diff(a.id, b.id)
        assert d.added == []
        assert len(d.removed) == 1
        assert d.removed[0].id == u.id
        assert d.modified == []

    async def test_modified_by_value(self, sf):
        s = await sf.create_session()
        u_before = units.kv(s.id, "x", 1)
        u_after = units.kv(s.id, "x", 2)
        a = await sf.snapshot(s.id, units=[u_before])
        b = await sf.snapshot(s.id, units=[u_after])
        d = await sf.diff(a.id, b.id)
        assert d.added == []
        assert d.removed == []
        assert len(d.modified) == 1
        bef, aft = d.modified[0]
        assert bef.value == 1
        assert aft.value == 2

    async def test_modified_by_metadata(self, sf):
        s = await sf.create_session()
        u_before = units.kv(s.id, "x", 1, metadata={"v": 1})
        u_after = units.kv(s.id, "x", 1, metadata={"v": 2})
        a = await sf.snapshot(s.id, units=[u_before])
        b = await sf.snapshot(s.id, units=[u_after])
        d = await sf.diff(a.id, b.id)
        assert len(d.modified) == 1

    async def test_modified_by_embedding(self, sf):
        s = await sf.create_session()
        u_before = units.embedding(s.id, "doc", [1.0, 2.0, 3.0], source="e")
        u_after = units.embedding(s.id, "doc", [1.0, 2.0, 9.0], source="e")
        a = await sf.snapshot(s.id, units=[u_before])
        b = await sf.snapshot(s.id, units=[u_after])
        d = await sf.diff(a.id, b.id)
        assert len(d.modified) == 1

    async def test_same_value_means_no_change(self, sf):
        s = await sf.create_session()
        u_before = units.kv(s.id, "x", 1)
        u_after = units.kv(s.id, "x", 1)  # different unit id, same logical state
        a = await sf.snapshot(s.id, units=[u_before])
        b = await sf.snapshot(s.id, units=[u_after])
        d = await sf.diff(a.id, b.id)
        # Same identity (session, key, type) AND identical content → no diff entry.
        assert d.added == []
        assert d.removed == []
        assert d.modified == []


class TestDiffIdentity:
    async def test_identity_uses_session_key_type(self, sf):
        s = await sf.create_session()
        # Same key, different type → different identities.
        u_msg = units.message(s.id, "x", "hello", source="user")
        u_kv = units.kv(s.id, "x", "hello", source="agent")
        a = await sf.snapshot(s.id, units=[u_msg])
        b = await sf.snapshot(s.id, units=[u_kv])
        d = await sf.diff(a.id, b.id)
        # message:x removed, kv:x added.
        assert len(d.added) == 1
        assert d.added[0].type.value == "kv"
        assert len(d.removed) == 1
        assert d.removed[0].type.value == "message"
        assert d.modified == []


class TestDuplicateIdentityLatestWins:
    async def test_two_units_same_identity_in_one_snapshot(self, sf):
        s = await sf.create_session()
        # Two units with same (session, key, type) — latest by created_at wins.
        u_early = units.kv(s.id, "k", "early")
        u_late = units.kv(s.id, "k", "late")
        # Ensure ordering by created_at.
        from dataclasses import replace
        u_early = replace(u_early, created_at=u_late.created_at - timedelta(seconds=1))

        a = await sf.snapshot(s.id, units=[])
        b = await sf.snapshot(s.id, units=[u_early, u_late])
        d = await sf.diff(a.id, b.id)
        # Only one identity present in b → one "added".
        assert len(d.added) == 1
        assert d.added[0].value == "late"  # latest wins


class TestDiffMissing:
    async def test_missing_from_raises(self, sf):
        s = await sf.create_session()
        b = await sf.snapshot(s.id, units=[])
        with pytest.raises(SnapshotNotFoundError):
            await sf.diff("nope", b.id)

    async def test_missing_to_raises(self, sf):
        s = await sf.create_session()
        a = await sf.snapshot(s.id, units=[])
        with pytest.raises(SnapshotNotFoundError):
            await sf.diff(a.id, "nope")


class TestDiffIter:
    async def test_yields_same_entries_as_diff(self, sf):
        s = await sf.create_session()
        u1 = units.kv(s.id, "x", 1)
        u2 = units.kv(s.id, "y", 2)
        a = await sf.snapshot(s.id, units=[u1])
        u1b = units.kv(s.id, "x", 99)  # modified
        b = await sf.snapshot(s.id, units=[u1b, u2])

        entries = []
        async for e in sf.diff_iter(a.id, b.id):
            entries.append(e)

        changes = {e.change for e in entries}
        assert changes == {"added", "modified"}
        d = await sf.diff(a.id, b.id)
        assert len(entries) == len(d.added) + len(d.modified) + len(d.removed)

    async def test_diff_iter_added_entries_have_only_after(self, sf):
        s = await sf.create_session()
        a = await sf.snapshot(s.id, units=[])
        u = units.kv(s.id, "x", 1)
        b = await sf.snapshot(s.id, units=[u])
        async for e in sf.diff_iter(a.id, b.id):
            assert e.change == "added"
            assert e.before is None
            assert e.after is not None
