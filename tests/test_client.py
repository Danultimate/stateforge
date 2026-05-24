from __future__ import annotations

import asyncio
import secrets

import pytest

from stateforge import StateForge, units
from stateforge.exceptions import (
    EncryptionKeyFormatError,
    HistoryTooDeepError,
    SessionNotFoundError,
    SnapshotNotFoundError,
    ValueTypeError,
)


# ────────────────────────────────────────────────────────────────────────────
# __init__ validation (synchronous)
# ────────────────────────────────────────────────────────────────────────────


class TestInitValidation:
    def test_default_construction_does_not_open_db(self, tmp_path):
        # Sync construction must not touch the disk yet.
        path = tmp_path / "x.db"
        StateForge(str(path))
        assert not path.exists()

    def test_both_encryption_args_raise(self):
        with pytest.raises(ValueError, match="at most one"):
            StateForge(
                ":memory:",
                encryption_key="0" * 64,
                encryption_key_provider=lambda: "0" * 64,
            )

    def test_pragmas_disallowed_key_raises(self):
        with pytest.raises(ValueError, match="not user-overridable"):
            StateForge(":memory:", pragmas={"journal_mode": "DELETE"})

    def test_pragmas_disallowed_key_raises_when_mixed_with_allowed(self):
        with pytest.raises(ValueError, match="not user-overridable"):
            StateForge(
                ":memory:",
                pragmas={"busy_timeout": "1000", "key": "abc"},
            )

    def test_pragmas_allowed_keys_accepted(self):
        StateForge(
            ":memory:",
            pragmas={
                "busy_timeout": "1000",
                "synchronous": "FULL",
                "foreign_keys": "ON",
            },
        )

    def test_bad_encryption_key_raises_at_init(self):
        with pytest.raises(EncryptionKeyFormatError):
            StateForge(":memory:", encryption_key="not-hex")

    def test_bad_encryption_key_via_provider_raises_at_init(self):
        with pytest.raises(EncryptionKeyFormatError):
            StateForge(":memory:", encryption_key_provider=lambda: b"\x00" * 16)

    def test_provider_called_once_at_init(self):
        calls = []

        def provider():
            calls.append(1)
            return secrets.token_hex(32)

        StateForge(":memory:", encryption_key_provider=provider)
        # Provider invoked exactly once during __init__.
        assert len(calls) == 1


# ────────────────────────────────────────────────────────────────────────────
# Lazy initialization
# ────────────────────────────────────────────────────────────────────────────


class TestLazyInit:
    async def test_first_call_initializes(self, sf):
        # No explicit initialize() needed; the first awaited call triggers it.
        sessions = await sf.list_sessions()
        assert sessions == []

    async def test_close_then_no_op(self, sf):
        await sf.list_sessions()  # init
        await sf.close()
        # Second close is a no-op.
        await sf.close()

    async def test_concurrent_first_calls_dedupe(self, sf):
        # Many concurrent first-touch calls must not double-initialize.
        await asyncio.gather(*[sf.list_sessions() for _ in range(10)])


# ────────────────────────────────────────────────────────────────────────────
# Sessions
# ────────────────────────────────────────────────────────────────────────────


class TestSessions:
    async def test_create_and_get(self, sf):
        s = await sf.create_session(label="x", metadata={"k": "v"})
        got = await sf.get_session(s.id)
        assert got.id == s.id
        assert got.label == "x"
        assert got.metadata == {"k": "v"}
        assert got.head_snapshot_id is None

    async def test_get_missing_raises(self, sf):
        with pytest.raises(SessionNotFoundError):
            await sf.get_session("nope")

    async def test_create_with_bad_metadata_rejected(self, sf):
        with pytest.raises(ValueTypeError):
            await sf.create_session(metadata={"bad": b"bytes"})

    async def test_list_sessions(self, sf):
        s1 = await sf.create_session(label="a")
        s2 = await sf.create_session(label="b")
        listed = await sf.list_sessions()
        ids = {s.id for s in listed}
        assert {s1.id, s2.id} <= ids


# ────────────────────────────────────────────────────────────────────────────
# Snapshot + head
# ────────────────────────────────────────────────────────────────────────────


class TestSnapshot:
    async def test_first_snapshot_has_no_parent(self, sf):
        s = await sf.create_session()
        snap = await sf.snapshot(s.id, units=[], label="first")
        assert snap.parent_id is None
        assert snap.label == "first"
        head = await sf.head(s.id)
        assert head is not None
        assert head.id == snap.id

    async def test_second_snapshot_chains_parent(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        assert snap2.parent_id == snap1.id

    async def test_snapshot_with_units(self, sf):
        s = await sf.create_session()
        u = units.message(s.id, "msg:0", "hello", source="user")
        snap = await sf.snapshot(s.id, units=[u])
        loaded = await sf.get_units(snap.id)
        assert len(loaded) == 1
        assert loaded[0].id == u.id
        assert loaded[0].value == "hello"

    async def test_snapshot_missing_session_raises(self, sf):
        with pytest.raises(SessionNotFoundError):
            await sf.snapshot("ghost", units=[])

    async def test_snapshot_bad_metadata_raises(self, sf):
        s = await sf.create_session()
        with pytest.raises(ValueTypeError):
            await sf.snapshot(s.id, units=[], metadata={"bad": b"x"})

    async def test_concurrent_snapshots_form_chain(self, sf):
        """Per-session lock guarantees that even concurrent snapshots on the
        same session chain linearly (no shared parent_id)."""
        s = await sf.create_session()
        results = await asyncio.gather(
            *[sf.snapshot(s.id, units=[], label=f"s{i}") for i in range(5)]
        )
        # Each snapshot's parent must be the previous one (in some serialization).
        # Build parent → child map and verify it's a linear chain rooted at None.
        by_id = {snap.id: snap for snap in results}
        children: dict[str | None, list[str]] = {}
        for snap in results:
            children.setdefault(snap.parent_id, []).append(snap.id)
        # Exactly one root (parent_id is None).
        assert len(children.get(None, [])) == 1
        # Each non-root has exactly one parent that exists in results.
        non_root = [s for s in results if s.parent_id is not None]
        for snap in non_root:
            assert snap.parent_id in by_id
        # Each node has at most one child (linear).
        for parent_id, kids in children.items():
            assert len(kids) == 1, f"branch detected at parent {parent_id}"

    async def test_head_for_empty_session_is_none(self, sf):
        s = await sf.create_session()
        assert await sf.head(s.id) is None


# ────────────────────────────────────────────────────────────────────────────
# Snapshot read paths
# ────────────────────────────────────────────────────────────────────────────


class TestReads:
    async def test_get_snapshot_missing_raises(self, sf):
        with pytest.raises(SnapshotNotFoundError):
            await sf.get_snapshot("nope")

    async def test_get_units_for_missing_snapshot_raises(self, sf):
        with pytest.raises(SnapshotNotFoundError):
            await sf.get_units("nope")

    async def test_list_snapshots(self, sf):
        s = await sf.create_session()
        snaps = [await sf.snapshot(s.id, units=[], label=f"s{i}") for i in range(3)]
        listed = await sf.list_snapshots(s.id)
        ids = {x.id for x in listed}
        assert {x.id for x in snaps} <= ids


# ────────────────────────────────────────────────────────────────────────────
# History
# ────────────────────────────────────────────────────────────────────────────


class TestHistory:
    async def test_empty_session(self, sf):
        s = await sf.create_session()
        assert await sf.history(s.id) == []

    async def test_single_snapshot(self, sf):
        s = await sf.create_session()
        snap = await sf.snapshot(s.id, units=[])
        h = await sf.history(s.id)
        assert [x.id for x in h] == [snap.id]

    async def test_chain_ordered_head_to_root(self, sf):
        s = await sf.create_session()
        snaps = [await sf.snapshot(s.id, units=[]) for _ in range(4)]
        h = await sf.history(s.id)
        assert [x.id for x in h] == [x.id for x in reversed(snaps)]

    async def test_max_depth(self, sf):
        s = await sf.create_session()
        for _ in range(5):
            await sf.snapshot(s.id, units=[])
        with pytest.raises(HistoryTooDeepError):
            await sf.history(s.id, max_depth=2)


# ────────────────────────────────────────────────────────────────────────────
# Quickstart-style end-to-end
# ────────────────────────────────────────────────────────────────────────────


class TestQuickstart:
    async def test_full_flow(self, sf):
        s = await sf.create_session(label="my-agent-run")

        snap1 = await sf.snapshot(
            s.id,
            units=[
                units.message(s.id, "msg:0", "hello", source="user"),
                units.kv(s.id, "goal", {"task": "summarize"}, source="agent"),
            ],
            label="initial",
        )

        snap2 = await sf.snapshot(
            s.id,
            units=[
                units.message(s.id, "msg:0", "hello", source="user"),
                units.message(s.id, "msg:1", "working on it", source="agent"),
                units.kv(
                    s.id, "goal",
                    {"task": "summarize", "progress": 0.5}, source="agent",
                ),
            ],
            label="after-step-1",
        )

        d = await sf.diff(snap1.id, snap2.id)
        # added: msg:1; modified: goal
        assert len(d.added) == 1
        assert d.added[0].key == "msg:1"
        assert len(d.modified) == 1
        before, after = d.modified[0]
        assert before.key == "goal"
        assert after.value["progress"] == 0.5
        assert len(d.removed) == 0

        snap3 = await sf.rollback(s.id, to_snapshot_id=snap1.id, label="undo")
        head = await sf.head(s.id)
        assert head is not None
        assert head.id == snap3.id
        # snap3 references the same unit identities as snap1.
        roll_units = await sf.get_units(snap3.id)
        roll_keys = {u.key for u in roll_units}
        snap1_keys = {u.key for u in await sf.get_units(snap1.id)}
        assert roll_keys == snap1_keys
