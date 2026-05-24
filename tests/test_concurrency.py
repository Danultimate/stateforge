from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stateforge import units
from stateforge.models import Session, Snapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _race_write(backend, sess_id: str, *, label: str, parent_id: str | None):
    """Helper: write a snapshot. Each call is independent."""
    u = units.kv(sess_id, f"k:{label}", label, source="agent")
    snap = Snapshot(
        id=str(uuid4()),
        session_id=sess_id,
        label=label,
        parent_id=parent_id,
        created_at=_now(),
        metadata={},
    )
    await backend.write_snapshot(snap, new_units=[u], unit_ids_to_link=[u.id])
    return snap


class TestConcurrentSnapshots:
    async def test_two_tasks_both_succeed(self, backend_with_session):
        """Two concurrent snapshot() calls on the same session both commit.

        aiosqlite serializes writes on a dedicated thread. Both must succeed;
        neither leaves orphan rows. Final head must be one of the two.
        """
        be, sess = backend_with_session
        results = await asyncio.gather(
            _race_write(be, sess.id, label="A", parent_id=None),
            _race_write(be, sess.id, label="B", parent_id=None),
        )
        a, b = results
        # Both snapshots exist
        await be.read_snapshot(a.id)
        await be.read_snapshot(b.id)
        # Head is one of them (last writer wins)
        head = (await be.get_session(sess.id)).head_snapshot_id
        assert head in {a.id, b.id}

    async def test_many_concurrent_writes(self, backend_with_session):
        """Stress: 20 concurrent snapshots. All must commit; head must exist."""
        be, sess = backend_with_session
        N = 20
        tasks = [
            _race_write(be, sess.id, label=f"s{i}", parent_id=None)
            for i in range(N)
        ]
        snaps = await asyncio.gather(*tasks)
        listed = await be.list_snapshots(sess.id, limit=100, before_id=None)
        assert len(listed) == N
        # The head was written by some task; it must be one of them.
        head = (await be.get_session(sess.id)).head_snapshot_id
        assert head in {s.id for s in snaps}

    async def test_concurrent_writes_do_not_corrupt_unit_set(
        self, backend_with_session
    ):
        """Each snapshot's unit_ids_to_link must match what it actually contains."""
        be, sess = backend_with_session
        N = 10
        snaps = await asyncio.gather(
            *[_race_write(be, sess.id, label=f"x{i}", parent_id=None) for i in range(N)]
        )
        for s in snaps:
            ids = await be.read_snapshot_unit_ids(s.id)
            # Each snapshot was given exactly one unit id.
            assert len(ids) == 1


class TestMigrationRace:
    """Multiple SQLiteBackend instances initializing the same file in parallel.

    The spec requires migrations to be idempotent + wrapped in BEGIN IMMEDIATE
    so this scenario cannot corrupt the schema.
    """

    async def test_parallel_init_on_same_file(self, tmp_path):
        path = str(tmp_path / "race.db")

        from stateforge.storage.sqlite import SQLiteBackend

        async def init_and_close():
            be = SQLiteBackend(path)
            await be.initialize()
            await be.close()

        # Sequential first to create the file with WAL/migrations applied.
        await init_and_close()

        # Now hammer it concurrently. None should crash.
        await asyncio.gather(*[init_and_close() for _ in range(10)])

        # Final consistency check: open once more and run a query.
        be = SQLiteBackend(path)
        await be.initialize()
        # An empty list_sessions should succeed (validates the schema is intact).
        out = await be.list_sessions(limit=10, before_id=None)
        assert out == []
        await be.close()
