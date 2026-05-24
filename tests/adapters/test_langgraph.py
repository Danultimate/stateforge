from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pytest

LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None
pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE, reason="langgraph not installed"
)

if LANGGRAPH_AVAILABLE:
    from langgraph.checkpoint.base import (
        Checkpoint,
        CheckpointMetadata,
        empty_checkpoint,
    )

    from stateforge.adapters.langgraph import StateForgeCheckpointer


def _ckpt(checkpoint_id: str, channel_values: dict, **extra) -> "Checkpoint":
    base = empty_checkpoint()
    base["id"] = checkpoint_id
    base["ts"] = datetime.now(timezone.utc).isoformat()
    base["channel_values"] = channel_values
    base["channel_versions"] = {k: 1 for k in channel_values}
    base.update(extra)
    return base  # type: ignore[return-value]


def _md(step: int = 0) -> "CheckpointMetadata":
    return {"source": "loop", "step": step, "parents": {}}  # type: ignore[typeddict-item]


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict:
    conf = {"thread_id": thread_id}
    if checkpoint_id is not None:
        conf["checkpoint_id"] = checkpoint_id
    return {"configurable": conf}


# ────────────────────────────────────────────────────────────────────────────
# aput + aget_tuple roundtrip
# ────────────────────────────────────────────────────────────────────────────


class TestPutGet:
    async def test_put_then_get_latest(self, sf):
        cp = StateForgeCheckpointer(sf)
        config = _cfg("thread-A")
        ck = _ckpt("ck-1", {"counter": 1, "name": "alice"})
        new_config = await cp.aput(config, ck, _md(step=1), {"counter": 1, "name": 1})
        # New config has the checkpoint_id stamped in.
        assert new_config["configurable"]["checkpoint_id"] == "ck-1"

        tup = await cp.aget_tuple(_cfg("thread-A"))
        assert tup is not None
        assert tup.checkpoint["id"] == "ck-1"
        assert tup.checkpoint["channel_values"] == {"counter": 1, "name": "alice"}

    async def test_get_specific_checkpoint_by_id(self, sf):
        cp = StateForgeCheckpointer(sf)
        ck1 = _ckpt("ck-1", {"x": 1})
        ck2 = _ckpt("ck-2", {"x": 2})
        cfg1 = await cp.aput(_cfg("t"), ck1, _md(0), {"x": 1})
        await cp.aput(cfg1, ck2, _md(1), {"x": 2})

        # Latest
        tup = await cp.aget_tuple(_cfg("t"))
        assert tup.checkpoint["channel_values"]["x"] == 2

        # Specific
        tup1 = await cp.aget_tuple(_cfg("t", checkpoint_id="ck-1"))
        assert tup1.checkpoint["channel_values"]["x"] == 1
        # Parent linkage on ck-2
        tup2 = await cp.aget_tuple(_cfg("t", checkpoint_id="ck-2"))
        assert tup2.parent_config is not None
        assert tup2.parent_config["configurable"]["checkpoint_id"] == "ck-1"

    async def test_get_for_unknown_thread_returns_none(self, sf):
        cp = StateForgeCheckpointer(sf)
        tup = await cp.aget_tuple(_cfg("never-seen"))
        assert tup is None

    async def test_missing_thread_id_raises(self, sf):
        cp = StateForgeCheckpointer(sf)
        from stateforge.exceptions import AdapterError
        with pytest.raises(AdapterError):
            await cp.aget_tuple({"configurable": {}})


# ────────────────────────────────────────────────────────────────────────────
# Per-field shredding → per-field diff
# ────────────────────────────────────────────────────────────────────────────


class TestShredding:
    async def test_each_channel_becomes_its_own_unit(self, sf):
        cp = StateForgeCheckpointer(sf)
        ck = _ckpt("ck-1", {"a": 1, "b": "two", "c": [1, 2, 3]})
        await cp.aput(_cfg("t"), ck, _md(), {"a": 1, "b": 1, "c": 1})

        tup = await cp.aget_tuple(_cfg("t"))
        snap_id = tup.config["configurable"]["checkpoint_id"]
        # Lookup the snapshot by label == checkpoint_id
        snaps = await sf._find_snapshots_by_label(
            (await sf.head((await cp._session_for_thread("t")))).session_id, snap_id
        )
        assert len(snaps) == 1
        snap_units = await sf.get_units(snaps[0].id)
        keys = {u.key for u in snap_units}
        assert keys == {"a", "b", "c"}

    async def test_diff_reports_field_level_changes(self, sf):
        cp = StateForgeCheckpointer(sf)
        ck1 = _ckpt("ck-1", {"counter": 0, "name": "alice"})
        ck2 = _ckpt("ck-2", {"counter": 1, "name": "alice", "extra": True})
        cfg = await cp.aput(_cfg("t"), ck1, _md(), {"counter": 1, "name": 1})
        await cp.aput(cfg, ck2, _md(1), {"counter": 2, "name": 1, "extra": 1})

        # Find both snapshots and diff them.
        session_id = await cp._session_for_thread("t")
        snaps_1 = await sf._find_snapshots_by_label(session_id, "ck-1")
        snaps_2 = await sf._find_snapshots_by_label(session_id, "ck-2")
        d = await sf.diff(snaps_1[0].id, snaps_2[0].id)
        # counter: 0 → 1 (modified); extra: new (added); name: unchanged.
        added_keys = {u.key for u in d.added}
        modified_keys = {b.key for b, a in d.modified}
        removed_keys = {u.key for u in d.removed}
        assert added_keys == {"extra"}
        assert modified_keys == {"counter"}
        assert removed_keys == set()


# ────────────────────────────────────────────────────────────────────────────
# Thread → session resolution
# ────────────────────────────────────────────────────────────────────────────


class TestThreadMapping:
    async def test_existing_session_label_resolves(self, sf):
        # Pre-create a session labeled "thread-X".
        sess = await sf.create_session(label="thread-X")
        cp = StateForgeCheckpointer(sf)
        ck = _ckpt("ck-1", {"a": 1})
        await cp.aput(_cfg("thread-X"), ck, _md(), {"a": 1})

        # The adapter should have reused our session, not created a new one.
        listed = await sf.list_sessions()
        labels = [s.label for s in listed]
        # Only one "thread-X" session must exist.
        assert labels.count("thread-X") == 1
        assert (await cp._session_for_thread("thread-X")) == sess.id

    async def test_new_thread_creates_session(self, sf):
        cp = StateForgeCheckpointer(sf)
        before = len(await sf.list_sessions())
        ck = _ckpt("ck-1", {"a": 1})
        await cp.aput(_cfg("brand-new"), ck, _md(), {"a": 1})
        after = len(await sf.list_sessions())
        assert after == before + 1

    async def test_thread_resolution_is_cached(self, sf):
        cp = StateForgeCheckpointer(sf)
        sid1 = await cp._session_for_thread("t1")
        sid2 = await cp._session_for_thread("t1")
        assert sid1 == sid2
        # And only one session was created.
        sessions = [s for s in await sf.list_sessions() if s.label == "t1"]
        assert len(sessions) == 1


# ────────────────────────────────────────────────────────────────────────────
# alist
# ────────────────────────────────────────────────────────────────────────────


class TestAList:
    async def test_lists_in_order(self, sf):
        cp = StateForgeCheckpointer(sf)
        cfg = _cfg("t")
        cfg = await cp.aput(cfg, _ckpt("ck-1", {"x": 1}), _md(0), {"x": 1})
        cfg = await cp.aput(cfg, _ckpt("ck-2", {"x": 2}), _md(1), {"x": 2})
        cfg = await cp.aput(cfg, _ckpt("ck-3", {"x": 3}), _md(2), {"x": 3})

        ids = []
        async for tup in cp.alist(_cfg("t")):
            ids.append(tup.checkpoint["id"])
        # Newest-first per list_snapshots semantics.
        assert ids == ["ck-3", "ck-2", "ck-1"]


# ────────────────────────────────────────────────────────────────────────────
# Non-JSON-safe values rejected
# ────────────────────────────────────────────────────────────────────────────


class TestValueContract:
    async def test_bytes_channel_value_raises(self, sf):
        from stateforge.exceptions import AdapterError

        cp = StateForgeCheckpointer(sf)
        ck = _ckpt("ck-1", {"blob": b"\x00\x01"})
        with pytest.raises(AdapterError, match="non-JSON-safe"):
            await cp.aput(_cfg("t"), ck, _md(), {"blob": 1})


# ────────────────────────────────────────────────────────────────────────────
# aput_writes is a no-op
# ────────────────────────────────────────────────────────────────────────────


class TestPutWritesNoOp:
    async def test_does_nothing(self, sf):
        cp = StateForgeCheckpointer(sf)
        await cp.aput(_cfg("t"), _ckpt("ck-1", {"x": 1}), _md(), {"x": 1})
        before = len(await sf.list_snapshots(await cp._session_for_thread("t")))
        await cp.aput_writes(_cfg("t", "ck-1"), [("x", 99)], task_id="task-1")
        after = len(await sf.list_snapshots(await cp._session_for_thread("t")))
        assert before == after  # no new snapshot
