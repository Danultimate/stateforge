from __future__ import annotations

import importlib.util

import pytest
import pytest_asyncio

LANGCHAIN_AVAILABLE = importlib.util.find_spec("langchain_core") is not None
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain-core not installed"
)

if LANGCHAIN_AVAILABLE:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from stateforge import StateForge
    from stateforge.adapters.langchain import StateForgeMessageHistory


@pytest_asyncio.fixture
async def history(sf):
    s = await sf.create_session(label="lc-test")
    return StateForgeMessageHistory(session_id=s.id, stateforge=sf)


# ────────────────────────────────────────────────────────────────────────────
# Construction
# ────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    async def test_defaults(self, sf):
        s = await sf.create_session()
        h = StateForgeMessageHistory(session_id=s.id, stateforge=sf)
        assert h.auto_snapshot is False
        assert h.snapshot_every == 1
        assert h.snapshot_label is None
        assert h.messages == []

    async def test_snapshot_every_zero_raises(self, sf):
        s = await sf.create_session()
        with pytest.raises(ValueError, match="snapshot_every"):
            StateForgeMessageHistory(
                session_id=s.id, stateforge=sf, snapshot_every=0
            )


# ────────────────────────────────────────────────────────────────────────────
# add / async semantics
# ────────────────────────────────────────────────────────────────────────────


class TestAdd:
    async def test_aadd_messages_appends_to_memory(self, history):
        await history.aadd_messages([HumanMessage(content="hello")])
        assert len(history.messages) == 1
        assert isinstance(history.messages[0], HumanMessage)
        assert history.messages[0].content == "hello"

    async def test_aadd_messages_does_not_snapshot_by_default(self, history, sf):
        await history.aadd_messages([HumanMessage(content="hi")])
        # auto_snapshot defaults to False; no snapshot exists yet.
        snaps = await sf.list_snapshots(history.session_id)
        assert snaps == []

    async def test_aadd_does_not_write_units_immediately(self, history, sf):
        """Adapter accumulates in-memory; nothing hits the DB until snapshot."""
        await history.aadd_messages([HumanMessage(content="A")])
        head = await sf.head(history.session_id)
        assert head is None

    async def test_sync_add_messages_in_running_loop_raises(self, history):
        from stateforge.exceptions import AdapterError

        with pytest.raises(AdapterError, match="event loop"):
            history.add_messages([HumanMessage(content="x")])


# ────────────────────────────────────────────────────────────────────────────
# Snapshot semantics
# ────────────────────────────────────────────────────────────────────────────


class TestSnapshot:
    async def test_explicit_snapshot_writes_all_pending(self, history, sf):
        await history.aadd_messages(
            [HumanMessage(content="a"), AIMessage(content="b")]
        )
        snap = await history.snapshot(label="cp-1")
        assert snap.label == "cp-1"
        units = await sf.get_units(snap.id)
        assert len(units) == 2

    async def test_repeated_snapshot_dedupes_via_insert_or_ignore(self, history, sf):
        # Two snapshots, same pending list. The second should produce a
        # snapshot referencing the same units without re-inserting them.
        await history.aadd_messages([HumanMessage(content="a")])
        snap1 = await history.snapshot()
        snap2 = await history.snapshot()
        ids1 = {u.id for u in await sf.get_units(snap1.id)}
        ids2 = {u.id for u in await sf.get_units(snap2.id)}
        assert ids1 == ids2  # same unit referenced by both


class TestAutoSnapshot:
    async def test_auto_snapshot_every_call(self, sf):
        s = await sf.create_session()
        h = StateForgeMessageHistory(
            session_id=s.id, stateforge=sf, auto_snapshot=True, snapshot_every=1
        )
        await h.aadd_messages([HumanMessage(content="1")])
        assert len((await sf.list_snapshots(s.id))) == 1
        await h.aadd_messages([HumanMessage(content="2")])
        assert len((await sf.list_snapshots(s.id))) == 2

    async def test_auto_snapshot_every_n(self, sf):
        s = await sf.create_session()
        h = StateForgeMessageHistory(
            session_id=s.id, stateforge=sf, auto_snapshot=True, snapshot_every=3
        )
        for i in range(2):
            await h.aadd_messages([HumanMessage(content=str(i))])
        # 2 calls < snapshot_every=3 → no snapshot yet
        assert (await sf.list_snapshots(s.id)) == []
        await h.aadd_messages([HumanMessage(content="3rd")])
        # 3rd call triggers snapshot
        snaps = await sf.list_snapshots(s.id)
        assert len(snaps) == 1
        units = await sf.get_units(snaps[0].id)
        assert len(units) == 3

    async def test_auto_snapshot_label_applied(self, sf):
        s = await sf.create_session()
        h = StateForgeMessageHistory(
            session_id=s.id, stateforge=sf,
            auto_snapshot=True, snapshot_every=1,
            snapshot_label="auto",
        )
        await h.aadd_messages([HumanMessage(content="x")])
        snaps = await sf.list_snapshots(s.id)
        assert snaps[0].label == "auto"


# ────────────────────────────────────────────────────────────────────────────
# Roundtrip
# ────────────────────────────────────────────────────────────────────────────


class TestRoundtrip:
    async def test_messages_survive_snapshot_and_reconstruct(self, sf):
        s = await sf.create_session()
        h = StateForgeMessageHistory(session_id=s.id, stateforge=sf)
        msgs = [
            SystemMessage(content="you are a helpful assistant"),
            HumanMessage(content="hi"),
            AIMessage(content="hello!"),
        ]
        await h.aadd_messages(msgs)
        snap = await h.snapshot(label="cp")

        # Reconstruct a fresh adapter from the snapshot.
        h2 = await StateForgeMessageHistory.from_snapshot(s.id, sf, snap.id)
        assert len(h2.messages) == 3
        # Order preserved by created_at.
        assert h2.messages[0].type == "system"
        assert h2.messages[1].type == "human"
        assert h2.messages[2].type == "ai"
        assert h2.messages[1].content == "hi"


# ────────────────────────────────────────────────────────────────────────────
# clear
# ────────────────────────────────────────────────────────────────────────────


class TestClear:
    async def test_aclear_empties_memory_and_writes_empty_snapshot(
        self, history, sf
    ):
        await history.aadd_messages([HumanMessage(content="a")])
        await history.snapshot()
        assert len(history.messages) == 1
        prev_snapshots = len(await sf.list_snapshots(history.session_id))

        await history.aclear()
        assert history.messages == []
        snaps = await sf.list_snapshots(history.session_id)
        assert len(snaps) == prev_snapshots + 1
        # The new head references no units.
        head = await sf.head(history.session_id)
        assert head is not None
        assert (await sf.get_units(head.id)) == []
