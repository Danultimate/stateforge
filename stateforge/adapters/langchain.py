"""LangChain ``BaseChatMessageHistory`` adapter.

Each call to ``aadd_messages`` (or ``add_messages``) converts every incoming
``BaseMessage`` into a ``MemoryUnit(type=MESSAGE)``. Units are accumulated in
memory; snapshots commit the entire accumulated set so the on-disk story
matches the spec's "snapshots are full reference sets, not deltas" guarantee.

If ``auto_snapshot=True``, a snapshot is committed every ``snapshot_every``
calls. Otherwise the caller is responsible for invoking
:meth:`snapshot` explicitly.

Install: ``pip install stateforge-llm[langchain]``.
"""
from __future__ import annotations

from typing import Any, Sequence

from stateforge.client import StateForge
from stateforge.exceptions import AdapterError
from stateforge.models import MemoryUnit, Snapshot
from stateforge.units import message as _build_message_unit

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        BaseMessage,
        messages_from_dict,
        messages_to_dict,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only when extra is missing
    _LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object  # type: ignore[assignment,misc]
    BaseMessage = Any  # type: ignore[assignment,misc]


def _require() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise AdapterError(
            "langchain-core is not installed. "
            "Install with: pip install stateforge-llm[langchain]"
        )


class StateForgeMessageHistory(BaseChatMessageHistory):
    """Drop-in replacement for ``BaseChatMessageHistory`` backed by StateForge.

    Args:
        session_id: An existing StateForge session id. Must be created on the
            ``stateforge`` client before constructing the adapter.
        stateforge: The :class:`StateForge` client instance.
        auto_snapshot: If True, snapshot on every ``snapshot_every`` calls to
            ``add_messages`` / ``aadd_messages``. Default False — callers must
            invoke :meth:`snapshot` explicitly.
        snapshot_every: Commit cadence when ``auto_snapshot`` is True.
        snapshot_label: Optional label applied to auto-snapshots.
    """

    def __init__(
        self,
        session_id: str,
        stateforge: StateForge,
        *,
        auto_snapshot: bool = False,
        snapshot_every: int = 1,
        snapshot_label: str | None = None,
    ) -> None:
        _require()
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
        self.session_id = session_id
        self.sf = stateforge
        self.auto_snapshot = auto_snapshot
        self.snapshot_every = snapshot_every
        self.snapshot_label = snapshot_label

        # In-memory state. The adapter is single-process; cross-process recovery
        # is out of v0 scope.
        self._messages: list[BaseMessage] = []   # ordered messages
        self._units: list[MemoryUnit] = []        # parallel list, same length
        self._add_count = 0                       # # of add_messages calls

    # ── Read side ────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages added through this adapter instance."""
        return list(self._messages)

    # ── Write side: sync wrappers ────────────────────────────────────────

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Sync entrypoint. Delegates to :meth:`aadd_messages` via the client's
        synchronous bridge.

        LangChain's sync API is the one most users start with. The async API
        is preferred for performance in long-running agents.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise AdapterError(
                "add_messages() called from within a running event loop; "
                "use the async aadd_messages() instead."
            )
        asyncio.run(self.aadd_messages(messages))

    def clear(self) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise AdapterError(
                "clear() called from within a running event loop; "
                "use aclear() instead."
            )
        asyncio.run(self.aclear())

    # ── Write side: async API ────────────────────────────────────────────

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            d = messages_to_dict([msg])[0]
            u = _build_message_unit(
                self.session_id,
                key=f"msg:{len(self._messages)}",
                value=d,
                source="langchain",
                source_ref=msg.type,
            )
            self._messages.append(msg)
            self._units.append(u)

        self._add_count += 1
        if self.auto_snapshot and self._add_count % self.snapshot_every == 0:
            await self.snapshot()

    async def aclear(self) -> None:
        """Drop in-memory history and write an empty snapshot.

        The DB is not destructive: prior snapshots and their units remain on
        disk. The new head snapshot simply references no units.
        """
        self._messages = []
        self._units = []
        await self.sf.snapshot(self.session_id, units=[], label=self.snapshot_label)

    # ── Snapshot ─────────────────────────────────────────────────────────

    async def snapshot(self, label: str | None = None) -> Snapshot:
        """Commit the accumulated units as a single snapshot.

        Re-passes the full unit list every call — the backend's ``INSERT OR
        IGNORE`` on ``memory_units`` deduplicates on id, so this is cheap even
        when most units already exist.
        """
        applied_label = label if label is not None else self.snapshot_label
        return await self.sf.snapshot(
            self.session_id, units=list(self._units), label=applied_label
        )

    # ── Reconstruction (best-effort) ─────────────────────────────────────

    @classmethod
    async def from_snapshot(
        cls, session_id: str, stateforge: StateForge, snapshot_id: str
    ) -> "StateForgeMessageHistory":
        """Construct an adapter pre-populated from an existing snapshot's MESSAGE units.

        Useful for resuming a conversation after a process restart.
        """
        _require()
        snap_units = await stateforge.get_units(snapshot_id)
        msg_units = [
            u for u in snap_units
            if u.type.value == "message" and u.source == "langchain"
        ]
        # Sort by created_at to recover insertion order.
        msg_units.sort(key=lambda u: u.created_at)

        adapter = cls(session_id=session_id, stateforge=stateforge)
        for u in msg_units:
            # value is the messages_to_dict shape: {"type": ..., "data": {...}}
            try:
                msgs = messages_from_dict([u.value])
            except Exception:  # malformed legacy unit
                continue
            adapter._messages.extend(msgs)
            adapter._units.append(u)
        return adapter
