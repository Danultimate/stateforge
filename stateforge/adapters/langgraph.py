"""LangGraph async ``BaseCheckpointSaver`` adapter.

**Per-field shredding.** A LangGraph checkpoint's ``channel_values`` is a dict
of per-channel state. We do **not** store it as one opaque blob; each top-level
key becomes its own ``MemoryUnit(type=KV)`` with ``key=<channel_name>``. The
practical payoff: ``sf.diff(prev_checkpoint, curr_checkpoint)`` reports field-
by-field changes, so a developer can see *what changed in the agent's mind*,
not just "1 unit modified".

**Thread mapping.** LangGraph's ``thread_id`` becomes the session label. The
first time we see a thread we either resolve it to an existing session (by
label or by uuid match) or create one. A ``thread_id → session_id`` cache
avoids repeated lookups.

**Limitations (v0):**
- Channel values must be JSON-safe (str/int/float/bool/None/list/dict). The
  adapter rejects other types at write time. If your state contains
  ``BaseMessage`` objects, serialize via ``messages_to_dict`` first or use
  the LangChain adapter for chat history specifically.
- ``aput_writes`` (intermediate writes from parallel nodes) is a no-op. Only
  fully-committed checkpoints from ``aput`` are versioned.

Install: ``pip install stateforge-llm[langgraph]``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Sequence

from stateforge.client import StateForge
from stateforge.exceptions import (
    AdapterError,
    AmbiguousRefError,
    SessionNotFoundError,
    SnapshotNotFoundError,
    ValueTypeError,
)
from stateforge.models import Snapshot
from stateforge.refs import resolve_session
from stateforge.units import kv as _build_kv_unit

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object  # type: ignore[assignment,misc]
    Checkpoint = dict  # type: ignore[assignment,misc]
    CheckpointMetadata = dict  # type: ignore[assignment,misc]
    CheckpointTuple = tuple  # type: ignore[assignment,misc]


def _require() -> None:
    if not _LANGGRAPH_AVAILABLE:
        raise AdapterError(
            "langgraph is not installed. "
            "Install with: pip install stateforge-llm[langgraph]"
        )


_SOURCE = "langgraph"
# Sentinel source label distinguishing channel-value units from other KV units
# the user might have put on the same session.
_CHANNEL_SOURCE_REF = "channel"


class StateForgeCheckpointer(BaseCheckpointSaver):
    """Drop-in async checkpointer for ``StateGraph.compile(checkpointer=...)``.

    Args:
        stateforge: The :class:`StateForge` client instance.
    """

    def __init__(self, stateforge: StateForge) -> None:
        _require()
        super().__init__()
        self.sf = stateforge
        self._thread_to_session: dict[str, str] = {}
        self._cache_lock = asyncio.Lock()

    # ── thread_id → session_id resolution ────────────────────────────────

    async def _session_for_thread(self, thread_id: str) -> str:
        if thread_id in self._thread_to_session:
            return self._thread_to_session[thread_id]
        async with self._cache_lock:
            if thread_id in self._thread_to_session:
                return self._thread_to_session[thread_id]
            try:
                sess = await resolve_session(self.sf, thread_id)
                session_id = sess.id
            except (SessionNotFoundError, AmbiguousRefError):
                created = await self.sf.create_session(label=thread_id)
                session_id = created.id
            self._thread_to_session[thread_id] = session_id
            return session_id

    @staticmethod
    def _thread_id(config: dict) -> str:
        try:
            return config["configurable"]["thread_id"]
        except (KeyError, TypeError) as e:
            raise AdapterError(
                "RunnableConfig.configurable.thread_id is required"
            ) from e

    @staticmethod
    def _checkpoint_id(config: dict) -> str | None:
        try:
            return config.get("configurable", {}).get("checkpoint_id")
        except AttributeError:
            return None

    # ── aput (write a checkpoint) ────────────────────────────────────────

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,  # type: ignore[valid-type]
        metadata: CheckpointMetadata,  # type: ignore[valid-type]
        new_versions: Any,
    ) -> dict:
        thread_id = self._thread_id(config)
        session_id = await self._session_for_thread(thread_id)

        # Shred channel_values into per-channel KV units.
        channel_values: dict[str, Any] = dict(checkpoint.get("channel_values", {}))
        units_list = []
        for channel_name, value in channel_values.items():
            try:
                u = _build_kv_unit(
                    session_id,
                    key=channel_name,
                    value=value,
                    source=_SOURCE,
                    source_ref=_CHANNEL_SOURCE_REF,
                )
            except ValueTypeError as e:
                raise AdapterError(
                    f"channel {channel_name!r} contains a non-JSON-safe value; "
                    "serialize it before checkpointing "
                    f"(underlying error: {e})"
                ) from e
            units_list.append(u)

        # Snapshot metadata captures the non-channel-values parts of the checkpoint.
        snap_metadata = {
            "checkpoint_id": checkpoint["id"],
            "checkpoint_ts": checkpoint["ts"],
            "checkpoint_v": checkpoint["v"],
            "channel_versions": _safe_json(checkpoint.get("channel_versions", {})),
            "versions_seen": _safe_json(checkpoint.get("versions_seen", {})),
            "updated_channels": _safe_json(checkpoint.get("updated_channels")),
            "lg_metadata": _safe_json(dict(metadata) if metadata else {}),
            "new_versions": _safe_json(_dictify(new_versions)),
            "parent_checkpoint_id": self._checkpoint_id(config),
        }

        await self.sf.snapshot(
            session_id,
            units=units_list,
            label=checkpoint["id"],
            metadata=snap_metadata,
        )

        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
            },
        }

    # ── aget_tuple (read latest or specific) ─────────────────────────────

    async def aget_tuple(self, config: dict) -> "CheckpointTuple | None":  # type: ignore[name-defined]
        thread_id = self._thread_id(config)
        try:
            session_id = await self._session_for_thread(thread_id)
        except SessionNotFoundError:
            return None

        checkpoint_id = self._checkpoint_id(config)
        if checkpoint_id is not None:
            matches = await self.sf._find_snapshots_by_label(session_id, checkpoint_id)
            if not matches:
                return None
            snap = matches[0]
        else:
            snap = await self.sf.head(session_id)
            if snap is None:
                return None

        return await self._snapshot_to_tuple(snap, thread_id)

    # ── alist (paginated history) ────────────────────────────────────────

    async def alist(
        self,
        config: dict | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> AsyncIterator["CheckpointTuple"]:  # type: ignore[name-defined]
        if config is None:
            return
        thread_id = self._thread_id(config)
        try:
            session_id = await self._session_for_thread(thread_id)
        except SessionNotFoundError:
            return

        before_snapshot_id: str | None = None
        if before is not None:
            before_ck = self._checkpoint_id(before)
            if before_ck is not None:
                matches = await self.sf._find_snapshots_by_label(session_id, before_ck)
                if matches:
                    before_snapshot_id = matches[0].id

        snaps = await self.sf.list_snapshots(
            session_id, limit=limit or 100, before_id=before_snapshot_id
        )
        for snap in snaps:
            tup = await self._snapshot_to_tuple(snap, thread_id)
            if tup is not None:
                yield tup

    # ── aput_writes (intermediate writes — v0 no-op) ─────────────────────

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # v0 does not version intermediate writes from parallel branches.
        # Only fully-committed checkpoints from aput() are recorded.
        return None

    # ── Sync wrappers ────────────────────────────────────────────────────

    def put(self, config, checkpoint, metadata, new_versions):
        return _run_sync(self.aput(config, checkpoint, metadata, new_versions))

    def get_tuple(self, config):
        return _run_sync(self.aget_tuple(config))

    def list(self, config, *, filter=None, before=None, limit=None):
        async def _collect():
            out = []
            async for t in self.alist(
                config, filter=filter, before=before, limit=limit
            ):
                out.append(t)
            return out
        return iter(_run_sync(_collect()))

    def put_writes(self, config, writes, task_id, task_path=""):
        return _run_sync(
            self.aput_writes(config, writes, task_id, task_path)
        )

    # ── Internal reconstruction ──────────────────────────────────────────

    async def _snapshot_to_tuple(
        self, snap: Snapshot, thread_id: str
    ) -> "CheckpointTuple | None":  # type: ignore[name-defined]
        snap_units = await self.sf.get_units(snap.id)
        channel_values: dict[str, Any] = {
            u.key: u.value
            for u in snap_units
            if u.source == _SOURCE and u.source_ref == _CHANNEL_SOURCE_REF
        }
        md = snap.metadata
        checkpoint: Checkpoint = {  # type: ignore[typeddict-item]
            "v": md.get("checkpoint_v", 1),
            "id": md.get("checkpoint_id", snap.id),
            "ts": md.get("checkpoint_ts", snap.created_at.isoformat()),
            "channel_values": channel_values,
            "channel_versions": md.get("channel_versions", {}),
            "versions_seen": md.get("versions_seen", {}),
            "updated_channels": md.get("updated_channels"),
        }
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
            }
        }
        parent_ck = md.get("parent_checkpoint_id")
        parent_config = (
            {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_ck}}
            if parent_ck
            else None
        )
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=md.get("lg_metadata", {}),
            parent_config=parent_config,
            pending_writes=None,
        )


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        raise AdapterError(
            "sync method called from within a running event loop; "
            "use the async variant"
        )
    return asyncio.run(coro)


def _dictify(obj: Any) -> Any:
    """Best-effort: convert mapping-like objects to dicts."""
    if hasattr(obj, "items"):
        return {k: v for k, v in obj.items()}
    return obj


def _safe_json(obj: Any) -> Any:
    """Recursively convert to a JSON-safe representation.

    Anything that can't be naturally serialized falls back to ``repr()``. This
    is best-effort — adapter metadata is not load-bearing for diff semantics,
    so lossy serialization here is acceptable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)
