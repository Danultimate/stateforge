"""StateForge client — the user-facing async API.

The client owns:
- arg validation at ``__init__`` (synchronous; encryption key format errors fire here)
- lazy async initialization on first awaited call
- per-session async locks so the read-head + write-snapshot sequence is atomic
  in the asyncio domain (without these, two concurrent snapshot() calls on the
  same session could both observe the same head and create a branch)
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from stateforge.crypto import KeyMaterial, validate_key
from stateforge.diff import compute_diff, compute_diff_iter
from stateforge.exceptions import (
    CrossSessionRollbackError,
    HistoryTooDeepError,
    SessionNotFoundError,
)
from stateforge.models import (
    DiffEntry,
    MemoryDiff,
    MemoryUnit,
    ProvenanceRecord,
    Session,
    Snapshot,
)
from stateforge.storage.sqlite import SQLiteBackend
from stateforge.value import validate_json_safe

_ALLOWED_PRAGMAS = {"busy_timeout", "synchronous", "foreign_keys"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class StateForge:
    """Versioned memory and state management for AI agents.

    The constructor is synchronous and validates arguments eagerly. The backend
    is opened lazily on the first awaited call.
    """

    def __init__(
        self,
        db_path: str = "stateforge.db",
        *,
        auto_migrate: bool = True,
        pragmas: dict[str, str] | None = None,
        encryption_key: str | bytes | None = None,
        encryption_key_provider: Callable[[], str | bytes] | None = None,
    ) -> None:
        if encryption_key is not None and encryption_key_provider is not None:
            raise ValueError(
                "pass at most one of `encryption_key` or "
                "`encryption_key_provider`, not both"
            )

        if pragmas:
            bad = set(pragmas) - _ALLOWED_PRAGMAS
            if bad:
                raise ValueError(
                    f"pragmas {sorted(bad)!r} are not user-overridable; "
                    f"allowed keys are {sorted(_ALLOWED_PRAGMAS)!r}. "
                    "journal_mode and key are managed by the library."
                )

        # Resolve key now so EncryptionKeyFormatError fires at __init__,
        # before any DB open (per spec § Reliability § Connection setup).
        key_material: KeyMaterial | None = None
        if encryption_key is not None:
            key_material = validate_key(encryption_key)
        elif encryption_key_provider is not None:
            # Provider is called once and cached for the instance lifetime.
            key_material = validate_key(encryption_key_provider())

        self._db_path = db_path
        self._auto_migrate = auto_migrate
        self._backend = SQLiteBackend(db_path, key=key_material, pragmas=pragmas)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _ensure(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            if self._auto_migrate:
                await self._backend.initialize()
            else:
                # Still open the connection + PRAGMAs; skip migrations.
                # initialize() is currently the only entry point; for v0 we
                # only support auto_migrate=True. Document and gate.
                raise NotImplementedError(
                    "auto_migrate=False is not supported in v0"
                )
            self._initialized = True

    async def close(self) -> None:
        """Close the backend connection. Idempotent."""
        if self._initialized:
            await self._backend.close()
            self._initialized = False

    # ── Sessions ─────────────────────────────────────────────────────────

    async def create_session(
        self, label: str | None = None, metadata: dict | None = None
    ) -> Session:
        await self._ensure()
        md = {} if metadata is None else dict(metadata)
        validate_json_safe(md, path="metadata")
        sess = Session(
            id=_new_id(),
            label=label,
            head_snapshot_id=None,
            created_at=_now(),
            metadata=md,
        )
        await self._backend.create_session(sess)
        return sess

    async def get_session(self, session_id: str) -> Session:
        await self._ensure()
        return await self._backend.get_session(session_id)

    async def list_sessions(
        self, limit: int = 100, before_id: str | None = None
    ) -> list[Session]:
        await self._ensure()
        return await self._backend.list_sessions(limit, before_id)

    async def head(self, session_id: str) -> Snapshot | None:
        """Return the current head snapshot, or None if the session has no snapshots."""
        await self._ensure()
        sess = await self._backend.get_session(session_id)
        if sess.head_snapshot_id is None:
            return None
        return await self._backend.read_snapshot(sess.head_snapshot_id)

    # ── Snapshots ────────────────────────────────────────────────────────

    async def snapshot(
        self,
        session_id: str,
        units: list[MemoryUnit],
        label: str | None = None,
        metadata: dict | None = None,
    ) -> Snapshot:
        """Create a snapshot atomically.

        Writes units (deduped by id), creates the snapshot, inserts
        ``snapshot_units`` rows for every unit in ``units``, and advances
        ``sessions.head_snapshot_id``. parent_id is set to the previous head.

        Raises:
            SessionNotFoundError: ``session_id`` is unknown.
            ValueTypeError: ``metadata`` is not JSON-safe.
            StorageError: backend I/O / integrity failure.
        """
        await self._ensure()
        md = {} if metadata is None else dict(metadata)
        validate_json_safe(md, path="metadata")

        async with self._lock_for(session_id):
            # Read head INSIDE the per-session lock so parent_id resolution
            # cannot race with another snapshot() / rollback() on this session.
            sess = await self._backend.get_session(session_id)
            snap = Snapshot(
                id=_new_id(),
                session_id=session_id,
                label=label,
                parent_id=sess.head_snapshot_id,
                created_at=_now(),
                metadata=md,
            )
            await self._backend.write_snapshot(
                snap,
                new_units=units,
                unit_ids_to_link=[u.id for u in units],
            )
            return snap

    async def get_snapshot(self, snapshot_id: str) -> Snapshot:
        await self._ensure()
        return await self._backend.read_snapshot(snapshot_id)

    async def list_snapshots(
        self, session_id: str, limit: int = 100, before_id: str | None = None
    ) -> list[Snapshot]:
        await self._ensure()
        return await self._backend.list_snapshots(session_id, limit, before_id)

    async def get_units(self, snapshot_id: str) -> list[MemoryUnit]:
        await self._ensure()
        ids = await self._backend.read_snapshot_unit_ids(snapshot_id)
        return await self._backend.read_units(ids)

    async def get_unit(self, unit_id: str) -> MemoryUnit:
        await self._ensure()
        return await self._backend.read_unit(unit_id)

    # ── Diff ─────────────────────────────────────────────────────────────

    async def diff(
        self, from_snapshot_id: str, to_snapshot_id: str
    ) -> MemoryDiff:
        await self._ensure()
        return await compute_diff(self._backend, from_snapshot_id, to_snapshot_id)

    def diff_iter(
        self, from_snapshot_id: str, to_snapshot_id: str
    ) -> AsyncIterator[DiffEntry]:
        """Stream DiffEntry instances. Caller awaits ``self._ensure()``
        implicitly on the first yield."""
        async def _gen() -> AsyncIterator[DiffEntry]:
            await self._ensure()
            async for entry in compute_diff_iter(
                self._backend, from_snapshot_id, to_snapshot_id
            ):
                yield entry
        return _gen()

    # ── Rollback ─────────────────────────────────────────────────────────

    async def rollback(
        self,
        session_id: str,
        to_snapshot_id: str,
        label: str | None = None,
    ) -> Snapshot:
        """Create a new snapshot whose unit set equals ``to_snapshot_id``'s.

        Units are referenced, not copied. parent_id is the previous head.
        New snapshot becomes head. No data is deleted; rollback is itself a
        versioned event.

        Raises:
            SessionNotFoundError: ``session_id`` is unknown.
            SnapshotNotFoundError: ``to_snapshot_id`` does not exist.
            CrossSessionRollbackError: target belongs to a different session.
        """
        await self._ensure()

        async with self._lock_for(session_id):
            sess = await self._backend.get_session(session_id)
            target = await self._backend.read_snapshot(to_snapshot_id)
            if target.session_id != session_id:
                raise CrossSessionRollbackError(
                    f"snapshot {to_snapshot_id} belongs to session "
                    f"{target.session_id!r}, not {session_id!r}"
                )

            new_label = label or f"rollback to {to_snapshot_id[:8]}"
            new_snap = Snapshot(
                id=_new_id(),
                session_id=session_id,
                label=new_label,
                parent_id=sess.head_snapshot_id,
                created_at=_now(),
                metadata={},
            )
            await self._backend.write_rollback(new_snap, copy_from_id=to_snapshot_id)
            return new_snap

    # ── History ──────────────────────────────────────────────────────────

    async def history(
        self, session_id: str, max_depth: int = 10_000
    ) -> list[Snapshot]:
        """Walk head → root via parent_id. Bulk-loaded.

        Raises:
            SessionNotFoundError: ``session_id`` is unknown.
            HistoryTooDeepError: chain exceeded ``max_depth``.
        """
        await self._ensure()
        sess = await self._backend.get_session(session_id)
        if sess.head_snapshot_id is None:
            return []
        return await self._backend.walk_parents(sess.head_snapshot_id, max_depth)

    # ── Provenance ───────────────────────────────────────────────────────

    async def get_provenance(self, unit_id: str) -> ProvenanceRecord:
        await self._ensure()
        return await self._backend.read_provenance(unit_id)

    async def write_provenance(self, record: ProvenanceRecord) -> None:
        """Convenience for ingesting a provenance record.

        The spec § API Surface only documents the read side (``get_provenance``).
        Writing provenance is a backend-level operation but exposed here so the
        client surface is self-contained for adapters and tests.
        """
        await self._ensure()
        await self._backend.write_provenance(record)

    # ── Ref-resolution helpers (used by CLI; single-underscore semi-public) ─

    async def _find_snapshots_by_id_prefix(
        self, prefix: str, session_id: str | None = None
    ) -> list[Snapshot]:
        await self._ensure()
        return await self._backend.find_snapshots_by_id_prefix(
            prefix, session_id=session_id
        )

    async def _find_snapshots_by_label(
        self, session_id: str, label: str
    ) -> list[Snapshot]:
        await self._ensure()
        return await self._backend.find_snapshots_by_label(session_id, label)

    async def _find_sessions_by_id_prefix(self, prefix: str) -> list[Session]:
        await self._ensure()
        return await self._backend.find_sessions_by_id_prefix(prefix)

    async def _find_sessions_by_label(self, label: str) -> list[Session]:
        await self._ensure()
        return await self._backend.find_sessions_by_label(label)
