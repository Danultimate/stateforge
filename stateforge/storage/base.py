from __future__ import annotations

from abc import ABC, abstractmethod

from stateforge.models import (
    MemoryUnit,
    ProvenanceRecord,
    Session,
    Snapshot,
)


class StorageBackend(ABC):
    """Narrow storage interface — only what v0 SQLite actually performs.

    Snapshot membership is defined by the (snapshot_id, memory_unit_id) junction.
    MemoryUnits are canonical and immutable; snapshots are reference sets.
    The head of a session is tracked on the session row.
    """

    # ─── Lifecycle ───────────────────────────────────────────────────────
    @abstractmethod
    async def initialize(self) -> None:
        """Open the connection, apply PRAGMAs, run idempotent migrations."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection. Idempotent."""

    # ─── Sessions ────────────────────────────────────────────────────────
    @abstractmethod
    async def create_session(self, session: Session) -> None: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session: ...

    @abstractmethod
    async def list_sessions(
        self, limit: int, before_id: str | None
    ) -> list[Session]: ...

    @abstractmethod
    async def set_head(self, session_id: str, snapshot_id: str) -> None:
        """Update sessions.head_snapshot_id. Called inside an outer transaction."""

    # ─── Snapshots + units (atomic write) ────────────────────────────────
    @abstractmethod
    async def write_snapshot(
        self,
        snapshot: Snapshot,
        new_units: list[MemoryUnit],
        unit_ids_to_link: list[str],
    ) -> None:
        """Atomic: insert new units (skipping any whose id already exists),
        insert the snapshot row, insert snapshot_units rows linking
        ``unit_ids_to_link``, and advance sessions.head_snapshot_id.

        ``new_units`` is the subset to persist; ``unit_ids_to_link`` is the full
        membership of the snapshot (must include the ids of all ``new_units``
        plus any previously-persisted unit ids carried forward).
        """

    @abstractmethod
    async def read_snapshot(self, snapshot_id: str) -> Snapshot: ...

    @abstractmethod
    async def read_snapshot_unit_ids(self, snapshot_id: str) -> list[str]: ...

    @abstractmethod
    async def read_units(self, unit_ids: list[str]) -> list[MemoryUnit]: ...

    @abstractmethod
    async def read_unit(self, unit_id: str) -> MemoryUnit: ...

    @abstractmethod
    async def list_snapshots(
        self, session_id: str, limit: int, before_id: str | None
    ) -> list[Snapshot]: ...

    @abstractmethod
    async def walk_parents(
        self, head_id: str, max_depth: int
    ) -> list[Snapshot]:
        """Walk parent chain from ``head_id`` toward root. Bulk-loaded.

        Raises HistoryTooDeepError if the chain exceeds ``max_depth``.
        """

    # ─── Rollback (atomic) ───────────────────────────────────────────────
    @abstractmethod
    async def write_rollback(
        self, new_snapshot: Snapshot, copy_from_id: str
    ) -> None:
        """Atomic: insert ``new_snapshot``, copy snapshot_units from
        ``copy_from_id`` into ``new_snapshot.id``, advance head.
        """

    # ─── Provenance ──────────────────────────────────────────────────────
    @abstractmethod
    async def write_provenance(self, record: ProvenanceRecord) -> None: ...

    @abstractmethod
    async def read_provenance(self, unit_id: str) -> ProvenanceRecord: ...
