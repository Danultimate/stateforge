"""SQLite storage backend for StateForge.

Supports two drivers:
- ``aiosqlite`` (plain SQLite) — default
- ``sqlcipher3`` (SQLCipher-encrypted) — used when a key is supplied

Both expose the same async surface through the internal :class:`_AsyncConn`.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import datetime
from typing import Any, AsyncIterator, Iterable

import aiosqlite

from stateforge.crypto import KeyMaterial, require_sqlcipher
from stateforge.enums import MemoryUnitType
from stateforge.exceptions import (
    EncryptionKeyError,
    HistoryTooDeepError,
    MemoryUnitNotFoundError,
    SessionNotFoundError,
    SnapshotNotFoundError,
    StorageError,
)
from stateforge.models import (
    MemoryUnit,
    ProvenanceHop,
    ProvenanceRecord,
    Session,
    Snapshot,
)
from stateforge.storage.base import StorageBackend

# SQLite has a default ~999 host-parameter limit. Stay well under it.
_PARAM_CHUNK = 500

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id                TEXT PRIMARY KEY,
        label             TEXT,
        head_snapshot_id  TEXT,
        created_at        TEXT NOT NULL,
        metadata          TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL REFERENCES sessions(id),
        label       TEXT,
        parent_id   TEXT REFERENCES snapshots(id),
        created_at  TEXT NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snapshots_session_created
        ON snapshots(session_id, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snapshots_parent
        ON snapshots(parent_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_units (
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL REFERENCES sessions(id),
        type        TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        embedding   BLOB,
        metadata    TEXT NOT NULL DEFAULT '{}',
        source      TEXT NOT NULL,
        source_ref  TEXT,
        created_at  TEXT NOT NULL
    )
    """,
    # Diff identity index. The Decision Log (D4) originally specified
    # (session_id, snapshot_id, key, type); the canonical-units model (D5)
    # removed snapshot_id from memory_units, so identity collapses to
    # (session_id, key, type). created_at is appended to break "duplicate
    # identity within a single snapshot" ties via latest-wins.
    """
    CREATE INDEX IF NOT EXISTS idx_units_identity
        ON memory_units(session_id, key, type, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_units_session
        ON memory_units(session_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot_units (
        snapshot_id     TEXT NOT NULL REFERENCES snapshots(id),
        memory_unit_id  TEXT NOT NULL REFERENCES memory_units(id),
        PRIMARY KEY (snapshot_id, memory_unit_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snapshot_units_unit
        ON snapshot_units(memory_unit_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS provenance (
        id              TEXT PRIMARY KEY,
        memory_unit_id  TEXT NOT NULL REFERENCES memory_units(id),
        source          TEXT NOT NULL,
        source_ref      TEXT,
        ingested_at     TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_provenance_unit
        ON provenance(memory_unit_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS provenance_hop (
        provenance_id  TEXT NOT NULL REFERENCES provenance(id),
        hop_index      INTEGER NOT NULL,
        source         TEXT NOT NULL,
        source_ref     TEXT,
        PRIMARY KEY (provenance_id, hop_index)
    )
    """,
]


# ────────────────────────────────────────────────────────────────────────────
# Internal connection abstraction
# ────────────────────────────────────────────────────────────────────────────


class _AsyncConn:
    """Driver-agnostic async wrapper. One instance per backend."""

    def __init__(self, path: str, key: KeyMaterial | None) -> None:
        self._path = path
        self._key = key
        self._aio: aiosqlite.Connection | None = None
        self._cipher: Any | None = None  # sqlcipher3 sync connection
        self._stmt_lock = asyncio.Lock()  # serialize sqlcipher per-statement ops
        self._txn_lock = asyncio.Lock()   # serialize BEGIN..COMMIT sequences

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Acquire the transaction lock and run a BEGIN IMMEDIATE..COMMIT block.

        aiosqlite serializes individual statements but not transaction
        sequences — two tasks can race to issue ``BEGIN IMMEDIATE`` on the same
        connection. This lock enforces asyncio-level atomicity. On exception,
        the transaction is rolled back and the error re-raises.
        """
        async with self._txn_lock:
            await self.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                await self.rollback()
                raise
            else:
                await self.commit()

    @property
    def encrypted(self) -> bool:
        return self._key is not None

    async def open(self) -> None:
        if self._key is None:
            self._aio = await aiosqlite.connect(self._path)
            # Make rows behave like tuples but with named access if needed.
            self._aio.row_factory = None
        else:
            require_sqlcipher()
            import sqlcipher3  # type: ignore[import-not-found]

            def _connect() -> Any:
                conn = sqlcipher3.dbapi2.connect(self._path)
                conn.isolation_level = None  # we manage txns explicitly
                return conn

            self._cipher = await asyncio.to_thread(_connect)

    async def close(self) -> None:
        if self._aio is not None:
            await self._aio.close()
            self._aio = None
        if self._cipher is not None:
            await asyncio.to_thread(self._cipher.close)
            self._cipher = None

    async def execute(self, sql: str, params: tuple = ()) -> None:
        if self._aio is not None:
            await self._aio.execute(sql, params)
            return
        async with self._stmt_lock:
            await asyncio.to_thread(self._cipher.execute, sql, params)  # type: ignore[union-attr]

    async def executemany(self, sql: str, params_list: Iterable[tuple]) -> None:
        params_list = list(params_list)
        if not params_list:
            return
        if self._aio is not None:
            await self._aio.executemany(sql, params_list)
            return
        async with self._stmt_lock:
            await asyncio.to_thread(
                self._cipher.executemany, sql, params_list  # type: ignore[union-attr]
            )

    async def fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self._aio is not None:
            async with self._aio.execute(sql, params) as cur:
                return list(await cur.fetchall())
        async with self._stmt_lock:
            def _run() -> list[tuple]:
                cur = self._cipher.execute(sql, params)  # type: ignore[union-attr]
                return list(cur.fetchall())
            return await asyncio.to_thread(_run)

    async def fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        if self._aio is not None:
            async with self._aio.execute(sql, params) as cur:
                return await cur.fetchone()
        async with self._stmt_lock:
            def _run() -> tuple | None:
                cur = self._cipher.execute(sql, params)  # type: ignore[union-attr]
                return cur.fetchone()
            return await asyncio.to_thread(_run)

    async def commit(self) -> None:
        if self._aio is not None:
            await self._aio.commit()
            return
        async with self._stmt_lock:
            await asyncio.to_thread(self._cipher.commit)  # type: ignore[union-attr]

    async def rollback(self) -> None:
        if self._aio is not None:
            await self._aio.rollback()
            return
        async with self._stmt_lock:
            await asyncio.to_thread(self._cipher.rollback)  # type: ignore[union-attr]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _row_to_session(row: tuple) -> Session:
    return Session(
        id=row[0],
        label=row[1],
        head_snapshot_id=row[2],
        created_at=_from_iso(row[3]),
        metadata=json.loads(row[4]),
    )


def _row_to_snapshot(row: tuple) -> Snapshot:
    return Snapshot(
        id=row[0],
        session_id=row[1],
        label=row[2],
        parent_id=row[3],
        created_at=_from_iso(row[4]),
        metadata=json.loads(row[5]),
    )


def _row_to_unit(row: tuple) -> MemoryUnit:
    return MemoryUnit(
        id=row[0],
        session_id=row[1],
        type=MemoryUnitType(row[2]),
        key=row[3],
        value=json.loads(row[4]),
        embedding=row[5],
        metadata=json.loads(row[6]),
        source=row[7],
        source_ref=row[8],
        created_at=_from_iso(row[9]),
    )


_SNAPSHOT_COLS = "id, session_id, label, parent_id, created_at, metadata"
_UNIT_COLS = "id, session_id, type, key, value, embedding, metadata, source, source_ref, created_at"
_SESSION_COLS = "id, label, head_snapshot_id, created_at, metadata"


# ────────────────────────────────────────────────────────────────────────────
# Backend
# ────────────────────────────────────────────────────────────────────────────


class SQLiteBackend(StorageBackend):
    """SQLite (and SQLCipher) storage backend.

    Concurrency: one connection per backend instance. aiosqlite runs SQL on a
    dedicated thread; sqlcipher3 ops are wrapped in ``asyncio.to_thread`` and
    serialized by an internal lock to mirror that thread-confinement model.
    """

    def __init__(
        self,
        db_path: str = "stateforge.db",
        *,
        key: KeyMaterial | None = None,
        pragmas: dict[str, str] | None = None,
    ) -> None:
        self._db_path = db_path
        self._key = key
        self._pragmas = pragmas or {}
        self._conn = _AsyncConn(db_path, key)
        self._initialized = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._initialized:
            return

        # Did the DB exist before we touched it? Used to decide whether to
        # probe the key on encrypted opens — a fresh DB has nothing to probe.
        pre_existing = self._db_path != ":memory:" and os.path.exists(self._db_path)

        await self._conn.open()

        # If ANY step below fails, the connection (and its worker thread) must
        # be closed before the exception propagates. Otherwise aiosqlite leaves
        # a non-daemon thread alive that blocks Python interpreter exit — which
        # manifests as a multi-minute hang at process teardown (the entire
        # test suite finishes in <2s but the process won't exit, blocking CI).
        try:
            # 1. PRAGMA key MUST precede every other statement, including journal_mode.
            if self._key is not None:
                await self._conn.execute(f"PRAGMA key = {self._key.pragma_value()}")
                # 2. Probe an existing DB to catch wrong/missing/extraneous-key cases.
                #    A fresh DB has no schema yet; the probe would always return 0.
                if pre_existing:
                    try:
                        await self._conn.fetchone(
                            "SELECT count(*) FROM sqlite_master"
                        )
                    except Exception as e:  # SQLCipher raises DatabaseError
                        raise EncryptionKeyError(
                            "failed to open encrypted DB — wrong key, missing key, "
                            f"or a key was supplied for an unencrypted DB: {e}"
                        ) from e

            # 3. Standard PRAGMAs (overridable except for journal_mode/key).
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute(
                f"PRAGMA busy_timeout = {self._pragmas.get('busy_timeout', '5000')}"
            )
            await self._conn.execute(
                f"PRAGMA synchronous = {self._pragmas.get('synchronous', 'NORMAL')}"
            )
            await self._conn.execute(
                f"PRAGMA foreign_keys = {self._pragmas.get('foreign_keys', 'ON')}"
            )

            # 4. Migrations under BEGIN IMMEDIATE — idempotent CREATE … IF NOT EXISTS.
            async with self._conn.transaction():
                for stmt in _SCHEMA:
                    await self._conn.execute(stmt)
        except BaseException:
            # Best-effort close so the worker thread terminates and the process
            # can exit cleanly. Swallow any close error — the original
            # exception is what the caller cares about.
            try:
                await self._conn.close()
            except Exception:
                pass
            raise

        self._initialized = True

    async def close(self) -> None:
        # Close even if initialize() never completed — the connection may have
        # been opened before the failing step, and its worker thread will
        # otherwise keep the process alive at exit.
        await self._conn.close()
        self._initialized = False

    # ── Sessions ─────────────────────────────────────────────────────────

    async def create_session(self, session: Session) -> None:
        await self._conn.execute(
            f"INSERT INTO sessions ({_SESSION_COLS}) VALUES (?, ?, ?, ?, ?)",
            (
                session.id,
                session.label,
                session.head_snapshot_id,
                _iso(session.created_at),
                json.dumps(session.metadata),
            ),
        )
        await self._conn.commit()

    async def get_session(self, session_id: str) -> Session:
        row = await self._conn.fetchone(
            f"SELECT {_SESSION_COLS} FROM sessions WHERE id = ?", (session_id,)
        )
        if row is None:
            raise SessionNotFoundError(session_id)
        return _row_to_session(row)

    async def list_sessions(
        self, limit: int, before_id: str | None
    ) -> list[Session]:
        if before_id is None:
            rows = await self._conn.fetchall(
                f"SELECT {_SESSION_COLS} FROM sessions "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        else:
            anchor = await self._conn.fetchone(
                "SELECT created_at FROM sessions WHERE id = ?", (before_id,)
            )
            if anchor is None:
                raise SessionNotFoundError(before_id)
            rows = await self._conn.fetchall(
                f"SELECT {_SESSION_COLS} FROM sessions WHERE created_at < ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (anchor[0], limit),
            )
        return [_row_to_session(r) for r in rows]

    async def set_head(self, session_id: str, snapshot_id: str) -> None:
        await self._conn.execute(
            "UPDATE sessions SET head_snapshot_id = ? WHERE id = ?",
            (snapshot_id, session_id),
        )
        await self._conn.commit()

    # ── Snapshot write (atomic) ──────────────────────────────────────────

    async def write_snapshot(
        self,
        snapshot: Snapshot,
        new_units: list[MemoryUnit],
        unit_ids_to_link: list[str],
    ) -> None:
        # Verify session exists before opening the transaction so the error is
        # raised cleanly without leaving an aborted txn behind.
        session_row = await self._conn.fetchone(
            "SELECT 1 FROM sessions WHERE id = ?", (snapshot.session_id,)
        )
        if session_row is None:
            raise SessionNotFoundError(snapshot.session_id)

        try:
            async with self._conn.transaction():
                if new_units:
                    await self._conn.executemany(
                        f"INSERT OR IGNORE INTO memory_units ({_UNIT_COLS}) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                u.id,
                                u.session_id,
                                u.type.value,
                                u.key,
                                json.dumps(u.value),
                                u.embedding,
                                json.dumps(u.metadata),
                                u.source,
                                u.source_ref,
                                _iso(u.created_at),
                            )
                            for u in new_units
                        ],
                    )

                await self._conn.execute(
                    f"INSERT INTO snapshots ({_SNAPSHOT_COLS}) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.id,
                        snapshot.session_id,
                        snapshot.label,
                        snapshot.parent_id,
                        _iso(snapshot.created_at),
                        json.dumps(snapshot.metadata),
                    ),
                )

                if unit_ids_to_link:
                    await self._conn.executemany(
                        "INSERT INTO snapshot_units (snapshot_id, memory_unit_id) "
                        "VALUES (?, ?)",
                        [(snapshot.id, uid) for uid in unit_ids_to_link],
                    )

                await self._conn.execute(
                    "UPDATE sessions SET head_snapshot_id = ? WHERE id = ?",
                    (snapshot.id, snapshot.session_id),
                )
        except Exception as e:
            raise StorageError(f"write_snapshot failed: {e}") from e

    async def read_snapshot(self, snapshot_id: str) -> Snapshot:
        row = await self._conn.fetchone(
            f"SELECT {_SNAPSHOT_COLS} FROM snapshots WHERE id = ?", (snapshot_id,)
        )
        if row is None:
            raise SnapshotNotFoundError(snapshot_id)
        return _row_to_snapshot(row)

    async def read_snapshot_unit_ids(self, snapshot_id: str) -> list[str]:
        # Confirm the snapshot exists so callers get a clear NotFound vs empty.
        exists = await self._conn.fetchone(
            "SELECT 1 FROM snapshots WHERE id = ?", (snapshot_id,)
        )
        if exists is None:
            raise SnapshotNotFoundError(snapshot_id)
        rows = await self._conn.fetchall(
            "SELECT memory_unit_id FROM snapshot_units WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        return [r[0] for r in rows]

    async def read_units(self, unit_ids: list[str]) -> list[MemoryUnit]:
        if not unit_ids:
            return []
        out: list[MemoryUnit] = []
        # Chunk to stay under SQLite's parameter limit.
        for i in range(0, len(unit_ids), _PARAM_CHUNK):
            chunk = unit_ids[i : i + _PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = await self._conn.fetchall(
                f"SELECT {_UNIT_COLS} FROM memory_units WHERE id IN ({placeholders})",
                tuple(chunk),
            )
            out.extend(_row_to_unit(r) for r in rows)
        return out

    async def read_unit(self, unit_id: str) -> MemoryUnit:
        row = await self._conn.fetchone(
            f"SELECT {_UNIT_COLS} FROM memory_units WHERE id = ?", (unit_id,)
        )
        if row is None:
            raise MemoryUnitNotFoundError(unit_id)
        return _row_to_unit(row)

    async def list_snapshots(
        self, session_id: str, limit: int, before_id: str | None
    ) -> list[Snapshot]:
        session_exists = await self._conn.fetchone(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        )
        if session_exists is None:
            raise SessionNotFoundError(session_id)

        if before_id is None:
            rows = await self._conn.fetchall(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots WHERE session_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            anchor = await self._conn.fetchone(
                "SELECT created_at FROM snapshots WHERE id = ? AND session_id = ?",
                (before_id, session_id),
            )
            if anchor is None:
                raise SnapshotNotFoundError(before_id)
            rows = await self._conn.fetchall(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE session_id = ? AND created_at < ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (session_id, anchor[0], limit),
            )
        return [_row_to_snapshot(r) for r in rows]

    async def walk_parents(
        self, head_id: str, max_depth: int
    ) -> list[Snapshot]:
        # Confirm the starting snapshot exists.
        exists = await self._conn.fetchone(
            "SELECT 1 FROM snapshots WHERE id = ?", (head_id,)
        )
        if exists is None:
            raise SnapshotNotFoundError(head_id)

        # Recursive CTE bounded by depth; we fetch (max_depth + 1) rows so we
        # can detect "still has a parent at the cap" and raise.
        rows = await self._conn.fetchall(
            f"""
            WITH RECURSIVE chain(id, session_id, label, parent_id, created_at, metadata, depth) AS (
                SELECT id, session_id, label, parent_id, created_at, metadata, 0
                FROM snapshots WHERE id = ?
                UNION ALL
                SELECT s.id, s.session_id, s.label, s.parent_id, s.created_at, s.metadata, c.depth + 1
                FROM snapshots s
                JOIN chain c ON s.id = c.parent_id
                WHERE c.depth < ?
            )
            SELECT {_SNAPSHOT_COLS} FROM chain ORDER BY depth ASC
            """,
            (head_id, max_depth),
        )
        snapshots = [_row_to_snapshot(r) for r in rows]
        # If the deepest row still has a parent_id, the chain exceeded max_depth.
        if snapshots and snapshots[-1].parent_id is not None and len(snapshots) > max_depth:
            raise HistoryTooDeepError(
                f"parent chain from {head_id} exceeded max_depth={max_depth}"
            )
        return snapshots

    # ── Rollback (atomic) ────────────────────────────────────────────────

    async def write_rollback(
        self, new_snapshot: Snapshot, copy_from_id: str
    ) -> None:
        # Validate source snapshot and same-session constraint outside the txn.
        source = await self._conn.fetchone(
            "SELECT session_id FROM snapshots WHERE id = ?", (copy_from_id,)
        )
        if source is None:
            raise SnapshotNotFoundError(copy_from_id)
        # CrossSessionRollbackError is the client's responsibility — backend just
        # writes what the client tells it. (See client.py for the check.)

        try:
            async with self._conn.transaction():
                await self._conn.execute(
                    f"INSERT INTO snapshots ({_SNAPSHOT_COLS}) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        new_snapshot.id,
                        new_snapshot.session_id,
                        new_snapshot.label,
                        new_snapshot.parent_id,
                        _iso(new_snapshot.created_at),
                        json.dumps(new_snapshot.metadata),
                    ),
                )

                await self._conn.execute(
                    "INSERT INTO snapshot_units (snapshot_id, memory_unit_id) "
                    "SELECT ?, memory_unit_id FROM snapshot_units "
                    "WHERE snapshot_id = ?",
                    (new_snapshot.id, copy_from_id),
                )

                await self._conn.execute(
                    "UPDATE sessions SET head_snapshot_id = ? WHERE id = ?",
                    (new_snapshot.id, new_snapshot.session_id),
                )
        except Exception as e:
            raise StorageError(f"write_rollback failed: {e}") from e

    # ── Provenance ───────────────────────────────────────────────────────

    async def write_provenance(self, record: ProvenanceRecord) -> None:
        # Confirm the referenced unit exists; FK would catch it but the error
        # message is nicer if we check first.
        exists = await self._conn.fetchone(
            "SELECT 1 FROM memory_units WHERE id = ?", (record.memory_unit_id,)
        )
        if exists is None:
            raise MemoryUnitNotFoundError(record.memory_unit_id)

        try:
            async with self._conn.transaction():
                await self._conn.execute(
                    "INSERT INTO provenance (id, memory_unit_id, source, source_ref, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.memory_unit_id,
                        record.source,
                        record.source_ref,
                        _iso(record.ingested_at),
                    ),
                )
                if record.trace:
                    await self._conn.executemany(
                        "INSERT INTO provenance_hop "
                        "(provenance_id, hop_index, source, source_ref) "
                        "VALUES (?, ?, ?, ?)",
                        [
                            (record.id, hop.index, hop.source, hop.source_ref)
                            for hop in record.trace
                        ],
                    )
        except Exception as e:
            raise StorageError(f"write_provenance failed: {e}") from e

    # ── Lookup helpers (not on the ABC — CLI ref-resolution support) ─────

    async def find_snapshots_by_id_prefix(
        self, prefix: str, *, session_id: str | None = None, limit: int = 100
    ) -> list[Snapshot]:
        """Return snapshots whose id starts with ``prefix`` (case-insensitive).

        Optionally scope to a single session. Used by CLI ref resolution.
        """
        if session_id is None:
            rows = await self._conn.fetchall(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots WHERE id LIKE ? LIMIT ?",
                (prefix.lower() + "%", limit),
            )
        else:
            rows = await self._conn.fetchall(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE session_id = ? AND id LIKE ? LIMIT ?",
                (session_id, prefix.lower() + "%", limit),
            )
        return [_row_to_snapshot(r) for r in rows]

    async def find_snapshots_by_label(
        self, session_id: str, label: str, *, limit: int = 100
    ) -> list[Snapshot]:
        rows = await self._conn.fetchall(
            f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
            "WHERE session_id = ? AND label = ? LIMIT ?",
            (session_id, label, limit),
        )
        return [_row_to_snapshot(r) for r in rows]

    async def find_sessions_by_id_prefix(
        self, prefix: str, *, limit: int = 100
    ) -> list[Session]:
        rows = await self._conn.fetchall(
            f"SELECT {_SESSION_COLS} FROM sessions WHERE id LIKE ? LIMIT ?",
            (prefix.lower() + "%", limit),
        )
        return [_row_to_session(r) for r in rows]

    async def find_sessions_by_label(
        self, label: str, *, limit: int = 100
    ) -> list[Session]:
        rows = await self._conn.fetchall(
            f"SELECT {_SESSION_COLS} FROM sessions WHERE label = ? LIMIT ?",
            (label, limit),
        )
        return [_row_to_session(r) for r in rows]

    async def read_provenance(self, unit_id: str) -> ProvenanceRecord:
        row = await self._conn.fetchone(
            "SELECT id, memory_unit_id, source, source_ref, ingested_at "
            "FROM provenance WHERE memory_unit_id = ? "
            "ORDER BY ingested_at DESC LIMIT 1",
            (unit_id,),
        )
        if row is None:
            raise MemoryUnitNotFoundError(unit_id)
        hop_rows = await self._conn.fetchall(
            "SELECT hop_index, source, source_ref FROM provenance_hop "
            "WHERE provenance_id = ? ORDER BY hop_index ASC",
            (row[0],),
        )
        return ProvenanceRecord(
            id=row[0],
            memory_unit_id=row[1],
            source=row[2],
            source_ref=row[3],
            ingested_at=_from_iso(row[4]),
            trace=[
                ProvenanceHop(index=h[0], source=h[1], source_ref=h[2])
                for h in hop_rows
            ],
        )
