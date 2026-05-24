"""Snapshot diffing.

Identity rule (spec § MemoryDiff): two units belong to the same logical
identity iff ``(session_id, key, type)`` matches. Modification is binary —
if value, metadata, or embedding differ, the unit is "modified".

Duplicate-identity tie-break: if a single snapshot contains multiple units
with the same identity, the latest by ``created_at`` wins.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from stateforge.models import DiffEntry, MemoryDiff, MemoryUnit
from stateforge.storage.base import StorageBackend


def _identity(unit: MemoryUnit) -> tuple[str, str, str]:
    return (unit.session_id, unit.key, unit.type.value)


def _identity_map(units: list[MemoryUnit]) -> dict[tuple[str, str, str], MemoryUnit]:
    """Build identity → unit, applying latest-wins for duplicates."""
    out: dict[tuple[str, str, str], MemoryUnit] = {}
    for u in sorted(units, key=lambda x: x.created_at):
        out[_identity(u)] = u
    return out


def _is_modified(before: MemoryUnit, after: MemoryUnit) -> bool:
    return (
        before.value != after.value
        or before.metadata != after.metadata
        or before.embedding != after.embedding
    )


async def _load_snapshot_units(
    backend: StorageBackend, snapshot_id: str
) -> list[MemoryUnit]:
    ids = await backend.read_snapshot_unit_ids(snapshot_id)
    return await backend.read_units(ids)


async def compute_diff_iter(
    backend: StorageBackend, from_snapshot_id: str, to_snapshot_id: str
) -> AsyncIterator[DiffEntry]:
    """Yield DiffEntry instances in deterministic order: added, modified, removed,
    each block sorted by identity. Both snapshots must be loaded (bounded by
    snapshot size, not by diff size).
    """
    from_units = await _load_snapshot_units(backend, from_snapshot_id)
    to_units = await _load_snapshot_units(backend, to_snapshot_id)
    from_map = _identity_map(from_units)
    to_map = _identity_map(to_units)

    added_ids = sorted(to_map.keys() - from_map.keys())
    removed_ids = sorted(from_map.keys() - to_map.keys())
    common_ids = sorted(to_map.keys() & from_map.keys())

    for identity in added_ids:
        yield DiffEntry(change="added", before=None, after=to_map[identity])

    for identity in common_ids:
        before = from_map[identity]
        after = to_map[identity]
        if _is_modified(before, after):
            yield DiffEntry(change="modified", before=before, after=after)

    for identity in removed_ids:
        yield DiffEntry(change="removed", before=from_map[identity], after=None)


async def compute_diff(
    backend: StorageBackend, from_snapshot_id: str, to_snapshot_id: str
) -> MemoryDiff:
    """Materialize the full diff. For small/medium snapshots."""
    added: list[MemoryUnit] = []
    removed: list[MemoryUnit] = []
    modified: list[tuple[MemoryUnit, MemoryUnit]] = []

    async for entry in compute_diff_iter(backend, from_snapshot_id, to_snapshot_id):
        if entry.change == "added":
            assert entry.after is not None
            added.append(entry.after)
        elif entry.change == "removed":
            assert entry.before is not None
            removed.append(entry.before)
        else:  # modified
            assert entry.before is not None and entry.after is not None
            modified.append((entry.before, entry.after))

    return MemoryDiff(
        from_snapshot_id=from_snapshot_id,
        to_snapshot_id=to_snapshot_id,
        added=added,
        removed=removed,
        modified=modified,
        created_at=datetime.now(timezone.utc),
    )
