from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeAlias

from stateforge.enums import MemoryUnitType

JSONValue: TypeAlias = "str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]"


@dataclass(slots=True, frozen=True)
class Session:
    id: str
    label: str | None
    head_snapshot_id: str | None
    created_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MemoryUnit:
    id: str
    session_id: str
    type: MemoryUnitType
    key: str
    value: JSONValue
    embedding: bytes | None
    metadata: dict
    source: str
    source_ref: str | None
    created_at: datetime


@dataclass(slots=True, frozen=True)
class Snapshot:
    id: str
    session_id: str
    label: str | None
    parent_id: str | None
    created_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MemoryDiff:
    from_snapshot_id: str
    to_snapshot_id: str
    added: list[MemoryUnit]
    removed: list[MemoryUnit]
    modified: list[tuple[MemoryUnit, MemoryUnit]]
    created_at: datetime


@dataclass(slots=True, frozen=True)
class DiffEntry:
    change: Literal["added", "removed", "modified"]
    before: MemoryUnit | None
    after: MemoryUnit | None


@dataclass(slots=True, frozen=True)
class ProvenanceHop:
    index: int
    source: str
    source_ref: str | None


@dataclass(slots=True, frozen=True)
class ProvenanceRecord:
    id: str
    memory_unit_id: str
    source: str
    source_ref: str | None
    ingested_at: datetime
    trace: list[ProvenanceHop]
