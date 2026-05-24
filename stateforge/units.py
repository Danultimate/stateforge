from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from stateforge.enums import MemoryUnitType
from stateforge.exceptions import ValueTypeError
from stateforge.models import MemoryUnit
from stateforge.value import validate_json_safe


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pack_embedding(vector: list[float] | bytes | None) -> bytes | None:
    if vector is None:
        return None
    if isinstance(vector, bytes):
        if len(vector) % 4 != 0:
            raise ValueTypeError(
                f"embedding bytes length {len(vector)} is not a multiple of 4 "
                "(expected packed float32 little-endian)"
            )
        return vector
    if isinstance(vector, list):
        for i, x in enumerate(vector):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueTypeError(
                    f"embedding[{i}] is not a number: {type(x).__name__}"
                )
        return struct.pack(f"<{len(vector)}f", *vector)
    raise ValueTypeError(
        f"embedding must be list[float] or bytes, got {type(vector).__name__}"
    )


def _normalize_metadata(metadata: dict | None) -> dict:
    md = {} if metadata is None else dict(metadata)
    validate_json_safe(md, path="metadata")
    return md


def _build(
    *,
    type: MemoryUnitType,
    session_id: str,
    key: str,
    value: Any,
    source: str,
    source_ref: str | None,
    metadata: dict | None,
    embedding: bytes | None = None,
) -> MemoryUnit:
    validate_json_safe(value, path="value")
    md = _normalize_metadata(metadata)
    return MemoryUnit(
        id=str(uuid4()),
        session_id=session_id,
        type=type,
        key=key,
        value=value,
        embedding=embedding,
        metadata=md,
        source=source,
        source_ref=source_ref,
        created_at=_now(),
    )


def message(
    session_id: str,
    key: str,
    value: Any,
    source: str = "user",
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> MemoryUnit:
    return _build(
        type=MemoryUnitType.MESSAGE,
        session_id=session_id,
        key=key,
        value=value,
        source=source,
        source_ref=source_ref,
        metadata=metadata,
    )


def kv(
    session_id: str,
    key: str,
    value: Any,
    source: str = "agent",
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> MemoryUnit:
    return _build(
        type=MemoryUnitType.KV,
        session_id=session_id,
        key=key,
        value=value,
        source=source,
        source_ref=source_ref,
        metadata=metadata,
    )


def embedding(
    session_id: str,
    key: str,
    vector: list[float] | bytes,
    source: str,
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> MemoryUnit:
    return _build(
        type=MemoryUnitType.EMBEDDING,
        session_id=session_id,
        key=key,
        value=None,
        source=source,
        source_ref=source_ref,
        metadata=metadata,
        embedding=_pack_embedding(vector),
    )


def tool_result(
    session_id: str,
    key: str,
    value: Any,
    source: str,
    source_ref: str | None,
    metadata: dict | None = None,
) -> MemoryUnit:
    return _build(
        type=MemoryUnitType.TOOL_RESULT,
        session_id=session_id,
        key=key,
        value=value,
        source=source,
        source_ref=source_ref,
        metadata=metadata,
    )


def summary(
    session_id: str,
    key: str,
    value: Any,
    source: str,
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> MemoryUnit:
    return _build(
        type=MemoryUnitType.SUMMARY,
        session_id=session_id,
        key=key,
        value=value,
        source=source,
        source_ref=source_ref,
        metadata=metadata,
    )
