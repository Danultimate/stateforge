"""Reference resolution for the CLI.

Three reference forms (resolution order is fixed):

1. ``head`` / ``head~N`` — relative to the current head of a session. Requires
   ``session_id`` context.
2. UUID prefix (8+ hex chars, optionally containing dashes). Globally unique
   when full; for prefixes, must match exactly one row.
3. Label (exact match). Requires ``session_id`` context for snapshots.

Ambiguous matches (multiple snapshots/sessions matching the same prefix or label)
raise :class:`AmbiguousRefError`. No match raises ``SnapshotNotFoundError`` or
``SessionNotFoundError``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stateforge.exceptions import (
    AmbiguousRefError,
    SessionNotFoundError,
    SnapshotNotFoundError,
)
from stateforge.models import Session, Snapshot

if TYPE_CHECKING:
    from stateforge.client import StateForge

# Minimum prefix length to disambiguate uuid4 ids. 8 chars of hex ≈ 32 bits
# which is collision-resistant enough for any session-sized population.
_MIN_PREFIX = 8

_HEAD_RE = re.compile(r"^head(?:~(\d+))?$")
# Matches a uuid prefix: hex (and optional dashes) of length >= 8 up to 36.
_UUID_PREFIX_RE = re.compile(r"^[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*$")


@dataclass(slots=True, frozen=True)
class ParsedHead:
    distance: int  # 0 for "head", N for "head~N"


def parse_head(ref: str) -> ParsedHead | None:
    """Return a ParsedHead if ``ref`` is ``head`` or ``head~N``, else None."""
    m = _HEAD_RE.match(ref)
    if m is None:
        return None
    n = m.group(1)
    return ParsedHead(distance=int(n) if n is not None else 0)


def looks_like_uuid_prefix(ref: str) -> bool:
    """True if ``ref`` is hex (and dashes) of at least 8 chars, no longer than 36."""
    if not (_MIN_PREFIX <= len(ref) <= 36):
        return False
    return bool(_UUID_PREFIX_RE.match(ref))


# ────────────────────────────────────────────────────────────────────────────
# Snapshot resolution
# ────────────────────────────────────────────────────────────────────────────


async def resolve_snapshot(
    sf: "StateForge", ref: str, session_id: str | None = None
) -> Snapshot:
    """Resolve a snapshot ref. Resolution order: head/head~N → uuid prefix → label.

    ``head`` / ``head~N`` and label refs require ``session_id``. A uuid prefix
    can be resolved without ``session_id`` (global lookup); if provided, the
    search is scoped to that session.
    """
    head_parsed = parse_head(ref)
    if head_parsed is not None:
        if session_id is None:
            raise ValueError(
                f"ref {ref!r} requires --session (head/head~N needs a session context)"
            )
        return await _resolve_head(sf, session_id, head_parsed.distance)

    if looks_like_uuid_prefix(ref):
        matches = await sf._find_snapshots_by_id_prefix(ref, session_id=session_id)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousRefError(
                f"snapshot ref {ref!r} matched {len(matches)} ids; "
                "give a longer prefix"
            )
        # No uuid match → fall through to label.

    if session_id is None:
        raise SnapshotNotFoundError(
            f"ref {ref!r} did not match a uuid prefix and no --session was "
            "given for label resolution"
        )
    label_matches = await sf._find_snapshots_by_label(session_id, ref)
    if len(label_matches) == 1:
        return label_matches[0]
    if len(label_matches) > 1:
        raise AmbiguousRefError(
            f"label {ref!r} matched {len(label_matches)} snapshots in session "
            f"{session_id!r}; use a uuid prefix instead"
        )
    raise SnapshotNotFoundError(
        f"no snapshot matches ref {ref!r} in session {session_id!r}"
    )


async def _resolve_head(sf: "StateForge", session_id: str, distance: int) -> Snapshot:
    """Walk ``distance`` parents back from the current head."""
    head = await sf.head(session_id)
    if head is None:
        raise SnapshotNotFoundError(
            f"session {session_id!r} has no head (no snapshots yet)"
        )
    if distance == 0:
        return head
    # Walk parents — load only what we need.
    chain = await sf.history(session_id, max_depth=distance + 1)
    if distance >= len(chain):
        raise SnapshotNotFoundError(
            f"head~{distance} is beyond the start of history "
            f"(chain length: {len(chain)})"
        )
    return chain[distance]


# ────────────────────────────────────────────────────────────────────────────
# Session resolution
# ────────────────────────────────────────────────────────────────────────────


async def resolve_session(sf: "StateForge", ref: str) -> Session:
    """Resolve a session ref. Order: uuid prefix → label.

    Sessions don't have head semantics, so head/head~N do not apply.
    """
    if looks_like_uuid_prefix(ref):
        matches = await sf._find_sessions_by_id_prefix(ref)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousRefError(
                f"session ref {ref!r} matched {len(matches)} ids; "
                "give a longer prefix"
            )

    label_matches = await sf._find_sessions_by_label(ref)
    if len(label_matches) == 1:
        return label_matches[0]
    if len(label_matches) > 1:
        raise AmbiguousRefError(
            f"label {ref!r} matched {len(label_matches)} sessions; "
            "use a uuid prefix instead"
        )
    raise SessionNotFoundError(f"no session matches ref {ref!r}")
