from __future__ import annotations

import pytest

from stateforge.exceptions import (
    AmbiguousRefError,
    SessionNotFoundError,
    SnapshotNotFoundError,
)
from stateforge.refs import (
    looks_like_uuid_prefix,
    parse_head,
    resolve_session,
    resolve_snapshot,
)


# ────────────────────────────────────────────────────────────────────────────
# Pure parsing
# ────────────────────────────────────────────────────────────────────────────


class TestParseHead:
    def test_head(self):
        assert parse_head("head") == parse_head("head")
        p = parse_head("head")
        assert p is not None
        assert p.distance == 0

    @pytest.mark.parametrize("ref,expected", [
        ("head~0", 0), ("head~1", 1), ("head~5", 5), ("head~999", 999),
    ])
    def test_head_tilde(self, ref, expected):
        p = parse_head(ref)
        assert p is not None
        assert p.distance == expected

    @pytest.mark.parametrize("ref", [
        "HEAD", "Head", "head~", "head~-1", "head~a", "main", "12345678",
        "head~1~2", "headlabel",
    ])
    def test_not_head(self, ref):
        assert parse_head(ref) is None


class TestUUIDPrefix:
    @pytest.mark.parametrize("ref", [
        "11111111", "11111111-1111", "11223344-5566-4778-89aa-bbccddeeff00",
        "AABBCCDD",
    ])
    def test_valid_prefix(self, ref):
        assert looks_like_uuid_prefix(ref)

    @pytest.mark.parametrize("ref", [
        "", "1234567", "abc",                       # too short
        "g" * 8, "head", "head~1", "my-label",      # non-hex
        "1" * 40,                                   # too long
    ])
    def test_invalid_prefix(self, ref):
        assert not looks_like_uuid_prefix(ref)


# ────────────────────────────────────────────────────────────────────────────
# Snapshot resolution
# ────────────────────────────────────────────────────────────────────────────


class TestResolveSnapshot:
    async def test_full_uuid(self, sf):
        s = await sf.create_session()
        snap = await sf.snapshot(s.id, units=[])
        got = await resolve_snapshot(sf, snap.id)
        assert got.id == snap.id

    async def test_uuid_prefix_8_chars(self, sf):
        s = await sf.create_session()
        snap = await sf.snapshot(s.id, units=[])
        got = await resolve_snapshot(sf, snap.id[:8])
        assert got.id == snap.id

    async def test_label_with_session_context(self, sf):
        s = await sf.create_session()
        snap = await sf.snapshot(s.id, units=[], label="my-checkpoint")
        got = await resolve_snapshot(sf, "my-checkpoint", session_id=s.id)
        assert got.id == snap.id

    async def test_label_without_session_raises(self, sf):
        s = await sf.create_session()
        await sf.snapshot(s.id, units=[], label="some-label")
        with pytest.raises(SnapshotNotFoundError):
            await resolve_snapshot(sf, "some-label")  # no session ctx

    async def test_head_resolves(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        got = await resolve_snapshot(sf, "head", session_id=s.id)
        assert got.id == snap2.id

    async def test_head_tilde_walks_parents(self, sf):
        s = await sf.create_session()
        snap1 = await sf.snapshot(s.id, units=[])
        snap2 = await sf.snapshot(s.id, units=[])
        snap3 = await sf.snapshot(s.id, units=[])
        assert (await resolve_snapshot(sf, "head", session_id=s.id)).id == snap3.id
        assert (await resolve_snapshot(sf, "head~1", session_id=s.id)).id == snap2.id
        assert (await resolve_snapshot(sf, "head~2", session_id=s.id)).id == snap1.id

    async def test_head_beyond_root_raises(self, sf):
        s = await sf.create_session()
        await sf.snapshot(s.id, units=[])
        with pytest.raises(SnapshotNotFoundError, match="beyond the start"):
            await resolve_snapshot(sf, "head~99", session_id=s.id)

    async def test_head_requires_session(self, sf):
        with pytest.raises(ValueError, match="requires --session"):
            await resolve_snapshot(sf, "head")

    async def test_head_on_empty_session_raises(self, sf):
        s = await sf.create_session()
        with pytest.raises(SnapshotNotFoundError, match="no head"):
            await resolve_snapshot(sf, "head", session_id=s.id)

    async def test_ambiguous_label_raises(self, sf):
        s = await sf.create_session()
        await sf.snapshot(s.id, units=[], label="dup")
        await sf.snapshot(s.id, units=[], label="dup")
        with pytest.raises(AmbiguousRefError, match="dup"):
            await resolve_snapshot(sf, "dup", session_id=s.id)

    async def test_uuid_prefix_falls_back_to_label_when_no_match(self, sf):
        """If a string is a valid hex prefix but no snapshot id matches, AND
        a snapshot has that string as its label, label resolution kicks in."""
        s = await sf.create_session()
        # Use a label that *looks* like a uuid prefix but doesn't match any real id.
        # Since uuid4 is random, "abcdef00" is extraordinarily unlikely to collide.
        snap = await sf.snapshot(s.id, units=[], label="abcdef00")
        got = await resolve_snapshot(sf, "abcdef00", session_id=s.id)
        assert got.id == snap.id

    async def test_no_match_raises_not_found(self, sf):
        s = await sf.create_session()
        with pytest.raises(SnapshotNotFoundError):
            await resolve_snapshot(sf, "no-such-thing", session_id=s.id)


# ────────────────────────────────────────────────────────────────────────────
# Session resolution
# ────────────────────────────────────────────────────────────────────────────


class TestResolveSession:
    async def test_full_uuid(self, sf):
        s = await sf.create_session()
        got = await resolve_session(sf, s.id)
        assert got.id == s.id

    async def test_uuid_prefix(self, sf):
        s = await sf.create_session()
        got = await resolve_session(sf, s.id[:8])
        assert got.id == s.id

    async def test_label(self, sf):
        s = await sf.create_session(label="run-42")
        got = await resolve_session(sf, "run-42")
        assert got.id == s.id

    async def test_ambiguous_label_raises(self, sf):
        await sf.create_session(label="dup")
        await sf.create_session(label="dup")
        with pytest.raises(AmbiguousRefError):
            await resolve_session(sf, "dup")

    async def test_no_match_raises(self, sf):
        with pytest.raises(SessionNotFoundError):
            await resolve_session(sf, "nope")
