from __future__ import annotations

import importlib.util
import os
import secrets
from datetime import datetime, timezone

import pytest

from stateforge import units
from stateforge.crypto import (
    KeyMaterial,
    is_sqlcipher_available,
    require_sqlcipher,
    validate_key,
)
from stateforge.exceptions import (
    EncryptionKeyError,
    EncryptionKeyFormatError,
    EncryptionUnavailableError,
)
from stateforge.models import Session, Snapshot
from stateforge.storage.sqlite import SQLiteBackend

SQLCIPHER_AVAILABLE = importlib.util.find_spec("sqlcipher3") is not None
sqlcipher_only = pytest.mark.skipif(
    not SQLCIPHER_AVAILABLE,
    reason="sqlcipher3-binary not installed; run `pip install stateforge-llm[encryption]`",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ────────────────────────────────────────────────────────────────────────────
# Key validation (no sqlcipher3 required)
# ────────────────────────────────────────────────────────────────────────────


class TestKeyValidation:
    def test_valid_hex(self):
        hex_key = "0" * 63 + "f"
        km = validate_key(hex_key)
        assert isinstance(km, KeyMaterial)
        assert len(km.raw) == 32
        assert km.hex == hex_key

    def test_valid_hex_uppercase(self):
        hex_key = "A" * 64
        km = validate_key(hex_key)
        assert len(km.raw) == 32

    def test_valid_raw_bytes(self):
        raw = secrets.token_bytes(32)
        km = validate_key(raw)
        assert km.raw == raw

    def test_pragma_value_uses_raw_form(self):
        km = validate_key("0" * 64)
        # Raw form so SQLCipher skips PBKDF2 derivation.
        assert km.pragma_value() == "\"x'" + "0" * 64 + "'\""

    @pytest.mark.parametrize(
        "bad",
        ["", "abc", "0" * 63, "0" * 65, "g" * 64, "0" * 32],
    )
    def test_bad_hex_strings_rejected(self, bad):
        with pytest.raises(EncryptionKeyFormatError):
            validate_key(bad)

    @pytest.mark.parametrize("length", [0, 16, 31, 33, 64])
    def test_wrong_byte_length_rejected(self, length):
        with pytest.raises(EncryptionKeyFormatError):
            validate_key(b"\x00" * length)

    def test_non_str_or_bytes_rejected(self):
        with pytest.raises(EncryptionKeyFormatError):
            validate_key(12345)  # type: ignore[arg-type]
        with pytest.raises(EncryptionKeyFormatError):
            validate_key(None)  # type: ignore[arg-type]
        with pytest.raises(EncryptionKeyFormatError):
            validate_key([0] * 32)  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────────────
# Driver availability detection
# ────────────────────────────────────────────────────────────────────────────


class TestDriverDetection:
    def test_is_available_returns_bool(self):
        assert isinstance(is_sqlcipher_available(), bool)

    def test_require_raises_when_unavailable(self, monkeypatch):
        # Force-fake the absence regardless of what's actually installed.
        monkeypatch.setattr(
            "stateforge.crypto.is_sqlcipher_available", lambda: False
        )
        with pytest.raises(EncryptionUnavailableError):
            require_sqlcipher()


# ────────────────────────────────────────────────────────────────────────────
# Backend init with key, sqlcipher3 missing
# ────────────────────────────────────────────────────────────────────────────


class TestBackendUnavailable:
    async def test_init_with_key_no_sqlcipher_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "stateforge.crypto.is_sqlcipher_available", lambda: False
        )
        km = validate_key("0" * 64)
        be = SQLiteBackend(str(tmp_path / "x.db"), key=km)
        with pytest.raises(EncryptionUnavailableError):
            await be.initialize()


# ────────────────────────────────────────────────────────────────────────────
# End-to-end encrypted operations (require sqlcipher3)
# ────────────────────────────────────────────────────────────────────────────


@sqlcipher_only
class TestEncryptedRoundtrip:
    async def test_create_close_reopen(self, tmp_path):
        path = str(tmp_path / "encrypted.db")
        key = validate_key(secrets.token_hex(32))

        # Open #1 — create encrypted DB, write a session.
        be1 = SQLiteBackend(path, key=key)
        await be1.initialize()
        sess = Session(
            id="11111111-1111-4111-8111-111111111111",
            label="enc-test",
            head_snapshot_id=None,
            created_at=_now(),
            metadata={"secret": True},
        )
        await be1.create_session(sess)
        u = units.message(sess.id, "msg:0", "confidential", source="user")
        snap = Snapshot(
            id="22222222-2222-4222-8222-222222222222",
            session_id=sess.id,
            label="s1",
            parent_id=None,
            created_at=_now(),
            metadata={},
        )
        await be1.write_snapshot(snap, new_units=[u], unit_ids_to_link=[u.id])
        await be1.close()

        # Open #2 — reopen with same key, read back.
        be2 = SQLiteBackend(path, key=key)
        await be2.initialize()
        got_sess = await be2.get_session(sess.id)
        assert got_sess.head_snapshot_id == snap.id
        got_unit = await be2.read_unit(u.id)
        assert got_unit.value == "confidential"
        await be2.close()

    async def test_wrong_key_raises(self, tmp_path):
        path = str(tmp_path / "encrypted.db")
        k1 = validate_key(secrets.token_hex(32))
        k2 = validate_key(secrets.token_hex(32))  # different

        # Create with k1
        be = SQLiteBackend(path, key=k1)
        await be.initialize()
        sess = Session(
            id="33333333-3333-4333-8333-333333333333",
            label=None, head_snapshot_id=None, created_at=_now(), metadata={},
        )
        await be.create_session(sess)
        await be.close()

        # Reopen with k2 — must fail on the post-key probe.
        be_bad = SQLiteBackend(path, key=k2)
        with pytest.raises(EncryptionKeyError):
            await be_bad.initialize()

    async def test_no_key_on_encrypted_db_raises(self, tmp_path):
        path = str(tmp_path / "encrypted.db")
        key = validate_key(secrets.token_hex(32))

        be = SQLiteBackend(path, key=key)
        await be.initialize()
        sess = Session(
            id="44444444-4444-4444-8444-444444444444",
            label=None, head_snapshot_id=None, created_at=_now(), metadata={},
        )
        await be.create_session(sess)
        await be.close()

        # Reopen with NO key — aiosqlite will see garbage pages and fail
        # somewhere during init (the migration's CREATE TABLE IF NOT EXISTS
        # tries to read the schema first).
        be_plain = SQLiteBackend(path)
        with pytest.raises(Exception):  # aiosqlite raises sqlite3.DatabaseError
            await be_plain.initialize()

    async def test_key_on_plain_db_raises(self, tmp_path):
        path = str(tmp_path / "plain.db")

        # Create a plain DB first.
        be = SQLiteBackend(path)
        await be.initialize()
        sess = Session(
            id="55555555-5555-4555-8555-555555555555",
            label=None, head_snapshot_id=None, created_at=_now(), metadata={},
        )
        await be.create_session(sess)
        await be.close()

        # Try to open it with a key — probe should reject.
        key = validate_key(secrets.token_hex(32))
        be_enc = SQLiteBackend(path, key=key)
        with pytest.raises(EncryptionKeyError):
            await be_enc.initialize()

    async def test_file_bytes_do_not_contain_plaintext(self, tmp_path):
        """Spot check: the on-disk bytes must not contain our plaintext.

        Not a cryptographic proof — just a smoke test that SQLCipher is
        actually engaged (vs. a silent fallback to plain SQLite).
        """
        path = str(tmp_path / "encrypted.db")
        key = validate_key(secrets.token_hex(32))
        be = SQLiteBackend(path, key=key)
        await be.initialize()
        sess = Session(
            id="66666666-6666-4666-8666-666666666666",
            label=None, head_snapshot_id=None, created_at=_now(), metadata={},
        )
        await be.create_session(sess)
        u = units.message(
            sess.id, "k", "UNIQUE_PLAINTEXT_MARKER_42", source="user"
        )
        snap = Snapshot(
            id="77777777-7777-4777-8777-777777777777",
            session_id=sess.id, label=None, parent_id=None,
            created_at=_now(), metadata={},
        )
        await be.write_snapshot(snap, new_units=[u], unit_ids_to_link=[u.id])
        await be.close()

        with open(path, "rb") as f:
            blob = f.read()
        assert b"UNIQUE_PLAINTEXT_MARKER_42" not in blob
