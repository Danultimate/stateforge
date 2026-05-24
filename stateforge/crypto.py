from __future__ import annotations

import re
from dataclasses import dataclass

from stateforge.exceptions import EncryptionKeyFormatError, EncryptionUnavailableError

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(slots=True, frozen=True)
class KeyMaterial:
    """Validated 32-byte encryption key, ready to be passed to SQLCipher."""

    raw: bytes  # exactly 32 bytes

    @property
    def hex(self) -> str:
        return self.raw.hex()

    def pragma_value(self) -> str:
        """Format for ``PRAGMA key = <value>`` using the raw-key form.

        SQLCipher accepts ``"x'<64-hex>'"`` to mean "use these bytes directly
        as the key, do NOT apply PBKDF2". This is what we want — the caller has
        already produced a high-entropy key (e.g., from a KMS); double-derivation
        is wasted work.
        """
        return f"\"x'{self.hex}'\""


def validate_key(key: str | bytes) -> KeyMaterial:
    """Validate and normalize an encryption key.

    Accepts:
    - a 64-character hex string (case-insensitive), or
    - exactly 32 raw bytes.

    Raises:
        EncryptionKeyFormatError: if neither form matches.
    """
    if isinstance(key, str):
        if not _HEX64.match(key):
            raise EncryptionKeyFormatError(
                "encryption key must be a 64-character hex string "
                f"(got length {len(key)})"
            )
        return KeyMaterial(raw=bytes.fromhex(key))
    if isinstance(key, (bytes, bytearray)):
        if len(key) != 32:
            raise EncryptionKeyFormatError(
                f"encryption key must be exactly 32 bytes (got {len(key)})"
            )
        return KeyMaterial(raw=bytes(key))
    raise EncryptionKeyFormatError(
        f"encryption key must be str or bytes, got {type(key).__name__}"
    )


def is_sqlcipher_available() -> bool:
    """Return True if the optional ``sqlcipher3`` driver is importable."""
    try:
        import sqlcipher3  # noqa: F401
        return True
    except ImportError:
        return False


def require_sqlcipher() -> None:
    """Raise EncryptionUnavailableError if sqlcipher3 is not installed."""
    if not is_sqlcipher_available():
        raise EncryptionUnavailableError(
            "encryption requested but sqlcipher3-binary is not installed. "
            "Install with: pip install stateforge-llm[encryption]"
        )
