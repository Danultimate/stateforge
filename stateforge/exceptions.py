class StateForgeError(Exception):
    """Base class for all StateForge errors."""


class SessionNotFoundError(StateForgeError):
    """A session with the given id does not exist."""


class SnapshotNotFoundError(StateForgeError):
    """A snapshot with the given id does not exist."""


class MemoryUnitNotFoundError(StateForgeError):
    """A memory unit with the given id does not exist."""


class CrossSessionRollbackError(StateForgeError):
    """Rollback target belongs to a different session than the requested rollback."""


class HistoryTooDeepError(StateForgeError):
    """Parent-chain walk exceeded the configured maximum depth."""


class ValueTypeError(StateForgeError):
    """Value or metadata contains a non-JSON-safe type.

    Raised at the API boundary before any database write.
    """


class StorageError(StateForgeError):
    """Backend I/O or integrity failure."""


class AdapterError(StateForgeError):
    """Framework adapter failure (LangChain, LangGraph)."""


class AmbiguousRefError(StateForgeError):
    """A CLI reference (label or uuid prefix) matched multiple snapshots."""


class EncryptionUnavailableError(StateForgeError):
    """An encryption key was supplied but ``sqlcipher3-binary`` is not installed."""


class EncryptionKeyError(StateForgeError):
    """Wrong, missing, or extraneous encryption key for the database.

    Raised on the first read after ``PRAGMA key`` is issued — including:
    - wrong key for an existing encrypted DB
    - missing key for an existing encrypted DB
    - a key supplied for a plain (unencrypted) DB
    """


class EncryptionKeyFormatError(StateForgeError):
    """Encryption key is neither a 64-character hex string nor 32 raw bytes.

    Raised synchronously at ``StateForge.__init__``, before any DB open.
    """
