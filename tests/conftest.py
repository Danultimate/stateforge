from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from stateforge import StateForge
from stateforge.models import Session
from stateforge.storage.sqlite import SQLiteBackend


@pytest.fixture
def session_id() -> str:
    return "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest_asyncio.fixture
async def backend() -> SQLiteBackend:
    """A fresh in-memory SQLite backend, initialized. Closed after test."""
    be = SQLiteBackend(":memory:")
    await be.initialize()
    try:
        yield be
    finally:
        await be.close()


@pytest_asyncio.fixture
async def backend_with_session(backend: SQLiteBackend, session_id: str, now: datetime):
    """Backend pre-populated with one empty session."""
    sess = Session(
        id=session_id,
        label="test",
        head_snapshot_id=None,
        created_at=now,
        metadata={},
    )
    await backend.create_session(sess)
    return backend, sess


@pytest.fixture
def new_id() -> "callable[[], str]":
    return lambda: str(uuid4())


@pytest_asyncio.fixture
async def sf() -> StateForge:
    """Fresh StateForge against in-memory SQLite. Closed after test."""
    forge = StateForge(":memory:")
    try:
        yield forge
    finally:
        await forge.close()
