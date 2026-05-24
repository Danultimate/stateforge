from __future__ import annotations

import json

import pytest
import pytest_asyncio
from typer.testing import CliRunner

from stateforge import StateForge, units
from stateforge.cli import app

runner = CliRunner()


@pytest_asyncio.fixture
async def populated_db(tmp_path):
    """A tmp DB pre-populated with one session and two snapshots.

    Returns (db_path, session, snap1, snap2, unit_id).
    """
    path = str(tmp_path / "cli.db")
    sf = StateForge(path)
    try:
        sess = await sf.create_session(label="run-1")
        u1 = units.kv(sess.id, "goal", {"task": "summarize"}, source="agent")
        snap1 = await sf.snapshot(sess.id, units=[u1], label="initial")
        u2 = units.kv(sess.id, "goal", {"task": "summarize", "progress": 0.5}, source="agent")
        snap2 = await sf.snapshot(sess.id, units=[u2], label="step-1")
        return path, sess, snap1, snap2, u1.id
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# sessions list
# ────────────────────────────────────────────────────────────────────────────


class TestSessionsList:
    def test_table_output(self, populated_db):
        path, sess, *_ = populated_db
        result = runner.invoke(app, ["--db", path, "sessions", "list"])
        assert result.exit_code == 0
        assert "run-1" in result.stdout

    def test_json_output(self, populated_db):
        path, sess, *_ = populated_db
        result = runner.invoke(app, ["--db", path, "--json", "sessions", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert any(s["label"] == "run-1" for s in data)


# ────────────────────────────────────────────────────────────────────────────
# snapshots list
# ────────────────────────────────────────────────────────────────────────────


class TestSnapshotsList:
    def test_by_session_uuid_prefix(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshots", "list", "--session", sess.id[:8]]
        )
        assert result.exit_code == 0
        assert "initial" in result.stdout
        assert "step-1" in result.stdout

    def test_by_session_label(self, populated_db):
        path, sess, *_ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshots", "list", "--session", "run-1"]
        )
        assert result.exit_code == 0
        assert "initial" in result.stdout

    def test_json_output(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "--json", "snapshots", "list", "--session", "run-1"]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        ids = {s["id"] for s in data}
        assert {snap1.id, snap2.id} <= ids


# ────────────────────────────────────────────────────────────────────────────
# snapshot show
# ────────────────────────────────────────────────────────────────────────────


class TestSnapshotShow:
    def test_by_uuid(self, populated_db):
        path, sess, snap1, *_ = populated_db
        result = runner.invoke(app, ["--db", path, "snapshot", "show", snap1.id])
        assert result.exit_code == 0
        assert snap1.id in result.stdout
        assert "goal" in result.stdout  # unit key

    def test_by_head(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshot", "show", "head", "--session", "run-1"]
        )
        assert result.exit_code == 0
        assert snap2.id in result.stdout

    def test_by_head_tilde(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshot", "show", "head~1", "--session", "run-1"]
        )
        assert result.exit_code == 0
        assert snap1.id in result.stdout

    def test_head_without_session_fails(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(app, ["--db", path, "snapshot", "show", "head"])
        assert result.exit_code != 0
        assert "session" in result.stderr.lower() or "session" in result.stdout.lower()

    def test_by_label(self, populated_db):
        path, sess, snap1, *_ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshot", "show", "initial", "--session", "run-1"]
        )
        assert result.exit_code == 0
        assert snap1.id in result.stdout

    def test_json_output(self, populated_db):
        path, sess, snap1, *_ = populated_db
        result = runner.invoke(
            app, ["--db", path, "--json", "snapshot", "show", snap1.id]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["snapshot"]["id"] == snap1.id
        assert len(data["units"]) == 1


# ────────────────────────────────────────────────────────────────────────────
# diff
# ────────────────────────────────────────────────────────────────────────────


class TestDiff:
    def test_by_uuid(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "diff", snap1.id, snap2.id]
        )
        assert result.exit_code == 0
        assert "modified" in result.stdout

    def test_by_head_tilde_with_session(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, [
                "--db", path, "diff", "head~1", "head", "--session", "run-1",
            ],
        )
        assert result.exit_code == 0
        assert "modified" in result.stdout

    def test_json_output(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "--json", "diff", snap1.id, snap2.id]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["from"] == snap1.id
        assert data["to"] == snap2.id
        assert len(data["modified"]) == 1


# ────────────────────────────────────────────────────────────────────────────
# rollback
# ────────────────────────────────────────────────────────────────────────────


class TestRollback:
    def test_with_yes_flag(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, [
                "--db", path, "rollback",
                "--session", "run-1", "--to", "initial", "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "rolled back" in result.stdout

    def test_without_yes_aborts(self, populated_db):
        path, *_ = populated_db
        # No --yes; CliRunner provides no stdin → confirm() aborts.
        result = runner.invoke(
            app, [
                "--db", path, "rollback",
                "--session", "run-1", "--to", "initial",
            ],
        )
        assert result.exit_code != 0

    def test_without_yes_with_no_input(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(
            app, [
                "--db", path, "rollback",
                "--session", "run-1", "--to", "initial",
            ],
            input="n\n",
        )
        assert result.exit_code != 0

    def test_with_yes_and_input(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(
            app, [
                "--db", path, "rollback",
                "--session", "run-1", "--to", "initial", "--label", "manual undo",
            ],
            input="y\n",
        )
        assert result.exit_code == 0


# ────────────────────────────────────────────────────────────────────────────
# provenance
# ────────────────────────────────────────────────────────────────────────────


class TestProvenance:
    def test_no_provenance_record_yields_not_found(self, populated_db):
        path, sess, snap1, snap2, unit_id = populated_db
        # No provenance has been written for this unit.
        result = runner.invoke(app, ["--db", path, "provenance", unit_id])
        assert result.exit_code != 0

    def test_with_record(self, tmp_path):
        from datetime import datetime, timezone
        from uuid import uuid4
        from stateforge.models import ProvenanceHop, ProvenanceRecord

        path = str(tmp_path / "p.db")

        async def setup():
            sf = StateForge(path)
            try:
                sess = await sf.create_session()
                u = units.kv(sess.id, "x", 1)
                await sf.snapshot(sess.id, units=[u])
                rec = ProvenanceRecord(
                    id=str(uuid4()),
                    memory_unit_id=u.id,
                    source="summarizer",
                    source_ref=None,
                    ingested_at=datetime.now(timezone.utc),
                    trace=[
                        ProvenanceHop(index=0, source="langchain", source_ref="ai"),
                        ProvenanceHop(index=1, source="tool", source_ref="web_search"),
                    ],
                )
                await sf.write_provenance(rec)
                return u.id
            finally:
                await sf.close()

        import asyncio
        unit_id = asyncio.run(setup())

        result = runner.invoke(app, ["--db", path, "--json", "provenance", unit_id])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["source"] == "summarizer"
        assert len(data["trace"]) == 2


# ────────────────────────────────────────────────────────────────────────────
# history
# ────────────────────────────────────────────────────────────────────────────


class TestHistory:
    def test_walks_chain(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "history", "--session", "run-1"]
        )
        assert result.exit_code == 0
        # Both snapshots in the chain output.
        assert snap1.id[:8] in result.stdout
        assert snap2.id[:8] in result.stdout

    def test_json_output(self, populated_db):
        path, sess, snap1, snap2, _ = populated_db
        result = runner.invoke(
            app, ["--db", path, "--json", "history", "--session", "run-1"]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        # head → root order
        assert [s["id"] for s in data] == [snap2.id, snap1.id]


# ────────────────────────────────────────────────────────────────────────────
# vacuum
# ────────────────────────────────────────────────────────────────────────────


class TestVacuum:
    def test_runs(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(app, ["--db", path, "vacuum"])
        assert result.exit_code == 0
        assert "vacuum complete" in result.stdout


# ────────────────────────────────────────────────────────────────────────────
# Error handling
# ────────────────────────────────────────────────────────────────────────────


class TestErrors:
    def test_missing_session_label(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshots", "list", "--session", "no-such-session"]
        )
        assert result.exit_code != 0

    def test_missing_snapshot_uuid(self, populated_db):
        path, *_ = populated_db
        result = runner.invoke(
            app, ["--db", path, "snapshot", "show", "11111111-deadbeef"]
        )
        assert result.exit_code != 0

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "stateforge" in result.stdout.lower() or "usage" in result.stdout.lower()
