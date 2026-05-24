"""StateForge command-line interface.

Conventions:
- Output is Rich tables by default, JSON with ``--json``.
- Snapshot refs accept ``head``, ``head~N``, label, or uuid (full or 8+ prefix).
- ``rollback`` prompts for confirmation unless ``--yes`` is passed.
- Connection path comes from ``--db`` or the ``STATEFORGE_DB`` env var.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from stateforge.client import StateForge
from stateforge.exceptions import (
    AmbiguousRefError,
    SessionNotFoundError,
    SnapshotNotFoundError,
    StateForgeError,
)
from stateforge.models import MemoryUnit, ProvenanceRecord, Session, Snapshot
from stateforge.refs import resolve_session, resolve_snapshot

app = typer.Typer(
    name="stateforge",
    help="Versioned memory and state management for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)

sessions_app = typer.Typer(help="Manage sessions.")
snapshots_app = typer.Typer(help="Inspect snapshots.")
snapshot_app = typer.Typer(help="Inspect a single snapshot.")

app.add_typer(sessions_app, name="sessions")
app.add_typer(snapshots_app, name="snapshots")
app.add_typer(snapshot_app, name="snapshot")


# ────────────────────────────────────────────────────────────────────────────
# Context, console, helpers
# ────────────────────────────────────────────────────────────────────────────


def _console() -> Console:
    # Disable color in tests via NO_COLOR env; Rich respects it automatically.
    return Console(file=sys.stdout)


@app.callback()
def main(
    ctx: typer.Context,
    db_path: str = typer.Option(
        "stateforge.db", "--db", envvar="STATEFORGE_DB",
        help="Path to the SQLite DB file.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of tables.",
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["json"] = json_output


def _run(coro):
    """Run an async command and translate StateForge errors to Typer exits."""
    try:
        return asyncio.run(coro)
    except (AmbiguousRefError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from None
    except (SessionNotFoundError, SnapshotNotFoundError) as e:
        typer.echo(f"not found: {e}", err=True)
        raise typer.Exit(code=1) from None
    except StateForgeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1) from None


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _session_dict(s: Session) -> dict[str, Any]:
    return {
        "id": s.id,
        "label": s.label,
        "head_snapshot_id": s.head_snapshot_id,
        "created_at": _iso(s.created_at),
        "metadata": s.metadata,
    }


def _snapshot_dict(s: Snapshot) -> dict[str, Any]:
    return {
        "id": s.id,
        "session_id": s.session_id,
        "label": s.label,
        "parent_id": s.parent_id,
        "created_at": _iso(s.created_at),
        "metadata": s.metadata,
    }


def _unit_dict(u: MemoryUnit) -> dict[str, Any]:
    return {
        "id": u.id,
        "session_id": u.session_id,
        "type": u.type.value,
        "key": u.key,
        "value": u.value,
        "embedding": "<bytes>" if u.embedding else None,
        "metadata": u.metadata,
        "source": u.source,
        "source_ref": u.source_ref,
        "created_at": _iso(u.created_at),
    }


def _provenance_dict(p: ProvenanceRecord) -> dict[str, Any]:
    return {
        "id": p.id,
        "memory_unit_id": p.memory_unit_id,
        "source": p.source,
        "source_ref": p.source_ref,
        "ingested_at": _iso(p.ingested_at),
        "trace": [
            {"index": h.index, "source": h.source, "source_ref": h.source_ref}
            for h in p.trace
        ],
    }


def _emit_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


def _short(s: str | None, n: int = 8) -> str:
    if s is None:
        return "—"
    return s[:n]


# ────────────────────────────────────────────────────────────────────────────
# sessions list
# ────────────────────────────────────────────────────────────────────────────


@sessions_app.command("list")
def sessions_list(ctx: typer.Context) -> None:
    """List all sessions."""
    _run(_sessions_list_impl(ctx))


async def _sessions_list_impl(ctx: typer.Context) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        sessions = await sf.list_sessions(limit=1000)
        if ctx.obj["json"]:
            _emit_json([_session_dict(s) for s in sessions])
            return
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Label")
        table.add_column("Head", style="dim")
        table.add_column("Created")
        for s in sessions:
            table.add_row(
                _short(s.id),
                s.label or "—",
                _short(s.head_snapshot_id),
                _iso(s.created_at),
            )
        _console().print(table)
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# snapshots list --session <ref>
# ────────────────────────────────────────────────────────────────────────────


@snapshots_app.command("list")
def snapshots_list(
    ctx: typer.Context,
    session: str = typer.Option(
        ..., "--session", "-s", help="Session ref (uuid prefix or label).",
    ),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List snapshots in a session."""
    _run(_snapshots_list_impl(ctx, session, limit))


async def _snapshots_list_impl(ctx: typer.Context, session_ref: str, limit: int) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        sess = await resolve_session(sf, session_ref)
        snaps = await sf.list_snapshots(sess.id, limit=limit)
        if ctx.obj["json"]:
            _emit_json([_snapshot_dict(s) for s in snaps])
            return
        table = Table(title=f"Snapshots in {sess.label or _short(sess.id)}")
        table.add_column("ID", style="cyan")
        table.add_column("Label")
        table.add_column("Parent", style="dim")
        table.add_column("Created")
        for s in snaps:
            table.add_row(
                _short(s.id),
                s.label or "—",
                _short(s.parent_id),
                _iso(s.created_at),
            )
        _console().print(table)
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# snapshot show <ref> [--session <ref>]
# ────────────────────────────────────────────────────────────────────────────


@snapshot_app.command("show")
def snapshot_show(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="Snapshot ref: head, head~N, label, or uuid prefix."),
    session: str | None = typer.Option(
        None, "--session", "-s",
        help="Session ref (required for head/head~N and label).",
    ),
) -> None:
    """Show a snapshot's metadata and its memory units."""
    _run(_snapshot_show_impl(ctx, ref, session))


async def _snapshot_show_impl(
    ctx: typer.Context, ref: str, session_ref: str | None
) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        session_id = None
        if session_ref is not None:
            session_id = (await resolve_session(sf, session_ref)).id
        snap = await resolve_snapshot(sf, ref, session_id=session_id)
        snap_units = await sf.get_units(snap.id)
        if ctx.obj["json"]:
            _emit_json({
                "snapshot": _snapshot_dict(snap),
                "units": [_unit_dict(u) for u in snap_units],
            })
            return
        console = _console()
        console.print(f"[bold cyan]Snapshot[/] {snap.id}")
        console.print(f"  label:     {snap.label or '—'}")
        console.print(f"  session:   {snap.session_id}")
        console.print(f"  parent:    {snap.parent_id or '—'}")
        console.print(f"  created:   {_iso(snap.created_at)}")
        if snap.metadata:
            console.print(f"  metadata:  {snap.metadata}")

        table = Table(title=f"Units ({len(snap_units)})")
        table.add_column("Key", style="cyan")
        table.add_column("Type")
        table.add_column("Source")
        table.add_column("Value (truncated)")
        for u in snap_units:
            val_repr = _truncate(repr(u.value), 60)
            table.add_row(u.key, u.type.value, u.source, val_repr)
        console.print(table)
    finally:
        await sf.close()


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ────────────────────────────────────────────────────────────────────────────
# diff <from> <to> [--session <ref>]
# ────────────────────────────────────────────────────────────────────────────


@app.command("diff")
def diff(
    ctx: typer.Context,
    from_ref: str = typer.Argument(..., metavar="FROM"),
    to_ref: str = typer.Argument(..., metavar="TO"),
    session: str | None = typer.Option(
        None, "--session", "-s",
        help="Session ref (required for head/head~N and label refs).",
    ),
) -> None:
    """Diff two snapshots."""
    _run(_diff_impl(ctx, from_ref, to_ref, session))


async def _diff_impl(
    ctx: typer.Context, from_ref: str, to_ref: str, session_ref: str | None
) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        session_id = None
        if session_ref is not None:
            session_id = (await resolve_session(sf, session_ref)).id
        snap_from = await resolve_snapshot(sf, from_ref, session_id=session_id)
        snap_to = await resolve_snapshot(sf, to_ref, session_id=session_id)
        d = await sf.diff(snap_from.id, snap_to.id)
        if ctx.obj["json"]:
            _emit_json({
                "from": snap_from.id,
                "to": snap_to.id,
                "added": [_unit_dict(u) for u in d.added],
                "removed": [_unit_dict(u) for u in d.removed],
                "modified": [
                    {"before": _unit_dict(b), "after": _unit_dict(a)}
                    for b, a in d.modified
                ],
            })
            return
        console = _console()
        console.print(
            f"[green]added[/]: {len(d.added)}  "
            f"[yellow]modified[/]: {len(d.modified)}  "
            f"[red]removed[/]: {len(d.removed)}"
        )
        for u in d.added:
            console.print(f"  [green]+[/] {u.key} ({u.type.value}) = {_truncate(repr(u.value), 60)}")
        for b, a in d.modified:
            console.print(
                f"  [yellow]~[/] {b.key} ({b.type.value}): "
                f"{_truncate(repr(b.value), 30)} → {_truncate(repr(a.value), 30)}"
            )
        for u in d.removed:
            console.print(f"  [red]-[/] {u.key} ({u.type.value})")
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# rollback --session <ref> --to <ref>
# ────────────────────────────────────────────────────────────────────────────


@app.command("rollback")
def rollback(
    ctx: typer.Context,
    session: str = typer.Option(..., "--session", "-s"),
    to: str = typer.Option(..., "--to", "-t", help="Snapshot ref to roll back to."),
    label: str | None = typer.Option(None, "--label"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Roll back a session's head to a previous snapshot.

    Non-destructive: creates a new snapshot referencing the target's units.
    """
    if not yes:
        typer.confirm(
            f"Roll back session {session!r} to {to!r}? "
            "(creates a new head snapshot; no data is deleted)",
            abort=True,
        )
    _run(_rollback_impl(ctx, session, to, label))


async def _rollback_impl(
    ctx: typer.Context, session_ref: str, to_ref: str, label: str | None
) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        sess = await resolve_session(sf, session_ref)
        target = await resolve_snapshot(sf, to_ref, session_id=sess.id)
        new_snap = await sf.rollback(sess.id, target.id, label=label)
        if ctx.obj["json"]:
            _emit_json(_snapshot_dict(new_snap))
            return
        console = _console()
        console.print(
            f"[green]rolled back[/] session {sess.label or _short(sess.id)} "
            f"to {_short(target.id)}"
        )
        console.print(f"new head: {new_snap.id}")
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# provenance <unit_id>
# ────────────────────────────────────────────────────────────────────────────


@app.command("provenance")
def provenance(
    ctx: typer.Context,
    unit_id: str = typer.Argument(..., help="Memory unit id (full uuid)."),
) -> None:
    """Show the provenance record for a memory unit."""
    _run(_provenance_impl(ctx, unit_id))


async def _provenance_impl(ctx: typer.Context, unit_id: str) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        rec = await sf.get_provenance(unit_id)
        if ctx.obj["json"]:
            _emit_json(_provenance_dict(rec))
            return
        console = _console()
        console.print(f"[bold cyan]Provenance[/] for unit {unit_id}")
        console.print(f"  source:      {rec.source}")
        console.print(f"  source_ref:  {rec.source_ref or '—'}")
        console.print(f"  ingested:    {_iso(rec.ingested_at)}")
        if rec.trace:
            table = Table(title="Trace")
            table.add_column("Hop")
            table.add_column("Source")
            table.add_column("Source Ref")
            for hop in rec.trace:
                table.add_row(str(hop.index), hop.source, hop.source_ref or "—")
            console.print(table)
        else:
            console.print("  trace:       (empty)")
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# history --session <ref>
# ────────────────────────────────────────────────────────────────────────────


@app.command("history")
def history(
    ctx: typer.Context,
    session: str = typer.Option(..., "--session", "-s"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """Walk the parent chain from head to root for a session."""
    _run(_history_impl(ctx, session, limit))


async def _history_impl(ctx: typer.Context, session_ref: str, limit: int) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        sess = await resolve_session(sf, session_ref)
        chain = await sf.history(sess.id, max_depth=limit)
        if ctx.obj["json"]:
            _emit_json([_snapshot_dict(s) for s in chain])
            return
        console = _console()
        console.print(f"[bold]History[/] for session {sess.label or _short(sess.id)} "
                      f"({len(chain)} snapshots, head → root)")
        for i, s in enumerate(chain):
            marker = "head" if i == 0 else f"head~{i}"
            console.print(
                f"  [cyan]{marker:>8}[/]  {_short(s.id)}  "
                f"{s.label or '—':<20}  {_iso(s.created_at)}"
            )
    finally:
        await sf.close()


# ────────────────────────────────────────────────────────────────────────────
# vacuum
# ────────────────────────────────────────────────────────────────────────────


@app.command("vacuum")
def vacuum(ctx: typer.Context) -> None:
    """Run SQLite VACUUM to compact the DB file.

    Does NOT delete any data. v0 has no retention policy — see § Security &
    Limitations in the spec.
    """
    _run(_vacuum_impl(ctx))


async def _vacuum_impl(ctx: typer.Context) -> None:
    sf = StateForge(ctx.obj["db_path"])
    try:
        # Ensure the connection is open, then issue VACUUM directly.
        await sf.list_sessions(limit=1)  # triggers _ensure
        # VACUUM cannot run inside a transaction; run it bare.
        await sf._backend._conn.execute("VACUUM")
        typer.echo("vacuum complete")
    finally:
        await sf.close()


if __name__ == "__main__":
    app()
