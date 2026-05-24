# StateForge

> **Versioned memory and state management for AI agents.**
> Git-like history (snapshot, diff, rollback) and queryable provenance over event-sourced memory units. SQLite-backed, async-first, optionally encrypted at rest.

Agent memory is mutable, opaque, and ephemeral by default. StateForge makes it versioned, inspectable, and reproducible. Every memory unit is written once, never mutated, and addressable forever. Every snapshot is an immutable set of references to those units. You can always answer: *what did this agent know, when did it know it, and where did that knowledge come from.*

[![tests](https://img.shields.io/badge/tests-265%20passing-brightgreen)](./tests) [![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml) [![license](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## Install

```bash
pip install stateforge-llm
```

Optional extras:

| Extra | Adds | When to use |
|---|---|---|
| `[langchain]` | `langchain-core>=0.2` | `StateForgeMessageHistory` adapter |
| `[langgraph]` | `langgraph>=0.1` | `StateForgeCheckpointer` adapter |
| `[encryption]` | `sqlcipher3>=0.6` | DB-level encryption at rest (SQLCipher) |
| `[dev]` | pytest, pytest-asyncio | Run the test suite |

```bash
pip install "stateforge-llm[langchain,langgraph,encryption]"
```

---

## Quickstart

```python
import asyncio
from stateforge import StateForge, units

async def main():
    sf = StateForge(db_path="agent.db")
    session = await sf.create_session(label="my-agent-run")

    # Take a snapshot of agent state
    snap1 = await sf.snapshot(
        session.id,
        units=[
            units.message(session.id, key="msg:0", value="hello", source="user"),
            units.kv(session.id, key="goal", value={"task": "summarize"}, source="agent"),
        ],
        label="initial",
    )

    # Move state forward
    snap2 = await sf.snapshot(
        session.id,
        units=[
            units.message(session.id, key="msg:0", value="hello", source="user"),
            units.message(session.id, key="msg:1", value="working on it", source="agent"),
            units.kv(session.id, key="goal",
                     value={"task": "summarize", "progress": 0.5}, source="agent"),
        ],
        label="after-step-1",
    )

    # See what changed
    d = await sf.diff(snap1.id, snap2.id)
    print(f"added: {len(d.added)} modified: {len(d.modified)} removed: {len(d.removed)}")
    # → added: 1 modified: 1 removed: 0

    # Roll back. Non-destructive: snap1 + snap2 stay readable, head moves to snap3.
    snap3 = await sf.rollback(session.id, to_snapshot_id=snap1.id, label="undo")
    head = await sf.head(session.id)
    assert head.id == snap3.id

    await sf.close()

asyncio.run(main())
```

---

## How it works (one minute)

Three first-class concepts:

| | What | Lifetime |
|---|---|---|
| **MemoryUnit** | An atomic piece of memory: a message, a KV fact, an embedding, a tool result, a summary. Has `id`, `key`, `type`, `value`, `source`. | Write-once. Immutable. |
| **Snapshot** | A named, immutable *set of references* to MemoryUnits at a point in time. Snapshots form a linked list via `parent_id`. | Write-once. Immutable. |
| **Session** | A logical grouping. Tracks the current `head_snapshot_id`. | Mutable head pointer; everything else is append-only. |

A snapshot does **not** copy unit content — it references it. Two snapshots that share a unit pay the storage cost once. This matches the Git object model honestly.

### Diff identity

Two units are "the same logical thing" iff `(session_id, key, type)` matches.

- **added** = identity present in `to`, absent in `from`
- **removed** = identity present in `from`, absent in `to`
- **modified** = same identity, different `value`, `metadata`, or `embedding`

If a snapshot accidentally contains two units with the same identity, the latest by `created_at` wins.

### Rollback

`sf.rollback(session_id, to=snap)` creates a **new** snapshot whose unit set equals `snap`'s, with `parent_id = previous head`. The previous head and every prior snapshot stay intact. Rollback is itself a versioned event.

### Value contract

`value` and `metadata` must be JSON-safe: `str | int | float | bool | None | list | dict`. `bytes`, `datetime`, `UUID`, `Decimal`, and custom classes raise `ValueTypeError` at write time. No silent corruption, no pickle attack surface. Base64-encode bytes yourself if you need them.

---

## Encryption at rest (opt-in)

Threat model: disk theft, leaked backup, misconfigured cloud volume. DB-level encryption via SQLCipher defends against all three — an attacker with the file and without the key learns nothing, not even index keys or session labels.

```python
import os
from stateforge import StateForge

sf = StateForge(
    db_path="agent.db",
    encryption_key=os.environ["STATEFORGE_KEY"],   # 64-char hex string
)
# All schema, data, indexes, and WAL pages are encrypted on disk.
```

Generate a key once and stash it in your secrets manager:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then `pip install "stateforge-llm[encryption]"`.

**What encryption does NOT protect against:** a malicious process running as the same user (the key lives in memory while StateForge is open); coredumps or swap leaks; an attacker who obtains the key (env var leak, compromised secrets manager); side-channel observation of access patterns (file size, timestamps). Defend those layers separately.

**Lifecycle caveats:**

- Switching an existing plain DB to encrypted (or vice versa) is **not** supported in v0 — the DB is one mode or the other from creation.
- Key rotation is **not** supported in v0. Lost key = lost data; no recovery.
- A wrong key for an existing encrypted DB raises `EncryptionKeyError` on the first read after `PRAGMA key`.

---

## CLI

```bash
stateforge sessions list
stateforge snapshots list --session run-1
stateforge snapshot show head --session run-1
stateforge snapshot show head~1 --session run-1
stateforge diff head~1 head --session run-1
stateforge rollback --session run-1 --to initial
stateforge history --session run-1
stateforge provenance <unit_id>
stateforge vacuum
```

Snapshot refs accept `head`, `head~N`, label, or uuid (full or 8+ char prefix). Add `--json` to any command for machine-readable output. Set `STATEFORGE_DB` to skip `--db` on every call.

Example:

```
$ stateforge --db agent.db history --session demo
History for session demo (2 snapshots, head → root)
      head  568e3c3d  step-1     2026-05-23T17:53:37+00:00
    head~1  d505aedf  initial    2026-05-23T17:53:37+00:00
```

---

## Framework adapters

### LangChain

```python
from stateforge import StateForge
from stateforge.adapters.langchain import StateForgeMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

sf = StateForge("agent.db")
session = await sf.create_session(label="chat-1")

history = StateForgeMessageHistory(
    session_id=session.id,
    stateforge=sf,
    auto_snapshot=True,    # snapshot every N add_messages calls
    snapshot_every=5,
)

await history.aadd_messages([
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!"),
])
```

`auto_snapshot=False` (the default) means you snapshot explicitly via `await history.snapshot()`. `from_snapshot(...)` lets you resume after a process restart.

### LangGraph

```python
from stateforge import StateForge
from stateforge.adapters.langgraph import StateForgeCheckpointer
from langgraph.graph import StateGraph

sf = StateForge("agent.db")
checkpointer = StateForgeCheckpointer(sf)

graph = StateGraph(AgentState)
# ... add nodes ...
app = graph.compile(checkpointer=checkpointer)
```

**Key feature: per-field shredding.** Each top-level field in `checkpoint["channel_values"]` becomes its own `MemoryUnit(KV)`. This means `sf.diff(prev_checkpoint, curr_checkpoint)` reports *which fields changed in the agent's state*, not just "1 unit modified". This is the difference between a useful audit trail and an opaque blob store.

---

## Security & limitations (read before deploying)

StateForge v0 is a **single-tenant, single-process Python library**. The DB file is the trust boundary.

| Concern | v0 stance |
|---|---|
| Disk theft / leaked backup | ✅ Defended by opt-in SQLCipher |
| JSON-safe value contract | ✅ Enforced at write boundary |
| Snapshot atomicity (crash mid-write) | ✅ Single-transaction guarantee |
| Multi-tenant isolation | ❌ Out of scope. Run one instance per tenant. |
| Field-level / per-row encryption | ❌ v1+ (DB-level is supported now) |
| Key rotation | ❌ v1+. Lost key = lost data; no recovery. |
| Cryptographically signed provenance | ❌ v1+. Provenance is descriptive, not attestable. |
| Retention / TTL | ❌ v1+. Snapshots accumulate; manage at app level. |
| Vector similarity search | ❌ v1+. Embeddings stored faithfully but inert. |

**Integrator checklist before deploying with sensitive data:**

- [ ] Is the data sensitive enough that disk theft is a concern? → use `encryption_key=...`.
- [ ] Is the key sourced from a hardened location (KMS, secrets manager, OS keychain) and *not* hardcoded?
- [ ] Is the process running in a trust boundary you control (no untrusted code in-process)?
- [ ] Are OS-level controls in place (swap encryption, no coredumps, restricted file permissions on the `.db`)?
- [ ] Is there exactly one tenant per `StateForge` instance / DB file?
- [ ] If using backups: are backups encrypted (either by encrypting at rest here, or by the backup tool)?

---

## Development

```bash
git clone https://github.com/Danultimate/stateforge-llm
cd stateforge-llm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,langchain,langgraph,encryption]"
pytest -q
# → 265 passed in <1s
```

Built through a multi-agent design review process: an internal technical specification (v0.4.2) is the source of truth for every design decision — data model, schema, API surface, reliability contract, security threat model, deferred v1+ items.

---

## Project status

**v0.4.2 (2026-05) — implementation complete.** 265 tests passing, 0 skipped. All in-scope items from the spec are shipped. The library has not been published to PyPI yet; install from source.

See the spec for the v1+ roadmap.

---

## License

MIT — see [`pyproject.toml`](./pyproject.toml).
