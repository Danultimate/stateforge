# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-05-25

### Added
- Initial public release of StateForge.
- Versioned memory store with Git-like snapshot / diff / rollback semantics over event-sourced memory units.
- Async SQLite backend (`aiosqlite`) with WAL mode, busy timeout, and per-session locking.
- Optional encryption-at-rest via SQLCipher (`stateforge-llm[encryption]`).
- LangChain `BaseChatMessageHistory` adapter (`stateforge-llm[langchain]`).
- LangGraph `BaseCheckpointSaver` adapter (`stateforge-llm[langgraph]`).
- Typer-based CLI (`stateforge`) for inspecting snapshots, diffs, and history.

[Unreleased]: https://github.com/Danultimate/stateforge/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/Danultimate/stateforge/releases/tag/v0.4.1
