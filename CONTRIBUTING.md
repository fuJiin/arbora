# Contributing to Arbora

Arbora is experimental research software — APIs are evolving and breaking
changes should be expected within 0.x. Issues are welcome; PRs are best
discussed in an issue first.

## Setup

```bash
./scripts/bootstrap.sh
```

This runs `uv sync`, installs pre-commit hooks, and optionally links a local
private notes directory if one exists alongside this repo (see below —
external contributors can ignore this).

## Running checks

```bash
uv run pytest tests/
uv run ruff check src/ tests/
uv run ty check src/arbora/
```

## Private notes (maintainer-only)

Some long-form research notes used by AI coding agents are kept in a private
sibling repo rather than this public one. The bootstrap script will symlink
them into `.agents/` if it finds either:

- a sibling directory at `../arbora-notes/`, or
- a path set via `ARBORA_NOTES_DIR`.

If neither exists, the bootstrap script skips this step without error.
There's no requirement to have such a directory to build, test, or
contribute to Arbora.
