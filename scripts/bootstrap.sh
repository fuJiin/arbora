#!/usr/bin/env bash
# Bootstrap an Arbora dev environment.
# Safe for external contributors: all private-notes steps are opt-in.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "==> Installing Python dependencies (uv sync)"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi
uv sync

echo "==> Installing pre-commit hooks"
uv run pre-commit install >/dev/null 2>&1 || \
  echo "   (pre-commit not configured or already installed, skipping)"

# Optional: link private research notes if they exist alongside this repo.
# External contributors don't need this; it silently no-ops for them.
NOTES_DIR="${ARBORA_NOTES_DIR:-$(dirname "$REPO_ROOT")/arbora-notes}"
if [ -d "$NOTES_DIR/.agents" ]; then
  if [ ! -e "$REPO_ROOT/.agents" ]; then
    ln -s "$NOTES_DIR/.agents" "$REPO_ROOT/.agents"
    echo "==> Linked private notes: $NOTES_DIR/.agents -> .agents"
  else
    echo "==> .agents already present, leaving alone"
  fi
else
  echo "==> No private notes repo at $NOTES_DIR (that's fine — optional)"
fi

echo
echo "Done. Try: uv run pytest tests/"
