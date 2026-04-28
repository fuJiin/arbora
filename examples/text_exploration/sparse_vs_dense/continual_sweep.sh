#!/usr/bin/env bash
# ARB-139: continual learning follow-up sweep (3 experiments).
#
# 1. Small-update sweep: phase 1 = 5M text8, phase 2 = {100k, 500k, 1M, 5M} Gutenberg
#    Tests whether SSH preserves better than w2v at smaller B sizes (less aggressive overwriting).
#    Reuses cached SSH phase-1 state across runs.
# 2. D=2048 cross-domain at 1M each phase
#    Tests whether more "room" reduces SSH's forgetting.
# 3. (Discussion-only — done in chat)
#
# Output dirs:
#   data/runs/arb139/continual_sweep/small_update_b{N}/
#   data/runs/arb139/continual_sweep/d2048/

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="data/runs/arb139/continual_sweep"
LOG_DIR="data/runs/arb139/logs"
mkdir -p "$SWEEP_DIR" "$LOG_DIR"

# Cache the phase 1 SSH state (D=1024) once and reuse for small-update sweep.
SSH_CACHE_5M_D1024="$SWEEP_DIR/ssh_phase1_5M_d1024.pkl"

# Experiment 1: small-update sweep
for B_TOKENS in 100000 500000 1000000 5000000; do
    OUT_DIR="$SWEEP_DIR/small_update_b${B_TOKENS}"
    LOG="$LOG_DIR/continual_small_update_b${B_TOKENS}.log"
    if [ -f "$OUT_DIR/results.csv" ]; then
        echo "=== skip small_update b=${B_TOKENS} (results.csv present) ==="
        continue
    fi
    echo "=== experiment 1: small_update b=${B_TOKENS} ==="
    uv run python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
        --n-per-phase 5000000 \
        --n-tokens-b "$B_TOKENS" \
        --split cross_domain \
        --out-dir "$OUT_DIR" \
        --ssh-cache-phase1 "$SSH_CACHE_5M_D1024" \
        2>&1 | tee "$LOG"
done

# Experiment 2: D=2048 at 1M each phase
OUT_DIR="$SWEEP_DIR/d2048_1M"
LOG="$LOG_DIR/continual_d2048_1M.log"
if [ -f "$OUT_DIR/results.csv" ]; then
    echo "=== skip d2048_1M (results.csv present) ==="
else
    echo "=== experiment 2: D=2048 at 1M each phase ==="
    uv run python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
        --n-per-phase 1000000 \
        --split cross_domain \
        --ssh-n-dims 2048 \
        --out-dir "$OUT_DIR" \
        2>&1 | tee "$LOG"
fi

echo "=== continual learning sweep complete ==="
