#!/usr/bin/env bash
# ARB-139: BCM-style consolidation sweep.
#
# Hypothesis: SSH catastrophic forgetting at large B (≥1M) is driven by the
# top-k aperture being a hard winner-take-all — phase-1 bits get booted from
# top-k and the SDR loses them irrecoverably. A meta-plasticity bonus that
# adds `λ` to the accumulator entries marked as "phase-1 winners" gives
# those bits a head-start in the ranking, requiring phase-2 evidence to
# accumulate enough to overcome `λ` before displacing them.
#
# Test at the failure regime: 5M text8 phase 1 (cached) → 5M Gutenberg phase 2.
# Compare three bonus magnitudes:
#   - 0.1   marginal (50x typical top-k boundary gap of 0.002)
#   - 0.5   strong (puts phase-1 bits at ~1.4, near the sigmoid cap)
#   - 1.0   hard freeze (puts them above the natural sigmoid cap of ~1.6)
#
# Reference (consolidation_bonus=0.0) is already at small_update_b5000000/.
#
# Output dirs: data/runs/arb139/consolidation_sweep/bonus_{N}/

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="data/runs/arb139/consolidation_sweep"
LOG_DIR="data/runs/arb139/logs"
mkdir -p "$SWEEP_DIR" "$LOG_DIR"

SSH_CACHE_5M_D1024="data/runs/arb139/continual_sweep/ssh_phase1_5M_d1024.pkl"

if [ ! -f "$SSH_CACHE_5M_D1024" ]; then
    echo "ERROR: SSH phase-1 cache missing: $SSH_CACHE_5M_D1024"
    echo "Run continual_sweep.sh first to populate it."
    exit 1
fi

for BONUS in 0.1 0.5 1.0; do
    SAFE_BONUS="${BONUS//./_}"
    OUT_DIR="$SWEEP_DIR/bonus_${SAFE_BONUS}"
    LOG="$LOG_DIR/consolidation_bonus_${SAFE_BONUS}.log"
    if [ -f "$OUT_DIR/results.csv" ]; then
        echo "=== skip bonus=${BONUS} (results.csv present) ==="
        continue
    fi
    echo "=== consolidation bonus=${BONUS} ==="
    uv run python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
        --n-per-phase 5000000 \
        --split cross_domain \
        --out-dir "$OUT_DIR" \
        --ssh-k-eval 160 \
        --ssh-cache-phase1 "$SSH_CACHE_5M_D1024" \
        --ssh-consolidation-bonus "$BONUS" \
        2>&1 | tee "$LOG"
done

echo "=== consolidation sweep complete ==="
