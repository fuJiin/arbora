#!/usr/bin/env bash
# ARB-139: within-run SimLex sweep over EMA alpha.
# Runs 3 within-run experiments at 1M tokens with different EMA decay
# rates to find the timescale that smooths SSH's oscillations.
#
# Usage: bash examples/text_exploration/sparse_vs_dense/within_run_alpha_sweep.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="data/runs/arb139"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

for alpha in 0.001 0.0005 0.0001; do
    safe_alpha=$(echo "$alpha" | tr '.' '_')
    csv="$OUT_DIR/within_run_alpha_${safe_alpha}.csv"
    log="$LOG_DIR/within_run_alpha_${safe_alpha}.log"
    if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -ge 21 ]; then
        echo "skip: $csv already complete"
        continue
    fi
    echo "=== alpha=$alpha → $csv ==="
    uv run python -m examples.text_exploration.sparse_vs_dense.within_run_simlex \
        --ema-alpha "$alpha" \
        --csv "$csv" 2>&1 | tee "$log"
done

echo "=== alpha sweep complete ==="
