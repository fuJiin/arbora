#!/usr/bin/env bash
# ARB-139 overnight sweep — sigmoid-bounded SSH battery.
#
# Phases (each writes its own CSV; resumable):
#   1. shuffle/epoch fairness test at 1M (~1.5 h)
#   2. within-run drift diagnostic at 1M with shuffle + sigmoid (~12 min)
#   3. cross-scale sigmoid-bounded with shuffle (~3 h)
#   4. postprocess: refresh diagnostics + within-run plot (~1 min)
#
# Usage: bash examples/text_exploration/sparse_vs_dense/overnight_sigmoid.sh
# All output: data/runs/arb139/*.csv + logs in data/runs/arb139/logs/

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="data/runs/arb139"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

phase() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} starting ===" | tee -a "$log"
    {
        time "$@"
    } 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} exit=${rc} ===" | tee -a "$log"
}

# Phase 1: shuffle + multi-epoch at 1M
phase "phase1_shuffle_epoch" \
    uv run python -m examples.text_exploration.sparse_vs_dense.shuffle_epoch_test \
        --csv "$OUT_DIR/shuffle_epoch_test.csv"

# Phase 2: within-run drift with sigmoid + shuffle at 1M
phase "phase2_within_run_drift" \
    uv run python -m examples.text_exploration.sparse_vs_dense.within_run_simlex \
        --sigmoid-bounded \
        --shuffle \
        --ema-alpha 0.0 \
        --csv "$OUT_DIR/within_run_sigmoid_shuffle.csv"

# Phase 3: cross-scale sigmoid-bounded with shuffle
phase "phase3_cross_scale_sigmoid" \
    uv run python -m examples.text_exploration.sparse_vs_dense.cross_scale_sigmoid \
        --csv "$OUT_DIR/cross_scale_sigmoid.csv"

# Phase 4: refresh diagnostics + plot
phase "phase4_diagnostics" \
    uv run python -m examples.text_exploration.sparse_vs_dense.diagnostics
phase "phase4_plot" \
    uv run python -m examples.text_exploration.sparse_vs_dense.plot_within_run

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] overnight sweep complete ==="
ls -la "$OUT_DIR" "$OUT_DIR/diagnostics" 2>/dev/null
