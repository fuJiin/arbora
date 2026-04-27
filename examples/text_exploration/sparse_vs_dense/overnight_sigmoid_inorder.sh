#!/usr/bin/env bash
# ARB-139 overnight continuation: Phase 2-4 in-order (no shuffle).
#
# Phase 1 of overnight_sigmoid.sh revealed that token shuffling kills the
# discriminative signal in skip-gram + unigram^0.75 negative sampling.
# This continuation runs Phase 2 (within-run drift) and Phase 3 (cross-
# scale) WITHOUT the shuffle flag.
#
# Usage: bash examples/text_exploration/sparse_vs_dense/overnight_sigmoid_inorder.sh

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

# Phase 2: within-run drift with sigmoid + inorder
phase "phase2_within_run_drift_inorder" \
    uv run python -m examples.text_exploration.sparse_vs_dense.within_run_simlex \
        --sigmoid-bounded \
        --ema-alpha 0.0 \
        --csv "$OUT_DIR/within_run_sigmoid_inorder.csv"

# Phase 3: cross-scale sigmoid-bounded inorder
phase "phase3_cross_scale_sigmoid_inorder" \
    uv run python -m examples.text_exploration.sparse_vs_dense.cross_scale_sigmoid \
        --no-shuffle \
        --csv "$OUT_DIR/cross_scale_sigmoid_inorder.csv"

# Phase 4: refresh diagnostics + plot
phase "phase4_diagnostics" \
    uv run python -m examples.text_exploration.sparse_vs_dense.diagnostics
phase "phase4_plot" \
    uv run python -m examples.text_exploration.sparse_vs_dense.plot_within_run

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] in-order continuation complete ==="
