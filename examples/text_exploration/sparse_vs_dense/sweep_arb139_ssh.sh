#!/usr/bin/env bash
# ARB-139 follow-up sweep: sparse-skipgram-Hebbian only.
# Same token-count sweep as phase 1 + same 500k variance at seeds 1, 2.
#
# Usage:
#   bash examples/text_exploration/sparse_vs_dense/sweep_arb139_ssh.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="data/runs/arb139"
DUMP_DIR="$OUT_DIR/dumps"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$DUMP_DIR" "$LOG_DIR"

VOCAB=5000
SKIP="--skip word2vec random_indexing brown_cluster t1"

run_phase() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} starting ===" | tee -a "$log"
    {
        time uv run python -m examples.text_exploration.sparse_vs_dense.compare \
            --vocab-size "$VOCAB" \
            --dump-dir "$DUMP_DIR" \
            $SKIP \
            "$@"
    } 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} exit=${rc} ===" | tee -a "$log"
    return $rc
}

# Main: seed=0 across full token-count sweep.
if [ -s "$OUT_DIR/ssh_seed0.csv" ] && [ "$(wc -l < "$OUT_DIR/ssh_seed0.csv")" -ge 7 ]; then
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=ssh_seed0 skipped (CSV complete) ==="
else
    run_phase "ssh_seed0" \
        --max-tokens 100000 500000 1000000 2000000 5000000 10000000 \
        --seed 0 \
        --csv "$OUT_DIR/ssh_seed0.csv" || echo "[warn] phase ssh_seed0 had failures, continuing"
fi

# Variance at 500k, seeds 1 and 2.
for seed in 1 2; do
    csv_path="$OUT_DIR/ssh_variance_seed${seed}.csv"
    if [ -s "$csv_path" ] && [ "$(wc -l < "$csv_path")" -ge 2 ]; then
        echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=ssh_variance_seed${seed} skipped (CSV complete) ==="
        continue
    fi
    run_phase "ssh_variance_seed${seed}" \
        --max-tokens 500000 \
        --seed "$seed" \
        --csv "$csv_path" \
        || echo "[warn] phase ssh_variance_seed${seed} had failures, continuing"
done

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ssh sweep complete ==="
