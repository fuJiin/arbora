#!/usr/bin/env bash
# Overnight sweep for ARB-139: four-way sparse-vs-dense comparison.
#
# Phases (all writing CSVs + pickle dumps to data/runs/arb139/):
#   1. Light models (word2vec, RI, Brown) at {100k, 500k, 1M, 2M, 5M, 10M}
#      tokens, seed=0. T1 skipped — too slow at 5M+.
#   2. T1 alone at {100k, 500k, 1M} tokens, seed=0. Caps T1 at 1M
#      because production T1 measured at ~14 min/100k tokens at
#      vocab=5000, n_columns=256 (so 1M ≈ 2.3 h, 2M would add 4.7 h).
#   3. Variance — all four models at 500k tokens for seeds {1, 2}.
#      500k chosen so T1's two replicates fit (~70 min × 2 = ~2.3 h)
#      while still giving variance bands at a meaningful corpus size.
#
# Each phase logs to its own file under data/runs/arb139/logs/ so a failure
# in one phase doesn't lose progress from the others. Total wall-clock
# budget estimate: 6-7 hours.
#
# Usage:
#   bash examples/text_exploration/sparse_vs_dense/sweep_arb139.sh
#
# Outputs:
#   data/runs/arb139/light_seed0.csv         — phase 1
#   data/runs/arb139/t1_seed0.csv            — phase 2
#   data/runs/arb139/variance_seed{1,2}.csv  — phase 3
#   data/runs/arb139/dumps/                  — pickled embeddings per (model, n_tokens, seed)
#   data/runs/arb139/logs/<phase>.log        — full stdout per phase

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="data/runs/arb139"
DUMP_DIR="$OUT_DIR/dumps"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$DUMP_DIR" "$LOG_DIR"

VOCAB=5000

run_phase() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} starting ===" | tee -a "$log"
    {
        time uv run python -m examples.text_exploration.sparse_vs_dense.compare \
            --vocab-size "$VOCAB" \
            --dump-dir "$DUMP_DIR" \
            "$@"
    } 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=${name} exit=${rc} ===" | tee -a "$log"
    return $rc
}

# Phase 1: light models across the full token-count sweep.
# Skip if CSV already exists (idempotent — lets us resume after a kill).
if [ -s "$OUT_DIR/light_seed0.csv" ]; then
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=light_seed0 skipped (CSV present) ==="
else
    run_phase "light_seed0" \
        --max-tokens 100000 500000 1000000 2000000 5000000 10000000 \
        --seed 0 \
        --skip t1 \
        --csv "$OUT_DIR/light_seed0.csv" || echo "[warn] phase light_seed0 had failures, continuing"
fi

# Phase 2: T1 alone, capped at 1M tokens (~3.5 hours wall).
run_phase "t1_seed0" \
    --max-tokens 100000 500000 1000000 \
    --seed 0 \
    --skip word2vec random_indexing brown_cluster \
    --csv "$OUT_DIR/t1_seed0.csv" || echo "[warn] phase t1_seed0 had failures, continuing"

# Phase 3: variance at 500k tokens, seeds 1 and 2, all models.
for seed in 1 2; do
    run_phase "variance_seed${seed}" \
        --max-tokens 500000 \
        --seed "$seed" \
        --csv "$OUT_DIR/variance_seed${seed}.csv" \
        || echo "[warn] phase variance_seed${seed} had failures, continuing"
done

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] sweep complete ==="
ls -la "$OUT_DIR"
