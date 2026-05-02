#!/usr/bin/env bash
# ARB-139: proportional-λ vs absolute-λ BCM consolidation sweep.
#
# Spec: vault/specs/proportional-lambda-consolidation.md
#
# 5 seeds × 4 variants = 20 cells. SSH phase-1 cached per (seed) and
# reused across the 3 SSH variants (baseline-no-bonus, BCM-absolute,
# BCM-proportional) so phase 1 runs once per seed, not three times.
#
# Scale: 1M tokens per phase (scaled down from spec's 5M to fit in a
# single Claude session — see investigator-execute log for justification).
# Spec's matched-8-epoch phase 1 + peak-detection revert preserved.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_ROOT="runs/arb139/proportional_lambda_2026-05-02"
LOG_DIR="$OUT_ROOT/logs"
CACHE_DIR="$OUT_ROOT/ssh_phase1_caches"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$CACHE_DIR"

N_PER_PHASE=1000000
N_DIMS=1024
K_ACTIVE=40
K_EVAL=160
N_EPOCHS_PHASE1=8
N_EPOCHS_PHASE2=1
N_WORKERS=4
PATIENCE=3
VOCAB_HINT=10000
BONUS=0.5
SEEDS="${SEEDS:-0 1 2 3 4}"

run_cell() {
    local seed="$1"
    local variant="$2"
    local out="$OUT_ROOT/seed_${seed}/${variant}"
    local log="$LOG_DIR/seed${seed}_${variant}.log"
    if [ -f "$out/results.csv" ]; then
        echo "=== skip seed=${seed} variant=${variant} (results.csv present) ==="
        return
    fi
    mkdir -p "$out"
    echo "=== seed=${seed} variant=${variant} ==="
    case "$variant" in
        w2v_baseline)
            uv run --extra embeddings --extra viz python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
                --n-per-phase "$N_PER_PHASE" \
                --split cross_domain \
                --methods word2vec \
                --seed "$seed" \
                --n-epochs-phase1 "$N_EPOCHS_PHASE1" \
                --n-epochs-phase2 "$N_EPOCHS_PHASE2" \
                --vocab-size-hint "$VOCAB_HINT" \
                --out-dir "$out" \
                2>&1 | tee "$log"
            ;;
        ssh_baseline)
            uv run --extra embeddings --extra viz python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
                --n-per-phase "$N_PER_PHASE" \
                --split cross_domain \
                --methods ssh \
                --seed "$seed" \
                --ssh-n-dims "$N_DIMS" \
                --ssh-k-active "$K_ACTIVE" \
                --ssh-k-eval "$K_EVAL" \
                --ssh-cache-phase1 "$CACHE_DIR/seed_${seed}.pkl" \
                --ssh-consolidation-bonus 0.0 \
                --ssh-early-stop-patience "$PATIENCE" \
                --n-epochs-phase1 "$N_EPOCHS_PHASE1" \
                --n-epochs-phase2 "$N_EPOCHS_PHASE2" \
                --ssh-n-workers "$N_WORKERS" \
                --vocab-size-hint "$VOCAB_HINT" \
                --out-dir "$out" \
                2>&1 | tee "$log"
            ;;
        ssh_absolute)
            uv run --extra embeddings --extra viz python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
                --n-per-phase "$N_PER_PHASE" \
                --split cross_domain \
                --methods ssh \
                --seed "$seed" \
                --ssh-n-dims "$N_DIMS" \
                --ssh-k-active "$K_ACTIVE" \
                --ssh-k-eval "$K_EVAL" \
                --ssh-cache-phase1 "$CACHE_DIR/seed_${seed}.pkl" \
                --ssh-consolidation-bonus "$BONUS" \
                --ssh-consolidation-mode absolute \
                --ssh-early-stop-patience "$PATIENCE" \
                --n-epochs-phase1 "$N_EPOCHS_PHASE1" \
                --n-epochs-phase2 "$N_EPOCHS_PHASE2" \
                --ssh-n-workers "$N_WORKERS" \
                --vocab-size-hint "$VOCAB_HINT" \
                --out-dir "$out" \
                2>&1 | tee "$log"
            ;;
        ssh_proportional)
            uv run --extra embeddings --extra viz python -m examples.text_exploration.sparse_vs_dense.continual_learning_v2 \
                --n-per-phase "$N_PER_PHASE" \
                --split cross_domain \
                --methods ssh \
                --seed "$seed" \
                --ssh-n-dims "$N_DIMS" \
                --ssh-k-active "$K_ACTIVE" \
                --ssh-k-eval "$K_EVAL" \
                --ssh-cache-phase1 "$CACHE_DIR/seed_${seed}.pkl" \
                --ssh-consolidation-bonus "$BONUS" \
                --ssh-consolidation-mode proportional \
                --ssh-early-stop-patience "$PATIENCE" \
                --n-epochs-phase1 "$N_EPOCHS_PHASE1" \
                --n-epochs-phase2 "$N_EPOCHS_PHASE2" \
                --ssh-n-workers "$N_WORKERS" \
                --vocab-size-hint "$VOCAB_HINT" \
                --out-dir "$out" \
                2>&1 | tee "$log"
            ;;
    esac
}

for seed in $SEEDS; do
    # Run SSH baseline first per seed so the phase-1 cache is built before
    # the absolute/proportional variants try to load it.
    run_cell "$seed" ssh_baseline
    run_cell "$seed" ssh_absolute
    run_cell "$seed" ssh_proportional
    run_cell "$seed" w2v_baseline
done

echo "=== sweep complete ==="
