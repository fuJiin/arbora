# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (40% sparse), S2+PFC→M2 (40% sparse)
  M2→M1

Apical (multi-source, per-source gain weights):
  S3→S2, S2→S1, M1→M2, M2→PFC, S1→M1, M1→S1

Surprise: S1→S2, S2→S3, S1→M1
```

## Learning: STDP Presynaptic Traces

Implemented in CorticalRegion base. Two separate trace systems:

**FF traces** (`_pre_trace`): decaying input trace for ff_weight LTP.
Inputs that preceded activation get temporal credit.

**Segment traces** (`_seg_trace_l23/l4`): decaying activity traces for
segment growth/adapt. Segments grow connections to recently-active
neurons (not just currently-active). Prediction stays boolean (current
state only) — traces affect plasticity, not activation. This is
biologically correct: STDP modifies synaptic strength, not firing.

Key insight from sweep: traces-for-learning-only gives best centroid
BPC ever (6.88 vs 7.79 baseline) but burst rate increases (57.6% vs
49.0%). Segments learn richer multi-step patterns but can only verify
single-step state at prediction time. Need longer training for segments
to adapt.

**300k trace run in background** — check results next session.
Checkpoint: `experiments/checkpoints/stage1_sensory_traces.ckpt`
Run: `experiments/runs/sensory-traces-300k--*`

## Key Parameters
- `pre_trace_decay`: 0.0 = disabled (default), 0.8 = good for sensory
- `_pre_trace_threshold`: sparsity control on ff traces
- Segment traces share decay rate with ff traces
- Three-factor (PFC, M1): pre_trace feeds eligibility → reward
- Two-factor (sensory, M2): pre_trace used directly in Hebbian LTP

## Validated Results
- Structural sparsity: 38% echo improvement (6.9% vs 5.0%)
- PFC three-factor: 3.1% → 8.2% echo
- Eligibility clip (0.05): only consistent tuning fix

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Next Steps
- [ ] **Check 300k trace results** — does burst rate converge?
- [ ] **Make traces default** once decay tuned per region
- [ ] **Performance**: numba for trace-based learning
- [ ] **Cerebellar forward model** — M1→predicted S1→error→M2
- [ ] **Recurrent PFC** — replace passive voltage decay
- [ ] **M2 three-factor** — credit assignment gap
