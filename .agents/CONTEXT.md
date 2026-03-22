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

In CorticalRegion base. Traces affect LEARNING only, not prediction:
- **FF traces** (`_pre_trace`): decaying input trace for ff_weight LTP
- **Segment traces** (`_seg_trace_l23/l4`): decaying activity traces for segment growth/adapt
- **Prediction stays boolean** (current state) — biologically correct
- Three-factor (PFC, M1): pre_trace feeds eligibility → reward
- Default pre_trace_decay=0.0 (disabled). 0.8 tested, shows improvement.

### 300k Trace Results
- **Decoder BPC: 3.63** (vs ~5.6 baseline) — representations highly decodable
- Centroid BPC: 7.23 (vs 7.79 baseline) — also improved
- Burst rate: unknown from probe, was 57.6% at 100k (higher than 49% baseline)
- Segments learn richer patterns via traces but predict from boolean — tension that may resolve with longer training

### Evaluation
- **Primary**: burst rate (surprise) — what the model optimizes
- **Secondary**: decoder BPC (dbpc in logs) — architecturally grounded
- Centroid BPC (cbpc): kept in logs but being deprecated

## Validated Results
- Structural sparsity: 38% echo improvement (6.9% vs 5.0%)
- PFC three-factor: 3.1% → 8.2% echo
- Eligibility clip (0.05): only consistent tuning fix

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Next Steps
- [ ] **Echo with traces** — do better representations help motor output?
- [ ] **Tune decay per region** — sensory/PFC/M1 may want different values
- [ ] **Make traces default** once echo results confirm
- [ ] **Performance**: numba for trace-based learning
- [ ] **Cerebellar forward model** — M1→predicted S1→error→M2
- [ ] **Recurrent PFC** — replace passive voltage decay
- [ ] **M2 three-factor** — credit assignment gap
