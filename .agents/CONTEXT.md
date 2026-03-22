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

## Learning: STDP Presynaptic Traces (DEFAULT ON)

`pre_trace_decay=0.8` in CortexConfig, all regions from construction.
- **FF traces**: decaying input → ff_weight LTP (temporal credit)
- **Segment traces**: decaying activity → segment growth/adapt
- **Prediction stays boolean** — traces for plasticity, not activation
- Three-factor (PFC, M1): pre_trace feeds eligibility → reward
- `_pre_trace_threshold`: sparsity control (default 0.0)

### Key validation
Echo with traces from construction: **7.3% avg, 7.5% last50** (best).
Traces patched on after construction: 3.6% (worse than no traces 6.0%).
**Lesson**: all regions must develop together with traces from step 1.

## Validated Results
- STDP traces from construction: 7.3% echo (best, still improving)
- Structural sparsity: 38% echo improvement (6.9% vs 5.0%)
- PFC three-factor: 3.1% → 8.2% echo
- 300k trace sensory: decoder BPC 3.63 (vs ~5.6 baseline)

## Evaluation
- **Primary**: burst rate (surprise)
- **Secondary**: decoder BPC (dbpc in logs)
- Centroid BPC (cbpc): in logs, being deprecated

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Next Steps
- [ ] **Longer echo with traces** (2k-5k episodes) — still improving at 500
- [ ] **Tune decay per region** — sensory/PFC/M1 may benefit from different values
- [ ] **Full staged training** with traces (sensory → babbling → echo)
- [ ] **Cerebellar forward model** — M1→predicted S1→error→M2
- [ ] **Recurrent PFC** — replace passive voltage decay
- [ ] **M2 three-factor** — credit assignment gap
- [ ] **Performance**: numba for trace-based learning
