# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated, source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (40% sparse per source)
  S2+PFC→M2 (40% sparse per source)
  M2→M1

Apical (multi-source, per-source gain weights):
  S3→S2, S2→S1, M1→M2, M2→PFC, S1→M1, M1→S1

Surprise: S1→S2, S2→S3, S1→M1
```

## Learning Mechanism: STDP-like Presynaptic Traces

Implemented in CorticalRegion base class, applies to ALL connections:
- `pre_trace_decay`: ff_weight learning uses decaying input trace
- `_seg_trace_l23/l4`: segment learning uses decaying activity traces
- `_pre_trace_threshold`: sparsity control (ignore faint echoes)
- Three-factor regions (PFC, M1): pre_trace feeds eligibility → reward
- Default pre_trace_decay=0.0 (disabled, coincidence only)

### Sweep Results (ff_weights only, no segment traces)
Decoder BPC improves (9.94→8.50) with longer traces. Burst rate unaffected because surprise is driven by segments, not ff_weights.

### Sweep Results (ff + segment traces)
- burst INCREASES (49%→59-63%) — segment predictions get LESS precise
- centroid BPC IMPROVES (7.79→7.35) — representations more discriminative
- Hypothesis: thresholded traces create broader "active context" for segments, making matches less precise. Segments need higher threshold or longer training to adapt.

### Evaluation
- **Primary**: burst rate (surprise) — what the model actually optimizes
- **Secondary**: dendritic decoder (how downstream regions would read)
- **Deprecated**: centroid BPC (external probe, not architecturally grounded)

## Key Results
- **Structural sparsity**: 40% per-source on PFC/M2 → 38% echo improvement
- **PFC three-factor**: baseline echo 3.1%→8.2%
- **Eligibility clip (0.05)**: only consistent tuning fix across sweeps

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Next Steps
- [ ] **Tune segment trace threshold** — current 0.01 too low, segments match too broadly. Try 0.1-0.5 or scale with decay.
- [ ] **Longer training with traces** — 100k may not be enough for segments to adapt to trace-based context
- [ ] **Make pre_trace_decay>0 default** once tuned — universal STDP learning
- [ ] **Performance**: threshold + numba for trace-based learning
- [ ] **Cerebellar forward model** — M1→predicted S1→error→M2
- [ ] **Recurrent PFC** — replace passive decay
- [ ] **M2 three-factor** — credit assignment gap
