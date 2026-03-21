# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated, source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (word+topic → goal, 40% sparse per source)
  S2+PFC→M2 (word+goal → sequence, 40% sparse per source)
  M2→M1 (sequence → execution)

Apical (multi-source, per-source gain weights):
  Sensory top-down: S3→S2, S2→S1
  Motor monitoring: M1→M2, M2→PFC (corollary discharge)
  Cross: S1→M1 (sensory context), M1→S1 (efference copy)

Surprise: S1→S2, S2→S3, S1→M1
```

Learning: STDP-like presynaptic traces implemented in CorticalRegion base.
- pre_trace_decay param on all regions (default 0.0 = coincidence only)
- Pre fires → trace accumulates. Post fires → synapses with traces strengthened.
- Three-factor (PFC, M1): pre_trace feeds into eligibility, consolidated by reward
- Two-factor (sensory, M2): pre_trace used directly in Hebbian LTP
- **Next**: extend traces to segment connections (lateral, feedback), not just ff_weights

Evaluation: burst rate (surprise) is primary metric. Dendritic decoder for predictions/interpretability. Centroid BPC to be deprecated — decoder is architecturally consistent ("how would a downstream region read this").

## Key Results

### STDP pre_trace sweep (ff_weights only)
Decoder BPC improves with traces (9.94 → 8.50) but burst rate doesn't benefit (49% → 50%). Pre_traces affect ff_weights (column selection) but surprise is driven by segments (temporal prediction). **Need traces on segment connections too** for surprise reduction.

### Structural sparsity validated
40% per-source on PFC/M2 improves echo 38% (6.9% vs 5.0%).

### Echo
- PFC three-factor biggest win: 3.1% → 8.2%
- Eligibility clip (0.05) only consistent tuning fix

### Biology comparison (language focus)
Good: columnar org, hierarchy, temporal integration, apical gain, segments, surprise modulation, BG gating, developmental stages, efference copy
Gaps: no cerebellum, passive PFC decay, no pattern completion, no dual stream

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Next Steps (Priority Order)
- [ ] **Pre_traces on segment connections** — extend STDP traces to lateral/feedback segments, not just ff_weights. This is where temporal prediction happens; traces here should reduce burst rate.
- [ ] **Make pre_trace_decay > 0 the default** for all regions, sweep for optimal decay per region type
- [ ] **Drop centroid BPC** from evaluation path, keep decoder + burst rate
- [ ] **Performance**: threshold pre_trace for sparsity, numba for _learn_ff
- [ ] **Minimal cerebellar forward model** — M1→predicted S1→error→M2
- [ ] **Recurrent PFC maintenance** — replace passive voltage decay
- [ ] **M2 three-factor** — credit assignment gap
