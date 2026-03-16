# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy + Numba, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `_default_region3_config()`
- **`src/step/cortex/`** — `region.py` (base + apical gain), `sensory.py`, `motor.py`, `_numba_kernels.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `bpc.py`, `word_selectivity.py`
- **`experiments/scripts/`** — `cortex_run.py` (--s3), `s3_probe.py`, `s2_sweep.py`, `feedback_ab.py`

## Architecture

### Sensory hierarchy
- **S1**: 128 cols, k=8. Char-level.
- **S2**: 32 cols, k=4, buf=4, burst gating. Word-level distributed patterns.
- **S3**: 32 cols, k=4, buf=8, burst gating. Topic/phrase-level patterns.

### Connection types
- **Feedforward**: Learned ff_weights (Hebbian). Additive L4 drive.
- **Lateral**: Dendritic segments only (L4 fb/lat, L2/3 lat). Dense L2/3 weights removed.
- **Apical feedback**: Per-neuron learned gain weights (BAC firing model). `gain = 1 + ctx @ weights`, multiplicative on voltage. 10x slower Hebbian learning + passive decay.

### Per-neuron apical gain — breakthrough result
- **A/B on BabyLM 100k with full feedback (S2→S1 + S3→S2):**
  - S1 BPC: 5.98 → 5.65 (**-0.33**, better than old instructive segments' -0.24)
  - S2 consistent words: 237 → 300 (**+26%**, sender HELPED not hurt)
  - S3 consistent words: 240 → 269 (+12%)
  - S3→S2 alone gives S2 words 237→295 — phrase context improves word representations
- **3 approaches tested**: instructive segments (hurt sender), scalar gain (no-op), per-neuron gain (helps both)

## In Progress
- **1M BabyLM training** running: S1→S2→S3+M1, full apical feedback, checkpoint `babylm_s3_1m`

## Key Decisions
- **Per-neuron apical gain**: Multiplicative, slow learning. Biologically grounded (BAC firing). Helps both sender and receiver.
- **BabyLM**: 53.5M chars child-directed speech
- **32c/k4 for S2 and S3**: Distributed co-activation
- **Segments only for L2/3 lateral**: Dense weight matrix removed
- **PFC needs working memory**: Persistent firing, not just another sensory region

## Next Steps
- [ ] **Analyze 1M results** — probe S1/S2/S3 at scale with apical gain
- [ ] **M1 babbling** — staged motor exploration (plan in memory)
- [ ] **PFC design** — working memory, multi-region input, goal maintenance
