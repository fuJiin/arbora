# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy + Numba, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, factory functions, `_default_region3_config()`
- **`src/step/cortex/`** — `region.py` (base), `sensory.py`, `motor.py`, `_numba_kernels.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`, `word_selectivity.py`
- **`src/step/data.py`** — BabyLM/PersonaChat/TinyDialogues loaders
- **`experiments/scripts/`** — `cortex_run.py` (--s3, --no-s3-feedback), `s2_probe.py`, `s3_probe.py`, `s2_sweep.py`, `feedback_ab.py`
- **`experiments/checkpoints/`** — `babylm_100k.ckpt`, `babylm_s3_100k.ckpt` (gitignored)

## Architecture

### Sensory hierarchy
- **S1**: 128 cols, k=8, PositionalCharEncoder. Char-level features.
- **S2**: 32 cols, k=4, buf=4, burst gating. Word-level distributed patterns.
- **S3**: 32 cols, k=4, buf=8, burst gating. Topic/phrase-level patterns.

### Connection types
- **Feedforward**: Learned ff_weights (Hebbian LTP/LTD). Additive drive to L4.
- **Lateral (L4)**: Dendritic segments (feedback L2/3→L4, lateral L4→L4). Voltage boost on predicted neurons.
- **Lateral (L2/3)**: Dendritic segments only. Dense l23_lateral_weights REMOVED — was redundant with segments.
- **Apical feedback**: Currently scalar gain modulation (no-op in A/B test). Needs per-neuron learned gain.

### Key APIs
- **`Topology.step()`** / **`Topology.run()`**: Single-token and batch processing.
- **`cortex_run.py --dataset babylm --s3`**: Full hierarchy training.
- **~280s for 100k** with 4 regions (S1+S2+S3+M1). 22% faster after removing dense lateral weights.

## Current State

### Architecture cleanup (2026-03-16)
- **Removed dense L2/3 lateral weights**: Redundant with L2/3 segments. No quality loss, 22% speedup (363s→282s for 100k).
- **Removed instructive apical segments**: Replaced with scalar gain modulation. Old approach helped receiver (S1 BPC -0.24) but hurt sender (S2 words 242→101).
- **Scalar gain is too weak**: A/B test shows it's a no-op. Need per-neuron learned gain (biological BAC firing model).

### S3 validated
- S3 BPC 5.42-5.48 (best region). Consistent words 232-250. Topic-specific patterns confirmed.
- Hierarchy working: each level extracts increasingly abstract features.

### Feedback A/B results (3 rounds)
1. **Instructive segments**: Helped receiver, hurt sender. S2 words 242→101 with full feedback.
2. **Scalar gain**: No-op. All metrics within noise.
3. **After lateral cleanup**: Same no-op for scalar gain. But clean baseline established.

## Key Decisions
- **BabyLM for training**: 53.5M chars child-directed speech
- **32c/k4 for S2 and S3**: Distributed co-activation
- **Segments only for L2/3 lateral**: No dense weight matrix
- **PFC needs working memory**: Persistent firing / high voltage_decay, not just another sensory region

## Next Steps (Priority Order)
- [ ] **Per-neuron apical gain** — weight matrix S2_l23 → S1_l4, multiplicative on voltage, slow Hebbian learning. Models biological BAC firing. A/B test to verify helps receiver without hurting sender.
- [ ] **Scale to 1M BabyLM** — once apical is working, train full hierarchy
- [ ] **M1 babbling** — staged motor exploration
- [ ] **PFC design** — working memory, multi-region input, goal maintenance
