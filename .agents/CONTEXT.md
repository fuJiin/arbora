# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, factory functions `make_sensory_region()`, `make_motor_region()`
- **`src/step/cortex/`** — `region.py` (base with ff weights), `sensory.py` (encoding masking + local connectivity), `motor.py` (L5 readout), `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`, `word_selectivity.py`
- **`src/step/data.py`** — `prepare_tokens_charlevel()`, `prepare_tokens_personachat()`, `inject_eom_tokens()`, BabyLM/PersonaChat/TinyDialogues
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`
- **`experiments/scripts/`** — `cortex_run.py`, `cortex_repl.py`, `s2_probe.py`, `s2_sweep.py`, `ef_sweep.py`
- **`experiments/checkpoints/`** — `personachat_k4_100k.ckpt`, `babylm_100k.ckpt` (gitignored)

## Architecture

### Class hierarchy
- **CorticalRegion** (`region.py`): ff_weights, process(), _learn_ff(), reconstruct(), efference copy (`efference_gain`), dendritic segments, L4/L2/3 pipeline
- **SensoryRegion** (`sensory.py`): encoding-width receptive field masking, local connectivity for segments + L2/3 lateral mask
- **MotorRegion** (`motor.py`): L5 readout, population vote, column→token mapping. Inherits SensoryRegion (for local connectivity)

### S1 → S2 → M1 + BasalGanglia
- **S1**: 128 cols, k=8, ltd=0.05, synapse_decay=1.0, PositionalCharEncoder
- **S2**: 32 cols k=4, temporal buffer (depth=4) + burst gating, apical feedback to S1, thalamic gating
- **M1**: 32 cols k=4, L5 readout threshold 0.3, population vote
- **BG**: Go/no-go gate, three-factor plasticity, supervised gate-error (Stage 1)
- **Efference copy**: During `force_gate_open`, S1 subtracts `efference_gain * predicted_drive`. Default gain=1.0.

### Key APIs
- **`Topology.step(token_id, token_str)`**: Lightweight single-token processing (no metrics overhead). ~1.4x faster than `run()`.
- **`Topology.run(tokens, metric_interval=N)`**: Full training loop. `metric_interval` controls expensive decode sampling.
- **`cortex_run.py --dataset babylm`**: BabyLM support with synthetic EOM every 200 chars.

## Training Results

### BabyLM S2 probe results (babylm_100k checkpoint, 50k probe chars)
- **S2 BPC: 5.67 — BEATS S1's 5.82** (first time S2 outperforms S1 at char prediction)
- **S2 consistent words: 288** (vs S1's 151, vs 123 on PersonaChat)
- S2 mean word consistency: 0.338 (vs S1's 0.291)
- Still no truly selective columns (entropy > 0.85 everywhere)
- **Diagnosis**: 32 columns can't specialize for individual words — they represent word *classes* via co-activation patterns. More columns may help.

### BabyLM dataset characteristics
- 53.5M total chars, we trained on 100k (0.2%)
- 79 words with 50+ occurrences in 100k — much better repetition than PersonaChat
- Child-directed speech: simple vocab, avg word length 4.1 (matches S2 buffer_depth=4)

### S2 architecture sweep results (BabyLM 100k, 6 configs)
- **32c/k4/buf4/burst is the best** — only config where S2 beats S1 at char prediction (5.57 vs 5.90 BPC)
- More columns hurts: 64c→6.2, 128c→7.2 BPC. Larger ff_weights needs more data than 100k.
- No burst gating (64c/k8) has most consistent words (276) but worse BPC — overwhelmed by redundant S1 signal
- Deeper buffer (buf8) doesn't help — avg word length 4.1 matches buf4
- **Zero selective columns in all configs** — word representation is fundamentally distributed
- **Conclusion**: 32c/k4 is the sweet spot. Don't scale columns — build S3 on top.

### M1 token map collapse (diagnosed 2026-03-16)
- Column→token mapping collapsed to ~7 high-frequency chars
- **Tabled**: needs babbling phases, waiting for S2/S3

### Performance (optimized 2026-03-16)
- **340 tok/s** with Numba JIT kernels (was 247). 100k chars in ~5 min.
- Numba `@njit` on: `predict_segments`, `grow_segment`, `adapt_segments_batch`
- Graceful fallback to NumPy if numba not installed

## Key Decisions
- **BabyLM for S2 tuning**: Better word repetition profile than PersonaChat
- **32c/k4 S2 is the sweet spot**: More columns doesn't help at 100k scale. Distributed co-activation is the right abstraction.
- **Architecture before scale**: Get S2↔S3 interface right at 100k, then scale to 1M+.
- **S2/S3 before M1 babbling**: Higher-level representations are the blocker for useful M1 generation.
- **Efference copy over neural adaptation**: Architecturally consistent, no special-case behavior.

## Next Steps (Priority Order)
- [ ] **Build minimal S3** — higher-level region on top of S2's 32c/k4 distributed word patterns. What abstraction does S3 learn?
- [ ] **Scale BabyLM training** — 1M+ chars once S2↔S3 architecture is validated
- [ ] **M1 babbling phases** — staged motor exploration once S2/S3 provide useful representations
- [ ] **Feedback gating** — adaptive S2→S1 once S2 is demonstrably useful
