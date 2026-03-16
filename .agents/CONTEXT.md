# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy + Numba, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, factory functions `make_sensory_region()`, `make_motor_region()`
- **`src/step/cortex/`** — `region.py` (base), `sensory.py`, `motor.py`, `_numba_kernels.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`, `word_selectivity.py`
- **`src/step/data.py`** — BabyLM/PersonaChat/TinyDialogues loaders, `inject_eom_tokens()`
- **`experiments/scripts/`** — `cortex_run.py`, `cortex_repl.py`, `s2_probe.py`, `s2_sweep.py`, `ef_sweep.py`
- **`experiments/checkpoints/`** — `personachat_k4_100k.ckpt`, `babylm_100k.ckpt` (gitignored)

## Full Architecture Vision

### Sensory hierarchy (current focus)
- **S1** (primary): 128 cols, k=8. Char-level features from PositionalCharEncoder.
- **S2** (secondary): 32 cols, k=4, buf=4, burst gating. Word-level distributed patterns.
- **S3** (association/temporal): Topic/theme/tone from S2's word patterns. **Building next.**

### Motor hierarchy (future)
- **PFC** (prefrontal): Decides *what* to respond — intent, reasoning, goal maintenance.
- **M2** (premotor/Broca's): Sequential planning — translates intent into word-level motor plan.
- **M1** (motor): 32 cols, k=4. Char-by-char output via population vote.

### Training stages
1. S1→S2→S3 sensory representation (current)
2. M1→S1 babbling (efference copy / surprise learning)
3. Imitation: hear word → S3 target → M2→M1 reproduce → efference copy error
4. S3→PFC→M2→M1 RL for coherence, then helpfulness

### Key APIs
- **`Topology.step(token_id, token_str)`**: Lightweight single-token processing.
- **`Topology.run(tokens, metric_interval=N)`**: Full training loop with deferred metrics.
- **340 tok/s** with Numba JIT (S1+S2+M1). 100k chars in ~5 min.

## Current State

### S2 settled (sweep complete)
- 32c/k4/buf4/burst is the sweet spot — only config where S2 beats S1 at BPC
- More columns hurts (needs more data). Word representation is fundamentally distributed.
- BabyLM (child-directed speech, 53.5M chars) is the right training dataset.

### M1 tabled
- Token map collapsed to ~7 frequent chars. Needs babbling phases.
- Waiting for S2/S3 to provide useful intent signals.

## Key Decisions
- **BabyLM for training**: Better word repetition than PersonaChat for S2+ regions
- **32c/k4 S2**: Distributed co-activation, not individual word detectors
- **PFC ≠ M2**: Broca's handles sequencing, PFC handles intent/reasoning. Different connectivity.
- **Architecture before scale**: Validate S2↔S3 at 100k, then scale to 1M+

## Next Steps
- [ ] **Build S3** — association region on S2's word patterns. Temporal buffer of S2 L2/3, similar architecture. Probe for topic/theme consistency.
- [ ] **Probe S3** — what abstractions does it learn? Informs PFC vs M2 decision.
- [ ] **Scale BabyLM** — 1M+ chars once S2↔S3 validated
- [ ] **M1 babbling** — after S3 provides useful representations
