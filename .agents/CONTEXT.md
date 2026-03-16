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

### S3 built and probed (babylm_s3_100k checkpoint)
- S3: 32c/k4, buf=8 on S2 L2/3, burst gating
- **S3 BPC: 5.45** — best of all regions. S3 consistent words: 232 (vs S2's 242 without feedback).
- S3's top words are topic-specific: "mosquitoes", "tickling", "microphone"
- **Hierarchy working**: each level extracts increasingly abstract features

### Apical feedback A/B test (BabyLM 100k, 4 configs)
- **S2→S1 helps S1**: BPC 5.87 → 5.64 (0.24 improvement). S2 word patterns ARE useful for S1.
- **But feedback hurts the sender**: S2 consistent words drop 242 → 195 (S2→S1) or 242 → 112 (S3→S2).
- **S3→S2 hurts S2 the most**: consistent words 242 → 112, BPC 5.51 → 5.79.
- **S3 is stable regardless** of feedback config (BPC 5.45-5.51).
- **Root cause**: apical feedback is too instructive, disrupts sender's internal learning. Should be modulatory (gain control), not additive. Fix later.
- **Best config for scaling: S2→S1 feedback ON, S3→S2 feedback OFF.**

### S2 settled (sweep complete)
- 32c/k4/buf4/burst. Distributed co-activation, no selective columns.

### M1 tabled
- Token map collapsed to ~7 frequent chars. Needs babbling after sensory hierarchy scales.

## Key Decisions
- **BabyLM for training**: 53.5M chars of child-directed speech
- **32c/k4 for S2 and S3**: Distributed co-activation at this scale
- **S2→S1 feedback ON, S3→S2 OFF**: Feedback helps receiver but hurts sender. Use only where net benefit is clear.
- **PFC ≠ M2**: Broca's = sequencing, PFC = intent/reasoning
- **Architecture before scale**: Validated at 100k, now scaling to 1M

## In Progress
- **1M BabyLM training**: S1→S2→S3+M1, S2→S1 apical ON, S3→S2 OFF. Checkpoint: `babylm_s3_1m`

## Next Steps
- [ ] **Analyze 1M results** — probe S1/S2/S3 at scale. Does S3 mature with more data?
- [ ] **Fix apical feedback mechanism** — make modulatory (gain) instead of instructive (additive) so it doesn't hurt sender
- [ ] **Dashboard updates** — add S3 panels, 3-region comparison
- [ ] **M1 babbling** — staged motor exploration
- [ ] **PFC/M2** — design intent→sequencing→output pipeline
