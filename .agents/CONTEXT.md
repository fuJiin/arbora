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
- S3: 32c/k4, buf=8 on S2 L2/3, burst gating, apical feedback to S2
- **S3 BPC: 5.52** — almost matches S1 (5.47), better than S2 (5.75)
- **S3 consistent words: 223** (vs S2's 109). 2x improvement with same column count.
- S3's top words are topic-specific: "mosquitoes", "tickling", "microphone", "catherine's"
- S2's top words are phonetically distinctive: "yummy", "chomp", "vroom"
- **Hierarchy is working**: each level extracts increasingly abstract features
- S3→S2 thalamic gate at 0.35-0.47 after 100k — still slowly opening

### S2 settled (sweep complete)
- 32c/k4/buf4/burst is the sweet spot — only config where S2 beats S1 at BPC
- Word representation is fundamentally distributed (no selective columns in any config)

### M1 tabled
- Token map collapsed to ~7 frequent chars. Needs babbling phases after sensory hierarchy is scaled.

## Key Decisions
- **BabyLM for training**: Child-directed speech, 53.5M chars, better word repetition than PersonaChat
- **32c/k4 for S2 and S3**: Distributed co-activation works well at this scale
- **S3 buf=8**: Spans ~8 words (one phrase). Captures topic-level patterns.
- **PFC ≠ M2**: Broca's handles sequencing, PFC handles intent/reasoning
- **Architecture before scale**: Validate hierarchy at 100k, then scale to 1M+

## Next Steps
- [ ] **Scale BabyLM training** — 1M+ chars with S1→S2→S3. Sensory hierarchy validated, needs more data for S3 to mature.
- [ ] **Dashboard updates** — add S3 panels, 3-region hierarchy comparison, word selectivity viz
- [ ] **S3→S2 apical feedback tuning** — gate is barely open at 100k. May need more training or lower threshold.
- [ ] **M1 babbling** — staged motor exploration once sensory hierarchy is scaled
- [ ] **PFC/M2 planning** — design intent→sequencing→output pipeline
