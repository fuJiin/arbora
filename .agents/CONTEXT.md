# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron feedforward weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Project Structure
- **`src/step/cortex/`** — `sensory.py`, `motor.py` (babbling), `topology.py` (freeze/enable), `stages.py`, `modulators.py`, `basal_ganglia.py`
- **`src/step/probes/`** — `centroid_bpc.py` (primary metric), `bpc.py` (dendritic, deprecated), `diagnostics.py`, `representation.py`
- **`experiments/scripts/`** — `cortex_run.py`, `cortex_staged.py` (staged runner), `cortex_repl.py` (with /info guardrails)

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **M1**: 32 cols, k=4. Motor output with babbling (direct column forcing).
- **Apical**: Per-neuron gain (BAC firing model). Near no-op at 1M scale (decay tuning needed).

## Key Results

### Centroid BPC (non-learned probe, replaces dendritic decoder)
- **1M sensory run**: cbpc 4.79 (100k) → 4.59 (300k) → 4.79 (1M). Plateaus at ~300k.
- Random baseline ~5.0 (log2(32)). Model is learning, but gains diminish after 300k.
- Dendritic decoder showed false regression (7.6→8.4); centroid confirmed monotonic learning.
- **Implication**: Sensory stage can be 300-500k tokens, not 1M.

### Motor babbling (Stage 2)
- Input noise mixing failed: M1 ff_weights collapsed to single dominant pattern.
- **Direct column forcing works**: bypasses ff_weights, randomly activates k columns. All 32 cols explored uniformly, 18/31 unique tokens in col_token_map.
- Meaningful col→token mapping needs Stage 3 reward signal (frequency dominates without RL).

## Training Stages (implemented)

Infrastructure: `region.learning_enabled`, `connection.enabled`, `Topology.freeze/unfreeze_region`, `TrainingStage.configure()`, `cortex_staged.py`.

1. **Sensory** (S1→S2→S3): 300-500k tokens. Self-supervised. Plateaus by 300k. ✅ Validated.
2. **Babbling** (M1 direct forcing, S1 frozen): 200k tokens. Diverse column exploration working. ✅ Mechanism built.
3. **Guided babbling** (M1+BG+S2): S2 word-recognition as reward. Needs implementation of reward pathway.
4. **Imitation** (S1→S2→M2→M1): Echolalia. Needs M2 region.
5. **Generation** (PFC→M2→M1): Goal-directed RL. Needs PFC region.

### Stages vs BG/thalamus design principle
- **Stages** = hardware readiness (which circuits online). Discrete, developmental.
- **BG/thalamus** = software (how circuits used). Continuous, learned within-stage.

## PFC Design (researched, not implemented)
- **Working memory**: Slow voltage decay (~0.97) + binary maint_flag toggled by BG per-stripe gate
- **Representations**: Mixed-selective, conjunctive. Receives S2+S3. Denser activation.
- **Learning**: Three-factor Hebbian (pre × post × reward). Eligibility trace decay ~0.98.
- **Minimal**: 16 cols, 4 stripes of 4, k=4. Input gating only (skip output gating initially).

## Parameter Tuning Needed
- **M1 col_token_map**: Frequency-dominated. Needs count normalization or recency weighting.
- **Apical gain decay**: Near no-op at 1M. Learning/decay balance needs A/B with centroid BPC.
- **Sensory stage length**: 300k sufficient (plateau), saves training time.

## Checkpoints
- `babylm_s3_1m_v2.ckpt` — 1M sensory with centroid probe
- `babylm_s3_100k.ckpt` / `stage1_sensory.ckpt` — 100k sensory (used for babbling tests)
- `stage2_babbling.ckpt` — after 200k babbling (direct column forcing)

## Next Steps
- [ ] **Stage 3 guided babbling** — wire S2 word-recognition as BG reward signal
- [ ] **M1 col_token_map normalization** — recency-weighted or frequency-normalized counts
- [ ] **Apical gain A/B** — tune decay rate using centroid BPC at 300k
- [ ] **REPL centroid integration** — replace dendritic decoder with centroid probe in REPL
- [ ] **M2 sequencing region** — bridge S2 word patterns to M1 character sequences
- [ ] **PFC implementation** — maint_voltage, maint_flag, per-stripe BG, three-factor learning
