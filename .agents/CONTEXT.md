# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron ff_weights, apical gain feedback, L5 output layer. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8.
- **M1**: 32 cols, k=4. L5 output weights (learned, sparse, three-factor). Direct column forcing for babbling.

## Reward Architecture (biologically grounded)
- **Curiosity (VTA dopamine RPE)**: per-bigram expected vs actual S1 burst. Drives exploration — known patterns stop being rewarding.
- **Caregiver (social reward)**: live prefix tracking against BabyLM vocabulary. Positive-only signal for word-like sequences. No dead-end penalty (caused attractor collapse). Attention span auto-reset at 8 chars.
- Combined in `CaregiverReward`: curiosity as base + word recognition as gentle bias toward English.

## Key Results This Session

### What worked:
- **Centroid BPC**: Non-learned observability. Sensory plateau at 300k (cbpc 4.59).
- **L5 output weights**: Replaced frequency-counting col_token_map. Same architecture as ff_weights. No more 'a' dominance.
- **Three-factor learning**: Eligibility trace + reward on both ff_weights and L5. Breaks attractor collapse.
- **Curiosity reward (RPE)**: Discovers all 32 chars. Natural explore→exploit cycle.
- **Adaptive noise**: Self-regulating via reward EMA delta. No hardcoded schedule.
- **Caregiver positive-only prefix**: "th" gets credit (prefix of "the"). No collapse. 28 instances of "th" bigram at 100k. Real words "go", "to" emerging.

### What failed:
- **S1 prediction reward alone**: Flat reward, no differentiation between repetition and diversity.
- **S2 word stability reward**: Uniformly -0.3 (single chars carry no word info).
- **Dead-end penalty in caregiver**: Collapsed M1 to single-char repetition at 55k. Penalty overwhelmed curiosity.
- **Repetition penalty**: Worked short-term but hacky, replaced by curiosity RPE.

### Current bottleneck:
L5 random initialization biases M1 toward arbitrary chars (q, f, x). Caregiver prefix nudges toward English (28 "th" instances) but can't overcome init bias in 100k. Need either longer runs or L5 pre-training from listening phase.

## Reward Design Lessons
- Curiosity (intrinsic) = exploration engine. Essential, never remove.
- Caregiver (extrinsic) = convergence toward language. Positive-only. Gentle nudge.
- Penalties cause collapse. Only reward progress, never punish.
- Reward asymmetry matters: positive caregiver + zero for non-matches >> positive + negative.

## Training Pipeline
1. **Sensory** (300k corpus): S1→S2→S3. ✅
2. **Babbling** (100k autoregressive): Curiosity + caregiver reward. 32 chars discovered, "th"/"go" emerging. ✅
3. Stages 2/3 collapsed into one (both use caregiver + curiosity).

## Next Steps
- [ ] **L5 pre-training via listening** — run corpus through M1 with L5 learning only (observe_token). Gives L5 a map to real chars before babbling starts. Biologically: infants hear language before babbling.
- [ ] **Longer babbling (500k)** — does gentle caregiver signal accumulate enough to shift L5 toward English?
- [ ] **Inspect L5 weight structure** — which L2/3 patterns map to which tokens? Is there any char clustering?
- [ ] **REPL babbling mode** — watch M1 babble interactively
- [ ] **M2 sequencing** — once M1 reliably produces common bigrams
- [ ] **PFC design** — working memory, per-stripe BG, three-factor learning

## Key Files
- `src/step/cortex/motor.py` — L5, three-factor, `_babble_direct()`
- `src/step/cortex/reward.py` — `CuriosityReward`, `CaregiverReward`, `WordReward`
- `src/step/cortex/topology.py` — `run_babbling()`, adaptive noise, pluggable reward
- `src/step/cortex/stages.py` — stage definitions
- `src/step/probes/centroid_bpc.py` — non-learned BPC
- `experiments/scripts/cortex_staged.py` — staged runner
