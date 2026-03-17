# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron ff_weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **M1**: 32 cols, k=4. Motor output. Three-factor learning (eligibility trace + reward on ff_weights). Direct column forcing for babbling.

## Key Files
- `src/step/cortex/topology.py` — `run()`, `run_babbling()` (autoregressive), freeze/enable APIs
- `src/step/cortex/stages.py` — `TrainingStage`, SENSORY/BABBLING/GUIDED stages
- `src/step/cortex/motor.py` — `_babble_direct()`, `_learn_ff()` (three-factor), `apply_reward()`
- `src/step/cortex/reward.py` — `S1PredictionReward` (burst rate + repetition penalty + diversity bonus)
- `src/step/probes/centroid_bpc.py` — non-learned BPC probe (primary metric)
- `experiments/scripts/cortex_staged.py` — staged runner, autoregressive babbling mode
- `experiments/scripts/cortex_repl.py` — REPL with /info guardrails

## This Session's Achievements (16 commits)

### Observability
- **Centroid BPC probe**: Non-learned (EMA centroids + dot product). Confirmed model learns monotonically at 1M. Sensory plateau at ~300k (cbpc 4.59).
- **Timeline downsampling** (`--timeline-interval`): Fixes OOM on long runs.

### Training Infrastructure
- **Stage system**: `region.learning_enabled`, `connection.enabled`, `TrainingStage.configure()`.
- **Staged runner** (`cortex_staged.py`): Checkpoint chaining, auto-detects babbling vs corpus mode.
- **Autoregressive babbling loop** (`run_babbling`): M1 drives, hears itself through frozen S1. No corpus.

### Motor Learning (Stages 2-3)
- **Direct column forcing**: Random k-hot activations + ff_weight training. Breaks frequency collapse.
- **Three-factor learning on M1**: Eligibility trace records Hebbian changes; reward consolidates or reverses. Breaks attractor collapse (unique chars: 1→20/window).
- **S1 prediction reward**: S1 burst rate as reward signal (natural transitions rewarded). Plus repetition penalty and diversity bonus.
- **Result**: Reward trends upward -0.144→-0.110 over 100k. M1 genuinely learning via RL to produce diverse, S1-predicted character transitions.

### Problems Solved
- Dendritic decoder broken thermometer → centroid BPC
- Timeline OOM → downsampling
- BG missing from topology → added
- Turn-taking reward drowning word reward → pluggable, replaces entirely
- Attractor collapse (M1 repeats one char) → three-factor learning
- S2 word reward uniformly -0.3 → S1 prediction reward with variance
- Flat reward from prediction alone → repetition penalty creates gradient

## Training Stages
1. **Sensory** (S1→S2→S3): Corpus-driven, 300k. cbpc 4.59. ✅
2. **Babbling** (M1→S1→M1): Autoregressive, direct forcing, 50k. 32/32 chars. ✅
3. **Guided babbling** (M1 + S1 reward): Autoregressive + three-factor RL. Reward trending up. ✅ Learning signal confirmed.
4. **Imitation** (S1→S2→M2→M1): Needs M2. Not started.
5. **Generation** (PFC→M2→M1): Needs PFC. Not started.

## Checkpoints
- `stage1_sensory.ckpt` — 300k sensory (32-char BabyLM alphabet)
- `stage2_babbling.ckpt` — 50k autoregressive babbling
- `stage3_guided.ckpt` — 100k guided babbling with S1 prediction reward

## Next Steps
- [ ] **Longer Stage 3 / noise annealing** — reward plateaus at -0.11. Lower noise (0.5→0.2) to let learned policy dominate. May need 500k+.
- [ ] **Inspect M1 output sequences** — what bigrams/trigrams is M1 actually producing? Are they language-like?
- [ ] **REPL babbling mode** — interactive mode where you watch M1 babble and see reward in real time
- [ ] **Col_token_map improvement** — recency-weighted counts, frequency normalization
- [ ] **Noise annealing schedule** — curriculum: high noise early, anneal to low noise as reward improves
- [ ] **M2 design** — once M1 produces ~10 recognizable bigrams/trigrams, add sequencing layer
- [ ] **Apical gain tuning** — deferred, use centroid BPC to A/B
