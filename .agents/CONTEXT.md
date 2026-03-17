# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron feedforward weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **M1**: 32 cols, k=4. Motor output with direct column forcing for babbling.
- **Apical**: Per-neuron gain (BAC firing). Near no-op at 1M (decay tuning deferred).

## Key Files
- `src/step/cortex/topology.py` — `run()` (corpus), `run_babbling()` (autoregressive), freeze/enable APIs
- `src/step/cortex/stages.py` — `TrainingStage`, predefined SENSORY/BABBLING/GUIDED stages
- `src/step/cortex/motor.py` — `_babble_direct()` (random k-hot columns + ff_weight training)
- `src/step/cortex/reward.py` — `WordReward` (continuous S2 stability + word boundary bonus)
- `src/step/probes/centroid_bpc.py` — non-learned BPC probe (primary metric)
- `experiments/scripts/cortex_staged.py` — staged runner with checkpoint chaining
- `experiments/scripts/cortex_repl.py` — REPL with /info guardrails, BabyLM default

## Session Results

### Centroid BPC (replaces dendritic decoder)
- 1M sensory: cbpc 4.79→4.59→4.79. Plateaus at ~300k. Random baseline ~5.0.
- Dendritic decoder was broken thermometer. Representations were learning all along.

### Autoregressive Babbling Pipeline (Stages 2+3)
- **Stage 2 works**: M1 babbles random chars, hears itself through frozen S1. All 32 chars discovered. ff_weights trained during forced column activation.
- **Stage 3 problem — attractor collapse**: M1 gets stuck producing same char repeatedly. Once col_token_map assigns 'e' to dominant columns, M1→S1→M1 loop creates stable attractor. Reward is consistently -0.300 (S2 sees incoherent repetition). BG receives negative signal but **penalty doesn't break the attractor** — current investigation.

### Bugs Found and Fixed This Session
- Timeline OOM: per-step frames accumulated ~13GB at 1M. Fixed with `--timeline-interval`.
- Dendritic decoder staleness: no forgetting mechanism. Added `perm_decay` (optional).
- BG missing from staged topology: reward computed but nowhere to send it.
- Turn-taking reward drowning word reward: pluggable source now replaces entirely.
- `--tokens` override lost babbling_noise/force_motor_active/reward_source fields.
- M1 produced 0 chars: col_token_map empty at cold start. Added bootstrap fallback.

## Training Stages (implemented)
1. **Sensory** (S1→S2→S3): Corpus-driven, 300k sufficient (plateau). ✅
2. **Babbling** (M1→S1→M1): Autoregressive, direct column forcing. 32/32 chars discovered. ✅
3. **Guided babbling** (M1+BG+S2): Autoregressive + word reward. Attractor collapse blocks learning. **Active investigation.**
4. **Imitation** (S1→S2→M2→M1): Needs M2 region. Not started.
5. **Generation** (PFC→M2→M1): Needs PFC region. Not started.

## Active Problem: Why doesn't negative reward break M1 attractor?
- M1 repeats same char → reward is -0.300 every step → BG gets consistent negative signal
- But BG gate stays 1.0 (force_gate_open=True) so negative reward can't close the gate
- Even if gate could close, M1 would just stop producing — not diversify
- The reward signal needs to change M1's *column activations*, not just the gate
- Possible fixes: anneal noise, exploration bonus for novel patterns, or reward needs to modulate M1's ff_weights/segments directly (three-factor learning)

## Checkpoints
- `stage1_sensory.ckpt` — 300k sensory (32-char alphabet, consistent dims)
- `stage2_babbling.ckpt` — 50k autoregressive babbling
- `stage3_guided.ckpt` — 50k guided babbling (attractor-collapsed)
- `babylm_s3_1m_v2.ckpt` — 1M sensory (old alphabet, incompatible with staged runner)

## Next Steps
- [ ] **Debug reward→M1 learning pathway** — why doesn't -0.300 change M1's behavior?
- [ ] **Exploration pressure** — noise annealing, novelty bonus, or diversity reward
- [ ] **Three-factor learning on M1 ff_weights** — reward should modulate weight updates
- [ ] **Col_token_map normalization** — frequency-dominated, needs recency weighting
- [ ] **Apical gain tuning** — deferred until motor pipeline works
