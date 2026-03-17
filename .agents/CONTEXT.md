# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron ff_weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **M1**: 32 cols, k=4. L5 output layer (learned weights, sparse, three-factor). Direct column forcing for babbling.

## Key Achievements This Session

### Centroid BPC (non-learned observability)
- Replaced broken dendritic decoder. Confirmed sensory learning plateaus at ~300k (cbpc 4.59).

### L5 Output Layer
- Replaced frequency-counting col_token_map with learned L5 weights (L2/3 → token scores)
- Structural sparsity, three-factor Hebbian (eligibility trace + reward consolidation)
- Vocabulary-constrained: L5 only outputs valid BabyLM chars
- Same architecture as ff_weights — biologically grounded L5 pyramidal projection

### Curiosity Reward (Dopamine RPE)
- Tracks expected S1 burst per bigram. Reward = expected - actual (prediction improvement)
- Known bigrams stop being rewarding → naturally drives exploration of new ones
- Result: **32/32 chars discovered** in 100k steps. No explicit diversity bonus needed.
- Explore→exploit cycle emerges: M1 expands to full vocab, then contracts to best bigrams
- Output evolves: `'itttiti'` → `"osko-forjxk"` → `"alfs.afaslsl"` → `'ii.riiriii.rr'`

### Adaptive Noise (Tonic Dopamine Model)
- Self-regulating: fast vs slow reward EMA drives noise up/down
- Stagnation detection: small |delta| → gentle noise increase
- Replaces hardcoded linear schedule

### Training Pipeline
1. **Sensory** (300k corpus): S1→S2→S3 representation learning. ✅
2. **Babbling** (autoregressive + curiosity): M1→S1→M1 with RPE reward. Full vocab discovered. ✅
3. Stages 2 and 3 collapsed — both use curiosity reward, L5 weights make them functionally identical.
4. **Imitation**: Needs M2. Not started.
5. **Generation**: Needs PFC. Not started.

## Key Files
- `src/step/cortex/motor.py` — L5 output weights, three-factor learning, `_babble_direct()`
- `src/step/cortex/reward.py` — `CuriosityReward` (RPE), `WordReward` (S2-based, backup)
- `src/step/cortex/topology.py` — `run_babbling()`, adaptive noise, pluggable reward
- `src/step/cortex/stages.py` — `TrainingStage`, SENSORY/BABBLING/GUIDED definitions
- `src/step/probes/centroid_bpc.py` — non-learned BPC probe

## Checkpoints
- `stage1_sensory.ckpt` — 300k sensory (32-char BabyLM)
- `stage2_babbling.ckpt` — 100k curiosity-driven babbling

## Next Steps
- [ ] **Longer babbling run (500k+)** — does M1 discover common English bigrams (th, he, in)?
- [ ] **Inspect bigram frequencies** — what transitions is M1 actually learning?
- [ ] **REPL babbling mode** — watch M1 babble interactively
- [ ] **M2 design** — once M1 produces recognizable sequences, add sequencing layer
- [ ] **PFC design** — working memory, per-stripe BG gating, three-factor learning
- [ ] **Apical gain tuning** — deferred, use centroid BPC to A/B
