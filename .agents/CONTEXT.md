# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron ff_weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **M1**: 32 cols, k=4. Three-factor RL learning. Direct column forcing + noise annealing.

## Key Results This Session

### Sensory (Stage 1)
- Centroid BPC: plateaus at ~4.59 by 300k. Non-learned probe confirmed monotonic learning.

### Motor RL (Stage 3) — Reward crossed zero!
- **100k guided babbling with noise annealing (0.5→0.1)**:
  - Reward: -0.137 (early) → -0.036 (late), best window **+0.042**
  - Output: `'aaaaaa'` → `'-abcdeagaaaaamaoaaraauvwxya .a'` (28 unique chars)
  - M1 produces diverse chars from learned policy at noise=0.1 (not just random forcing)
- **Three-factor learning works**: eligibility trace + reward on ff_weights breaks attractor collapse
- **S1 prediction reward works**: burst rate + repetition penalty + diversity bonus gives gradient
- **Current bottleneck**: col_token_map is frequency-dominated. Most columns map to 'a'. Limits output diversity even when ff_weights produce diverse column patterns.

### Key Architectural Decisions
- Stages control hardware readiness (freeze/enable); BG/thalamus control software (learned within-stage)
- Autoregressive babbling (no corpus): M1→S1→M1 closed loop
- S1 prediction reward over S2 word reward: char-level signal with real gradient
- Noise annealing: linear 0.5→0.1, learned policy increasingly dominates

## Training Pipeline
1. **Sensory** (300k corpus): S1→S2→S3, all learning. ✅
2. **Babbling** (50k autoregressive): M1 direct forcing + ff_weight training. ✅
3. **Guided babbling** (100k autoregressive + RL): Three-factor, S1 prediction reward, noise annealing. ✅ Reward approaching zero.
4. **Imitation**: Needs M2. Not started.
5. **Generation**: Needs PFC. Not started.

## Active Investigation: col_token_map frequency dominance
- `_col_token_counts[col][token_id]` accumulates all-time counts
- `_col_token_map[col] = argmax(counts)` → most frequent token wins
- During babbling, 'a' is most frequent (both from corpus and attractor phase)
- Even when ff_weights produce diverse column patterns, `get_population_output()` maps most columns to 'a'
- Needs recency weighting or reset mechanism

## Checkpoints
- `stage1_sensory.ckpt` — 300k (32-char BabyLM)
- `stage2_babbling.ckpt` — 50k autoregressive
- `stage3_guided.ckpt` — 100k guided babbling

## Next Steps
- [ ] **Fix col_token_map** — recency-weighted counts or decay
- [ ] **Longer Stage 3** — 500k with fixed map, see if reward goes solidly positive
- [ ] **Inspect natural bigrams** — is M1 producing "th", "he", "an" etc?
- [ ] **REPL babbling mode** — watch M1 babble interactively
- [ ] **M2 design** — once M1 produces recognizable sequences
