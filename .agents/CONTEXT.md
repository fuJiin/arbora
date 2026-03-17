# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn architecture (L4/L2/3), dendritic segments, per-neuron ff_weights, apical gain, L5 output layer. NumPy + Numba, Python 3.12+, uv.

## Architecture
```
CorticalRegion (region.py) — global connectivity
  ├── SensoryRegion (sensory.py) — local/topographic → S1, S2, S3
  ├── MotorRegion (motor.py) — global + L5 output → M1
  └── PFCRegion (next) — global + slow decay + BG gating → PFC
```
- **S1**: 128 cols, k=8. **S2**: 32 cols, k=4, buf=4. **S3**: 32 cols, k=4, buf=8.
- **M1**: 32 cols, k=4. L5 output (learned, sparse, three-factor). Babbling via direct column forcing.
- **Note**: S2/S3 use SensoryRegion's local segment connectivity but their inputs have no spatial structure (encoding_width=0). May benefit from CorticalRegion's global connectivity — experiment for later.

## Reward Architecture
- **Curiosity (RPE)**: per-bigram expected vs actual S1 burst. Habituates per-bigram.
- **Caregiver**: live prefix tracking with optionality-scaled reward (more possible completions → more reward). Massive word completion bonus (scales with word length). Habituation on repeated words (0.7^count decay).
- **Anti-attractor stack**: curiosity RPE + caregiver habituation + adaptive noise + three-factor negative reward. All biologically grounded.

## Best Results: Interleaved Listen+Babble
100k babble (+ 400k listen): Space #1 char. 25 English bigrams. 6 real 3-letter words: the, ask, has, not, him, mom. "hi":18, "it":7, "is":2.

## Recent Engineering
- **Code quality**: Extracted shared run loop methods (_propagate_feedforward, _propagate_signals, _step_motor_reward). Fixed 5 bugs (learning gate, eligibility resets, checkpoint save/load). dataclasses.replace() for stage overrides.
- **Perf**: Pre-allocated BG context buffer, source pools, centroid BPC matrix.
- **Inheritance**: MotorRegion now inherits CorticalRegion directly (not SensoryRegion). Clean hierarchy for PFC.

## PFC Design (from research, ready to implement)

### Biological basis
- PFC is agranular cortex (thin/absent L4) — receives processed cortical input, not raw sensory
- Same minicolumn architecture as other regions, different parameters:
  - Stronger L2/3 lateral connections (sustain patterns = working memory)
  - Slower voltage decay (~0.97 vs 0.5)
  - Multi-source input (S2 + S3 concatenated)
  - Dense dopamine modulation (three-factor learning more prominent)
- BG per-stripe gating: Go/NoGo controls when PFC updates vs maintains

### Developmental sequence (infant PFC)
1. Novelty preference (~2-3mo) — already have via curiosity RPE
2. Turn-taking (~2-3mo) — alternating listen/babble structure
3. Goal maintenance (~8-12mo) — hold intention across multi-char output
4. Echolalia (~8-14mo) — reproduce heard words (first PFC-gated mode)
5. Context-dependent response (~12-18mo) — "what's that?" → produce label
6. Metacognition (~20mo+) — "I don't know" when uncertain

### Minimal PFC implementation
- CorticalRegion with voltage_decay=0.97, 16 cols, k=4
- Input: S2.l23 + S3.l23 concatenated
- BG per-stripe gating (Go/NoGo per group of columns)
- Apical output to M1 (biases token production)
- First task: echo mode vs babble mode (simplest goal maintenance)

### Wang et al. meta-RL insight
Slow three-factor learning on PFC recurrent weights (outer loop) → PFC dynamics spontaneously learn fast adaptation (inner loop). Don't engineer fast learning — it emerges.

## Training Pipeline
1. **Sensory + M1 listening** (300k corpus) ✅
2. **Interleaved babble** (100k+ babble) — English words emerge ✅
3. **PFC echo/babble mode** — next
4. **Dialogue training** — listen → PFC maintains → M1 responds
5. **M2 sequencing** — when word length is the bottleneck

## Runs In Progress
- 500k interleaved (old reward) — ~150k/500k done
- 100k interleaved (new reward with habituation) — ~63k/100k done

## Key Files
- `src/step/cortex/region.py` — CorticalRegion base (all core logic)
- `src/step/cortex/sensory.py` — SensoryRegion (local connectivity)
- `src/step/cortex/motor.py` — MotorRegion (L5, babbling, three-factor)
- `src/step/cortex/reward.py` — CuriosityReward, CaregiverReward
- `src/step/cortex/topology.py` — run(), run_babbling(), run_interleaved()
- `src/step/cortex/stages.py` — TrainingStage definitions

## Next Steps
- [ ] **Implement PFCRegion** — CorticalRegion subclass with slow decay, BG input gating
- [ ] **Echo mode training** — PFC maintains "echo" goal, M1 reproduces input
- [ ] **Dialogue structure** — listen → PFC → respond training loop
- [ ] **Analyze 500k/100k run results** when they finish
- [ ] **S2/S3 global connectivity experiment** — do they benefit from dropping local segments?
