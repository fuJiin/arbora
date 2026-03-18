# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop.

## Architecture
```
CorticalRegion (L4/L2/3, segments, apical, Hebbian)
  ├── SensoryRegion → S1 (128c/k8), S2 (32c/k4), S3 (32c/k4)
  ├── MotorRegion (L5, three-factor, goal drive) → M1 (32c/k4)
  ├── PFCRegion (slow decay 0.97, global gate) → PFC (16c/k4)
  └── M2Region (next) — temporal sequencing via lateral segments
```

## Key Architectural Insight: Apical vs Feedforward
- **Apical** = bias/mode (gain modulation). Good for attention, context.
- **Feedforward** = content/command (additive drive). Required for specific outputs.
- PFC→M1 apical: 4.2% echo (failed). PFC→M1 ff: 9% peak then destabilized.
- PFC→M1 direct hits ceiling at ~9%. **M2 is needed as intermediary.**

## Results Summary
- **Babbling**: English words ("the", "mom", "ask") from interleaved listen+babble
- **Echo**: 9% peak match (PFC→M1 ff + three-factor), degrades over 20k episodes
- **Dialogue**: 3.8% match at 5k turns. M1 captures prominent chars but can't reproduce words.

## Why M2 Is Needed Now
PFC holds a static goal pattern. M1 needs step-by-step character commands. The dimensional gap (abstract goal → specific char) is too large for one weight matrix. M2 decomposes the goal into a temporal sequence using lateral segments (same mechanism S2 uses for word patterns, but generating instead of predicting).

## M2 Design
```
Inputs:  PFC (ff, static goal) + S2 (ff, word context) + own L2/3 (lateral segments, temporal state)
Process: PFC biases which sequence unfolds. Segments drive temporal ordering.
Output:  M1 (ff, one step at a time)
```
- Same CorticalRegion architecture, different wiring
- Lateral segments learn character sequences (like S2 learns word patterns)
- PFC→M2 feedforward (goal), M2→M1 feedforward (sequence step)
- Three-factor learning on all weights

## Training Pipeline
1. ✅ **Sensory + M1/PFC listening** (300k)
2. ✅ **Interleaved babble** (English words emerge)
3. ✅ **Echo mode** (PFC→M1 direct, ~9% ceiling)
4. ✅ **Dialogue training** (structured turns, 3.8% match)
5. **→ M2 implementation** (temporal sequence generation)
6. **M2 echo training** (PFC→M2→M1 pathway)

## Runs
- 500k babble: ~412k/500k, still running

## Key Files
- `src/step/cortex/pfc.py`, `motor.py`, `region.py`, `sensory.py`
- `src/step/cortex/topology.py` — run(), run_babbling(), run_interleaved(), run_echo(), run_dialogue()
- `src/step/cortex/reward.py` — CuriosityReward, CaregiverReward, EchoReward
