# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop.

## Architecture
```
CorticalRegion (L4/L2/3, segments, apical, Hebbian)
  ├── SensoryRegion → S1 (128c/k8), S2 (32c/k4), S3 (32c/k4)
  ├── MotorRegion (L5, three-factor, goal drive) → M1 (32c/k4)
  └── PFCRegion (slow decay 0.97, global gate) → PFC (16c/k4)

Feedforward: S1→S2→S3, S2→PFC, S1→M1, PFC→M1 (goal drive)
Apical: S3→S2→S1, PFC→M1 (modulatory bias)
```

## Key Architectural Discovery: Apical vs Feedforward

**Apical (gain modulation)**: Biases which neurons are excitable. Good for mode selection, attention. Cannot select specific outputs — M1 collapses to 'e' regardless.

**Feedforward (additive drive)**: Directly drives column competition. Can select specific outputs. PFC→M1 goal drive works for echo (7.6% match, trending 6%→10%).

**Implication**: PFC→M1 apical = mode bias. PFC→M1 ff (goal_weights) = content command. Both coexist. M2 will eventually replace the ff path for longer sequences.

## Echo Mode (working, improving)
- Listen: word → S1→S2→PFC (gate open)
- PFC snapshots goal, closes gate
- Speak: PFC goal_drive → M1 (feedforward), reward for char matches
- Three-factor learning on goal_weights: PFC activity × M1 winners × reward
- Reward→PFC replay: modulates PFC learning rate, replays heard word
- **Result**: 7.6% match at 2k episodes (2.5x chance), trending up 6%→10%
- "you"→"yoy", "huh"→"uuu", "the"→" h " (partial matches emerging)

## Motor Babbling (completed)
- Interleaved listen+babble, curiosity + caregiver reward
- Produces English words: "the", "mom", "ask", "him"
- 500k run was in progress

## Reward Stack
- Curiosity (RPE): per-bigram, habituating
- Caregiver: optionality-scaled prefix + word completion bonus + habituation
- Echo: position-tolerant char matching + curiosity base

## Engineering
- Shared run loop methods (DRY), 5 bug fixes, perf pre-allocation
- MotorRegion inherits CorticalRegion (not SensoryRegion)
- Code/perf audits completed, README updated, REPL with /babble /probe

## Next Steps
- [ ] **Longer echo training (10k+ episodes)** with full 300k sensory pre-training
- [ ] **Analyze whether echo match rate keeps climbing** — ceiling indicates when M2 is needed
- [ ] **M2 design** — PFC→M2 (ff goal→plan), S2→M2 (ff context), M2→M1 (ff sequence)
- [ ] **Dialogue training** — structured listen→respond with PFC mode gating
- [ ] **500k babble analysis** when complete
