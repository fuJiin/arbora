# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated for multi-source):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (word+topic → goal)
  S2+PFC→M2 (word+goal → sequence)
  M2→M1 (sequence → execution)

Apical:
  Sensory top-down: S3→S2, S2→S1
  Motor monitoring: M1→M2, M2→PFC (corollary discharge)
  Cross: S1→M1 (sensory context), M1→S1 (efference copy)

Surprise: S1→S2, S2→S3, S1→M1
```

Region types (all inherit CorticalRegion):
- SensoryRegion (S1/S2/S3): local connectivity
- PFCRegion (PFC): slow decay 0.97, global gate
- PremotorRegion (M2): temporal sequencing via lateral segments
- MotorRegion (M1): L5 output, three-factor RL, babbling

## Key Results

### Babbling
- English words ("the", "mom", "ask") from interleaved listen+babble
- Curiosity RPE + caregiver reward (optionality-scaled, habituating)

### Echo (PFC→M2→M1)
- **Best: 12.8% avg match** (batch + clip + symmetric + curriculum)
- **Previous baseline: 3.1%** (per-step reward, oscillating)
- Sweep tested 17 configurations — see `experiments/results/echo_sweep.json`

### Echo Oscillation Fix (this session)
**Root cause**: 10:1 reward asymmetry (match +2.0, mismatch -0.2) + per-step reward application + unbounded eligibility traces. Any accidental match massively strengthens goal weights; mismatches can't undo it fast enough → attractor lock → slow decay → cycle repeats.

**Fixes applied (now defaults):**
- **Batch reward**: accumulate over episode, apply once (prevents 5x per-step weight push)
- **Eligibility clip** (0.05): caps trace magnitude, prevents unbounded weight deltas
- **Symmetric mismatch** (0.5x): mismatch penalty = 50% of match bonus (was 10%)
- **Curriculum**: start with 2-char words, advance when avg match > 25% over 50 episodes

**What didn't work:**
- Reward baseline subtraction: too aggressive, killed learning entirely
- All fixes combined: over-constrained, worse than subsets
- Batch + symmetric alone: 1.3% (worse than either individually — unclear interaction)

### Architecture Insights
- **Apical = bias/mode, feedforward = content/command.** Echo via apical: 4.2%. Echo via ff: 13% peak.
- **Multiple ff to same target**: concatenated (biologically correct convergent input)
- **Motor monitoring apical** mirrors sensory feedback (M1→M2→PFC ↔ S3→S2→S1)
- **S2 ff_weight pollution is not an issue**: downstream targets have separate weights in their own regions. Upstream reward modulation on S2 is the concern, but not blocking.

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules (needs workflow OAuth scope to push)

## Next Steps (Priority Order)
- [ ] **Run longer echo training** (5k-10k episodes) with new defaults to verify convergence
- [ ] **Curriculum advancement**: current threshold (25% over 50 episodes) may be too high — curriculum never advanced past maxlen=2 in 2k episodes
- [ ] **Dialogue training** with stable echo as foundation
- [ ] **Numba for L2/3 segments** — 10-20% speedup, defer until training objective stabilizes
- [ ] **Push CI workflow change** (requires workflow scope)
