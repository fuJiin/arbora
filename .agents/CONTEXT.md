# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`, `_default_motor_config()`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `motor.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — thin wrappers `run_cortex()`, `run_hierarchy()` delegating to `Topology`
- **`src/step/data.py`** — token loading + `EOM_TOKEN`, `inject_eom_tokens()`, `STORY_BOUNDARY`
- **`src/step/runs.py`** — run serialization: `save_run`/`load_run`/`list_runs`/`auto_name`
- **`src/step/viz/`** — dashboard chart builders (`cards.py`, `charts.py`, `layout.py`, `build_index_html`)
- **`src/step/encoders/`** — `CharbitEncoder`, `OneHotCharEncoder`, `PositionalCharEncoder`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`

## Three-Region Architecture

### S1 (Sensory)
- Encoder -> 32 cols, k=4, ltd=0.05 (char-level with PositionalCharEncoder, 256-dim)

### S2 (Secondary Sensory)
- S1's L2/3 firing rate -> 32 cols, sliding window receptive fields
- Feedforward: `firing_rate_l23` EMA signal S1->S2, with temporal buffer + burst gating
- Feedback: S2->S1 apical segments (precision-weighted + thalamic-gated)
- Surprise modulation: S1 burst rate -> SurpriseTracker -> scales S2 learning

### M1 (Motor Cortex)
- **`MotorRegion`** subclasses `SensoryRegion` (encoding_width=0), adds L5 output gating
- Receives S1's L2/3, learns context->token mapping via same Hebbian/dendritic algorithm
- **L5 readout**: per-column mean L2/3 firing rate, thresholded (default 0.3) for output
- **k=1 (winner-take-all)**: k=1 -> 60% accuracy, k=4 -> 26%. Architecture dominates tuning.
- Wiring: S1->M1 feedforward, S1->M1 surprise, M1->S1 apical (thalamic-gated)

### BasalGanglia (Go/No-Go Gating) — NEW
- **Go/no-go gate** on M1 output, learned via dopamine (three-factor: `dw = lr * reward * eligibility_trace`)
- Receives S1's L2/3 firing rates as context (same signal M1 sees)
- D1/D2 pathway model: positive reward -> open gate (go), negative reward -> close gate (no-go)
- Gate is scalar [0, 1] that multiplies M1's output_scores before speak/silent decision
- Eligibility trace with decay=0.95 tracks which context features were active
- Resets at story boundaries

### ThalamicGate
- **Receiver-side gating** on feedback connections (S2->S1, M1->S1)
- `readiness = 1.0 - smoothed_burst_rate` (EMA, decay=0.95, starts closed)
- `effective_feedback = signal * sender_confidence * receiver_readiness`

### RewardModulator
- **Dopaminergic signal** for cortical consolidation gating (NOT phasic RPE)
- Smoothed EMA of reward, modulates Hebbian learning rate: `effective_lr = base_lr * surprise_mod * reward_mod`
- Range [0, 2], baseline 1.0. Slow consolidation gate, not fast teaching signal.

## Motor RL System (Active — uncommitted on main)

### Turn-Taking (Stage 1) — IMPLEMENTED
- `EOM_TOKEN = -2` injected before story boundaries + every `segment_length` tokens
- `speak_window` pads EOM with repeated tokens so M1 has time to practice
- Reward function: speak during EOM (+0.5), silent during input (+0.2), speak during input (-0.5), silent during EOM (-0.3), rambling past max steps (-1.0)
- Turn-taking counters: interruptions, unresponsive, correct_speak, correct_silent, rambles
- BasalGanglia receives reward signal directly; cortical RewardModulator is slow consolidation gate
- CLI flags: `--reward`, `--eom`, `--eom-segment`

### First BG Run Results (5k chars, segment=100, window=10)
- BG gate oscillates 0.40-0.63, centering ~0.5 — not yet decisive
- Interruptions ~28% (stable), unresponsive ~68% (down from 100% initial)
- Learning rate 0.01 may be too conservative for 5k tokens
- S1 L2/3 may not distinguish "post-EOM" from "normal input" strongly enough

## Key Decisions
- **k=1 for M1**: robust to other params (lr, ltd, voltage_decay all give 58-61%)
- **BG for gating, not reward-modulated cortical learning**: RewardModulator was scaling Hebbian weights (wrong target). BG provides learnable go/no-go gate on the actual speak decision.
- **Reward routing (from neuroscience research)**: Phasic RPE -> BG only (fast, per-trial). Smoothed dopamine -> M1 consolidation gate (slow). Cortex learns Hebbian; reward only decides which patterns persist. Validated by songbird vocal learning: lesion Area X (BG) in juveniles -> never learn to sing.
- **Direction 1 (RL) before Direction 2 (Chat Interface)**: M1 at 60% char accuracy, chat interface premature
- **`surprise.py` renamed to `modulators.py`**: contains SurpriseTracker, RewardModulator, ThalamicGate

## Current Work (uncommitted on main)
- ~20 files modified/added: modulators rename, RewardModulator, EOM injection, turn-taking metrics, BasalGanglia, BG integration in topology
- All 186 tests passing, lint clean
- New files: `basal_ganglia.py`, `modulators.py`, `test_basal_ganglia.py`, `test_reward.py`

## Next Steps
- [ ] Commit current BG + reward work
- [ ] Tune BG learning rate (try 0.05-0.1) and/or longer runs (20k+)
- [ ] Investigate if S1 L2/3 representations differentiate EOM vs input contexts
- [ ] Stage 2: coherent speech reward (BG exploratory bias + M1 consolidation)
- [ ] Interactive REPL for debugging turn-taking
- [ ] PFC / reasoning region (receives S2, plans multi-step)
