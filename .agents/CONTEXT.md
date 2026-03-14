# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`, `_default_motor_config()`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `motor.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`
- **`src/step/data.py`** — token loading + `EOM_TOKEN`, `inject_eom_tokens()`, `prepare_tokens_tinydialogues()`, `STORY_BOUNDARY`
- **`src/step/runs.py`** — run serialization: `save_run`/`load_run`/`list_runs`/`auto_name`
- **`src/step/viz/`** — dashboard chart builders (includes BPC + BG gate charts)
- **`src/step/encoders/`** — `CharbitEncoder`, `OneHotCharEncoder`, `PositionalCharEncoder`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`
- **`experiments/scripts/`** — `bg_sweep.py`, `td_sweep.py` (TinyDialogues param sweep)

## Architecture

### S1 (Sensory) → S2 (Secondary) → M1 (Motor) + BasalGanglia

- **S1**: Encoder → configurable cols (default 32, best 64-128), k=4-8, ltd=0.05, PositionalCharEncoder
- **S2**: S1 L2/3 → 32 cols, temporal buffer + burst gating, apical feedback to S1
- **M1**: S1 L2/3 → 32 cols k=1 (winner-take-all), L5 readout thresholded at 0.3
- **BasalGanglia**: Go/no-go gate on M1 output, learned via three-factor plasticity

### BasalGanglia — Tuned & Working
- **Context**: per-column precision state (1=predicted, 0=bursting) + precision fraction. Models L5/6 → striatum projection.
- **Signal**: Gate-error (target_gate - actual_gate) every step. Supervised for Stage 1.
- **Exploration**: Gaussian noise (σ=0.5) on activation prevents gate collapse.

### Modulators
- **SurpriseTracker**: scales learning rate at downstream regions
- **RewardModulator**: slow consolidation gate (NOT phasic RPE). Cortex learns Hebbian; reward only gates which traces persist.
- **ThalamicGate**: receiver-side feedback suppression until receiver stabilizes

### BPC Metric
- `BPCProbe` in topology run loop, uses dendritic decoder overlap scores → softmax → -log₂(P)
- Tracks overall + rolling 500-char window. Shows `bpc=X.XX` in log lines.
- Random baseline: log₂(V) ≈ 6.0 for 65 chars. Good: <5.0.

## Stage 1 Turn-Taking — COMPLETE

Validated on TinyDialogues with real speaker-alternation turns:
- 0.9-1.1% interruptions, 69-90% speak rate, 52-57% M1 acc (S1=128col k=8)
- BG gating transfers cleanly from synthetic to real turn boundaries

## TinyDialogues Param Sweep — Key Results

Sweep explored S1 cols (32/48/64/128), k (2/4/6/8/16), learning rates, M1 cols on 10-50k chars.

**Optimal activation fraction ≈ 5-6%:**
- 64col k=4 (6.25%): BPC 4.72, best at 30k chars
- 128col k=8 (6.25%): BPC 4.89, den=21.8%, M1=57.0%, best for scale

**Capacity scales with data:**
- 10k chars: S1=64 wins (BPC 4.88 vs 5.22 baseline)
- 30k chars: S1=64 degrades to 4.72, S1=128 k=8 better at 4.89
- 50k chars: S1=128 k=4 gets BPC 5.14 — k too sparse, k=8 needed

**M1 expansion alone hurts** — bottleneck is S1 representation quality, not M1 capacity.

## Key Decisions
- **k=1 for M1**: robust across params, architecture > tuning
- **Precision context for BG**: burst/precise columns are the discriminative signal
- **Gate-error signal (supervised Stage 1)**: Stage 2/3 will use RL
- **Coherence training deferred**: current char-level model wouldn't produce useful chatbot
- **Switch to TinyDialogues** (`styfeng/TinyDialogues`): real speaker-alternation turns. 110k train dialogues, ~65 unique chars, ~15.8 turns/dialogue.
- **BPC metric**: formalizes prediction quality. Trend shows learning.
- **Activation fraction ~6%**: 64col/k=4 or 128col/k=8 both optimal

## Next Steps (Active)
- [ ] Update default S1 config to 128col k=8 (or 64col k=4 for speed)
- [ ] Longer training runs (50-100k) to confirm BPC trend improves
- [ ] Evaluate BLiMP sensitivity with tuned config
- [ ] Freeze canonical training regime based on BPC convergence
- [ ] Then: Stage 2 coherence (BG exploratory bias + M1 consolidation)
