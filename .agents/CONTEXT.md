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
- **`src/step/viz/`** — dashboard chart builders
- **`src/step/encoders/`** — `CharbitEncoder`, `OneHotCharEncoder`, `PositionalCharEncoder`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`
- **`experiments/scripts/bg_sweep.py`** — BG parameter sweep tool

## Architecture

### S1 (Sensory) → S2 (Secondary) → M1 (Motor) + BasalGanglia

- **S1**: Encoder → 32 cols, k=4, ltd=0.05, PositionalCharEncoder
- **S2**: S1 L2/3 → 32 cols, temporal buffer + burst gating, apical feedback to S1
- **M1**: S1 L2/3 → 32 cols k=1 (winner-take-all), L5 readout thresholded at 0.3
- **BasalGanglia**: Go/no-go gate on M1 output, learned via three-factor plasticity

### BasalGanglia — Tuned & Working
- **Context**: per-column precision state (1=predicted, 0=bursting) + precision fraction. NOT L2/3 firing rates (cosine=0.91 between EOM/input, indistinguishable).
- **Signal**: Gate-error (target_gate - actual_gate) every step. Supervised for Stage 1. Avoids sparse-reward collapse from 90/10 input/EOM imbalance.
- **Exploration**: Gaussian noise on activation (default 0.5) prevents gate collapse during early learning.
- **Key insight**: L2/3 rates don't distinguish phases. Burst rate does (0.22 EOM vs 0.40 input). Precision columns are the right BG context — models L5/6 → striatum projection.

### Modulators
- **SurpriseTracker**: scales learning rate at downstream regions
- **RewardModulator**: slow consolidation gate (NOT phasic RPE). Cortex learns Hebbian; reward only gates which traces persist.
- **ThalamicGate**: receiver-side feedback suppression until receiver stabilizes

## Motor RL Results — Stage 1 Turn-Taking

### BG Gating Performance
| Dataset | Tokens | Int% | Speak% | Unr% | M1 Acc |
|---------|--------|------|--------|------|--------|
| TinyStories 15 stories | 10k | 0.6% | **82%** | **18%** | 74% |
| TinyStories 31 stories | 20k | 0.6% | 74% | 26% | **81%** |
| TinyStories 69 stories | 50k | 0.5% | 66% | 34% | 54% |
| BabyLM synthetic | 20k | 4.0% | 71% | 29% | 73% |

### Key Findings
- Natural boundaries (TinyStories) >> synthetic (BabyLM): 0.6% vs 4% interruptions
- **Sweet spot: 10-20k TinyStories** (15-31 stories). 50k degrades — 64 chars overwhelms 32 columns.
- BG gating works immediately (71% speak even at 1k tokens)
- ~29% unresponsive floor is M1 capacity (gate opens but output_scores < threshold), not BG
- speak_window=20 gives 79% speak (vs 60% at win=5); M1 needs ramp-up time
- Reward routing validated by neuroscience: phasic RPE → BG only, smoothed dopamine → cortex consolidation gate

## Key Decisions
- **k=1 for M1**: sweep of k={1,2,4}, k=1 gives 60% accuracy, robust to other params
- **Precision context for BG**: L2/3 rates indistinguishable between phases; burst/precise columns are the discriminative signal
- **Gate-error signal (supervised Stage 1)**: pure RL collapsed to "never speak" due to 90/10 imbalance. Gate-error provides gradient from both phases. Stage 2/3 will use RL.
- **Direction 1 (RL) before Direction 2 (Chat Interface)**

## Next Steps
- [ ] Investigate 29% unresponsive floor (M1 output threshold / capacity)
- [ ] Scale M1 columns for larger vocabularies (TinyStories 64+ chars)
- [ ] Stage 2: coherent speech reward (BG exploratory bias + M1 consolidation)
- [ ] Interactive REPL for debugging turn-taking
- [ ] PFC / reasoning region (receives S2, plans multi-step)
