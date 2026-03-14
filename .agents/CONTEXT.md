# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`, `_default_motor_config()`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `motor.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — thin wrappers delegating to `Topology`
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
- **Context**: per-column precision state (1=predicted, 0=bursting) + precision fraction. NOT L2/3 firing rates (cosine=0.91, indistinguishable between phases). Models L5/6 → striatum projection.
- **Signal**: Gate-error (target_gate - actual_gate) every step. Supervised for Stage 1. Pure RL collapsed to "never speak" due to 90/10 imbalance.
- **Exploration**: Gaussian noise (σ=0.5) on activation prevents gate collapse.

### Modulators
- **SurpriseTracker**: scales learning rate at downstream regions
- **RewardModulator**: slow consolidation gate (NOT phasic RPE). Cortex learns Hebbian; reward only gates which traces persist.
- **ThalamicGate**: receiver-side feedback suppression until receiver stabilizes

## Stage 1 Turn-Taking — COMPLETE

Best results: TinyStories 10-20k, natural boundaries, 0.6% interruptions, 74-82% speak, 74-81% M1 acc. BabyLM with synthetic boundaries: 4% interruptions, 71% speak, 73% M1 acc. ~29% unresponsive floor is M1 capacity (gate opens but output < threshold).

## Key Decisions
- **k=1 for M1**: robust across params, architecture > tuning
- **Precision context for BG**: burst/precise columns are the discriminative signal
- **Gate-error signal (supervised Stage 1)**: Stage 2/3 will use RL
- **Coherence training deferred**: current char-level model on toy data wouldn't produce useful chatbot. Need stronger foundation first.
- **Switch to TinyDialogues** (`styfeng/TinyDialogues`): real speaker-alternation turns instead of synthetic segment_length. Age-2 subset for simple vocab. EMNLP 2024 paper found conversational data essential for grammar learning.
- **Add BPC metric**: bits-per-character formalizes prediction quality into a single comparable number. Trend over training shows whether representations are learning structure.
- **Canonical training regime**: freeze after TinyDialogues + BPC validation cycle. If BPC improves and BLiMP shows syntactic sensitivity → proceed to Stage 2. If BPC plateaus → bottleneck is S1/S2 capacity, fix that first.

## Evaluation Roadmap
1. Current metrics (M1 acc, turn-taking, representation quality) — keep
2. **BPC on held-out data** — standard char-level metric, shows learning trend
3. **BLiMP via cumulative BPC** — minimal-pair grammaticality, connects to BabyLM community
4. **AoA correlation** — compare learning order against MacArthur-Bates CDI norms
5. BLiMP Supplement turn-taking task — directly validates BG gating

## Next Steps (Active)
- [ ] Add TinyDialogues dataset loader (speaker labels → turn boundaries)
- [ ] Implement BPC metric in Topology/probes
- [ ] Validate BG gating with real speaker-alternation turns
- [ ] Evaluate: does BPC improve with training? BLiMP sensitivity?
- [ ] Freeze canonical training regime based on results
- [ ] Then: Stage 2 coherence (BG exploratory bias + M1 consolidation)
