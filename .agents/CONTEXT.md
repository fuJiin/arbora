# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, factory functions `make_sensory_region()`, `make_motor_region()`
- **`src/step/cortex/`** — `region.py` (base with ff weights), `sensory.py` (encoding masking + local connectivity), `motor.py` (L5 readout), `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`
- **`src/step/data.py`** — token loading: `prepare_tokens_tinydialogues()`, `prepare_tokens_personachat()`, `EOM_TOKEN`, `STORY_BOUNDARY`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`
- **`experiments/scripts/`** — `cortex_run.py`, `cortex_repl.py`, `td_sweep.py`, `bg_sweep.py`, `ef_sweep.py`
- **`experiments/checkpoints/`** — saved model checkpoints (gitignored)

## Architecture

### Class hierarchy
- **CorticalRegion** (`region.py`): ff_weights, process(), _learn_ff(), reconstruct(), efference copy (with `efference_gain`), dendritic segments, L4/L2/3 pipeline
- **SensoryRegion** (`sensory.py`): encoding-width receptive field masking, local connectivity for segments + L2/3 lateral mask
- **MotorRegion** (`motor.py`): L5 readout, population vote, column→token mapping. Inherits SensoryRegion (for local connectivity)

### S1 → S2 → M1 + BasalGanglia
- **S1**: 128 cols, k=8, ltd=0.05, synapse_decay=1.0, PositionalCharEncoder
- **S2**: 32 cols k=4, temporal buffer (depth=4) + burst gating, apical feedback to S1, thalamic gating
- **M1**: 32 cols k=4, L5 readout threshold 0.3, population vote output (L5 population coding)
- **BG**: Go/no-go gate, three-factor plasticity, supervised gate-error (Stage 1)
- **Modulators**: SurpriseTracker, ThalamicGate (receiver-side feedback gating)
- **Efference copy**: When M1 speaks during `force_gate_open`, its output encoding is set on S1 via `set_efference_copy()`. S1's next `process()` subtracts `efference_gain * predicted_drive` from actual drive. Default gain=1.0.

### Topology features
- Persistent `_in_eom`/`_eom_steps`/`_total_steps` across `run()` calls (supports single-token stepping)
- `force_gate_open` flag for interactive use (bypasses BG gating + triggers efference copy)
- `save_checkpoint()`/`load_checkpoint()` for full model persistence
- Dataset presets (`--dataset personachat`) auto-set buffer_depth=4, burst_gate, apical, gate_feedback

## Training Results

### 100k PersonaChat (k=4, full architecture)
- S1 BPC: **5.27** (steady-state 5.38, random baseline 5.43)
- M1 population vote: **33%** training accuracy (steadily improving)
- M1 representation: NON-TRIVIAL, context discrimination 0.942

### Efference copy gain sweep (`ef_sweep.py`)
- **gain=0.0** (no efference): mostly silence/spaces, occasional `e` — fixed point
- **gain=1.0** (full cancellation): 3-4 unique chars, `e`/`?`/`t`/`o`/spaces. Fixed point broken but output not coherent
- **gain=2.0**: 5 unique chars, `i` appears. Most diverse but still `e`-dominated
- **Root cause**: efference copy breaks the **feedforward** fixed point, but S1's other drive sources (voltage decay, lateral weights, excitability) still bias toward same columns. Feedforward is necessary but not sufficient.

## Key Decisions
- **synapse_decay=1.0**: No passive decay. Sparse encoding prevents catastrophic forgetting.
- **Forced BG gate in REPL**: BG is for turn-taking (when to speak), not content. REPL controls turns.
- **PersonaChat over TinyDialogues**: Human-written > GPT-generated for natural language patterns.
- **Population vote over DendriticDecoder for M1 output**: Biologically grounded (L5 population coding) and empirically better (33% vs 13%).
- **Efference copy over neural adaptation**: Architecturally consistent — addresses root cause (closed sensorimotor loop) rather than symptom (L2/3 collapse). No special-case behavior in any region.
- **ff machinery in CorticalRegion**: All regions receive input via ff_weights. SensoryRegion just adds encoding-specific masking and local connectivity.
- **efference_gain=1.0 default**: Full cancellation is biologically correct. Higher gains add marginal diversity.

## Next Steps (Active)
- [ ] **Address remaining S1 drive sources during generation** — lateral weights + excitability still push toward `e` even after ff suppression. May need to suppress S1 learning during generation, or reset lateral/excitability state
- [ ] **M1 recurrent sequence generation** — L2/3 lateral connections need to drive sequential state evolution so M1 produces varied output from its own dynamics, not just residual S1 noise
- [ ] **Optimize S2 + apical feedback** — make top-down signals more useful for S1 prediction
- [ ] Stage 2 coherence (content-based reward for M1)
