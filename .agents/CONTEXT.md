# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, factory functions `make_sensory_region()`, `make_motor_region()`
- **`src/step/cortex/`** — `region.py`, `sensory.py`, `motor.py`, `modulators.py`, `basal_ganglia.py`, `topology.py`
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `timeline.py`, `bpc.py`
- **`src/step/data.py`** — token loading: `prepare_tokens_tinydialogues()`, `prepare_tokens_personachat()`, `EOM_TOKEN`, `STORY_BOUNDARY`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`
- **`experiments/scripts/`** — `cortex_run.py`, `cortex_repl.py`, `td_sweep.py`, `bg_sweep.py`
- **`experiments/checkpoints/`** — saved model checkpoints (gitignored)

## Architecture

### S1 → S2 → M1 + BasalGanglia
- **S1**: 128 cols, k=8, ltd=0.05, synapse_decay=1.0, PositionalCharEncoder
- **S2**: 32 cols k=4, temporal buffer (depth=4) + burst gating, apical feedback to S1, thalamic gating
- **M1**: 32 cols k=4, L5 readout threshold 0.3, population vote output (L5 population coding), DendriticDecoder tracked as diagnostic
- **BG**: Go/no-go gate, three-factor plasticity, supervised gate-error (Stage 1)
- **Modulators**: SurpriseTracker, ThalamicGate (receiver-side feedback gating)

### Topology features
- Persistent `_in_eom`/`_eom_steps`/`_total_steps` across `run()` calls (supports single-token stepping)
- `force_gate_open` flag for interactive use (bypasses BG gating)
- `save_checkpoint()`/`load_checkpoint()` for full model persistence
- `--checkpoint` flag on `cortex_run.py` for saving checkpoints after training
- Dataset presets (`--dataset personachat`) auto-set buffer_depth=4, burst_gate, apical, gate_feedback

## cortex_repl — Interactive REPL

Full S1→S2→M1+BG pipeline for qualitative exploration:
- Input phase: per-char surprise, predictions, gate value, M1 interruptions
- EOM injection → speak phase with forced-open BG gate
- Autoregressive M1 generation until silence or ramble limit
- Commands: `/help`, `/reset`, `/stats`, `/warmup N`, `/save`, `/load`
- `--checkpoint name` for instant model loading
- `--dataset personachat|tinydialogues` for vocab/warmup source
- Vocab sample uses 100k chars when loading checkpoint (rare chars need many dialogues)

## Training Results

### 100k PersonaChat (k=4, full architecture)
- S1 BPC: **5.27** (steady-state 5.38, random baseline 5.43)
- S1 dendritic decoder: 15% accuracy
- M1 population vote: **33%** training accuracy (steadily improving)
- M1 dendritic decoder: 13% (insufficient capacity for M1 L2/3 patterns)
- M1 representation: NON-TRIVIAL, context discrimination 0.942

### M1 generation quality (REPL testing)
- Outputs degenerate `eeeee` loops regardless of voting mechanism
- Tested: raw frequency vote (33% train), full selectivity (13%), tempered sqrt (32%) — all produce same fixed-point at generation time
- **Root cause**: autoregressive loop creates fixed point. Feeding M1 output back through S1 → S1 L2/3 stabilizes → M1 always sees same pattern → same output
- M1 lacks recurrent dynamics for sequence generation

### Background: 1M PersonaChat training in progress
- Running `personachat-k4-1m` to test if more data improves S1 representations

## Key Decisions
- **synapse_decay=1.0**: No passive decay. Sparse encoding prevents catastrophic forgetting.
- **Forced BG gate in REPL**: BG is for turn-taking (when to speak), not content. REPL controls turns.
- **PersonaChat over TinyDialogues**: Human-written > GPT-generated for natural language patterns.
- **Population vote over DendriticDecoder for M1 output**: Biologically grounded (L5 population coding) and empirically better (33% vs 13%).
- **Selectivity normalization doesn't help generation**: The bottleneck is the autoregressive loop, not the voting mechanism.

## Next Steps (Active)
- [ ] **M1 recurrent sequence generation** — make M1's L2/3 lateral connections drive sequential state evolution so M1 produces varied output from a single "go" signal (EOM), rather than echoing S1's fixed-point input. Core architectural gap.
- [ ] **Optimize S2 + apical feedback** — make top-down signals more useful for S1 prediction
- [ ] **Efference copy** — M1→S1 signal distinguishing self-generated from external input
- [ ] Stage 2 coherence (content-based reward for M1)
