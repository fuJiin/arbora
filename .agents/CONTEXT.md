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
- **Efference copy**: When M1 speaks during `force_gate_open`, S1 subtracts `efference_gain * predicted_drive` from actual drive. Default gain=1.0.

### Topology features
- `force_gate_open` flag for interactive use (bypasses BG gating + triggers efference copy)
- `save_checkpoint()`/`load_checkpoint()` for full model persistence
- Dataset presets (`--dataset personachat`) auto-set buffer_depth=4, burst_gate, apical, gate_feedback

## Training Results

### 100k PersonaChat (k=4, full architecture)
- S1 BPC: **5.27** (steady-state 5.38, random baseline 5.43)
- M1 population vote: **33%** training accuracy
- M1 representation: NON-TRIVIAL, context discrimination 0.942

### M1 token map collapse (diagnosed 2026-03-16)
- Only **7 unique tokens** across 30 assigned columns: 12×space, 5×`?`, 4×`e`, 3×`.`, 3×`o`, 2×`t`, 1×`i`
- Column→token mapping dominated by character frequency — no mechanism pushes columns to specialize for rare chars
- Output scores during generation all near zero (max 0.003 vs 0.3 threshold) — M1 barely fires
- **Verdict: "bad" not "fixable"** — collapsed mappings, not diverse-mappings-with-wrong-activation

### M1 babbling plan (designed, tabled)
- Modeled on infant vocal development: random motor exploration → sensorimotor mapping → guided imitation
- **Phase 1**: Pure motor babbling — random/noisy M1 drive, no S1 loop, builds diverse column→token map
- **Phase 2**: Sensorimotor mapping — M1→S1→M1 closed loop, learns what own output looks like through S1 (addresses exposure bias)
- **Phase 3**: Guided babbling — interleaved with corpus reading (current training)
- **Tabled**: S2/S3 representations are the blocker — M1 needs higher-level intent signals to be useful after babbling

## Key Architectural Findings

### S2 interpretability: blind spot
- Same aggregate metrics as S1 (selectivity, context discrimination) but no word-level analysis
- S2 gets 4-frame temporal buffer of S1's burst-gated activity — could capture word patterns, but no probe to verify
- **Need**: Word-level selectivity probe, S2→S1 column pairing analysis, S2 prediction decoder

### Feedback gating: passive, not adaptive
- ThalamicGate (receiver burst rate) + precision weighting (sender confidence) — both passive/reactive
- Apical feedback disabled by default because S2 was "precise but wrong"
- **Defer** until S2 interpretability shows S2 is learning something useful

### Performance
- ~250-500 tokens/sec, 100k chars in ~10-15 min with full hierarchy
- Bottleneck: Python loops in segment learning (~6K-12K micro-ops/step)
- **Defer** until 1M+ training becomes routine need

## Key Decisions
- **synapse_decay=1.0**: No passive decay. Sparse encoding prevents catastrophic forgetting.
- **PersonaChat over TinyDialogues**: Human-written > GPT-generated for natural language patterns.
- **Efference copy over neural adaptation**: Architecturally consistent, no special-case behavior.
- **ff machinery in CorticalRegion**: All regions receive input via ff_weights.
- **S2/S3 before M1 babbling**: Higher-level representations are the blocker for M1 usefulness. Get S2 working first.

## Next Steps (Priority Order)
- [ ] **S2 interpretability probe** — word-level selectivity analysis to determine if S2 is learning multi-char patterns. Build probes before changing architecture.
- [ ] **S2 architecture tuning** — based on probe results, tune S2 to extract word-level context (may need more columns, different buffer depth, etc.)
- [ ] **S3 / concept region** — higher-level abstractions that would eventually drive M1 via intent signals
- [ ] **M1 babbling phases** — implement staged motor exploration once S2/S3 provide useful representations
- [ ] **Feedback gating** — adaptive S2→S1 gating once S2 is demonstrably useful
- [ ] **Performance** — Numba on segment learning when 1M+ training needed
