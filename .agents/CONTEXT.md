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
- **`experiments/checkpoints/`** — saved model checkpoints (e.g., `personachat_100k.ckpt`)

## Architecture

### S1 → S2 → M1 + BasalGanglia
- **S1**: 128 cols, k=8, ltd=0.05, synapse_decay=1.0, PositionalCharEncoder
- **S2**: 32 cols, temporal buffer (depth=4) + burst gating, apical feedback to S1
- **M1**: 32 cols k=4, L5 readout threshold 0.3, population vote output (L5 population coding), DendriticDecoder tracked as diagnostic
- **BG**: Go/no-go gate, three-factor plasticity, supervised gate-error (Stage 1)
- **Modulators**: SurpriseTracker, ThalamicGate (receiver-side feedback gating)

### Topology features
- Persistent `_in_eom`/`_eom_steps`/`_total_steps` across `run()` calls (supports single-token stepping)
- `force_gate_open` flag for interactive use (bypasses BG gating)
- `save_checkpoint()`/`load_checkpoint()` for full model persistence

## cortex_repl — Interactive REPL

Full S1→S2→M1+BG pipeline for qualitative exploration:
- Input phase: per-char surprise, predictions, gate value, M1 interruptions
- EOM injection → speak phase with forced-open BG gate
- Autoregressive M1 generation until silence or ramble limit
- Commands: `/help`, `/reset`, `/stats`, `/warmup N`, `/save`, `/load`
- `--checkpoint name` for instant model loading
- `--dataset personachat|tinydialogues` for vocab/warmup source

## Datasets
- **PersonaChat** (`AlekseyKorshuk/persona-chat`): 17.8k human-written dialogues, ~43 unique chars. Now default for REPL.
- **TinyDialogues** (`styfeng/TinyDialogues`): GPT-generated, ~65 chars, 110k dialogues. Used for original BG/BPC work.
- DailyDialog was preferred but broken on HuggingFace (old loading script format).

## Training Results

### 100k TinyDialogues (synapse_decay=1.0)
- BPC floor: **4.38** at 50k (random baseline 6.0)
- Oscillates 4.38-4.98 — dialogue diversity, not catastrophic forgetting
- S2 context discrimination 0.91, S1 0.76

### 100k PersonaChat
- BPC floor: **4.81** at 100k (random baseline 5.43 for 43 chars)
- Higher burst rate (34% vs 25%) — more diverse patterns
- Better column selectivity (0.317 vs 0.466)
- Checkpoint saved: `experiments/checkpoints/personachat_100k.ckpt` (9.3 MB, k=1 — incompatible with current k=4)

### M1 output comparison (100k PersonaChat, k=4)
- **Population vote: 33%** accuracy at 100k, steadily improving
- **Dendritic decoder: 13%** accuracy at 100k, erratic — insufficient capacity for M1 L2/3 patterns
- Population vote is biologically grounded (L5 population coding) and empirically superior
- Old k=1 was degenerate (1 active column can't distinguish 43 chars with 32 cols)

## Key Decisions
- **synapse_decay=1.0**: No passive decay. Sparse encoding prevents catastrophic forgetting; decay was the real culprit.
- **Forced BG gate in REPL**: BG is for turn-taking (when to speak), not content (what to say). REPL controls turns.
- **PersonaChat over TinyDialogues**: Human-written > GPT-generated for natural language patterns.
- **S2 value unconfirmed**: Context discrimination 0.91 but no BPC ablation yet.

## Next Steps (Active)
- [ ] **Save new checkpoint + test REPL** — population vote output with k=4 confirmed superior, need fresh checkpoint and qualitative testing
- [ ] **Optimize S2 + apical feedback** — make top-down signals more useful for S1 prediction
- [ ] Evaluate BLiMP sensitivity with tuned config
- [ ] Stage 2 coherence (content-based reward for M1)
