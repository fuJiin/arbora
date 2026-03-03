# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Dual model implementations**:
  - `model.py`: Functional core with explicit state threading via `ModelState` NamedTuple (in-memory, numpy arrays)
  - `db.py`: SQLite-backed `StepModel` class with lazy weight decay, SQL aggregation for predict/learn, persistence
- **Model protocol** (`protocol.py`): `Model` protocol for running STEP and baselines through the same harness
- **Key modules**: `sdr.py` (encoding), `metrics.py` (IoU, rolling mean), `data.py` (tokenizer + dataset streaming, yields `(t, token_id, sdr)` tuples)
- **Experiment infra**: `experiment.py` (seed-controlled runner, JSON logging), `figures.py` (matplotlib plots)
- **Scripts**: `experiments/scripts/run_step.py`, `run_sweep.py` (JSON output only), `visualize.py` (reads JSON, produces figures)
- **Configs**: `EncoderConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`
- **Naming**: experiments use `exp{N}_{name}` convention (e.g., `exp0_tinystories`)

## Current Work
- Completed major refactoring:
  - Renamed `update()` -> `learn()` across all modules
  - Merged `normalize.py` into `model.py` as `_local_normalize()`
  - Moved experiment scripts to `experiments/scripts/`, separated visualization
  - Added experiment versioning (`expN` naming, `seedN.json` output files)
  - Updated `data.py` to yield `(t, token_id, sdr)` 3-tuples
  - Implemented SQLite-backed `StepModel` in `db.py` with:
    - Lazy weight decay via `POWER(decay, dt)` at read time
    - SQL aggregation for predict (synapse voting) and learn (reinforce + penalize)
    - Inverted index decode (SDR -> token_id)
    - Persistence, metrics logging, checkpoint/resume
  - Created `Model` protocol in `protocol.py`
  - Added `*.db` to `.gitignore`
- 60 tests passing (20 new SQLite model tests)

## Key Decisions
- `learn()` renamed from `update()` for clarity
- `_local_normalize()` made private, merged into `model.py` (only caller)
- SQLite model uses lazy decay: `w * POWER(decay, current_t - last_updated_t)` at read time, no batch updates needed
- `StepModel` owns its SQLite connection, uses context manager pattern
- Experiment scripts save JSON only; visualization is a separate script
- `data.py` yields 3-tuples `(t, token_id, sdr)` — token_id needed for SDR definitions and accuracy tracking

## Next Steps
- [ ] Run exp0 with TinyStories data to verify learning signal
- [ ] Wire `StepModel` (SQLite) into experiment runner as alternative to functional model
- [ ] Test MiniGPT on a machine with torch support
- [ ] Build full comparison figures (STEP vs GPT on shared accuracy metric via Model protocol)
