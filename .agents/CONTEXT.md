# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Functional core**: Pure functions with explicit state threading via `ModelState` NamedTuple
- **Key modules**: `sdr.py` (encoding), `model.py` (predict/update/observe), `training.py` (train loop), `data.py` (tokenizer + dataset streaming)
- **Configs**: Three dataclasses — `EncoderConfig`, `ModelConfig`, `TrainingConfig`
- **In-place mutation**: Weight arrays mutated inside `update()` for performance (documented, intentional)

## Current Work
- Core STEP package extracted and tested (40 tests passing)
- Baseline stubs created but MiniGPT not yet implemented

## Key Decisions
- `compute_iou` uses proper set IoU (intersection/union), not overlap/k from dump.py
- `penalty_factor` separated from `weight_decay` (fixes original bug)
- SDRs are `frozenset[int]` for immutability and hashability
- Tokenizer isolated in `data.py`; `sdr.py` is pure math with zero I/O deps

## Next Steps
- [ ] Implement MiniGPT (~100 line causal transformer from scratch)
- [ ] Build comparison harness (shared next-token accuracy metric)
- [ ] Run experiments and produce figures for paper
