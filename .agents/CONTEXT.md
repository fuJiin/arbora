# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Functional core**: Pure functions with explicit state threading via `ModelState` NamedTuple
- **Key modules**: `sdr.py` (encoding), `model.py` (predict/update/observe), `training.py` (train loop), `data.py` (tokenizer + dataset streaming)
- **Experiment infra**: `experiment.py` (seed-controlled runner, JSON logging), `figures.py` (matplotlib plots with error bars)
- **Configs**: `EncoderConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`
- **In-place mutation**: Weight arrays mutated inside `update()` for performance (documented, intentional)

## Current Work
- Core STEP package extracted and tested (40 tests passing)
- MiniGPT implemented (from-scratch causal transformer, ~100 lines) but untested — torch unavailable on current platform (macOS x86_64, needs arm64 or Linux)
- Comparison harness: `step_next_token_accuracy()` implemented in `baselines/compare.py`
- Initial experiments complete with figures:
  - `step_quick`: 5 seeds x 5k tokens, learning curve + IoU distribution
  - SDR size sweep: n={256,512,1024}, 3 seeds each
  - Learning rate sweep: lr={0.1,0.3,0.5,0.8}, 3 seeds each
- **Key finding**: IoU is flat at ~0.04 across all configs. Model is not learning above chance. This is the critical issue to investigate.

## Key Decisions
- `compute_iou` uses proper set IoU (intersection/union), not overlap/k from dump.py
- `penalty_factor` separated from `weight_decay` (fixes original bug)
- SDRs are `frozenset[int]` for immutability and hashability
- Tokenizer isolated in `data.py`; `sdr.py` is pure math with zero I/O deps
- Experiments use random token sequences (seeded) rather than real dataset streaming, for speed and reproducibility
- Research outputs: `experiments/configs/` committed, `experiments/runs/` gitignored, `experiments/figures/` committed

## Next Steps
- [ ] Investigate why IoU is flat — likely the random token stream has no learnable structure (no repeated sequences). Try repeating short patterns or real text.
- [ ] Run experiments with real TinyStories data (requires network, slower)
- [ ] Test MiniGPT on a machine with torch support
- [ ] Build full comparison figures (STEP vs GPT on shared accuracy metric)
- [ ] Rename directory `nano_egpt` → `step` (can't be done mid-session; breaks Bash tool cwd)
