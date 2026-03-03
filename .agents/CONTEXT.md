# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Dual model implementations**:
  - `model.py`: Functional core with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix)
  - `db.py`: SQLite-backed `StepModel` class — dropped from exp0 due to O(80K) SQL upserts/token. Future: hybrid numpy compute + SQLite checkpoint.
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP (decode currently uses dense matrix multiply — needs inverted index fix)
  - `MiniGPTModel`: Inference-only GPT wrapper with rolling context buffer
- **Model protocol** (`protocol.py`): `Model` protocol for running STEP and baselines through the same harness
- **Key modules**: `sdr.py` (encoding), `metrics.py` (IoU, rolling mean), `data.py` (tokenizer + dataset caching + .env HF token)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()`, JSON logging)
- **Baselines**: `mini_gpt.py` (2-layer transformer), `pretrain.py` (training loop), `wrappers.py`

## exp0 Results (completed)
Branch: `worktree-exp0` — Two-model comparison on TinyStories (50K vocab)

| Model | Params | Final Rolling Accuracy | Native Metric | Pre-train | Eval |
|-------|--------|----------------------|---------------|-----------|------|
| step_memory | 4.2M (n=2048) | 6.0% | IoU: 7.95% | 26.7 min | 4.9 min |
| mini_gpt | 6.8M (94% embeddings) | 6.0% | CE Loss: 5.90 | ~2 min | 3.7 min |

**Key finding**: STEP matches MiniGPT at 6% accuracy. BUT both models are severely undertrained — converged TinyStories models achieve val loss ~1.3 (perplexity ~3.7) vs our 5.90 (perplexity ~365). The parity is encouraging but not yet meaningful.

Config: `n=2048, k=40, eligibility_window=101, max_lr=0.5, weight_decay=1.0, penalty_factor=0.0`
Pre-trained on 100K train tokens (full dataset has ~476M tokens), evaluated on 10K validation tokens.

## TinyStories Baselines (from literature)

| Model | Params | Val Loss | Perplexity | Source |
|-------|--------|----------|------------|--------|
| pluto-1M | 1.5M | 1.71 | ~5.5 | github.com/tanaydesai/pluto |
| gpt-tinystories-8M | 8M | ~1.62 | ~5.1 | github.com/raymond-van/gpt-tinystories |
| pluto-15M | 15M | 1.29 | ~3.6 | github.com/tanaydesai/pluto |
| gpt-tinystories-28M | 28M | ~1.32 | ~3.7 | github.com/raymond-van/gpt-tinystories |

**Important**: Original TinyStories paper uses **10K vocab** (top 10K tokens from GPT-Neo tokenizer). HuggingFace models expanded to 50K for compatibility. No published top-1 accuracy benchmarks exist — paper uses GPT-4 grading. We'd be establishing a new accuracy benchmark.

## Known Performance Issues

### Decode bottleneck (critical)
`StepMemoryModel._decode()` uses dense matrix multiply over 98%-sparse SDR matrix:
- Current: O(V_seen × n) = ~7.2M ops/call on 28.7MB matrix
- Inverted index: O(k² × V_seen / n) = ~2,700 ops/call on ~560KB
- **~2,600x speedup possible** with identical results (argmax IoU = argmax overlap when all SDRs have k bits)
- Also: `row_sums` recomputed every call (always k=40), `np.vstack` copies entire matrix per new token

### GPU acceleration
CuPy (drop-in numpy replacement) or JAX for predict/learn. Wins at n=8K+ where weight matrix exceeds CPU cache. Not needed at n=2048.

## Key Decisions
- No weight decay or penalty for exp0 — pure Hebbian reinforcement matches undertrained transformer
- STEP pre-trained on train split (same 100K tokens as MiniGPT) for fair comparison
- MiniGPT inference-only during eval (learn returns loss but no backward/optimizer)
- Dataset downloaded locally (non-streaming) with HF token from .env

## Future Research Ideas

### SDR adaptation / learned embeddings (high priority, later)
Fixed random SDRs are STEP's equivalent of random embeddings that never train. ~46% of token pairs have zero bit overlap (k²/n ≈ 0.78 expected shared bits). This limits association learning.

Proposed mechanisms:
- **Bit swapping (structural plasticity)**: Find "dead" bits (near-zero outgoing weight) per token, swap for "desired" bits (used by strong-weight neighbors). Analogous to brain pruning unused synapses + growing new ones.
- **SDR drift via co-occurrence**: When tokens frequently predict each other but share zero bits, migrate bits toward shared space. Hebbian wiring at the representation level.
- **Competitive bit allocation**: Track per-bit utility (avg weight magnitude). Low-utility bits get released and reassigned.

**Sleep-like consolidation** (preferred approach): Perform SDR mutations BETWEEN epochs, not during training. Inspired by memory consolidation during sleep:
1. After an epoch, identify dead/low-utility bits and move them (structural mutation)
2. Instead of manual weight transfer (which may have unintended consequences), **replay episodes where IoU was low** (the model was "surprised") to let the learning rule naturally update weights for the new bit configuration
3. This is analogous to the brain replaying surprising/important experiences during sleep to consolidate memory
4. Requires: storing a buffer of high-surprise episodes during training (token, t, IoU) for replay

### Sweeps (later)
- Weight decay: controls forgetting/generalization
- Penalty factor: penalizes false positives
- False-negative boosting: per-bit activation frequency tracking with adaptive LR scaling (homeostatic plasticity)
- Learning rate schedules

### Visualization & Interpretability (later)
- Synapse heatmaps, prediction decomposition, token neighborhoods, learning dynamics, eligibility trace visualization

## Immediate Roadmap
1. **Ceiling**: Train proper transformer baseline on TinyStories with 10K vocab (matching published benchmarks). Target: val loss ~1.7 (pluto-1M level). This establishes the accuracy ceiling STEP needs to reach.
2. **STEP decode fix**: Replace dense matrix multiply with inverted index (~2,600x speedup). Unblocks scaling experiments.
3. **Scale STEP**: More training tokens, multi-epoch, match param count to ceiling model. If n needs to grow past ~8K, add CuPy/JAX GPU support.
4. **SDR adaptation**: Implement bit swapping to let representations learn (STEP's version of embedding training).
5. **Sweeps**: Weight decay, penalty factor, false-negative boosting.
