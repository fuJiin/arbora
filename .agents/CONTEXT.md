# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Dual model implementations**:
  - `model.py`: Functional core with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix)
  - `db.py`: SQLite-backed `StepModel` class — dropped from exp0 due to O(80K) SQL upserts/token. Future: hybrid numpy compute + SQLite checkpoint.
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP with inverted index decode (O(k) per query)
  - `MiniGPTModel`: Inference-only GPT wrapper with rolling context buffer
- **Model protocol** (`protocol.py`): `Model` protocol for running STEP and baselines through the same harness
- **Key modules**: `sdr.py` (encoding), `metrics.py` (IoU, rolling mean), `data.py` (tokenizer + dataset caching + vocab filtering + .env HF token)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()`, JSON logging)
- **Baselines**: `mini_gpt.py` (2-layer transformer), `pretrain.py` (training loop with cosine LR schedule), `wrappers.py`

## exp1 Results (completed)
Branch: `worktree-exp0` — Ceiling transformer attempt with 10K vocab, 2M tokens

| Model | Params | Final Rolling Accuracy | Native Metric | Pre-train | Eval |
|-------|--------|----------------------|---------------|-----------|------|
| step_memory | 4.2M (n=2048) | 3.0% | IoU: 7.35% | ~6.5h (2M tokens) | 226s |
| mini_gpt | 1.7M (10K vocab) | 3.0% | CE Loss: 5.45 | ~10 min | 97s |

**Key finding**: 2M tokens is still severely insufficient. MiniGPT loss only reached 5.30 (vs target ~1.7 from pluto-1M). Published models train on 100-500M tokens. Both models stuck at 3% accuracy.

**What worked**: Inverted index decode (226s eval vs prior ~hours), 10K vocab clamping, cosine LR schedule with warmup.

**What didn't work**: 2M tokens is ~0.4% of the full TinyStories dataset. Need 50-100x more data for a proper ceiling.

## exp0 Results (completed)
Two-model comparison on TinyStories (50K vocab, 100K train tokens)

| Model | Params | Final Rolling Accuracy | Native Metric |
|-------|--------|----------------------|---------------|
| step_memory | 4.2M (n=2048) | 6.0% | IoU: 7.95% |
| mini_gpt | 6.8M (50K vocab) | 6.0% | CE Loss: 5.90 |

## TinyStories Baselines (from literature)

| Model | Params | Val Loss | Perplexity | Source |
|-------|--------|----------|------------|--------|
| pluto-1M | 1.5M | 1.71 | ~5.5 | github.com/tanaydesai/pluto |
| gpt-tinystories-8M | 8M | ~1.62 | ~5.1 | github.com/raymond-van/gpt-tinystories |

**Important**: Original TinyStories paper uses **10K vocab**. Our exp1 uses 10K vocab to match.

## Key Decisions
- `vocab_size` added to `EncoderConfig` (default 50257 for backward compat)
- Tokens >= vocab_size clamped to UNK (token 0) in data.py and pretrain.py
- Inverted index decode replaces dense matrix multiply (identical results, ~2600x fewer ops)
- Cosine LR schedule with linear warmup added to MiniGPT pretraining

## Immediate Roadmap
1. **Scale training data**: Need 50-100M+ tokens to get MiniGPT to val loss ~1.7. Current 2M is far too little.
2. **GPU training**: CPU pretraining at 2M tokens took ~10min for MiniGPT. Need GPU for 50M+ tokens.
3. **SDR adaptation**: Implement bit swapping to let representations learn (STEP's version of embedding training).
4. **Sweeps**: Weight decay, penalty factor, false-negative boosting.

## Future Research Ideas

### SDR adaptation / learned embeddings (high priority, later)
Fixed random SDRs are STEP's equivalent of random embeddings that never train. ~46% of token pairs have zero bit overlap (k²/n ≈ 0.78 expected shared bits). This limits association learning.

**Sleep-like consolidation** (preferred approach): Perform SDR mutations BETWEEN epochs. After an epoch, identify dead/low-utility bits, mutate them, then replay high-surprise episodes to let weights adjust naturally.
