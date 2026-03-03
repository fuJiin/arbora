# STEP — Sparse Temporal Eligibility Propagation

A research project exploring biologically-plausible sequence learning using eligibility traces and sparse distributed representations (SDRs), compared against standard backprop transformers.

STEP learns online, one token at a time, with no backpropagation. It maintains sparse synaptic connections between SDR bits, reinforcing connections that predict correctly and penalizing false positives, modulated by an exponentially-decaying eligibility trace over recent history.

## How it works

1. Each token is encoded as a **sparse distributed representation** — a fixed-size binary vector with `k` active bits out of `n` total (e.g., 40/2048).

2. To **predict** the next token, STEP looks at recently observed SDRs within an eligibility window, weighs each by recency, and aggregates synapse votes to produce a predicted SDR.

3. After observing the actual token, STEP **learns** by:
   - Reinforcing synapses from recent bits to correctly predicted bits
   - Penalizing synapses to false-positive bits
   - Scaling updates by prediction error (1 - IoU)

4. Weights decay exponentially over time, applied lazily at read time.

## Project structure

```
src/step/
├── model.py       # Functional in-memory model (NumPy)
├── db.py          # SQLite-backed model with lazy decay and SQL aggregation
├── protocol.py    # Model protocol for uniform evaluation
├── sdr.py         # Token -> SDR encoding
├── config.py      # EncoderConfig, ModelConfig, TrainingConfig
├── data.py        # TinyStories streaming tokenizer
├── training.py    # Training loop
├── experiment.py  # Seed-controlled experiment runner
├── metrics.py     # IoU, rolling mean
└── figures.py     # Matplotlib plotting utilities

src/baselines/
├── mini_gpt.py    # From-scratch causal transformer (~100 lines)
└── compare.py     # STEP vs GPT comparison harness

experiments/
├── configs/       # Experiment configs (JSON)
├── scripts/       # Run scripts and visualization
├── runs/          # Output data (gitignored)
└── figures/       # Generated plots
```

### Two model backends

**Functional** (`model.py`): In-memory NumPy arrays with explicit state threading via `ModelState` NamedTuple. Fast, good for prototyping.

**SQLite** (`db.py`): `StepModel` class backed by SQLite. Uses SQL aggregation for predict/learn, lazy weight decay via `POWER(decay, dt)`, and an inverted index for SDR-to-token decoding. Gives persistence, replay, and queryability for free.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync           # core dependencies
uv sync --dev     # + dev tools (pytest, ruff, ty)
```

Optional extras:

```bash
uv sync --extra viz          # matplotlib for figures
uv sync --extra comparison   # torch for MiniGPT baseline
```

## Usage

Run an experiment:

```bash
uv run python experiments/scripts/run_step.py experiments/configs/exp0_tinystories.json
```

Visualize results:

```bash
uv run python experiments/scripts/visualize.py experiments/runs/exp0_tinystories/
```

Run a parameter sweep:

```bash
uv run python experiments/scripts/run_sweep.py
```

## Development

```bash
uv run python -m pytest tests/ -v   # 60 tests
uv run ruff check src/ tests/       # lint
uv run ruff format --check src/     # format check
uv run ty check src/step/           # type check
```

CI runs all of the above on push/PR via GitHub Actions.

## Experiment naming

Experiments use the convention `exp{N}_{descriptive_name}`:

- `exp0_tinystories` — baseline STEP on TinyStories dataset

Configs live in `experiments/configs/`, results in `experiments/runs/` (gitignored, reproducible from code + seed).
