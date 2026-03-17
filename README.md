# STEP — Sparse Temporal Eligibility Propagation

A research project exploring biologically-plausible language learning using cortical minicolumn architecture. No backpropagation — learning uses local Hebbian rules, eligibility traces, and reward-modulated plasticity (three-factor learning).

## What it does

STEP builds a cortical hierarchy that learns to understand and produce language through developmental stages, mirroring infant speech acquisition:

1. **Listening** — Sensory regions (S1→S2→S3) learn character, word, and topic representations from text. Motor region (M1) observes passively.
2. **Babbling** — M1 produces characters autoregressively, hears itself through S1. Curiosity reward (dopamine RPE) drives exploration. Caregiver reward nudges toward real words.
3. **Speaking** — (in progress) M1 produces English words like "the", "mom", "ask" through interleaved listening and babbling.

## Architecture

```
Sensory hierarchy          Motor output
S3 (topic/phrase)
  ↑ apical
S2 (word-level)
  ↑ apical
S1 (char-level) ─────────→ M1 (L5 output)
  128 cols, k=8              32 cols, k=4
```

- **Feedforward**: Learned per-neuron weights (Hebbian). Structural sparsity.
- **Lateral**: Dendritic segments for temporal prediction (L4 and L2/3).
- **Apical feedback**: Per-neuron gain weights (BAC firing model).
- **L5 output**: Learned weights mapping M1 L2/3 → token scores. Three-factor RL.
- **Reward**: Curiosity (RPE on S1 burst rate) + caregiver (live prefix matching against vocabulary).

## Quick start

```bash
uv sync  # Python 3.12+, requires uv

# Stage 1: Sensory learning + M1 listening (300k tokens)
uv run experiments/scripts/cortex_staged.py --stage sensory --tokens 300000

# Stage 2: Interleaved listening + babbling (100k babble steps)
uv run experiments/scripts/cortex_staged.py --stage babbling --tokens 100000

# Interactive REPL (load a checkpoint)
uv run experiments/scripts/cortex_repl.py --checkpoint stage2_babbling --dataset babylm
```

## REPL commands

```
/help            Show available commands
/info            Model capabilities, vocabulary, sample prompts
/babble [N]      Watch M1 babble N chars in real time (default 200)
/probe           Show S1/S2/S3/M1 representation quality
/stats           BPC statistics
/warmup [N]      Train on N more corpus chars
/save [name]     Save checkpoint
/load [name]     Load checkpoint
/reset           Clear working memory
/quit            Exit
```

Type text to feed through the hierarchy. After your input, M1 gets a turn to speak.

## Project structure

```
src/step/
├── cortex/
│   ├── region.py       # Base cortical region (L4/L2/3, segments, apical)
│   ├── sensory.py      # Encoding-aware region with local connectivity
│   ├── motor.py        # M1 with L5 output, three-factor learning, babbling
│   ├── topology.py     # Region wiring, run loops (corpus, babbling, interleaved)
│   ├── stages.py       # Training stage definitions (SENSORY, BABBLING, GUIDED)
│   ├── reward.py       # CuriosityReward (RPE), CaregiverReward (prefix matching)
│   ├── basal_ganglia.py # Go/no-go gating with three-factor plasticity
│   └── modulators.py   # Surprise, reward, thalamic gate modulators
├── probes/
│   ├── centroid_bpc.py # Non-learned BPC probe (primary metric)
│   ├── bpc.py          # Dendritic decoder BPC (deprecated)
│   ├── diagnostics.py  # Per-step diagnostics
│   └── representation.py # Selectivity, discrimination, RF quality
├── encoders/
│   └── positional.py   # Positional character encoder
├── decoders/
│   └── dendritic.py    # Segment-based decoder with permanence decay
├── config.py           # Region configs and factory functions
├── data.py             # BabyLM, TinyDialogues, PersonaChat loaders
└── runs.py             # Run saving utilities

experiments/scripts/
├── cortex_staged.py    # Staged training runner (sensory → babbling)
├── cortex_repl.py      # Interactive REPL with babbling mode
├── cortex_run.py       # Single-stage runner (legacy)
└── cortex_dashboard.py # Web visualization dashboard
```

## Development

```bash
uv run python -m pytest tests/ -v   # 192 tests
uv run ruff check src/ tests/       # lint
```

## Key results

- **Centroid BPC**: 4.59 at 300k (random baseline 5.0). Non-learned probe confirms monotonic learning.
- **Motor babbling**: M1 discovers all 32 BabyLM chars through curiosity-driven RL.
- **English words from babbling**: "the", "mom", "ask", "him", "not", "has" emerge from interleaved listen+babble training. 25 distinct English bigrams, space as most frequent char.
- **Developmental stages**: Listening → babbling → speaking mirrors infant acquisition. Interleaved training prevents L5 drift.
