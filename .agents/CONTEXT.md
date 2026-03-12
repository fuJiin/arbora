# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning via cortical representation building. Neocortical minicolumn model with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Architecture (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise activation
- **Dendritic segments** for prediction: fb (L2/3→L4), lat (L4→L4), l23 (L2/3→L2/3), HTM-style permanence learning
- **Per-neuron ff_weights**: each L4 neuron has own weights within column's structural mask
- **SensoryRegion**: local connectivity (radius = n_columns//4), structural masks on ff and segments
- **L2/3 lateral weights**: dense Hebbian (broad context) + dendritic segments (selective pattern predictions)
- **RepresentationTracker** (`representation.py`): primary metrics for evaluating cortex quality
- **Decoders** (`src/step/decoders/`): InvertedIndexDecoder + SynapticDecoder (monitoring only)
- **SurpriseTracker** (`surprise.py`): EMA-based burst rate tracker, outputs learning modulator [0, 2]
- **HierarchyConfig** (`config.py`): two-region config with R2 defaults
- **run_hierarchy()** (`runner.py`): two-region training loop with surprise modulation
- **Dashboard** (`cortex_dashboard.py`): single-region default, `--hierarchy` for dual-region view

## Two-Region Hierarchy (complete, tuned)
- **Region 1** (sensory): CharbitEncoder → 32 cols, standard config
- **Region 2** (secondary): R1's L2/3 firing rate → 16 cols, sliding window receptive fields
- **Inter-region signal**: `firing_rate_l23` — EMA of L2/3 spikes using `voltage_decay` as time constant (not boolean). Biologically grounded: postsynaptic temporal integration of spike trains.
- **R2 receptive fields**: `encoding_width=0` (sliding window). Tiled mode gave 75% overlap / 0.600 Jaccard between columns — columns couldn't differentiate. Sliding window gives 12.5% coverage with topographic neighbor-only overlap.
- **Surprise modulation**: R1 burst rate → SurpriseTracker → scales all R2 learning (ff LTP/LTD, lateral Hebbian, segment permanences). Models NE from locus coeruleus.
- **R2 tuned defaults**: lr=0.01, ltd=0.4, voltage_decay=0.8, eligibility_decay=0.98, synapse_decay=0.9999

## Results at 5k tokens (BabyLM, pre-segment-resweep config)
| Metric | R1 | R2 |
|--------|----|----|
| Burst rate | 63.7% | 60.8% |
| Selectivity | 0.607 | 0.694 |
| Context discrimination | 0.546 | **0.867** |
| Cross-col cosine | 0.073 | 0.098 |
| FF sparsity | 0.913 | 0.950 |
| Runtime overhead | — | ~1.3x |

## Canonical Setup
- **CharbitEncoder** — 6x richer similarity structure vs random encoder
- **BabyLM** dataset — more naturalistic, developmentally plausible
- **Per-neuron ff_weights** — always on, column-level path removed
- **Dendritic segments** — sole prediction mechanism
- **Segment params**: thresh=2, perm_inc=0.2, n_synapses=24 (re-swept with CharbitEncoder)
- **L2/3 segment params**: 4 segments, shared permanence params, `l23_prediction_boost=0` (uses fb_boost)

## Key Decisions
- **Representation quality over decoder accuracy** — sensory cortex builds representations for downstream use
- **Generation is planning, not next-token prediction** — brain does hierarchical planning (intention → plan → structure → words), not autoregressive completion. Future generation architecture should be planning-based (prefrontal/basal ganglia territory), not "motor cortex as decoder"
- **Prediction is the learning signal, not the generation mechanism** — predictive coding: cortex learns by predicting and updating on errors. Burst = surprise = learn. This is correct for building representations, separate from how output is generated.
- **Linear probe sanity check (2026-03-12)**: L2/3 representations at 20k tokens show weak next-token signal (1.5x majority baseline). Representations encode context structure (sentence boundaries, punctuation) but not lexical prediction. This is expected — sensory cortex builds contextual features, not prediction lookups. Not a validity test for the architecture.
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded and gives R2 smooth gradients to learn from
- **Sliding window > tiled for R2 receptive fields** — R1's L2/3 output has no character-position structure; tiled mode created near-total overlap
- **Low lr + high LTD for R2** — secondary region should learn slowly and prune aggressively for stable higher-level features
- **Feedback R2→R1 deferred** — requires thalamic relay (pulvinar) + apical dendrite compartments
- **Surprise-modulated learning (third-factor)** — R1 burst rate modulates R2 plasticity via NE-like signal
- **24 synapses/segment** — CharbitEncoder re-sweep showed wider segments capture richer context
- **L2/3 segments coexist with dense Hebbian** — dense weights provide broad lateral context, segments add selective pattern-specific predictions
- **Old STEP model removed (2026-03-12)** — flat weight matrix model, SQLite backend, baselines, old experiment harness all removed. Cortex architecture is the sole path forward.

## Codebase (post-cleanup 2026-03-12)
- `src/step/cortex/` — region, sensory, config, runner, diagnostics, representation, surprise, timeline
- `src/step/encoders/` — CharbitEncoder only (RandomEncoder/AdaptiveEncoder removed)
- `src/step/decoders/` — InvertedIndexDecoder, SynapticDecoder
- `src/step/env.py` — environment config
- `experiments/scripts/` — cortex dashboard, hierarchy eval, representation eval, sweep scripts, probe_l23
- Old STEP model (`model.py`, `db.py`, `wrappers.py`, `experiment.py`, etc.) and baselines removed

## Next Steps
- [ ] Add thalamic relay + feedback R2→R1 (activates prediction_gain, apical dendrites) — in progress in separate session
- [ ] Explore goal-directed planning architecture for response generation (not autoregressive)
