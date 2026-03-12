# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Architecture (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise activation
- **Dendritic segments** for prediction: fb (L2/3→L4), lat (L4→L4), l23 (L2/3→L2/3), apical (R2 L2/3→R1 L4), HTM-style permanence learning
- **Per-neuron ff_weights**: each L4 neuron has own weights within column's structural mask
- **SensoryRegion**: local connectivity (radius = n_columns//4), structural masks on ff and segments
- **L2/3 lateral weights**: dense Hebbian (broad context) + dendritic segments (selective pattern predictions)
- **Apical feedback** (`init_apical_segments()`): R2 L2/3 → R1 L4 via dendritic segments + `prediction_gain` (column-level multiplicative boost)
- **RepresentationTracker** (`representation.py`): primary metrics for evaluating cortex quality
- **Decoders** (`src/step/decoders/`): InvertedIndexDecoder + SynapticDecoder (monitoring only)
- **SurpriseTracker** (`surprise.py`): EMA-based burst rate tracker, outputs learning modulator [0, 2]
- **HierarchyConfig** (`config.py`): two-region config with R2 defaults
- **run_hierarchy()** (`runner.py`): two-region training loop with surprise modulation + apical feedback
- **Dashboard** (`cortex_dashboard.py`): single-region default, `--hierarchy` for dual-region view

## Two-Region Hierarchy
- **Region 1** (sensory): CharbitEncoder → 32 cols, standard config
- **Region 2** (secondary): R1's L2/3 firing rate → 16 cols, sliding window receptive fields
- **Feedforward**: `firing_rate_l23` EMA signal R1→R2
- **Feedback**: R2 `firing_rate_l23` → R1 apical segments → `prediction_gain` column boost
- **Surprise modulation**: R1 burst rate → SurpriseTracker → scales all R2 learning
- **R2 tuned defaults**: lr=0.01, ltd=0.4, voltage_decay=0.8, eligibility_decay=0.98, synapse_decay=0.9999

## Apical Feedback Sweep Results (BabyLM)

| Gain | 10k Burst Δ | 10k Ctx Δ | 20k Burst Δ | 20k Ctx Δ | 50k Burst Δ | 50k Ctx Δ |
|------|-----------|---------|-----------|---------|-----------|---------|
| 1.5  | -2.5%     | -0.020  | -2.2%     | -0.028  | -1.9%     | -0.013  |
| 2.0  | -2.1%     | +0.016  | -1.9%     | -0.019  | -1.1%     | +0.005  |
| 2.5  | —         | —       | -12.0%    | +0.090  | -7.3%     | +0.083  |
| 3.0  | -26.3%    | +0.215  | -22.9%    | +0.200  | -13.5%    | +0.146  |

**Best: gain=2.5** — sustained burst reduction + ctx disc improvement + best apical connectivity (4.1% at 50k).

## Canonical Setup
- **CharbitEncoder** — 6x richer similarity structure vs random encoder
- **BabyLM** dataset — more naturalistic, developmentally plausible
- **Per-neuron ff_weights** — always on, column-level path removed
- **Dendritic segments** — sole prediction mechanism
- **Segment params**: thresh=2, perm_inc=0.2, n_synapses=24
- **L2/3 segment params**: 4 segments, shared permanence params, `l23_prediction_boost=0` (uses fb_boost)
- **Apical feedback**: `prediction_gain=2.5`, 4 apical segments per L4 neuron

## Key Decisions
- **Representation quality over decoder accuracy** — sensory cortex builds representations for downstream regions
- **Motor cortex will generate responses** (not predict next token)
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded
- **Sliding window > tiled for R2 receptive fields** — R1's L2/3 output has no character-position structure
- **Low lr + high LTD for R2** — secondary region learns slowly, prunes aggressively
- **Direct cortico-cortical feedback (no thalamic relay)** — R2 L2/3 → R1 apical segments. L5 (motor output) and L6 (thalamic control) deferred.
- **Apical = column-level, fb = neuron-level** — apical prediction_gain is multiplicative column boost ("this column is relevant"), fb_boost is additive per-neuron ("this specific neuron will fire")
- **prediction_gain=2.5** — sweep showed 2.5 is sweet spot: -7.3% burst, +0.083 ctx disc at 50k. gain=3.0 too aggressive (apical connectivity declining from overpunishment).

## Next Steps
- [ ] Update default `prediction_gain` to 2.5 in CortexConfig
- [ ] Explore motor cortex design for response generation
