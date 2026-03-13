# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure (post-refactor)

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `surprise.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — `run_cortex()`, `run_hierarchy()`
- **`src/step/data.py`** — shared token loading (`prepare_tokens()`, `STORY_BOUNDARY`)
- **`src/step/viz/`** — dashboard chart builders (`cards.py`, `charts.py`, `layout.py`)
- **`src/step/encoders/`** — `CharbitEncoder` (canonical)
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`

## Two-Region Hierarchy
- **Region 1** (sensory): CharbitEncoder → 32 cols, standard config
- **Region 2** (secondary): R1's L2/3 firing rate → 16 cols, sliding window receptive fields
- **Feedforward**: `firing_rate_l23` EMA signal R1→R2
- **Feedback**: R2 `firing_rate_l23` → R1 apical segments → `prediction_gain` column boost
- **Surprise modulation**: R1 burst rate → SurpriseTracker → scales all R2 learning

## Canonical Setup
- **CharbitEncoder**, **BabyLM** dataset, **per-neuron ff_weights** (always on)
- **Dendritic segments**: fb (L2/3→L4), lat (L4→L4), l23 (L2/3→L2/3), apical (R2→R1)
- **Apical feedback**: `prediction_gain=2.5`, 4 apical segments per L4 neuron
- **R2 tuned defaults**: lr=0.01, ltd=0.4, voltage_decay=0.8, eligibility_decay=0.98

## Dashboard
- Single-region default, `--hierarchy` for dual-region tabbed view (Overview/R1/R2/Feedback)
- Stat cards use directional color logic (R2 vs R1 comparison, not absolute values)
- ASCII `->` for arrows (Unicode/HTML entities don't render reliably)
- Apical viz: segment connectivity over time, apical prediction hit rate

## Key Decisions
- **Representation quality over decoder accuracy** — sensory cortex builds representations for downstream regions
- **Motor cortex will generate responses** (not predict next token)
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded
- **Direct cortico-cortical feedback** — R2 L2/3 → R1 apical segments (no thalamic relay yet)
- **Apical = column-level, fb = neuron-level** — apical prediction_gain is multiplicative column boost, fb_boost is additive per-neuron
- **prediction_gain=2.5** — sweep showed 2.5 is sweet spot: -7.3% burst, +0.083 ctx disc at 50k

## Next Steps
- [ ] Explore motor cortex design for response generation
- [ ] Consider L5 (motor output) and L6 (thalamic control) layers
