# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3 layers, dendritic segments, per-neuron feedforward weights, apical gain feedback. NumPy + Numba, Python 3.12+, uv.

## Project Structure
- **`src/step/cortex/`** — `sensory.py`, `motor.py`, `topology.py`, `modulators.py`, `basal_ganglia.py`, `stages.py` (new)
- **`src/step/probes/`** — `diagnostics.py`, `representation.py`, `bpc.py`, `centroid_bpc.py` (new)
- **`src/step/decoders/`** — `dendritic.py` (permanence decay added)
- **`experiments/scripts/`** — `cortex_run.py`

## Architecture
- **S1**: 128 cols, k=8. Char-level. **S2**: 32 cols, k=4, buf=4. Word-level. **S3**: 32 cols, k=4, buf=8. Topic-level.
- **Feedforward**: Learned ff_weights (Hebbian). **Lateral**: Dendritic segments only. **Apical**: Per-neuron gain (BAC firing model).

## Centroid BPC — Key Breakthrough This Session

Dendritic decoder BPC was unreliable (learned component couldn't track drift). Replaced with **centroid-based BPC** (non-learned, EMA centroids + dot-product similarity).

**100k comparison**: dendritic 8.49, centroid **4.95** (random baseline ~5.0). Centroid shows monotonic improvement.

**1M run in progress** (at ~280k): centroid BPC 6.93 (10k) → 4.79 (100k) → **4.67 (280k)** and still improving. Dendritic BPC showed false regression at 200k; centroid confirms model was learning all along.

## Uncommitted Changes (main branch)
- `src/step/probes/centroid_bpc.py` — non-learned BPC probe
- `src/step/cortex/topology.py` — timeline_interval, decoder_perm_decay, centroid probe, freeze/enable APIs
- `src/step/cortex/region.py` — `learning_enabled` flag gates all plasticity
- `src/step/cortex/stages.py` — `TrainingStage` dataclass + predefined SENSORY/BABBLING/GUIDED stages
- `src/step/decoders/dendritic.py` — perm_decay parameter
- `experiments/scripts/cortex_run.py` — --timeline-interval, --decoder-decay flags

## Training Stages (developmental progression)

1. **Sensory** (S1→S2→S3): Self-supervised representation learning. All sensory regions learn, M1 disconnected. *Current focus — validating at 1M.*
2. **Babbling** (M1→S1→M1): Self-supervised motor exploration. S1 frozen (forward pass only). M1 learns forward model via efference copy.
3. **Guided babbling** (M1+BG+S2): RL. S2 word-recognition as natural reward signal. BG gates "keep going" vs "try something else."
4. **Imitation** (S1→S2→M2→M1): Echolalia. Hear word → encode → sequence → reproduce. Bootstraps M2 as sequencer. *Needs M2 region.*
5. **Generation** (PFC→M2→M1): Goal-directed RL. PFC maintains goals, BG per-stripe gating. *Needs PFC region.*

### Stage infrastructure built:
- `region.learning_enabled` — freeze all plasticity per region
- `connection.enabled` — disable specific connections
- `Topology.freeze_region()` / `unfreeze_region()` / `disable_connection()` / `enable_connection()`
- `TrainingStage.configure(topology)` — applies stage config
- Predefined: `SENSORY_STAGE`, `BABBLING_STAGE`, `GUIDED_BABBLING_STAGE`

### Stages vs BG/thalamus — design principle:
- **Stages** control "hardware readiness" — which circuits are online (discrete, developmental)
- **BG/thalamus** control "software" — how circuits are used once online (continuous, learned)
- Thalamic gate is within-stage (learns when to allow feedback). Stage determines the connection exists.

## PFC Design (from research, not yet implemented)
- **Working memory**: Slow voltage decay (~0.97) + binary maint_flag toggled by BG input gate per stripe
- **Representations**: Mixed-selective, conjunctive. Receives S2+S3. Denser activation.
- **Learning**: Three-factor Hebbian (pre × post × reward). Eligibility trace decay ~0.98.
- **BG gating**: Per-stripe input gate. RL via dopamine RPE. Output gating deferred.

## Key Decisions
- **Centroid BPC as primary metric**: Non-learned, can't break. Dendritic decoder kept as optional.
- **BabyLM as canonical dataset**: 53.5M chars child-directed speech.
- **REPL guardrails needed**: Show model capabilities, vocabulary hints, age-appropriate prompts. Users shouldn't talk to the model like an adult.
- **Apical gain tuning deferred**: Need centroid BPC at 1M scale first.

## Next Steps
- [ ] **Build staged runner script** — chains stages, loads/saves checkpoints
- [ ] **Finish 1M centroid run** — confirm cbpc keeps improving (in progress)
- [ ] **Commit session changes** — centroid probe, stage infra, timeline fix, freeze APIs
- [ ] **REPL guardrails** — show training stage, vocabulary, sample prompts
- [ ] **M1 babbling (Stage 2)** — first consumer of stage infra
- [ ] **Tune apical gain** — A/B with reliable centroid BPC
