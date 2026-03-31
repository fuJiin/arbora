# Context: Arbora

## Overview
Biologically-plausible cortical learning framework. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Public neuroAI framework at v0.1.0.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)
- **GitHub:** fuJiin/arbora

## Architecture
```
from arbora import Circuit, SensoryRegion, MotorRegion, BasalGangliaRegion

NeuronGroup          -- universal base (firing_rate + modulation)
  Lamina             -- cortex-specific (+ voltage, active, predicted, trace)

Region types:
  Sensory (S1):  L4 --> L2/3       (granular, n_l5=0)
  Motor (M1):    L2/3 --> L5       (agranular, n_l4=0)
  BG:            striatum --> gpi  (subcortical, NeuronGroup)
  Full (S2+):    L4 --> L2/3 --> L5

Connection roles: FEEDFORWARD, APICAL, MODULATORY
Region protocol: input_port / output_port (NeuronGroup)
```

## Package structure (v0.1.0)
```
src/arbora/               # Core framework
  __init__.py            # 24 public exports
  neuron_group.py        # NeuronGroup base
  basal_ganglia.py       # BasalGangliaRegion
  agent.py               # BaseAgent, TrainResult
  cortex/                # CorticalRegion, SensoryRegion, MotorRegion, Circuit
  encoders/              # CharbitEncoder, OneHotCharEncoder, PositionalCharEncoder
  decoders/              # DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
  probes/                # Probe protocol, LaminaProbe
  snapshots/             # InputSnapshot, AssociationSnapshot

examples/                # Application-specific
  chat/                  # ChatAgent, harness, presets, stages, reward, data
  minigrid/              # MiniGridAgent, harness, encoder, probes, benchmark
```

## Session: 2026-03-28 → 2026-03-30

### Merged PRs
1. PR #51 — MiniGrid Phase 0
2. PR #52 — Agranular regions (STEP-77) + reward wiring (STEP-88)
3. PR #54 — BG redesign: Go/NoGo + tonic DA (STEP-90)
4. PR #56 — Rename step → arbor
5. PR #57 — Move app-specific code to examples/
6. PR #58 — Public API + README + v0.1.0

### Key decisions
- **Arbora** — renamed from STEP/Arbor. Dendritic arbors + cultivating a framework.
- **Core vs examples** — framework building blocks in src/arbora/, app-specific in examples/
- **NeuronGroup** — universal connectable surface (Brian2 naming). String group_id.
- **Lamina(NeuronGroup)** — cortex-specific dynamics
- **Region protocol** — input_port/output_port, process, apply_reward, reset
- **ConnectionRole.MODULATORY** — BG→M1 additive bias before k-WTA
- **v0.1.0** — working system, expect breaking changes in 0.x

### Honest eval findings
- BG tonic DA: 90% on lucky seed, 29% ± 25% across 5 seeds (p=0.95 vs random)
- Learning signal too weak — BG noise dominates three-factor contribution
- Empty-5x5 too easy for random (41%) — hard to show learning signal

## Next: ARC-AGI

Exploring ARC-AGI as next example/benchmark. Few-shot spatial reasoning — fundamentally different from RL. Tests abstract rule inference from 2-5 examples. LLMs struggle (~5-20%), humans excel (85%+). No published bio-plausible results.

## Remaining tickets

**Done:**
- [x] STEP-63, 77, 78-84, 86-90 (probes + agranular + reward + BG)
- [x] Rename to Arbora, restructure for public framework

**Open:**
- [ ] STEP-93 Tune BG tonic DA
- [ ] STEP-89 PFC working memory (MiniGrid Phase 2)
- [ ] STEP-30 Region Protocol typing (partially done)
- [ ] STEP-61 Adaptive gating (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] ARC-AGI example (new)
