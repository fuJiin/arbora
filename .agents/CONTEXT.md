# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
ChatEnv.step(action) -> (obs, reward)
  └── ChatAgent.step(obs) + decode_action()
        └── Circuit.process(encoding, motor_active) -> ndarray  [pure neural]

ChatTrainHarness orchestrates: agent.step → probes.observe → agent.decode_action → env.step

Topo: S1 → S2 → S3 → PFC → M2 → M1
Layers: L4 (input) → L2/3 (associative) → L5 (output/feedback)

Circuit.process() is environment-agnostic. No EOM state, no hooks, no logging.
All telemetry flows through probes → typed snapshots.
```

## Package structure
```
src/step/
├── cortex/          # Pure neural: Circuit, Region, connections, modulators
├── harness/chat/    # ChatTrainHarness (env + agent + probes + reporter)
├── agent/chat.py    # ChatAgent (step, decode_action, reset)
├── probes/
│   ├── core.py      # Probe protocol, LaminaProbe
│   ├── chat.py      # ChatLaminaProbe, ChatMotorProbe
│   └── modulators.py # ModulatorProbe (surprise, thalamic, reward)
├── snapshots/
│   ├── core.py      # L4Snapshot, L23Snapshot, LaminaRegionSnapshot
│   └── chat.py      # ChatL23Snapshot, MotorRegionSnapshot
├── reporting/chat.py # ChatReporter (log lines from typed probe snapshots)
├── environment.py   # ChatEnv, ChatObs
└── encoders/        # PositionalCharEncoder, CharbitEncoder

experiments/scripts/chat/  # train.py, repl.py, sweep_s1.py
```

## Probe framework (complete)

All metrics flow through `probe.observe(circuit, **kwargs) → probe.snapshot()`.

| Probe | Location | Metrics |
|-------|----------|---------|
| LaminaProbe | probes/core.py | L4 recall/precision/sparseness, L2/3 eff_dim |
| ChatLaminaProbe | probes/chat.py | + linear probe accuracy, ctx discrimination |
| ChatMotorProbe | probes/chat.py | Motor accuracy, BG gate, turn-taking, rewards |
| ModulatorProbe | probes/modulators.py | Surprise, thalamic, reward modulator time series |

TrainResult = `{probe_snapshots, elapsed_seconds}`. That's it.

## Session: 2026-03-27 → 2026-03-28 (probe framework + simplification)

### Merged PRs (10 total)
1. PR #42 — Probe protocol, LaminaProbe, ChatLaminaProbe (STEP-80)
2. PR #43 — Vectorize Hebbian learning hot paths
3. PR #44 — Wire probes into train() (STEP-81)
4. PR #45 — ChatMotorProbe, prune RunMetrics, kill run_hierarchy (STEP-82)
5. PR #46 — Streamline Circuit: pure process(), remove EOM state (STEP-79)
6. PR #47 — ChatReporter, delete RunHooks/RunMetrics/CortexResult (STEP-83)
7. PR #48 — ChatTrainHarness, delete train.py, prune 14 scripts (STEP-84)
8. PR #49 — Agent step/decode_action split (STEP-87)
9. PR #50 — ModulatorProbe, simplify TrainResult (STEP-86)

### Key decisions
- **Harness not Runner** — ChatTrainHarness (instrumentation, not execution)
- **Chat prefix convention** — chat-specific: ChatEnv, ChatAgent, ChatLaminaProbe, ChatMotorProbe, ChatTrainHarness
- **Linear probe is L2/3 KPI** — dendritic decoder BPC → REPL-only
- **Typed snapshots** — L4Snapshot, L23Snapshot, etc. in snapshots/ package
- **Agent step/decode_action split** — harness calls step → probes → decode_action
- **Scripts by modality** — experiments/scripts/chat/ (ready for scripts/rl/)

### Deleted
- RunHooks (585 lines), RunMetrics, CortexResult, StepHooks
- run_hierarchy(), HierarchyMetrics
- runner.py, runs.py, viz/ module
- 14 stale experiment scripts
- RepresentationTracker calls from training loop

## Remaining tickets

**Done this session:**
- [x] STEP-63 Probe protocol (parent, all 5 phases + 86 + 87)
- [x] STEP-78 Fix L4 firing_rate bug
- [x] STEP-79 Streamline Circuit
- [x] STEP-80–84 Phases 1–5
- [x] STEP-86 ModulatorProbe
- [x] STEP-87 Agent step/decode split

**Next: RL/Puffer integration**
- [ ] Puffer harness — pressure-test abstractions with RL environment
- [ ] PFC/BG/cerebellum/hippocampus design in RL context

**Architecture:**
- [ ] STEP-77 Agranular motor/PFC regions — skip L4 (M)
- [ ] STEP-30 Region Protocol typing (M)

**Features:**
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
