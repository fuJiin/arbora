"""Ablation runner for the HC v1 MiniGrid memory benchmark (ARB-118).

Runs two arms — `build_baseline_circuit` (no HC) and
`build_hippocampal_circuit` (with HC) — on the same environment and
same env seeds, then reports success rate, time-to-first-success, and
mean steps per arm. The HC arm additionally reports mechanistic
diagnostics from `HippocampalProbe` (per-step observables) and
`RetentionTracker` (non-destructive retention of a fixed reference
pattern set).

Topology (post-ARB-123)
-----------------------
Both arms share `T1 → M1` and `T1 → BG → M1 (mod)`. The HC arm adds:

    T1 → HC → BG

HC projects into BG (ventral-striatum-analog) rather than M1; the
ablation tests whether memory-informed value signals to BG help on
memory-gated tasks. See `examples/minigrid/presets.py` for the full
wiring rationale.

Usage
-----
::

    uv run python -m examples.minigrid.ablation \
        --env MiniGrid-MemoryS13-v0 --seeds 5 --episodes 200

Falsifiable claim (per ARB-118):

    Hippocampal pattern completion is necessary for one-shot relational
    binding on MemoryS13 under Arbora's minimal sensorimotor architecture.

If `hippocampal.success_rate > baseline.success_rate` with non-
overlapping error bars AND the HC-probe mechanistic stats corroborate
(ca3_revisit_stability high, ca1_match positive on revisits, retention
not collapsing), claim supported. Equal or worse → three possible
causes spelled out in the ARB-118 ticket; the probe stats help
disambiguate HC-bug from task-not-gated from baseline-already-solves.

Task-variant notes
------------------
MiniGrid exposes `MemoryS7`, `MemoryS11`, `MemoryS13` as standard envs
(S-number sets the grid size; larger grids mean longer corridors and
stronger memory demand). `MemoryS17` exists only as the
`MemoryS17Random-v0` variant (randomized target placement). Use
`MemoryS17Random-v0` as a harder follow-up if MemoryS13 isn't
memory-gated enough at your step budget.

Success metric
--------------
The `success_rate` column counts episodes with `reward > 0`, not
episodes that terminated. MemoryEnv terminates on *any* choice (target
or distractor), so "terminated" would conflate solving with
wrong-answer endings. `term_rate` is reported alongside for
transparency.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit
from arbora.hippocampus import HippocampalRegion
from arbora.probes import CortexStabilityTracker, HippocampalProbe, RetentionTracker
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness
from examples.minigrid.presets import (
    build_baseline_circuit,
    build_hippocampal_circuit,
)

if TYPE_CHECKING:
    pass

# Seed offset for probe-pattern collection. Kept far from the training-seed
# range so probe-pattern initial states never coincide with training-seed
# initial states.
_PROBE_PATTERN_SEED_OFFSET = 100_000

# Cap on HippocampalProbe's per-step log size. Long runs can generate
# 100k+ steps; rolling counters in the probe cover summary stats
# beyond this cap.
_HC_PROBE_MAX_STEPS = 10_000


@dataclass
class EpisodeEvent:
    """Per-episode outcome recorded by the probe.

    - `terminated` mirrors the gymnasium `terminated` flag (MDP ended
      naturally — can include wrong-answer endings in MemoryEnv).
    - `reward` is the total reward for the episode.
    - `solved` is the load-bearing success metric: `reward > 0`. In
      MiniGrid memory tasks, a terminated-but-unrewarded episode means
      the agent picked the distractor, not the target.
    """

    terminated: bool
    steps: int
    reward: float

    @property
    def solved(self) -> bool:
        return self.reward > 0.0


class EpisodeProbe:
    """Minimal probe that records per-episode outcomes.

    Duck-typed against `MiniGridHarness`: it looks for `observe()`,
    `boundary()`, `episode_end()`, and `snapshot()` on any probe. We
    only care about `episode_end`.
    """

    name = "episode"

    def __init__(self) -> None:
        self.events: list[EpisodeEvent] = []

    def observe(self, circuit, step: int = 0) -> None:
        pass

    def boundary(self) -> None:
        pass

    def episode_end(self, success: bool, steps: int, reward: float) -> None:
        # `success` here is `terminated` from the harness; the actual
        # "did the agent solve the task" metric is derived from reward.
        self.events.append(EpisodeEvent(bool(success), int(steps), float(reward)))

    def snapshot(self) -> dict:
        return {"events": self.events}


@dataclass
class ArmResult:
    """Outcome of running one arm on one seed."""

    name: str
    seed: int
    events: list[EpisodeEvent] = field(default_factory=list)
    # Mechanistic diagnostics.
    # hc_summary / final_retention are HC-arm-only (None/empty on baseline).
    # t1_stability is populated on *both* arms — T1 is shared, and drift
    # there is the primary diagnostic for HC retention failures.
    hc_summary: dict = field(default_factory=dict)
    final_retention: list[float] | None = None
    t1_stability: list[float] | None = None

    @property
    def success_rate(self) -> float:
        """Fraction of episodes the agent actually solved (reward > 0).

        Uses the reward-based `solved` flag rather than the
        `terminated` flag, because MemoryEnv terminates on wrong-answer
        picks too.
        """
        if not self.events:
            return 0.0
        return sum(e.solved for e in self.events) / len(self.events)

    @property
    def termination_rate(self) -> float:
        """Fraction of episodes that ended via `terminated=True`."""
        if not self.events:
            return 0.0
        return sum(e.terminated for e in self.events) / len(self.events)

    @property
    def time_to_first_success(self) -> int | None:
        """Episode number (1-indexed) of the first solved episode."""
        for i, e in enumerate(self.events):
            if e.solved:
                return i + 1
        return None

    @property
    def mean_steps(self) -> float:
        if not self.events:
            return 0.0
        return statistics.mean(e.steps for e in self.events)

    @property
    def mean_retention(self) -> float | None:
        """Mean retention across all probe patterns at run end, or None."""
        if not self.final_retention:
            return None
        return statistics.mean(self.final_retention)

    @property
    def mean_t1_stability(self) -> float | None:
        """Mean T1 L2/3 stability across reference encodings at run end."""
        if not self.t1_stability:
            return None
        return statistics.mean(self.t1_stability)


def _find_hippocampal_region(circuit: Circuit) -> HippocampalRegion | None:
    """Locate a HippocampalRegion in the circuit, if any."""
    for state in circuit._regions.values():
        region = state.region
        if isinstance(region, HippocampalRegion):
            return region
    return None


def _find_sensory_region(circuit: Circuit, name: str = "T1") -> SensoryRegion | None:
    """Locate the named SensoryRegion (default T1) in the circuit."""
    state = circuit._regions.get(name)
    if state is None:
        return None
    region = state.region
    return region if isinstance(region, SensoryRegion) else None


def _reference_encodings(
    env_id: str,
    encoder: MiniGridEncoder,
    n: int,
    seed_offset: int = _PROBE_PATTERN_SEED_OFFSET,
) -> list[np.ndarray]:
    """Collect N encoded initial observations from fresh envs.

    These are real MiniGrid encodings (984-dim), used as reference
    inputs for the T1 stability tracker. Deliberately disjoint from
    the training env: each encoding comes from a fresh env at a
    distinct seed.
    """
    encodings: list[np.ndarray] = []
    for i in range(n):
        env = MiniGridEnv(env_id, max_episodes=1, seed=seed_offset + i)
        obs = env.reset()
        encodings.append(encoder.encode(obs))
    return encodings


def _synthetic_probe_patterns(
    dim: int,
    n: int,
    *,
    sparsity: float = 0.06,
    seed: int = _PROBE_PATTERN_SEED_OFFSET,
) -> list[np.ndarray]:
    """Generate N sparse binary probe patterns at HC's input dim.

    HC's `input_port` expects cortical-dim vectors (T1 L2/3 output),
    not raw 984-dim encoder output. Rather than cloning T1 to generate
    realistic L2/3 patterns for probes, we use synthetic sparse
    binary vectors matching cortical sparsity (~6%). This keeps the
    retention test decoupled from T1 — it measures HC's ability to
    retain what it binds, not the combined T1+HC system.

    Each probe pattern has a distinct active-unit set (same seed, different
    draws), so they're mutually near-orthogonal and bind to distinct
    CA3 attractors.
    """
    rng = np.random.default_rng(seed)
    k = max(1, round(dim * sparsity))
    probe_patterns: list[np.ndarray] = []
    for _ in range(n):
        pat = np.zeros(dim, dtype=np.bool_)
        pat[rng.choice(dim, size=k, replace=False)] = True
        probe_patterns.append(pat)
    return probe_patterns


def run_arm(
    name: str,
    builder: Callable[[MiniGridEncoder], Circuit],
    *,
    env_id: str,
    episodes: int,
    seed: int,
    n_probe_patterns: int = 8,
    n_stability_refs: int = 8,
    trace: bool = False,
    trace_every: int = 1,
) -> ArmResult:
    """Run one arm on one seed and return the episode history + diagnostics.

    Set `trace=True` to stream a compact per-step line (see
    `examples.minigrid.trace.TraceProbe`). Intended for debugging
    individual episodes, not for large sweeps.
    """
    encoder = MiniGridEncoder()
    circuit = builder(encoder)
    env = MiniGridEnv(env_id, max_episodes=episodes, seed=seed)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)

    # T1 stability tracker — primes a deepcopy-based non-destructive
    # measurement with N real env encodings. Populates on both arms
    # because T1 is shared; drift is the primary diagnostic for HC
    # retention failures under v1 defaults.
    t1 = _find_sensory_region(circuit)
    stability = None
    if t1 is not None:
        ref_encodings = _reference_encodings(env_id, encoder, n=n_stability_refs)
        stability = CortexStabilityTracker(t1, encodings=ref_encodings)

    # Retention tracker on the HC arm only. HC's lateral weights
    # acquire these synthetic patterns at setup; `.measure()` at run
    # end tells us whether they're still retrievable after the training
    # run has laid down many more memories on top.
    hc = _find_hippocampal_region(circuit)
    retention = None
    if hc is not None:
        probe_patterns = _synthetic_probe_patterns(dim=hc.input_dim, n=n_probe_patterns)
        retention = RetentionTracker(hc, patterns=probe_patterns)

    episode_probe = EpisodeProbe()
    hc_probe = HippocampalProbe(max_steps=_HC_PROBE_MAX_STEPS)
    probes: list = [episode_probe, hc_probe]
    if trace:
        # Import here so main-path doesn't pay the cost when unused.
        from examples.minigrid.trace import TraceProbe

        probes.append(TraceProbe(every=trace_every))
    harness = MiniGridHarness(env, agent, probes=probes, log_interval=10**9)
    harness.run()

    hc_summary = hc_probe.snapshot().get("summary", {})
    final_retention = retention.measure() if retention is not None else None
    t1_stability = stability.measure() if stability is not None else None

    return ArmResult(
        name=name,
        seed=seed,
        events=episode_probe.events,
        hc_summary=dict(hc_summary),
        final_retention=final_retention,
        t1_stability=t1_stability,
    )


@dataclass
class AblationResult:
    """Aggregated results across seeds for both arms."""

    env_id: str
    baseline: list[ArmResult]
    hippocampal: list[ArmResult]

    def _summary(self, results: list[ArmResult]) -> dict[str, float]:
        srs = [r.success_rate for r in results]
        trs = [r.termination_rate for r in results]
        ttfs = [r.time_to_first_success for r in results if r.time_to_first_success]
        return {
            "success_rate_mean": statistics.mean(srs) if srs else 0.0,
            "success_rate_stdev": statistics.stdev(srs) if len(srs) > 1 else 0.0,
            "termination_rate_mean": statistics.mean(trs) if trs else 0.0,
            "ttfs_mean": statistics.mean(ttfs) if ttfs else float("nan"),
            "n_seeds_with_success": len(ttfs),
            "mean_steps": (
                statistics.mean(r.mean_steps for r in results) if results else 0.0
            ),
        }

    def _hc_mechanistic_summary(self) -> dict[str, float] | None:
        """Aggregate HC-only mechanistic stats across seeds."""
        if not self.hippocampal:
            return None

        def _collect(key: str) -> list[float]:
            return [r.hc_summary[key] for r in self.hippocampal if key in r.hc_summary]

        revisit_stability = _collect("ca3_revisit_stability")
        match_delta = _collect("ca1_match_revisit_minus_first")
        sat_frac = _collect("final_ca3_lateral_sat_frac")
        retentions = [
            r.mean_retention for r in self.hippocampal if r.mean_retention is not None
        ]

        summary: dict[str, float] = {}
        if revisit_stability:
            summary["ca3_revisit_stability_mean"] = statistics.mean(revisit_stability)
        if match_delta:
            summary["ca1_match_delta_mean"] = statistics.mean(match_delta)
        if sat_frac:
            summary["ca3_lateral_saturation_mean"] = statistics.mean(sat_frac)
        if retentions:
            summary["retention_mean"] = statistics.mean(retentions)
            summary["retention_stdev"] = (
                statistics.stdev(retentions) if len(retentions) > 1 else 0.0
            )
        return summary or None

    def _t1_stability_summary(
        self, results: list[ArmResult]
    ) -> dict[str, float] | None:
        """Aggregate T1 stability across seeds for a single arm."""
        stabilities = [
            r.mean_t1_stability for r in results if r.mean_t1_stability is not None
        ]
        if not stabilities:
            return None
        return {
            "mean": statistics.mean(stabilities),
            "stdev": statistics.stdev(stabilities) if len(stabilities) > 1 else 0.0,
        }

    def format_table(self) -> str:
        n = len(self.baseline)
        lines = [
            f"Ablation on {self.env_id} ({n} seeds)",
            "  success_rate = fraction of episodes with reward > 0",
            "  term_rate    = fraction of episodes terminated (incl. wrong choice)",
            "  ttfs         = time-to-first-success (episode index, 1-based)",
            "",
            f"{'arm':<14} {'success_rate':>18} {'term_rate':>12} "
            f"{'ttfs':>8} {'mean_steps':>12}",
        ]
        for arm_name, results in (
            ("baseline", self.baseline),
            ("hippocampal", self.hippocampal),
        ):
            s = self._summary(results)
            sr = f"{s['success_rate_mean']:.3f}±{s['success_rate_stdev']:.3f}"
            tr = f"{s['termination_rate_mean']:.3f}"
            ttfs = f"{s['ttfs_mean']:.1f}" if s["n_seeds_with_success"] > 0 else "n/a"
            lines.append(
                f"{arm_name:<14} {sr:>18} {tr:>12} {ttfs:>8} {s['mean_steps']:>12.1f}"
            )

        # T1 stability — populated on both arms (T1 is shared).
        lines += ["", "T1 representational stability (higher = less drift):"]
        for arm_name, results in (
            ("baseline", self.baseline),
            ("hippocampal", self.hippocampal),
        ):
            stab = self._t1_stability_summary(results)
            if stab is None:
                continue
            lines.append(
                f"  {arm_name:<14} T1 L2/3 stability: "
                f"{stab['mean']:.3f}±{stab['stdev']:.3f}"
            )

        # HC-only mechanistic summary
        hc = self._hc_mechanistic_summary()
        if hc:
            lines += ["", "HC mechanistic stats (hippocampal arm only):"]
            if "ca3_revisit_stability_mean" in hc:
                lines.append(
                    f"  ca3_revisit_stability (higher = more stable): "
                    f"{hc['ca3_revisit_stability_mean']:.3f}"
                )
            if "ca1_match_delta_mean" in hc:
                lines.append(
                    f"  ca1_match delta (revisit - first, higher = familiar): "
                    f"{hc['ca1_match_delta_mean']:+.3f}"
                )
            if "ca3_lateral_saturation_mean" in hc:
                lines.append(
                    f"  ca3 lateral saturation (frac at clip ceiling): "
                    f"{hc['ca3_lateral_saturation_mean']:.3f}"
                )
            if "retention_mean" in hc:
                rstd = hc.get("retention_stdev", 0.0)
                lines.append(
                    f"  probe-pattern retention at run end (mean±stdev): "
                    f"{hc['retention_mean']:.3f}±{rstd:.3f}"
                )
        return "\n".join(lines)


def run_ablation(
    env_id: str,
    *,
    n_seeds: int = 5,
    episodes_per_seed: int = 100,
    verbose: bool = True,
    n_probe_patterns: int = 8,
    n_stability_refs: int = 8,
    trace: bool = False,
    trace_every: int = 1,
) -> AblationResult:
    """Run both arms across `n_seeds` seeds and return aggregated results."""
    baseline_results: list[ArmResult] = []
    hc_results: list[ArmResult] = []
    for seed in range(n_seeds):
        if verbose:
            print(f"  seed {seed}: baseline...", end=" ", flush=True)
        baseline_results.append(
            run_arm(
                "baseline",
                build_baseline_circuit,
                env_id=env_id,
                episodes=episodes_per_seed,
                seed=seed,
                n_probe_patterns=n_probe_patterns,
                n_stability_refs=n_stability_refs,
                trace=trace,
                trace_every=trace_every,
            )
        )
        if verbose:
            print("hippocampal...", end=" ", flush=True)
        hc_results.append(
            run_arm(
                "hippocampal",
                build_hippocampal_circuit,
                env_id=env_id,
                episodes=episodes_per_seed,
                seed=seed,
                n_probe_patterns=n_probe_patterns,
                n_stability_refs=n_stability_refs,
                trace=trace,
                trace_every=trace_every,
            )
        )
        if verbose:
            print("done")
    return AblationResult(
        env_id=env_id, baseline=baseline_results, hippocampal=hc_results
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HC v1 ablation on MiniGrid memory tasks (ARB-118)"
    )
    parser.add_argument("--env", default="MiniGrid-MemoryS13-v0")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--probe-patterns", type=int, default=8)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Stream a compact per-step trace to stdout. For debugging "
        "individual episodes — produces a lot of output on long runs.",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="With --trace, print every N-th step. Default 1.",
    )
    args = parser.parse_args()

    print(
        f"Running HC ablation: env={args.env}, seeds={args.seeds}, "
        f"episodes={args.episodes}, probe_patterns={args.probe_patterns}"
    )
    result = run_ablation(
        args.env,
        n_seeds=args.seeds,
        episodes_per_seed=args.episodes,
        n_probe_patterns=args.probe_patterns,
        trace=args.trace,
        trace_every=args.trace_every,
    )
    print()
    print(result.format_table())


if __name__ == "__main__":
    main()
