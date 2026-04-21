"""Ablation runner for the HC v1 MiniGrid memory benchmark (ARB-118).

Runs two arms — `build_baseline_circuit` (no HC) and
`build_hippocampal_circuit` (with HC) — on the same environment and
same env seeds, then reports success rate, time-to-first-success, and
mean steps per arm.

The ablation holds S1, BG, and M1 configurations identical between
arms (see presets.py); the only difference is whether HC mediates the
S1 → M1 feedforward path.

Usage
-----
::

    uv run python -m examples.minigrid.ablation \
        --env MiniGrid-MemoryS13-v0 --seeds 5 --episodes 200

Falsifiable claim (per ARB-118):

    Hippocampal pattern completion is necessary for one-shot relational
    binding on MemoryS13 under Arbora's minimal sensorimotor architecture.

If `hippocampal.success_rate > baseline.success_rate` with non-
overlapping error bars, claim supported. Equal or worse → three
possible causes spelled out in the ARB-118 ticket; diagnose S1 drift
before suspecting HC bugs.

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

from arbora.cortex.circuit import Circuit
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness
from examples.minigrid.presets import (
    build_baseline_circuit,
    build_hippocampal_circuit,
)


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
        """Fraction of episodes that ended via `terminated=True`.

        Included alongside success_rate so users can distinguish
        "agent made a (possibly wrong) choice" from "agent succeeded."
        """
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


def run_arm(
    name: str,
    builder: Callable[[MiniGridEncoder], Circuit],
    *,
    env_id: str,
    episodes: int,
    seed: int,
) -> ArmResult:
    """Run one arm on one seed and return the episode history."""
    encoder = MiniGridEncoder()
    circuit = builder(encoder)
    env = MiniGridEnv(env_id, max_episodes=episodes, seed=seed)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)
    probe = EpisodeProbe()
    harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10**9)
    harness.run()
    return ArmResult(name=name, seed=seed, events=probe.events)


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
        for name, results in (
            ("baseline", self.baseline),
            ("hippocampal", self.hippocampal),
        ):
            s = self._summary(results)
            sr = f"{s['success_rate_mean']:.3f}±{s['success_rate_stdev']:.3f}"
            tr = f"{s['termination_rate_mean']:.3f}"
            ttfs = f"{s['ttfs_mean']:.1f}" if s["n_seeds_with_success"] > 0 else "n/a"
            lines.append(
                f"{name:<14} {sr:>18} {tr:>12} {ttfs:>8} {s['mean_steps']:>12.1f}"
            )
        return "\n".join(lines)


def run_ablation(
    env_id: str,
    *,
    n_seeds: int = 5,
    episodes_per_seed: int = 100,
    verbose: bool = True,
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
    args = parser.parse_args()

    print(
        f"Running HC ablation: env={args.env}, "
        f"seeds={args.seeds}, episodes={args.episodes}"
    )
    result = run_ablation(
        args.env,
        n_seeds=args.seeds,
        episodes_per_seed=args.episodes,
    )
    print()
    print(result.format_table())


if __name__ == "__main__":
    main()
