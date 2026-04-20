"""ARC-AGI-3 baseline training loop.

Runs an Arbor V1→BG→M1 agent on ARC-AGI-3 interactive environments.
Focused on keyboard-only games (actions 1-5). Trains across multiple
episodes with reward shaping: +1.0 for level completion, -1.0 for death.

Usage:
    # Train on a specific game for 10 episodes:
    uv run python -m examples.arc.train --game ls20 --episodes 10

    # Train on all keyboard-only games:
    uv run python -m examples.arc.train --keyboard-only --episodes 5

    # Single-episode eval on all public games:
    uv run python -m examples.arc.train --all --episodes 1
"""

from __future__ import annotations

import argparse
import time

import arc_agi
import numpy as np
from arcengine import GameAction

from examples.arc.agent import ArcAgent, build_circuit
from examples.arc.data import keyboard_only_games
from examples.arc.encoder import ArcGridEncoder
from examples.arc.probes import ArcProbeBundle

# GameAction enum doesn't support GameAction(int) — build a lookup table.
_ACTION_BY_VALUE = {a.value: a for a in GameAction}


def run_episode(
    game_id: str,
    arcade: arc_agi.Arcade,
    agent: ArcAgent,
    encoder: ArcGridEncoder,
    *,
    max_steps: int = 500,
    verbose: bool = False,
    probes: ArcProbeBundle | None = None,
) -> dict:
    """Run one episode of a game. Returns results dict."""
    env = arcade.make(game_id)
    frame = env.reset()

    grid = frame.frame[0]
    total_steps = 0
    levels_completed = 0
    level_steps = 0
    died = False
    won = False

    # Reset per-episode state (but preserve learned weights)
    agent.reset_episode()
    if probes is not None:
        probes.reset()

    for _step_i in range(max_steps):
        # Agent processes frame and picks action
        # Reward is applied inside act() from previous step
        action_id, data = agent.act(grid, 0.0)

        # Probe observation: after circuit.process() (inside act()),
        # record V1/V2 state relative to the grid that was just processed.
        if probes is not None:
            probes.observe(agent.circuit, encoder, grid)

        # Step environment
        game_action = _ACTION_BY_VALUE[action_id]
        frame = env.step(game_action, data=data)
        if frame is None:
            break

        total_steps += 1
        level_steps += 1

        # Game over — no external penalty. The agent's intrinsic
        # signals (burst rate) are the only reward source. External
        # events like death should be learned, not hardcoded.
        if not frame.frame:
            died = True
            if verbose:
                print(f"    Died at step {total_steps} ({frame.state})")
            break

        grid = frame.frame[0]

        # Level completion — tracked for scoring. No external reward;
        # the visual change to a new level naturally produces a burst
        # spike which is the intrinsic signal.
        new_levels = frame.levels_completed
        if new_levels > levels_completed:
            if verbose:
                print(
                    f"    Level {new_levels} at step"
                    f" {total_steps} ({level_steps} actions)"
                )
            levels_completed = new_levels
            level_steps = 0

        # Win — done
        state_str = str(frame.state)
        if "WIN" in state_str:
            won = True
            if verbose:
                print(f"    WIN at step {total_steps}")
            break

    # If the loop exited without a terminal outcome (hit max_steps,
    # or the env returned None earlier), the episode timed out.
    timed_out = not won and not died
    return {
        "levels_completed": levels_completed,
        "total_steps": total_steps,
        "died": died,
        "won": won,
        "timed_out": timed_out,
    }


def train_game(
    game_id: str,
    arcade: arc_agi.Arcade,
    *,
    agent: ArcAgent | None = None,
    encoder: ArcGridEncoder | None = None,
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 42,
    verbose: bool = True,
    probes: ArcProbeBundle | None = None,
) -> dict:
    """Train an agent on one game over multiple episodes.

    If agent/encoder are provided, reuses them (weights carry across games).
    Otherwise creates fresh ones.
    """
    # Probe the game for its action space
    env = arcade.make(game_id)
    frame = env.reset()
    available_actions = frame.available_actions
    n_actions = len(available_actions)
    win_levels = frame.win_levels

    if verbose:
        print(f"  Actions: {available_actions}, Levels: {win_levels}")

    if encoder is None:
        encoder = ArcGridEncoder()
    if agent is None:
        circuit = build_circuit(encoder, n_actions=n_actions, seed=seed)
        agent = ArcAgent(encoder, circuit, available_actions=available_actions)
    else:
        # Update action map for this game's action space
        agent.update_actions(available_actions)

    episode_results = []

    for ep in range(n_episodes):
        result = run_episode(
            game_id,
            arcade,
            agent,
            encoder,
            max_steps=max_steps,
            verbose=verbose,
            probes=probes,
        )
        episode_results.append(result)

        if verbose:
            status = "died" if result["died"] else "alive"
            print(
                f"  Ep {ep + 1:3d}: {result['levels_completed']}/{win_levels} levels, "
                f"{result['total_steps']:4d} steps, {status}"
            )

        if probes is not None:
            probes.print_report()

    # Summary
    best_levels = max(r["levels_completed"] for r in episode_results)
    avg_steps = float(np.mean([r["total_steps"] for r in episode_results]))
    n_won = sum(1 for r in episode_results if r["won"])
    n_died = sum(1 for r in episode_results if r["died"])
    n_timed_out = sum(1 for r in episode_results if r["timed_out"])
    levels_by_episode = [r["levels_completed"] for r in episode_results]

    # Did the agent improve? Compare first half vs second half.
    # With < 2 episodes there is no late half.
    if n_episodes < 2:
        early_levels = late_levels = float(levels_by_episode[0])
    else:
        half = n_episodes // 2
        early_levels = float(np.mean(levels_by_episode[:half]))
        late_levels = float(np.mean(levels_by_episode[half:]))

    if verbose:
        print("  ---")
        print(f"  Best: {best_levels}/{win_levels} levels")
        print(
            f"  Outcomes (W/D/T): {n_won}/{n_died}/{n_timed_out}, "
            f"avg steps: {avg_steps:.0f}"
        )
        print(f"  Learning: levels {early_levels:.1f} -> {late_levels:.1f}")

    return {
        "game_id": game_id,
        "win_levels": win_levels,
        "best_levels": best_levels,
        "avg_steps": avg_steps,
        "n_won": n_won,
        "n_died": n_died,
        "n_timed_out": n_timed_out,
        "n_episodes": n_episodes,
        "early_levels": float(early_levels),
        "late_levels": float(late_levels),
        "episode_levels": levels_by_episode,
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 baseline with Arbor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game", type=str, help="Run on a specific game ID")
    group.add_argument(
        "--keyboard-only", action="store_true", help="Keyboard-only games"
    )
    group.add_argument("--all", action="store_true", help="All public games")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per game")
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--share-weights",
        action="store_true",
        help="Share weights across games",
    )
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Enable visual decodability probes",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    arcade = arc_agi.Arcade()

    if args.game:
        game_ids = [args.game]
    elif args.keyboard_only:
        game_ids = keyboard_only_games(arcade)
    else:
        game_ids = [e.game_id.split("-")[0] for e in arcade.get_environments()]

    probe_bundle = ArcProbeBundle() if args.probes else None

    if verbose:
        print(f"Training {len(game_ids)} games, {args.episodes} episodes each")
        if args.share_weights:
            print("  (sharing weights across games)")
        if args.probes:
            print("  (visual decodability probes enabled)")
        print()

    # Shared agent for cross-game learning (if enabled).
    # Built on first game, reused for all subsequent games.
    # Different games may have different action counts — we build for
    # the max action count and the agent remaps per game.
    shared_encoder: ArcGridEncoder | None = None
    shared_agent: ArcAgent | None = None
    if args.share_weights:
        shared_encoder = ArcGridEncoder()
        # Build circuit with max possible actions (7)
        circuit = build_circuit(shared_encoder, n_actions=7, seed=args.seed)
        # Will be initialized with first game's actions
        shared_agent = ArcAgent(
            shared_encoder, circuit, available_actions=list(range(1, 8))
        )

    results = []
    t0 = time.time()

    for i, game_id in enumerate(game_ids):
        if verbose:
            print(f"[{i + 1}/{len(game_ids)}] {game_id}")
        result = train_game(
            game_id,
            arcade,
            agent=shared_agent if args.share_weights else None,
            encoder=shared_encoder if args.share_weights else None,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=verbose,
            probes=probe_bundle,
        )
        results.append(result)
        if verbose:
            print()

    elapsed = time.time() - t0

    # Overall summary
    print("=" * 60)
    print(f"Results: {len(results)} games x {args.episodes} episodes, {elapsed:.1f}s")
    print()
    # W/D/T = Won / Died / Timed-out episodes (sum == episodes run).
    header = f"{'Game':8s} {'Best':>5s} {'W/D/T':>8s} {'Early→Late Levels':>20s}"
    print(header)
    print("-" * 60)
    for r in results:
        marker = "*" if r["best_levels"] > 0 else " "
        wdt = f"{r['n_won']}/{r['n_died']}/{r['n_timed_out']}"
        print(
            f"{marker}{r['game_id']:7s} "
            f"{r['best_levels']:2d}/{r['win_levels']:<2d} "
            f"{wdt:>8s}  "
            f"{r['early_levels']:4.1f} -> {r['late_levels']:4.1f}"
        )

    total_best = sum(r["best_levels"] for r in results)
    total_possible = sum(r["win_levels"] for r in results)
    any_learning = sum(1 for r in results if r["late_levels"] > r["early_levels"])
    print("-" * 60)
    print(f" Total best: {total_best}/{total_possible} levels")
    print(f" Games with improving levels: {any_learning}/{len(results)}")


if __name__ == "__main__":
    main()
