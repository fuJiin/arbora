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
import sys
import time

import numpy as np

import arc_agi
from arcengine import GameAction

from examples.arc.agent import ArcAgent, build_circuit
from examples.arc.encoder import ArcGridEncoder

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
) -> dict:
    """Run one episode of a game. Returns results dict."""
    env = arcade.make(game_id)
    frame = env.reset()

    grid = frame.frame[0]
    total_steps = 0
    levels_completed = 0
    level_steps = 0
    died = False

    # Reset per-episode state (but preserve learned weights)
    agent.reset_episode()

    for step_i in range(max_steps):
        # Agent processes frame and picks action
        # Reward is applied inside act() from previous step
        action_id, data = agent.act(grid, 0.0)

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
                print(f"    Level {new_levels} at step {total_steps} ({level_steps} actions)")
            levels_completed = new_levels
            level_steps = 0

        # Win — done
        state_str = str(frame.state)
        if "WIN" in state_str:
            if verbose:
                print(f"    WIN at step {total_steps}")
            break

    return {
        "levels_completed": levels_completed,
        "total_steps": total_steps,
        "died": died,
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
        agent.available_actions = available_actions
        agent._action_map = agent._build_action_map(available_actions)

    episode_results = []

    for ep in range(n_episodes):
        result = run_episode(
            game_id, arcade, agent, encoder,
            max_steps=max_steps, verbose=verbose,
        )
        episode_results.append(result)

        if verbose:
            status = "died" if result["died"] else "alive"
            print(
                f"  Ep {ep+1:3d}: {result['levels_completed']}/{win_levels} levels, "
                f"{result['total_steps']:4d} steps, {status}"
            )

    # Summary
    best_levels = max(r["levels_completed"] for r in episode_results)
    avg_steps = np.mean([r["total_steps"] for r in episode_results])
    survival_rate = 1.0 - np.mean([r["died"] for r in episode_results])
    levels_by_episode = [r["levels_completed"] for r in episode_results]

    # Did the agent improve? Compare first half vs second half
    half = max(1, n_episodes // 2)
    early_levels = np.mean(levels_by_episode[:half])
    late_levels = np.mean(levels_by_episode[half:])
    early_survival = np.mean([1 - r["died"] for r in episode_results[:half]])
    late_survival = np.mean([1 - r["died"] for r in episode_results[half:]])

    if verbose:
        print(f"  ---")
        print(f"  Best: {best_levels}/{win_levels} levels")
        print(f"  Avg steps: {avg_steps:.0f}, Survival: {survival_rate:.0%}")
        print(
            f"  Learning: levels {early_levels:.1f} -> {late_levels:.1f}, "
            f"survival {early_survival:.0%} -> {late_survival:.0%}"
        )

    return {
        "game_id": game_id,
        "win_levels": win_levels,
        "best_levels": best_levels,
        "avg_steps": float(avg_steps),
        "survival_rate": float(survival_rate),
        "early_levels": float(early_levels),
        "late_levels": float(late_levels),
        "early_survival": float(early_survival),
        "late_survival": float(late_survival),
        "episode_levels": levels_by_episode,
    }


def get_keyboard_only_games(arcade: arc_agi.Arcade) -> list[str]:
    """Return game IDs that only use keyboard actions (1-5)."""
    game_ids = []
    for e in arcade.get_environments():
        gid = e.game_id.split("-")[0]
        env = arcade.make(gid)
        f = env.reset()
        if all(a <= 5 for a in f.available_actions):
            game_ids.append(gid)
    return game_ids


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 baseline with Arbor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game", type=str, help="Run on a specific game ID")
    group.add_argument("--keyboard-only", action="store_true", help="Keyboard-only games")
    group.add_argument("--all", action="store_true", help="All public games")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per game")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--share-weights", action="store_true", help="Share weights across games")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    arcade = arc_agi.Arcade()

    if args.game:
        game_ids = [args.game]
    elif args.keyboard_only:
        game_ids = get_keyboard_only_games(arcade)
    else:
        game_ids = [e.game_id.split("-")[0] for e in arcade.get_environments()]

    if verbose:
        print(f"Training {len(game_ids)} games, {args.episodes} episodes each")
        if args.share_weights:
            print("  (sharing weights across games)")
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
            print(f"[{i+1}/{len(game_ids)}] {game_id}")
        result = train_game(
            game_id, arcade,
            agent=shared_agent if args.share_weights else None,
            encoder=shared_encoder if args.share_weights else None,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=verbose,
        )
        results.append(result)
        if verbose:
            print()

    elapsed = time.time() - t0

    # Overall summary
    print("=" * 60)
    print(f"Results: {len(results)} games x {args.episodes} episodes, {elapsed:.1f}s")
    print()
    print(f"{'Game':8s} {'Best':>5s} {'Surv':>6s} {'Early→Late Levels':>20s} {'Early→Late Surv':>18s}")
    print("-" * 60)
    for r in results:
        marker = "*" if r["best_levels"] > 0 else " "
        print(
            f"{marker}{r['game_id']:7s} "
            f"{r['best_levels']:2d}/{r['win_levels']:<2d} "
            f"{r['survival_rate']:5.0%}  "
            f"{r['early_levels']:4.1f} -> {r['late_levels']:4.1f}          "
            f"{r['early_survival']:4.0%} -> {r['late_survival']:4.0%}"
        )

    total_best = sum(r["best_levels"] for r in results)
    total_possible = sum(r["win_levels"] for r in results)
    any_learning = sum(1 for r in results if r["late_levels"] > r["early_levels"])
    print("-" * 60)
    print(f" Total best: {total_best}/{total_possible} levels")
    print(f" Games with improving levels: {any_learning}/{len(results)}")


if __name__ == "__main__":
    main()
