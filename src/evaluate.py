"""Deterministic evaluation of a trained PPO checkpoint on ViZDoom.

The environment configuration is read from the ``.env.json`` sidecar written by
``src.train`` next to the checkpoint, so evaluation always reproduces the exact
observation and action space the model was trained with.  Results are printed
and written to a JSON file for later comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np

from src.config import SCENARIOS, EnvConfig, resolve_checkpoint_env_config
from src.env import VizDoomEnv, has_display
from src.model import load_ppo_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained .zip checkpoint.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=sorted(SCENARIOS),
        help="Scenario to evaluate on (default: taken from the checkpoint sidecar).",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=100, help="Base seed for evaluation episodes.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device to use.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="Only used when the checkpoint has no .env.json sidecar.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: <checkpoint>.eval.json).",
    )
    parser.add_argument(
        "--render", action="store_true", help="Show the game window while evaluating."
    )
    parser.add_argument(
        "--engine-log",
        action="store_true",
        help="Show the ViZDoom engine log (hidden by default).",
    )
    return parser.parse_args(argv)


def build_eval_env(
    checkpoint: str,
    scenario: str | None = None,
    seed: int = 100,
    frame_stack: int | None = None,
    window_visible: bool = False,
    silence_engine: bool | None = None,
) -> EnvConfig:
    """Return the env config that matches ``checkpoint`` (see config module)."""
    return resolve_checkpoint_env_config(
        checkpoint,
        scenario=scenario,
        seed=seed,
        frame_stack=frame_stack,
        window_visible=window_visible,
        silence_engine=silence_engine,
    )


def summarize_results(
    rewards: Sequence[float],
    lengths: Sequence[int],
    outcomes: Sequence[str | None],
) -> dict[str, Any]:
    """Aggregate per-episode results into a JSON-serialisable summary."""
    if not rewards:
        raise ValueError("Cannot summarise zero episodes")

    n = len(rewards)
    counts: dict[str, int] = {"goal": 0, "death": 0, "timeout": 0, "unknown": 0}
    for outcome in outcomes:
        counts[outcome if outcome in ("goal", "death", "timeout") else "unknown"] += 1

    successes = counts["goal"]
    return {
        "episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "goal_rate": 100.0 * successes / n,
        "death_rate": 100.0 * counts["death"] / n,
        "timeout_rate": 100.0 * counts["timeout"] / n,
        "outcome_counts": counts,
    }


def evaluate(
    checkpoint_path: str,
    scenario: str | None = None,
    episodes: int = 10,
    seed: int = 100,
    device: str = "auto",
    frame_stack: int | None = None,
    window_visible: bool = False,
    output: str | None = None,
    silence_engine: bool | None = None,
) -> dict[str, Any]:
    """Run deterministic episodes and return a result summary.

    Raises:
        ValueError: If ``episodes`` is not positive.
        SystemExit: If the checkpoint and environment spaces disagree.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    env_cfg = build_eval_env(
        checkpoint_path,
        scenario=scenario,
        seed=seed,
        frame_stack=frame_stack,
        window_visible=window_visible,
        silence_engine=silence_engine,
    )
    env = VizDoomEnv(config=env_cfg)
    try:
        model = load_ppo_model(checkpoint_path, device=device)

        if model.action_space != env.action_space:
            raise SystemExit(
                f"Action space mismatch: checkpoint {model.action_space} vs "
                f"environment {env.action_space}. The checkpoint was trained "
                "with a different scenario/action catalogue."
            )
        if model.observation_space.shape != env.observation_space.shape:
            raise SystemExit(
                f"Observation shape mismatch: checkpoint "
                f"{model.observation_space.shape} vs environment "
                f"{env.observation_space.shape}. Pass --frame-stack to match."
            )

        print("=" * 70)
        print(SCENARIOS[env_cfg.scenario].summary())
        print("=" * 70)

        rewards: list[float] = []
        lengths: list[int] = []
        outcomes: list[str | None] = []
        records: list[dict[str, Any]] = []

        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            done = False
            ep_reward = 0.0
            ep_len = 0
            outcome: str | None = None

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated
                outcome = info.get("episode_outcome")

            rewards.append(ep_reward)
            lengths.append(ep_len)
            outcomes.append(outcome)
            records.append(
                {
                    "episode": episode + 1,
                    "seed": seed + episode,
                    "reward": ep_reward,
                    "length": ep_len,
                    "outcome": outcome,
                    "ticks": info.get("episode_time_ticks"),
                }
            )
            print(
                f"Episode {episode + 1:3d}/{episodes}: "
                f"reward={ep_reward:9.3f} | steps={ep_len:4d} | "
                f"outcome={outcome}"
            )
    finally:
        env.close()

    summary = summarize_results(rewards, lengths, outcomes)
    print("\n=== Evaluation Summary ===")
    print(f"Scenario:           {env_cfg.scenario}")
    print(f"Episodes:           {summary['episodes']}")
    print(
        f"Mean reward:        {summary['mean_reward']:.3f} "
        f"+/- {summary['std_reward']:.3f} "
        f"(min {summary['min_reward']:.3f}, max {summary['max_reward']:.3f})"
    )
    if SCENARIOS[env_cfg.scenario].has_goal:
        print(f"Goal rate:          {summary['goal_rate']:.1f}%")
    else:
        print("Goal rate:          n/a (this scenario has no goal state)")
    print(f"Death rate:         {summary['death_rate']:.1f}%")
    print(f"Timeout rate:       {summary['timeout_rate']:.1f}%")
    print(f"Mean episode length:{summary['mean_length']:.1f} steps")
    print(f"Success definition: {SCENARIOS[env_cfg.scenario].success_signal}")

    payload: dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "scenario": env_cfg.scenario,
        "has_goal": SCENARIOS[env_cfg.scenario].has_goal,
        "success_signal": SCENARIOS[env_cfg.scenario].success_signal,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "env_config": env_cfg.to_dict(),
        "summary": summary,
        "episodes": records,
    }

    output_path = output or f"{os.path.splitext(checkpoint_path)[0]}.eval.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Results written to: {output_path}")

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render and not has_display():
        print(
            "Error: --render needs a display, but no DISPLAY/WAYLAND_DISPLAY was found.",
        )
        return 2
    try:
        evaluate(
            checkpoint_path=args.checkpoint,
            scenario=args.scenario,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            frame_stack=args.frame_stack,
            window_visible=args.render,
            output=args.output,
            silence_engine=not args.engine_log,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
