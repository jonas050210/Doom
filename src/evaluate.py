"""Deterministic evaluation of a trained PPO checkpoint on ViZDoom."""

import argparse
from typing import List

import numpy as np

from src.config import EnvConfig, SCENARIOS
from src.env import VizDoomEnv
from src.model import load_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained .zip checkpoint.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="my_way_home",
        choices=sorted(SCENARIOS.keys()),
        help="ViZDoom scenario to evaluate on.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of deterministic evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device to use.",
    )
    return parser.parse_args()


def evaluate(
    checkpoint_path: str,
    scenario: str = "my_way_home",
    episodes: int = 10,
    seed: int = 100,
    device: str = "auto",
) -> None:
    model = load_ppo_model(checkpoint_path, device=device)
    env = VizDoomEnv(EnvConfig(scenario=scenario, seed=seed, window_visible=False))

    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_success = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                ep_len += 1
                if info.get("is_success", False):
                    ep_success = True
                done = terminated or truncated

            rewards.append(ep_reward)
            lengths.append(ep_len)
            successes.append(ep_success)
            print(
                f"Episode {ep + 1:2d}/{episodes}: "
                f"reward={ep_reward:7.3f} | length={ep_len:4d} | success={ep_success}"
            )
    finally:
        env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    completion_rate = 100.0 * float(np.mean(successes))
    avg_length = float(np.mean(lengths))

    print("\n=== Evaluation Summary ===")
    print(f"Scenario:               {scenario}")
    print(f"Episodes:               {episodes}")
    print(f"Mean Episode Reward:    {mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"Completion Rate:        {completion_rate:.1f}%")
    print(f"Average Episode Length: {avg_length:.1f} steps")


def main() -> None:
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
