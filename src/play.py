"""Watch a trained PPO agent (or a random agent) play ViZDoom.

Uses a visible ViZDoom window when a display is available and falls back to
headless mode automatically.  The environment configuration is taken from the
checkpoint's ``.env.json`` sidecar so the agent always sees the observations it
was trained on.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from stable_baselines3 import PPO

from src.config import SCENARIOS, EnvConfig, resolve_checkpoint_env_config
from src.env import VizDoomEnv, has_display
from src.model import load_ppo_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play ViZDoom with a trained agent or random policy."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the trained .zip checkpoint (omit when using --random).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Run a random-agent smoke test instead of loading a checkpoint.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=sorted(SCENARIOS),
        help="Scenario to run (default: taken from the checkpoint sidecar).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode even if a display is available.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="Only used when the checkpoint has no .env.json sidecar.",
    )
    parser.add_argument(
        "--engine-log",
        action="store_true",
        help="Show the ViZDoom engine log (hidden by default).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep between steps when a window is visible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device to use.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.random and not args.checkpoint:
        print("Error: provide --checkpoint <path> or --random.")
        return 2

    display_available = has_display()
    window_visible = display_available and not args.headless
    if not window_visible:
        reason = "--headless requested" if args.headless else "no DISPLAY/WAYLAND_DISPLAY detected"
        print(f"[Display] Running headless ({reason}).")
    else:
        print("[Display] Display detected; opening the ViZDoom window.")

    model: PPO | None = None
    if args.checkpoint and not args.random:
        try:
            env_cfg: EnvConfig = resolve_checkpoint_env_config(
                args.checkpoint,
                scenario=args.scenario,
                seed=args.seed,
                frame_stack=args.frame_stack,
                window_visible=window_visible,
                silence_engine=not args.engine_log,
            )
            model = load_ppo_model(args.checkpoint, device=args.device)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
    else:
        env_cfg = EnvConfig(
            scenario=args.scenario or "my_way_home",
            frame_stack=4 if args.frame_stack is None else args.frame_stack,
            seed=args.seed,
            window_visible=window_visible,
            silence_engine=not args.engine_log,
        )

    env = VizDoomEnv(config=env_cfg)
    try:
        if model is not None and model.action_space != env.action_space:
            print(
                f"Error: action space mismatch: checkpoint {model.action_space} vs "
                f"environment {env.action_space}."
            )
            return 2

        print("=" * 70)
        print(SCENARIOS[env_cfg.scenario].summary())
        print(
            f"Policy: {'random' if model is None else args.checkpoint} | "
            f"actions: {', '.join(env.action_names)}"
        )
        print("=" * 70)

        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            outcome: str | None = None

            while not done:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated
                outcome = info.get("episode_outcome")
                if window_visible and args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"Episode {episode + 1}/{args.episodes}: "
                f"reward={ep_reward:9.3f} | steps={ep_steps:4d} | outcome={outcome}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
