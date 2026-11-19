"""Watch a trained PPO agent (or random agent) play ViZDoom.

Uses a visible ViZDoom window when a display is available, and automatically
falls back to headless mode when running without an X11/Wayland display.
"""

import argparse
import os
import time
from typing import Optional

from src.config import EnvConfig, SCENARIOS
from src.env import VizDoomEnv
from src.model import load_ppo_model


def has_display() -> bool:
    """Return True if an X11 or Wayland display environment variable is set."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play ViZDoom with a trained agent or random policy.")
    parser.add_argument(
        "--checkpoint",
        type=str,
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
        type=str,
        default="my_way_home",
        choices=sorted(SCENARIOS.keys()),
        help="ViZDoom scenario to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode even if a display is available.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep between steps when a window is visible.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.random and not args.checkpoint:
        raise SystemExit("Error: provide --checkpoint <path> or --random.")

    display_available = has_display()
    window_visible = display_available and not args.headless

    if not window_visible:
        reason = "--headless requested" if args.headless else "no DISPLAY/WAYLAND_DISPLAY detected"
        print(f"[Display] Running in headless fallback mode ({reason}).")
    else:
        print("[Display] Display detected; opening ViZDoom window.")

    model: Optional[object] = None
    if not args.random and args.checkpoint:
        model = load_ppo_model(args.checkpoint, device=args.device)

    env = VizDoomEnv(
        EnvConfig(
            scenario=args.scenario,
            seed=args.seed,
            window_visible=window_visible,
        )
    )

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                if args.random or model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated
                if window_visible and args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"Episode {ep + 1}/{args.episodes}: "
                f"reward={ep_reward:.3f} | steps={ep_steps} | success={info.get('is_success', False)}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
