"""Train a PPO agent on a free ViZDoom scenario."""

import argparse
from datetime import datetime
import os

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import EnvConfig, SCENARIOS, TrainConfig
from src.env import VizDoomEnv
from src.model import create_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on ViZDoom.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="my_way_home",
        choices=sorted(SCENARIOS.keys()),
        help="ViZDoom scenario to train on.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device to use.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Rollout steps per PPO update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard and CSV logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.scenario}_{timestamp}"

    run_log_dir = os.path.join(args.log_dir, run_name)
    run_ckpt_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    env_cfg = EnvConfig(scenario=args.scenario, seed=args.seed, window_visible=False)
    train_cfg = TrainConfig(
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        save_dir=run_ckpt_dir,
        log_dir=run_log_dir,
        device=args.device,
    )

    def _init_env():
        env = VizDoomEnv(config=env_cfg)
        return Monitor(env, filename=os.path.join(run_log_dir, "monitor.csv"))

    vec_env = DummyVecEnv([_init_env])

    model = create_ppo_model(
        env=vec_env,
        train_config=train_cfg,
        seed=args.seed,
        tensorboard_log=run_log_dir,
    )
    logger = configure(run_log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=run_ckpt_dir,
        name_prefix=f"ppo_{args.scenario}",
    )

    print(f"Starting PPO training on '{args.scenario}' for {args.timesteps} timesteps...")
    print(f"Logs:        {run_log_dir}")
    print(f"Checkpoints: {run_ckpt_dir}")

    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
        final_path = os.path.join(args.save_dir, f"{args.scenario}_final.zip")
        model.save(final_path)
        print(f"\nTraining complete. Final model saved to: {final_path}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
