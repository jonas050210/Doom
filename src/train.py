"""Train a PPO agent on a free ViZDoom scenario."""

import argparse
from datetime import datetime
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import EnvConfig, SCENARIOS, TrainConfig, get_low_ram_configs
from src.env import VizDoomEnv
from src.model import create_ppo_model, resolve_device


class BestModelCallback(BaseCallback):
    """Save only the best model (by rolling mean episode reward) to save disk space."""

    def __init__(self, best_path: str, check_freq: int = 1024, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.best_path = best_path
        self.check_freq = max(1, check_freq)
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(
            self.model.ep_info_buffer
        )
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.best_path)
            if self.verbose > 0:
                print(
                    f"[BestModel] New best mean reward: {mean_reward:.3f} -> saved {self.best_path}"
                )
        return True


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
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help="Use low-RAM preset (frame_stack=1, n_steps=256, batch_size=32, CPU) for 2 GB RAM laptops.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the ViZDoom window during training so you can watch the agent learn live.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start training from scratch instead of resuming from the last checkpoint.",
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

    if args.low_ram:
        env_cfg, train_cfg = get_low_ram_configs(scenario=args.scenario, seed=args.seed)
        env_cfg.window_visible = args.render
        train_cfg.total_timesteps = args.timesteps if args.timesteps != 100_000 else train_cfg.total_timesteps
        train_cfg.save_dir = run_ckpt_dir
        train_cfg.log_dir = run_log_dir
    else:
        env_cfg = EnvConfig(
            scenario=args.scenario,
            include_use_attack=bool(SCENARIOS[args.scenario].get("combat", False)),
            seed=args.seed,
            window_visible=args.render,
        )
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

    final_path = os.path.join(args.save_dir, f"{args.scenario}_final.zip")
    best_path = os.path.join(args.save_dir, f"{args.scenario}_best.zip")
    resuming = os.path.exists(final_path) and not args.fresh

    if resuming:
        device = resolve_device(train_cfg.device)
        print(f"[Resume] Loading existing checkpoint: {final_path}")
        model = PPO.load(final_path, env=vec_env, device=device)
    else:
        model = create_ppo_model(
            env=vec_env,
            train_config=train_cfg,
            seed=args.seed,
            tensorboard_log=run_log_dir,
        )

    logger = configure(run_log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    best_cb = BestModelCallback(
        best_path=best_path,
        check_freq=max(1, args.checkpoint_freq),
    )

    print(f"Starting PPO training on '{args.scenario}' for {args.timesteps} timesteps...")
    print(f"Logs:       {run_log_dir}")
    print(f"Best model: {best_path}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=best_cb,
            reset_num_timesteps=not resuming,
        )
        model.save(final_path)
        print(f"\nTraining complete. Final model saved to: {final_path}")
    except KeyboardInterrupt:
        model.save(final_path)
        print(f"\n[Interrupted] Progress saved to: {final_path}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
