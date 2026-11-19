"""PPO model creation, device selection, and checkpoint loading."""

from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import torch

from src.config import TrainConfig


def resolve_device(requested: str = "auto") -> str:
    """Resolve the PyTorch device ('cuda' when available, else 'cpu')."""
    if requested == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] Using CUDA GPU: {gpu_name}")
            return "cuda"
        print("[Device] CUDA not available; falling back to CPU.")
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("[Device] Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    print(f"[Device] Using requested device: {requested}")
    return requested


def create_ppo_model(
    env: VecEnv,
    train_config: Optional[TrainConfig] = None,
    seed: int = 42,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Create a Stable-Baselines3 PPO agent with CnnPolicy."""
    cfg = train_config or TrainConfig()
    device = resolve_device(cfg.device)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=1,
    )
    return model


def load_ppo_model(checkpoint_path: str, device: str = "auto") -> PPO:
    """Load a trained PPO checkpoint from disk."""
    resolved_device = resolve_device(device)
    return PPO.load(checkpoint_path, device=resolved_device)
