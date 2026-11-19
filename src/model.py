"""PPO model creation, device selection, and checkpoint loading."""

from __future__ import annotations

import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from src.config import TrainConfig

VALID_DEVICES = ("auto", "cuda", "cpu")


def resolve_device(requested: str = "auto") -> str:
    """Resolve the PyTorch device (``cuda`` when available, else ``cpu``).

    Raises:
        ValueError: If ``requested`` is not one of ``auto``/``cuda``/``cpu``.
    """
    requested = requested.lower()
    if requested not in VALID_DEVICES:
        raise ValueError(f"Unknown device {requested!r}. Choose one of {list(VALID_DEVICES)}.")

    if requested == "auto":
        if torch.cuda.is_available():
            print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        print("[Device] CUDA not available; falling back to CPU.")
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("[Device] Warning: CUDA requested but not available. Using CPU.")
        return "cpu"

    print(f"[Device] Using requested device: {requested}")
    return requested


def create_ppo_model(
    env: VecEnv,
    train_config: TrainConfig | None = None,
    seed: int = 42,
    tensorboard_log: str | None = None,
    verbose: int = 1,
) -> PPO:
    """Create a Stable-Baselines3 PPO agent with ``CnnPolicy``.

    Args:
        env: Vectorised environment whose observation space is channels-first
            ``uint8`` image data (see :class:`src.env.VizDoomEnv`).
        train_config: Hyper-parameters; defaults to :class:`TrainConfig`.
        seed: Seed forwarded to PPO (torch/numpy seeding inside SB3).
        tensorboard_log: Directory for TensorBoard events.
        verbose: SB3 verbosity level.
    """
    if not isinstance(env, VecEnv):
        raise TypeError(
            "create_ppo_model expects a stable_baselines3 VecEnv; "
            f"got {type(env).__name__}. Wrap it in DummyVecEnv/SubprocVecEnv."
        )

    cfg = train_config or TrainConfig()
    device = resolve_device(cfg.device)

    return PPO(
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
        verbose=verbose,
    )


def load_ppo_model(checkpoint_path: str, device: str = "auto") -> PPO:
    """Load a trained PPO checkpoint from disk.

    Raises:
        FileNotFoundError: If the checkpoint does not exist (with a hint about
            Stable-Baselines3 file naming).
    """
    if not os.path.exists(checkpoint_path):
        hint = ""
        if not checkpoint_path.endswith(".zip"):
            hint = " Stable-Baselines3 appends '.zip'; try adding it."
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.{hint}")

    resolved_device = resolve_device(device)
    return PPO.load(checkpoint_path, device=resolved_device)
