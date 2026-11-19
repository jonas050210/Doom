"""PPO model creation, device selection, and checkpoint loading.

GPU notes (Ubuntu primary, Windows supported)
---------------------------------------------
* ``device="auto"`` (the default) uses CUDA when ``torch.cuda.is_available()``
  is true — i.e. on the Ubuntu workstation with an NVIDIA GPU and a
  CUDA-enabled PyTorch install — and falls back to CPU on the Windows laptop
  and in CI.  No flag is needed to "unlock" the GPU; just install the
  CUDA-enabled ``torch`` wheel (available from PyTorch for Linux/Windows) and
  make sure the NVIDIA driver is installed (``nvidia-smi`` works).
* ``--low-ram`` forces ``device="cpu"`` for the 2 GB laptop.  Every other run
  defaults to ``auto``.
* ViZDoom environments always run on CPU; only the PPO network/optimiser
  uses the GPU.  ``--n-envs 4`` therefore scales CPU rendering while keeping
  gradient computation on the GPU.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from src.config import TrainConfig

VALID_DEVICES = ("auto", "cuda", "cpu")


def gpu_info() -> dict[str, Any]:
    """Return a JSON-serialisable summary of the CUDA environment.

    Safe to call on CPU-only machines (reports ``available=False``).  Used in
    startup banners and ``run.json`` so a run log always records whether the
    GPU was actually used.
    """
    info: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "device_count": 0,
        "device_name": None,
        "total_memory_mb": None,
    }
    if torch.cuda.is_available():
        try:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["total_memory_mb"] = int(props.total_memory // (1024 * 1024))
        except Exception:  # pragma: no cover - defensive; driver quirks
            pass
    return info


def resolve_device(requested: str = "auto") -> str:
    """Resolve the PyTorch device (``cuda`` when available, else ``cpu``).

    Accepts ``auto``/``cuda``/``cpu`` (case-insensitive) and ``cuda:N`` for a
    specific GPU.  ``cuda`` falls back to ``cpu`` with a warning when CUDA is
    unavailable, so the same command works on the Ubuntu GPU workstation and
    the Windows CPU laptop.

    Raises:
        ValueError: If ``requested`` is not a recognised device.
    """
    requested = requested.strip().lower()
    if requested not in VALID_DEVICES and not requested.startswith("cuda:"):
        raise ValueError(f"Unknown device {requested!r}. Choose one of {list(VALID_DEVICES)}.")

    if requested == "auto":
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover - defensive
                name = "unknown GPU"
            print(f"[Device] Using CUDA GPU: {name}")
            return "cuda"
        print("[Device] CUDA not available; falling back to CPU.")
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
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
    try:
        return PPO.load(checkpoint_path, device=resolved_device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path!r} on device "
            f"{resolved_device!r}: {exc}. Check that the checkpoint was "
            "trained with a compatible stable-baselines3/torch version "
            "(see requirements.txt upper bounds) and that the file is not "
            "truncated."
        ) from exc
