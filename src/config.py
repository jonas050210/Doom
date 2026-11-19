"""Configuration presets and defaults for NeuralDoom-Lab.

Designed for Ubuntu Linux with an NVIDIA RTX 4060 Ti (8 GB VRAM) and 32 GB RAM,
with automatic CPU fallback for smoke tests and headless environments.
"""

from dataclasses import dataclass
from typing import Dict


SCENARIOS: Dict[str, Dict[str, str]] = {
    "my_way_home": {
        "cfg": "my_way_home.cfg",
        "description": (
            "Free maze/navigation scenario: reach the goal vest in a series of "
            "rooms with a small time penalty per step."
        ),
    },
    "health_gathering": {
        "cfg": "health_gathering.cfg",
        "description": (
            "Free survival/navigation scenario: collect health packs to survive "
            "as long as possible on a damaging floor."
        ),
    },
}


@dataclass
class EnvConfig:
    """Environment configuration."""

    scenario: str = "my_way_home"
    frame_width: int = 84
    frame_height: int = 84
    grayscale: bool = True
    frame_skip: int = 4
    frame_stack: int = 4
    include_use_attack: bool = False
    window_visible: bool = False
    seed: int = 42


@dataclass
class TrainConfig:
    """PPO training configuration (safe defaults for 8 GB VRAM)."""

    total_timesteps: int = 100_000
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    checkpoint_freq: int = 10_000
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "auto"


def get_low_ram_configs(scenario: str = "my_way_home", seed: int = 42) -> tuple[EnvConfig, TrainConfig]:
    """Return memory-minimized configs for 2 GB RAM laptops (e.g. i3-1215U)."""
    env_cfg = EnvConfig(
        scenario=scenario,
        frame_stack=1,
        window_visible=False,
        seed=seed,
    )
    train_cfg = TrainConfig(
        total_timesteps=2_048,
        n_steps=256,
        batch_size=32,
        n_epochs=2,
        checkpoint_freq=1_024,
        device="cpu",
    )
    return env_cfg, train_cfg
