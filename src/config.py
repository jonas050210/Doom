"""Configuration: scenario metadata, validated configs, and checkpoint sidecars.

Design notes
------------
* :class:`ScenarioSpec` centralises what each ViZDoom scenario exposes
  (episode timeout, reward shape, what "success" means).  Values are asserted
  against the shipped ``.cfg`` files in ``tests/test_config.py``.
* :class:`EnvConfig` validates itself, can be serialised to JSON, and is saved
  next to every checkpoint so that ``evaluate``/``play`` reproduce the exact
  observation and action space that a model was trained with.  This is what
  prevents the classic "trained with ``--low-ram``, evaluated without it"
  mismatch.
* :func:`build_configs` resolves CLI overrides against presets so that
  ``--low-ram`` only changes defaults and every explicitly passed flag wins.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    """Static description of a ViZDoom scenario shipped with the package."""

    name: str
    cfg_file: str
    combat: bool
    description: str
    success_signal: str
    has_goal: bool
    episode_timeout_ticks: int
    reward_range: tuple[float, float]

    def summary(self) -> str:
        """Multi-line human readable summary for CLI banners."""
        low, high = self.reward_range
        return (
            f"{self.name}: {self.description}\n"
            f"  success: {self.success_signal}\n"
            f"  reward per episode: roughly [{low:g}, {high:g}]\n"
            f"  episode timeout: {self.episode_timeout_ticks} tics"
        )


SCENARIOS: dict[str, ScenarioSpec] = {
    "my_way_home": ScenarioSpec(
        name="my_way_home",
        cfg_file="my_way_home.cfg",
        combat=False,
        description=("Level 1 (Navigation): find the goal vest in a maze of coloured rooms."),
        success_signal="reached the goal vest (+1); timeouts are not successes",
        episode_timeout_ticks=2100,
        has_goal=True,
        reward_range=(-0.25, 1.0),
    ),
    "health_gathering": ScenarioSpec(
        name="health_gathering",
        cfg_file="health_gathering.cfg",
        combat=False,
        description=("Level 2 (Survival): collect medkits to survive on a damaging floor."),
        success_signal=("no goal exists; the score is survival time (+1/tic, -100 on death)"),
        episode_timeout_ticks=2100,
        has_goal=False,
        reward_range=(-105.0, 2105.0),
    ),
    "basic": ScenarioSpec(
        name="basic",
        cfg_file="basic.cfg",
        combat=True,
        description=("Level 3 (Combat intro): strafe and shoot a single monster in a room."),
        success_signal="killed the monster (+106); -1/tic and -5/shot otherwise",
        episode_timeout_ticks=300,
        has_goal=True,
        reward_range=(-320.0, 112.0),
    ),
    "defend_the_center": ScenarioSpec(
        name="defend_the_center",
        cfg_file="defend_the_center.cfg",
        combat=True,
        description=("Level 4 (Arena combat): turn and shoot monsters closing in from all sides."),
        success_signal="each kill is +1 and the episode always ends in death (-1)",
        episode_timeout_ticks=2100,
        has_goal=False,
        reward_range=(-2.0, 30.0),
    ),
    "deadly_corridor": ScenarioSpec(
        name="deadly_corridor",
        cfg_file="deadly_corridor.cfg",
        combat=True,
        description=("Level 5 (Hard combat): fight down a corridor of monsters to reach the vest."),
        success_signal="reached the vest (+1000); death is -100 and shaping is +/-dX",
        episode_timeout_ticks=2100,
        has_goal=True,
        reward_range=(-1100.0, 1010.0),
    ),
}


def resolve_scenario(name: str) -> ScenarioSpec:
    """Look up a scenario by name.

    Raises:
        ValueError: If the scenario is unknown (message lists valid names).
    """
    try:
        return SCENARIOS[name]
    except KeyError:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario {name!r}. Available: {available}") from None


def scenario_names() -> list[str]:
    """Return sorted scenario names (used for argparse choices/help)."""
    return sorted(SCENARIOS)


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        scenario: Name of a scenario in :data:`SCENARIOS`.
        frame_width/frame_height: Observation size after resizing.
        grayscale: ``True`` for 1-channel GRAY8 frames, ``False`` for RGB24.
        frame_skip: Tics executed per agent step (ViZDoom ``make_action`` skip).
        frame_stack: Number of consecutive frames stacked along the channel axis.
        window_visible: Open the ViZDoom window (requires a display).
        seed: Default seed used when ``reset()`` is called without one.
    """

    scenario: str = "my_way_home"
    frame_width: int = 84
    frame_height: int = 84
    grayscale: bool = True
    frame_skip: int = 4
    frame_stack: int = 4
    window_visible: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        resolve_scenario(self.scenario)
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

    @property
    def channels_per_frame(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of a single observation: ``(C * frame_stack, H, W)``."""
        return (
            self.channels_per_frame * self.frame_stack,
            self.frame_height,
            self.frame_width,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvConfig:
        """Rebuild a config from a mapping, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            warnings.warn(f"Ignoring unknown EnvConfig keys: {sorted(unknown)}", stacklevel=2)
        return cls(**{key: data[key] for key in data if key in known})

    def save(self, path: str) -> None:
        """Write the config as JSON (used for checkpoint sidecars)."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str) -> EnvConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def env_config_path_for_model(model_path: str) -> str:
    """Return the sidecar JSON path for a checkpoint (``model.zip.env.json``)."""
    return f"{model_path}.env.json"


def save_env_config_for_model(config: EnvConfig, model_path: str) -> str:
    """Persist ``config`` next to a checkpoint so it can be reproduced later."""
    path = env_config_path_for_model(model_path)
    config.save(path)
    return path


def load_env_config_for_model(model_path: str) -> EnvConfig | None:
    """Load the sidecar written by :func:`save_env_config_for_model`, if any."""
    path = env_config_path_for_model(model_path)
    if not os.path.exists(path):
        return None
    return EnvConfig.load(path)


@dataclass
class TrainConfig:
    """PPO training configuration with validated defaults for 8 GB VRAM."""

    total_timesteps: int = 100_000
    n_envs: int = 1
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 3
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be >= 1")
        if self.checkpoint_freq < 1 or self.eval_freq < 1:
            raise ValueError("checkpoint_freq and eval_freq must be >= 1")

        rollout_size = self.n_steps * self.n_envs
        if self.batch_size > rollout_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= n_steps * n_envs ({rollout_size})"
            )
        if rollout_size % self.batch_size != 0:
            warnings.warn(
                f"n_steps * n_envs ({rollout_size}) is not divisible by "
                f"batch_size ({self.batch_size}); the last minibatch will be smaller.",
                stacklevel=2,
            )

    @property
    def rollout_size(self) -> int:
        return self.n_steps * self.n_envs


def get_low_ram_configs(
    scenario: str = "my_way_home", seed: int = 42
) -> tuple[EnvConfig, TrainConfig]:
    """Return memory-minimised presets for 2 GB RAM / integrated-GPU machines."""
    env_cfg = EnvConfig(
        scenario=scenario,
        frame_stack=1,
        window_visible=False,
        seed=seed,
    )
    train_cfg = TrainConfig(
        total_timesteps=2_048,
        n_envs=1,
        n_steps=256,
        batch_size=32,
        n_epochs=2,
        checkpoint_freq=1_024,
        eval_freq=1_024,
        device="cpu",
    )
    return env_cfg, train_cfg


def build_configs(
    scenario: str,
    *,
    low_ram: bool = False,
    timesteps: int | None = None,
    seed: int = 42,
    frame_stack: int | None = None,
    frame_skip: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_envs: int | None = None,
    ent_coef: float | None = None,
    checkpoint_freq: int | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    device: str | None = None,
    save_dir: str = "checkpoints",
    log_dir: str = "runs",
    window_visible: bool = False,
) -> tuple[EnvConfig, TrainConfig]:
    """Merge presets with explicit CLI overrides.

    ``low_ram`` selects the memory-minimised preset; every non-``None`` keyword
    argument then overrides it.  This makes ``--low-ram --batch-size 128``
    behave the way a caller expects instead of silently ignoring the flag.
    """
    resolve_scenario(scenario)

    if low_ram:
        env_cfg, train_cfg = get_low_ram_configs(scenario=scenario, seed=seed)
    else:
        env_cfg = EnvConfig(scenario=scenario, seed=seed)
        train_cfg = TrainConfig()

    env_cfg.window_visible = window_visible or env_cfg.window_visible

    if timesteps is not None:
        train_cfg.total_timesteps = timesteps
    if frame_stack is not None:
        env_cfg.frame_stack = frame_stack
    if frame_skip is not None:
        env_cfg.frame_skip = frame_skip
    if n_steps is not None:
        train_cfg.n_steps = n_steps
    if batch_size is not None:
        train_cfg.batch_size = batch_size
    if n_envs is not None:
        train_cfg.n_envs = n_envs
    if ent_coef is not None:
        train_cfg.ent_coef = ent_coef
    if checkpoint_freq is not None:
        train_cfg.checkpoint_freq = checkpoint_freq
    if eval_freq is not None:
        train_cfg.eval_freq = eval_freq
    if n_eval_episodes is not None:
        train_cfg.n_eval_episodes = n_eval_episodes
    if device is not None:
        train_cfg.device = device

    train_cfg.save_dir = save_dir
    train_cfg.log_dir = log_dir

    # Re-run dataclass validation after mutation (e.g. batch_size vs n_steps).
    env_cfg = EnvConfig(**env_cfg.to_dict())
    train_cfg = TrainConfig(**{f.name: getattr(train_cfg, f.name) for f in fields(TrainConfig)})
    return env_cfg, train_cfg


def resolve_checkpoint_env_config(
    checkpoint_path: str,
    *,
    scenario: str | None = None,
    seed: int = 100,
    frame_stack: int | None = None,
    window_visible: bool = False,
) -> EnvConfig:
    """Return the environment config that reproduces ``checkpoint_path``.

    The ``.env.json`` sidecar written during training is authoritative; CLI
    values are only a fallback for checkpoints that predate sidecars.

    Raises:
        SystemExit: If an explicit ``scenario`` contradicts the sidecar.
    """
    stored = load_env_config_for_model(checkpoint_path)
    if stored is not None:
        if scenario is not None and scenario != stored.scenario:
            raise SystemExit(
                f"Checkpoint was trained on {stored.scenario!r} but "
                f"scenario={scenario!r} was requested. Omit the override or pass "
                f"--scenario {stored.scenario}."
            )
        if frame_stack is not None and frame_stack != stored.frame_stack:
            print(
                f"[Config] Note: frame_stack={frame_stack} ignored; the checkpoint "
                f"was trained with frame_stack={stored.frame_stack}."
            )
        stored.window_visible = window_visible
        return stored

    print(
        "[Config] Warning: no .env.json sidecar next to the checkpoint; falling "
        "back to CLI flags. A space mismatch will be reported if they differ."
    )
    return EnvConfig(
        scenario=scenario or "my_way_home",
        frame_stack=4 if frame_stack is None else frame_stack,
        seed=seed,
        window_visible=window_visible,
    )
