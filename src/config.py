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
* :class:`TrainConfig` is persisted next to every checkpoint as
  ``<model>.train.json`` so ``--resume`` can warn when hyper-parameters differ
  from the original run instead of silently drifting.
* :func:`build_configs` resolves CLI overrides against presets so that
  ``--low-ram`` only changes defaults and every explicitly passed flag wins.
* All sidecar/JSON writes are atomic (temp file + ``os.replace``) so a crash
  mid-write never leaves a truncated sidecar behind.  This works identically
  on Windows and Linux.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

VALID_DEVICES = ("auto", "cuda", "cpu")


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (cross-platform).

    The data is written to a temporary file in the same directory and then
    moved over ``path`` with :func:`os.replace`, which is atomic on both
    Windows and POSIX.  A crash or ``Ctrl-C`` mid-write therefore leaves
    either the old file or the new file — never a truncated one.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp", prefix=".neuraldoom-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    return value


def _require_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}: {value!r}")
    return float(value)


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
        silence_engine: Suppress the ViZDoom engine log (it prints engine
            start-up noise to the terminal on every environment instance).
        seed: Default seed used when ``reset()`` is called without one.
    """

    scenario: str = "my_way_home"
    frame_width: int = 84
    frame_height: int = 84
    grayscale: bool = True
    frame_skip: int = 4
    frame_stack: int = 4
    window_visible: bool = False
    silence_engine: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        resolve_scenario(self.scenario)
        _require_int("frame_width", self.frame_width)
        _require_int("frame_height", self.frame_height)
        _require_int("frame_skip", self.frame_skip)
        _require_int("frame_stack", self.frame_stack)
        _require_int("seed", self.seed)
        if not isinstance(self.grayscale, bool):
            raise ValueError(f"grayscale must be a bool, got {self.grayscale!r}")
        if not isinstance(self.window_visible, bool):
            raise ValueError(f"window_visible must be a bool, got {self.window_visible!r}")
        if not isinstance(self.silence_engine, bool):
            raise ValueError(f"silence_engine must be a bool, got {self.silence_engine!r}")
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
        """Rebuild a config from a mapping, ignoring unknown keys.

        Raises:
            ValueError: If ``data`` is not a mapping or a known field has
                the wrong type (e.g. a hand-edited sidecar with
                ``"frame_stack": "4"``).
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"EnvConfig sidecar must be a JSON object, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            warnings.warn(f"Ignoring unknown EnvConfig keys: {sorted(unknown)}", stacklevel=2)
        try:
            return cls(**{key: data[key] for key in data if key in known})
        except TypeError as exc:
            raise ValueError(f"Invalid EnvConfig sidecar: {exc}") from exc

    def save(self, path: str) -> None:
        """Write the config as JSON (used for checkpoint sidecars)."""
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> EnvConfig:
        """Load a config from JSON, with a clear error for corrupt files.

        Raises:
            ValueError: If the file is not valid JSON or fails validation.
            FileNotFoundError: If the file does not exist.
        """
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt EnvConfig sidecar {path!r}: invalid JSON ({exc}). "
                "Delete it or retrain to regenerate it."
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupt EnvConfig sidecar {path!r}: not UTF-8 ({exc}).") from exc
        return cls.from_dict(data)


def env_config_path_for_model(model_path: str) -> str:
    """Return the sidecar JSON path for a checkpoint (``model.zip.env.json``)."""
    return f"{model_path}.env.json"


def save_env_config_for_model(config: EnvConfig, model_path: str) -> str:
    """Persist ``config`` next to a checkpoint so it can be reproduced later."""
    path = env_config_path_for_model(model_path)
    config.save(path)
    return path


def load_env_config_for_model(model_path: str) -> EnvConfig | None:
    """Load the sidecar written by :func:`save_env_config_for_model`, if any.

    Raises:
        ValueError: If the sidecar exists but is corrupt/invalid.
    """
    path = env_config_path_for_model(model_path)
    if not os.path.exists(path):
        return None
    return EnvConfig.load(path)


@dataclass
class TrainConfig:
    """PPO training configuration with validated defaults for 8 GB VRAM.

    ``device="auto"`` uses CUDA when available (Ubuntu workstation with an
    NVIDIA GPU) and falls back to CPU otherwise (Windows laptop, CI).
    """

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
        _require_int("total_timesteps", self.total_timesteps)
        _require_int("n_envs", self.n_envs)
        _require_int("n_steps", self.n_steps)
        _require_int("batch_size", self.batch_size)
        _require_int("n_epochs", self.n_epochs)
        _require_int("checkpoint_freq", self.checkpoint_freq)
        _require_int("eval_freq", self.eval_freq)
        _require_int("n_eval_episodes", self.n_eval_episodes)
        _require_float("learning_rate", self.learning_rate)
        _require_float("gamma", self.gamma)
        _require_float("gae_lambda", self.gae_lambda)
        _require_float("clip_range", self.clip_range)
        _require_float("ent_coef", self.ent_coef)
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be >= 1")
        if self.checkpoint_freq < 1 or self.eval_freq < 1:
            raise ValueError("checkpoint_freq and eval_freq must be >= 1")
        if not (self.learning_rate > 0):
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if not (self.clip_range > 0):
            raise ValueError("clip_range must be > 0")
        if not (self.ent_coef >= 0):
            raise ValueError("ent_coef must be >= 0")
        if not isinstance(self.device, str) or self.device.lower() not in VALID_DEVICES:
            raise ValueError(f"device must be one of {list(VALID_DEVICES)}, got {self.device!r}")
        # Normalise so downstream comparisons are case-insensitive.
        self.device = self.device.lower()

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        """Rebuild a config from a mapping, ignoring unknown keys."""
        if not isinstance(data, Mapping):
            raise ValueError(
                f"TrainConfig sidecar must be a JSON object, got {type(data).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            warnings.warn(f"Ignoring unknown TrainConfig keys: {sorted(unknown)}", stacklevel=2)
        try:
            return cls(**{key: data[key] for key in data if key in known})
        except TypeError as exc:
            raise ValueError(f"Invalid TrainConfig sidecar: {exc}") from exc

    def save(self, path: str) -> None:
        """Write the config as JSON (used for checkpoint sidecars)."""
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> TrainConfig:
        """Load a config from JSON, with a clear error for corrupt files."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt TrainConfig sidecar {path!r}: invalid JSON ({exc}). "
                "Delete it or retrain to regenerate it."
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupt TrainConfig sidecar {path!r}: not UTF-8 ({exc}).") from exc
        return cls.from_dict(data)


def train_config_path_for_model(model_path: str) -> str:
    """Return the training sidecar path for a checkpoint (``model.zip.train.json``)."""
    return f"{model_path}.train.json"


def save_train_config_for_model(config: TrainConfig, model_path: str) -> str:
    """Persist ``config`` next to a checkpoint so resume can compare it."""
    path = train_config_path_for_model(model_path)
    config.save(path)
    return path


def load_train_config_for_model(model_path: str) -> TrainConfig | None:
    """Load the training sidecar written by :func:`save_train_config_for_model`, if any."""
    path = train_config_path_for_model(model_path)
    if not os.path.exists(path):
        return None
    return TrainConfig.load(path)


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
        base_env, base_train = get_low_ram_configs(scenario=scenario, seed=seed)
    else:
        base_env = EnvConfig(scenario=scenario, seed=seed)
        base_train = TrainConfig()

    env_data = base_env.to_dict()
    train_data = base_train.to_dict()

    env_data["window_visible"] = bool(window_visible or env_data["window_visible"])
    if frame_stack is not None:
        env_data["frame_stack"] = frame_stack
    if frame_skip is not None:
        env_data["frame_skip"] = frame_skip

    if timesteps is not None:
        train_data["total_timesteps"] = timesteps
    if n_steps is not None:
        train_data["n_steps"] = n_steps
    if batch_size is not None:
        train_data["batch_size"] = batch_size
    if n_envs is not None:
        train_data["n_envs"] = n_envs
    if ent_coef is not None:
        train_data["ent_coef"] = ent_coef
    if checkpoint_freq is not None:
        train_data["checkpoint_freq"] = checkpoint_freq
    if eval_freq is not None:
        train_data["eval_freq"] = eval_freq
    if n_eval_episodes is not None:
        train_data["n_eval_episodes"] = n_eval_episodes
    if device is not None:
        train_data["device"] = device
    train_data["save_dir"] = save_dir
    train_data["log_dir"] = log_dir

    # Single construction validates the final values exactly once (no
    # double warning for indivisible batch sizes).
    return EnvConfig(**env_data), TrainConfig(**train_data)


def _checkpoint_hint(checkpoint_path: str) -> str:
    """Explain a missing checkpoint: the usual ``.zip`` slip, or what is nearby.

    Pointing at the files that *do* exist turns the most common mistake (a
    typo'd or not-yet-written path) into a one-line fix instead of a traceback.
    """
    parts: list[str] = []
    if not checkpoint_path.endswith(".zip"):
        parts.append("Stable-Baselines3 saves models as '.zip'; try adding it.")

    folder = os.path.dirname(checkpoint_path) or "."
    target = os.path.basename(checkpoint_path)
    try:
        candidates = sorted(n for n in os.listdir(folder) if n.endswith(".zip"))
    except OSError:
        candidates = []
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda name: difflib.SequenceMatcher(None, target, name).ratio(),
            reverse=True,
        )
        listed = ", ".join(os.path.join(folder, name) for name in ranked[:3])
        parts.append(f"Checkpoints that do exist: {listed}.")

    return (" " + " ".join(parts)) if parts else ""


def resolve_checkpoint_env_config(
    checkpoint_path: str,
    *,
    scenario: str | None = None,
    seed: int = 100,
    frame_stack: int | None = None,
    frame_skip: int | None = None,
    window_visible: bool = False,
    silence_engine: bool | None = None,
) -> EnvConfig:
    """Return the environment config that reproduces ``checkpoint_path``.

    The ``.env.json`` sidecar written during training is authoritative; CLI
    values are only a fallback for checkpoints that predate sidecars.

    Raises:
        SystemExit: If an explicit ``scenario`` contradicts the sidecar.
        ValueError: If the sidecar exists but is corrupt/invalid.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}.{_checkpoint_hint(checkpoint_path)}"
        )

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
        if frame_skip is not None and frame_skip != stored.frame_skip:
            print(
                f"[Config] Note: frame_skip={frame_skip} ignored; the checkpoint "
                f"was trained with frame_skip={stored.frame_skip}."
            )
        stored.window_visible = window_visible
        if silence_engine is not None:
            stored.silence_engine = silence_engine
        return stored

    print(
        "[Config] Warning: no .env.json sidecar next to the checkpoint; falling "
        "back to CLI flags. A space mismatch will be reported if they differ."
    )
    return EnvConfig(
        scenario=scenario or "my_way_home",
        frame_stack=4 if frame_stack is None else frame_stack,
        frame_skip=4 if frame_skip is None else frame_skip,
        seed=seed,
        window_visible=window_visible,
        **({} if silence_engine is None else {"silence_engine": silence_engine}),
    )
