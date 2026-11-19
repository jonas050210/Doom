"""Tests for device resolution and PPO model helpers."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import TrainConfig
from src.model import create_ppo_model, load_ppo_model, resolve_device


class FakeImageEnv(gym.Env):
    """Minimal channels-first image env, so model tests need no ViZDoom."""

    metadata = {"render_modes": []}

    def __init__(self, channels: int = 1, size: int = 84, n_actions: int = 3):
        self.observation_space = spaces.Box(0, 255, shape=(channels, size, size), dtype=np.uint8)
        self.action_space = spaces.Discrete(n_actions)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._steps += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        terminated = self._steps >= 5
        return obs, 1.0, terminated, False, {}


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_valid_device():
    assert resolve_device("auto") in ("cpu", "cuda")


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_create_ppo_model_requires_vec_env():
    with pytest.raises(TypeError, match="expects a stable_baselines3 VecEnv"):
        create_ppo_model(FakeImageEnv())


def test_create_and_train_tiny_ppo(tmp_path):
    venv = DummyVecEnv([lambda: FakeImageEnv()])
    cfg = TrainConfig(total_timesteps=64, n_steps=32, batch_size=16, device="cpu")
    model = create_ppo_model(venv, train_config=cfg, seed=0, tensorboard_log=str(tmp_path))
    assert model.action_space.n == 3
    model.learn(total_timesteps=64)

    checkpoint = tmp_path / "model.zip"
    model.save(str(checkpoint))
    reloaded = load_ppo_model(str(checkpoint), device="cpu")
    assert reloaded.observation_space.shape == (1, 84, 84)
    assert reloaded.action_space == model.action_space
    venv.close()


def test_load_missing_checkpoint_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="appends '.zip'"):
        load_ppo_model(str(tmp_path / "nope"), device="cpu")
