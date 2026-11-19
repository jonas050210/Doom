"""Tests for the Gymnasium-compatible ViZDoom environment wrapper."""

import numpy as np
import pytest

from src.config import EnvConfig
from src.env import VizDoomEnv


def test_env_reset_and_observation_shape():
    env = VizDoomEnv(EnvConfig(scenario="my_way_home", frame_stack=4, window_visible=False))
    try:
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
    finally:
        env.close()


def test_all_actions_valid_and_random_rollout():
    env = VizDoomEnv(EnvConfig(scenario="my_way_home", frame_stack=4, window_visible=False))
    try:
        obs, _ = env.reset(seed=123)
        assert env.action_space.n == 5

        # Step every valid action once
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (4, 84, 84)
            assert obs.dtype == np.uint8
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "is_success" in info
            if terminated or truncated:
                obs, _ = env.reset()

        # Random agent smoke rollout for 30 steps
        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert obs.shape == (4, 84, 84)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


def test_invalid_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        VizDoomEnv(EnvConfig(scenario="nonexistent_scenario"))


def test_health_gathering_scenario():
    env = VizDoomEnv(EnvConfig(scenario="health_gathering", frame_stack=4, window_visible=False))
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (4, 84, 84)
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


def test_low_ram_frame_stack_one():
    env = VizDoomEnv(EnvConfig(scenario="my_way_home", frame_stack=1, window_visible=False))
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (1, 84, 84)
        assert obs.dtype == np.uint8
    finally:
        env.close()


@pytest.mark.parametrize("scenario", ["basic", "defend_the_center", "deadly_corridor"])
def test_combat_scenarios(scenario):
    env = VizDoomEnv(
        EnvConfig(
            scenario=scenario,
            frame_stack=1,
            include_use_attack=True,
            window_visible=False,
        )
    )
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (1, 84, 84)
        assert env.action_space.n == 7
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
