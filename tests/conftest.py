"""Shared pytest fixtures for NeuralDoom-Lab."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.config import EnvConfig
from src.env import VizDoomEnv


@pytest.fixture
def make_env() -> Callable[..., VizDoomEnv]:
    """Factory for headless envs that are always closed at test teardown."""
    created: list[VizDoomEnv] = []

    def _make(scenario: str = "my_way_home", **kwargs) -> VizDoomEnv:
        kwargs.setdefault("frame_stack", 1)
        env = VizDoomEnv(EnvConfig(scenario=scenario, **kwargs))
        created.append(env)
        return env

    yield _make

    for env in created:
        env.close()
