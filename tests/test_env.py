"""Tests for the Gymnasium-compatible ViZDoom environment wrapper.

Regression focus: episode endings must distinguish goal/death (terminated) from
the scenario timeout (truncated), and ``is_success`` must not be guessed from
the reward sign.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import EnvConfig
from src.env import VizDoomEnv, silent_engine_ini_path


def test_silent_engine_ini_requests_noconsole(tmp_path, monkeypatch):
    """The engine only stays quiet if its config says so before it boots."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path), raising=False)
    path = silent_engine_ini_path()

    text = Path(path).read_text(encoding="utf-8")
    assert "[GlobalSettings]" in text
    assert "viz_noconsole=1" in text
    assert Path(path).parent == tmp_path


def test_silent_engine_ini_is_unique_per_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path), raising=False)
    assert silent_engine_ini_path() != silent_engine_ini_path()


def test_silent_env_points_the_engine_at_its_own_config(make_env):
    env = make_env(scenario="basic", silence_engine=True)
    config_path = env.game.get_doom_config_path()

    assert os.path.basename(config_path).startswith("neuraldoom-engine-")
    assert "viz_noconsole=1" in Path(config_path).read_text(encoding="utf-8")


def test_engine_log_env_keeps_the_default_config(make_env):
    env = make_env(scenario="basic", silence_engine=False)

    # ``--engine-log`` must not move the engine's config file around; an empty
    # path is how ViZDoom reports "use the default _vizdoom.ini in the cwd".
    assert env.game.get_doom_config_path() == ""


def test_env_can_init_with_engine_log_visible(make_env):
    """``silence_engine=False`` (``--engine-log``) must still work."""
    env = VizDoomEnv(config=EnvConfig(scenario="basic", frame_stack=1, silence_engine=False))
    try:
        env.reset(seed=0)
    finally:
        env.close()


EXPECTED_ACTION_COUNTS = {
    "my_way_home": 7,
    "health_gathering": 5,
    "basic": 3,
    "defend_the_center": 3,
    "deadly_corridor": 9,
}


@pytest.mark.parametrize("scenario,expected", sorted(EXPECTED_ACTION_COUNTS.items()))
def test_scenario_action_counts(make_env, scenario, expected):
    env = make_env(scenario, frame_stack=1)
    assert env.action_space.n == expected
    assert len(env.action_names) == expected
    assert len(set(env.action_names)) == expected


def test_reset_observation_shape_and_dtype(make_env):
    env = make_env("my_way_home", frame_stack=4)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert info["scenario"] == "my_way_home"


def test_seed_is_deterministic(make_env):
    env = make_env("my_way_home")
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert np.array_equal(first, second)


def test_different_seeds_produce_different_episodes(make_env):
    env = make_env("my_way_home")
    first, _ = env.reset(seed=1)
    second, _ = env.reset(seed=2)
    assert not np.array_equal(first, second)


def test_frame_stack_composition(make_env):
    env = make_env("my_way_home", frame_stack=4)
    obs0, _ = env.reset(seed=3)
    # A freshly reset stack repeats the same frame `frame_stack` times.
    for index in range(1, env.config.frame_stack):
        assert np.array_equal(obs0[0], obs0[index])

    obs1, _, _, _, _ = env.step(0)
    obs2, _, _, _, _ = env.step(0)
    # The stacked window slides by exactly one frame per step.
    assert np.array_equal(obs1[:-1], obs0[1:])
    assert np.array_equal(obs2[:-1], obs1[1:])


def test_all_actions_are_executable(make_env):
    env = make_env("health_gathering")
    env.reset(seed=123)
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert not (terminated and truncated)
        if terminated or truncated:
            env.reset()


def test_invalid_action_raises(make_env):
    env = make_env("my_way_home")
    env.reset(seed=1)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(env.action_space.n)


def test_step_before_reset_raises():
    env = VizDoomEnv(EnvConfig(scenario="my_way_home", frame_stack=1))
    try:
        with pytest.raises(RuntimeError, match="before reset"):
            env.step(0)
    finally:
        env.close()


def test_step_after_done_raises(make_env):
    env = make_env("my_way_home")
    env.game.set_episode_timeout(12)
    env.reset(seed=5)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
    with pytest.raises(RuntimeError, match="after the episode finished"):
        env.step(0)


def test_timeout_is_reported_as_truncation(make_env):
    """A timeout must be truncated, not terminated (PPO bootstrapping)."""
    env = make_env("my_way_home")
    env.game.set_episode_timeout(12)
    env.reset(seed=11)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
    assert info["episode_outcome"] == "timeout"
    assert truncated is True
    assert terminated is False
    assert info["is_success"] is False


@pytest.mark.parametrize(
    "timeout_reached,player_dead,expected",
    [
        (True, False, "timeout"),
        (False, True, "death"),
        (False, False, "goal"),
        (True, True, "timeout"),  # timeout wins over a death at the same tic
    ],
)
def test_episode_outcome_classification(make_env, timeout_reached, player_dead, expected):
    """Unit-test the classifier with a stub instead of playing long episodes."""
    env = make_env("my_way_home")
    real_game = env.game
    try:
        env.game = SimpleNamespace(
            is_episode_timeout_reached=lambda: timeout_reached,
            is_player_dead=lambda: player_dead,
            get_episode_time=lambda: 10,
        )
        assert env._classify_episode_end() == expected
    finally:
        env.game = real_game


def test_health_gathering_death_is_terminated(make_env):
    """Death ends the episode as terminated (not truncated, not a success)."""
    env = make_env("health_gathering")
    env.game.set_episode_timeout(4000)  # let a death happen before the timeout
    env.reset(seed=4)
    done = False
    info = {}
    steps = 0
    while not done and steps < 2000:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    if info.get("episode_outcome") == "death":
        assert info["is_success"] is False
        assert terminated is True
    else:  # pragma: no cover - random policy that survives; still a valid run
        assert info.get("episode_outcome") in ("goal", "timeout")


def test_render_rgb_array(make_env):
    env = VizDoomEnv(EnvConfig(scenario="my_way_home", frame_stack=1), render_mode="rgb_array")
    try:
        env.reset(seed=1)
        frame = env.render()
        assert frame is not None
        assert frame.shape == (84, 84, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_render_returns_none_without_render_mode(make_env):
    env = make_env("my_way_home")
    env.reset(seed=1)
    assert env.render() is None


def test_invalid_render_mode_raises():
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        VizDoomEnv(EnvConfig(scenario="my_way_home"), render_mode="hologram")


def test_rgb_observation_path(make_env):
    """The (previously untested) RGB24 branch must produce CHW uint8 frames."""
    env = make_env("my_way_home", grayscale=False, frame_stack=2)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6, 84, 84)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_close_is_idempotent(make_env):
    env = make_env("my_way_home")
    env.reset(seed=1)
    env.close()
    env.close()


def test_timeout_ticks_match_scenario(make_env):
    env = make_env("basic")
    assert env.timeout_ticks == 300


def test_episode_time_ticks_are_reported_for_every_outcome():
    """ViZDoom zeroes its tic counter when a goal ends the episode.

    Regression: ``info["episode_time_ticks"]`` used to be 0 for goal episodes,
    which made logged episode durations wrong.
    """
    env = VizDoomEnv(EnvConfig(scenario="basic", frame_stack=1, seed=0))
    try:
        seen = {}
        for episode in range(20):
            env.reset(seed=episode)
            while True:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    seen.setdefault(info["episode_outcome"], info["episode_time_ticks"])
                    break
        assert seen, "no episode finished"
        for outcome, ticks in seen.items():
            assert ticks > 0, f"{outcome} episode reported {ticks} tics"
    finally:
        env.close()
