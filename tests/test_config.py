"""Tests for scenario metadata, config validation, and checkpoint sidecars."""

import json
import os

import pytest
import vizdoom as vzd

from src.config import (
    SCENARIOS,
    EnvConfig,
    TrainConfig,
    build_configs,
    env_config_path_for_model,
    get_low_ram_configs,
    load_env_config_for_model,
    resolve_checkpoint_env_config,
    resolve_scenario,
    save_env_config_for_model,
)


def test_every_scenario_has_a_shipped_cfg_file():
    for spec in SCENARIOS.values():
        path = os.path.join(vzd.scenarios_path, spec.cfg_file)
        assert os.path.exists(path), f"missing scenario cfg: {path}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_declared_timeout_matches_vizdoom(name):
    """Our documented timeout must match what ViZDoom actually loads."""
    spec = SCENARIOS[name]
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, spec.cfg_file))
    game.set_window_visible(False)
    try:
        game.init()
        assert game.get_episode_timeout() == spec.episode_timeout_ticks
    finally:
        game.close()


def test_unknown_scenario_lists_alternatives():
    with pytest.raises(ValueError, match="Unknown scenario"):
        resolve_scenario("nonexistent_scenario")


def test_scenario_summary_mentions_success_definition():
    summary = SCENARIOS["my_way_home"].summary()
    assert "my_way_home" in summary
    assert "success" in summary.lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_stack": 0},
        {"frame_skip": 0},
        {"frame_width": 0},
        {"frame_height": -8},
    ],
)
def test_env_config_validation(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_env_config_observation_shape():
    cfg = EnvConfig(frame_stack=4)
    assert cfg.observation_shape == (4, 84, 84)
    rgb = EnvConfig(frame_stack=2, grayscale=False)
    assert rgb.observation_shape == (6, 84, 84)


def test_env_config_json_roundtrip(tmp_path):
    cfg = EnvConfig(scenario="basic", frame_stack=1, frame_skip=3, seed=7)
    path = tmp_path / "env.json"
    cfg.save(str(path))
    assert EnvConfig.load(str(path)) == cfg
    assert json.loads(path.read_text())["scenario"] == "basic"


def test_env_config_ignores_unknown_keys(tmp_path):
    path = tmp_path / "env.json"
    path.write_text(json.dumps({"scenario": "basic", "legacy_flag": True}))
    with pytest.warns(UserWarning, match="unknown EnvConfig keys"):
        cfg = EnvConfig.load(str(path))
    assert cfg.scenario == "basic"


def test_checkpoint_sidecar_roundtrip(tmp_path):
    model_path = str(tmp_path / "model.zip")
    cfg = EnvConfig(scenario="health_gathering", frame_stack=1)
    sidecar = save_env_config_for_model(cfg, model_path)
    assert sidecar == env_config_path_for_model(model_path)
    assert os.path.exists(sidecar)
    assert load_env_config_for_model(model_path) == cfg


def test_missing_sidecar_returns_none(tmp_path):
    assert load_env_config_for_model(str(tmp_path / "nope.zip")) is None


def test_sidecar_is_authoritative(tmp_path):
    model_path = str(tmp_path / "model.zip")
    save_env_config_for_model(EnvConfig(scenario="basic", frame_stack=1), model_path)
    cfg = resolve_checkpoint_env_config(
        model_path, scenario=None, frame_stack=4, window_visible=False
    )
    assert cfg.scenario == "basic"
    assert cfg.frame_stack == 1  # sidecar wins over the CLI value


def test_sidecar_rejects_conflicting_scenario(tmp_path):
    model_path = str(tmp_path / "model.zip")
    save_env_config_for_model(EnvConfig(scenario="basic"), model_path)
    with pytest.raises(SystemExit, match="trained on 'basic'"):
        resolve_checkpoint_env_config(model_path, scenario="my_way_home")


def test_fallback_without_sidecar_uses_cli(tmp_path):
    cfg = resolve_checkpoint_env_config(
        str(tmp_path / "legacy.zip"), scenario="basic", frame_stack=2
    )
    assert cfg.scenario == "basic"
    assert cfg.frame_stack == 2


def test_train_config_rejects_batch_larger_than_rollout():
    with pytest.raises(ValueError, match="must be <= n_steps"):
        TrainConfig(n_steps=64, batch_size=128, n_envs=1)


def test_train_config_warns_on_indivisible_batch():
    with pytest.warns(UserWarning, match="not divisible"):
        TrainConfig(n_steps=100, batch_size=64, n_envs=1)


def test_low_ram_preset_values():
    env_cfg, train_cfg = get_low_ram_configs(seed=11)
    assert env_cfg.frame_stack == 1
    assert env_cfg.seed == 11
    assert train_cfg.device == "cpu"
    assert train_cfg.n_steps == 256
    assert train_cfg.batch_size == 32
    assert train_cfg.total_timesteps == 2048


def test_build_configs_defaults():
    env_cfg, train_cfg = build_configs("my_way_home")
    assert env_cfg.frame_stack == 4
    assert train_cfg.total_timesteps == 100_000
    assert train_cfg.device == "auto"


def test_explicit_flags_override_low_ram_preset():
    """Regression: --low-ram used to silently discard --batch-size etc."""
    env_cfg, train_cfg = build_configs(
        "basic",
        low_ram=True,
        timesteps=5_000,
        frame_stack=2,
        n_steps=512,
        batch_size=128,
        n_envs=2,
        checkpoint_freq=250,
        eval_freq=250,
        device="cuda",
    )
    assert env_cfg.frame_stack == 2
    assert train_cfg.total_timesteps == 5_000
    assert train_cfg.n_steps == 512
    assert train_cfg.batch_size == 128
    assert train_cfg.n_envs == 2
    assert train_cfg.checkpoint_freq == 250
    assert train_cfg.eval_freq == 250
    assert train_cfg.device == "cuda"


def test_build_configs_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_configs("does_not_exist")
