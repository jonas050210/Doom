"""Tests for training CLI parsing, resume handling, and helpers."""

import os

import pytest

from src import train
from src.config import EnvConfig, save_env_config_for_model
from src.train import (
    OutcomeCounterCallback,
    latest_pointer_path,
    make_vec_env,
    parse_args,
    promote_to_latest,
    resolve_resume_path,
)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.scenario == "my_way_home"
    assert args.timesteps is None
    assert args.device is None
    assert args.low_ram is False
    assert args.resume is None


def test_resume_without_value_means_latest():
    args = parse_args(["--resume"])
    assert args.resume == "latest"


def test_resume_with_explicit_path():
    args = parse_args(["--resume", "checkpoints/x.zip"])
    assert args.resume == "checkpoints/x.zip"


def test_fresh_and_resume_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--fresh", "--resume"])


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "nope"])


def test_resolve_resume_missing_latest_explains(tmp_path):
    with pytest.raises(SystemExit, match="Nothing to resume"):
        resolve_resume_path("latest", str(tmp_path), "my_way_home")


def test_resolve_resume_explicit_missing(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        resolve_resume_path(str(tmp_path / "missing.zip"), str(tmp_path), "basic")


def test_resolve_resume_latest_exists(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    latest = tmp_path / "my_way_home_latest.zip"
    latest.write_bytes(b"fake")
    assert resolve_resume_path("latest", str(tmp_path), "my_way_home") == str(latest)


def test_promote_to_latest_copies_model_and_sidecar(tmp_path):
    run_dir = tmp_path / "run"
    os.makedirs(run_dir)
    model_path = run_dir / "basic_final.zip"
    model_path.write_bytes(b"model")
    save_env_config_for_model(EnvConfig(scenario="basic"), str(model_path))

    latest = promote_to_latest(str(model_path), str(tmp_path), "basic")

    assert latest == latest_pointer_path(str(tmp_path), "basic")
    assert os.path.exists(latest)
    assert os.path.exists(f"{latest}.env.json")


def test_make_vec_env_reports_correct_spaces(tmp_path):
    venv = make_vec_env(
        EnvConfig(scenario="basic", frame_stack=1),
        1,
        monitor_path=str(tmp_path / "monitor.csv"),
    )
    try:
        assert venv.num_envs == 1
        assert venv.observation_space.shape == (1, 84, 84)
        assert venv.action_space.n == 3
        assert venv.get_attr("action_names")[0] == ["strafe_left", "strafe_right", "attack"]
        venv.reset()
        venv.step([0])
    finally:
        venv.close()


def test_outcome_counter_callback():
    callback = OutcomeCounterCallback()
    callback.locals = {
        "infos": [
            {"episode_outcome": "goal"},
            {"episode_outcome": "death"},
            {"episode_outcome": None},
            {"episode_outcome": "timeout"},
            {"episode_outcome": "goal"},
        ]
    }
    assert callback._on_step() is True
    assert callback.counts == {"goal": 2, "death": 1, "timeout": 1}
    assert "4 episodes" in callback.summary()


def test_render_without_display_returns_error(monkeypatch):
    monkeypatch.setattr(train, "has_display", lambda: False)
    assert train.main(["--render", "--timesteps", "64"]) == 2


def test_invalid_config_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "has_display", lambda: False)
    # n_steps smaller than batch_size must fail cleanly, not with a traceback.
    code = train.main(
        [
            "--scenario",
            "basic",
            "--n-steps",
            "32",
            "--batch-size",
            "128",
            "--save-dir",
            str(tmp_path / "ckpt"),
            "--log-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 2


def test_sigterm_handler_is_installed_and_restored():
    import signal

    before = signal.getsignal(signal.SIGTERM)
    with train.SigtermToInterrupt():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before
