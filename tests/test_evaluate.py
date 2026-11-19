"""Tests for evaluation summaries and the checkpoint/report pipeline."""

import json
import os

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import EnvConfig, save_env_config_for_model
from src.env import VizDoomEnv
from src.evaluate import evaluate, summarize_results


def test_summarize_results_math():
    summary = summarize_results(
        rewards=[1.0, 2.0, 3.0],
        lengths=[10, 20, 30],
        outcomes=["goal", "death", "timeout"],
    )
    assert summary["episodes"] == 3
    assert summary["mean_reward"] == pytest.approx(2.0)
    assert summary["std_reward"] == pytest.approx(float(np.std([1.0, 2.0, 3.0])))
    assert summary["mean_length"] == pytest.approx(20.0)
    assert summary["goal_rate"] == pytest.approx(100 / 3)
    assert summary["death_rate"] == pytest.approx(100 / 3)
    assert summary["timeout_rate"] == pytest.approx(100 / 3)
    assert summary["outcome_counts"] == {
        "goal": 1,
        "death": 1,
        "timeout": 1,
        "unknown": 0,
    }


def test_summarize_unknown_outcome_is_counted():
    summary = summarize_results([0.0], [5], [None])
    assert summary["outcome_counts"]["unknown"] == 1


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError, match="zero episodes"):
        summarize_results([], [], [])


def test_evaluate_rejects_zero_episodes(tmp_path):
    with pytest.raises(ValueError, match="episodes must be"):
        evaluate(str(tmp_path / "model.zip"), episodes=0)


@pytest.fixture
def basic_checkpoint(tmp_path):
    """Tiny (untrained) PPO checkpoint plus sidecar for the `basic` scenario."""
    cfg = EnvConfig(scenario="basic", frame_stack=1, seed=0)
    venv = DummyVecEnv([lambda: VizDoomEnv(cfg)])
    model = PPO("CnnPolicy", venv, n_steps=16, batch_size=8, device="cpu", seed=0, verbose=0)
    path = str(tmp_path / "basic_model.zip")
    model.save(path)
    save_env_config_for_model(cfg, path)
    venv.close()
    return cfg, path


def test_evaluate_writes_report(basic_checkpoint, tmp_path):
    cfg, checkpoint = basic_checkpoint
    payload = evaluate(checkpoint, episodes=1, seed=5, device="cpu")

    assert payload["scenario"] == "basic"
    assert payload["has_goal"] is True
    assert payload["success_signal"]
    assert payload["summary"]["episodes"] == 1
    assert payload["env_config"]["frame_stack"] == 1
    assert len(payload["episodes"]) == 1
    episode = payload["episodes"][0]
    assert episode["outcome"] in ("goal", "death", "timeout")
    assert episode["length"] > 0

    report = f"{os.path.splitext(checkpoint)[0]}.eval.json"
    assert os.path.exists(report)
    with open(report, encoding="utf-8") as handle:
        assert json.load(handle)["summary"]["episodes"] == 1


def test_evaluate_uses_sidecar_config(basic_checkpoint):
    """Regression: evaluation used to guess frame_stack/actions from flags."""
    _, checkpoint = basic_checkpoint
    payload = evaluate(checkpoint, episodes=1, frame_stack=4, device="cpu")
    assert payload["env_config"]["frame_stack"] == 1  # sidecar wins


def test_evaluate_rejects_conflicting_scenario(basic_checkpoint):
    _, checkpoint = basic_checkpoint
    with pytest.raises(SystemExit, match="trained on 'basic'"):
        evaluate(checkpoint, scenario="my_way_home", episodes=1, device="cpu")
