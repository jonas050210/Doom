"""Regression coverage for platform, device, and persistence paths."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.config import (
    EnvConfig,
    TrainConfig,
    load_train_config_for_model,
    save_train_config_for_model,
)
from src.env import _to_grayscale_hw, _to_rgb_hwc, has_display
from src.model import gpu_info, resolve_device
from src.train import CheckpointWithSidecarCallback, make_vec_env


def test_force_headless(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("NEURALDOOM_NO_DISPLAY", "1")
    assert not has_display()


@pytest.mark.parametrize("layout", [(8, 9), (8, 9, 1), (1, 8, 9)])
def test_grayscale_layouts(layout):
    assert _to_grayscale_hw(np.zeros(layout, dtype=np.uint8)).shape == (8, 9)


@pytest.mark.parametrize("layout", [(8, 9, 3), (3, 8, 9)])
def test_rgb_layouts(layout):
    assert _to_rgb_hwc(np.zeros(layout, dtype=np.uint8)).shape == (8, 9, 3)


def test_cuda_auto_and_fallback(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"
    assert gpu_info()["cuda_available"] is False
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: "Test GPU")
    assert resolve_device("auto") == "cuda"


def test_train_sidecar_roundtrip_and_corruption(tmp_path):
    path = str(tmp_path / "model.zip")
    cfg = TrainConfig()
    save_train_config_for_model(cfg, path)
    assert load_train_config_for_model(path) == cfg
    (tmp_path / "model.zip.train.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        load_train_config_for_model(path)


def test_periodic_checkpoint_sidecars(tmp_path):
    cfg = EnvConfig(frame_stack=1)
    cb = CheckpointWithSidecarCallback(cfg, TrainConfig(), save_freq=1, save_path=str(tmp_path))
    cb.model = Mock()
    cb.n_calls = 1
    cb.num_timesteps = 32
    path = tmp_path / "rl_model_32_steps.zip"
    cb.model.save.side_effect = lambda filename: path.write_bytes(b"model")
    assert cb._on_step()
    assert path.with_name(path.name + ".env.json").exists()
    assert path.with_name(path.name + ".train.json").exists()


def test_spawn_workers():
    env = make_vec_env(EnvConfig(scenario="basic", frame_stack=1), 2)
    try:
        obs = env.reset()
        assert obs.shape == (2, 1, 84, 84)
        obs, rewards, dones, infos = env.step(np.array([0, 0]))
        assert len(infos) == 2
    finally:
        env.close()
