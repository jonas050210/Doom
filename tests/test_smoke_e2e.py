"""End-to-end smoke tests: train -> resume -> evaluate -> play.

Marked ``slow`` (excluded by default, run with ``pytest -m slow``).  These are
the tests that prove the whole pipeline still works after a refactor.
"""

import json
import os
import subprocess

import pytest

from src import train
from src.evaluate import evaluate
from src.play import main as play_main

pytestmark = pytest.mark.slow


def _train_args(tmp_path, *extra):
    return [
        "--scenario",
        "basic",
        "--timesteps",
        "512",
        "--n-steps",
        "64",
        "--batch-size",
        "32",
        "--checkpoint-freq",
        "256",
        "--eval-freq",
        "256",
        "--n-eval-episodes",
        "1",
        "--frame-stack",
        "1",
        "--device",
        "cpu",
        "--save-dir",
        str(tmp_path / "ckpt"),
        "--log-dir",
        str(tmp_path / "runs"),
        *extra,
    ]


def test_train_resume_evaluate_play(tmp_path):
    assert train.main(_train_args(tmp_path)) == 0

    latest = tmp_path / "ckpt" / "basic_latest.zip"
    assert latest.exists(), "latest pointer should be created"
    assert os.path.exists(f"{latest}.env.json"), "env config sidecar is required"

    run_dirs = sorted((tmp_path / "ckpt").glob("basic_2*"))
    assert run_dirs, "timestamped run directory should exist"
    metas = [json.loads((run_dir / "run.json").read_text()) for run_dir in run_dirs]
    assert any(meta["num_timesteps"] == 512 for meta in metas)

    first_meta = metas[0]
    assert first_meta["action_names"] == ["strafe_left", "strafe_right", "attack"]
    assert first_meta["observation_shape"] == [1, 84, 84]
    assert first_meta["env_config"]["frame_stack"] == 1
    assert sum(first_meta["episode_outcomes"].values()) > 0
    assert any((run_dir / "best_model.zip").exists() for run_dir in run_dirs)

    # Resuming to 1024 total steps must work and must not silently no-op.
    assert train.main(_train_args(tmp_path, "--resume", "latest", "--timesteps", "1024")) == 0
    metas = [
        json.loads((run_dir / "run.json").read_text())
        for run_dir in sorted((tmp_path / "ckpt").glob("basic_2*"))
    ]
    assert any(meta["num_timesteps"] == 1024 for meta in metas)

    payload = evaluate(str(latest), episodes=1, seed=1, device="cpu")
    assert payload["summary"]["episodes"] == 1
    assert payload["scenario"] == "basic"

    assert (
        play_main(["--checkpoint", str(latest), "--episodes", "1", "--headless", "--device", "cpu"])
        == 0
    )


def _count_vizdoom_processes() -> int:
    """Count live ViZDoom game processes (Linux only; 0 elsewhere)."""
    import sys

    if not sys.platform.startswith("linux"):
        return 0
    try:
        listing = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:  # pragma: no cover - ps missing
        return 0
    return sum(1 for line in listing.splitlines() if "vizdoom -iwad" in line)


def test_sigterm_saves_and_reaps_vizdoom(tmp_path):
    """SIGTERM (from ``timeout``/systemd/CI) must save and leave no orphans."""
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "train.log"
    before = _count_vizdoom_processes()

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "src.train",
                "--scenario",
                "basic",
                "--timesteps",
                "200000",
                "--frame-stack",
                "1",
                "--device",
                "cpu",
                "--save-dir",
                str(tmp_path / "ckpt"),
                "--log-dir",
                str(tmp_path / "runs"),
            ],
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.time() + 120
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if "Training for" in log_path.read_text(encoding="utf-8"):
                break
            time.sleep(0.2)

        output = log_path.read_text(encoding="utf-8")
        assert "Training for" in output, f"training never started:\n{output}"

        process.send_signal(signal.SIGTERM)
        returncode = process.wait(timeout=120)

    output = log_path.read_text(encoding="utf-8")
    assert returncode == 130, output
    assert "Progress saved" in output, output
    assert list((tmp_path / "ckpt").glob("*/basic_final.zip")), output

    # ViZDoom children must be gone once the trainer has exited.
    deadline = time.time() + 30
    while time.time() < deadline and _count_vizdoom_processes() > before:
        time.sleep(0.5)
    assert _count_vizdoom_processes() <= before, "ViZDoom child processes were orphaned"
