"""Tests for the command-line entry points' error handling.

Regression focus: a typo'd ``--checkpoint`` path used to surface as a raw
traceback (``src/play.py``) or as a misleading "no .env.json sidecar" warning
printed *before* the real error (``src/evaluate.py``).
"""

import pytest

from src.evaluate import main as evaluate_main
from src.evaluate import parse_args as evaluate_parse_args
from src.play import main as play_main
from src.play import parse_args as play_parse_args
from src.train import parse_args as train_parse_args


@pytest.mark.parametrize("suffix", [".zip", ""])
def test_play_missing_checkpoint_exits_cleanly(tmp_path, capsys, suffix):
    missing = str(tmp_path / f"nope{suffix}")
    rc = play_main(["--checkpoint", missing, "--headless"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "Checkpoint not found" in out
    assert "Traceback" not in out
    assert "sidecar" not in out


def test_play_still_requires_checkpoint_or_random(capsys):
    assert play_main([]) == 2
    assert "provide --checkpoint" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [".zip", ""])
def test_evaluate_missing_checkpoint_exits_cleanly(tmp_path, capsys, suffix):
    missing = str(tmp_path / f"nope{suffix}")
    rc = evaluate_main(["--checkpoint", missing])
    out = capsys.readouterr().out

    assert rc == 2
    assert "Checkpoint not found" in out
    assert "Traceback" not in out
    # The sidecar warning must not appear before the real error.
    assert "sidecar" not in out


def test_engine_log_flag_defaults_to_silent():
    assert play_parse_args(["--random"]).engine_log is False
    assert evaluate_parse_args(["--checkpoint", "x.zip"]).engine_log is False
    assert train_parse_args([]).engine_log is False


def test_engine_log_flag_can_be_enabled():
    assert play_parse_args(["--random", "--engine-log"]).engine_log is True
    assert evaluate_parse_args(["--checkpoint", "x.zip", "--engine-log"]).engine_log is True
    assert train_parse_args(["--engine-log"]).engine_log is True
