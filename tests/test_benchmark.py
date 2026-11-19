"""Benchmark validation and a complete disposable PPO trial."""

import pytest

from src.benchmark import main, parse_args, run_trial


@pytest.mark.parametrize(
    "args",
    [
        ["--seconds", "0"],
        ["--seconds", "nan"],
        ["--seconds", "inf"],
        ["--env-counts", "0"],
        ["--threads", "0"],
        ["--n-steps", "0"],
    ],
)
def test_invalid_args(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_defaults():
    args = parse_args([])
    assert args.env_counts == [1, 2, 4, 8]
    assert args.seconds == 20


@pytest.mark.slow
def test_complete_trial():
    result = run_trial(
        1,
        0.01,
        scenario="basic",
        device="cpu",
        n_steps=32,
        batch_size=32,
        seed=1,
    )
    assert result["steps"] >= 32
    assert result["steps_per_second"] > 0
    assert result["complete_cycles"] >= 1


def test_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.benchmark.run_trial",
        lambda *a, **kw: {
            "n_envs": a[0],
            "steps_per_second": 100 * a[0],
            "steps": 100,
            "measured_seconds": 1,
            "device": "cpu",
        },
    )
    path = tmp_path / "report.json"
    assert main(["--env-counts", "1", "2", "--device", "cpu", "--output", str(path)]) == 0
    assert '"steps_per_second": 200' in path.read_text()
