"""Compare end-to-end PPO throughput without touching saved checkpoints.

Each trial uses a fresh disposable model, warms up for one complete PPO
rollout/update, then measures complete rollout/update cycles. Duration is a
soft target: the last cycle finishes rather than interrupting an optimizer.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from datetime import datetime

import torch
from stable_baselines3.common.logger import configure

from src.config import EnvConfig, TrainConfig, atomic_write_json, scenario_names
from src.model import create_ppo_model, gpu_info, resolve_device
from src.train import make_vec_env, set_global_seeds


def run_trial(
    n_envs: int,
    seconds: float,
    *,
    scenario: str,
    device: str,
    n_steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Measure full training cycles; setup/warmup are reported separately."""
    started = time.perf_counter()
    env = make_vec_env(EnvConfig(scenario=scenario, seed=seed), n_envs)
    model = None
    try:
        set_global_seeds(seed)
        cfg = TrainConfig(
            n_envs=n_envs,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
            ent_coef=0.05,
        )
        model = create_ppo_model(env, cfg, seed=seed, verbose=0)
        model.set_logger(configure(folder=None, format_strings=[]))
        cycle_steps = n_envs * n_steps
        model.learn(total_timesteps=cycle_steps)
        warmup_seconds = time.perf_counter() - started
        initial_steps = model.num_timesteps
        measured_start = time.perf_counter()
        cycles = 0
        while time.perf_counter() - measured_start < seconds:
            model.learn(total_timesteps=cycle_steps, reset_num_timesteps=False)
            cycles += 1
        # Synchronize before stopping the clock so queued GPU work is included.
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - measured_start
        steps = int(model.num_timesteps - initial_steps)
        return {
            "n_envs": n_envs,
            "device": str(model.device),
            "measured_seconds": elapsed,
            "steps": steps,
            "steps_per_second": steps / elapsed,
            "complete_cycles": cycles,
            "setup_and_warmup_seconds": warmup_seconds,
        }
    finally:
        env.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=scenario_names(), default="defend_the_center")
    parser.add_argument("--env-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/benchmark-results.json")
    args = parser.parse_args(argv)
    if not 0 < args.seconds < float("inf"):
        parser.error("--seconds must be finite and positive")
    if args.threads < 1 or any(n < 1 for n in args.env_counts):
        parser.error("thread and environment counts must be positive")
    for n in args.env_counts:
        try:
            TrainConfig(n_envs=n, n_steps=args.n_steps, batch_size=args.batch_size)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    results = []
    print("Disposable models only; existing checkpoints will not be loaded or changed.")
    print("Duration excludes setup/warmup and may overrun by one full PPO cycle.")
    for n in args.env_counts:
        print(f"Testing {n} environment(s) for ~{args.seconds:g}s...", flush=True)
        try:
            result = run_trial(
                n,
                args.seconds,
                scenario=args.scenario,
                device=device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        except (RuntimeError, OSError, EOFError) as exc:
            result = {"n_envs": n, "error": str(exc)}
            print(f"  FAILED: {exc}", flush=True)
        else:
            print(
                f"  {result['steps_per_second']:.0f} steps/s | "
                f"{result['steps']} steps in {result['measured_seconds']:.2f}s | "
                f"device={result['device']}",
                flush=True,
            )
        results.append(result)
        # Preserve completed trials even if a later trial is interrupted.
        atomic_write_json(
            args.output,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "settings": vars(args),
                "gpu": gpu_info(),
                "results": results,
            },
        )
    successful = [r for r in results if "steps_per_second" in r]
    if not successful:
        return 1
    best = max(successful, key=lambda r: r["steps_per_second"])
    print(f"\nFastest measured: --n-envs {best['n_envs']} ({best['steps_per_second']:.0f} steps/s)")
    print("Short noisy throughput test, not a learning-quality ranking. Repeat to confirm.")
    print("Changing n_envs also changes rollout size and optimizer work per cycle.")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
