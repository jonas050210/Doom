"""Train a PPO agent on a free ViZDoom scenario.

Checkpoint layout
-----------------
Every run gets its own timestamped directory::

    checkpoints/<scenario>_<timestamp>/
        <scenario>_ckpt_<steps>_steps.zip   # periodic checkpoints
        best_model.zip                      # best deterministic evaluation
        <scenario>_final.zip                # saved at the end / on Ctrl-C
        *.env.json                          # sidecar env config for each model
        *.train.json                        # sidecar training config for each model
    checkpoints/<scenario>_latest.zip       # stable pointer to the newest model

The sidecars record the exact observation/action configuration and the PPO
hyper-parameters so ``evaluate`` and ``play`` reproduce them instead of
guessing, and ``--resume`` can warn when flags differ from the original run.

Cross-platform / GPU
--------------------
* Works on Ubuntu (primary, GPU workstation) and Windows PowerShell (laptop).
  ``SubprocVecEnv`` always uses ``spawn`` so it behaves identically on both
  (Windows has no ``fork``).
* ``--device auto`` (default) uses CUDA on Ubuntu when available and CPU
  elsewhere.  ``--low-ram`` forces CPU for the 2 GB laptop.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import shutil
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from src.config import (
    SCENARIOS,
    EnvConfig,
    TrainConfig,
    atomic_write_json,
    build_configs,
    load_env_config_for_model,
    load_train_config_for_model,
    save_env_config_for_model,
    save_train_config_for_model,
    scenario_names,
)
from src.env import VizDoomEnv, has_display
from src.model import create_ppo_model, gpu_info, resolve_device


class OutcomeCounterCallback(BaseCallback):
    """Count ``goal``/``death``/``timeout`` episode endings during training.

    Reward alone hides *why* an episode ended (a timeout and a death at the
    same shaped reward look identical), so this records the env's
    ``episode_outcome`` info key and prints a summary after training.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.counts: dict[str, int] = {"goal": 0, "death": 0, "timeout": 0}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []) or []:
            outcome = info.get("episode_outcome")
            if outcome:
                self.counts[outcome] = self.counts.get(outcome, 0) + 1
        return True

    def summary(self) -> str:
        total = sum(self.counts.values())
        if total == 0:
            return "no completed episodes"
        parts = [
            f"{name}={count} ({100 * count / total:.0f}%)" for name, count in self.counts.items()
        ]
        return f"{total} episodes: " + ", ".join(parts)


class CheckpointWithSidecarCallback(CheckpointCallback):
    """``CheckpointCallback`` that also writes ``.env.json``/``.train.json``.

    Stock ``CheckpointCallback`` saves only the ``.zip``.  Without sidecars,
    periodic checkpoints cannot be evaluated (the fallback guesses
    ``frame_stack=4`` and mismatches ``--low-ram`` runs).  This subclass
    writes both sidecars next to every periodic checkpoint, so crash-recovery
    checkpoints are directly usable with ``evaluate``/``play``.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        train_cfg: TrainConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._env_cfg = env_cfg
        self._train_cfg = train_cfg

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            if os.path.exists(model_path):
                save_env_config_for_model(self._env_cfg, model_path)
                save_train_config_for_model(self._train_cfg, model_path)
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on ViZDoom.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Scenarios:\n" + "\n".join(f"  {spec.summary()}" for spec in SCENARIOS.values()),
    )
    parser.add_argument(
        "--scenario",
        default="my_way_home",
        choices=scenario_names(),
        help="ViZDoom scenario to train on.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: 100000, or 2048 with --low-ram). "
        "When resuming, this is the *total* target, not an increment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device (default: auto — CUDA on Ubuntu GPU, else CPU).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Rollout steps per PPO update (default: 1024, low-ram: 256).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PPO minibatch size (default: 64, low-ram: 32).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="PPO entropy coefficient (default: 0.01). Raise to ~0.05 for combat "
        "scenarios, where a policy that never shoots can score better than one "
        "that shoots randomly and the agent learns to stop shooting.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel ViZDoom processes (default: 1). "
        "On the Ubuntu GPU workstation use 4-8; on the 2 GB laptop keep 1.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Save a checkpoint every N timesteps (default: 10000).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Run a deterministic evaluation every N timesteps (default: 10000).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Episodes per evaluation (default: 3).",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="Frames stacked per observation (default: 4, low-ram: 1).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Tics executed per agent step (default: 4).",
    )
    parser.add_argument(
        "--save-dir", default="checkpoints", help="Directory for model checkpoints."
    )
    parser.add_argument("--log-dir", default="runs", help="Directory for TensorBoard and CSV logs.")
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help="Low-memory preset (frame_stack=1, n_steps=256, batch_size=32, CPU).",
    )
    parser.add_argument(
        "--engine-log",
        action="store_true",
        help="Show the ViZDoom engine log (hidden by default; it is very noisy).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the ViZDoom window while training (needs a display).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a checkpoint (default path: <save-dir>/<scenario>_latest.zip).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Deprecated no-op: runs start fresh unless --resume is given.",
    )
    args = parser.parse_args(argv)

    if args.fresh and args.resume:
        parser.error("--fresh and --resume are mutually exclusive")
    return args


def _interruptible_signals() -> list[int]:
    """Signals that should trigger a clean save (platform-dependent).

    Linux: ``SIGTERM`` (``timeout``/systemd) + ``SIGINT`` (Ctrl-C).
    Windows: ``SIGINT`` (Ctrl-C) + ``SIGBREAK`` (Ctrl-Break) + ``SIGTERM``
    where available.  Without this, schedulers kill the process and leave
    ViZDoom children orphaned.
    """
    signals = []
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signals.append(sig)
    return signals


class SigtermToInterrupt:
    """Turn termination signals into KeyboardInterrupt so saving still happens.

    Without this, ``timeout``/systemd/job schedulers kill the Python process
    and leave the ViZDoom child processes orphaned.  Those orphans also
    inherit the parent's stdout pipe, which can hang whatever supervises
    the run.  Works on both Linux and Windows (PowerShell).
    """

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> SigtermToInterrupt:
        def _handler(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        for sig in _interruptible_signals():
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, _handler)
            except (ValueError, OSError, RuntimeError):  # pragma: no cover - non-main thread
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for sig, prev in self._previous.items():
            with contextlib.suppress(ValueError, OSError, RuntimeError):
                signal.signal(sig, prev)


def set_global_seeds(seed: int) -> None:
    """Seed every RNG we can reach, so short runs are reproducible.

    Seeds Python, NumPy, and PyTorch (CPU *and* CUDA when available) and
    pins cuDNN to deterministic mode.  Note: CUDA kernels with
    non-deterministic algorithms can still vary run-to-run; set
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` for stricter determinism.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - very old torch
        pass


def make_vec_env(
    env_cfg: EnvConfig,
    n_envs: int,
    monitor_path: str | None = None,
    info_keywords: Sequence[str] = ("is_success",),
) -> VecEnv:
    """Create ``n_envs`` ViZDoom instances with distinct, reproducible seeds.

    ``n_envs == 1`` uses :class:`DummyVecEnv` (in-process, cheapest).
    ``n_envs > 1`` uses :class:`SubprocVecEnv` with ``spawn`` — the only
    start method that works identically on Windows and Linux.  Workers are
    CPU-only (ViZDoom rendering); the PPO model may still run on CUDA in
    the main process.
    """
    factories: list[Callable[[], Any]] = []
    for rank in range(n_envs):
        worker_cfg = replace(env_cfg, seed=env_cfg.seed + rank)

        def _init(cfg: EnvConfig = worker_cfg) -> VizDoomEnv:
            env = VizDoomEnv(config=cfg)
            env.reset(seed=cfg.seed)
            return env

        factories.append(_init)

    venv: VecEnv = (
        DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    )
    if monitor_path:
        venv = VecMonitor(venv, filename=monitor_path, info_keywords=tuple(info_keywords))
    return venv


def resolve_resume_path(requested: str, save_dir: str, scenario: str) -> str:
    """Turn ``--resume`` (``latest`` or a path) into a concrete checkpoint path."""
    if requested == "latest":
        path = os.path.join(save_dir, f"{scenario}_latest.zip")
        if not os.path.exists(path):
            raise SystemExit(
                f"Nothing to resume: {path} does not exist yet. "
                "Run without --resume to start from scratch."
            )
        return path
    if not os.path.exists(requested):
        raise SystemExit(f"Checkpoint to resume from not found: {requested}")
    return requested


def latest_pointer_path(save_dir: str, scenario: str) -> str:
    return os.path.join(save_dir, f"{scenario}_latest.zip")


def unique_dir(path: str) -> str:
    """Return ``path`` or ``path_2``/``path_3``... so runs never overwrite.

    Two runs started within the same second would otherwise share a directory
    named from a timestamp and silently clobber each other's checkpoints.
    """
    if not os.path.exists(path):
        return path
    suffix = 2
    while os.path.exists(f"{path}_{suffix}"):
        suffix += 1
    return f"{path}_{suffix}"


def promote_to_latest(model_path: str, save_dir: str, scenario: str) -> str:
    """Copy ``model_path`` (and its sidecars) to the stable ``_latest`` name."""
    os.makedirs(save_dir, exist_ok=True)
    latest = latest_pointer_path(save_dir, scenario)
    shutil.copyfile(model_path, latest)
    for suffix in (".env.json", ".train.json"):
        sidecar = f"{model_path}{suffix}"
        if os.path.exists(sidecar):
            shutil.copyfile(sidecar, f"{latest}{suffix}")
    return latest


def _package_versions() -> dict[str, str]:
    import gymnasium
    import stable_baselines3
    import torch
    import vizdoom

    return {
        "python": sys.version.split()[0],
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "vizdoom": getattr(vizdoom, "__version__", "unknown"),
        "numpy": np.__version__,
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _save_model_and_sidecar(
    model: PPO, path: str, env_cfg: EnvConfig, train_cfg: TrainConfig | None = None
) -> None:
    model.save(path)
    save_env_config_for_model(env_cfg, path)
    if train_cfg is not None:
        save_train_config_for_model(train_cfg, path)


def _warn_on_resume_drift(
    args: argparse.Namespace,
    stored_env: EnvConfig,
    resume_path: str,
    train_cfg: TrainConfig,
) -> None:
    """Warn about CLI flags that will be ignored because the sidecar wins."""
    for flag, cli_val, stored_val in (
        ("--frame-stack", args.frame_stack, stored_env.frame_stack),
        ("--frame-skip", args.frame_skip, stored_env.frame_skip),
    ):
        if cli_val is not None and cli_val != stored_val:
            print(
                f"[Resume] Note: {flag} {cli_val} ignored; "
                f"the checkpoint was trained with {flag[2:].replace('-', '_')}={stored_val}."
            )
    stored_train = load_train_config_for_model(resume_path)
    if stored_train is not None:
        diffs = [
            f"{name}: CLI={getattr(train_cfg, name)!r} vs checkpoint={getattr(stored_train, name)!r}"
            for name in (
                "n_steps",
                "batch_size",
                "n_epochs",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "n_envs",
                "device",
            )
            if getattr(train_cfg, name) != getattr(stored_train, name)
        ]
        if diffs:
            print(
                "[Resume] Note: training hyper-parameters differ from the "
                "checkpoint's .train.json (PPO hyper-parameters are restored "
                "from the checkpoint; device and run-control flags use CLI values):"
            )
            for line in diffs:
                print(f"           {line}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.render and not has_display():
        print(
            "[Display] Error: --render needs a display, but no display was "
            "detected (DISPLAY/WAYLAND_DISPLAY on Linux, interactive session "
            "on Windows). Re-run without --render.",
            file=sys.stderr,
        )
        return 2

    try:
        env_cfg, train_cfg = build_configs(
            args.scenario,
            low_ram=args.low_ram,
            timesteps=args.timesteps,
            seed=args.seed,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_envs=args.n_envs,
            ent_coef=args.ent_coef,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            window_visible=args.render,
        )
    except ValueError as exc:
        print(f"[Config] Error: {exc}", file=sys.stderr)
        return 2

    env_cfg.silence_engine = not args.engine_log
    set_global_seeds(args.seed)
    scenario = SCENARIOS[args.scenario]

    resume_path: str | None = None
    if args.resume:
        try:
            resume_path = resolve_resume_path(args.resume, args.save_dir, args.scenario)
        except SystemExit as exc:
            print(f"[Resume] Error: {exc}", file=sys.stderr)
            return 2
        try:
            stored_cfg = load_env_config_for_model(resume_path)
        except ValueError as exc:
            print(f"[Resume] Error: {exc}", file=sys.stderr)
            return 2
        if stored_cfg is not None:
            try:
                _warn_on_resume_drift(args, stored_cfg, resume_path, train_cfg)
            except ValueError as exc:
                print(f"[Resume] Error: {exc}", file=sys.stderr)
                return 2
            stored_cfg.window_visible = args.render
            stored_cfg.silence_engine = not args.engine_log
            env_cfg = stored_cfg
            print(f"[Resume] Loaded env config from {resume_path}.env.json")
        else:
            print(
                "[Resume] Warning: no .env.json sidecar found; "
                "assuming the current flags match the checkpoint."
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.scenario}_{timestamp}"
    run_log_dir = unique_dir(os.path.join(args.log_dir, run_name))
    run_ckpt_dir = unique_dir(os.path.join(args.save_dir, run_name))
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    print("=" * 70)
    print(scenario.summary())
    print("=" * 70)
    print(f"Run directory:  {run_ckpt_dir}")
    print(f"Log directory:  {run_log_dir}")
    print(
        f"Observation:    {env_cfg.observation_shape} uint8 "
        f"({'grayscale' if env_cfg.grayscale else 'RGB'}, frame_skip={env_cfg.frame_skip})"
    )
    print(
        f"Target:         {train_cfg.total_timesteps:,} timesteps "
        f"({train_cfg.n_envs} env(s), n_steps={train_cfg.n_steps}, batch={train_cfg.batch_size})"
    )
    info = gpu_info()
    if info["cuda_available"]:
        print(f"Device:         CUDA ({info['device_name']}, {info['total_memory_mb']} MB)")
    else:
        print("Device:         CPU (no CUDA GPU detected)")

    vec_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    try:
        vec_env = make_vec_env(
            env_cfg,
            train_cfg.n_envs,
            monitor_path=os.path.join(run_log_dir, "monitor.csv"),
        )
        eval_env = make_vec_env(
            replace(env_cfg, seed=env_cfg.seed + 1000, window_visible=False),
            1,
            monitor_path=os.path.join(run_log_dir, "eval_monitor.csv"),
        )
    except Exception as exc:
        print(f"[Env] Error: failed to create environments: {exc}", file=sys.stderr)
        if vec_env is not None:
            vec_env.close()
        if eval_env is not None:
            eval_env.close()
        return 2
    action_names: list[str] = list(vec_env.get_attr("action_names")[0])
    print(f"Actions:        {', '.join(action_names)}")

    outcome_counter = OutcomeCounterCallback()

    with SigtermToInterrupt():
        return _run_training(
            args=args,
            env_cfg=env_cfg,
            train_cfg=train_cfg,
            vec_env=vec_env,
            eval_env=eval_env,
            action_names=action_names,
            outcome_counter=outcome_counter,
            run_log_dir=run_log_dir,
            run_ckpt_dir=run_ckpt_dir,
            run_name=run_name,
            resume_path=resume_path,
        )


def _run_training(
    *,
    args: argparse.Namespace,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    vec_env: VecEnv,
    eval_env: VecEnv,
    action_names: list[str],
    outcome_counter: OutcomeCounterCallback,
    run_log_dir: str,
    run_ckpt_dir: str,
    run_name: str,
    resume_path: str | None,
) -> int:
    model: PPO | None = None
    start_timesteps = 0

    try:
        if resume_path:
            device = resolve_device(train_cfg.device)
            try:
                model = PPO.load(resume_path, env=vec_env, device=device)
            except Exception as exc:
                print(f"[Resume] Error: failed to load {resume_path!r}: {exc}", file=sys.stderr)
                return 2
            if model.action_space != vec_env.action_space:
                print(
                    "[Resume] Error: Checkpoint action space "
                    f"({model.action_space}) does not match the environment "
                    f"({vec_env.action_space}). The scenario or action catalogue "
                    "changed since this checkpoint was trained; retrain from scratch.",
                    file=sys.stderr,
                )
                return 2
            if model.observation_space.shape != vec_env.observation_space.shape:
                print(
                    "[Resume] Error: Checkpoint observation shape "
                    f"{model.observation_space.shape} != environment "
                    f"{vec_env.observation_space.shape}.",
                    file=sys.stderr,
                )
                return 2
            # PPO.load restores optimizer/rollout settings. Record those actual
            # values rather than the fresh-run defaults supplied by the CLI.
            for name in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef"):
                setattr(train_cfg, name, getattr(model, name))
            train_cfg.learning_rate = float(model.lr_schedule(1.0))
            train_cfg.clip_range = float(model.clip_range(1.0))
            start_timesteps = int(model.num_timesteps)
            print(f"[Resume] Continuing from {start_timesteps:,} timesteps.")
        else:
            model = create_ppo_model(
                env=vec_env,
                train_config=train_cfg,
                seed=args.seed,
                tensorboard_log=run_log_dir,
            )

        remaining = train_cfg.total_timesteps - start_timesteps
        if remaining <= 0:
            print(
                f"[Resume] Already trained for {start_timesteps:,} timesteps >= "
                f"target {train_cfg.total_timesteps:,}. "
                "Pass a larger --timesteps to continue.",
                file=sys.stderr,
            )
            return 1

        model.set_logger(configure(run_log_dir, ["stdout", "csv", "tensorboard"]))

        callbacks: list[BaseCallback] = []
        checkpoint_freq = max(1, train_cfg.checkpoint_freq // train_cfg.n_envs)
        callbacks.append(
            CheckpointWithSidecarCallback(
                env_cfg,
                train_cfg,
                save_freq=checkpoint_freq,
                save_path=run_ckpt_dir,
                name_prefix=f"{args.scenario}_ckpt",
                verbose=1,
            )
        )
        eval_freq = max(1, train_cfg.eval_freq // train_cfg.n_envs)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=run_ckpt_dir,
                log_path=run_log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=train_cfg.n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
        )
        callbacks.append(outcome_counter)

        print(
            f"Training for {remaining:,} timesteps "
            f"(checkpoint every {train_cfg.checkpoint_freq:,}, "
            f"eval every {train_cfg.eval_freq:,})..."
        )
        # Stable-Baselines3 treats ``total_timesteps`` as *additional* steps when
        # ``reset_num_timesteps=False``; pass the remaining budget so that
        # ``--timesteps`` always means "train until this total is reached".
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not bool(resume_path),
        )

    except KeyboardInterrupt:
        if model is None:
            print("\n[Interrupted] Nothing to save yet.")
            return 130
        print("\n[Interrupted] Saving current model...")
        final_path = os.path.join(run_ckpt_dir, f"{args.scenario}_final.zip")
        _save_model_and_sidecar(model, final_path, env_cfg, train_cfg)
        promote_to_latest(final_path, args.save_dir, args.scenario)
        print(f"[Interrupted] Progress saved to: {final_path}")
        return 130
    finally:
        best_candidate = os.path.join(run_ckpt_dir, "best_model.zip")
        if os.path.exists(best_candidate):
            save_env_config_for_model(env_cfg, best_candidate)
            save_train_config_for_model(train_cfg, best_candidate)

        vec_env.close()
        eval_env.close()

    if model is None:  # pragma: no cover - only if learn() never started
        print("[Train] Error: no model was trained.", file=sys.stderr)
        return 1

    final_path = os.path.join(run_ckpt_dir, f"{args.scenario}_final.zip")
    _save_model_and_sidecar(model, final_path, env_cfg, train_cfg)
    latest_path = promote_to_latest(final_path, args.save_dir, args.scenario)

    _write_json(
        os.path.join(run_ckpt_dir, "run.json"),
        {
            "run_name": run_name,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "scenario": args.scenario,
            "argv": vars(args),
            "num_timesteps": int(model.num_timesteps),
            "env_config": env_cfg.to_dict(),
            "train_config": asdict(train_cfg),
            "action_names": action_names,
            "observation_shape": list(env_cfg.observation_shape),
            "episode_outcomes": outcome_counter.counts,
            "packages": _package_versions(),
            "gpu": gpu_info(),
        },
    )

    print("\n" + "=" * 70)
    print(f"Training complete after {int(model.num_timesteps):,} timesteps.")
    print(f"Final model:  {final_path}")
    print(f"Latest copy:  {latest_path}")
    print(f"Episodes:     {outcome_counter.summary()}")
    print("Next steps:")
    print(f"  python -m src.evaluate --checkpoint {latest_path} --episodes 5")
    print(f"  python -m src.play --checkpoint {latest_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
