# NeuralDoom-Lab

**NeuralDoom-Lab** is an offline reinforcement-learning (RL) research project in which an AI agent learns to play free **ViZDoom** scenarios from raw screen pixels.

> **Offline research only.** This project is strictly for local, offline reinforcement-learning research using ViZDoom's official Python API and the free Freedoom-compatible assets that ship with the `vizdoom` package. It does **not** require commercial Doom WAD files and is not intended for online play or automation.

---

## What happens in one episode

1. **Observe** — ViZDoom renders `320x240`; we resize to `84x84` grayscale (or RGB) `uint8` and stack the last `frame_stack` frames along the channel axis, so the policy can perceive motion.
2. **Act** — the agent picks one of the scenario's discrete actions. The action set is derived from the buttons the scenario actually exposes (see below).
3. **Reward** — whatever the `.cfg` scenario defines; ranges per scenario are listed in the table further down.
4. **Learn** — PPO from Stable-Baselines3 with a `CnnPolicy` updates the network.

### Three design decisions worth knowing

**1. The action space is derived per scenario, never hard-coded.**
ViZDoom scenarios declare which buttons exist, and pressing a missing button is a no-op. A static action table therefore creates dead actions: `basic` only exposes strafe-left/right and attack, so a "forward" action does nothing at all. This project filters a catalogue of named actions (`src/actions.py`) against `game.get_available_buttons()`, drops anything unsupported, and **fails loudly** if an action would press no button or duplicate another one.

**2. Episodes report *why* they ended, not just "done".**
The wrapper maps ViZDoom's own signals to Gymnasium semantics:

| End reason | `terminated` | `truncated` | `info["episode_outcome"]` | `info["is_success"]` |
|---|---|---|---|---|
| Reached the scenario goal | `True` | `False` | `"goal"` | `True` |
| Player died | `True` | `False` | `"death"` | `False` |
| `episode_timeout` reached | `False` | `True` | `"timeout"` | `False` |

Getting this right matters: PPO bootstraps the value function at `truncated` states, so reporting timeouts as terminal states biases the value target on every long episode.

**3. Every checkpoint carries the environment config that produced it.**
Each saved model gets a `<model>.zip.env.json` sidecar with the observation shape, frame stack/skip, and scenario. `evaluate` and `play` read it instead of guessing, so a model trained with `--low-ram` (one stacked frame) can never be silently evaluated with four. Mismatched action or observation spaces raise a clear error.

---

## Scenarios

| Scenario | Level | Actions | Buttons exposed by ViZDoom | Timeout | Reward per episode | What "success" means |
|---|---|---|---|---|---|---|
| `my_way_home` | 1 — Navigation | 7 | turn L/R, forward, strafe L/R | 2100 tics | ≈ [-0.25, +1] | reached the vest (+1) |
| `health_gathering` | 2 — Survival | 5 | turn L/R, forward | 2100 tics | ≈ [-105, +2105] | no goal; score is survival time (+1/tic, -100 death) |
| `basic` | 3 — Combat intro | 3 | strafe L/R, attack | 300 tics | ≈ [-320, +112] | killed the monster (+106) |
| `defend_the_center` | 4 — Arena combat | 3 | turn L/R, attack | 2100 tics | ≈ [-2, +30] | each kill +1; episode ends in death (-1) |
| `deadly_corridor` | 5 — Hard combat | 9 | all movement + attack | 2100 tics | ≈ [-1100, +1010] | reached the vest (+1000) |

`--help` prints the same information, and `src/config.py` is the single source of truth (values are asserted against the shipped `.cfg` files in the test suite).

Because the scenarios have such different reward scales, **do not compare mean rewards across scenarios**; compare goal/timeout/death rates instead.

---

## Project layout

```text
NeuralDoom-Lab/
├── README.md
├── Makefile                # make test / lint / smoke / train / demo
├── pyproject.toml          # pytest + ruff configuration
├── requirements.txt        # runtime deps (pinned with upper bounds)
├── requirements-dev.txt    # + pytest, pytest-cov, ruff
├── setup.sh / setup.ps1    # venv + deps + import smoke check
├── src/
│   ├── config.py           # scenario metadata, validated configs, sidecars
│   ├── actions.py          # button-aware action catalogue
│   ├── env.py              # Gymnasium ViZDoom wrapper (outcome-aware)
│   ├── model.py            # PPO builder, device resolution, checkpoint loading
│   ├── train.py            # training loop, checkpointing, evaluation, resume
│   ├── evaluate.py         # deterministic evaluation + JSON report
│   └── play.py             # visual demo with headless fallback
├── tests/
│   ├── test_actions.py     # no dead/duplicate actions per scenario
│   ├── test_config.py      # cfg/metadata agreement, validation, sidecars
│   ├── test_env.py         # obs, determinism, termination vs truncation
│   ├── test_model.py       # device handling, PPO wiring (no ViZDoom needed)
│   ├── test_train.py       # CLI, resume paths, callbacks
│   ├── test_evaluate.py    # summaries, sidecar use, JSON report
│   └── test_smoke_e2e.py   # train → resume → evaluate → play (marked slow)
├── runs/                   # TensorBoard/CSV/monitor logs (gitignored)
└── checkpoints/            # models + .env.json sidecars (gitignored)
```

---

## Quickstart

### Linux / macOS

```bash
git clone <your-fork-url> && cd Doom
chmod +x setup.sh
./setup.sh              # creates .venv, installs deps, verifies imports
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Both scripts end by importing `vizdoom`, `gymnasium`, `stable_baselines3`, `torch`, and `src.config`, so a broken install fails immediately instead of on your first training run.

### Verify

```bash
python -m pytest -q              # ~90 fast tests, about 10 s
python -m src.play --random --episodes 1
```

### Train

```bash
# default: 100k steps, my_way_home, auto device (CUDA if available)
python -m src.train --scenario my_way_home

# short run for a laptop
python -m src.train --scenario my_way_home --timesteps 10000

# 2 GB RAM / integrated GPU preset
python -m src.train --scenario my_way_home --low-ram --timesteps 2048

# two parallel ViZDoom processes
python -m src.train --scenario basic --n-envs 2 --timesteps 50000

# watch it learn (needs a display)
python -m src.train --scenario basic --render --timesteps 5000
```

`--low-ram` only changes **defaults** (`frame_stack=1`, `n_steps=256`, `batch_size=32`, `device=cpu`). Any flag you pass explicitly wins.

### Resume

```bash
python -m src.train --scenario my_way_home --resume --timesteps 200000
```

`--resume` (or `--resume path/to/model.zip`) continues from `<save-dir>/<scenario>_latest.zip` and restores the saved env config. `--timesteps` is always the **total target**: resuming a 100k model with `--timesteps 150000` trains 50k more and says so. If the target is already reached, it says that and exits instead of silently doing nothing.

### Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/my_way_home_latest.zip --episodes 10
```

Prints mean/std/min/max reward, goal/death/timeout rates, mean episode length, and writes `<checkpoint>.eval.json` with per-episode records plus the env config — so results are diffable in git and comparable across runs.

### Watch the agent

```bash
python -m src.play --checkpoint checkpoints/my_way_home_latest.zip --episodes 3
```

Falls back to headless automatically when no `DISPLAY`/`WAYLAND_DISPLAY` exists.

### TensorBoard

```bash
tensorboard --logdir runs
```

Each run logs TensorBoard events, `progress.csv`, `monitor.csv` (episode rewards), `eval_monitor.csv`, and `run.json` (exact args, package versions, action names, episode outcome counts).

---

## Checkpoint layout

```text
checkpoints/
├── my_way_home_20260101_120000/
│   ├── my_way_home_ckpt_10000_steps.zip
│   ├── best_model.zip                 # best deterministic evaluation
│   ├── my_way_home_final.zip
│   ├── *.zip.env.json                 # sidecar for each model
│   └── run.json                       # metadata of this run
└── my_way_home_latest.zip             # stable pointer to the newest model
```

Runs are timestamped and never overwrite each other (a run started in the same second gets a `_2` suffix). Checkpoints are written every `--checkpoint-freq` steps, so a crash costs at most that many steps.

`Ctrl-C` **and** `SIGTERM` (what `timeout`, systemd, and most schedulers send) both save the current model and shut the ViZDoom processes down cleanly, so interrupting a run never leaves stray game processes behind.

> Loading a checkpoint runs `torch.load` underneath, which unpickles code. Only load models you produced yourself or trust.

---

## Testing

```bash
python -m pytest                  # fast suite (default)
python -m pytest -m slow          # end-to-end: train → resume → evaluate → play
python -m pytest --cov=src        # with coverage
ruff check src tests && ruff format --check src tests
```

The fast suite covers action-matrix invariants per scenario, config validation and sidecar round-trips, observation determinism and frame stacking, termination-vs-truncation semantics, device handling, CLI parsing, resume-path logic, and evaluation summaries. `.github/workflows/ci.yml` runs lint plus the full suite on Python 3.11 and 3.12, and the end-to-end smoke job separately.

---

## Reproducibility

* `--seed` seeds Python, NumPy, Torch, PPO, and every env worker (`base + rank`).
* Evaluation uses fixed per-episode seeds (`--seed + episode`).
* `run.json` records args, resolved config, action names, episode outcome counts, and library versions.
* `requirements.txt` uses upper bounds because PPO checkpoints are version-sensitive.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run `python -m ...` from the repository root (or use `pytest`, which sets `pythonpath` via `pyproject.toml`). |
| `--render` exits with code 2 | No display detected. Drop `--render`, or run under `xvfb-run`. |
| CUDA OOM on 8 GB | Lower `--batch-size` / `--n-steps`, or use `--device cpu`. Defaults are small (84x84, 1M-parameter CNN). |
| `nvidia-smi` fails / `torch.cuda.is_available()` is `False` | Install NVIDIA drivers; training falls back to CPU automatically. |
| Checkpoint won't load (`action space mismatch`) | The scenario or action catalogue changed since training. Retrain, or evaluate with the scenario it was trained on. |
| `setup.sh` fails on Python 3.13+ | Prebuilt `vizdoom`/`torch` wheels may not exist yet; use 3.11 or 3.12. |
| Training is slow on one env | `--n-envs 2`/`4` runs parallel ViZDoom processes; each adds ~200 MB RAM. |

---

## Scope and licensing

The project ships no game assets: scenarios and Freedoom-compatible WADs come from the `vizdoom` package. No license file is included yet — add one before distributing the code, and check ViZDoom's own license (GPL-3.0 for the engine) if you plan to redistribute binaries.
