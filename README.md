# NeuralDoom-Lab

**NeuralDoom-Lab** is an offline reinforcement-learning (RL) research project where an AI agent learns to play free **ViZDoom** scenarios using only raw screen pixels.

> **Offline Research Only:** This project is strictly for local, offline reinforcement-learning research using ViZDoom's official Python API and free, included Freedoom-compatible assets. It does **not** require any commercial Doom WAD files and is not intended for online games or automation.

---

## What Is Reinforcement Learning? (Beginner Overview)

In reinforcement learning, an **agent** (our neural network) interacts with an **environment** (ViZDoom) in a loop:

1. **Observe:** The agent sees the current game screen as an `84x84` grayscale image (stacked 4 frames at a time so it can perceive motion).
2. **Act:** The agent chooses a discrete action (`forward`, `left`, `right`, `forward_left`, or `forward_right`).
3. **Reward:** The environment returns a reward signal (e.g., `+1.0` for reaching the goal vest in `my_way_home`, with a small time penalty each step).
4. **Learn:** Using **PPO (Proximal Policy Optimization)** from Stable-Baselines3, the agent updates its neural network to choose actions that maximize future rewards.

### Project Architecture

```text
NeuralDoom-Lab/
├── README.md            # Project guide, commands, and troubleshooting
├── requirements.txt     # Python dependencies (ViZDoom, Gymnasium, SB3, PyTorch)
├── setup.sh             # Creates .venv and installs dependencies
├── src/
│   ├── config.py        # Safe defaults for 8 GB VRAM + scenario presets
│   ├── actions.py       # Discrete action space -> ViZDoom button mapping
│   ├── env.py           # Gymnasium-compatible ViZDoom wrapper (4x84x84 uint8)
│   ├── model.py         # SB3 PPO builder, CUDA/CPU selection, checkpoint loader
│   ├── train.py         # Training loop with checkpoints, CSV, and TensorBoard
│   ├── evaluate.py      # Deterministic evaluation (reward, completion rate, length)
│   └── play.py          # Visual agent demo (with automatic headless fallback)
├── tests/
│   ├── test_actions.py  # Unit tests for action mapping
│   └── test_env.py      # Environment reset, observation, and rollout tests
├── runs/                # TensorBoard & CSV logs (gitignored)
└── checkpoints/         # Saved model .zip checkpoints (gitignored)
```

---

## Quickstart (Ubuntu Linux)

### 1. Verify NVIDIA Driver & CUDA
```bash
nvidia-smi
```
You should see your GPU (e.g., NVIDIA RTX 4060 Ti 8 GB) and CUDA version.

### 2. Set Up the Environment
**Ubuntu / Linux:**
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### 3. Verify GPU PyTorch
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Run Automated Tests
```bash
python -m pytest -v
```

### 5. Run a Random-Agent Smoke Test
```bash
python -m src.play --random --episodes 1
```

### 6. Start a Short Training Run
```bash
python -m src.train --scenario my_way_home --timesteps 10000
```
For the optional survival scenario (collecting health packs):
```bash
python -m src.train --scenario health_gathering --timesteps 10000
```
For **2 GB RAM laptops** (e.g. i3-1215U on Windows), use `--low-ram` (`frame_stack=1`, `n_steps=256`, `batch_size=32`, CPU):
```bash
python -m src.train --low-ram --timesteps 2048
python -m src.evaluate --checkpoint checkpoints/my_way_home_final.zip --low-ram --episodes 3
python -m src.play --checkpoint checkpoints/my_way_home_final.zip --low-ram --episodes 1
```

### 7. Evaluate a Trained Checkpoint
```bash
python -m src.evaluate --checkpoint checkpoints/my_way_home_final.zip --episodes 10
```

### 8. Watch the Trained Agent Play
```bash
python -m src.play --checkpoint checkpoints/my_way_home_final.zip --episodes 3
```
*(If no display is detected, `play.py` automatically falls back to headless mode.)*

### 9. View Training Metrics in TensorBoard
```bash
tensorboard --logdir runs
```
Then open `http://localhost:6006` in your browser.

---

## Troubleshooting

- **`nvidia-smi` fails or `torch.cuda.is_available()` is `False`:**
  - Ensure NVIDIA drivers are installed (`sudo apt update && sudo apt install nvidia-driver-550` or your distro's recommended driver) and reboot.
  - The project will automatically fall back to CPU so you can still run tests and smoke-training.
- **ViZDoom window / display errors (`DISPLAY` not set or X11 error):**
  - `train.py` and `evaluate.py` run headless by default.
  - `play.py` automatically detects if `DISPLAY`/`WAYLAND_DISPLAY` is missing and switches to headless mode. You can also pass `--headless`.
- **CUDA Out-of-Memory (OOM) on 8 GB VRAM:**
  - The default settings (`84x84` grayscale, `n_steps=1024`, `batch_size=64`) use minimal VRAM.
  - If needed, reduce `--batch-size 32` or `--n-steps 512` in `src/train.py`.
- **`ModuleNotFoundError: No module named 'src'`:**
  - Always run commands from the repository root using `python -m ...` (e.g., `python -m pytest -v`, `python -m src.train`).
