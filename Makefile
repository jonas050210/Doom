#!/usr/bin/env make
# Works on Ubuntu/Linux and on Windows (Git Bash / WSL / PowerShell via make).
# On Windows, `make setup` dispatches to setup.ps1; elsewhere to setup.sh.

PYTHON ?= python

ifeq ($(OS),Windows_NT)
SETUP_CMD = powershell -ExecutionPolicy Bypass -File setup.ps1
else
SETUP_CMD = ./setup.sh
endif

.PHONY: help setup test test-all lint format smoke train demo evaluate gpu-check clean

help:
	@echo "make setup      - create .venv and install dependencies (CUDA torch on GPU)"
	@echo "make test       - fast unit tests (no training)"
	@echo "make test-all   - unit tests + slow end-to-end tests"
	@echo "make lint       - ruff check"
	@echo "make format     - ruff format"
	@echo "make gpu-check  - print torch/CUDA status (GPU unlock check)"
	@echo "make smoke      - shortest possible training run (2048 steps)"
	@echo "make train      - default training run (100k steps, my_way_home)"
	@echo "make demo       - train 10k steps, evaluate, then watch the agent"
	@echo "make clean      - remove caches (keeps runs/ and checkpoints/)"

setup:
	$(SETUP_CMD)

test:
	$(PYTHON) -m pytest

test-all:
	$(PYTHON) -m pytest -o addopts=""

lint:
	ruff check src tests

format:
	ruff format src tests

gpu-check:
	$(PYTHON) -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

smoke:
	$(PYTHON) -m src.train --scenario my_way_home --low-ram --timesteps 2048

train:
	$(PYTHON) -m src.train --scenario my_way_home

demo:
	$(PYTHON) -m src.train --scenario basic --timesteps 10000
	$(PYTHON) -m src.evaluate --checkpoint checkpoints/basic_latest.zip --episodes 5
	$(PYTHON) -m src.play --checkpoint checkpoints/basic_latest.zip --episodes 2

evaluate:
	$(PYTHON) -m src.evaluate --checkpoint checkpoints/my_way_home_latest.zip --episodes 10

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache']]"
