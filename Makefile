#!/usr/bin/env make

PYTHON ?= python

.PHONY: help setup test test-all lint format smoke train demo evaluate clean

help:
	@echo "make setup      - create .venv and install dependencies"
	@echo "make test       - fast unit tests (no training)"
	@echo "make test-all   - unit tests + slow end-to-end tests"
	@echo "make lint       - ruff check"
	@echo "make format     - ruff format"
	@echo "make smoke      - shortest possible training run (2048 steps)"
	@echo "make train      - default training run (100k steps, my_way_home)"
	@echo "make demo       - train 10k steps, evaluate, then watch the agent"
	@echo "make clean      - remove caches (keeps runs/ and checkpoints/)"

setup:
	./setup.sh

test:
	$(PYTHON) -m pytest

test-all:
	$(PYTHON) -m pytest -m "not slow or slow"

lint:
	ruff check src tests

format:
	ruff format src tests

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
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
