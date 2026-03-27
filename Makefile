PYTHON ?= python3
SRC_PATHS := src tests scripts

.PHONY: help venv install format lint test inspect-data prepare-data run-baseline prepare-dirs

help:
	@echo "Available targets:"
	@echo "  make venv          Create a local Python 3.11 virtual environment"
	@echo "  make install       Install the project and dev dependencies"
	@echo "  make format        Run code formatter"
	@echo "  make lint          Run static lint checks"
	@echo "  make test          Run unit tests"
	@echo "  make inspect-data  Run a raw-data sanity inspection"
	@echo "  make prepare-data  Build the stage-1 processed demand dataset"
	@echo "  make run-baseline  Run a baseline experiment (default: persistence)"
	@echo "  make prepare-dirs  Create derived-data and artifact directories"

venv:
	$(PYTHON) -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format $(SRC_PATHS)

lint:
	$(PYTHON) -m ruff check $(SRC_PATHS)

test:
	PYTHONPATH=src $(PYTHON) -m pytest

inspect-data:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli inspect-data --limit 2 --sample-rows 3

prepare-data:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli prepare-data

run-baseline:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli run-baseline

prepare-dirs:
	mkdir -p artifacts/checkpoints artifacts/logs artifacts/reports
	mkdir -p data/external data/interim data/processed
