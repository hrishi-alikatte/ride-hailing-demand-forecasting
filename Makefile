PYTHON ?= python3
SRC_PATHS := src tests scripts

.PHONY: help venv install format lint test inspect-data prepare-data download-road-network build-graphs run-baseline run-full-model compare-zone-tracks prepare-dirs

help:
	@echo "Available targets:"
	@echo "  make venv          Create a local Python 3.11 virtual environment"
	@echo "  make install       Install the project and dev dependencies"
	@echo "  make format        Run code formatter"
	@echo "  make lint          Run static lint checks"
	@echo "  make test          Run unit tests"
	@echo "  make inspect-data  Run a raw-data sanity inspection"
	@echo "  make prepare-data  Build the stage-1 processed demand dataset"
	@echo "  make download-road-network  Download the official NYC centerline road-network asset"
	@echo "  make build-graphs  Build the stage-3 distance and OD-flow graph artifacts"
	@echo "  make run-baseline  Run a baseline experiment (default: persistence)"
	@echo "  make run-full-model  Run the final LPE-STGTN graph model"
	@echo "  make compare-zone-tracks  Compare the main 68-zone track with the 67-zone sensitivity track"
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

download-road-network:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli download-road-network

build-graphs:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli build-graphs

run-baseline:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli run-baseline

run-full-model:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli run-full-model --config configs/experiments/nyc_lpe_stgtn_sensitivity_67.yaml

compare-zone-tracks:
	PYTHONPATH=src $(PYTHON) -m lpe_stgtn.cli compare-zone-tracks

prepare-dirs:
	mkdir -p artifacts/checkpoints artifacts/logs artifacts/reports
	mkdir -p data/external data/interim data/processed
