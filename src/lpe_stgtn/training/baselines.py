"""Baseline training and evaluation runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lpe_stgtn.config import load_yaml
from lpe_stgtn.data.datasets import (
    ProcessedDatasetBundle,
    WindowedDemandDataset,
    load_processed_dataset,
)
from lpe_stgtn.evaluation.metrics import mae, mape, rmse
from lpe_stgtn.models.baselines.lstm import LSTMBaseline
from lpe_stgtn.models.baselines.persistence import PersistenceBaseline
from lpe_stgtn.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class BaselineRunSummary:
    """Summary of one baseline run."""

    experiment_name: str
    model_name: str
    report_path: Path
    metrics: dict[str, dict[str, float]]
    checkpoint_path: Path | None = None


def run_baseline_experiment(
    experiment_config_path: Path,
    *,
    project_root: Path,
    training_config_path: Path | None = None,
) -> BaselineRunSummary:
    """Load configs, run the selected baseline, and write a JSON report."""
    experiment_config = load_yaml(experiment_config_path)
    training_defaults = load_yaml(training_config_path) if training_config_path is not None else {}
    merged_config = deep_merge(training_defaults, experiment_config)

    experiment_name = _require_string(merged_config, "experiment_name")
    processed_data_dir = project_root / _require_string(merged_config, "processed_data_dir")
    model_config = _require_mapping(merged_config, "model")
    model_name = _require_string(model_config, "name")
    artifacts_config = _require_mapping(merged_config, "artifacts")
    report_dir = project_root / _require_string(artifacts_config, "report_dir")
    checkpoint_dir = project_root / str(
        artifacts_config.get("checkpoint_dir", artifacts_config["report_dir"])
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reproducibility = merged_config.get("reproducibility", {})
    seed_everything(
        int(reproducibility.get("seed", 2026)),
        deterministic_torch=bool(reproducibility.get("deterministic", True)),
    )

    bundle = load_processed_dataset(processed_data_dir)

    if model_name == "persistence":
        metrics = run_persistence_baseline(bundle)
        checkpoint_path = None
        training_history: list[dict[str, float | int]] = []
    elif model_name == "lstm":
        metrics, checkpoint_path, training_history = train_lstm_baseline(
            bundle,
            experiment_name=experiment_name,
            model_config=model_config,
            trainer_config=_require_mapping(merged_config, "trainer"),
            optimizer_config=_require_mapping(merged_config, "optimizer"),
            checkpoint_dir=checkpoint_dir,
            device_name=str(merged_config.get("runtime", {}).get("device", "auto")),
        )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    report_path = report_dir / f"{experiment_name}.json"
    report_payload = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "experiment_config_path": str(experiment_config_path),
        "training_config_path": str(training_config_path) if training_config_path else None,
        "processed_data_dir": str(processed_data_dir),
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "training_history": training_history,
        "dataset": {
            "history_steps": bundle.history_steps,
            "forecast_steps": bundle.forecast_steps,
            "num_zones": bundle.num_zones,
            "normalization_mean": bundle.normalization_mean,
            "normalization_std": bundle.normalization_std,
        },
        "study_area_summary": bundle.metadata.get("study_area_summary", {}),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return BaselineRunSummary(
        experiment_name=experiment_name,
        model_name=model_name,
        report_path=report_path,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
    )


def run_persistence_baseline(bundle: ProcessedDatasetBundle) -> dict[str, dict[str, float]]:
    """Evaluate the persistence baseline on validation and test splits."""
    baseline = PersistenceBaseline(bundle.forecast_steps)
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        x_raw, y_raw = bundle.build_windows(split, normalized=False)
        predictions = baseline.predict(torch.as_tensor(x_raw, dtype=torch.float32)).numpy()
        metrics_by_split[split] = compute_metrics(y_raw, predictions)
    return metrics_by_split


def train_lstm_baseline(
    bundle: ProcessedDatasetBundle,
    *,
    experiment_name: str,
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    checkpoint_dir: Path,
    device_name: str,
) -> tuple[dict[str, dict[str, float]], Path, list[dict[str, float | int]]]:
    """Train an LSTM baseline and return validation/test metrics plus report details."""
    device = resolve_device(device_name)
    model = LSTMBaseline(
        input_dim=bundle.num_zones,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        forecast_steps=bundle.forecast_steps,
        num_layers=int(model_config.get("num_layers", 1)),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)

    train_dataset = WindowedDemandDataset(bundle, split="train", normalized=True)
    validation_dataset = WindowedDemandDataset(bundle, split="validation", normalized=True)

    batch_size = int(trainer_config.get("batch_size", 64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(optimizer_config.get("learning_rate", 0.001))
    )
    max_epochs = int(trainer_config.get("max_epochs", 20))
    patience = int(trainer_config.get("early_stopping_patience", 5))
    gradient_clip_norm = trainer_config.get("gradient_clip_norm")

    checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_mae = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_metrics, validation_loss = evaluate_model(
            model,
            validation_loader,
            bundle=bundle,
            criterion=criterion,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_mape": validation_metrics["mape"],
            }
        )

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state_dict is None:
        raise RuntimeError("LSTM training finished without producing a best checkpoint.")

    torch.save(
        {
            "model_name": "lstm",
            "model_config": model_config,
            "best_epoch": best_epoch,
            "state_dict": best_state_dict,
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state_dict)
    test_dataset = WindowedDemandDataset(bundle, split="test", normalized=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    validation_metrics, _ = evaluate_model(
        model,
        validation_loader,
        bundle=bundle,
        criterion=criterion,
        device=device,
    )
    test_metrics, _ = evaluate_model(
        model,
        test_loader,
        bundle=bundle,
        criterion=criterion,
        device=device,
    )
    metrics = {"validation": validation_metrics, "test": test_metrics}
    return metrics, checkpoint_path, history


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> float:
    """Run one LSTM training epoch on normalized demand windows."""
    model.train()
    losses: list[float] = []
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss = criterion(prediction, target)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    bundle: ProcessedDatasetBundle,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Evaluate the model and report raw-scale metrics plus normalized validation loss."""
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            prediction = model(history)
            loss = criterion(prediction, target)
            losses.append(float(loss.detach().cpu().item()))

            prediction_raw = bundle.denormalize(prediction.cpu().numpy())
            target_raw = bundle.denormalize(target.cpu().numpy())
            predictions.append(prediction_raw)
            targets.append(target_raw)

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    return compute_metrics(y_true, y_pred), float(np.mean(losses))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the paper metrics on a forecast tensor."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def format_baseline_run_summary(summary: BaselineRunSummary) -> str:
    """Render a terminal-friendly baseline run summary."""
    lines = [
        f"Experiment: {summary.experiment_name}",
        f"Model: {summary.model_name}",
    ]
    for split_name, split_metrics in summary.metrics.items():
        lines.append(
            f"{split_name.title()} metrics: "
            f"MAE={split_metrics['mae']:.6f}, "
            f"RMSE={split_metrics['rmse']:.6f}, "
            f"MAPE={split_metrics['mape']:.6f}"
        )
    if summary.checkpoint_path is not None:
        lines.append(f"Wrote checkpoint: {summary.checkpoint_path}")
    lines.append(f"Wrote report: {summary.report_path}")
    return "\n".join(lines)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested runtime device with an `auto` fallback."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for config key '{key}'")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for config key '{key}'")
    return value
