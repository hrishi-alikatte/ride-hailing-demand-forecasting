"""Training runner for the full LPE-STGTN architecture."""

from __future__ import annotations

import json
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
from lpe_stgtn.graphs.artifacts import GraphArtifactsBundle, load_graph_artifacts
from lpe_stgtn.models.lpe_stgtn import LPE_STGTN
from lpe_stgtn.training.baselines import (
    BaselineRunSummary,
    deep_merge,
    graph_summary_payload,
    resolve_device,
    validate_graph_alignment,
)


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


def run_lpe_stgtn_experiment(
    experiment_config_path: Path,
    *,
    project_root: Path,
    training_config_path: Path | None = None,
) -> BaselineRunSummary:
    """Load configs, run the full LPE-STGTN model, and write a JSON report."""
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

    from lpe_stgtn.utils.reproducibility import seed_everything
    reproducibility = merged_config.get("reproducibility", {})
    seed_everything(
        int(reproducibility.get("seed", 2026)),
        deterministic_torch=bool(reproducibility.get("deterministic", True)),
    )

    bundle = load_processed_dataset(processed_data_dir)
    graph_data_dir = project_root / str(
        merged_config.get("graph_data_dir", processed_data_dir / "graphs")
    )
    graph_bundle = load_graph_artifacts(graph_data_dir)
    validate_graph_alignment(bundle, graph_bundle)

    if model_name != "lpe_stgtn":
        raise ValueError(f"Unsupported model for this runner: {model_name}")

    metrics, checkpoint_path, training_history = train_lpe_stgtn(
        bundle,
        graph_bundle=graph_bundle,
        experiment_name=experiment_name,
        model_config=model_config,
        trainer_config=_require_mapping(merged_config, "trainer"),
        optimizer_config=_require_mapping(merged_config, "optimizer"),
        checkpoint_dir=checkpoint_dir,
        device_name=str(merged_config.get("runtime", {}).get("device", "auto")),
    )

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
        },
        "graph_data_dir": str(graph_bundle.root_dir),
        "graph_summary": graph_summary_payload(graph_bundle),
        "study_area_summary": bundle.metadata.get("study_area_summary", {}),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return BaselineRunSummary(
        experiment_name=experiment_name,
        model_name=model_name,
        report_path=report_path,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        graph_data_dir=graph_bundle.root_dir,
    )


def train_lpe_stgtn(
    bundle: ProcessedDatasetBundle,
    *,
    graph_bundle: GraphArtifactsBundle,
    experiment_name: str,
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    checkpoint_dir: Path,
    device_name: str,
) -> tuple[dict[str, dict[str, float]], Path, list[dict[str, float | int]]]:
    device = resolve_device(device_name)
    model = LPE_STGTN(
        num_zones=bundle.num_zones,
        forecast_steps=bundle.forecast_steps,
        distance_adjacency=torch.as_tensor(
            graph_bundle.distance_adjacency_normalized,
            dtype=torch.float32,
        ),
        od_flow_adjacency=torch.as_tensor(
            graph_bundle.od_flow_adjacency_normalized,
            dtype=torch.float32,
        ),
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        attention_heads=int(model_config.get("attention_heads", 4)),
        aft_window_size=int(model_config.get("aft_window_size", 4)),
        gru_layers=int(model_config.get("gru_layers", 1)),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)

    checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"
    
    train_dataset = WindowedDemandDataset(bundle, split="train", normalized=True, include_time_features=True)
    validation_dataset = WindowedDemandDataset(bundle, split="validation", normalized=True, include_time_features=True)

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

    scheduler_config = trainer_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type")
    scheduler = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=float(scheduler_config.get("factor", 0.5)), 
            patience=int(scheduler_config.get("patience", 2)),
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=int(scheduler_config.get("T_max", max_epochs)),
        )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch_tuple(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_metrics, validation_loss = _evaluate_model_tuple(
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
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_metrics["mae"])
            else:
                scheduler.step()

        if epochs_without_improvement >= patience:
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_name": "lpe_stgtn",
            "model_config": model_config,
            "state_dict": best_state_dict,
        },
        checkpoint_path,
    )

    test_dataset = WindowedDemandDataset(bundle, split="test", normalized=True, include_time_features=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    validation_metrics, _ = _evaluate_model_tuple(
        model, validation_loader, bundle=bundle, criterion=criterion, device=device
    )
    test_metrics, _ = _evaluate_model_tuple(
        model, test_loader, bundle=bundle, criterion=criterion, device=device
    )
    return {"validation": validation_metrics, "test": test_metrics}, checkpoint_path, history


def _train_one_epoch_tuple(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> float:
    """Run one epoch with `(demand, tod, dow)` tuples."""
    model.train()
    losses: list[float] = []
    for x_tuple, target in loader:
        history, tod, dow = (t.to(device) for t in x_tuple)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(history, tod, dow)
        loss = criterion(prediction, target)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def _evaluate_model_tuple(
    model: nn.Module,
    loader: DataLoader,
    *,
    bundle: ProcessedDatasetBundle,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Evaluate model with tuple inputs."""
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for x_tuple, target in loader:
            history, tod, dow = (t.to(device) for t in x_tuple)
            target = target.to(device)
            
            prediction = model(history, tod, dow)
            loss = criterion(prediction, target)
            losses.append(float(loss.detach().cpu().item()))

            prediction_raw = bundle.denormalize(prediction.cpu().numpy())
            target_raw = bundle.denormalize(target.cpu().numpy())
            predictions.append(prediction_raw)
            targets.append(target_raw)

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    metrics = {
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
    }
    return metrics, float(np.mean(losses))
