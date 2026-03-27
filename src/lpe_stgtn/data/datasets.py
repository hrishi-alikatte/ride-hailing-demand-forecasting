"""Dataset loading utilities built on the stage-1 processed demand artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProcessedDatasetBundle:
    """In-memory view of the processed demand artifacts."""

    root_dir: Path
    demand: np.ndarray
    normalized_demand: np.ndarray
    zone_ids: np.ndarray
    time_of_day: np.ndarray
    day_of_week: np.ndarray
    train_sample_start_indices: np.ndarray
    validation_sample_start_indices: np.ndarray
    test_sample_start_indices: np.ndarray
    metadata: dict
    history_steps: int
    forecast_steps: int
    normalization_mean: float
    normalization_std: float

    @property
    def num_zones(self) -> int:
        return int(self.zone_ids.shape[0])

    def sample_start_indices(self, split: str) -> np.ndarray:
        split_map = {
            "train": self.train_sample_start_indices,
            "validation": self.validation_sample_start_indices,
            "test": self.test_sample_start_indices,
        }
        try:
            return split_map[split]
        except KeyError as error:
            raise ValueError(f"Unsupported split: {split}") from error

    def build_windows(self, split: str, *, normalized: bool) -> tuple[np.ndarray, np.ndarray]:
        """Materialize `(x, y)` windows for the requested split."""
        source = self.normalized_demand if normalized else self.demand
        start_indices = self.sample_start_indices(split)
        x_windows = np.stack(
            [source[start : start + self.history_steps] for start in start_indices],
            axis=0,
        )
        y_windows = np.stack(
            [
                source[
                    start + self.history_steps : start
                    + self.history_steps
                    + self.forecast_steps
                ]
                for start in start_indices
            ],
            axis=0,
        )
        return x_windows, y_windows

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        """Map normalized demand values back to the raw count scale."""
        return array * self.normalization_std + self.normalization_mean


class WindowedDemandDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy supervised dataset over the processed demand timeline."""

    def __init__(self, bundle: ProcessedDatasetBundle, *, split: str, normalized: bool) -> None:
        self.bundle = bundle
        self.split = split
        self.normalized = normalized
        self.start_indices = bundle.sample_start_indices(split)

    def __len__(self) -> int:
        return int(self.start_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.start_indices[index])
        source = self.bundle.normalized_demand if self.normalized else self.bundle.demand
        x_window = source[start : start + self.bundle.history_steps]
        y_window = source[
            start
            + self.bundle.history_steps : start
            + self.bundle.history_steps
            + self.bundle.forecast_steps
        ]
        return (
            torch.as_tensor(x_window, dtype=torch.float32),
            torch.as_tensor(y_window, dtype=torch.float32),
        )


def load_processed_dataset(root_dir: Path) -> ProcessedDatasetBundle:
    """Load the processed demand artifacts written by the stage-1 pipeline."""
    arrays_path = root_dir / "dataset_arrays.npz"
    metadata_path = root_dir / "metadata.json"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing dataset arrays artifact: {arrays_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata artifact: {metadata_path}")

    arrays = np.load(arrays_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    normalization = metadata["normalization"]
    return ProcessedDatasetBundle(
        root_dir=root_dir,
        demand=arrays["demand"],
        normalized_demand=arrays["normalized_demand"],
        zone_ids=arrays["zone_ids"],
        time_of_day=arrays["time_of_day"],
        day_of_week=arrays["day_of_week"],
        train_sample_start_indices=arrays["train_sample_start_indices"],
        validation_sample_start_indices=arrays["validation_sample_start_indices"],
        test_sample_start_indices=arrays["test_sample_start_indices"],
        metadata=metadata,
        history_steps=int(metadata["history_steps"]),
        forecast_steps=int(metadata["forecast_steps"]),
        normalization_mean=float(normalization["train_mean"]),
        normalization_std=float(normalization["train_std"]),
    )
