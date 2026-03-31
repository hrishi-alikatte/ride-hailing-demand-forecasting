"""Load saved graph artifacts aligned to one processed dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraphArtifactsBundle:
    """In-memory view of the saved stage-3 graph artifacts."""

    root_dir: Path
    zone_ids: np.ndarray
    distance_adjacency: np.ndarray
    distance_adjacency_normalized: np.ndarray
    od_flow_counts: np.ndarray
    od_flow_adjacency: np.ndarray
    od_flow_adjacency_normalized: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_zones(self) -> int:
        return int(self.zone_ids.shape[0])


def load_graph_artifacts(root_dir: Path) -> GraphArtifactsBundle:
    """Load the saved graph arrays and metadata from one graph build directory."""
    metadata_path = root_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing graph metadata artifact: {metadata_path}")

    required_paths = {
        "zone_ids": root_dir / "zone_ids.npy",
        "distance_adjacency": root_dir / "distance_adjacency.npy",
        "distance_adjacency_normalized": root_dir / "distance_adjacency_normalized.npy",
        "od_flow_counts": root_dir / "od_flow_counts.npy",
        "od_flow_adjacency": root_dir / "od_flow_adjacency.npy",
        "od_flow_adjacency_normalized": root_dir / "od_flow_adjacency_normalized.npy",
    }
    for path in required_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing graph array artifact: {path}")

    return GraphArtifactsBundle(
        root_dir=root_dir,
        zone_ids=np.load(required_paths["zone_ids"]),
        distance_adjacency=np.load(required_paths["distance_adjacency"]),
        distance_adjacency_normalized=np.load(required_paths["distance_adjacency_normalized"]),
        od_flow_counts=np.load(required_paths["od_flow_counts"]),
        od_flow_adjacency=np.load(required_paths["od_flow_adjacency"]),
        od_flow_adjacency_normalized=np.load(required_paths["od_flow_adjacency_normalized"]),
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
    )
