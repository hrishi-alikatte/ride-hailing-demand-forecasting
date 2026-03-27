"""Project path helpers for terminal-first workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    configs: Path
    artifacts: Path
    interim: Path
    processed: Path


def get_project_paths() -> ProjectPaths:
    """Resolve the repository paths, allowing an explicit environment override."""
    root_override = os.environ.get("LPE_STGTN_PROJECT_ROOT")
    root = Path(root_override) if root_override else Path(__file__).resolve().parents[3]
    return ProjectPaths(
        root=root,
        data=root / "data",
        configs=root / "configs",
        artifacts=root / "artifacts",
        interim=root / "data" / "interim",
        processed=root / "data" / "processed",
    )
