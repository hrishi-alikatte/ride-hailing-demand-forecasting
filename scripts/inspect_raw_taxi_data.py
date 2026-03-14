#!/usr/bin/env python3
"""Run the raw-data inspection command without requiring an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def run() -> int:
    from lpe_stgtn.cli import main

    return main(["inspect-data", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(run())
