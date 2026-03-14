"""Command-line entrypoints for common repository tasks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lpe_stgtn.data.inspection import format_dataset_report, summarize_dataset
from lpe_stgtn.utils.paths import get_project_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="lpe-stgtn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    project_paths = get_project_paths()

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Summarize raw parquet files and validate required columns.",
    )
    inspect_parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_paths.data,
        help="Directory containing raw parquet files.",
    )
    inspect_parser.add_argument(
        "--pattern",
        default="yellow_tripdata_2018-*.parquet",
        help="Glob pattern used to match raw data files.",
    )
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of matching files to inspect.",
    )
    inspect_parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of leading rows to sample from each file.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-data":
        summaries = summarize_dataset(
            data_dir=args.data_dir,
            pattern=args.pattern,
            limit=args.limit,
            sample_rows=args.sample_rows,
        )
        if not summaries:
            parser.error(f"No parquet files matched pattern '{args.pattern}' in {args.data_dir}.")
        print(format_dataset_report(summaries))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
