"""Command-line entrypoints for common repository tasks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lpe_stgtn.data.inspection import format_dataset_report, summarize_dataset
from lpe_stgtn.data.preprocessing import (
    format_prepared_dataset_summary,
    prepare_dataset_from_config,
)
from lpe_stgtn.data.taxi_zones import format_study_area_summary, summarize_study_area
from lpe_stgtn.training.baselines import format_baseline_run_summary, run_baseline_experiment
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

    study_area_parser = subparsers.add_parser(
        "inspect-study-area",
        help="Summarize the current borough study area from the TLC lookup asset.",
    )
    study_area_parser.add_argument(
        "--lookup-path",
        type=Path,
        default=project_paths.data / "external" / "taxi_zone_lookup.csv",
        help="Path to the TLC taxi-zone lookup CSV.",
    )
    study_area_parser.add_argument(
        "--borough",
        default="Manhattan",
        help="Borough name to filter from the lookup asset.",
    )
    study_area_parser.add_argument(
        "--expected-zone-count",
        type=int,
        default=68,
        help="Expected zone count for the configured study area.",
    )
    study_area_parser.add_argument(
        "--exclude-location-id",
        type=int,
        action="append",
        default=[],
        help="LocationID values to exclude from the borough zone set.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Build the stage-1 processed demand dataset from raw NYC parquet files.",
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        default=project_paths.configs / "data" / "nyc_yellow_2018.yaml",
        help="Path to the data preprocessing config.",
    )

    baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run a config-driven baseline experiment on the processed dataset.",
    )
    baseline_parser.add_argument(
        "--config",
        type=Path,
        default=project_paths.configs / "experiments" / "nyc_persistence_baseline.yaml",
        help="Path to the baseline experiment config.",
    )
    baseline_parser.add_argument(
        "--training-config",
        type=Path,
        default=project_paths.configs / "training" / "default.yaml",
        help="Path to the shared training defaults.",
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

    if args.command == "inspect-study-area":
        summary = summarize_study_area(
            args.lookup_path,
            borough=args.borough,
            expected_zone_count=args.expected_zone_count,
            excluded_location_ids=args.exclude_location_id,
        )
        print(format_study_area_summary(summary))
        return 0

    if args.command == "prepare-data":
        project_paths = get_project_paths()
        summary = prepare_dataset_from_config(args.config, project_root=project_paths.root)
        print(format_prepared_dataset_summary(summary))
        return 0

    if args.command == "run-baseline":
        project_paths = get_project_paths()
        summary = run_baseline_experiment(
            args.config,
            project_root=project_paths.root,
            training_config_path=args.training_config,
        )
        print(format_baseline_run_summary(summary))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
