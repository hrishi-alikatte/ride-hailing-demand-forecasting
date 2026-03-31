"""Command-line entrypoints for common repository tasks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lpe_stgtn.analysis.zone_tracks import (
    build_zone_track_comparison,
    format_zone_track_comparison,
    zone_track_comparison_to_dict,
)
from lpe_stgtn.data.inspection import format_dataset_report, summarize_dataset
from lpe_stgtn.data.preprocessing import (
    format_prepared_dataset_summary,
    prepare_dataset_from_config,
)
from lpe_stgtn.data.taxi_zones import format_study_area_summary, summarize_study_area
from lpe_stgtn.graphs.pipeline import (
    build_graph_artifacts_from_config,
    format_graph_build_summary,
)
from lpe_stgtn.graphs.road_network import (
    download_centerline_geojson,
    format_road_network_download_summary,
)
from lpe_stgtn.training.baselines import format_baseline_run_summary, run_baseline_experiment
from lpe_stgtn.training.lpe_stgtn_runner import run_lpe_stgtn_experiment
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

    full_model_parser = subparsers.add_parser(
        "run-full-model",
        help="Run the full LPE-STGTN config-driven experiment.",
    )
    full_model_parser.add_argument(
        "--config",
        type=Path,
        default=project_paths.configs / "experiments" / "nyc_lpe_stgtn_sensitivity_67.yaml",
        help="Path to the full model experiment config.",
    )
    full_model_parser.add_argument(
        "--training-config",
        type=Path,
        default=project_paths.configs / "training" / "default.yaml",
        help="Path to the shared training defaults.",
    )
    baseline_parser.add_argument(
        "--training-config",
        type=Path,
        default=project_paths.configs / "training" / "default.yaml",
        help="Path to the shared training defaults.",
    )

    graph_parser = subparsers.add_parser(
        "build-graphs",
        help="Build the stage-3 distance and OD-flow graph artifacts.",
    )
    graph_parser.add_argument(
        "--config",
        type=Path,
        default=project_paths.configs / "graphs" / "nyc_manhattan_default.yaml",
        help="Path to the graph construction config.",
    )

    road_network_parser = subparsers.add_parser(
        "download-road-network",
        help="Download the official Manhattan NYC street-centerline subset.",
    )
    road_network_parser.add_argument(
        "--output",
        type=Path,
        default=project_paths.data / "external" / "nyc_centerline" / "centerline_citywide.geojson",
        help="Output path for the downloaded official centerline GeoJSON asset.",
    )
    road_network_parser.add_argument(
        "--borough-code",
        type=int,
        default=None,
        help="Optional NYC borough code to subset the official centerline dataset.",
    )

    compare_tracks_parser = subparsers.add_parser(
        "compare-zone-tracks",
        help="Compare the main 68-zone track against the 67-zone sensitivity track.",
    )
    compare_tracks_parser.add_argument(
        "--primary-label",
        default="paper_count_68_track",
        help="Label for the primary study-area track.",
    )
    compare_tracks_parser.add_argument(
        "--primary-processed-data-dir",
        type=Path,
        default=project_paths.data / "processed" / "nyc_yellow_taxi_2018_manhattan_pickups_15min",
        help="Processed data directory for the primary track.",
    )
    compare_tracks_parser.add_argument(
        "--primary-persistence-report",
        type=Path,
        default=project_paths.root
        / "artifacts"
        / "reports"
        / "baselines"
        / "nyc_persistence_baseline.json",
        help="Persistence baseline report for the primary track.",
    )
    compare_tracks_parser.add_argument(
        "--primary-lstm-report",
        type=Path,
        default=project_paths.root
        / "artifacts"
        / "reports"
        / "baselines"
        / "nyc_lstm_baseline.json",
        help="LSTM baseline report for the primary track.",
    )
    compare_tracks_parser.add_argument(
        "--secondary-label",
        default="road_network_clean_67_track",
        help="Label for the secondary study-area track.",
    )
    compare_tracks_parser.add_argument(
        "--secondary-processed-data-dir",
        type=Path,
        default=project_paths.data
        / "processed"
        / "nyc_yellow_taxi_2018_manhattan_pickups_15min_sensitivity_67",
        help="Processed data directory for the secondary track.",
    )
    compare_tracks_parser.add_argument(
        "--secondary-persistence-report",
        type=Path,
        default=project_paths.root
        / "artifacts"
        / "reports"
        / "baselines"
        / "nyc_persistence_baseline_sensitivity_67.json",
        help="Persistence baseline report for the secondary track.",
    )
    compare_tracks_parser.add_argument(
        "--secondary-lstm-report",
        type=Path,
        default=project_paths.root
        / "artifacts"
        / "reports"
        / "baselines"
        / "nyc_lstm_baseline_sensitivity_67.json",
        help="LSTM baseline report for the secondary track.",
    )
    compare_tracks_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the comparison payload as JSON.",
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

    if args.command == "run-full-model":
        project_paths = get_project_paths()
        summary = run_lpe_stgtn_experiment(
            args.config,
            project_root=project_paths.root,
            training_config_path=args.training_config,
        )
        print(format_baseline_run_summary(summary))
        return 0

    if args.command == "build-graphs":
        project_paths = get_project_paths()
        summary = build_graph_artifacts_from_config(
            args.config,
            project_root=project_paths.root,
        )
        print(format_graph_build_summary(summary))
        return 0

    if args.command == "download-road-network":
        summary = download_centerline_geojson(
            args.output,
            borough_code=args.borough_code,
        )
        print(format_road_network_download_summary(summary))
        return 0

    if args.command == "compare-zone-tracks":
        comparison = build_zone_track_comparison(
            primary_label=args.primary_label,
            primary_processed_data_dir=args.primary_processed_data_dir,
            primary_persistence_report_path=args.primary_persistence_report,
            primary_lstm_report_path=args.primary_lstm_report,
            secondary_label=args.secondary_label,
            secondary_processed_data_dir=args.secondary_processed_data_dir,
            secondary_persistence_report_path=args.secondary_persistence_report,
            secondary_lstm_report_path=args.secondary_lstm_report,
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(zone_track_comparison_to_dict(comparison), indent=2),
                encoding="utf-8",
            )
        print(format_zone_track_comparison(comparison))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
