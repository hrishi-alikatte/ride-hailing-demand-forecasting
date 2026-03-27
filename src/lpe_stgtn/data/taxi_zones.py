"""Taxi-zone lookup utilities for reproducible study-area definitions."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TAXI_ZONE_LOOKUP_REQUIRED_COLUMNS = (
    "LocationID",
    "Borough",
    "Zone",
    "service_zone",
)


@dataclass(frozen=True)
class TaxiZone:
    """One TLC taxi-zone lookup row."""

    location_id: int
    borough: str
    zone: str
    service_zone: str


@dataclass(frozen=True)
class StudyAreaSummary:
    """Lookup-backed summary of a configured study area."""

    lookup_path: Path
    borough: str
    available_zone_count: int
    included_zone_count: int
    expected_zone_count: int | None
    excluded_location_ids: tuple[int, ...]
    included_location_ids: tuple[int, ...]

    @property
    def matches_expected_zone_count(self) -> bool | None:
        """Return whether the included zone count matches the requested expectation."""
        if self.expected_zone_count is None:
            return None
        return self.included_zone_count == self.expected_zone_count


def load_taxi_zone_lookup(path: Path) -> list[TaxiZone]:
    """Load the TLC taxi-zone lookup CSV into typed rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = sorted(set(TAXI_ZONE_LOOKUP_REQUIRED_COLUMNS) - set(fieldnames))
        if missing_columns:
            missing_list = ", ".join(missing_columns)
            raise ValueError(f"Taxi-zone lookup at {path} is missing columns: {missing_list}")

        rows: list[TaxiZone] = []
        for raw_row in reader:
            rows.append(
                TaxiZone(
                    location_id=int(raw_row["LocationID"]),
                    borough=raw_row["Borough"].strip(),
                    zone=raw_row["Zone"].strip(),
                    service_zone=raw_row["service_zone"].strip(),
                )
            )
    return rows


def summarize_study_area(
    lookup_path: Path,
    *,
    borough: str,
    expected_zone_count: int | None = None,
    excluded_location_ids: Iterable[int] = (),
) -> StudyAreaSummary:
    """Summarize a borough-defined study area from the lookup asset."""
    excluded_ids = tuple(sorted(set(excluded_location_ids)))
    excluded_id_set = set(excluded_ids)

    borough_zone_ids = sorted(
        zone.location_id for zone in load_taxi_zone_lookup(lookup_path) if zone.borough == borough
    )
    included_zone_ids = tuple(
        location_id for location_id in borough_zone_ids if location_id not in excluded_id_set
    )

    return StudyAreaSummary(
        lookup_path=lookup_path,
        borough=borough,
        available_zone_count=len(borough_zone_ids),
        included_zone_count=len(included_zone_ids),
        expected_zone_count=expected_zone_count,
        excluded_location_ids=excluded_ids,
        included_location_ids=included_zone_ids,
    )


def format_study_area_summary(summary: StudyAreaSummary) -> str:
    """Render a terminal-friendly study-area summary."""
    lines = [
        f"Lookup file: {summary.lookup_path}",
        f"Borough: {summary.borough}",
        f"Lookup zones in borough: {summary.available_zone_count}",
        f"Included zones after exclusions: {summary.included_zone_count}",
    ]

    excluded_display = ", ".join(str(location_id) for location_id in summary.excluded_location_ids)
    lines.append(f"Excluded location IDs: {excluded_display or 'none'}")

    if summary.expected_zone_count is not None:
        lines.append(f"Expected zone count: {summary.expected_zone_count}")
        match_display = "yes" if summary.matches_expected_zone_count else "no"
        lines.append(f"Matches expected zone count: {match_display}")

    zone_ids = ", ".join(str(location_id) for location_id in summary.included_location_ids)
    lines.append(f"Included location IDs: {zone_ids}")

    if summary.matches_expected_zone_count is False:
        lines.append(
            "Note: the current lookup asset does not match the requested zone count without an"
            " explicit exclusion rule."
        )

    return "\n".join(lines)
