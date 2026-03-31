"""Pure-Python readers for the NYC taxi-zone shapefile assets."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DbfField:
    """One DBF field descriptor."""

    name: str
    field_type: str
    length: int
    decimal_count: int


@dataclass(frozen=True)
class PolygonShape:
    """A polygon or multi-part polygon from the shapefile."""

    shape_type: int
    parts: tuple[tuple[tuple[float, float], ...], ...]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class TaxiZoneGeometry:
    """Geometry record joined with the DBF attributes used by the project."""

    location_id: int
    borough: str
    zone: str
    shape: PolygonShape


@dataclass(frozen=True)
class ZoneCentroid:
    """Centroid coordinates for one taxi zone in the shapefile CRS."""

    location_id: int
    borough: str
    zone: str
    x: float
    y: float


def load_taxi_zone_geometries(shapefile_path: Path) -> list[TaxiZoneGeometry]:
    """Load joined geometry and DBF attributes from the NYC taxi-zone asset."""
    dbf_path = shapefile_path.with_suffix(".dbf")
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Missing shapefile: {shapefile_path}")
    if not dbf_path.exists():
        raise FileNotFoundError(f"Missing DBF for shapefile: {dbf_path}")

    shapes = read_polygon_shapes(shapefile_path)
    records = read_dbf_records(dbf_path)
    if len(shapes) != len(records):
        raise ValueError(
            "Shape and DBF record counts do not match: "
            f"{len(shapes)} shapes vs {len(records)} records"
        )

    geometries: list[TaxiZoneGeometry] = []
    for shape, record in zip(shapes, records, strict=True):
        geometries.append(
            TaxiZoneGeometry(
                location_id=int(record["LocationID"]),
                borough=str(record["borough"]).strip(),
                zone=str(record["zone"]).strip(),
                shape=shape,
            )
        )
    return geometries


def compute_zone_centroid(geometry: TaxiZoneGeometry) -> ZoneCentroid:
    """Compute a polygon centroid from the zone's multi-part geometry."""
    centroid_x, centroid_y = polygon_centroid(geometry.shape.parts)
    return ZoneCentroid(
        location_id=geometry.location_id,
        borough=geometry.borough,
        zone=geometry.zone,
        x=centroid_x,
        y=centroid_y,
    )


def polygon_centroid(parts: tuple[tuple[tuple[float, float], ...], ...]) -> tuple[float, float]:
    """Compute an area-weighted centroid across polygon parts."""
    weighted_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    all_points: list[tuple[float, float]] = []

    for part in parts:
        if not part:
            continue
        all_points.extend(part)
        area, centroid_x, centroid_y = ring_area_and_centroid(part)
        if abs(area) < 1e-12:
            continue
        weighted_area += area
        weighted_x += centroid_x * area
        weighted_y += centroid_y * area

    if abs(weighted_area) >= 1e-12:
        return weighted_x / weighted_area, weighted_y / weighted_area

    if not all_points:
        raise ValueError("Cannot compute centroid for an empty polygon.")
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    return sum(x_values) / len(x_values), sum(y_values) / len(y_values)


def ring_area_and_centroid(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float]:
    """Return signed area and centroid for one polygon ring."""
    if len(points) < 3:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return 0.0, sum(x_values) / len(x_values), sum(y_values) / len(y_values)

    ring = list(points)
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    twice_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=True):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross

    if abs(twice_area) < 1e-12:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return 0.0, sum(x_values) / len(x_values), sum(y_values) / len(y_values)

    area = twice_area / 2.0
    centroid_x /= 3.0 * twice_area
    centroid_y /= 3.0 * twice_area
    return area, centroid_x, centroid_y


def read_dbf_records(path: Path) -> list[dict[str, Any]]:
    """Read DBF attribute records without external dependencies."""
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"DBF header is truncated: {path}")
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields: list[DbfField] = []
        while True:
            descriptor = handle.read(32)
            if not descriptor:
                raise ValueError(f"Unexpected end of DBF header: {path}")
            if descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            field_type = chr(descriptor[11])
            length = descriptor[16]
            decimal_count = descriptor[17]
            fields.append(
                DbfField(
                    name=name,
                    field_type=field_type,
                    length=length,
                    decimal_count=decimal_count,
                )
            )

        handle.seek(header_length)

        records: list[dict[str, Any]] = []
        for _ in range(record_count):
            raw_record = handle.read(record_length)
            if len(raw_record) != record_length:
                raise ValueError(f"Unexpected end of DBF records: {path}")
            if raw_record[:1] == b"*":
                continue
            record: dict[str, Any] = {}
            offset = 1
            for field in fields:
                raw_value = raw_record[offset : offset + field.length]
                record[field.name] = parse_dbf_value(
                    raw_value,
                    field_type=field.field_type,
                    decimal_count=field.decimal_count,
                )
                offset += field.length
            records.append(record)
    return records


def parse_dbf_value(raw_value: bytes, *, field_type: str, decimal_count: int) -> Any:
    """Parse one DBF field payload into a Python value."""
    text = raw_value.decode("latin1", errors="ignore").strip()
    if field_type == "C":
        return text
    if field_type in {"N", "F"}:
        if text == "":
            return 0.0 if decimal_count > 0 else 0
        return float(text) if decimal_count > 0 else int(text)
    if field_type == "L":
        return text.upper() in {"Y", "T"}
    if field_type == "D":
        return text
    return text


def read_polygon_shapes(path: Path) -> list[PolygonShape]:
    """Read polygon records from a `.shp` file without external dependencies."""
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100:
            raise ValueError(f"Shapefile header is truncated: {path}")
        file_length_words = struct.unpack(">i", header[24:28])[0]
        file_length_bytes = file_length_words * 2
        handle.seek(100)

        shapes: list[PolygonShape] = []
        while handle.tell() < file_length_bytes:
            record_header = handle.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError(f"Malformed shapefile record header in {path}")
            content_length_words = struct.unpack(">i", record_header[4:8])[0]
            content = handle.read(content_length_words * 2)
            if len(content) != content_length_words * 2:
                raise ValueError(f"Malformed shapefile record body in {path}")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in {5, 15, 25}:
                raise ValueError(f"Unsupported shapefile shape type {shape_type} in {path}")
            shapes.append(parse_polygon_shape(content))
    return shapes


def parse_polygon_shape(content: bytes) -> PolygonShape:
    """Parse one polygon record from a shapefile content payload."""
    bbox = struct.unpack("<4d", content[4:36])
    num_parts = struct.unpack("<i", content[36:40])[0]
    num_points = struct.unpack("<i", content[40:44])[0]
    parts = struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts])
    points_offset = 44 + 4 * num_parts
    points_end = points_offset + 16 * num_points
    points_raw = struct.unpack(
        f"<{num_points * 2}d",
        content[points_offset:points_end],
    )
    points = [
        (points_raw[index], points_raw[index + 1])
        for index in range(0, len(points_raw), 2)
    ]

    part_slices = list(parts) + [num_points]
    polygon_parts: list[tuple[tuple[float, float], ...]] = []
    for start, end in zip(part_slices[:-1], part_slices[1:], strict=True):
        polygon_parts.append(tuple(points[start:end]))

    shape_type = struct.unpack("<i", content[:4])[0]
    return PolygonShape(
        shape_type=shape_type,
        parts=tuple(polygon_parts),
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )
