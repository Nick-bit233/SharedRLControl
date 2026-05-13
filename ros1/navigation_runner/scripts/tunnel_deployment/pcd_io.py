"""Small PCD reader/writer utilities for tunnel deployment tools.

Supports PCD v0.7 ASCII and uncompressed binary files with scalar fields
including at least x, y, z. Binary-compressed PCD is intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PcdHeader:
    fields: list[str]
    size: list[int]
    type: list[str]
    count: list[int]
    width: int
    height: int
    points: int
    data: str
    data_offset: int
    raw_lines: list[str]


def _parse_header(blob: bytes) -> PcdHeader:
    offset = 0
    raw_lines: list[str] = []
    header_map: dict[str, list[str]] = {}

    while offset < len(blob):
        next_offset = blob.find(b"\n", offset)
        if next_offset < 0:
            raise ValueError("PCD header is missing DATA line")
        line_bytes = blob[offset:next_offset].rstrip(b"\r")
        offset = next_offset + 1
        line = line_bytes.decode("ascii", errors="strict")
        raw_lines.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        key = parts[0].upper()
        header_map[key] = parts[1:]
        if key == "DATA":
            break
    else:
        raise ValueError("PCD header is missing DATA line")

    fields = header_map.get("FIELDS")
    if not fields:
        raise ValueError("PCD header is missing FIELDS")

    size = [int(v) for v in header_map.get("SIZE", [])]
    typ = header_map.get("TYPE", [])
    count = [int(v) for v in header_map.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(size) == len(typ) == len(count)):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths do not match")

    width = int(header_map.get("WIDTH", ["0"])[0])
    height = int(header_map.get("HEIGHT", ["1"])[0])
    points = int(header_map.get("POINTS", [str(width * height)])[0])
    data = header_map.get("DATA", [""])[0].lower()
    if data not in {"ascii", "binary"}:
        raise ValueError(f"Unsupported PCD DATA format: {data}")

    return PcdHeader(
        fields=fields,
        size=size,
        type=typ,
        count=count,
        width=width,
        height=height,
        points=points,
        data=data,
        data_offset=offset,
        raw_lines=raw_lines,
    )


def _numpy_dtype(header: PcdHeader) -> np.dtype:
    dtype_fields = []
    for name, size, typ, count in zip(
        header.fields, header.size, header.type, header.count
    ):
        typ = typ.upper()
        if typ == "F" and size == 4:
            base = "<f4"
        elif typ == "F" and size == 8:
            base = "<f8"
        elif typ == "U" and size == 1:
            base = "<u1"
        elif typ == "U" and size == 2:
            base = "<u2"
        elif typ == "U" and size == 4:
            base = "<u4"
        elif typ == "I" and size == 1:
            base = "<i1"
        elif typ == "I" and size == 2:
            base = "<i2"
        elif typ == "I" and size == 4:
            base = "<i4"
        else:
            raise ValueError(f"Unsupported PCD field type: {name} {typ}{size}")

        if count == 1:
            dtype_fields.append((name, base))
        else:
            dtype_fields.append((name, base, (count,)))
    return np.dtype(dtype_fields)


def read_pcd_fields(path: str | Path, fields: Sequence[str]) -> np.ndarray:
    """Read selected scalar PCD fields into an ``N x len(fields)`` float32 array."""
    path = Path(path)
    blob = path.read_bytes()
    header = _parse_header(blob)
    missing = [field for field in fields if field not in header.fields]
    if missing:
        raise ValueError(f"{path} is missing required PCD fields: {missing}")

    if header.data == "binary":
        structured = np.frombuffer(
            blob[header.data_offset:],
            dtype=_numpy_dtype(header),
            count=header.points,
        )
        cols = [np.asarray(structured[field], dtype=np.float32) for field in fields]
        return np.column_stack(cols).astype(np.float32, copy=False)

    text = blob[header.data_offset:].decode("ascii", errors="strict")
    if not text.strip():
        return np.empty((0, len(fields)), dtype=np.float32)
    data = np.loadtxt(text.splitlines(), dtype=np.float32, ndmin=2)
    field_to_col = {name: idx for idx, name in enumerate(header.fields)}
    cols = [data[:, field_to_col[field]] for field in fields]
    return np.column_stack(cols).astype(np.float32, copy=False)


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    """Read x/y/z points from an ASCII or binary PCD file."""
    points = read_pcd_fields(path, ("x", "y", "z"))
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {matrix.shape}")
    hom = np.ones((points.shape[0], 4), dtype=np.float32)
    hom[:, :3] = points
    return (hom @ matrix.T)[:, :3].astype(np.float32, copy=False)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point per voxel using deterministic first-point selection."""
    if voxel_size <= 0:
        return points
    if len(points) == 0:
        return points
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(voxels, axis=0, return_index=True)
    keep.sort()
    return points[keep]


def crop_points(
    points: np.ndarray,
    crop_min: Sequence[float] | None = None,
    crop_max: Sequence[float] | None = None,
) -> np.ndarray:
    """Crop points by inclusive axis-aligned bounds."""
    mask = np.ones(points.shape[0], dtype=bool)
    if crop_min is not None:
        mn = np.asarray(crop_min, dtype=np.float32)
        if mn.shape != (3,):
            raise ValueError("crop_min must contain three values")
        mask &= np.all(points >= mn, axis=1)
    if crop_max is not None:
        mx = np.asarray(crop_max, dtype=np.float32)
        if mx.shape != (3,):
            raise ValueError("crop_max must contain three values")
        mask &= np.all(points <= mx, axis=1)
    return points[mask]


def bounds(points: np.ndarray) -> Mapping[str, list[float]]:
    if len(points) == 0:
        return {"min": [], "max": []}
    return {
        "min": [float(v) for v in points.min(axis=0)],
        "max": [float(v) for v in points.max(axis=0)],
    }


def write_pcd_ascii_xyz(path: str | Path, points: np.ndarray) -> None:
    """Write x/y/z points as ASCII PCD compatible with existing ROS scripts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    with path.open("w") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z\n")
        handle.write("SIZE 4 4 4\n")
        handle.write("TYPE F F F\n")
        handle.write("COUNT 1 1 1\n")
        handle.write(f"WIDTH {len(points)}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {len(points)}\n")
        handle.write("DATA ascii\n")
        np.savetxt(handle, points, fmt="%.6f %.6f %.6f")


def load_transforms(path: str | Path | None) -> dict[str, np.ndarray]:
    """Load optional per-file 4x4 transforms from a JSON object."""
    if not path:
        return {}
    import json

    with Path(path).open() as handle:
        payload = json.load(handle)
    transforms: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        matrix_values = value.get("matrix", value) if isinstance(value, dict) else value
        matrix = np.asarray(matrix_values, dtype=np.float32)
        if matrix.size != 16:
            raise ValueError(f"Transform for {name} must contain 16 values")
        transforms[name] = matrix.reshape(4, 4)
    return transforms


def iter_pcd_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pcd")))
        else:
            files.append(path)
    return files
