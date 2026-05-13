#!/usr/bin/env python3
"""Crop a real PCD map into a bounded dry-run workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .pcd_io import bounds, crop_points, read_pcd_xyz, voxel_downsample, write_pcd_ascii_xyz
except ImportError:
    from pcd_io import bounds, crop_points, read_pcd_xyz, voxel_downsample, write_pcd_ascii_xyz  # type: ignore


def _vec3(values: list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input merged PCD file")
    parser.add_argument("--output", required=True, help="Output cropped ASCII XYZ PCD")
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 2.0])
    parser.add_argument("--size", nargs=3, type=float, default=[6.0, 6.0, 5.0])
    parser.add_argument("--crop-min", nargs=3, type=float, default=None)
    parser.add_argument("--crop-max", nargs=3, type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--manifest", default="", help="Manifest JSON output; defaults to <output>.manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")

    points = read_pcd_xyz(input_path)
    input_bounds = bounds(points)

    if args.crop_min is not None or args.crop_max is not None:
        if args.crop_min is None or args.crop_max is None:
            raise ValueError("--crop-min and --crop-max must be provided together")
        crop_min = _vec3(args.crop_min, "crop_min")
        crop_max = _vec3(args.crop_max, "crop_max")
        center = ((crop_min + crop_max) * 0.5).astype(np.float32)
        size = (crop_max - crop_min).astype(np.float32)
    else:
        center = _vec3(args.center, "center")
        size = _vec3(args.size, "size")
        crop_min = center - size * 0.5
        crop_max = center + size * 0.5

    cropped = crop_points(points, crop_min, crop_max)
    if args.voxel_size > 0.0:
        cropped = voxel_downsample(cropped, args.voxel_size)
    if len(cropped) == 0:
        raise RuntimeError(
            f"Crop is empty: min={crop_min.tolist()} max={crop_max.tolist()} input={input_path}"
        )

    write_pcd_ascii_xyz(output_path, cropped)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "input_bounds": input_bounds,
        "crop_center": [float(v) for v in center],
        "crop_size": [float(v) for v in size],
        "crop_min": [float(v) for v in crop_min],
        "crop_max": [float(v) for v in crop_max],
        "voxel_size": float(args.voxel_size),
        "points_before_crop": int(len(points)),
        "points_after_crop": int(len(cropped)),
        "output_bounds": bounds(cropped),
        "suggested_fake_initial_xyz": [0.0, 0.0, float(center[2])],
        "suggested_geofence_x": [float(-size[0] * 0.5), float(size[0] * 0.5)],
        "suggested_geofence_y": [float(-size[1] * 0.5), float(size[1] * 0.5)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(cropped)} cropped points to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
