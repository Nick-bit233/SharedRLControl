#!/usr/bin/env python3
"""Merge registered real-environment PCD scans into a ROS-compatible map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tunnel_deployment.pcd_io import (
        apply_transform,
        bounds,
        crop_points,
        iter_pcd_files,
        load_transforms,
        read_pcd_xyz,
        voxel_downsample,
        write_pcd_ascii_xyz,
    )
except ImportError:
    from pcd_io import (  # type: ignore
        apply_transform,
        bounds,
        crop_points,
        iter_pcd_files,
        load_transforms,
        read_pcd_xyz,
        voxel_downsample,
        write_pcd_ascii_xyz,
    )


def _parse_vec3(values):
    if values is None:
        return None
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three values")
    return [float(v) for v in values]


def _manifest_path(output: Path, manifest_arg: str | None) -> Path:
    if manifest_arg:
        return Path(manifest_arg)
    return output.with_suffix(".manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge registered binary/ascii PCD scans into ASCII x/y/z PCD."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing input .pcd files.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Explicit input .pcd files. Can be combined with --input-dir.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ASCII x/y/z PCD path.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel downsample size in metres. Use 0 to disable.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Add this offset to z after transforms and before cropping.",
    )
    parser.add_argument(
        "--crop-min",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Inclusive minimum crop bound.",
    )
    parser.add_argument(
        "--crop-max",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Inclusive maximum crop bound.",
    )
    parser.add_argument(
        "--transform-json",
        default=None,
        help="Optional JSON mapping file names to flattened 4x4 transforms.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest JSON path. Defaults to <output>.manifest.json.",
    )
    args = parser.parse_args()

    search_paths = list(args.inputs)
    if args.input_dir:
        search_paths.append(args.input_dir)
    if not search_paths:
        parser.error("provide --input-dir or --inputs")

    files = iter_pcd_files(search_paths)
    if not files:
        parser.error("no .pcd inputs found")

    transforms = load_transforms(args.transform_json)
    crop_min = _parse_vec3(args.crop_min)
    crop_max = _parse_vec3(args.crop_max)

    merged = []
    input_stats = []
    for path in files:
        points = read_pcd_xyz(path)
        raw_count = int(len(points))
        transform = transforms[path.name] if path.name in transforms else transforms.get(str(path))
        if transform is not None:
            points = apply_transform(points, transform)
        if args.z_offset:
            points = points.copy()
            points[:, 2] += float(args.z_offset)
        points = crop_points(points, crop_min=crop_min, crop_max=crop_max)
        merged.append(points)
        stat = {
            "file": str(path),
            "raw_points": raw_count,
            "kept_points_before_global_downsample": int(len(points)),
            "bounds": bounds(points),
            "transform": transform.tolist() if transform is not None else None,
        }
        input_stats.append(stat)
        print(
            f"[merge_real_pcd_maps] {path.name}: raw={raw_count} "
            f"kept={len(points)} bounds={stat['bounds']}"
        )

    all_points = np.vstack(merged).astype(np.float32, copy=False)
    before_downsample = int(len(all_points))
    all_points = voxel_downsample(all_points, args.voxel_size)

    output = Path(args.output)
    write_pcd_ascii_xyz(output, all_points)
    manifest = {
        "inputs": input_stats,
        "output": str(output),
        "output_format": "ascii_xyz",
        "voxel_size": float(args.voxel_size),
        "z_offset": float(args.z_offset),
        "crop_min": crop_min,
        "crop_max": crop_max,
        "points_before_downsample": before_downsample,
        "points_after_downsample": int(len(all_points)),
        "bounds": bounds(all_points),
    }
    manifest_path = _manifest_path(output, args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"[merge_real_pcd_maps] wrote {output} with {len(all_points)} points; "
        f"manifest={manifest_path}"
    )
    print(f"[merge_real_pcd_maps] merged bounds={manifest['bounds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
