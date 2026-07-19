#!/usr/bin/env python3
"""Benchmark the real PCD policy-ray and raw-clearance channels together."""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from srlc_real_deployment.clearance_geometry import PcdClearanceGeometry
from srlc_real_deployment.pcd_raycast import (
    PcdRaycaster,
    policy_surface_distances,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", required=True, help="67,726-point aligned PCD")
    parser.add_argument("--log", required=True, help="recorder JSON with a collision frame")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-p95-ms", type=float, default=25.0)
    return parser.parse_args()


def _collision_pose(log_path):
    payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
    sample = next(
        item
        for item in payload["samples"]
        if str(item.get("fault_reason", "")).upper() == "COLLISION"
    )
    position = np.asarray(sample["position"], dtype=np.float64)
    yaw = float(sample["yaw"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return position, rotation


def main():
    args = _parse_args()
    if args.samples <= 0 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup non-negative")
    if not math.isfinite(args.max_p95_ms) or args.max_p95_ms <= 0.0:
        raise ValueError("--max-p95-ms must be positive and finite")

    position, rotation = _collision_pose(args.log)
    raycaster = PcdRaycaster(
        args.pcd,
        resolution=0.1,
        inflate=(0.0, 0.0, 0.0),
    )
    geometry = PcdClearanceGeometry(args.pcd)
    durations = []
    beam_count = 0
    for index in range(args.warmup + args.samples):
        started = time.perf_counter()
        raw = raycaster.raycast_raw(
            position,
            0.0,
            4.0,
            -10.0,
            20.0,
            4,
            10.0,
        )
        policy = policy_surface_distances(
            raw.entry_distances,
            raw.directions_world,
            raw.hit_mask,
            rotation,
            (0.20, 0.20, 0.05),
            max_range=4.0,
        )
        clearance = geometry.query(
            position,
            rotation,
            (0.15, 0.15, 0.05),
            clearance_cap=1.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(clearance.surface_clearance):
            raise RuntimeError("clearance query returned a non-finite value")
        beam_count = int(policy.size)
        if index >= args.warmup:
            durations.append(elapsed_ms)

    point_count = int(geometry.points.shape[0])
    p95_ms = float(np.percentile(durations, 95.0))
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "point_count": point_count,
        "beam_count": beam_count,
        "warmup": args.warmup,
        "samples": args.samples,
        "median_ms": float(np.median(durations)),
        "p95_ms": p95_ms,
        "max_p95_ms": args.max_p95_ms,
    }
    print(json.dumps(report, sort_keys=True))

    if point_count != 67726:
        print("expected the production 67,726-point PCD", file=sys.stderr)
        return 2
    if beam_count != 144:
        print("expected the production 36x4 beam tensor", file=sys.stderr)
        return 2
    if p95_ms >= args.max_p95_ms:
        print("combined channel p95 exceeds the configured gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
