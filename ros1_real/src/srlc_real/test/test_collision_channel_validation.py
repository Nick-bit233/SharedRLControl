#!/usr/bin/env python3

import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from srlc_real_deployment.clearance_geometry import PcdClearanceGeometry  # noqa: E402
from srlc_real_deployment.clearance_guard import (  # noqa: E402
    ClearanceGuard,
    ClearanceState,
)
from srlc_real_deployment.pcd_io import write_pcd_ascii_xyz  # noqa: E402
from srlc_real_deployment.pcd_raycast import (  # noqa: E402
    PcdRaycaster,
    policy_surface_distances,
)


DEFAULT_PCD = Path(
    "/home/nickbit/uav/SharedRLControl/ros1/real_maps/room601/"
    "0624_section_resampled6w_ascii_aligned_yaw_m4p50_floor_level_z0.pcd"
)
DEFAULT_LOGS = {
    "060212": Path(
        "/home/nickbit/uav/SharedRLControl/ros1_real/results_v2/"
        "real_px4_nokov_20260717_060212.json"
    ),
    "055747": Path(
        "/home/nickbit/uav/SharedRLControl/ros1_real/results_v2/"
        "real_px4_nokov_20260717_055747.json"
    ),
}
EXPECTED_CLEARANCES = {
    "060212": 0.248,
    "055747": 0.178,
}
# The logs retain yaw but not full roll/pitch, and PCD preprocessing may round
# points. A 1.5 cm tolerance covers those replay limitations without getting
# close to either the 0.10 m soft threshold or 0.02 m hard threshold.
COUNTERFACTUAL_TOLERANCE_M = 0.015


def _with_temporary_path(test_function):
    def wrapped():
        directory = Path(tempfile.mkdtemp(prefix="srlc-validation-test-"))
        try:
            return test_function(directory)
        finally:
            shutil.rmtree(str(directory))

    wrapped.__name__ = test_function.__name__
    wrapped.__doc__ = test_function.__doc__
    return wrapped


def _external_asset_path(environment_name, default):
    return Path(os.environ.get(environment_name, str(default)))


def _collision_sample(log_path):
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    return next(
        sample
        for sample in payload["samples"]
        if str(sample.get("fault_reason", "")).upper() == "COLLISION"
    )


def _yaw_rotation(yaw):
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _reference_dda(occupied, resolution, origin, direction, max_range):
    voxel = [int(math.floor(value / resolution)) for value in origin]
    if tuple(voxel) in occupied:
        return 0.0

    step = [1 if value > 0.0 else -1 if value < 0.0 else 0 for value in direction]
    t_max = [math.inf, math.inf, math.inf]
    t_delta = [math.inf, math.inf, math.inf]
    for axis in range(3):
        component = float(direction[axis])
        if component > 0.0:
            boundary = (voxel[axis] + 1) * resolution
        elif component < 0.0:
            boundary = voxel[axis] * resolution
        else:
            continue
        t_max[axis] = (boundary - float(origin[axis])) / component
        t_delta[axis] = resolution / abs(component)

    while True:
        entry = min(t_max)
        if not math.isfinite(entry) or entry > max_range:
            return None
        tolerance = 1e-12 * max(1.0, abs(entry))
        for axis in range(3):
            if abs(t_max[axis] - entry) <= tolerance:
                voxel[axis] += step[axis]
                t_max[axis] += t_delta[axis]
        if tuple(voxel) in occupied:
            return max(0.0, entry)


@_with_temporary_path
def test_dda_optimization_matches_independent_reference(tmp_path):
    points = np.array(
        [
            [-1.95, -0.95, 0.05],
            [-0.95, 0.05, 0.05],
            [0.05, 0.05, 0.05],
            [0.95, 0.95, 0.95],
            [1.95, -0.95, 0.05],
            [2.05, 0.05, -0.95],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "dda_reference.pcd"
    write_pcd_ascii_xyz(path, points)
    raycaster = PcdRaycaster(path, resolution=0.1, inflate=(0.0, 0.0, 0.0))

    cases = [
        (np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)),
        (np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, -1.0]) / math.sqrt(3.0)),
        (np.array([0.05, 0.05, 0.05]), np.array([1.0, 0.0, 0.0])),
    ]
    random = np.random.RandomState(20260719)
    for _ in range(200):
        origin = random.uniform(-2.5, 2.5, size=3)
        direction = random.normal(size=3)
        direction /= np.linalg.norm(direction)
        cases.append((origin, direction))

    for origin, direction in cases:
        expected = _reference_dda(
            raycaster._occupied,
            raycaster.res,
            origin,
            direction,
            4.0,
        )
        actual = raycaster._dda_entry_distance(origin, direction, 4.0)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert abs(actual - expected) <= 1e-12


def test_recorded_false_triggers_have_large_raw_surface_clearance():
    pcd_path = _external_asset_path("SRLC_COUNTERFACTUAL_PCD", DEFAULT_PCD)
    log_paths = {
        run: _external_asset_path(
            "SRLC_COUNTERFACTUAL_LOG_" + run,
            default,
        )
        for run, default in DEFAULT_LOGS.items()
    }
    missing = [path for path in [pcd_path, *log_paths.values()] if not path.exists()]
    if missing:
        import unittest

        raise unittest.SkipTest("external replay assets unavailable: " + ", ".join(map(str, missing)))

    geometry = PcdClearanceGeometry(pcd_path)
    assert geometry.points.shape == (67726, 3)
    for run in ("060212", "055747"):
        sample = _collision_sample(log_paths[run])
        result = geometry.query(
            sample["position"],
            _yaw_rotation(sample["yaw"]),
            (0.15, 0.15, 0.05),
            clearance_cap=1.0,
        )
        assert abs(result.surface_clearance - EXPECTED_CLEARANCES[run]) <= (
            COUNTERFACTUAL_TOLERANCE_M
        )

        guard = ClearanceGuard()
        decision = guard.update(
            now=1.0,
            source_stamp=1.0,
            valid=True,
            surface_clearance=result.surface_clearance,
            escape_direction=result.escape_direction,
            human_velocity_world=(0.0, 0.0, 0.0),
            px4_local_position=sample["position"],
        )
        assert guard.config.proximity_enabled is False
        assert decision.state == ClearanceState.NORMAL


def test_reproducible_benchmark_entrypoint_is_installed():
    script = SCRIPT_DIR / "benchmark_collision_channels.py"
    assert script.exists()
    source = script.read_text(encoding="utf-8")
    for required in (
        "--pcd",
        "--log",
        "--samples",
        "--warmup",
        "--max-p95-ms",
        "PcdRaycaster",
        "policy_surface_distances",
        "PcdClearanceGeometry",
        "platform.platform()",
        '"point_count"',
        '"beam_count"',
        '"p95_ms"',
    ):
        assert required in source

    cmake = (PACKAGE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "scripts/benchmark_collision_channels.py" in cmake
    assert "catkin_add_nosetests(test/test_collision_channel_validation.py)" in cmake


def test_real_pcd_combined_channel_p95_under_25ms_opt_in():
    if os.environ.get("SRLC_RUN_REAL_PCD_BENCHMARK") != "1":
        import unittest

        raise unittest.SkipTest(
            "set SRLC_RUN_REAL_PCD_BENCHMARK=1 to run the 67,726-point benchmark"
        )

    pcd_path = _external_asset_path("SRLC_COUNTERFACTUAL_PCD", DEFAULT_PCD)
    log_path = _external_asset_path(
        "SRLC_COUNTERFACTUAL_LOG_060212",
        DEFAULT_LOGS["060212"],
    )
    if not pcd_path.exists() or not log_path.exists():
        import unittest

        raise unittest.SkipTest("external benchmark assets unavailable")

    samples = int(os.environ.get("SRLC_BENCHMARK_SAMPLES", "60"))
    warmup = int(os.environ.get("SRLC_BENCHMARK_WARMUP", "10"))
    sample = _collision_sample(log_path)
    position = np.asarray(sample["position"], dtype=np.float64)
    rotation = _yaw_rotation(sample["yaw"])
    raycaster = PcdRaycaster(pcd_path, resolution=0.1, inflate=(0.0, 0.0, 0.0))
    geometry = PcdClearanceGeometry(pcd_path)
    assert geometry.points.shape == (67726, 3)

    durations = []
    beam_count = None
    for index in range(warmup + samples):
        started = time.perf_counter()
        raw = raycaster.raycast_raw(position, 0.0, 4.0, -10.0, 20.0, 4, 10.0)
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
        beam_count = int(policy.size)
        assert math.isfinite(clearance.surface_clearance)
        if index >= warmup:
            durations.append(elapsed_ms)

    p95_ms = float(np.percentile(durations, 95.0))
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "point_count": int(geometry.points.shape[0]),
                "beam_count": beam_count,
                "warmup": warmup,
                "samples": samples,
                "median_ms": float(np.median(durations)),
                "p95_ms": p95_ms,
            },
            sort_keys=True,
        )
    )
    assert beam_count == 144
    assert p95_ms < 25.0
