#!/usr/bin/env python3

import importlib
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _snapshot_module():
    path = SCRIPT_DIR / "srlc_real_deployment" / "recorder_snapshots.py"
    assert path.exists(), "ROS-free recorder snapshot store is missing"
    return importlib.import_module("srlc_real_deployment.recorder_snapshots")


def _guard_status_module():
    path = SCRIPT_DIR / "srlc_real_deployment" / "guard_status.py"
    assert path.exists(), "ROS-free guard source metadata helper is missing"
    return importlib.import_module("srlc_real_deployment.guard_status")


def _clearance(module, marker):
    value = float(marker)
    return module.ClearanceObservation(
        valid=bool(marker % 2),
        source_stamp=value,
        source_frame_id="frame-%d" % marker,
        surface_clearance=value + 0.1,
        center_distance=value + 0.2,
        nearest_obstacle_point=(value + 1.0, value + 2.0, value + 3.0),
        escape_direction=(-value - 1.0, -value - 2.0, -value - 3.0),
    )


def _clearance_signature(observation):
    return (
        observation.valid,
        observation.source_stamp,
        observation.source_frame_id,
        observation.surface_clearance,
        observation.center_distance,
        observation.nearest_obstacle_point,
        observation.escape_direction,
    )


def test_clearance_replacement_never_exposes_a_mixed_source_frame():
    module = _snapshot_module()
    store = module.RecorderObservationStore()
    first = _clearance(module, 10)
    second = _clearance(module, 20)
    allowed = {_clearance_signature(first), _clearance_signature(second)}
    store.replace_clearance(first)
    start = threading.Event()

    def writer():
        start.wait()
        for index in range(20000):
            store.replace_clearance(first if index % 2 == 0 else second)

    thread = threading.Thread(target=writer)
    thread.start()
    start.set()
    observed = set()
    for _ in range(20000):
        observed.add(_clearance_signature(store.read().clearance))
    thread.join()

    assert observed
    assert observed <= allowed


def test_one_read_is_an_immutable_dual_channel_and_guard_snapshot():
    module = _snapshot_module()
    store = module.RecorderObservationStore()
    first_clearance = _clearance(module, 1)
    first_guard = module.GuardObservation(
        source_valid=True,
        source_stamp=1.25,
        source_frame_id="nokov_local",
        raw_state="PROXIMITY_HOLD",
        effective_state="NORMAL",
    )
    store.replace_clearance(first_clearance)
    store.replace_guard(first_guard)
    store.replace_raw_center_distance(1.5)
    store.replace_policy_ranges(0.7, 0.8)

    captured = store.read()
    store.replace_clearance(_clearance(module, 2))
    store.replace_guard(module.GuardObservation())
    store.replace_raw_center_distance(9.5)
    store.replace_policy_ranges(9.7, 9.8)

    assert captured.clearance == first_clearance
    assert captured.guard == first_guard
    assert captured.raw_center_distance == 1.5
    assert captured.policy_min_distance == 0.7
    assert captured.front_distance == 0.8


def test_guard_source_preserves_exact_ros_header_parts_and_validity_changes():
    module = _guard_status_module()
    header = SimpleNamespace(
        seq=17,
        stamp=SimpleNamespace(secs=1234567890, nsecs=987654321),
        frame_id="nokov_local",
    )
    source = module.GuardStatusSource.from_header(
        header,
        source_valid=True,
    )

    assert source.source_valid is True
    assert source.seq == 17
    assert source.stamp_secs == 1234567890
    assert source.stamp_nsecs == 987654321
    assert source.frame_id == "nokov_local"
    assert math.isclose(source.stamp_seconds, 1234567890.9876542)

    unusable = source.with_validity(False)
    assert unusable.source_valid is False
    assert unusable.seq == source.seq
    assert unusable.stamp_secs == source.stamp_secs
    assert unusable.stamp_nsecs == source.stamp_nsecs
    assert unusable.frame_id == source.frame_id


def test_guard_source_default_is_explicitly_sourceless():
    module = _guard_status_module()
    source = module.GuardStatusSource()

    assert source.source_valid is False
    assert source.has_source is False
    assert math.isnan(source.stamp_seconds)
    assert source.frame_id == ""
