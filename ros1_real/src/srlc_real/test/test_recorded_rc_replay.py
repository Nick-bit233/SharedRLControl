#!/usr/bin/env python3

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.recorded_rc_replay import (  # noqa: E402
    RcChannelOverlay,
    RecordingFormatError,
    load_recorded_rc_timeline,
    wrapped_angle_error,
)


def channels(forward=1500, lateral=1500, vertical=1500, auxiliary=1200):
    values = [1500] * 8
    values[0] = lateral
    values[1] = forward
    values[2] = vertical
    values[6] = auxiliary
    return values


def sample(
    timestamp,
    lifecycle,
    rc=None,
    position=(1.0, 2.0, 1.1),
    yaw=0.25,
    control_mode="ASSIST",
    effective_mode="ASSIST",
):
    return {
        "t": timestamp,
        "lifecycle_state": lifecycle,
        "rc": channels() if rc is None else rc,
        "position": list(position),
        "yaw": yaw,
        "control_mode": control_mode,
        "effective_mode": effective_mode,
    }


class RecordingFixture:
    def __init__(self, testcase):
        self.testcase = testcase
        self.tempdir = tempfile.TemporaryDirectory()

    def close(self):
        self.tempdir.cleanup()

    def json(self, payload, name="recording.json"):
        path = Path(self.tempdir.name) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def npz(self, payload, name="recording.npz"):
        path = Path(self.tempdir.name) / name
        arrays = {}
        for key, value in payload.items():
            if key in ("samples", "rc_events"):
                arrays[key] = np.array(value, dtype=object)
            else:
                arrays[key] = value
        np.savez_compressed(path, **arrays)
        return str(path)


class RecordedRcTimelineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = RecordingFixture(self)

    def tearDown(self):
        self.fixture.close()

    @staticmethod
    def active_payload():
        return {
            "summary": {"run_id": "manual-flight"},
            "samples": [
                sample(4.95, "TAKEOFF", rc=channels(forward=1500)),
                sample(
                    5.00,
                    "ACTIVE",
                    rc=channels(forward=1550),
                    position=(1.2, -0.4, 1.0),
                    yaw=-0.5,
                    effective_mode="ASSIST_IDLE",
                ),
                sample(
                    5.05,
                    "ACTIVE",
                    rc=channels(forward=1600, lateral=1450),
                    position=(1.21, -0.4, 1.0),
                    yaw=-0.5,
                ),
                sample(
                    5.10,
                    "ACTIVE",
                    rc=channels(forward=1700, lateral=1400),
                    position=(1.23, -0.4, 1.0),
                    yaw=-0.49,
                ),
                sample(5.15, "FAULT_HOLD", rc=channels(forward=1500)),
            ],
        }

    def test_loads_unique_active_window_and_uses_zero_order_hold(self):
        path = self.fixture.json(self.active_payload())

        timeline = load_recorded_rc_timeline(path)

        self.assertEqual(timeline.source_stream, "sample_snapshots")
        self.assertEqual(timeline.source_run_id, "manual-flight")
        self.assertEqual(len(timeline.samples), 3)
        self.assertAlmostEqual(timeline.duration, 0.15)
        self.assertAlmostEqual(timeline.source_start_time, 5.0)
        self.assertAlmostEqual(timeline.source_end_time, 5.15)
        self.assertEqual(timeline.reference_position, (1.2, -0.4, 1.0))
        self.assertAlmostEqual(timeline.reference_yaw, -0.5)
        self.assertEqual(timeline.control_modes, ("ASSIST",))
        self.assertEqual(timeline.effective_modes, ("ASSIST_IDLE", "ASSIST"))

        first_index, first = timeline.sample_at(0.049)
        second_index, second = timeline.sample_at(0.050)
        final_index, final = timeline.sample_at(999.0)
        self.assertEqual(first_index, 0)
        self.assertEqual(first.channels[1], 1550)
        self.assertEqual(second_index, 1)
        self.assertEqual(second.channels[1], 1600)
        self.assertEqual(final_index, 2)
        self.assertEqual(final.channels[1], 1700)

    def test_prefers_high_rate_rc_events_but_uses_samples_for_start_pose(self):
        payload = self.active_payload()
        payload["rc_events"] = [
            {
                "t": 4.98,
                "lifecycle_state": "TAKEOFF",
                "channels": channels(forward=1500),
            },
            {
                "t": 5.01,
                "lifecycle_state": "ACTIVE",
                "channels": channels(forward=1510),
                "position": [1.205, -0.4, 1.0],
                "yaw": -0.5,
                "control_mode": "ASSIST",
                "effective_mode": "ASSIST_IDLE",
            },
            {
                "t": 5.03,
                "lifecycle_state": "ACTIVE",
                "channels": channels(forward=1520),
            },
            {
                "t": 5.05,
                "lifecycle_state": "ACTIVE",
                "channels": channels(forward=1530),
            },
            {
                "t": 5.07,
                "lifecycle_state": "FAULT_HOLD",
                "channels": channels(forward=1500),
            },
        ]
        path = self.fixture.json(payload)

        timeline = load_recorded_rc_timeline(path)

        self.assertEqual(timeline.source_stream, "rc_events")
        self.assertEqual(len(timeline.samples), 3)
        self.assertAlmostEqual(timeline.duration, 0.06)
        self.assertEqual(timeline.samples[1].channels[1], 1520)
        self.assertEqual(timeline.reference_position, (1.205, -0.4, 1.0))

    def test_loads_recorder_npz_object_arrays(self):
        payload = self.active_payload()
        path = self.fixture.npz(payload)

        timeline = load_recorded_rc_timeline(path)

        self.assertEqual(len(timeline.samples), 3)
        self.assertEqual(timeline.source_run_id, "manual-flight")
        self.assertEqual(timeline.samples[-1].channels[0], 1400)

    def test_monotonic_event_stream_is_independent_of_sample_clock_jump(self):
        payload = self.active_payload()
        payload["samples"][4]["t"] = 4.0
        payload["rc_events"] = [
            {
                "t": 2.00,
                "lifecycle_state": "ACTIVE",
                "channels": channels(forward=1550),
                "position": [1.2, -0.4, 1.0],
                "yaw": -0.5,
            },
            {
                "t": 2.02,
                "lifecycle_state": "ACTIVE",
                "channels": channels(forward=1600),
                "position": [1.2, -0.4, 1.0],
                "yaw": -0.5,
            },
            {
                "t": 2.04,
                "lifecycle_state": "FAULT_HOLD",
                "channels": channels(),
            },
        ]
        path = self.fixture.json(payload)

        timeline = load_recorded_rc_timeline(path)

        self.assertEqual(timeline.source_stream, "rc_events")
        self.assertEqual(len(timeline.samples), 2)
        self.assertAlmostEqual(timeline.duration, 0.04)

    def test_legacy_recording_requires_and_accepts_explicit_window(self):
        payload = {
            "summary": {"run_id": "legacy"},
            "samples": [
                sample(0.95, "", rc=channels(forward=1500)),
                sample(
                    1.00,
                    "",
                    rc=channels(forward=1600),
                    position=(-1.0, 0.5, 1.2),
                    yaw=0.75,
                ),
                sample(1.05, "", rc=channels(forward=1700)),
                sample(1.10, "", rc=channels(forward=1800)),
                sample(1.15, "", rc=channels(forward=1500)),
            ],
        }
        path = self.fixture.json(payload)

        with self.assertRaisesRegex(
            RecordingFormatError, "replay_start_time"
        ):
            load_recorded_rc_timeline(path)

        timeline = load_recorded_rc_timeline(
            path,
            replay_start_time=0.99,
            replay_end_time=1.11,
        )
        self.assertEqual(len(timeline.samples), 3)
        self.assertAlmostEqual(timeline.duration, 0.11)
        self.assertEqual(timeline.reference_position, (-1.0, 0.5, 1.2))
        self.assertAlmostEqual(timeline.reference_yaw, 0.75)

        with self.assertRaisesRegex(RecordingFormatError, "gap"):
            load_recorded_rc_timeline(
                path,
                replay_start_time=0.0,
                replay_end_time=1.11,
            )

    def test_rejects_ambiguous_active_intervals(self):
        payload = {
            "samples": [
                sample(0.00, "ACTIVE"),
                sample(0.05, "ACTIVE"),
                sample(0.10, "WAIT_READY"),
                sample(0.15, "ACTIVE"),
                sample(0.20, "ACTIVE"),
                sample(0.25, "TERMINATED"),
            ]
        }
        path = self.fixture.json(payload)

        with self.assertRaisesRegex(RecordingFormatError, "disjoint"):
            load_recorded_rc_timeline(path)

    def test_rejects_sparse_or_out_of_range_motion_data(self):
        sparse = self.active_payload()
        sparse["samples"][2]["t"] = 5.40
        sparse["samples"][3]["t"] = 5.45
        sparse["samples"][4]["t"] = 5.50
        sparse_path = self.fixture.json(sparse, "sparse.json")
        with self.assertRaisesRegex(RecordingFormatError, "gap"):
            load_recorded_rc_timeline(sparse_path)

        invalid = self.active_payload()
        invalid["samples"][1]["rc"][1] = 2500
        invalid_path = self.fixture.json(invalid, "invalid.json")
        with self.assertRaisesRegex(RecordingFormatError, "outside"):
            load_recorded_rc_timeline(invalid_path)


class RcChannelOverlayTest(unittest.TestCase):
    def test_replays_only_motion_channels_and_preserves_live_safety_switches(self):
        overlay = RcChannelOverlay()
        live = channels(forward=1510, lateral=1520, vertical=1530, auxiliary=1900)
        recorded = channels(
            forward=1700,
            lateral=1300,
            vertical=1100,
            auxiliary=1000,
        )

        output = overlay.overlay(live, recorded)

        self.assertEqual(output[0:3], [1300, 1700, 1100])
        self.assertEqual(output[6], 1900)
        self.assertEqual(live[0:3], [1520, 1510, 1530])

    def test_neutral_centers_only_motion_channels(self):
        overlay = RcChannelOverlay(pwm_mid=1494)
        live = channels(forward=1700, lateral=1300, vertical=1100, auxiliary=1900)

        output = overlay.neutral(live)

        self.assertEqual(output[0:3], [1494, 1494, 1494])
        self.assertEqual(output[6], 1900)

    def test_wrapped_start_yaw_error_crosses_pi_boundary(self):
        error = wrapped_angle_error(
            math.radians(-179.0), math.radians(179.0)
        )

        self.assertAlmostEqual(math.degrees(error), 2.0, places=7)


if __name__ == "__main__":
    unittest.main()
