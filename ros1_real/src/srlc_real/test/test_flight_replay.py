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

from srlc_real_deployment.flight_replay import (  # noqa: E402
    FlightRecordingError,
    integrate_human_input_prediction,
    load_flight_timeline,
)
from srlc_real_deployment.replay_video_overlay import (  # noqa: E402
    StickHudRenderer,
    replay_elapsed_for_video_time,
)


def sample(
    timestamp,
    position,
    yaw=0.0,
    lifecycle="",
    armed=None,
    landed=None,
    human_input=None,
):
    value = {
        "t": timestamp,
        "position": list(position),
        "yaw": yaw,
        "lifecycle_state": lifecycle,
    }
    if armed is not None:
        value["armed"] = armed
    if landed is not None:
        value["landed_state"] = landed
    if human_input is not None:
        value["human_action"] = list(human_input)
    return value


class RecordingFixture:
    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def close(self):
        self.tempdir.cleanup()

    def json(self, payload, name="flight.json"):
        path = Path(self.tempdir.name) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def npz(self, payload, name="flight.npz"):
        path = Path(self.tempdir.name) / name
        np.savez_compressed(
            path,
            samples=np.asarray(payload["samples"], dtype=object),
            summary=payload.get("summary", {}),
        )
        return str(path)


class FlightReplayTimelineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = RecordingFixture()

    def tearDown(self):
        self.fixture.close()

    @staticmethod
    def current_payload():
        return {
            "summary": {"run_id": "real-flight"},
            "samples": [
                sample(0.95, (1.0, 0.0, 0.1), armed=False, landed=1),
                sample(
                    1.00,
                    (1.0, 0.0, 0.1),
                    lifecycle="TAKEOFF",
                    armed=True,
                    landed=1,
                ),
                sample(
                    1.05,
                    (1.1, 0.0, 0.4),
                    lifecycle="TAKEOFF",
                    armed=True,
                    landed=2,
                ),
                sample(
                    1.10,
                    (1.2, 0.2, 0.8),
                    lifecycle="ACTIVE",
                    armed=True,
                    landed=2,
                ),
                sample(
                    1.20,
                    (1.3, 0.4, 0.1),
                    lifecycle="TERMINATED",
                    armed=False,
                    landed=1,
                ),
                sample(1.25, (1.3, 0.4, 0.1), armed=False, landed=1),
            ],
        }

    def test_auto_crops_takeoff_to_landing_and_applies_map_transform(self):
        path = self.fixture.json(self.current_payload())
        timeline = load_flight_timeline(
            path,
            map_yaw_deg=90.0,
            map_origin_xyz=(10.0, 20.0, 30.0),
        )

        self.assertAlmostEqual(timeline.source_start_time, 1.0)
        self.assertAlmostEqual(timeline.source_end_time, 1.2)
        self.assertAlmostEqual(timeline.duration, 0.2)
        self.assertEqual(timeline.window_reason, "auto_takeoff_to_landed")
        self.assertEqual(timeline.source_run_id, "real-flight")
        self.assertAlmostEqual(timeline.samples[0].position[0], 10.0)
        self.assertAlmostEqual(timeline.samples[0].position[1], 21.0)
        self.assertAlmostEqual(timeline.samples[0].position[2], 30.1)
        self.assertAlmostEqual(timeline.samples[0].yaw, math.pi / 2.0)

    def test_shortest_arc_yaw_interpolation_crosses_pi(self):
        payload = {
            "samples": [
                sample(0.0, (0.0, 0.0, 0.0), yaw=math.radians(179.0)),
                sample(1.0, (1.0, 0.0, 0.0), yaw=math.radians(-179.0)),
            ]
        }
        path = self.fixture.json(payload)
        timeline = load_flight_timeline(
            path, start_time=0.0, end_time=1.0, max_sample_gap=1.1
        )

        midpoint = timeline.sample_at(0.5)
        self.assertAlmostEqual(abs(midpoint.yaw), math.pi, places=6)
        self.assertAlmostEqual(midpoint.position[0], 0.5)
        self.assertAlmostEqual(midpoint.linear_velocity[0], 1.0)
        self.assertAlmostEqual(midpoint.yaw_rate, math.radians(2.0))

    def test_human_input_is_held_and_integrated_in_recorded_yaw_frame(self):
        payload = {
            "samples": [
                sample(
                    0.0,
                    (2.0, -1.0, 0.4),
                    yaw=math.pi / 2.0,
                    human_input=(1.0, 0.0, 0.0),
                ),
                sample(
                    1.0,
                    (2.0, -1.0, 0.8),
                    yaw=math.pi / 2.0,
                    human_input=(0.0, -1.0, 0.0),
                ),
                sample(
                    2.0,
                    (2.0, -1.0, 0.8),
                    yaw=math.pi / 2.0,
                    human_input=(0.0, 0.0, 0.0),
                ),
            ]
        }
        timeline = load_flight_timeline(
            self.fixture.json(payload),
            start_time=0.0,
            end_time=2.0,
            max_sample_gap=1.1,
        )

        self.assertTrue(timeline.has_human_input)
        self.assertEqual(timeline.human_input_at(0.5), (1.0, 0.0, 0.0))
        self.assertEqual(timeline.human_input_at(1.5), (0.0, -1.0, 0.0))
        predicted = integrate_human_input_prediction(
            timeline,
            max_xy_speed=0.5,
        )
        self.assertAlmostEqual(predicted[1][0], 2.0)
        self.assertAlmostEqual(predicted[1][1], -0.5)
        self.assertAlmostEqual(predicted[1][2], 0.8)
        self.assertAlmostEqual(predicted[2][0], 2.5)
        self.assertAlmostEqual(predicted[2][1], -0.5)

    def test_input_prediction_requires_recorded_human_action(self):
        timeline = load_flight_timeline(
            self.fixture.json(
                {
                    "samples": [
                        sample(0.0, (0.0, 0.0, 0.0)),
                        sample(1.0, (0.0, 0.0, 0.0)),
                    ]
                }
            ),
            start_time=0.0,
            end_time=1.0,
            max_sample_gap=1.1,
        )
        self.assertFalse(timeline.has_human_input)
        with self.assertRaisesRegex(FlightRecordingError, "no human_action"):
            integrate_human_input_prediction(timeline)

    def test_right_stick_hud_is_rgba_and_video_time_maps_to_replay(self):
        renderer = StickHudRenderer(width=180, height=180)
        neutral = renderer.render((0.0, 0.0, 0.0))
        forward_right = renderer.render(
            (1.0, -1.0, 0.0),
            history=[(0.0, 0.0, 0.0), (0.5, -0.5, 0.0)],
        )
        self.assertEqual(len(neutral), 180 * 180 * 4)
        self.assertNotEqual(neutral, forward_right)
        self.assertEqual(
            replay_elapsed_for_video_time(
                0.5,
                lead_seconds=1.0,
                speed=2.0,
                replay_duration=10.0,
            ),
            0.0,
        )
        self.assertEqual(
            replay_elapsed_for_video_time(
                3.0,
                lead_seconds=1.0,
                speed=2.0,
                replay_duration=10.0,
            ),
            4.0,
        )

    def test_legacy_recording_requires_explicit_window_and_interpolates_boundaries(self):
        payload = {
            "samples": [
                sample(0.0, (0.0, 0.0, 0.0)),
                sample(0.5, (1.0, 0.0, 0.0)),
                sample(1.0, (2.0, 0.0, 0.0)),
            ]
        }
        path = self.fixture.json(payload)

        with self.assertRaisesRegex(FlightRecordingError, "cannot infer"):
            load_flight_timeline(path)

        timeline = load_flight_timeline(
            path,
            start_time=0.25,
            end_time=0.75,
            max_sample_gap=0.3,
        )
        self.assertEqual(timeline.window_reason, "explicit")
        self.assertAlmostEqual(timeline.samples[0].position[0], 0.5)
        self.assertAlmostEqual(timeline.samples[-1].position[0], 1.5)
        self.assertAlmostEqual(timeline.duration, 0.5)

    def test_loads_trusted_recorder_npz(self):
        path = self.fixture.npz(self.current_payload())
        timeline = load_flight_timeline(path)
        self.assertEqual(timeline.source_run_id, "real-flight")
        self.assertEqual(timeline.window_reason, "auto_takeoff_to_landed")

    def test_rejects_large_gap_inside_selected_window(self):
        payload = {
            "samples": [
                sample(
                    0.0,
                    (0.0, 0.0, 0.0),
                    lifecycle="TAKEOFF",
                    armed=True,
                    landed=1,
                ),
                sample(
                    0.5,
                    (1.0, 0.0, 1.0),
                    lifecycle="ACTIVE",
                    armed=True,
                    landed=2,
                ),
                sample(
                    0.6,
                    (1.0, 0.0, 0.0),
                    lifecycle="TERMINATED",
                    armed=False,
                    landed=1,
                ),
            ]
        }
        path = self.fixture.json(payload)
        with self.assertRaisesRegex(FlightRecordingError, "sample gap"):
            load_flight_timeline(path, max_sample_gap=0.2)

    def test_rejects_nonfinite_pose_and_nonmonotonic_time(self):
        nonfinite = {
            "samples": [
                sample(0.0, (0.0, 0.0, 0.0)),
                sample(1.0, (float("nan"), 0.0, 0.0)),
            ]
        }
        with self.assertRaisesRegex(FlightRecordingError, "must be finite"):
            load_flight_timeline(
                self.fixture.json(nonfinite, "nonfinite.json"),
                start_time=0.0,
                end_time=1.0,
            )

        unordered = {
            "samples": [
                sample(1.0, (0.0, 0.0, 0.0)),
                sample(0.5, (1.0, 0.0, 0.0)),
            ]
        }
        with self.assertRaisesRegex(FlightRecordingError, "strictly increasing"):
            load_flight_timeline(
                self.fixture.json(unordered, "unordered.json"),
                start_time=0.5,
                end_time=1.0,
            )


if __name__ == "__main__":
    unittest.main()
