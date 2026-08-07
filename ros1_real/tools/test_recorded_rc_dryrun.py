#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools/recorded_rc_dryrun.py"
SPEC = importlib.util.spec_from_file_location("recorded_rc_dryrun_tool", TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class RecordedRcDryRunToolTests(unittest.TestCase):
    def test_plot_defaults_to_real_yx_axis_order_and_left_right_mirror(self):
        args = TOOL._build_parser().parse_args(["recording.json"])
        self.assertEqual(args.plot_axis_order, "yx")
        self.assertEqual(args.plot_mirror, "horizontal")
        points = np.asarray([[1.0, 2.0], [-3.0, 4.0]])
        np.testing.assert_allclose(
            TOOL.plot_coordinates(points, args.plot_axis_order, np),
            [[2.0, 1.0], [4.0, -3.0]],
        )
        np.testing.assert_allclose(
            TOOL.plot_coordinates(points, "xy", np),
            points,
        )
        self.assertEqual(
            TOOL.plot_axis_limits(
                -6.0,
                2.0,
                plot_axis="horizontal",
                mirror=args.plot_mirror,
            ),
            (2.0, -6.0),
        )
        self.assertEqual(
            TOOL.plot_axis_limits(
                -3.0,
                4.0,
                plot_axis="vertical",
                mirror=args.plot_mirror,
            ),
            (-3.0, 4.0),
        )

    def test_axis_mapping_matches_deadband_saturation_and_reverse(self):
        common = {
            "pwm_min": 1000.0,
            "pwm_mid": 1500.0,
            "pwm_max": 2000.0,
            "deadband": 0.05,
        }
        self.assertEqual(
            TOOL.axis_from_pwm(1524, reverse=False, **common),
            0.0,
        )
        self.assertAlmostEqual(
            TOOL.axis_from_pwm(1525, reverse=False, **common),
            0.05,
        )
        self.assertEqual(
            TOOL.axis_from_pwm(2200, reverse=False, **common),
            1.0,
        )
        self.assertEqual(
            TOOL.axis_from_pwm(1000, reverse=True, **common),
            1.0,
        )

    def test_rc_integration_uses_zoh_yaw_rotation_and_xy_clamp(self):
        timeline = SimpleNamespace(
            duration=2.0,
            samples=[
                SimpleNamespace(
                    elapsed=0.0,
                    source_time=10.0,
                    channels=(1500, 2000, 1500),
                ),
                SimpleNamespace(
                    elapsed=1.0,
                    source_time=11.0,
                    channels=(1000, 1500, 1500),
                ),
            ],
        )
        parameters = {
            "forward_index": 1,
            "lateral_index": 0,
            "vertical_index": 2,
            "pwm_min": 1000.0,
            "pwm_mid": 1500.0,
            "pwm_max": 2000.0,
            "deadband": 0.05,
            "max_forward_speed": 1.0,
            "max_lateral_speed": 1.0,
            "forward_reverse": False,
            "lateral_reverse": True,
        }
        elapsed, positions, body, world, yaws = TOOL.integrate_recorded_rc(
            timeline,
            source_sample_times=[10.0, 11.0, 12.0],
            source_yaws=[0.0, 0.0, 0.0],
            start_xy=[2.0, -1.0],
            rc_parameters=parameters,
            max_xy_speed=0.8,
            np=np,
        )
        np.testing.assert_allclose(elapsed, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            positions,
            [[2.0, -1.0], [2.8, -1.0], [2.8, -0.2]],
            atol=1e-9,
        )
        np.testing.assert_allclose(body, [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(world, [[0.8, 0.0], [0.0, 0.8]])
        np.testing.assert_allclose(yaws, [0.0, 0.0])

    def test_replay_window_matches_the_complete_transition_subsequence(self):
        def event(t, channels):
            return {
                "t": float(t),
                "channels": list(channels),
                "lifecycle_state": "ACTIVE",
            }

        neutral = (1500, 1500, 1500)
        first = (1400, 1600, 1000)
        second = (1300, 1700, 1000)
        payload = {
            "rc_events": [
                event(0.0, neutral),
                event(0.5, neutral),
                event(1.0, first),
                event(1.5, first),
                event(2.0, second),
                event(2.5, second),
                event(3.0, neutral),
            ]
        }
        start, end, selected, exact = TOOL.find_replay_event_window(
            payload,
            expected_transitions=[
                (1600, 1400, 1000),
                (1700, 1300, 1000),
            ],
            motion_indices=(1, 0, 2),
            duration=2.0,
            start_delay=1.0,
        )
        self.assertEqual(start, 1.0)
        self.assertEqual(end, 3.0)
        self.assertEqual(len(selected), 4)
        self.assertTrue(exact)


if __name__ == "__main__":
    unittest.main()
