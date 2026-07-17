#!/usr/bin/env python3

import unittest
from pathlib import Path


RECORDER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "srlc_real_recorder.py"
)


class RecorderContractTest(unittest.TestCase):
    def test_records_one_shot_lifecycle_and_px4_state(self):
        source = RECORDER.read_text(encoding="utf-8")

        for field in (
            '"lifecycle_state"',
            '"effective_mode"',
            '"policy_active"',
            '"session_consumed"',
            '"fault_reason"',
            '"mavros_mode"',
            '"armed"',
            '"landed_state"',
            '"safety_distance"',
            '"px4_local_position"',
            '"px4_local_velocity"',
            '"setpoint_type_mask"',
        ):
            self.assertIn(field, source)

        for topic in (
            "/mavros/state",
            "/mavros/extended_state",
            "/tunnel_nav/lifecycle_state",
            "/tunnel_nav/effective_mode",
            "/tunnel_nav/policy_active",
            "/tunnel_nav/session_consumed",
            "/tunnel_nav/fault_reason",
            "/srlc/lidar/min_safety_distance",
            "/mavros/local_position/odom",
            "/mavros/local_position/velocity_local",
        ):
            self.assertIn(topic, source)


if __name__ == "__main__":
    unittest.main()
