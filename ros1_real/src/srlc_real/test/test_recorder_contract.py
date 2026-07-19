#!/usr/bin/env python3

import unittest
from pathlib import Path


RECORDER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "srlc_real_recorder.py"
)


class RecorderContractTest(unittest.TestCase):
    def test_records_clearance_policy_and_guard_observability(self):
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
            '"px4_local_position"',
            '"px4_local_velocity"',
            '"setpoint_type_mask"',
            '"raw_center_distance"',
            '"policy_min_distance"',
            '"model_min_distance"',
            '"clearance_valid"',
            '"surface_clearance"',
            '"clearance_center_distance"',
            '"clearance_source_stamp"',
            '"clearance_source_age"',
            '"nearest_obstacle_point"',
            '"escape_direction"',
            '"effective_guard_state"',
            '"shadow_guard_state"',
            '"shadow_decision"',
            '"shadow_would_intervene"',
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
            "/srlc/lidar/min_distance",
            "/srlc/lidar/obstacle_clearance",
            "/tunnel_nav/clearance_guard_state",
            "/tunnel_nav/clearance_guard_shadow_state",
            "/mavros/local_position/odom",
            "/mavros/local_position/velocity_local",
        ):
            self.assertIn(topic, source)

        self.assertIn("from srlc_real.msg import ObstacleClearance", source)
        self.assertIn("msg.header.stamp.to_sec()", source)
        self.assertIn("now.to_sec() - self.clearance_source_stamp", source)
        self.assertIn("self.lidar_range * (1.0 - policy_norm)", source)
        self.assertIn("PROXIMITY_HOLD", source)
        self.assertIn("PROXIMITY_ESCAPE", source)
        self.assertIn("COLLISION", source)
        self.assertNotIn("/srlc/lidar/min_safety_distance", source)
        self.assertNotIn("lidar_safety_distance", source)
        self.assertNotIn('"safety_distance"', source)
        self.assertNotIn('"min_safety_distance"', source)
        self.assertNotIn("safety_d=", source)

    def test_summary_uses_unambiguous_dual_channel_names(self):
        source = RECORDER.read_text(encoding="utf-8")

        for field in (
            '"min_raw_center_distance"',
            '"min_policy_distance"',
            '"min_surface_clearance"',
            '"latest_guard_state"',
            '"latest_shadow_decision"',
        ):
            self.assertIn(field, source)

        self.assertNotIn("min_safety_distance", source)


if __name__ == "__main__":
    unittest.main()
