#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.predefined_rc_replay import (  # noqa: E402
    RcAxisEncoding,
    SCurveIntent,
    local_position_to_map,
    map_velocity_to_body,
    wrapped_angle_error,
)


class SCurveIntentTest(unittest.TestCase):
    def make_intent(self):
        return SCurveIntent(
            start_xy=(-1.85, 2.55),
            goal_xy=(2.75, -1.50),
            lateral_amplitude=0.65,
            duration=24.0,
        )

    def make_route2_intent(self):
        return SCurveIntent(
            start_xy=(-1.85, 2.55),
            goal_xy=(2.75, -1.50),
            lateral_amplitude=0.65,
            duration=26.0,
            lateral_profile=(
                (0.00, 0.00),
                (0.12, 0.28),
                (0.23, -0.09),
                (0.29, -0.80),
                (0.39, -1.25),
                (0.51, -1.01),
                (0.60, -0.37),
                (0.67, 0.32),
                (0.77, 0.83),
                (0.89, 0.55),
                (1.00, 0.00),
            ),
            arc_length_timing=True,
            arc_length_samples=4001,
        )

    def test_geometry_has_two_opposite_lobes_and_exact_endpoints(self):
        intent = self.make_intent()

        self.assertEqual(intent.position_at_progress(0.0), intent.start_xy)
        self.assertAlmostEqual(intent.position_at_progress(1.0)[0], intent.goal_xy[0])
        self.assertAlmostEqual(intent.position_at_progress(1.0)[1], intent.goal_xy[1])

        start_x, start_y = intent.start_xy
        dx, dy = intent.displacement
        normal_x, normal_y = intent.normal
        first = intent.position_at_progress(0.25)
        third = intent.position_at_progress(0.75)
        first_lateral = (
            (first[0] - (start_x + 0.25 * dx)) * normal_x
            + (first[1] - (start_y + 0.25 * dy)) * normal_y
        )
        third_lateral = (
            (third[0] - (start_x + 0.75 * dx)) * normal_x
            + (third[1] - (start_y + 0.75 * dy)) * normal_y
        )
        self.assertAlmostEqual(first_lateral, 0.65)
        self.assertAlmostEqual(third_lateral, -0.65)

    def test_smooth_time_scaling_starts_and_finishes_at_zero_speed(self):
        intent = self.make_intent()

        initial = intent.sample(0.0)
        final = intent.sample(intent.duration)
        late = intent.sample(intent.duration + 10.0)

        self.assertEqual(initial.velocity_xy, (0.0, 0.0))
        self.assertEqual(final.velocity_xy, (0.0, 0.0))
        self.assertTrue(final.complete)
        self.assertTrue(late.complete)
        self.assertAlmostEqual(final.position_xy[0], intent.goal_xy[0])
        self.assertAlmostEqual(final.position_xy[1], intent.goal_xy[1])

    def test_endpoint_tangents_follow_start_goal_diagonal(self):
        intent = self.make_intent()

        for progress in (0.0, 0.5, 1.0):
            tangent = intent.tangent_at_progress(progress)
            self.assertAlmostEqual(tangent[0], intent.displacement[0], places=7)
            self.assertAlmostEqual(tangent[1], intent.displacement[1], places=7)

    def test_default_replay_stays_below_real_flight_speed_limit(self):
        peak = self.make_intent().sampled_max_speed()

        self.assertGreater(peak, 0.45)
        self.assertLessEqual(peak, 0.50)

    def test_integrated_velocity_reaches_goal_without_position_feedback(self):
        intent = self.make_intent()
        count = 24001
        dt = intent.duration / float(count - 1)
        x_pos, y_pos = intent.start_xy
        previous = intent.sample(0.0).velocity_xy
        for index in range(1, count):
            current = intent.sample(index * dt).velocity_xy
            x_pos += 0.5 * (previous[0] + current[0]) * dt
            y_pos += 0.5 * (previous[1] + current[1]) * dt
            previous = current

        self.assertAlmostEqual(x_pos, intent.goal_xy[0], places=6)
        self.assertAlmostEqual(y_pos, intent.goal_xy[1], places=6)

    def test_route2_profile_interpolates_asymmetric_control_points(self):
        intent = self.make_route2_intent()

        self.assertEqual(intent.lateral_bounds, (-1.25, 0.83))
        for progress, expected_offset in intent.lateral_profile:
            self.assertAlmostEqual(
                intent.lateral_offset_at_progress(progress), expected_offset
            )
        sampled_offsets = [
            intent.lateral_offset_at_progress(index / 1000.0)
            for index in range(1001)
        ]
        self.assertGreaterEqual(min(sampled_offsets), -1.25)
        self.assertLessEqual(max(sampled_offsets), 0.83)

    def test_route2_arc_timing_preserves_speed_limit_and_endpoint_tangent(self):
        intent = self.make_route2_intent()

        self.assertAlmostEqual(intent.arc_length, 8.10768, places=4)
        self.assertAlmostEqual(intent.progress_at_arc_fraction(0.0), 0.0)
        self.assertAlmostEqual(intent.progress_at_arc_fraction(1.0), 1.0)
        self.assertGreater(intent.sampled_max_speed(), 0.46)
        self.assertLessEqual(intent.sampled_max_speed(), 0.50)
        for progress in (0.0, 1.0):
            tangent = intent.tangent_at_progress(progress)
            self.assertAlmostEqual(tangent[0], intent.displacement[0], places=7)
            self.assertAlmostEqual(tangent[1], intent.displacement[1], places=7)

    def test_route2_integrated_velocity_reaches_goal(self):
        intent = self.make_route2_intent()
        count = 26001
        dt = intent.duration / float(count - 1)
        x_pos, y_pos = intent.start_xy
        previous = intent.sample(0.0).velocity_xy
        for index in range(1, count):
            current = intent.sample(index * dt).velocity_xy
            x_pos += 0.5 * (previous[0] + current[0]) * dt
            y_pos += 0.5 * (previous[1] + current[1]) * dt
            previous = current

        self.assertAlmostEqual(x_pos, intent.goal_xy[0], places=4)
        self.assertAlmostEqual(y_pos, intent.goal_xy[1], places=4)


class RcEncodingTest(unittest.TestCase):
    @staticmethod
    def decode(pwm, maximum, reverse=False):
        if pwm >= 1500.0:
            value = (pwm - 1500.0) / 500.0
        else:
            value = (pwm - 1500.0) / 500.0
        if reverse:
            value = -value
        return value * maximum

    def test_overwrites_motion_axes_and_preserves_auxiliary_channels(self):
        encoding = RcAxisEncoding(lateral_reverse=True)
        source = [1500] * 18
        source[6] = 1900

        output = encoding.encode_motion(source, (0.4, -0.2, 0.0))

        self.assertEqual(source[0], 1500)
        self.assertEqual(output[6], 1900)
        self.assertAlmostEqual(self.decode(output[1], 1.0), 0.4, places=3)
        self.assertAlmostEqual(
            self.decode(output[0], 1.0, reverse=True), -0.2, places=3
        )
        self.assertEqual(output[2], 1500)

    def test_world_intent_is_yaw_compensated_before_rc_encoding(self):
        body = map_velocity_to_body(
            (1.0, 0.0, 0.0),
            yaw_local=math.radians(60.0),
            map_yaw=math.radians(30.0),
        )

        self.assertAlmostEqual(body[0], 0.0, places=7)
        self.assertAlmostEqual(body[1], -1.0, places=7)
        self.assertEqual(body[2], 0.0)

    def test_start_gate_uses_same_transform_as_map_lidar(self):
        mapped = local_position_to_map(
            (1.0, 0.0, 0.5),
            map_yaw=math.pi / 2.0,
            map_origin_xyz=(2.0, 3.0, -0.1),
        )

        self.assertAlmostEqual(mapped[0], 2.0)
        self.assertAlmostEqual(mapped[1], 4.0)
        self.assertAlmostEqual(mapped[2], 0.4)

    def test_start_yaw_error_wraps_across_pi_boundary(self):
        error = wrapped_angle_error(math.radians(-179.0), math.radians(179.0))

        self.assertAlmostEqual(math.degrees(error), 2.0, places=7)


if __name__ == "__main__":
    unittest.main()
