#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.one_shot_flight import (  # noqa: E402
    FlightAction,
    FlightSnapshot,
    LifecycleState,
    OneShotFlightConfig,
    OneShotFlightLifecycle,
)


class OneShotFlightLifecycleTest(unittest.TestCase):
    def make_core(self, **overrides):
        values = {
            "takeoff_height": 1.0,
            "takeoff_lower_margin": 0.2,
            "takeoff_upper_margin": 0.2,
            "takeoff_max_abs_vz": 0.25,
            "takeoff_confirm_duration": 0.5,
            "takeoff_timeout": 15.0,
            "takeoff_max_overshoot": 0.5,
            "takeoff_max_xy_drift": 0.5,
            "takeoff_max_climb_speed": 0.4,
            "takeoff_max_vertical_accel": 0.5,
            "takeoff_max_tracking_error": 0.25,
            "enable_proximity_hold": True,
            "proximity_enter_dist": 0.10,
            "proximity_release_dist": 0.15,
            "proximity_release_duration": 0.20,
            "enable_collision_detection": True,
            "collision_dist": 0.05,
            "safety_activation_height": 0.3,
            "input_recovery_grace": 1.0,
            "fault_response": "auto_land",
            "fault_land_mode": "AUTO.LAND",
            "fault_land_confirm_timeout": 2.0,
            "fault_land_retry_interval": 0.5,
            "fault_land_max_attempts": 3,
        }
        values.update(overrides)
        return OneShotFlightLifecycle(OneShotFlightConfig(**values))

    @staticmethod
    def snapshot(
        now,
        *,
        connected=True,
        armed=True,
        mode="OFFBOARD",
        position=(-1.8, 0.0, 0.1),
        velocity=(0.0, 0.0, 0.0),
        odom_fresh=True,
        rc_fresh=True,
        lidar_fresh=True,
        safety_distance=2.0,
        landed=True,
        external_fault=None,
    ):
        return FlightSnapshot(
            now=now,
            connected=connected,
            armed=armed,
            mode=mode,
            position=position,
            velocity=velocity,
            odom_fresh=odom_fresh,
            rc_fresh=rc_fresh,
            lidar_fresh=lidar_fresh,
            safety_distance=safety_distance,
            landed=landed,
            external_fault=external_fault,
        )

    def start_takeoff(self, core, now=0.0):
        decision = core.update(self.snapshot(now))
        self.assertEqual(decision.state, LifecycleState.TAKEOFF)
        self.assertEqual(decision.action, FlightAction.TAKEOFF_HOLD)
        self.assertTrue(decision.session_consumed)
        return decision

    def advance_profile_to_target(self, core, *, start_time=0.0, dt=0.05):
        decision = self.start_takeoff(core, now=start_time)
        position = list(decision.target)
        now = start_time
        history = [decision]
        for _ in range(400):
            now += dt
            if decision.target is not None:
                position[2] = decision.target[2]
            vz = (
                decision.target_velocity[2]
                if decision.target_velocity is not None
                else 0.0
            )
            decision = core.update(
                self.snapshot(
                    now,
                    position=tuple(position),
                    velocity=(0.0, 0.0, vz),
                    landed=False,
                )
            )
            history.append(decision)
            if (
                decision.target == core.takeoff_target
                and decision.target_velocity == (0.0, 0.0, 0.0)
            ):
                return now, position, history
        self.fail("takeoff profile did not reach its final target")

    def advance_to_active(self, core):
        reached_at, _, _ = self.advance_profile_to_target(core)
        target = core.takeoff_target
        core.update(
            self.snapshot(
                reached_at + 0.05,
                position=target,
                velocity=(0.0, 0.0, 0.0),
                landed=False,
            )
        )
        active = core.update(
            self.snapshot(
                reached_at + 0.05 + core.config.takeoff_confirm_duration + 0.01,
                position=target,
                velocity=(0.0, 0.0, 0.0),
                landed=False,
            )
        )
        self.assertEqual(active.state, LifecycleState.ACTIVE)
        self.assertEqual(active.action, FlightAction.ACTIVE_CONTROL)
        return active

    def test_waits_for_manual_arm_before_first_offboard_takeoff(self):
        core = self.make_core()

        decision = core.update(
            self.snapshot(0.0, armed=False, mode="POSCTL", landed=True)
        )
        self.assertEqual(decision.state, LifecycleState.WAIT_ARMED)
        self.assertEqual(decision.action, FlightAction.PRESTREAM_HOLD)

        decision = core.update(
            self.snapshot(0.1, armed=True, mode="POSCTL", landed=True)
        )
        self.assertEqual(decision.state, LifecycleState.WAIT_OFFBOARD)
        self.assertEqual(decision.action, FlightAction.PRESTREAM_HOLD)

        decision = core.update(
            self.snapshot(0.2, armed=True, mode="OFFBOARD", landed=True)
        )
        self.assertEqual(decision.state, LifecycleState.TAKEOFF)
        self.assertEqual(decision.target, (-1.8, 0.0, 0.1))
        self.assertEqual(decision.target_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(core.takeoff_target, (-1.8, 0.0, 1.1))

    def test_takeoff_profile_limits_acceleration_speed_and_position_step(self):
        core = self.make_core()
        first = self.start_takeoff(core)
        self.assertEqual(first.target, self.snapshot(0.0).position)

        position = list(first.target)
        previous = first
        previous_vz = 0.0
        previous_z = first.target[2]
        max_seen_vz = 0.0
        for step in range(1, 81):
            now = step * 0.05
            position[2] = previous.target[2]
            current = core.update(
                self.snapshot(
                    now,
                    position=tuple(position),
                    velocity=(0.0, 0.0, previous_vz),
                    landed=False,
                )
            )
            current_vz = current.target_velocity[2]
            self.assertGreaterEqual(current.target[2] + 1e-9, previous_z)
            self.assertLessEqual(current_vz, 0.4 + 1e-9)
            self.assertLessEqual(abs(current_vz - previous_vz), 0.5 * 0.05 + 1e-9)
            max_seen_vz = max(max_seen_vz, current_vz)
            previous = current
            previous_vz = current_vz
            previous_z = current.target[2]
            if current.target == core.takeoff_target:
                break

        self.assertGreater(max_seen_vz, 0.35)
        self.assertEqual(previous.target, core.takeoff_target)
        self.assertEqual(previous.target_velocity, (0.0, 0.0, 0.0))

    def test_takeoff_profile_never_leads_measured_height_beyond_tracking_limit(self):
        core = self.make_core()
        decision = self.start_takeoff(core)
        measured_position = self.snapshot(0.0).position

        for step in range(1, 101):
            decision = core.update(
                self.snapshot(
                    step * 0.05,
                    position=measured_position,
                    velocity=(0.0, 0.0, 0.0),
                    landed=False,
                )
            )

        self.assertLessEqual(
            decision.target[2] - measured_position[2],
            0.25 + 1e-9,
        )
        self.assertEqual(decision.target_velocity, (0.0, 0.0, 0.0))

    def test_profile_replans_after_tracking_pause_during_deceleration(self):
        core = self.make_core()
        decision = self.start_takeoff(core)
        position = list(decision.target)

        for step in range(1, 100):
            position[2] = decision.target[2]
            decision = core.update(
                self.snapshot(
                    step * 0.05,
                    position=tuple(position),
                    velocity=(0.0, 0.0, decision.target_velocity[2]),
                    landed=False,
                )
            )
            if core._takeoff_decelerating:
                break
        self.assertTrue(core._takeoff_decelerating)
        self.assertLess(decision.target[2], core.takeoff_target[2])

        lagged_position = (
            decision.target[0],
            decision.target[1],
            decision.target[2] - core.config.takeoff_max_tracking_error,
        )
        paused = core.update(
            self.snapshot(
                (step + 1) * 0.05,
                position=lagged_position,
                velocity=(0.0, 0.0, 0.0),
                landed=False,
            )
        )
        self.assertEqual(paused.target_velocity, (0.0, 0.0, 0.0))

        position = list(paused.target)
        for resume_step in range(step + 2, step + 102):
            position[2] = paused.target[2]
            paused = core.update(
                self.snapshot(
                    resume_step * 0.05,
                    position=tuple(position),
                    velocity=(0.0, 0.0, paused.target_velocity[2]),
                    landed=False,
                )
            )
            if paused.target == core.takeoff_target:
                break

        self.assertEqual(paused.target, core.takeoff_target)

    def test_offboard_loss_terminates_and_never_creates_another_target(self):
        core = self.make_core()
        self.start_takeoff(core)
        final_target = core.takeoff_target

        terminated = core.update(
            self.snapshot(
                1.0,
                mode="POSCTL",
                position=(-1.7, 0.1, 0.6),
                landed=False,
            )
        )
        self.assertEqual(terminated.state, LifecycleState.TERMINATED)
        self.assertEqual(terminated.action, FlightAction.STOP_STREAM)
        self.assertEqual(terminated.reason, "OFFBOARD_LOST")

        reentered = core.update(
            self.snapshot(
                2.0,
                mode="OFFBOARD",
                position=(-1.7, 0.1, 0.6),
                landed=False,
            )
        )
        self.assertEqual(reentered.state, LifecycleState.TERMINATED)
        self.assertEqual(reentered.action, FlightAction.STOP_STREAM)
        self.assertEqual(core.takeoff_target, final_target)

    def test_activates_after_vertical_band_is_stable_without_xy_success_gate(self):
        core = self.make_core()
        reached_at, _, _ = self.advance_profile_to_target(core)

        within_band = core.update(
            self.snapshot(
                reached_at + 0.1,
                position=(-1.5, 0.3, 0.95),
                velocity=(0.3, 0.0, 0.20),
                landed=False,
            )
        )
        self.assertEqual(within_band.state, LifecycleState.TAKEOFF)

        active = core.update(
            self.snapshot(
                reached_at + 0.7,
                position=(-1.5, 0.3, 1.05),
                velocity=(0.3, 0.0, 0.10),
                landed=False,
            )
        )
        self.assertEqual(active.state, LifecycleState.ACTIVE)
        self.assertEqual(active.action, FlightAction.ACTIVE_CONTROL)

    def test_height_band_cannot_activate_before_profile_reaches_final_target(self):
        core = self.make_core(takeoff_confirm_duration=0.1)
        self.start_takeoff(core)

        core.update(
            self.snapshot(
                0.05,
                position=(-1.8, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                landed=False,
            )
        )
        still_takeoff = core.update(
            self.snapshot(
                0.25,
                position=(-1.8, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                landed=False,
            )
        )

        self.assertEqual(still_takeoff.state, LifecycleState.TAKEOFF)
        self.assertLess(still_takeoff.target[2], core.takeoff_target[2])

    def test_takeoff_target_never_changes_while_input_recovers(self):
        core = self.make_core()
        first = self.start_takeoff(core)
        final_target = core.takeoff_target

        climbing = core.update(
            self.snapshot(
                0.5,
                position=first.target,
                landed=False,
            )
        )

        hold = core.update(
            self.snapshot(
                1.0,
                position=(-1.7, 0.0, climbing.target[2]),
                lidar_fresh=False,
                landed=False,
            )
        )
        self.assertEqual(hold.action, FlightAction.FAULT_HOLD)
        self.assertEqual(hold.reason, "INPUT_RECOVERY_HOLD")

        resumed = core.update(
            self.snapshot(
                1.5,
                position=(-1.7, 0.0, climbing.target[2]),
                lidar_fresh=True,
                landed=False,
            )
        )
        self.assertEqual(resumed.action, FlightAction.TAKEOFF_HOLD)
        self.assertLess(resumed.target[2], final_target[2])
        self.assertLessEqual(resumed.target_velocity[2], 0.05 + 1e-9)
        self.assertEqual(core.takeoff_target, final_target)

    def test_active_input_recovery_hold_keeps_takeoff_altitude_and_xy(self):
        core = self.make_core()
        active = self.advance_to_active(core)
        target_z = active.target[2]
        now = core._takeoff_confirm_started_at + core.config.takeoff_confirm_duration + 0.1

        first = core.update(
            self.snapshot(
                now,
                position=(-1.5, 0.2, target_z + 0.18),
                lidar_fresh=False,
                landed=False,
            )
        )
        second = core.update(
            self.snapshot(
                now + 0.1,
                position=(-1.3, 0.4, target_z + 0.25),
                lidar_fresh=False,
                landed=False,
            )
        )

        self.assertEqual(first.reason, "INPUT_RECOVERY_HOLD")
        self.assertEqual(first.target, (-1.5, 0.2, target_z))
        self.assertEqual(second.target, first.target)

    def test_proximity_hold_captures_one_target_and_releases_with_hysteresis(self):
        core = self.make_core()
        active = self.advance_to_active(core)
        target_z = active.target[2]
        now = core._takeoff_confirm_started_at + core.config.takeoff_confirm_duration + 0.1

        entered = core.update(
            self.snapshot(
                now,
                position=(-1.4, 0.2, target_z - 0.16),
                safety_distance=0.10,
                landed=False,
            )
        )
        near_release = core.update(
            self.snapshot(
                now + 0.05,
                position=(-1.2, 0.4, target_z + 0.12),
                safety_distance=0.14,
                landed=False,
            )
        )
        release_started = core.update(
            self.snapshot(
                now + 0.10,
                position=(-1.1, 0.5, target_z + 0.10),
                safety_distance=0.15,
                landed=False,
            )
        )
        still_holding = core.update(
            self.snapshot(
                now + 0.29,
                position=(-1.0, 0.6, target_z),
                safety_distance=0.20,
                landed=False,
            )
        )
        released = core.update(
            self.snapshot(
                now + 0.31,
                position=(-1.0, 0.6, target_z),
                safety_distance=0.20,
                landed=False,
            )
        )

        self.assertEqual(entered.reason, "PROXIMITY_HOLD")
        self.assertEqual(entered.target, (-1.4, 0.2, target_z))
        self.assertEqual(near_release.target, entered.target)
        self.assertEqual(release_started.target, entered.target)
        self.assertEqual(still_holding.target, entered.target)
        self.assertEqual(released.action, FlightAction.ACTIVE_CONTROL)
        self.assertEqual(released.reason, "ACTIVE")

    def test_proximity_hold_can_be_disabled_without_disabling_collision(self):
        core = self.make_core(enable_proximity_hold=False)
        active = self.advance_to_active(core)
        now = core._takeoff_confirm_started_at + core.config.takeoff_confirm_duration + 0.1

        allowed = core.update(
            self.snapshot(
                now,
                position=active.target,
                safety_distance=0.08,
                landed=False,
            )
        )
        collision = core.update(
            self.snapshot(
                now + 0.1,
                position=active.target,
                safety_distance=0.05,
                landed=False,
            )
        )

        self.assertEqual(allowed.action, FlightAction.ACTIVE_CONTROL)
        self.assertEqual(collision.reason, "COLLISION")
        self.assertEqual(collision.state, LifecycleState.FAULT_LAND)

    def test_all_map_distance_guards_can_be_disabled(self):
        core = self.make_core(
            enable_proximity_hold=False,
            enable_collision_detection=False,
        )
        active = self.advance_to_active(core)
        now = core._takeoff_confirm_started_at + core.config.takeoff_confirm_duration + 0.1

        allowed = core.update(
            self.snapshot(
                now,
                position=active.target,
                safety_distance=float("nan"),
                landed=False,
            )
        )

        self.assertEqual(allowed.action, FlightAction.ACTIVE_CONTROL)

    def test_active_fault_hold_uses_takeoff_altitude_not_live_height(self):
        core = self.make_core(fault_response="hold")
        active = self.advance_to_active(core)
        target_z = active.target[2]
        now = core._takeoff_confirm_started_at + core.config.takeoff_confirm_duration + 0.1

        fault = core.update(
            self.snapshot(
                now,
                position=(-1.2, 0.3, target_z + 0.22),
                external_fault="GEOFENCE_X",
                landed=False,
            )
        )

        self.assertEqual(fault.state, LifecycleState.FAULT_HOLD)
        self.assertEqual(fault.target, (-1.2, 0.3, target_z))

    def test_persistent_input_timeout_requests_auto_land(self):
        core = self.make_core()
        self.start_takeoff(core)

        core.update(
            self.snapshot(1.0, rc_fresh=False, landed=False)
        )
        fault = core.update(
            self.snapshot(2.1, rc_fresh=False, landed=False)
        )

        self.assertEqual(fault.state, LifecycleState.FAULT_LAND)
        self.assertEqual(fault.action, FlightAction.REQUEST_MODE)
        self.assertEqual(fault.request_mode, "AUTO.LAND")
        self.assertEqual(fault.reason, "RC_TIMEOUT")

    def test_auto_land_confirmation_terminates_the_session(self):
        core = self.make_core()
        self.start_takeoff(core)

        fault = core.update(
            self.snapshot(
                16.0,
                position=(-1.8, 0.0, 0.4),
                landed=False,
            )
        )
        self.assertEqual(fault.state, LifecycleState.FAULT_LAND)
        self.assertEqual(fault.request_mode, "AUTO.LAND")
        self.assertEqual(fault.reason, "TAKEOFF_TIMEOUT")

        landed_mode = core.update(
            self.snapshot(
                16.2,
                mode="AUTO.LAND",
                position=(-1.8, 0.0, 0.4),
                landed=False,
            )
        )
        self.assertEqual(landed_mode.state, LifecycleState.TERMINATED)
        self.assertEqual(landed_mode.reason, "AUTO_LAND_CONFIRMED")
        self.assertEqual(landed_mode.action, FlightAction.STOP_STREAM)

    def test_auto_land_failure_falls_back_to_last_position_hold(self):
        core = self.make_core()
        self.start_takeoff(core)

        first = core.update(
            self.snapshot(
                16.0,
                position=(-1.6, 0.2, 0.5),
                landed=False,
            )
        )
        self.assertEqual(first.request_mode, "AUTO.LAND")

        retry = core.update(
            self.snapshot(
                16.5,
                position=(-1.5, 0.3, 0.6),
                landed=False,
            )
        )
        self.assertEqual(retry.request_mode, "AUTO.LAND")

        fallback = core.update(
            self.snapshot(
                18.1,
                position=(-1.4, 0.4, 0.7),
                landed=False,
            )
        )
        self.assertEqual(fallback.state, LifecycleState.FAULT_HOLD)
        self.assertEqual(fallback.action, FlightAction.FAULT_HOLD)
        self.assertEqual(fallback.target, (-1.6, 0.2, 0.5))
        self.assertEqual(fallback.reason, "AUTO_LAND_UNCONFIRMED")

    def test_hold_fault_response_skips_mode_request(self):
        core = self.make_core(fault_response="hold")
        self.start_takeoff(core)

        fault = core.update(
            self.snapshot(
                1.0,
                position=(-1.6, 0.2, 1.7),
                landed=False,
            )
        )
        self.assertEqual(fault.state, LifecycleState.FAULT_HOLD)
        self.assertEqual(fault.action, FlightAction.FAULT_HOLD)
        self.assertIsNone(fault.request_mode)
        self.assertEqual(fault.reason, "TAKEOFF_OVERHEIGHT")

    def test_external_geofence_fault_requests_auto_land(self):
        core = self.make_core()
        self.start_takeoff(core)

        fault = core.update(
            self.snapshot(
                1.0,
                position=(3.1, 0.0, 0.8),
                landed=False,
                external_fault="GEOFENCE_X",
            )
        )

        self.assertEqual(fault.state, LifecycleState.FAULT_LAND)
        self.assertEqual(fault.action, FlightAction.REQUEST_MODE)
        self.assertEqual(fault.reason, "GEOFENCE_X")

    def test_floor_distance_is_ignored_until_takeoff_is_airborne(self):
        core = self.make_core()
        started = core.update(
            self.snapshot(0.0, safety_distance=0.05, landed=True)
        )
        self.assertEqual(started.state, LifecycleState.TAKEOFF)
        self.assertEqual(started.action, FlightAction.TAKEOFF_HOLD)

        airborne_fault = core.update(
            self.snapshot(
                1.0,
                position=(-1.8, 0.0, 0.41),
                safety_distance=0.05,
                landed=False,
            )
        )
        self.assertEqual(airborne_fault.state, LifecycleState.FAULT_LAND)
        self.assertEqual(airborne_fault.reason, "COLLISION")


if __name__ == "__main__":
    unittest.main()
