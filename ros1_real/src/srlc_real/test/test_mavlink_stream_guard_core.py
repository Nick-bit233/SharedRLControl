#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.mavlink_stream_guard_core import (  # noqa: E402
    GuardAction,
    GuardState,
    MavlinkStreamGuardCore,
    apply_stream_requests,
    build_stream_requests,
)


class StreamRequestTest(unittest.TestCase):
    def test_builds_px4_attitude_and_local_position_requests(self):
        requests = build_stream_requests(local_position_rate_hz=30.0, attitude_rate_hz=20.0)

        self.assertEqual([(request.message_id, request.rate_hz) for request in requests], [(32, 30.0), (30, 20.0)])

    def test_applies_every_request_and_requires_all_to_succeed(self):
        requests = build_stream_requests(local_position_rate_hz=30.0, attitude_rate_hz=20.0)
        calls = []

        def send(request):
            calls.append((request.message_id, request.rate_hz))
            return request.message_id != 32

        self.assertFalse(apply_stream_requests(requests, send))
        self.assertEqual(calls, [(32, 30.0), (30, 20.0)])

    def test_applies_successful_request_batch(self):
        requests = build_stream_requests(local_position_rate_hz=30.0, attitude_rate_hz=20.0)

        self.assertTrue(apply_stream_requests(requests, lambda _request: True))


class MavlinkStreamGuardCoreTest(unittest.TestCase):
    def make_core(self):
        return MavlinkStreamGuardCore(
            verify_timeout_sec=3.0,
            stale_timeout_sec=2.0,
            retry_interval_sec=2.0,
            max_attempts=3,
        )

    def connect_and_request(self, core, now=0.0):
        core.on_connection(True, now)
        action = core.tick(now, service_available=True)
        self.assertEqual(action, GuardAction.REQUEST_STREAMS)
        return now

    def make_healthy(self, core, request_time=0.0):
        core.on_request_result(request_time, success=True)
        core.on_local_pose(request_time + 0.1)
        core.on_imu(request_time + 0.1)
        self.assertIsNone(core.tick(request_time + 0.2, service_available=True))
        self.assertEqual(core.state, GuardState.HEALTHY)

    def test_disconnected_guard_never_requests_streams(self):
        core = self.make_core()

        self.assertIsNone(core.tick(100.0, service_available=True))
        self.assertEqual(core.state, GuardState.DISCONNECTED)
        self.assertEqual(core.attempts, 0)

    def test_connected_guard_waits_for_service_without_consuming_attempt(self):
        core = self.make_core()
        core.on_connection(True, 0.0)

        self.assertIsNone(core.tick(0.0, service_available=False))
        self.assertIsNone(core.tick(10.0, service_available=False))
        self.assertEqual(core.state, GuardState.WAITING_SERVICE)
        self.assertEqual(core.attempts, 0)

    def test_success_requires_pose_and_imu_received_after_request(self):
        core = self.make_core()
        core.on_local_pose(-1.0)
        core.on_imu(-1.0)
        request_time = self.connect_and_request(core)

        core.on_request_result(request_time, success=True)
        self.assertIsNone(core.tick(0.5, service_available=True))
        self.assertEqual(core.state, GuardState.VERIFYING)

        core.on_local_pose(0.6)
        self.assertIsNone(core.tick(0.7, service_available=True))
        self.assertEqual(core.state, GuardState.VERIFYING)

        core.on_imu(0.8)
        self.assertIsNone(core.tick(0.9, service_available=True))
        self.assertEqual(core.state, GuardState.HEALTHY)

    def test_verification_timeout_retries_three_times_then_fails(self):
        core = self.make_core()
        request_time = self.connect_and_request(core, 0.0)

        for attempt in range(1, 4):
            core.on_request_result(request_time, success=True)
            self.assertIsNone(core.tick(request_time + 3.0, service_available=True))
            if attempt < 3:
                self.assertEqual(core.state, GuardState.REQUESTING)
                request_time += 5.0
                self.assertEqual(core.tick(request_time, service_available=True), GuardAction.REQUEST_STREAMS)

        self.assertEqual(core.state, GuardState.FAILED)
        self.assertEqual(core.attempts, 3)
        self.assertIsNone(core.tick(100.0, service_available=True))

    def test_failed_service_call_obeys_retry_interval(self):
        core = self.make_core()
        self.connect_and_request(core, 0.0)
        core.on_request_result(0.1, success=False)

        self.assertIsNone(core.tick(2.0, service_available=True))
        self.assertEqual(core.tick(2.1, service_available=True), GuardAction.REQUEST_STREAMS)
        self.assertEqual(core.attempts, 2)

    def test_stale_healthy_stream_starts_new_recovery_batch(self):
        core = self.make_core()
        self.connect_and_request(core, 0.0)
        self.make_healthy(core, request_time=0.0)

        self.assertEqual(core.tick(2.2, service_available=True), GuardAction.REQUEST_STREAMS)
        self.assertEqual(core.state, GuardState.REQUESTING)
        self.assertEqual(core.attempts, 1)

    def test_disconnect_reconnect_resets_failed_guard(self):
        core = self.make_core()
        self.connect_and_request(core, 0.0)
        core.on_request_result(0.0, success=False)
        self.assertIsNone(core.tick(1.9, service_available=True))
        self.assertEqual(core.tick(2.0, service_available=True), GuardAction.REQUEST_STREAMS)
        core.on_request_result(2.0, success=False)
        self.assertEqual(core.tick(4.0, service_available=True), GuardAction.REQUEST_STREAMS)
        core.on_request_result(4.0, success=False)
        self.assertEqual(core.state, GuardState.FAILED)

        core.on_connection(False, 5.0)
        self.assertEqual(core.state, GuardState.DISCONNECTED)
        core.on_connection(True, 6.0)

        self.assertEqual(core.tick(6.0, service_available=True), GuardAction.REQUEST_STREAMS)
        self.assertEqual(core.attempts, 1)


if __name__ == "__main__":
    unittest.main()
