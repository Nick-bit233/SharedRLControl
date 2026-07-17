#!/usr/bin/env python3

import threading
import time
import unittest

import rospy
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, SetModeRequest
from std_msgs.msg import Bool, String


class OneShotFlightRosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rospy.init_node("one_shot_flight_ros_test", anonymous=True)
        cls._lock = threading.Lock()
        cls.lifecycle = ""
        cls.effective_mode = ""
        cls.policy_active = False
        cls.session_consumed = False
        cls.setpoint_count = 0
        cls.target_z_values = []
        cls.target_vz_values = []

        rospy.Subscriber(
            "/tunnel_nav/lifecycle_state",
            String,
            cls._lifecycle_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/tunnel_nav/effective_mode",
            String,
            cls._effective_mode_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/tunnel_nav/policy_active",
            Bool,
            cls._policy_active_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/tunnel_nav/session_consumed",
            Bool,
            cls._session_consumed_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/mavros/setpoint_raw/local",
            PositionTarget,
            cls._setpoint_cb,
            queue_size=100,
        )
        rospy.wait_for_service("/mavros/set_mode", timeout=15.0)
        cls.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    @classmethod
    def _lifecycle_cb(cls, msg):
        with cls._lock:
            cls.lifecycle = str(msg.data)

    @classmethod
    def _effective_mode_cb(cls, msg):
        with cls._lock:
            cls.effective_mode = str(msg.data)

    @classmethod
    def _policy_active_cb(cls, msg):
        with cls._lock:
            cls.policy_active = bool(msg.data)

    @classmethod
    def _session_consumed_cb(cls, msg):
        with cls._lock:
            cls.session_consumed = bool(msg.data)

    @classmethod
    def _setpoint_cb(cls, msg):
        with cls._lock:
            cls.setpoint_count += 1
            if (
                cls.session_consumed
                and not (msg.type_mask & PositionTarget.IGNORE_PZ)
            ):
                cls.target_z_values.append(float(msg.position.z))
                cls.target_vz_values.append(
                    0.0
                    if msg.type_mask & PositionTarget.IGNORE_VZ
                    else float(msg.velocity.z)
                )

    @classmethod
    def wait_for(cls, predicate, timeout, description):
        deadline = time.time() + timeout
        while time.time() < deadline and not rospy.is_shutdown():
            with cls._lock:
                if predicate():
                    return
            rospy.sleep(0.05)
        with cls._lock:
            state = {
                "lifecycle": cls.lifecycle,
                "effective_mode": cls.effective_mode,
                "policy_active": cls.policy_active,
                "session_consumed": cls.session_consumed,
                "setpoint_count": cls.setpoint_count,
            }
        raise AssertionError(f"Timed out waiting for {description}: {state}")

    def test_one_shot_takeoff_assist_and_terminal_offboard_loss(self):
        self.wait_for(
            lambda: self.lifecycle == "ACTIVE",
            timeout=30.0,
            description="ACTIVE lifecycle",
        )
        self.wait_for(
            lambda: self.policy_active,
            timeout=5.0,
            description="ASSIST policy output",
        )
        with self._lock:
            self.assertTrue(self.session_consumed)
            self.assertIn(self.effective_mode, {"ASSIST", "ASSIST_IDLE"})
            active_targets = list(self.target_z_values)
        self.assertTrue(active_targets)
        self.assertGreater(max(active_targets) - min(active_targets), 0.5)
        self.assertTrue(
            all(
                later + 1e-5 >= earlier
                for earlier, later in zip(active_targets, active_targets[1:])
            )
        )
        with self._lock:
            active_target_vz = list(self.target_vz_values)
        self.assertGreater(max(active_target_vz), 0.0)
        self.assertLessEqual(max(active_target_vz), 0.4 + 1e-5)

        response = self.set_mode(SetModeRequest(custom_mode="POSCTL"))
        self.assertTrue(response.mode_sent)
        self.wait_for(
            lambda: self.lifecycle == "TERMINATED",
            timeout=5.0,
            description="TERMINATED lifecycle",
        )

        rospy.sleep(0.3)
        with self._lock:
            stopped_count = self.setpoint_count
        response = self.set_mode(SetModeRequest(custom_mode="OFFBOARD"))
        self.assertTrue(response.mode_sent)
        rospy.sleep(1.0)
        with self._lock:
            self.assertEqual(self.lifecycle, "TERMINATED")
            self.assertEqual(self.setpoint_count, stopped_count)


if __name__ == "__main__":
    import rostest

    rostest.rosrun("srlc_real", "one_shot_flight_ros", OneShotFlightRosTest)
