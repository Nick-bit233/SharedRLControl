#!/usr/bin/env python3

import threading
import time
import unittest

import rospy
import rostest
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import MessageInterval, MessageIntervalResponse
from sensor_msgs.msg import Imu
from std_msgs.msg import String


class MavlinkStreamGuardRosTest(unittest.TestCase):
    def setUp(self):
        self._lock = threading.Lock()
        self._requests = []
        self._status = None
        self._state_pub = rospy.Publisher("/mavros/state", State, queue_size=1, latch=True)
        self._pose_pub = rospy.Publisher("/mavros/local_position/pose", PoseStamped, queue_size=5)
        self._imu_pub = rospy.Publisher("/mavros/imu/data", Imu, queue_size=5)
        self._status_sub = rospy.Subscriber(
            "/srlc/mavlink_stream_guard/status", String, self._status_cb, queue_size=1
        )
        self._service = rospy.Service(
            "/mavros/set_message_interval", MessageInterval, self._message_interval_cb
        )

    def _message_interval_cb(self, request):
        with self._lock:
            self._requests.append((request.message_id, request.message_rate))
        return MessageIntervalResponse(success=True)

    def _status_cb(self, message):
        with self._lock:
            self._status = message.data

    def test_connected_guard_requests_streams_and_becomes_healthy(self):
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            stamp = rospy.Time.now()

            state = State()
            state.header.stamp = stamp
            state.connected = True
            self._state_pub.publish(state)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.pose.orientation.w = 1.0
            self._pose_pub.publish(pose)

            imu = Imu()
            imu.header.stamp = stamp
            imu.orientation.w = 1.0
            self._imu_pub.publish(imu)

            with self._lock:
                if self._status == "HEALTHY":
                    break
            rospy.sleep(0.05)

        with self._lock:
            status = self._status
            requests = list(self._requests)

        self.assertEqual(status, "HEALTHY")
        self.assertGreaterEqual(len(requests), 2)
        self.assertEqual(requests[0][0], 32)
        self.assertAlmostEqual(requests[0][1], 30.0)
        self.assertEqual(requests[1][0], 30)
        self.assertAlmostEqual(requests[1][1], 20.0)


if __name__ == "__main__":
    rospy.init_node("mavlink_stream_guard_ros_test")
    rostest.rosrun("srlc_real", "mavlink_stream_guard_ros", MavlinkStreamGuardRosTest)
