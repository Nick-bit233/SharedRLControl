#!/usr/bin/env python3

import threading
import time
import unittest

import rospy
import rostest
from std_msgs.msg import String


class FakeRuntimeStreamGuardRosTest(unittest.TestCase):
    def setUp(self):
        self._lock = threading.Lock()
        self._status = None
        self._subscriber = rospy.Subscriber(
            "/srlc/mavlink_stream_guard/status", String, self._status_cb, queue_size=1
        )

    def _status_cb(self, message):
        with self._lock:
            self._status = message.data

    def test_fake_runtime_drives_guard_to_healthy(self):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                if self._status == "HEALTHY":
                    return
            rospy.sleep(0.05)

        with self._lock:
            status = self._status
        self.fail("guard did not become HEALTHY; last status={!r}".format(status))


if __name__ == "__main__":
    rospy.init_node("fake_runtime_stream_guard_ros_test")
    rostest.rosrun(
        "srlc_real",
        "fake_runtime_stream_guard_ros",
        FakeRuntimeStreamGuardRosTest,
    )
