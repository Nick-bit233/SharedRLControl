#!/usr/bin/env python3
"""Restore PX4 ATTITUDE and LOCAL_POSITION_NED streams after FCU connection."""

import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import MessageInterval
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from srlc_real_deployment.mavlink_stream_guard_core import (
    GuardAction,
    GuardState,
    MavlinkStreamGuardCore,
    apply_stream_requests,
    build_stream_requests,
)


class MavlinkStreamGuardNode:
    def __init__(self):
        rospy.init_node("mavlink_stream_guard", anonymous=False)

        self.state_topic = rospy.get_param("~state_topic", "/mavros/state")
        self.local_pose_topic = rospy.get_param(
            "~local_pose_topic", "/mavros/local_position/pose"
        )
        self.imu_topic = rospy.get_param("~imu_topic", "/mavros/imu/data")
        self.service_name = rospy.get_param(
            "~message_interval_service", "/mavros/set_message_interval"
        )
        self.check_period_sec = float(rospy.get_param("~check_period_sec", 0.2))
        self.service_wait_timeout_sec = float(
            rospy.get_param("~service_wait_timeout_sec", 0.05)
        )

        self.stream_requests = build_stream_requests(
            local_position_rate_hz=float(
                rospy.get_param("~local_position_rate_hz", 30.0)
            ),
            attitude_rate_hz=float(rospy.get_param("~attitude_rate_hz", 20.0)),
        )
        self.core = MavlinkStreamGuardCore(
            verify_timeout_sec=float(rospy.get_param("~verify_timeout_sec", 3.0)),
            stale_timeout_sec=float(rospy.get_param("~stale_timeout_sec", 2.0)),
            retry_interval_sec=float(rospy.get_param("~retry_interval_sec", 2.0)),
            max_attempts=int(rospy.get_param("~max_attempts", 3)),
        )

        self._lock = threading.RLock()
        self._last_published_state = None
        self._set_interval = rospy.ServiceProxy(self.service_name, MessageInterval)
        self._status_pub = rospy.Publisher(
            "/srlc/mavlink_stream_guard/status", String, queue_size=1, latch=True
        )
        self._state_sub = rospy.Subscriber(
            self.state_topic, State, self._state_cb, queue_size=1
        )
        self._pose_sub = rospy.Subscriber(
            self.local_pose_topic, PoseStamped, self._pose_cb, queue_size=1
        )
        self._imu_sub = rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=1)
        self._timer = rospy.Timer(rospy.Duration(self.check_period_sec), self._timer_cb)

        self._publish_status_if_changed()
        rospy.loginfo(
            "[MAVLinkStreamGuard] local_position=%.1fHz attitude=%.1fHz "
            "verify=%.1fs stale=%.1fs retries=%d",
            self.stream_requests[0].rate_hz,
            self.stream_requests[1].rate_hz,
            self.core.verify_timeout_sec,
            self.core.stale_timeout_sec,
            self.core.max_attempts,
        )

    def _state_cb(self, message):
        with self._lock:
            self.core.on_connection(message.connected, time.monotonic())
            self._publish_status_if_changed()

    def _pose_cb(self, _message):
        with self._lock:
            self.core.on_local_pose(time.monotonic())

    def _imu_cb(self, _message):
        with self._lock:
            self.core.on_imu(time.monotonic())

    def _timer_cb(self, _event):
        with self._lock:
            now = time.monotonic()
            service_available = self._service_available()
            action = self.core.tick(now, service_available)
            self._publish_status_if_changed()

            if action == GuardAction.REQUEST_STREAMS:
                success = apply_stream_requests(
                    self.stream_requests, self._send_stream_request
                )
                self.core.on_request_result(time.monotonic(), success)
                self._publish_status_if_changed()

    def _service_available(self):
        try:
            rospy.wait_for_service(
                self.service_name, timeout=self.service_wait_timeout_sec
            )
            return True
        except rospy.ROSException:
            return False

    def _send_stream_request(self, request):
        try:
            response = self._set_interval(
                message_id=request.message_id,
                message_rate=request.rate_hz,
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn(
                "[MAVLinkStreamGuard] message %d at %.1fHz request failed: %s",
                request.message_id,
                request.rate_hz,
                exc,
            )
            return False

        if not response.success:
            rospy.logwarn(
                "[MAVLinkStreamGuard] FCU rejected message %d at %.1fHz",
                request.message_id,
                request.rate_hz,
            )
            return False

        rospy.loginfo(
            "[MAVLinkStreamGuard] requested message %d at %.1fHz",
            request.message_id,
            request.rate_hz,
        )
        return True

    def _publish_status_if_changed(self):
        state = self.core.state
        if state == self._last_published_state:
            return

        self._status_pub.publish(String(data=state.value))
        if state == GuardState.FAILED:
            rospy.logerr(
                "[MAVLinkStreamGuard] recovery failed after %d attempts; "
                "waiting for FCU reconnect or node restart",
                self.core.attempts,
            )
        elif state == GuardState.HEALTHY:
            rospy.loginfo("[MAVLinkStreamGuard] critical MAVLink streams are healthy")
        else:
            rospy.loginfo("[MAVLinkStreamGuard] state=%s", state.value)
        self._last_published_state = state


def main():
    MavlinkStreamGuardNode()
    rospy.spin()


if __name__ == "__main__":
    main()
