#!/usr/bin/env python3
"""Replay a deterministic map-frame S intent through a private RCIn topic.

The node copies all auxiliary channels from the physical/fake MAVROS RC input
and replaces only forward, lateral, and vertical stick channels.  This keeps
the PX4 receiver topic untouched and prevents two publishers from competing on
``/mavros/rc/in``.
"""

import math
import os
import sys
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import RCIn
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.predefined_rc_replay import (  # noqa: E402
    RcAxisEncoding,
    SCurveIntent,
    local_position_to_map,
    map_velocity_to_body,
    wrapped_angle_error,
)


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class PredefinedRcReplayNode:
    def __init__(self):
        rospy.init_node("predefined_rc_replay_node", anonymous=False)

        self.input_rc_topic = rospy.get_param("~input_rc_topic", "/mavros/rc/in")
        self.output_rc_topic = rospy.get_param(
            "~output_rc_topic", "/srlc/predefined_rc/in"
        )
        self.odom_topic = rospy.get_param(
            "~odom_topic", "/nokov/local_position/odom"
        )
        self.lifecycle_topic = rospy.get_param(
            "~lifecycle_topic", "/tunnel_nav/lifecycle_state"
        )
        self.status_topic = rospy.get_param(
            "~status_topic", "/srlc/predefined_rc/status"
        )
        self.complete_topic = rospy.get_param(
            "~complete_topic", "/srlc/predefined_rc/complete"
        )
        self.path_topic = rospy.get_param(
            "~path_topic", "/srlc/predefined_rc/intent_path"
        )
        self.map_frame = str(rospy.get_param("~map_frame", "map"))
        if self.input_rc_topic == self.output_rc_topic:
            raise ValueError("input_rc_topic and output_rc_topic must differ")

        start_xy = rospy.get_param("~start_xy", [-1.85, 2.55])
        goal_xy = rospy.get_param("~goal_xy", [2.75, -1.50])
        lateral_profile = tuple(
            (float(point[0]), float(point[1]))
            for point in rospy.get_param("~lateral_profile", [])
        )
        self.intent = SCurveIntent(
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            lateral_amplitude=float(rospy.get_param("~lateral_amplitude", 0.65)),
            duration=float(rospy.get_param("~duration", 24.0)),
            lateral_profile=lateral_profile,
            arc_length_timing=_param_bool(
                rospy.get_param("~arc_length_timing", False)
            ),
            arc_length_samples=int(rospy.get_param("~arc_length_samples", 2001)),
        )
        self.path_z = float(rospy.get_param("~path_z", 1.1))
        self.path_samples = max(2, int(rospy.get_param("~path_samples", 201)))
        self.start_delay = max(0.0, float(rospy.get_param("~start_delay", 1.0)))
        self.start_tolerance = float(rospy.get_param("~start_tolerance", 0.35))
        self.start_yaw = math.radians(
            float(rospy.get_param("~start_yaw_deg", -45.0))
        )
        self.start_yaw_tolerance = math.radians(
            float(rospy.get_param("~start_yaw_tolerance_deg", 5.0))
        )
        self.activation_value = str(rospy.get_param("~activation_value", "ACTIVE"))
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.3))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.3))
        self.lifecycle_timeout = float(rospy.get_param("~lifecycle_timeout", 0.5))
        self.command_speed_limit = float(
            rospy.get_param("~command_speed_limit", 0.5)
        )

        self.map_yaw = math.radians(float(rospy.get_param("~map_yaw_deg", 0.0)))
        self.map_origin_xyz = tuple(
            float(value)
            for value in rospy.get_param("~map_origin_xyz", [0.0, 0.0, 0.0])
        )
        if len(self.map_origin_xyz) != 3:
            raise ValueError("map_origin_xyz must contain three values")

        self.encoding = RcAxisEncoding(
            channel_base=int(rospy.get_param("~channel_base", 1)),
            forward_channel=int(rospy.get_param("~forward_channel", 2)),
            lateral_channel=int(rospy.get_param("~lateral_channel", 1)),
            vertical_channel=int(rospy.get_param("~vertical_channel", 3)),
            pwm_min=float(rospy.get_param("~pwm_min", 1000.0)),
            pwm_mid=float(rospy.get_param("~pwm_mid", 1500.0)),
            pwm_max=float(rospy.get_param("~pwm_max", 2000.0)),
            max_forward_speed=float(rospy.get_param("~max_forward_speed", 1.0)),
            max_lateral_speed=float(rospy.get_param("~max_lateral_speed", 1.0)),
            max_vertical_speed=float(rospy.get_param("~max_vertical_speed", 0.4)),
            forward_reverse=_param_bool(rospy.get_param("~forward_reverse", False)),
            lateral_reverse=_param_bool(rospy.get_param("~lateral_reverse", False)),
            vertical_reverse=_param_bool(rospy.get_param("~vertical_reverse", False)),
        )

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if min(self.input_timeout, self.odom_timeout, self.lifecycle_timeout) <= 0.0:
            raise ValueError("input freshness timeouts must be positive")
        if self.start_tolerance <= 0.0:
            raise ValueError("start_tolerance must be positive")
        if not (0.0 < self.start_yaw_tolerance <= math.pi):
            raise ValueError("start_yaw_tolerance_deg must be in (0, 180]")
        maximum_speed = self.intent.sampled_max_speed(count=4001)
        if maximum_speed > self.command_speed_limit + 1e-6:
            raise ValueError(
                "S-curve peak speed %.3f exceeds command_speed_limit %.3f"
                % (maximum_speed, self.command_speed_limit)
            )

        self._lock = threading.RLock()
        self._raw_rc = None
        self._raw_rc_time = None
        self._position_local = None
        self._yaw_local = None
        self._odom_time = None
        self._lifecycle = ""
        self._lifecycle_time = None
        self._active_since = None
        self._started_at = None
        self._complete = False
        self._abort_reason = ""

        self.rc_pub = rospy.Publisher(self.output_rc_topic, RCIn, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=2)
        self.complete_pub = rospy.Publisher(
            self.complete_topic, Bool, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)

        self.rc_sub = rospy.Subscriber(
            self.input_rc_topic, RCIn, self._raw_rc_cb, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=1
        )
        self.lifecycle_sub = rospy.Subscriber(
            self.lifecycle_topic, String, self._lifecycle_cb, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb
        )

        self._publish_complete(False)
        self._publish_intent_path()
        lateral_min, lateral_max = self.intent.lateral_bounds
        rospy.logwarn(
            "[PredefinedRC] ARMED FOR REPLAY: %s -> %s duration=%.1fs "
            "lateral=[%+.2f,%+.2f]m arc=%.2fm peak=%.3fm/s "
            "start_yaw=%.1f+/-%.1fdeg; waiting for lifecycle=%s",
            self.intent.start_xy,
            self.intent.goal_xy,
            self.intent.duration,
            lateral_min,
            lateral_max,
            self.intent.arc_length,
            maximum_speed,
            math.degrees(self.start_yaw),
            math.degrees(self.start_yaw_tolerance),
            self.activation_value,
        )

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _raw_rc_cb(self, msg):
        with self._lock:
            self._raw_rc = msg
            self._raw_rc_time = rospy.Time.now()

    def _odom_cb(self, msg):
        position = msg.pose.pose.position
        with self._lock:
            self._position_local = (
                float(position.x),
                float(position.y),
                float(position.z),
            )
            self._yaw_local = self._yaw_from_odom(msg)
            self._odom_time = rospy.Time.now()

    def _lifecycle_cb(self, msg):
        with self._lock:
            self._lifecycle = str(msg.data)
            self._lifecycle_time = rospy.Time.now()

    @staticmethod
    def _fresh(now, stamp, timeout):
        return stamp is not None and (now - stamp).to_sec() <= timeout

    def _abort(self, reason):
        if self._abort_reason:
            return
        self._abort_reason = str(reason)
        rospy.logerr("[PredefinedRC] Replay aborted: %s", self._abort_reason)

    def _publish_rc(self, raw_rc, body_velocity):
        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = raw_rc.header.frame_id
        msg.rssi = raw_rc.rssi
        msg.channels = self.encoding.encode_motion(raw_rc.channels, body_velocity)
        try:
            self.rc_pub.publish(msg)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def _publish_status(self, text):
        try:
            self.status_pub.publish(String(data=str(text)))
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def _publish_complete(self, complete):
        try:
            self.complete_pub.publish(Bool(data=bool(complete)))
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def _publish_intent_path(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.map_frame
        for index in range(self.path_samples):
            progress = index / float(self.path_samples - 1)
            x_pos, y_pos = self.intent.position_at_progress(progress)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x_pos
            pose.pose.position.y = y_pos
            pose.pose.position.z = self.path_z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _timer_cb(self, _event):
        if rospy.is_shutdown():
            return

        now = rospy.Time.now()
        with self._lock:
            raw_rc = self._raw_rc
            raw_rc_time = self._raw_rc_time
            position_local = self._position_local
            yaw_local = self._yaw_local
            odom_time = self._odom_time
            lifecycle = self._lifecycle
            lifecycle_time = self._lifecycle_time

            raw_fresh = self._fresh(now, raw_rc_time, self.input_timeout)
            odom_fresh = self._fresh(now, odom_time, self.odom_timeout)
            lifecycle_fresh = self._fresh(
                now, lifecycle_time, self.lifecycle_timeout
            )

            if raw_rc is None or not raw_fresh:
                if self._started_at is not None and not self._complete:
                    self._abort("RAW_RC_TIMEOUT")
                else:
                    self._active_since = None
                self._publish_status("WAIT_RAW_RC")
                return

            neutral = (0.0, 0.0, 0.0)
            if self._abort_reason:
                self._publish_rc(raw_rc, neutral)
                self._publish_status("ABORTED reason=%s" % self._abort_reason)
                return
            if self._complete:
                self._publish_rc(raw_rc, neutral)
                self._publish_status("COMPLETE")
                return

            if not odom_fresh or position_local is None or yaw_local is None:
                if self._started_at is not None:
                    self._abort("ODOM_TIMEOUT")
                else:
                    self._active_since = None
                self._publish_rc(raw_rc, neutral)
                self._publish_status("WAIT_ODOM")
                return
            if not lifecycle_fresh:
                if self._started_at is not None:
                    self._abort("LIFECYCLE_TIMEOUT")
                else:
                    self._active_since = None
                self._publish_rc(raw_rc, neutral)
                self._publish_status("WAIT_LIFECYCLE")
                return

            position_map = local_position_to_map(
                position_local,
                map_yaw=self.map_yaw,
                map_origin_xyz=self.map_origin_xyz,
            )
            start_error = math.hypot(
                position_map[0] - self.intent.start_xy[0],
                position_map[1] - self.intent.start_xy[1],
            )
            yaw_map = yaw_local + self.map_yaw
            start_yaw_error = abs(wrapped_angle_error(yaw_map, self.start_yaw))

            if self._started_at is None:
                if lifecycle != self.activation_value:
                    self._active_since = None
                    self._publish_rc(raw_rc, neutral)
                    self._publish_status(
                        "WAIT_ACTIVE lifecycle=%s" % (lifecycle or "UNKNOWN")
                    )
                    return
                if start_error > self.start_tolerance:
                    self._active_since = None
                    self._publish_rc(raw_rc, neutral)
                    self._publish_status(
                        "WAIT_START error=%.3f tolerance=%.3f map_xy=(%.3f,%.3f)"
                        % (
                            start_error,
                            self.start_tolerance,
                            position_map[0],
                            position_map[1],
                        )
                    )
                    return
                if start_yaw_error > self.start_yaw_tolerance:
                    self._active_since = None
                    self._publish_rc(raw_rc, neutral)
                    self._publish_status(
                        "WAIT_START_YAW error_deg=%.2f tolerance_deg=%.2f "
                        "map_yaw_deg=%.2f target_deg=%.2f"
                        % (
                            math.degrees(start_yaw_error),
                            math.degrees(self.start_yaw_tolerance),
                            math.degrees(yaw_map),
                            math.degrees(self.start_yaw),
                        )
                    )
                    return
                if self._active_since is None:
                    self._active_since = now
                delay_elapsed = (now - self._active_since).to_sec()
                if delay_elapsed < self.start_delay:
                    self._publish_rc(raw_rc, neutral)
                    self._publish_status(
                        "START_DELAY remaining=%.2f start_error=%.3f "
                        "yaw_error_deg=%.2f"
                        % (
                            self.start_delay - delay_elapsed,
                            start_error,
                            math.degrees(start_yaw_error),
                        )
                    )
                    return
                self._started_at = now
                rospy.logwarn(
                    "[PredefinedRC] Replay started at map_xy=(%.3f, %.3f), "
                    "start_error=%.3fm map_yaw=%.2fdeg yaw_error=%.2fdeg",
                    position_map[0],
                    position_map[1],
                    start_error,
                    math.degrees(yaw_map),
                    math.degrees(start_yaw_error),
                )

            if lifecycle != self.activation_value:
                self._abort("LIFECYCLE_LEFT_%s" % self.activation_value)
                self._publish_rc(raw_rc, neutral)
                self._publish_status("ABORTED reason=%s" % self._abort_reason)
                return

            elapsed = (now - self._started_at).to_sec()
            sample = self.intent.sample(elapsed)
            if sample.complete:
                self._complete = True
                self._publish_complete(True)
                self._publish_rc(raw_rc, neutral)
                self._publish_status("COMPLETE")
                rospy.logwarn(
                    "[PredefinedRC] Replay complete; motion channels are neutral"
                )
                return

            body_velocity = map_velocity_to_body(
                (sample.velocity_xy[0], sample.velocity_xy[1], 0.0),
                yaw_local=yaw_local,
                map_yaw=self.map_yaw,
            )
            self._publish_rc(raw_rc, body_velocity)
            self._publish_status(
                "RUNNING t=%.2f/%.2f u=%.3f intent_xy=(%.3f,%.3f) "
                "map_v=(%.3f,%.3f) body_v=(%.3f,%.3f)"
                % (
                    elapsed,
                    self.intent.duration,
                    sample.progress,
                    sample.position_xy[0],
                    sample.position_xy[1],
                    sample.velocity_xy[0],
                    sample.velocity_xy[1],
                    body_velocity[0],
                    body_velocity[1],
                )
            )


if __name__ == "__main__":
    try:
        PredefinedRcReplayNode()
        rospy.spin()
    except (rospy.ROSInterruptException, rospy.ROSException):
        pass
