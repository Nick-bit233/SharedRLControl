#!/usr/bin/env python3
"""Replay recorded motion-stick channels through a private MAVROS RCIn topic.

The physical receiver remains the authority for PX4 and for every auxiliary or
switch channel.  Only the forward, lateral, and vertical channels consumed by
``rc_input_node`` are overlaid from the selected recording.
"""

import math
import os
import sys
import threading

import rospy
from mavros_msgs.msg import RCIn
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.recorded_rc_replay import (  # noqa: E402
    RcChannelOverlay,
    load_recorded_rc_timeline,
    wrapped_angle_error,
)


class RecordedRcReplayNode:
    def __init__(self):
        rospy.init_node("recorded_rc_replay_node", anonymous=False)

        self.input_rc_topic = str(
            rospy.get_param("~input_rc_topic", "/mavros/rc/in")
        )
        self.output_rc_topic = str(
            rospy.get_param("~output_rc_topic", "/srlc/recorded_rc/in")
        )
        self.odom_topic = str(
            rospy.get_param("~odom_topic", "/nokov/local_position/odom")
        )
        self.lifecycle_topic = str(
            rospy.get_param("~lifecycle_topic", "/tunnel_nav/lifecycle_state")
        )
        self.status_topic = str(
            rospy.get_param("~status_topic", "/srlc/recorded_rc/status")
        )
        self.complete_topic = str(
            rospy.get_param("~complete_topic", "/srlc/recorded_rc/complete")
        )
        if self.input_rc_topic == self.output_rc_topic:
            raise ValueError("input_rc_topic and output_rc_topic must differ")

        self.activation_value = str(
            rospy.get_param("~activation_value", "ACTIVE")
        )
        self.start_delay = max(
            0.0, float(rospy.get_param("~start_delay", 1.0))
        )
        self.start_xy_tolerance = float(
            rospy.get_param("~start_xy_tolerance", 0.35)
        )
        self.start_yaw_tolerance = math.radians(
            float(rospy.get_param("~start_yaw_tolerance_deg", 5.0))
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 50.0))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.3))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.3))
        self.lifecycle_timeout = float(
            rospy.get_param("~lifecycle_timeout", 0.5)
        )

        self.overlay = RcChannelOverlay(
            channel_base=int(rospy.get_param("~channel_base", 1)),
            forward_channel=int(rospy.get_param("~forward_channel", 2)),
            lateral_channel=int(rospy.get_param("~lateral_channel", 1)),
            vertical_channel=int(rospy.get_param("~vertical_channel", 3)),
            pwm_mid=int(round(float(rospy.get_param("~pwm_mid", 1500.0)))),
        )
        replay_start_time = float(
            rospy.get_param("~replay_start_time", -1.0)
        )
        replay_end_time = float(rospy.get_param("~replay_end_time", -1.0))
        self.timeline = load_recorded_rc_timeline(
            str(rospy.get_param("~recording_file", "")),
            activation_value=self.activation_value,
            replay_start_time=(
                replay_start_time if replay_start_time >= 0.0 else None
            ),
            replay_end_time=(
                replay_end_time if replay_end_time >= 0.0 else None
            ),
            motion_indices=self.overlay.motion_indices,
            pwm_lower_bound=int(
                rospy.get_param("~pwm_lower_bound", 800)
            ),
            pwm_upper_bound=int(
                rospy.get_param("~pwm_upper_bound", 2200)
            ),
            max_sample_gap=float(
                rospy.get_param("~max_sample_gap", 0.25)
            ),
            max_replay_duration=float(
                rospy.get_param("~max_replay_duration", 300.0)
            ),
        )

        if self.timeline.reference_position is None:
            raise ValueError(
                "recording replay window has no finite position for its start gate"
            )
        if self.timeline.reference_yaw is None:
            raise ValueError(
                "recording replay window has no finite yaw for its start gate"
            )
        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if min(
            self.input_timeout,
            self.odom_timeout,
            self.lifecycle_timeout,
        ) <= 0.0:
            raise ValueError("input freshness timeouts must be positive")
        if self.start_xy_tolerance <= 0.0:
            raise ValueError("start_xy_tolerance must be positive")
        if not (0.0 < self.start_yaw_tolerance <= math.pi):
            raise ValueError("start_yaw_tolerance_deg must be in (0, 180]")

        self._lock = threading.RLock()
        self._raw_rc = None
        self._raw_rc_time = None
        self._position = None
        self._yaw = None
        self._odom_time = None
        self._lifecycle = ""
        self._lifecycle_time = None
        self._ready_since = None
        self._started_at = None
        self._complete = False
        self._abort_reason = ""

        self.rc_pub = rospy.Publisher(
            self.output_rc_topic, RCIn, queue_size=10
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=2
        )
        self.complete_pub = rospy.Publisher(
            self.complete_topic, Bool, queue_size=1, latch=True
        )
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
        reference = self.timeline.reference_position
        rospy.logwarn(
            "[RecordedRC] ARMED FOR ONE-SHOT REPLAY: file=%s stream=%s "
            "source_t=[%.3f, %.3f] duration=%.3fs samples=%d "
            "start_xy=(%.3f, %.3f) start_yaw=%.2fdeg modes=%s/%s; "
            "waiting for lifecycle=%s",
            self.timeline.source_path,
            self.timeline.source_stream,
            self.timeline.source_start_time,
            self.timeline.source_end_time,
            self.timeline.duration,
            len(self.timeline.samples),
            reference[0],
            reference[1],
            math.degrees(self.timeline.reference_yaw),
            ",".join(self.timeline.control_modes) or "unknown",
            ",".join(self.timeline.effective_modes) or "unknown",
            self.activation_value,
        )

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _fresh(now, stamp, timeout):
        return stamp is not None and (now - stamp).to_sec() <= timeout

    def _raw_rc_cb(self, msg):
        with self._lock:
            self._raw_rc = msg
            self._raw_rc_time = rospy.Time.now()

    def _odom_cb(self, msg):
        position = msg.pose.pose.position
        with self._lock:
            self._position = (
                float(position.x),
                float(position.y),
                float(position.z),
            )
            self._yaw = self._yaw_from_odom(msg)
            self._odom_time = rospy.Time.now()

    def _lifecycle_cb(self, msg):
        with self._lock:
            self._lifecycle = str(msg.data)
            self._lifecycle_time = rospy.Time.now()

    def _abort(self, reason):
        if self._abort_reason:
            return
        self._abort_reason = str(reason)
        rospy.logerr("[RecordedRC] Replay aborted: %s", self._abort_reason)

    def _publish_rc(self, raw_rc, recorded_channels=None):
        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = raw_rc.header.frame_id
        msg.rssi = raw_rc.rssi
        if recorded_channels is None:
            msg.channels = self.overlay.neutral(raw_rc.channels)
        else:
            msg.channels = self.overlay.overlay(
                raw_rc.channels, recorded_channels
            )
        try:
            self.rc_pub.publish(msg)
        except rospy.ROSException as exc:
            if not rospy.is_shutdown() and "closed topic" not in str(exc):
                raise

    def _publish_status(self, text):
        try:
            self.status_pub.publish(String(data=str(text)))
        except rospy.ROSException as exc:
            if not rospy.is_shutdown() and "closed topic" not in str(exc):
                raise

    def _publish_complete(self, complete):
        try:
            self.complete_pub.publish(Bool(data=bool(complete)))
        except rospy.ROSException as exc:
            if not rospy.is_shutdown() and "closed topic" not in str(exc):
                raise

    def _timer_cb(self, _event):
        if rospy.is_shutdown():
            return

        now = rospy.Time.now()
        with self._lock:
            raw_rc = self._raw_rc
            raw_rc_time = self._raw_rc_time
            position = self._position
            yaw = self._yaw
            odom_time = self._odom_time
            lifecycle = self._lifecycle
            lifecycle_time = self._lifecycle_time

            if raw_rc is None or not self._fresh(
                now, raw_rc_time, self.input_timeout
            ):
                if self._started_at is not None and not self._complete:
                    self._abort("RAW_RC_TIMEOUT")
                else:
                    self._ready_since = None
                self._publish_status("WAIT_RAW_RC")
                return
            if not self.overlay.has_motion_channels(raw_rc.channels):
                if self._started_at is not None and not self._complete:
                    self._abort("RAW_RC_CHANNELS_MISSING")
                else:
                    self._ready_since = None
                self._publish_status("WAIT_RAW_RC_CHANNELS")
                return

            if self._abort_reason:
                self._publish_rc(raw_rc)
                self._publish_status(
                    "ABORTED reason=%s" % self._abort_reason
                )
                return
            if self._complete:
                self._publish_rc(raw_rc)
                self._publish_status("COMPLETE")
                return

            if (
                position is None
                or yaw is None
                or not self._fresh(now, odom_time, self.odom_timeout)
            ):
                if self._started_at is not None:
                    self._abort("ODOM_TIMEOUT")
                else:
                    self._ready_since = None
                self._publish_rc(raw_rc)
                self._publish_status("WAIT_ODOM")
                return
            if not self._fresh(
                now, lifecycle_time, self.lifecycle_timeout
            ):
                if self._started_at is not None:
                    self._abort("LIFECYCLE_TIMEOUT")
                else:
                    self._ready_since = None
                self._publish_rc(raw_rc)
                self._publish_status("WAIT_LIFECYCLE")
                return

            reference = self.timeline.reference_position
            start_error = math.hypot(
                position[0] - reference[0],
                position[1] - reference[1],
            )
            yaw_error = abs(
                wrapped_angle_error(yaw, self.timeline.reference_yaw)
            )

            if self._started_at is None:
                if lifecycle != self.activation_value:
                    self._ready_since = None
                    self._publish_rc(raw_rc)
                    self._publish_status(
                        "WAIT_ACTIVE lifecycle=%s"
                        % (lifecycle or "UNKNOWN")
                    )
                    return
                if start_error > self.start_xy_tolerance:
                    self._ready_since = None
                    self._publish_rc(raw_rc)
                    self._publish_status(
                        "WAIT_START error=%.3f tolerance=%.3f "
                        "xy=(%.3f,%.3f) reference=(%.3f,%.3f)"
                        % (
                            start_error,
                            self.start_xy_tolerance,
                            position[0],
                            position[1],
                            reference[0],
                            reference[1],
                        )
                    )
                    return
                if yaw_error > self.start_yaw_tolerance:
                    self._ready_since = None
                    self._publish_rc(raw_rc)
                    self._publish_status(
                        "WAIT_START_YAW error_deg=%.2f tolerance_deg=%.2f "
                        "yaw_deg=%.2f reference_deg=%.2f"
                        % (
                            math.degrees(yaw_error),
                            math.degrees(self.start_yaw_tolerance),
                            math.degrees(yaw),
                            math.degrees(self.timeline.reference_yaw),
                        )
                    )
                    return
                if self._ready_since is None:
                    self._ready_since = now
                delay_elapsed = (now - self._ready_since).to_sec()
                if delay_elapsed < self.start_delay:
                    self._publish_rc(raw_rc)
                    self._publish_status(
                        "START_DELAY remaining=%.2f start_error=%.3f "
                        "yaw_error_deg=%.2f"
                        % (
                            self.start_delay - delay_elapsed,
                            start_error,
                            math.degrees(yaw_error),
                        )
                    )
                    return
                self._started_at = now
                rospy.logwarn(
                    "[RecordedRC] Replay started at xy=(%.3f, %.3f), "
                    "start_error=%.3fm yaw=%.2fdeg yaw_error=%.2fdeg",
                    position[0],
                    position[1],
                    start_error,
                    math.degrees(yaw),
                    math.degrees(yaw_error),
                )

            if lifecycle != self.activation_value:
                self._abort("LIFECYCLE_LEFT_%s" % self.activation_value)
                self._publish_rc(raw_rc)
                self._publish_status(
                    "ABORTED reason=%s" % self._abort_reason
                )
                return

            elapsed = (now - self._started_at).to_sec()
            if elapsed >= self.timeline.duration:
                self._complete = True
                self._publish_complete(True)
                self._publish_rc(raw_rc)
                self._publish_status("COMPLETE")
                rospy.logwarn(
                    "[RecordedRC] Replay complete; motion channels are neutral"
                )
                return

            index, sample = self.timeline.sample_at(elapsed)
            self._publish_rc(raw_rc, sample.channels)
            motion = tuple(
                sample.channels[channel_index]
                for channel_index in self.overlay.motion_indices
            )
            self._publish_status(
                "RUNNING t=%.3f/%.3f sample=%d/%d "
                "source_t=%.3f motion_pwm=%s"
                % (
                    elapsed,
                    self.timeline.duration,
                    index + 1,
                    len(self.timeline.samples),
                    sample.source_time,
                    motion,
                )
            )


if __name__ == "__main__":
    try:
        RecordedRcReplayNode()
        rospy.spin()
    except (rospy.ROSInterruptException, rospy.ROSException):
        pass
