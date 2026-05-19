#!/usr/bin/env python3
"""MAVROS RC bridge for real PX4 SRLC deployment."""
import math

import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, String

try:
    from mavros_msgs.msg import RCIn
    HAS_MAVROS = True
except ImportError:
    HAS_MAVROS = False


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class RcInputNode:
    def __init__(self):
        rospy.init_node("rc_input_node", anonymous=False)
        if not HAS_MAVROS:
            rospy.logfatal("[RCInput] mavros_msgs/RCIn is unavailable")
            rospy.signal_shutdown("missing mavros_msgs")
            return

        self.rc_topic = rospy.get_param("~rc_topic", "/mavros/rc/in")
        self.human_action_topic = rospy.get_param("~human_action_topic", "/srlc/human_action")
        self.stop_topic = rospy.get_param("~stop_topic", "/experiment_control/stop")
        self.assist_topic = rospy.get_param("~assist_enable_topic", "/srlc/assist_enable")
        self.status_topic = rospy.get_param("~status_topic", "/srlc/rc_status")

        self.channel_base = int(rospy.get_param("~channel_base", 1))
        self.forward_channel = int(rospy.get_param("~forward_channel", 2))
        self.lateral_channel = int(rospy.get_param("~lateral_channel", 1))
        self.vertical_channel = int(rospy.get_param("~vertical_channel", 3))
        self.estop_channel = int(rospy.get_param("~estop_channel", 7))
        self.assist_channel = int(rospy.get_param("~assist_channel", 9))
        self.reset_channel = int(rospy.get_param("~reset_channel", 0))

        self.pwm_min = float(rospy.get_param("~pwm_min", 1000.0))
        self.pwm_mid = float(rospy.get_param("~pwm_mid", 1500.0))
        self.pwm_max = float(rospy.get_param("~pwm_max", 2000.0))
        self.deadband = float(rospy.get_param("~deadband", 0.05))

        self.max_forward_speed = float(rospy.get_param("~max_forward_speed", 2.0))
        self.max_lateral_speed = float(rospy.get_param("~max_lateral_speed", 2.0))
        self.max_vertical_speed = float(rospy.get_param("~max_vertical_speed", 1.0))
        self.forward_reverse = bool(rospy.get_param("~forward_reverse", False))
        self.lateral_reverse = bool(rospy.get_param("~lateral_reverse", False))
        self.vertical_reverse = bool(rospy.get_param("~vertical_reverse", False))

        self.switch_threshold = float(rospy.get_param("~switch_threshold", 1700.0))
        self.estop_high_is_stop = _param_bool(rospy.get_param("~estop_high_is_stop", True))
        self.assist_high_is_enable = _param_bool(rospy.get_param("~assist_high_is_enable", True))
        self.assist_toggle_mode = _param_bool(rospy.get_param("~assist_toggle_mode", False))
        self.reset_high_is_reset = _param_bool(rospy.get_param("~reset_high_is_reset", True))
        self.latch_stop = _param_bool(rospy.get_param("~latch_stop", True))
        self.enable_stop_output = _param_bool(rospy.get_param("~enable_stop_output", True))
        self.stop_on_timeout = _param_bool(rospy.get_param("~stop_on_timeout", True))
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 0.3))
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))

        self.last_rc = None
        self.last_rc_time = None
        self.stop_latched = bool(self.enable_stop_output and self.latch_stop)
        self.assist_enable = False
        self._last_assist_switch_active = None

        self.rc_sub = rospy.Subscriber(self.rc_topic, RCIn, self._rc_cb, queue_size=1)
        self.human_pub = rospy.Publisher(self.human_action_topic, TwistStamped, queue_size=2)
        self.stop_pub = None
        if self.enable_stop_output:
            self.stop_pub = rospy.Publisher(self.stop_topic, Bool, queue_size=2, latch=True)
        self.assist_pub = rospy.Publisher(self.assist_topic, Bool, queue_size=2, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=2)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._timer_cb)

        if self.stop_pub is not None:
            self.stop_pub.publish(Bool(data=bool(self.stop_latched)))
        self.assist_pub.publish(Bool(data=False))
        rospy.loginfo(
            "[RCInput] Ready: rc=%s human_action=%s assist_ch=%d assist_toggle=%s stop_output=%s",
            self.rc_topic,
            self.human_action_topic,
            self.assist_channel,
            self.assist_toggle_mode,
            self.enable_stop_output,
        )

    def _rc_cb(self, msg):
        self.last_rc = list(msg.channels)
        self.last_rc_time = rospy.Time.now()

    def _idx(self, channel):
        if channel <= 0:
            return None
        return channel - self.channel_base

    def _channel_value(self, channel, default=None):
        idx = self._idx(channel)
        if idx is None or self.last_rc is None or idx < 0 or idx >= len(self.last_rc):
            return default
        return float(self.last_rc[idx])

    def _axis(self, channel, reverse=False):
        pwm = self._channel_value(channel, self.pwm_mid)
        if pwm >= self.pwm_mid:
            denom = max(1.0, self.pwm_max - self.pwm_mid)
        else:
            denom = max(1.0, self.pwm_mid - self.pwm_min)
        value = (pwm - self.pwm_mid) / denom
        value = max(-1.0, min(1.0, value))
        if abs(value) < self.deadband:
            value = 0.0
        if reverse:
            value = -value
        return value

    def _switch_active(self, channel, high_is_active):
        pwm = self._channel_value(channel)
        if pwm is None:
            return False
        active = pwm >= self.switch_threshold
        return active if high_is_active else not active

    def _rc_fresh(self):
        if self.last_rc_time is None:
            return False
        return (rospy.Time.now() - self.last_rc_time).to_sec() <= self.timeout_sec

    def _timer_cb(self, _event):
        rc_fresh = self._rc_fresh()
        if not rc_fresh:
            if self.enable_stop_output and self.stop_on_timeout:
                self.stop_latched = True
            self.assist_enable = False
            self._last_assist_switch_active = None
            self._publish_zero("RC_TIMEOUT")
            return

        estop_active = (
            self.enable_stop_output
            and self._switch_active(self.estop_channel, self.estop_high_is_stop)
        )
        reset_active = self._switch_active(self.reset_channel, self.reset_high_is_reset)
        if self.enable_stop_output:
            if estop_active:
                self.stop_latched = True
            elif self.latch_stop and reset_active:
                self.stop_latched = False
            elif not self.latch_stop:
                self.stop_latched = False
        else:
            self.stop_latched = False

        assist_switch_active = self._switch_active(
            self.assist_channel,
            self.assist_high_is_enable,
        )
        if self.stop_latched:
            self.assist_enable = False
        elif self.assist_toggle_mode:
            if self._last_assist_switch_active is None:
                self._last_assist_switch_active = assist_switch_active
            elif assist_switch_active and not self._last_assist_switch_active:
                self.assist_enable = not self.assist_enable
            self._last_assist_switch_active = assist_switch_active
        else:
            self.assist_enable = assist_switch_active

        vx = self._axis(self.forward_channel, self.forward_reverse) * self.max_forward_speed
        vy = self._axis(self.lateral_channel, self.lateral_reverse) * self.max_lateral_speed
        vz = self._axis(self.vertical_channel, self.vertical_reverse) * self.max_vertical_speed
        if self.stop_latched:
            vx = vy = vz = 0.0

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        self.human_pub.publish(msg)
        if self.stop_pub is not None:
            self.stop_pub.publish(Bool(data=bool(self.stop_latched)))
        self.assist_pub.publish(Bool(data=bool(self.assist_enable)))

        status = "STOP" if self.stop_latched else ("ASSIST" if self.assist_enable else "DIRECT")
        self.status_pub.publish(
            String(
                data=(
                    f"{status} vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
                    f"assist_ch={self.assist_channel} estop={estop_active} reset={reset_active}"
                )
            )
        )

    def _publish_zero(self, reason):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        self.human_pub.publish(msg)
        if self.stop_pub is not None:
            self.stop_pub.publish(Bool(data=bool(self.stop_latched)))
        self.assist_pub.publish(Bool(data=False))
        self.status_pub.publish(String(data=reason))


if __name__ == "__main__":
    try:
        RcInputNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
