#!/usr/bin/env python3
"""Deterministic RCIn publisher for SRLC dry-run tests."""

import rospy
from mavros_msgs.msg import RCIn


class SrlcFakeRcNode:
    def __init__(self):
        rospy.init_node("srlc_fake_rc_node", anonymous=False)

        self.rc_topic = rospy.get_param("~rc_topic", "/mavros/rc/in")
        self.rate_hz = float(rospy.get_param("~rate", 30.0))
        self.channel_base = int(rospy.get_param("~channel_base", 1))
        self.forward_channel = int(rospy.get_param("~forward_channel", 2))
        self.lateral_channel = int(rospy.get_param("~lateral_channel", 1))
        self.vertical_channel = int(rospy.get_param("~vertical_channel", 3))
        self.assist_channel = int(rospy.get_param("~assist_channel", 6))
        self.estop_channel = int(rospy.get_param("~estop_channel", 7))
        self.reset_channel = int(rospy.get_param("~reset_channel", 8))

        self.pwm_mid = int(rospy.get_param("~pwm_mid", 1500))
        self.pwm_span = int(rospy.get_param("~pwm_span", 450))
        self.inactive_pwm = int(rospy.get_param("~inactive_pwm", 1100))
        self.active_pwm = int(rospy.get_param("~active_pwm", 1900))
        self.reset_duration = float(rospy.get_param("~reset_duration", 1.0))
        self.reset_after = float(rospy.get_param("~reset_after", 1.0))
        self.assist_after = float(rospy.get_param("~assist_after", 2.0))
        self.motion_after = float(rospy.get_param("~motion_after", 3.0))
        self.estop_after = float(rospy.get_param("~estop_after", 0.0))

        self.forward_stick = float(rospy.get_param("~forward_stick", 0.45))
        self.lateral_stick = float(rospy.get_param("~lateral_stick", 0.05))
        self.vertical_stick = float(rospy.get_param("~vertical_stick", 0.0))

        self.start_time = rospy.Time.now()
        self.pub = rospy.Publisher(self.rc_topic, RCIn, queue_size=10)
        rospy.loginfo(
            "[SRLC Fake RC] Publishing %s reset@%.1fs assist@%.1fs motion@%.1fs",
            self.rc_topic,
            self.reset_after,
            self.assist_after,
            self.motion_after,
        )

    def _idx(self, channel):
        return channel - self.channel_base

    def _set_channel(self, channels, channel, pwm):
        idx = self._idx(channel)
        if 0 <= idx < len(channels):
            channels[idx] = int(pwm)

    def _stick_pwm(self, value):
        value = max(-1.0, min(1.0, float(value)))
        return int(round(self.pwm_mid + value * self.pwm_span))

    def _message(self):
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        channels = [self.pwm_mid] * 18

        reset_active = self.reset_after <= elapsed < self.reset_after + self.reset_duration
        assist_active = elapsed >= self.assist_after
        estop_active = self.estop_after > 0.0 and elapsed >= self.estop_after
        motion_active = elapsed >= self.motion_after and assist_active and not estop_active

        self._set_channel(channels, self.reset_channel, self.active_pwm if reset_active else self.inactive_pwm)
        self._set_channel(channels, self.assist_channel, self.active_pwm if assist_active else self.inactive_pwm)
        self._set_channel(channels, self.estop_channel, self.active_pwm if estop_active else self.inactive_pwm)

        self._set_channel(channels, self.forward_channel, self._stick_pwm(self.forward_stick if motion_active else 0.0))
        self._set_channel(channels, self.lateral_channel, self._stick_pwm(self.lateral_stick if motion_active else 0.0))
        self._set_channel(channels, self.vertical_channel, self._stick_pwm(self.vertical_stick if motion_active else 0.0))

        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.channels = channels
        return msg

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.pub.publish(self._message())
            rate.sleep()


if __name__ == "__main__":
    try:
        SrlcFakeRcNode().run()
    except rospy.ROSInterruptException:
        pass
