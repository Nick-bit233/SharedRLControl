#!/usr/bin/env python3
"""Fake MAVROS Node — provides MAVROS services and state for IPC.

IPC calls MAVROS services during initialization. This node provides
them with always-success responses, and publishes /mavros/state
indicating the vehicle is armed and in OFFBOARD mode.
"""

import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, SetModeResponse
from mavros_msgs.srv import CommandBool, CommandBoolResponse
from mavros_msgs.srv import CommandLong, CommandLongResponse
from sensor_msgs.msg import BatteryState


class FakeMAVROS:
    def __init__(self):
        rospy.init_node('mavros_fake_node', anonymous=False)

        self.state_pub = rospy.Publisher(
            '/mavros/state', State, queue_size=1, latch=True)
        self.battery_pub = rospy.Publisher(
            '/mavros/battery', BatteryState, queue_size=1, latch=True)

        self.set_mode_srv = rospy.Service(
            '/mavros/set_mode', SetMode, self._set_mode_cb)
        self.arming_srv = rospy.Service(
            '/mavros/cmd/arming', CommandBool, self._arming_cb)
        self.cmd_srv = rospy.Service(
            '/mavros/cmd/command', CommandLong, self._command_cb)

        self._publish_state()
        self._publish_battery()

        self.timer = rospy.Timer(rospy.Duration(1.0), self._timer_cb)
        rospy.loginfo("[MAVROS Fake] Services ready")

    def _set_mode_cb(self, req):
        rospy.loginfo("[MAVROS Fake] SetMode: %s", req.custom_mode)
        return SetModeResponse(mode_sent=True)

    def _arming_cb(self, req):
        rospy.loginfo("[MAVROS Fake] Arming: %s", req.value)
        return CommandBoolResponse(success=True, result=0)

    def _command_cb(self, req):
        rospy.loginfo("[MAVROS Fake] Command: cmd=%d", req.command)
        return CommandLongResponse(success=True, result=0)

    def _publish_state(self):
        s = State()
        s.header.stamp = rospy.Time.now()
        s.connected = True
        s.armed = True
        s.guided = True
        s.mode = "OFFBOARD"
        s.system_status = 4  # MAV_STATE_ACTIVE
        self.state_pub.publish(s)

    def _publish_battery(self):
        b = BatteryState()
        b.header.stamp = rospy.Time.now()
        b.voltage = 16.8
        b.percentage = 1.0
        b.present = True
        self.battery_pub.publish(b)

    def _timer_cb(self, event):
        self._publish_state()
        self._publish_battery()


if __name__ == '__main__':
    try:
        FakeMAVROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
