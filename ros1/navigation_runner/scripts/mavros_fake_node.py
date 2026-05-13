#!/usr/bin/env python3
"""Fake MAVROS Node — provides MAVROS services, state, and optional odom.

IPC calls MAVROS services during initialization. This node provides
them with always-success responses, and publishes /mavros/state
indicating the vehicle is armed and in OFFBOARD mode.
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode, SetModeResponse
from mavros_msgs.srv import CommandBool, CommandBoolResponse
from mavros_msgs.srv import CommandLong, CommandLongResponse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState


class FakeMAVROS:
    def __init__(self):
        rospy.init_node('mavros_fake_node', anonymous=False)

        self.state = State()
        self.state.connected = True
        self.state.armed = bool(rospy.get_param('~armed', True))
        self.state.guided = True
        self.state.mode = rospy.get_param('~mode', 'OFFBOARD')
        self.state.system_status = 4  # MAV_STATE_ACTIVE

        self.publish_odom = bool(rospy.get_param('~publish_odom', False))
        self.odom_rate = float(rospy.get_param('~odom_rate', 30.0))
        self.integrate_cmd = bool(rospy.get_param('~integrate_cmd', True))
        self.cmd_timeout = float(rospy.get_param('~cmd_timeout', 0.4))
        self.max_xy_speed = float(rospy.get_param('~max_xy_speed', 1.5))
        self.max_z_speed = float(rospy.get_param('~max_z_speed', 0.5))
        self.position = [
            float(rospy.get_param('~initial_x', 0.0)),
            float(rospy.get_param('~initial_y', 0.0)),
            float(rospy.get_param('~initial_z', 2.0)),
        ]
        self.yaw = math.radians(float(rospy.get_param('~initial_yaw_deg', 0.0)))
        self.velocity = [0.0, 0.0, 0.0]
        self.position_target_z = None
        self.last_cmd_time = None
        self.last_odom_time = rospy.Time.now()

        self.state_pub = rospy.Publisher(
            '/mavros/state', State, queue_size=1, latch=True)
        self.battery_pub = rospy.Publisher(
            '/mavros/battery', BatteryState, queue_size=1, latch=True)
        self.odom_pub = rospy.Publisher(
            '/mavros/local_position/odom', Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher(
            '/mavros/local_position/pose', PoseStamped, queue_size=10)

        self.raw_setpoint_sub = rospy.Subscriber(
            '/mavros/setpoint_raw/local', PositionTarget, self._raw_setpoint_cb, queue_size=1)
        self.pose_setpoint_sub = rospy.Subscriber(
            '/mavros/setpoint_position/local', PoseStamped, self._pose_setpoint_cb, queue_size=1)

        self.set_mode_srv = rospy.Service(
            '/mavros/set_mode', SetMode, self._set_mode_cb)
        self.arming_srv = rospy.Service(
            '/mavros/cmd/arming', CommandBool, self._arming_cb)
        self.cmd_srv = rospy.Service(
            '/mavros/cmd/command', CommandLong, self._command_cb)

        self._publish_state()
        self._publish_battery()

        self.timer = rospy.Timer(rospy.Duration(1.0), self._timer_cb)
        self.odom_timer = None
        if self.publish_odom:
            self.odom_timer = rospy.Timer(rospy.Duration(1.0 / self.odom_rate), self._odom_timer_cb)
        rospy.loginfo(
            "[MAVROS Fake] Services ready (odom=%s, initial=[%.2f, %.2f, %.2f])",
            self.publish_odom,
            self.position[0],
            self.position[1],
            self.position[2],
        )

    def _set_mode_cb(self, req):
        rospy.loginfo("[MAVROS Fake] SetMode: %s", req.custom_mode)
        if req.custom_mode:
            self.state.mode = req.custom_mode
        return SetModeResponse(mode_sent=True)

    def _arming_cb(self, req):
        rospy.loginfo("[MAVROS Fake] Arming: %s", req.value)
        self.state.armed = bool(req.value)
        return CommandBoolResponse(success=True, result=0)

    def _command_cb(self, req):
        rospy.loginfo("[MAVROS Fake] Command: cmd=%d", req.command)
        return CommandLongResponse(success=True, result=0)

    def _publish_state(self):
        self.state.header.stamp = rospy.Time.now()
        self.state_pub.publish(self.state)

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

    def _raw_setpoint_cb(self, msg):
        vx = 0.0 if msg.type_mask & PositionTarget.IGNORE_VX else float(msg.velocity.x)
        vy = 0.0 if msg.type_mask & PositionTarget.IGNORE_VY else float(msg.velocity.y)
        vz = 0.0 if msg.type_mask & PositionTarget.IGNORE_VZ else float(msg.velocity.z)
        hspeed = math.hypot(vx, vy)
        if hspeed > self.max_xy_speed:
            scale = self.max_xy_speed / hspeed
            vx *= scale
            vy *= scale
        vz = max(-self.max_z_speed, min(self.max_z_speed, vz))
        self.velocity = [vx, vy, vz]
        self.position_target_z = None if msg.type_mask & PositionTarget.IGNORE_PZ else float(msg.position.z)
        self.last_cmd_time = rospy.Time.now()

    def _pose_setpoint_cb(self, msg):
        self.velocity = [0.0, 0.0, 0.0]
        self.position_target_z = float(msg.pose.position.z)
        self.last_cmd_time = rospy.Time.now()

    def _odom_timer_cb(self, event):
        now = rospy.Time.now()
        dt = max(0.0, (now - self.last_odom_time).to_sec())
        self.last_odom_time = now

        if self.integrate_cmd and self.last_cmd_time is not None:
            age = (now - self.last_cmd_time).to_sec()
            if age <= self.cmd_timeout:
                self.position[0] += self.velocity[0] * dt
                self.position[1] += self.velocity[1] * dt
                if self.position_target_z is None:
                    self.position[2] += self.velocity[2] * dt
                else:
                    err = self.position_target_z - self.position[2]
                    vz = max(-self.max_z_speed, min(self.max_z_speed, err * 1.5))
                    self.position[2] += vz * dt
                    self.velocity[2] = vz
            else:
                self.velocity = [0.0, 0.0, 0.0]

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.position[0]
        odom.pose.pose.position.y = self.position[1]
        odom.pose.pose.position.z = self.position[2]
        odom.pose.pose.orientation = self._yaw_quat(self.yaw)
        odom.twist.twist.linear.x = self.velocity[0]
        odom.twist.twist.linear.y = self.velocity[1]
        odom.twist.twist.linear.z = self.velocity[2]
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

    @staticmethod
    def _yaw_quat(yaw):
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q


if __name__ == '__main__':
    try:
        FakeMAVROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
