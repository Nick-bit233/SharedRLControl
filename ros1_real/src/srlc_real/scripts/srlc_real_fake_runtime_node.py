#!/usr/bin/env python3
"""Deterministic fake PX4/MAVROS/RC runtime for SRLC real-stack smoke tests."""

import math

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from mavros_msgs.msg import PositionTarget, RCIn, State
from mavros_msgs.srv import CommandBool, CommandBoolResponse
from mavros_msgs.srv import CommandLong, CommandLongResponse
from mavros_msgs.srv import MessageInterval, MessageIntervalResponse
from mavros_msgs.srv import SetMode, SetModeResponse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, Imu


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class FakeRealRuntime:
    def __init__(self):
        rospy.init_node("srlc_real_fake_runtime_node", anonymous=False)

        self.state = State()
        self.state.connected = _param_bool(rospy.get_param("~connected", True))
        self.state.armed = _param_bool(rospy.get_param("~armed", False))
        self.state.guided = True
        self.state.mode = str(rospy.get_param("~mode", "OFFBOARD"))
        self.state.system_status = 4

        self.position = [
            float(rospy.get_param("~initial_x", 0.0)),
            float(rospy.get_param("~initial_y", 0.0)),
            float(rospy.get_param("~initial_z", 0.0)),
        ]
        self.yaw = math.radians(float(rospy.get_param("~initial_yaw_deg", 0.0)))
        self.velocity = [0.0, 0.0, 0.0]
        self.target_position = [None, None, None]
        self.last_cmd_time = None
        self.last_odom_time = rospy.Time.now()

        self.state_rate = float(rospy.get_param("~state_rate", 10.0))
        self.odom_rate = float(rospy.get_param("~odom_rate", 30.0))
        self.rc_rate = float(rospy.get_param("~rc_rate", 30.0))
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.max_xy_speed = float(rospy.get_param("~fake_max_xy_speed", 1.5))
        self.max_z_speed = float(rospy.get_param("~fake_max_z_speed", 0.8))
        self.xy_hold_kp = float(rospy.get_param("~xy_hold_kp", 1.5))
        self.z_hold_kp = float(rospy.get_param("~z_hold_kp", 1.5))

        self.channel_base = int(rospy.get_param("~channel_base", 1))
        self.forward_channel = int(rospy.get_param("~forward_channel", 2))
        self.lateral_channel = int(rospy.get_param("~lateral_channel", 1))
        self.vertical_channel = int(rospy.get_param("~vertical_channel", 3))
        self.pwm_min = float(rospy.get_param("~pwm_min", 1000.0))
        self.pwm_mid = float(rospy.get_param("~pwm_mid", 1500.0))
        self.pwm_max = float(rospy.get_param("~pwm_max", 2000.0))
        self.forward_reverse = _param_bool(rospy.get_param("~forward_reverse", False))
        self.lateral_reverse = _param_bool(rospy.get_param("~lateral_reverse", False))
        self.vertical_reverse = _param_bool(rospy.get_param("~vertical_reverse", False))
        self.max_forward_speed = float(rospy.get_param("~max_forward_speed", 1.0))
        self.max_lateral_speed = float(rospy.get_param("~max_lateral_speed", 1.0))
        self.max_vertical_speed = float(rospy.get_param("~max_vertical_speed", 0.4))
        self.fake_forward_speed = float(rospy.get_param("~fake_forward_speed", 1.0))
        self.fake_lateral_speed = float(rospy.get_param("~fake_lateral_speed", 0.0))
        self.fake_vertical_speed = float(rospy.get_param("~fake_vertical_speed", 0.0))
        self.motion_after = float(rospy.get_param("~motion_after", 5.0))
        self.stop_after = float(rospy.get_param("~stop_after", 0.0))
        self.start_time = rospy.Time.now()

        self.state_pub = rospy.Publisher("/mavros/state", State, queue_size=2, latch=True)
        self.battery_pub = rospy.Publisher("/mavros/battery", BatteryState, queue_size=1, latch=True)
        self.rc_pub = rospy.Publisher("/mavros/rc/in", RCIn, queue_size=10)
        self.nokov_odom_pub = rospy.Publisher("/nokov/local_position/odom", Odometry, queue_size=10)
        self.mavros_odom_pub = rospy.Publisher("/mavros/local_position/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher("/mavros/local_position/pose", PoseStamped, queue_size=10)
        self.vision_pose_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=10)
        self.imu_pub = rospy.Publisher("/nokov/imu/data", Imu, queue_size=10)
        self.mavros_imu_pub = rospy.Publisher("/mavros/imu/data", Imu, queue_size=10)

        self.raw_setpoint_sub = rospy.Subscriber(
            "/mavros/setpoint_raw/local", PositionTarget, self._raw_setpoint_cb, queue_size=1
        )
        self.set_mode_srv = rospy.Service("/mavros/set_mode", SetMode, self._set_mode_cb)
        self.arming_srv = rospy.Service("/mavros/cmd/arming", CommandBool, self._arming_cb)
        self.command_srv = rospy.Service("/mavros/cmd/command", CommandLong, self._command_cb)
        self.message_interval_srv = rospy.Service(
            "/mavros/set_message_interval", MessageInterval, self._message_interval_cb
        )

        self.state_timer = rospy.Timer(rospy.Duration(1.0 / self.state_rate), self._state_timer_cb)
        self.odom_timer = rospy.Timer(rospy.Duration(1.0 / self.odom_rate), self._odom_timer_cb)
        self.rc_timer = rospy.Timer(rospy.Duration(1.0 / self.rc_rate), self._rc_timer_cb)

        rospy.loginfo(
            "[SRLC Fake] mode=%s armed=%s initial=[%.2f, %.2f, %.2f] motion_after=%.1fs vx=%.2f",
            self.state.mode,
            self.state.armed,
            self.position[0],
            self.position[1],
            self.position[2],
            self.motion_after,
            self.fake_forward_speed,
        )

    def _set_mode_cb(self, req):
        if req.custom_mode:
            self.state.mode = str(req.custom_mode)
        rospy.loginfo("[SRLC Fake] SetMode -> %s", self.state.mode)
        return SetModeResponse(mode_sent=True)

    def _arming_cb(self, req):
        self.state.armed = bool(req.value)
        rospy.loginfo("[SRLC Fake] Arming -> %s", self.state.armed)
        return CommandBoolResponse(success=True, result=0)

    def _command_cb(self, req):
        rospy.loginfo("[SRLC Fake] CommandLong command=%d", req.command)
        return CommandLongResponse(success=True, result=0)

    def _message_interval_cb(self, req):
        rospy.loginfo(
            "[SRLC Fake] MessageInterval id=%d rate=%.1fHz",
            req.message_id,
            req.message_rate,
        )
        return MessageIntervalResponse(success=True)

    def _raw_setpoint_cb(self, msg):
        cmd = [0.0, 0.0, 0.0]
        target = [None, None, None]

        if not (msg.type_mask & PositionTarget.IGNORE_VX):
            cmd[0] = float(msg.velocity.x)
        if not (msg.type_mask & PositionTarget.IGNORE_VY):
            cmd[1] = float(msg.velocity.y)
        if not (msg.type_mask & PositionTarget.IGNORE_VZ):
            cmd[2] = float(msg.velocity.z)

        if not (msg.type_mask & PositionTarget.IGNORE_PX):
            target[0] = float(msg.position.x)
        if not (msg.type_mask & PositionTarget.IGNORE_PY):
            target[1] = float(msg.position.y)
        if not (msg.type_mask & PositionTarget.IGNORE_PZ):
            target[2] = float(msg.position.z)

        hspeed = math.hypot(cmd[0], cmd[1])
        if hspeed > self.max_xy_speed:
            scale = self.max_xy_speed / hspeed
            cmd[0] *= scale
            cmd[1] *= scale
        cmd[2] = self._clamp(cmd[2], -self.max_z_speed, self.max_z_speed)

        self.velocity = cmd
        self.target_position = target
        self.last_cmd_time = rospy.Time.now()

    def _state_timer_cb(self, _event):
        self.state.header.stamp = rospy.Time.now()
        self.state_pub.publish(self.state)

        battery = BatteryState()
        battery.header.stamp = rospy.Time.now()
        battery.voltage = 12.0
        battery.percentage = 0.9
        battery.present = True
        self.battery_pub.publish(battery)

    def _odom_timer_cb(self, _event):
        now = rospy.Time.now()
        dt = max(0.0, (now - self.last_odom_time).to_sec())
        self.last_odom_time = now

        if self.last_cmd_time is None or (now - self.last_cmd_time).to_sec() > self.cmd_timeout:
            self.velocity = [0.0, 0.0, 0.0]
        else:
            self._apply_position_hold_targets()
            self.position[0] += self.velocity[0] * dt
            self.position[1] += self.velocity[1] * dt
            self.position[2] += self.velocity[2] * dt

        odom = self._build_odom(now)
        self.nokov_odom_pub.publish(odom)
        self.mavros_odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)
        self.vision_pose_pub.publish(pose)

        imu = Imu()
        imu.header = odom.header
        imu.orientation = odom.pose.pose.orientation
        self.imu_pub.publish(imu)
        self.mavros_imu_pub.publish(imu)

    def _apply_position_hold_targets(self):
        if self.target_position[0] is not None and self.target_position[1] is not None:
            vx = (self.target_position[0] - self.position[0]) * self.xy_hold_kp
            vy = (self.target_position[1] - self.position[1]) * self.xy_hold_kp
            hspeed = math.hypot(vx, vy)
            if hspeed > self.max_xy_speed:
                scale = self.max_xy_speed / hspeed
                vx *= scale
                vy *= scale
            self.velocity[0] = vx
            self.velocity[1] = vy
        if self.target_position[2] is not None:
            vz = (self.target_position[2] - self.position[2]) * self.z_hold_kp
            self.velocity[2] = self._clamp(vz, -self.max_z_speed, self.max_z_speed)

    def _rc_timer_cb(self, _event):
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        motion_active = elapsed >= self.motion_after
        if self.stop_after > 0.0 and elapsed >= self.stop_after:
            motion_active = False

        forward = self.fake_forward_speed if motion_active else 0.0
        lateral = self.fake_lateral_speed if motion_active else 0.0
        vertical = self.fake_vertical_speed if motion_active else 0.0

        channels = [int(round(self.pwm_mid))] * 18
        self._set_channel(
            channels,
            self.forward_channel,
            self._speed_to_pwm(forward, self.max_forward_speed, self.forward_reverse),
        )
        self._set_channel(
            channels,
            self.lateral_channel,
            self._speed_to_pwm(lateral, self.max_lateral_speed, self.lateral_reverse),
        )
        self._set_channel(
            channels,
            self.vertical_channel,
            self._speed_to_pwm(vertical, self.max_vertical_speed, self.vertical_reverse),
        )

        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.channels = channels
        self.rc_pub.publish(msg)

    def _build_odom(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])
        odom.pose.pose.orientation = self._yaw_quat(self.yaw)
        odom.twist.twist.linear.x = float(self.velocity[0])
        odom.twist.twist.linear.y = float(self.velocity[1])
        odom.twist.twist.linear.z = float(self.velocity[2])
        return odom

    def _set_channel(self, channels, channel, pwm):
        idx = int(channel) - self.channel_base
        if 0 <= idx < len(channels):
            channels[idx] = int(round(pwm))

    def _speed_to_pwm(self, speed, max_speed, reverse):
        if max_speed <= 0.0:
            value = 0.0
        else:
            value = self._clamp(speed / max_speed, -1.0, 1.0)
        if reverse:
            value = -value
        if value >= 0.0:
            return self.pwm_mid + value * (self.pwm_max - self.pwm_mid)
        return self.pwm_mid + value * (self.pwm_mid - self.pwm_min)

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _yaw_quat(yaw):
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q


if __name__ == "__main__":
    try:
        FakeRealRuntime()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
