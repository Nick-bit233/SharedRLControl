#!/usr/bin/env python3
"""Command Bridge Node — converts IPC's PositionCommand to CERLAB cmd_vel.

CERLAB's Gazebo plugin consumes horizontal velocity commands in a yaw-aligned
local frame rather than the full pitched/rolled body frame. Mirror the RL
deployment path: rotate XY by yaw only, and generate Z command from the IPC
position/velocity setpoint so the bridge does not drive the drone downward
while the planner is still trying to climb.
"""

import math
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

try:
    from quadrotor_msgs.msg import PositionCommand
    HAS_QUADROTOR_MSGS = True
except ImportError:
    HAS_QUADROTOR_MSGS = False


def yaw_from_quat(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class CmdBridge:
    def __init__(self):
        rospy.init_node('cmd_bridge_node', anonymous=False)

        self.max_vel = rospy.get_param('~max_vel', 3.0)
        self.yaw_gain = rospy.get_param('~yaw_gain', 1.0)
        self.cmd_timeout = rospy.get_param('~cmd_timeout', 0.5)
        self.altitude_kp = rospy.get_param('~altitude_kp', 1.0)
        self.max_vz = rospy.get_param('~max_vz', 1.0)

        self.current_yaw = 0.0
        self.current_position = None
        self.current_z = None
        self.last_cmd_time = None
        self.stop_requested = False

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.stop_sub = rospy.Subscriber(
            '/experiment_control/stop', Bool, self._stop_cb, queue_size=1)

        if HAS_QUADROTOR_MSGS:
            self.cmd_sub = rospy.Subscriber(
                '/planning/pos_cmd', PositionCommand, self._cmd_cb, queue_size=1)
            rospy.loginfo("[Cmd Bridge] Using quadrotor_msgs/PositionCommand")
        else:
            rospy.logwarn("[Cmd Bridge] quadrotor_msgs not found — "
                          "compile slope_inspection msgs first")

        self.vel_pub = rospy.Publisher(
            '/CERLAB/quadcopter/cmd_vel', TwistStamped, queue_size=10)
        self.pose_pub = rospy.Publisher(
            '/CERLAB/quadcopter/setpoint_pose', PoseStamped, queue_size=10)

        self.safety_timer = rospy.Timer(rospy.Duration(0.05), self._safety_cb)
        rospy.loginfo("[Cmd Bridge] Ready. max_vel=%.1f", self.max_vel)

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.current_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        p = msg.pose.pose.position
        self.current_position = (p.x, p.y, p.z)
        self.current_z = msg.pose.pose.position.z

    def _stop_cb(self, msg):
        self.stop_requested = bool(msg.data)
        if self.stop_requested:
            self._publish_hold()

    def _cmd_cb(self, msg):
        if self.current_z is None:
            return
        if self.stop_requested:
            self._publish_hold()
            return

        # IPC publishes world-frame commands. The CERLAB velocity controller is
        # effectively yaw-aligned local-frame, matching the RL Gazebo bridge.
        cos_yaw = math.cos(self.current_yaw)
        sin_yaw = math.sin(self.current_yaw)
        vx_local = cos_yaw * msg.velocity.x + sin_yaw * msg.velocity.y
        vy_local = -sin_yaw * msg.velocity.x + cos_yaw * msg.velocity.y
        vz_local = msg.velocity.z + self.altitude_kp * (msg.position.z - self.current_z)

        # Clamp
        hspeed = math.hypot(vx_local, vy_local)
        if hspeed > self.max_vel:
            scale = self.max_vel / hspeed
            vx_local *= scale
            vy_local *= scale
        vz_local = max(-self.max_vz, min(self.max_vz, vz_local))

        # Yaw rate: use yaw_dot if available, else proportional
        yaw_rate = 0.0
        if hasattr(msg, 'yaw_dot') and abs(msg.yaw_dot) > 0.01:
            yaw_rate = msg.yaw_dot
        else:
            yaw_error = msg.yaw - self.current_yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = self.yaw_gain * yaw_error

        twist = TwistStamped()
        twist.header.stamp = rospy.Time.now()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = float(vx_local)
        twist.twist.linear.y = float(vy_local)
        twist.twist.linear.z = float(vz_local)
        twist.twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(twist)
        self.last_cmd_time = rospy.Time.now()

    def _publish_hold(self):
        if self.current_position is None:
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = 'base_link'
            self.vel_pub.publish(twist)
            return

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'world'
        pose.pose.position.x = float(self.current_position[0])
        pose.pose.position.y = float(self.current_position[1])
        pose.pose.position.z = float(self.current_position[2])
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

    def _safety_cb(self, event):
        if self.stop_requested:
            self._publish_hold()
            return
        if self.last_cmd_time is None:
            return
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > self.cmd_timeout:
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = 'base_link'
            self.vel_pub.publish(twist)


if __name__ == '__main__':
    try:
        CmdBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
