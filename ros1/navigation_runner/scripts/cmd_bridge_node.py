#!/usr/bin/env python3
"""Command Bridge Node — converts IPC's PositionCommand to CERLAB cmd_vel.

Subscribes to quadrotor_msgs/PositionCommand (world-frame velocity),
rotates to body frame, and publishes as TwistStamped for CERLAB Gazebo.
"""

import math
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

try:
    from quadrotor_msgs.msg import PositionCommand
    HAS_QUADROTOR_MSGS = True
except ImportError:
    HAS_QUADROTOR_MSGS = False


def quat_to_rotmat(qx, qy, qz, qw):
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix (body-to-world)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])


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

        self.rot_matrix = None  # body-to-world
        self.current_yaw = 0.0
        self.last_cmd_time = None

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)

        if HAS_QUADROTOR_MSGS:
            self.cmd_sub = rospy.Subscriber(
                '/planning/pos_cmd', PositionCommand, self._cmd_cb, queue_size=1)
            rospy.loginfo("[Cmd Bridge] Using quadrotor_msgs/PositionCommand")
        else:
            rospy.logwarn("[Cmd Bridge] quadrotor_msgs not found — "
                          "compile slope_inspection msgs first")

        self.vel_pub = rospy.Publisher(
            '/CERLAB/quadcopter/cmd_vel', TwistStamped, queue_size=10)

        self.safety_timer = rospy.Timer(rospy.Duration(0.05), self._safety_cb)
        rospy.loginfo("[Cmd Bridge] Ready. max_vel=%.1f", self.max_vel)

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.rot_matrix = quat_to_rotmat(q.x, q.y, q.z, q.w)
        self.current_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def _cmd_cb(self, msg):
        if self.rot_matrix is None:
            return

        # World-frame velocity from IPC
        vel_world = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])

        # Rotate to body frame: v_body = R^T @ v_world
        vel_body = self.rot_matrix.T @ vel_world

        # Clamp
        speed = np.linalg.norm(vel_body)
        if speed > self.max_vel:
            vel_body *= self.max_vel / speed

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
        twist.twist.linear.x = float(vel_body[0])
        twist.twist.linear.y = float(vel_body[1])
        twist.twist.linear.z = float(vel_body[2])
        twist.twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(twist)
        self.last_cmd_time = rospy.Time.now()

    def _safety_cb(self, event):
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
