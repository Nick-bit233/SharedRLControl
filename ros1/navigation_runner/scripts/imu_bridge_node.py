#!/usr/bin/env python3
"""IMU Bridge Node — generates sensor_msgs/Imu from CERLAB Gazebo data.

Combines orientation + angular velocity from odom with linear acceleration
from acc_raw to produce standard IMU messages for IPC.
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu


class IMUBridge:
    def __init__(self):
        rospy.init_node('imu_bridge_node', anonymous=False)

        self.rate = rospy.get_param('~rate', 100)
        self.frame_id = rospy.get_param('~frame_id', 'base_link')

        self.orientation = None
        self.angular_velocity = None
        self.linear_acceleration = None

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.acc_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/acc_raw', TwistStamped, self._acc_cb, queue_size=1)

        self.imu_pub = rospy.Publisher('/mavros/imu/data', Imu, queue_size=10)
        self.imu_raw_pub = rospy.Publisher('/mavros/imu/data_raw', Imu, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)
        rospy.loginfo("[IMU Bridge] Ready at %dHz", self.rate)

    def _odom_cb(self, msg):
        self.orientation = msg.pose.pose.orientation
        self.angular_velocity = msg.twist.twist.angular

    def _acc_cb(self, msg):
        self.linear_acceleration = msg.twist.linear

    def _timer_cb(self, event):
        if self.orientation is None:
            return

        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = self.frame_id

        imu.orientation = self.orientation

        if self.angular_velocity is not None:
            imu.angular_velocity = self.angular_velocity

        if self.linear_acceleration is not None:
            imu.linear_acceleration = self.linear_acceleration
        else:
            imu.linear_acceleration.z = 9.81

        imu.orientation_covariance = [0.0] * 9
        imu.angular_velocity_covariance = [0.0] * 9
        imu.linear_acceleration_covariance = [0.0] * 9

        self.imu_pub.publish(imu)
        self.imu_raw_pub.publish(imu)


if __name__ == '__main__':
    try:
        IMUBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
