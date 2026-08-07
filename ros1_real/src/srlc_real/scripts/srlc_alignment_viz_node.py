#!/usr/bin/env python3
"""Publish map-aligned odometry and dry-run alignment markers."""

import math

import rospy
import tf.transformations
from geometry_msgs.msg import Point, Quaternion, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SrlcAlignmentVizNode:
    def __init__(self):
        rospy.init_node("srlc_alignment_viz_node", anonymous=False)

        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        self.odom_map_topic = rospy.get_param("~odom_map_topic", "/srlc/alignment/odom_map")
        self.marker_topic = rospy.get_param("~marker_topic", "/srlc/alignment/markers")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.map_origin = [float(v) for v in rospy.get_param("~map_origin_xyz", [0.0, 0.0, 0.0])]
        self.map_yaw = math.radians(float(rospy.get_param("~map_yaw_deg", 0.0)))
        self.geofence_x = [float(v) for v in rospy.get_param("~geofence_x", [-3.0, 3.0])]
        self.geofence_y = [float(v) for v in rospy.get_param("~geofence_y", [-3.0, 3.0])]
        self.min_altitude = float(rospy.get_param("~min_altitude", 0.5))
        self.max_altitude = float(rospy.get_param("~max_altitude", 5.0))
        self.forward_length = float(rospy.get_param("~forward_length", 2.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))

        self._cos_yaw = math.cos(self.map_yaw)
        self._sin_yaw = math.sin(self.map_yaw)

        self.odom_pub = rospy.Publisher(self.odom_map_topic, Odometry, queue_size=5)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._timer_cb)
        rospy.loginfo(
            "[SRLC Alignment] origin=%s yaw=%.1fdeg geofence_x=%s geofence_y=%s",
            self.map_origin,
            math.degrees(self.map_yaw),
            self.geofence_x,
            self.geofence_y,
        )

    def _local_to_map_xy(self, x, y):
        mx = self._cos_yaw * x - self._sin_yaw * y + self.map_origin[0]
        my = self._sin_yaw * x + self._cos_yaw * y + self.map_origin[1]
        return mx, my

    def _local_to_map_point(self, x, y, z):
        mx, my = self._local_to_map_xy(x, y)
        return Point(x=mx, y=my, z=z + self.map_origin[2])

    def _odom_cb(self, msg):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = msg.child_frame_id or "base_link"
        p = msg.pose.pose.position
        out.pose.pose.position = self._local_to_map_point(p.x, p.y, p.z)

        q = msg.pose.pose.orientation
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw + self.map_yaw)
        out.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        out.pose.covariance = msg.pose.covariance

        v = msg.twist.twist.linear
        out.twist.twist.linear.x = self._cos_yaw * v.x - self._sin_yaw * v.y
        out.twist.twist.linear.y = self._sin_yaw * v.x + self._cos_yaw * v.y
        out.twist.twist.linear.z = v.z
        out.twist.twist.angular = msg.twist.twist.angular
        out.twist.covariance = msg.twist.covariance
        self.odom_pub.publish(out)

    def _timer_cb(self, _event):
        markers = MarkerArray()
        markers.markers.append(self._geofence_marker())
        markers.markers.append(self._takeoff_marker())
        markers.markers.append(self._forward_marker())
        self.marker_pub.publish(markers)

    def _base_marker(self, marker_id, marker_type, ns):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        return marker

    def _geofence_marker(self):
        marker = self._base_marker(0, Marker.CUBE, "geofence")
        center_x = 0.5 * (self.geofence_x[0] + self.geofence_x[1])
        center_y = 0.5 * (self.geofence_y[0] + self.geofence_y[1])
        center_z = 0.5 * (self.min_altitude + self.max_altitude)
        marker.pose.position = self._local_to_map_point(center_x, center_y, center_z)
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, self.map_yaw)
        marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        marker.scale = Vector3(
            x=abs(self.geofence_x[1] - self.geofence_x[0]),
            y=abs(self.geofence_y[1] - self.geofence_y[0]),
            z=abs(self.max_altitude - self.min_altitude),
        )
        marker.color = ColorRGBA(r=1.0, g=0.7, b=0.0, a=0.12)
        return marker

    def _takeoff_marker(self):
        marker = self._base_marker(1, Marker.SPHERE, "takeoff")
        marker.pose.position = self._local_to_map_point(0.0, 0.0, 0.0)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.25, y=0.25, z=0.25)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
        return marker

    def _forward_marker(self):
        marker = self._base_marker(2, Marker.ARROW, "forward")
        marker.points = [
            self._local_to_map_point(0.0, 0.0, self.min_altitude),
            self._local_to_map_point(self.forward_length, 0.0, self.min_altitude),
        ]
        marker.scale = Vector3(x=0.05, y=0.12, z=0.12)
        marker.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)
        return marker


if __name__ == "__main__":
    try:
        SrlcAlignmentVizNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
