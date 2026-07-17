#!/usr/bin/env python3
"""PCD-map LiDAR publisher for real PX4 SRLC deployment.

The SRLC policy should consume LiDAR data as a ROS topic on the real drone.
This node owns the map dependency: it raycasts the pre-scanned PCD map from
MAVROS odometry and publishes a training-compatible normalized range image.
"""
import math
import os
import sys

import numpy as np
import rospy
import tf.transformations
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Float32MultiArray, Header, MultiArrayDimension

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.pcd_raycast import PcdRaycaster, minimum_raycast_distance


class MapLidarNode:
    def __init__(self):
        rospy.init_node("map_lidar_node", anonymous=False)

        self.pcd_file = rospy.get_param("~pcd_file", "")
        if not self.pcd_file:
            rospy.logfatal("[MapLiDAR] ~pcd_file is required")
            rospy.signal_shutdown("missing pcd_file")
            return

        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        self.range_topic = rospy.get_param("~range_image_topic", "/srlc/lidar/range_image")
        self.points_topic = rospy.get_param("~raycast_points_topic", "/srlc/lidar/raycast_points")
        self.min_dist_topic = rospy.get_param("~min_distance_topic", "/srlc/lidar/min_distance")
        self.safety_dist_topic = rospy.get_param(
            "~min_safety_distance_topic", "/srlc/lidar/min_safety_distance"
        )
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))

        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_vfov = rospy.get_param("~lidar_vfov", [-10.0, 20.0])
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.lidar_hres = float(rospy.get_param("~lidar_hres", 10.0))
        self.lidar_hbeams = int(360.0 / self.lidar_hres)

        self.map_origin_xyz = np.array(
            rospy.get_param("~map_origin_xyz", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        self.map_yaw = math.radians(float(rospy.get_param("~map_yaw_deg", 0.0)))
        self._cos_yaw = math.cos(self.map_yaw)
        self._sin_yaw = math.sin(self.map_yaw)

        resolution = float(rospy.get_param("~map_resolution", 0.1))
        inflate = tuple(rospy.get_param("~inflate_xyz", [0.15, 0.15, 0.05]))
        self.raycaster = PcdRaycaster(self.pcd_file, resolution=resolution, inflate=inflate)

        self.odom = None
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        self.range_pub = rospy.Publisher(self.range_topic, Float32MultiArray, queue_size=2)
        self.points_pub = rospy.Publisher(self.points_topic, PointCloud2, queue_size=2)
        self.min_dist_pub = rospy.Publisher(self.min_dist_topic, Float32, queue_size=2)
        self.safety_dist_pub = rospy.Publisher(
            self.safety_dist_topic, Float32, queue_size=2
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._timer_cb)
        rospy.loginfo(
            "[MapLiDAR] Ready: pcd=%s lidar=%dx%d range=%.2fm topic=%s",
            self.pcd_file,
            self.lidar_hbeams,
            self.lidar_vbeams,
            self.lidar_range,
            self.range_topic,
        )

    def _odom_cb(self, msg):
        self.odom = msg

    def _local_to_map_position(self, pos_local):
        x = self._cos_yaw * pos_local[0] - self._sin_yaw * pos_local[1]
        y = self._sin_yaw * pos_local[0] + self._cos_yaw * pos_local[1]
        return np.array([x, y, pos_local[2]], dtype=np.float32) + self.map_origin_xyz

    def _local_to_map_yaw(self, yaw_local):
        return yaw_local + self.map_yaw

    def _timer_cb(self, _event):
        if self.odom is None:
            return

        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        _, _, yaw_local = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        pos_local = np.array([p.x, p.y, p.z], dtype=np.float32)
        pos_map = self._local_to_map_position(pos_local)
        yaw_map = self._local_to_map_yaw(yaw_local)

        points = self.raycaster.raycast(
            pos_map,
            yaw_map,
            self.lidar_range,
            float(self.lidar_vfov[0]),
            float(self.lidar_vfov[1]),
            self.lidar_vbeams,
            self.lidar_hres,
        )
        self._publish_range_image(points, pos_map)
        self._publish_points(points)
        self.min_dist_pub.publish(Float32(data=float(self.raycaster.nearest_distance(pos_map))))
        self.safety_dist_pub.publish(
            Float32(
                data=minimum_raycast_distance(
                    points,
                    pos_map,
                    max_range=self.lidar_range,
                )
            )
        )

    def _publish_range_image(self, points, pos_map):
        expected = self.lidar_hbeams * self.lidar_vbeams
        if points.shape[0] != expected:
            rospy.logwarn_throttle(
                2.0,
                "[MapLiDAR] Unexpected raycast point count: %d != %d",
                points.shape[0],
                expected,
            )
            return

        dists = np.linalg.norm(points - pos_map.reshape(1, 3), axis=-1)
        dists = np.clip(dists, 0.0, self.lidar_range)
        ranges = ((self.lidar_range - dists) / self.lidar_range).astype(np.float32)
        ranges = np.clip(ranges, 0.0, 1.0).astype(np.float32)

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="channel", size=1, stride=expected),
            MultiArrayDimension(label="horizontal", size=self.lidar_hbeams, stride=self.lidar_vbeams),
            MultiArrayDimension(label="vertical", size=self.lidar_vbeams, stride=1),
        ]
        msg.layout.data_offset = 0
        msg.data = ranges.reshape(-1).tolist()
        self.range_pub.publish(msg)

    def _publish_points(self, points):
        msg = PointCloud2()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        self.points_pub.publish(msg)


if __name__ == "__main__":
    try:
        MapLidarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
