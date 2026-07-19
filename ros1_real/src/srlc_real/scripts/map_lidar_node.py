#!/usr/bin/env python3
"""PCD-map LiDAR publisher for real PX4 SRLC deployment.

The SRLC policy should consume LiDAR data as a ROS topic on the real drone.
This node owns the map dependency: it raycasts the pre-scanned PCD map from
source-stamped Nokov odometry and publishes a training-compatible range image.
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

from srlc_real.msg import ObstacleClearance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.clearance_geometry import PcdClearanceGeometry  # noqa: E402
from srlc_real_deployment.pcd_raycast import (  # noqa: E402
    PcdRaycaster,
    policy_surface_distances,
)


class MapLidarNode:
    def __init__(self):
        rospy.init_node("map_lidar_node", anonymous=False)

        self.pcd_file = rospy.get_param("~pcd_file", "")
        if not self.pcd_file:
            rospy.logfatal("[MapLiDAR] ~pcd_file is required")
            rospy.signal_shutdown("missing pcd_file")
            return

        self.odom_topic = rospy.get_param("~odom_topic", "/nokov/local_position/odom")
        self.range_topic = rospy.get_param("~range_image_topic", "/srlc/lidar/range_image")
        self.points_topic = rospy.get_param("~raycast_points_topic", "/srlc/lidar/raycast_points")
        self.min_dist_topic = rospy.get_param("~min_distance_topic", "/srlc/lidar/min_distance")
        self.safety_dist_topic = rospy.get_param(
            "~min_safety_distance_topic", "/srlc/lidar/min_safety_distance"
        )
        self.clearance_topic = rospy.get_param(
            "~clearance_topic", "/srlc/lidar/obstacle_clearance"
        )
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))

        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_vfov = rospy.get_param("~lidar_vfov", [-10.0, 20.0])
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.lidar_hres = float(rospy.get_param("~lidar_hres", 10.0))
        self.lidar_hbeams = int(360.0 / self.lidar_hres)

        self.map_origin_xyz = np.asarray(
            rospy.get_param("~map_origin_xyz", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        self.map_yaw = math.radians(float(rospy.get_param("~map_yaw_deg", 0.0)))
        self._cos_yaw = math.cos(self.map_yaw)
        self._sin_yaw = math.sin(self.map_yaw)
        self._rotation_map_from_local = np.array(
            [
                [self._cos_yaw, -self._sin_yaw, 0.0],
                [self._sin_yaw, self._cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.vehicle_half_extents = np.asarray(
            rospy.get_param("~vehicle_half_extents", [0.15, 0.15, 0.05]),
            dtype=np.float64,
        )
        self.policy_extra_margin = np.asarray(
            rospy.get_param("~policy_extra_margin", [0.05, 0.05, 0.0]),
            dtype=np.float64,
        )
        self.policy_half_extents = (
            self.vehicle_half_extents + self.policy_extra_margin
        )
        self.clearance_cap = float(rospy.get_param("~clearance_cap", 1.0))
        self._validate_geometry_config()

        resolution = float(rospy.get_param("~map_resolution", 0.1))
        self.raycaster = PcdRaycaster(
            self.pcd_file,
            resolution=resolution,
            inflate=(0.0, 0.0, 0.0),
        )
        self.clearance_geometry = PcdClearanceGeometry(self.pcd_file)

        self.odom = None
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self._odom_cb,
            queue_size=50,
        )
        self.range_pub = rospy.Publisher(self.range_topic, Float32MultiArray, queue_size=2)
        self.points_pub = rospy.Publisher(self.points_topic, PointCloud2, queue_size=2)
        self.min_dist_pub = rospy.Publisher(
            self.min_dist_topic,
            Float32,
            queue_size=50,
        )
        self.safety_dist_pub = rospy.Publisher(
            self.safety_dist_topic, Float32, queue_size=2
        )
        self.clearance_pub = rospy.Publisher(
            self.clearance_topic, ObstacleClearance, queue_size=50
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
        self._publish_source_clearance(msg)
        self.odom = msg

    def _validate_geometry_config(self):
        for name, value in (
            ("map_origin_xyz", self.map_origin_xyz),
            ("vehicle_half_extents", self.vehicle_half_extents),
            ("policy_extra_margin", self.policy_extra_margin),
        ):
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"~{name} must contain three finite values")
        if np.any(self.vehicle_half_extents < 0.0):
            raise ValueError("~vehicle_half_extents must be non-negative")
        if np.any(self.policy_extra_margin < 0.0):
            raise ValueError("~policy_extra_margin must be non-negative")
        if not math.isfinite(self.clearance_cap) or self.clearance_cap <= 0.0:
            raise ValueError("~clearance_cap must be a positive finite value")

    def _local_to_map_position(self, pos_local):
        x = self._cos_yaw * pos_local[0] - self._sin_yaw * pos_local[1]
        y = self._sin_yaw * pos_local[0] + self._cos_yaw * pos_local[1]
        return np.array([x, y, pos_local[2]], dtype=np.float64) + self.map_origin_xyz

    def _pose_from_odom(self, odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        pos_local = np.asarray([p.x, p.y, p.z], dtype=np.float64)
        quaternion_local = np.asarray([q.x, q.y, q.z, q.w], dtype=np.float64)
        if not np.isfinite(pos_local).all() or not np.isfinite(
            quaternion_local
        ).all():
            raise ValueError("odometry pose must be finite")
        quaternion_norm = float(np.linalg.norm(quaternion_local))
        if quaternion_norm <= 0.0:
            raise ValueError("odometry quaternion must have non-zero length")
        quaternion_local /= quaternion_norm

        rotation_local_from_body = np.asarray(
            tf.transformations.quaternion_matrix(quaternion_local.tolist())[:3, :3],
            dtype=np.float64,
        )
        rotation_map_from_body = (
            self._rotation_map_from_local @ rotation_local_from_body
        )
        return self._local_to_map_position(pos_local), rotation_map_from_body

    def _publish_source_clearance(self, odom):
        try:
            pos_map, rotation_map_from_body = self._pose_from_odom(odom)
            clearance_result = self.clearance_geometry.query(
                pos_map,
                rotation_map_from_body,
                self.vehicle_half_extents,
                clearance_cap=self.clearance_cap,
            )
        except (TypeError, ValueError) as exc:
            rospy.logerr_throttle(
                1.0,
                "[MapLiDAR] Invalid odometry/clearance geometry: %s",
                exc,
            )
            self._publish_invalid_clearance(odom)
            return

        self.min_dist_pub.publish(
            Float32(data=float(clearance_result.center_distance))
        )
        self._publish_clearance(
            odom,
            clearance_result,
            self._rotation_map_from_local,
        )

    def _timer_cb(self, _event):
        if self.odom is None:
            return

        odom = self.odom
        try:
            pos_map, rotation_map_from_body = self._pose_from_odom(odom)

            raw_raycast = self.raycaster.raycast_raw(
                pos_map,
                0.0,
                self.lidar_range,
                float(self.lidar_vfov[0]),
                float(self.lidar_vfov[1]),
                self.lidar_vbeams,
                self.lidar_hres,
                direction_frame_yaw=self.map_yaw,
            )
            policy_distances = policy_surface_distances(
                raw_raycast.entry_distances,
                raw_raycast.directions_world,
                raw_raycast.hit_mask,
                rotation_map_from_body,
                self.policy_half_extents,
                max_range=self.lidar_range,
            )
        except (TypeError, ValueError) as exc:
            rospy.logerr_throttle(
                1.0,
                "[MapLiDAR] Invalid odometry/policy raycast: %s",
                exc,
            )
            return

        self._publish_range_image(policy_distances)
        self._publish_points(raw_raycast.points, odom.header.stamp)
        self.safety_dist_pub.publish(
            Float32(data=float(np.min(policy_distances)))
        )

    def _publish_range_image(self, policy_distances):
        expected = self.lidar_hbeams * self.lidar_vbeams
        if policy_distances.shape != (expected,):
            rospy.logwarn_throttle(
                2.0,
                "[MapLiDAR] Unexpected policy range count: %d != %d",
                policy_distances.size,
                expected,
            )
            return

        dists = np.clip(policy_distances, 0.0, self.lidar_range)
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

    def _publish_points(self, points, source_stamp):
        msg = PointCloud2()
        msg.header = Header(stamp=source_stamp, frame_id=self.frame_id)
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

    def _publish_clearance(
        self,
        odom,
        clearance_result,
        rotation_map_from_local,
    ):
        nearest_point_local = rotation_map_from_local.T @ (
            clearance_result.nearest_point - self.map_origin_xyz
        )
        escape_direction_local = (
            rotation_map_from_local.T @ clearance_result.escape_direction
        )

        clearance_msg = ObstacleClearance()
        clearance_msg.header = odom.header
        clearance_msg.valid = True
        clearance_msg.surface_clearance = float(clearance_result.surface_clearance)
        clearance_msg.center_distance = float(clearance_result.center_distance)
        clearance_msg.nearest_obstacle_point.x = float(nearest_point_local[0])
        clearance_msg.nearest_obstacle_point.y = float(nearest_point_local[1])
        clearance_msg.nearest_obstacle_point.z = float(nearest_point_local[2])
        clearance_msg.escape_direction.x = float(escape_direction_local[0])
        clearance_msg.escape_direction.y = float(escape_direction_local[1])
        clearance_msg.escape_direction.z = float(escape_direction_local[2])
        self.clearance_pub.publish(clearance_msg)

    def _publish_invalid_clearance(self, odom):
        clearance_msg = ObstacleClearance()
        clearance_msg.header = odom.header
        clearance_msg.valid = False
        self.clearance_pub.publish(clearance_msg)


if __name__ == "__main__":
    try:
        MapLidarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
