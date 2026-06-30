#!/usr/bin/env python3
"""LiDAR Simulator Node — generates PointCloud2 from static PCD map.

Loads a PCD file, builds a KD-tree, and for each cycle queries points
near the drone's position. Applies FOV filtering and angle-bin occlusion
to simulate realistic LiDAR output for IPC's ROG-Map.
"""

import os
import sys

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.pcd_io import read_pcd_xyz

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def quat_to_rotmat(qx, qy, qz, qw):
    """Quaternion [x,y,z,w] to 3x3 rotation matrix (body-to-world)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])


class LiDARSimulator:
    def __init__(self):
        rospy.init_node('lidar_sim_node', anonymous=False)

        if not HAS_SCIPY:
            rospy.logfatal("[LiDAR Sim] scipy not installed (pip3 install scipy)")
            rospy.signal_shutdown("Missing scipy")
            return

        self.pcd_file = rospy.get_param('~pcd_file', '')
        self.rate = rospy.get_param('~rate', 10.0)
        self.max_range = rospy.get_param('~max_range', 50.0)
        self.min_range = rospy.get_param('~min_range', 0.3)
        self.v_fov_up = rospy.get_param('~v_fov_up', 30.0)
        self.v_fov_down = rospy.get_param('~v_fov_down', -30.0)
        self.h_resolution = rospy.get_param('~h_resolution', 1.0)
        self.v_resolution = rospy.get_param('~v_resolution', 2.0)
        self.noise_stddev = rospy.get_param('~noise_stddev', 0.01)
        self.frame_id = rospy.get_param('~frame_id', 'base_link')

        self.points_world = self._load_pcd(self.pcd_file)
        if self.points_world is None or len(self.points_world) == 0:
            rospy.logfatal("[LiDAR Sim] Failed to load PCD: %s", self.pcd_file)
            rospy.signal_shutdown("No PCD")
            return

        self.kdtree = cKDTree(self.points_world)
        rospy.loginfo("[LiDAR Sim] Loaded %d points, KD-tree built", len(self.points_world))

        self.n_h_bins = int(360.0 / self.h_resolution)
        self.n_v_bins = int((self.v_fov_up - self.v_fov_down) / self.v_resolution)
        self.n_total_bins = self.n_h_bins * self.n_v_bins

        self.position = None
        self.rot_matrix = None  # body-to-world
        self.odom_received = False

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.cloud_pub = rospy.Publisher(
            '/pcl_render_node/cloud', PointCloud2, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)

        rospy.loginfo("[LiDAR Sim] Ready. range=[%.1f,%.1f]m, rate=%.0fHz, bins=%dx%d",
                      self.min_range, self.max_range, self.rate,
                      self.n_h_bins, self.n_v_bins)

    @staticmethod
    def _load_pcd(filepath):
        try:
            points = read_pcd_xyz(filepath)
            return points if len(points) else None
        except Exception as e:
            rospy.logerr("[LiDAR Sim] PCD load error: %s", str(e))
            return None

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.rot_matrix = quat_to_rotmat(q.x, q.y, q.z, q.w)
        self.odom_received = True

    def _timer_cb(self, event):
        if not self.odom_received:
            return

        # Ball query for nearby points
        indices = self.kdtree.query_ball_point(self.position, self.max_range)
        if not indices:
            return

        nearby = self.points_world[indices]

        # World -> body frame: R^T * (p_world - pos)
        local = (nearby - self.position) @ self.rot_matrix  # equivalent to R^T @ diff

        # Range filter
        dists = np.linalg.norm(local, axis=1)
        mask = (dists >= self.min_range) & (dists <= self.max_range)
        local = local[mask]
        dists = dists[mask]
        if len(local) == 0:
            return

        # Azimuth and elevation
        azimuth = np.arctan2(local[:, 1], local[:, 0])  # [-pi, pi]
        elevation = np.arcsin(np.clip(local[:, 2] / dists, -1, 1))

        # Vertical FOV filter
        v_up = np.radians(self.v_fov_up)
        v_down = np.radians(self.v_fov_down)
        fov_mask = (elevation >= v_down) & (elevation <= v_up)
        local = local[fov_mask]
        dists = dists[fov_mask]
        azimuth = azimuth[fov_mask]
        elevation = elevation[fov_mask]
        if len(local) == 0:
            return

        # Angle-bin occlusion: nearest point per (azimuth, elevation) bin
        h_bin = ((np.degrees(azimuth) + 180.0) / self.h_resolution).astype(np.int32) % self.n_h_bins
        v_bin = ((np.degrees(elevation) - self.v_fov_down) / self.v_resolution).astype(np.int32)
        v_bin = np.clip(v_bin, 0, self.n_v_bins - 1)
        bin_idx = h_bin * self.n_v_bins + v_bin

        # Vectorized: for each bin keep closest point
        best_dist = np.full(self.n_total_bins, np.inf)
        best_idx = np.full(self.n_total_bins, -1, dtype=np.int64)
        for i in range(len(local)):
            b = bin_idx[i]
            if dists[i] < best_dist[b]:
                best_dist[b] = dists[i]
                best_idx[b] = i

        valid = best_idx >= 0
        result = local[best_idx[valid]]

        if len(result) == 0:
            return

        # Add noise
        if self.noise_stddev > 0:
            result = result + np.random.normal(0, self.noise_stddev, result.shape).astype(np.float32)

        # IPC's ROGMap::updateMap() expects point coordinates in the published frame.
        # For IPC we publish world-frame points directly; for RL/RViz we can still keep
        # the original body-frame cloud by leaving frame_id as base_link.
        if self.frame_id in ('world', 'map'):
            result = result @ self.rot_matrix.T + self.position

        self._publish_cloud(result)

    def _publish_cloud(self, points):
        msg = PointCloud2()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        self.cloud_pub.publish(msg)


if __name__ == '__main__':
    try:
        LiDARSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
