#!/usr/bin/env python3
"""Publish a static PCD map as PointCloud2 for RViz."""

import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.pcd_io import read_pcd_xyz, voxel_downsample


def _cloud_msg(points, frame_id):
    msg = PointCloud2()
    msg.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points)
    msg.is_dense = True
    msg.data = np.asarray(points, dtype=np.float32).tobytes()
    return msg


def main():
    rospy.init_node("pcd_map_publisher", anonymous=False)
    pcd_file = rospy.get_param("~pcd_file", "")
    frame_id = rospy.get_param("~frame_id", "world")
    topic = rospy.get_param("~topic", "/real_map/cloud")
    voxel_size = float(rospy.get_param("~voxel_size", 0.0))
    max_points = int(rospy.get_param("~max_points", 300000))
    z_min = float(rospy.get_param("~z_min", "-inf"))
    z_max = float(rospy.get_param("~z_max", "inf"))
    publish_rate = float(rospy.get_param("~publish_rate", 0.2))
    latch = bool(rospy.get_param("~latch", True))

    if not pcd_file:
        rospy.logfatal("[PCD Map Publisher] ~pcd_file is required")
        return
    if not os.path.exists(pcd_file):
        rospy.logfatal("[PCD Map Publisher] PCD file not found: %s", pcd_file)
        return
    if not z_min < z_max:
        rospy.logfatal(
            "[PCD Map Publisher] expected z_min < z_max, got %.3f >= %.3f",
            z_min,
            z_max,
        )
        return

    points = read_pcd_xyz(pcd_file)
    points = points[(points[:, 2] >= z_min) & (points[:, 2] < z_max)]
    if voxel_size > 0 and len(points) > 0:
        points = voxel_downsample(points, voxel_size)
    if max_points > 0 and len(points) > max_points:
        step = int(np.ceil(len(points) / float(max_points)))
        points = points[::step]

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=latch)
    rospy.loginfo(
        "[PCD Map Publisher] Loaded %d points from %s in z=[%.3f, %.3f); "
        "publishing %s in frame %s",
        len(points),
        pcd_file,
        z_min,
        z_max,
        topic,
        frame_id,
    )

    rate = rospy.Rate(publish_rate) if publish_rate > 0 else None
    while not rospy.is_shutdown():
        pub.publish(_cloud_msg(points, frame_id))
        if latch or rate is None:
            rospy.spin()
            return
        rate.sleep()


if __name__ == "__main__":
    main()
