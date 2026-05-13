#!/usr/bin/env python3
"""Dry-run recorder for the SRLC MAVROS topic contract."""

import json
import math
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PositionTarget, RCIn
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class SrlcDryRunRecorder:
    def __init__(self):
        rospy.init_node("srlc_dry_run_recorder", anonymous=False)

        self.output_dir = rospy.get_param("~output_dir", "/tmp/srlc_dry_run")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_hbeams = int(rospy.get_param("~lidar_hbeams", 36))
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.run_id = rospy.get_param("~run_id", "srlc_dry_run")

        self.odom = None
        self.rc = None
        self.human_cmd = None
        self.policy_cmd = None
        self.setpoint = None
        self.assist_enabled = False
        self.min_distance = float("inf")
        self.front_distance = float("inf")
        self.status = ""
        self.start_time = rospy.Time.now()
        self.initial_position = None
        self.saved = False
        self.samples = []

        os.makedirs(self.output_dir, exist_ok=True)

        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/mavros/rc/in", RCIn, self._rc_cb, queue_size=1)
        rospy.Subscriber("/srlc/human_action", TwistStamped, self._human_cb, queue_size=1)
        rospy.Subscriber("/srlc/assist_enable", Bool, self._assist_cb, queue_size=1)
        rospy.Subscriber("/tunnel_nav/policy_cmd", TwistStamped, self._policy_cb, queue_size=1)
        rospy.Subscriber("/mavros/setpoint_raw/local", PositionTarget, self._setpoint_cb, queue_size=1)
        rospy.Subscriber("/srlc/lidar/min_distance", Float32, self._min_distance_cb, queue_size=1)
        rospy.Subscriber("/srlc/lidar/range_image", Float32MultiArray, self._range_image_cb, queue_size=1)
        rospy.Subscriber("/tunnel_nav/status", String, self._status_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._record_cb)
        rospy.on_shutdown(self._save)
        rospy.loginfo("[SRLC Recorder] Recording dry-run data to %s", self.output_dir)

    def _odom_cb(self, msg):
        self.odom = msg
        if self.initial_position is None:
            p = msg.pose.pose.position
            self.initial_position = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _rc_cb(self, msg):
        self.rc = msg

    def _human_cb(self, msg):
        self.human_cmd = msg

    def _assist_cb(self, msg):
        self.assist_enabled = bool(msg.data)

    def _policy_cb(self, msg):
        self.policy_cmd = msg

    def _setpoint_cb(self, msg):
        self.setpoint = msg

    def _min_distance_cb(self, msg):
        self.min_distance = float(msg.data)

    def _status_cb(self, msg):
        self.status = msg.data

    def _range_image_cb(self, msg):
        expected = self.lidar_hbeams * self.lidar_vbeams
        if len(msg.data) != expected:
            return
        ranges = np.asarray(msg.data, dtype=np.float32).reshape(self.lidar_hbeams, self.lidar_vbeams)
        front_bins = [0, 1, self.lidar_hbeams - 2, self.lidar_hbeams - 1]
        front_norm = float(np.max(ranges[front_bins, :]))
        self.front_distance = self.lidar_range * (1.0 - front_norm)

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _twist_vec(msg):
        if msg is None:
            return [0.0, 0.0, 0.0]
        v = msg.twist.linear
        return [float(v.x), float(v.y), float(v.z)]

    def _record_cb(self, _event):
        if self.odom is None:
            return
        now = rospy.Time.now()
        p = self.odom.pose.pose.position
        v = self.odom.twist.twist.linear
        position = np.array([p.x, p.y, p.z], dtype=np.float32)
        rel = position - self.initial_position if self.initial_position is not None else position
        setpoint_v = [0.0, 0.0, 0.0]
        setpoint_z = float("nan")
        if self.setpoint is not None:
            setpoint_v = [
                float(self.setpoint.velocity.x),
                float(self.setpoint.velocity.y),
                float(self.setpoint.velocity.z),
            ]
            setpoint_z = float(self.setpoint.position.z)

        self.samples.append(
            {
                "t": float((now - self.start_time).to_sec()),
                "position": position.tolist(),
                "position_rel": rel.tolist(),
                "velocity": [float(v.x), float(v.y), float(v.z)],
                "yaw": float(self._yaw_from_odom(self.odom)),
                "rc": list(self.rc.channels) if self.rc is not None else [],
                "assist_enabled": bool(self.assist_enabled),
                "human_action": self._twist_vec(self.human_cmd),
                "policy_cmd": self._twist_vec(self.policy_cmd),
                "setpoint_velocity": setpoint_v,
                "setpoint_z": setpoint_z,
                "min_distance": float(self.min_distance),
                "front_distance": float(self.front_distance),
                "status": self.status,
            }
        )

    def _save(self):
        if self.saved:
            return
        self.saved = True
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{self.run_id}_{stamp}"
        json_path = os.path.join(self.output_dir, stem + ".json")
        npz_path = os.path.join(self.output_dir, stem + ".npz")

        summary = {
            "run_id": self.run_id,
            "samples": len(self.samples),
            "min_distance": min((s["min_distance"] for s in self.samples), default=float("inf")),
            "min_front_distance": min((s["front_distance"] for s in self.samples), default=float("inf")),
            "max_abs_setpoint_xy": max(
                (math.hypot(s["setpoint_velocity"][0], s["setpoint_velocity"][1]) for s in self.samples),
                default=0.0,
            ),
            "latest_status": self.samples[-1]["status"] if self.samples else "",
            "data_file": os.path.basename(npz_path),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "samples": self.samples}, handle, indent=2)
        np.savez_compressed(npz_path, samples=np.array(self.samples, dtype=object), summary=summary)
        rospy.loginfo("[SRLC Recorder] Saved %d samples to %s", len(self.samples), json_path)


if __name__ == "__main__":
    try:
        SrlcDryRunRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
