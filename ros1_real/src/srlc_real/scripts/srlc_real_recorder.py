#!/usr/bin/env python3
"""Recorder for the SRLC MAVROS topic contract."""

import json
import math
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import ExtendedState, PositionTarget, RCIn, State
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class SrlcRealRecorder:
    def __init__(self):
        rospy.init_node("srlc_real_recorder", anonymous=False)

        self.output_dir = rospy.get_param("~output_dir", "/root/real_outputs")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_hbeams = int(rospy.get_param("~lidar_hbeams", 36))
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.run_id = rospy.get_param("~run_id", "srlc_real")
        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        self.px4_local_odom_topic = rospy.get_param(
            "~px4_local_odom_topic", "/mavros/local_position/odom"
        )
        self.px4_local_velocity_topic = rospy.get_param(
            "~px4_local_velocity_topic", "/mavros/local_position/velocity_local"
        )
        self.rc_topic = rospy.get_param("~rc_topic", "/mavros/rc/in")
        self.human_action_topic = rospy.get_param("~human_action_topic", "/srlc/human_action")
        self.control_mode_topic = rospy.get_param("~control_mode_topic", "/tunnel_nav/control_mode")
        self.lifecycle_state_topic = rospy.get_param(
            "~lifecycle_state_topic", "/tunnel_nav/lifecycle_state"
        )
        self.effective_mode_topic = rospy.get_param(
            "~effective_mode_topic", "/tunnel_nav/effective_mode"
        )
        self.policy_active_topic = rospy.get_param(
            "~policy_active_topic", "/tunnel_nav/policy_active"
        )
        self.session_consumed_topic = rospy.get_param(
            "~session_consumed_topic", "/tunnel_nav/session_consumed"
        )
        self.fault_reason_topic = rospy.get_param(
            "~fault_reason_topic", "/tunnel_nav/fault_reason"
        )
        self.policy_cmd_topic = rospy.get_param("~policy_cmd_topic", "/tunnel_nav/policy_cmd")
        self.setpoint_raw_topic = rospy.get_param("~setpoint_raw_topic", "/mavros/setpoint_raw/local")
        self.lidar_min_distance_topic = rospy.get_param("~lidar_min_distance_topic", "/srlc/lidar/min_distance")
        self.lidar_safety_distance_topic = rospy.get_param(
            "~lidar_safety_distance_topic", "/srlc/lidar/min_safety_distance"
        )
        self.lidar_range_image_topic = rospy.get_param("~lidar_range_image_topic", "/srlc/lidar/range_image")
        self.status_topic = rospy.get_param("~status_topic", "/tunnel_nav/status")

        self.odom = None
        self.px4_local_odom = None
        self.px4_local_velocity = None
        self.rc = None
        self.human_cmd = None
        self.policy_cmd = None
        self.setpoint = None
        self.last_setpoint_time = None
        self.control_mode = ""
        self.lifecycle_state = ""
        self.effective_mode = ""
        self.policy_active = False
        self.session_consumed = False
        self.fault_reason = ""
        self.mavros_state = None
        self.extended_state = None
        self.min_distance = float("inf")
        self.safety_distance = float("inf")
        self.front_distance = float("inf")
        self.status = ""
        self.start_time = rospy.Time.now()
        self.initial_position = None
        self.initial_px4_local_position = None
        self.saved = False
        self.samples = []

        os.makedirs(self.output_dir, exist_ok=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(
            self.px4_local_odom_topic,
            Odometry,
            self._px4_local_odom_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.px4_local_velocity_topic,
            TwistStamped,
            self._px4_local_velocity_cb,
            queue_size=1,
        )
        rospy.Subscriber(self.rc_topic, RCIn, self._rc_cb, queue_size=1)
        rospy.Subscriber(self.human_action_topic, TwistStamped, self._human_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self._mavros_state_cb, queue_size=1)
        rospy.Subscriber(
            "/mavros/extended_state",
            ExtendedState,
            self._extended_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(self.control_mode_topic, String, self._control_mode_cb, queue_size=1)
        rospy.Subscriber(
            self.lifecycle_state_topic, String, self._lifecycle_state_cb, queue_size=1
        )
        rospy.Subscriber(
            self.effective_mode_topic, String, self._effective_mode_cb, queue_size=1
        )
        rospy.Subscriber(
            self.policy_active_topic, Bool, self._policy_active_cb, queue_size=1
        )
        rospy.Subscriber(
            self.session_consumed_topic, Bool, self._session_consumed_cb, queue_size=1
        )
        rospy.Subscriber(
            self.fault_reason_topic, String, self._fault_reason_cb, queue_size=1
        )
        rospy.Subscriber(self.policy_cmd_topic, TwistStamped, self._policy_cb, queue_size=1)
        rospy.Subscriber(self.setpoint_raw_topic, PositionTarget, self._setpoint_cb, queue_size=1)
        rospy.Subscriber(self.lidar_min_distance_topic, Float32, self._min_distance_cb, queue_size=1)
        rospy.Subscriber(
            self.lidar_safety_distance_topic,
            Float32,
            self._safety_distance_cb,
            queue_size=1,
        )
        rospy.Subscriber(self.lidar_range_image_topic, Float32MultiArray, self._range_image_cb, queue_size=1)
        rospy.Subscriber(self.status_topic, String, self._status_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._record_cb)
        rospy.on_shutdown(self._save)
        rospy.loginfo("[SRLC Recorder] Recording real-flight data to %s", self.output_dir)

    def _odom_cb(self, msg):
        self.odom = msg
        if self.initial_position is None:
            p = msg.pose.pose.position
            self.initial_position = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _px4_local_odom_cb(self, msg):
        self.px4_local_odom = msg
        if self.initial_px4_local_position is None:
            p = msg.pose.pose.position
            self.initial_px4_local_position = np.array(
                [p.x, p.y, p.z], dtype=np.float32
            )

    def _px4_local_velocity_cb(self, msg):
        self.px4_local_velocity = msg

    def _rc_cb(self, msg):
        self.rc = msg

    def _human_cb(self, msg):
        self.human_cmd = msg

    def _mavros_state_cb(self, msg):
        self.mavros_state = msg

    def _extended_state_cb(self, msg):
        self.extended_state = msg

    def _control_mode_cb(self, msg):
        self.control_mode = str(msg.data)

    def _lifecycle_state_cb(self, msg):
        self.lifecycle_state = str(msg.data)

    def _effective_mode_cb(self, msg):
        self.effective_mode = str(msg.data)

    def _policy_active_cb(self, msg):
        self.policy_active = bool(msg.data)

    def _session_consumed_cb(self, msg):
        self.session_consumed = bool(msg.data)

    def _fault_reason_cb(self, msg):
        self.fault_reason = str(msg.data)

    def _policy_cb(self, msg):
        self.policy_cmd = msg

    def _setpoint_cb(self, msg):
        self.setpoint = msg
        self.last_setpoint_time = rospy.Time.now()

    def _min_distance_cb(self, msg):
        self.min_distance = float(msg.data)

    def _safety_distance_cb(self, msg):
        self.safety_distance = float(msg.data)

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
        setpoint_type_mask = -1
        setpoint_age = float("inf")
        if self.setpoint is not None:
            setpoint_v = [
                float(self.setpoint.velocity.x),
                float(self.setpoint.velocity.y),
                float(self.setpoint.velocity.z),
            ]
            setpoint_z = float(self.setpoint.position.z)
            setpoint_type_mask = int(self.setpoint.type_mask)
            if self.last_setpoint_time is not None:
                setpoint_age = float((now - self.last_setpoint_time).to_sec())

        mavros_mode = self.mavros_state.mode if self.mavros_state is not None else ""
        armed = bool(self.mavros_state.armed) if self.mavros_state is not None else False
        landed_state = (
            int(self.extended_state.landed_state)
            if self.extended_state is not None
            else int(ExtendedState.LANDED_STATE_UNDEFINED)
        )
        px4_local_position = [float("nan"), float("nan"), float("nan")]
        px4_local_position_rel = [float("nan"), float("nan"), float("nan")]
        if self.px4_local_odom is not None:
            px4_p = self.px4_local_odom.pose.pose.position
            px4_position_np = np.array(
                [px4_p.x, px4_p.y, px4_p.z], dtype=np.float32
            )
            px4_local_position = px4_position_np.tolist()
            if self.initial_px4_local_position is not None:
                px4_local_position_rel = (
                    px4_position_np - self.initial_px4_local_position
                ).tolist()

        self.samples.append(
            {
                "t": float((now - self.start_time).to_sec()),
                "position": position.tolist(),
                "position_rel": rel.tolist(),
                "velocity": [float(v.x), float(v.y), float(v.z)],
                "px4_local_position": px4_local_position,
                "px4_local_position_rel": px4_local_position_rel,
                "px4_local_velocity": self._twist_vec(self.px4_local_velocity),
                "yaw": float(self._yaw_from_odom(self.odom)),
                "rc": list(self.rc.channels) if self.rc is not None else [],
                "control_mode": self.control_mode,
                "assist_enabled": self.control_mode.upper() == "ASSIST",
                "lifecycle_state": self.lifecycle_state,
                "effective_mode": self.effective_mode,
                "policy_active": bool(self.policy_active),
                "session_consumed": bool(self.session_consumed),
                "fault_reason": self.fault_reason,
                "mavros_mode": str(mavros_mode),
                "armed": armed,
                "landed_state": landed_state,
                "human_action": self._twist_vec(self.human_cmd),
                "policy_cmd": self._twist_vec(self.policy_cmd),
                "setpoint_velocity": setpoint_v,
                "setpoint_z": setpoint_z,
                "setpoint_type_mask": setpoint_type_mask,
                "setpoint_age": setpoint_age,
                "min_distance": float(self.min_distance),
                "safety_distance": float(self.safety_distance),
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
            "min_safety_distance": min(
                (s["safety_distance"] for s in self.samples),
                default=float("inf"),
            ),
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
        SrlcRealRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
