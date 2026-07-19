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
from srlc_real.msg import ClearanceGuardStatus, ObstacleClearance

from srlc_real_deployment.recorder_snapshots import (
    ClearanceObservation,
    GuardObservation,
    RecorderObservationStore,
)


SHADOW_INTERVENTION_STATES = frozenset(
    ("PROXIMITY_HOLD", "PROXIMITY_ESCAPE", "COLLISION")
)


def _minimum_finite(samples, field, valid_field=None):
    values = (
        float(sample[field])
        for sample in samples
        if (valid_field is None or bool(sample[valid_field]))
        and math.isfinite(float(sample[field]))
    )
    return min(values, default=float("inf"))


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
        self.lidar_min_distance_topic = rospy.get_param(
            "~lidar_min_distance_topic", "/srlc/lidar/min_distance"
        )
        self.clearance_topic = rospy.get_param(
            "~clearance_topic", "/srlc/lidar/obstacle_clearance"
        )
        self.clearance_guard_status_topic = rospy.get_param(
            "~clearance_guard_status_topic",
            "/tunnel_nav/clearance_guard_status",
        )
        self.lidar_range_image_topic = rospy.get_param(
            "~lidar_range_image_topic", "/srlc/lidar/range_image"
        )
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
        self._observation_store = RecorderObservationStore()
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
        rospy.Subscriber(
            self.lidar_min_distance_topic,
            Float32,
            self._min_distance_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.clearance_topic,
            ObstacleClearance,
            self._clearance_cb,
            queue_size=50,
        )
        rospy.Subscriber(
            self.clearance_guard_status_topic,
            ClearanceGuardStatus,
            self._clearance_guard_status_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            self.lidar_range_image_topic,
            Float32MultiArray,
            self._range_image_cb,
            queue_size=1,
        )
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
        self._observation_store.replace_raw_center_distance(float(msg.data))

    def _clearance_cb(self, msg):
        clearance = ClearanceObservation(
            valid=bool(msg.valid),
            source_stamp=float(msg.header.stamp.to_sec()),
            source_frame_id=str(msg.header.frame_id),
            surface_clearance=float(msg.surface_clearance),
            center_distance=float(msg.center_distance),
            nearest_obstacle_point=(
                float(msg.nearest_obstacle_point.x),
                float(msg.nearest_obstacle_point.y),
                float(msg.nearest_obstacle_point.z),
            ),
            escape_direction=(
                float(msg.escape_direction.x),
                float(msg.escape_direction.y),
                float(msg.escape_direction.z),
            ),
        )
        self._observation_store.replace_clearance(clearance)

    def _clearance_guard_status_cb(self, msg):
        has_source = bool(
            msg.source_valid
            or msg.header.seq
            or msg.header.stamp.secs
            or msg.header.stamp.nsecs
            or msg.header.frame_id
        )
        source_stamp = (
            float(msg.header.stamp.to_sec())
            if has_source
            else float("nan")
        )
        guard = GuardObservation(
            source_valid=bool(msg.source_valid),
            source_stamp=source_stamp,
            source_frame_id=str(msg.header.frame_id),
            raw_state=str(msg.raw_state),
            effective_state=str(msg.effective_state),
        )
        self._observation_store.replace_guard(guard)

    def _status_cb(self, msg):
        self.status = msg.data

    def _range_image_cb(self, msg):
        expected = self.lidar_hbeams * self.lidar_vbeams
        if len(msg.data) != expected:
            return
        ranges = np.asarray(msg.data, dtype=np.float32).reshape(self.lidar_hbeams, self.lidar_vbeams)
        policy_norm = float(np.max(ranges))
        policy_min_distance = self.lidar_range * (1.0 - policy_norm)
        front_bins = [0, 1, self.lidar_hbeams - 2, self.lidar_hbeams - 1]
        front_norm = float(np.max(ranges[front_bins, :]))
        front_distance = self.lidar_range * (1.0 - front_norm)
        self._observation_store.replace_policy_ranges(
            policy_min_distance,
            front_distance,
        )

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
        observations = self._observation_store.read()
        clearance = observations.clearance
        guard = observations.guard
        clearance_source_age = float("inf")
        if math.isfinite(clearance.source_stamp):
            clearance_source_age = float(
                now.to_sec() - clearance.source_stamp
            )
        guard_source_age = float("inf")
        if math.isfinite(guard.source_stamp):
            guard_source_age = float(now.to_sec() - guard.source_stamp)
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
                "raw_center_distance": float(
                    observations.raw_center_distance
                ),
                "policy_min_distance": float(
                    observations.policy_min_distance
                ),
                "model_min_distance": float(
                    observations.policy_min_distance
                ),
                "clearance_valid": bool(clearance.valid),
                "surface_clearance": float(clearance.surface_clearance),
                "clearance_center_distance": float(clearance.center_distance),
                "clearance_source_stamp": float(clearance.source_stamp),
                "clearance_source_age": clearance_source_age,
                "clearance_source_frame_id": clearance.source_frame_id,
                "nearest_obstacle_point": list(
                    clearance.nearest_obstacle_point
                ),
                "escape_direction": list(clearance.escape_direction),
                "guard_source_valid": bool(guard.source_valid),
                "guard_source_stamp": float(guard.source_stamp),
                "guard_source_age": guard_source_age,
                "guard_source_frame_id": guard.source_frame_id,
                "raw_guard_state": guard.raw_state,
                "effective_guard_state": guard.effective_state,
                "shadow_guard_state": guard.raw_state,
                # Compatibility fields mirror the combined status's raw
                # decision even while effective enforcement stays NORMAL.
                "shadow_decision": guard.raw_state,
                "shadow_would_intervene": (
                    guard.raw_state in SHADOW_INTERVENTION_STATES
                ),
                "would_intervene": (
                    guard.raw_state in SHADOW_INTERVENTION_STATES
                ),
                "front_distance": float(observations.front_distance),
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
            "min_raw_center_distance": _minimum_finite(
                self.samples,
                "raw_center_distance",
            ),
            "min_policy_distance": _minimum_finite(
                self.samples,
                "policy_min_distance",
            ),
            "min_surface_clearance": _minimum_finite(
                self.samples,
                "surface_clearance",
                valid_field="clearance_valid",
            ),
            "min_front_distance": min((s["front_distance"] for s in self.samples), default=float("inf")),
            "max_abs_setpoint_xy": max(
                (math.hypot(s["setpoint_velocity"][0], s["setpoint_velocity"][1]) for s in self.samples),
                default=0.0,
            ),
            "latest_guard_state": (
                self.samples[-1]["effective_guard_state"]
                if self.samples
                else "UNKNOWN"
            ),
            "latest_shadow_decision": (
                self.samples[-1]["shadow_decision"]
                if self.samples
                else "UNKNOWN"
            ),
            "latest_shadow_would_intervene": (
                self.samples[-1]["shadow_would_intervene"]
                if self.samples
                else False
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
