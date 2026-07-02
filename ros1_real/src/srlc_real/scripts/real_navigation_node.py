#!/usr/bin/env python3
"""Real PX4/Nokov SRLC navigation node.

This is the real-flight-only version of the tunnel SRLC navigator.  It consumes
Nokov odometry, MAVROS RC-derived human action, and PCD LiDAR topics, then
publishes PX4 raw local setpoints.
"""

import math
import os
import sys
import threading

import numpy as np
import rospy
import tf.transformations
import torch

from geometry_msgs.msg import Point, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.policy_net import TunnelPolicyNet  # noqa: E402


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class RealConfig:
    def __init__(self):
        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.lidar_hres = float(rospy.get_param("~lidar_hres", 10.0))
        self.lidar_hbeams = int(360.0 / self.lidar_hres)
        self.lidar_range_image_topic = rospy.get_param(
            "~lidar_range_image_topic", "/srlc/lidar/range_image"
        )
        self.lidar_min_distance_topic = rospy.get_param(
            "~lidar_min_distance_topic", "/srlc/lidar/min_distance"
        )
        self.lidar_timeout = float(rospy.get_param("~lidar_timeout", 0.3))

        self.action_limit = float(rospy.get_param("~action_limit", 2.0))
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "")
        self.policy_mode = str(rospy.get_param("~policy_mode", "residual")).lower()
        self.device = self._resolve_device(rospy.get_param("~device", "cpu"))
        self.deterministic = _param_bool(rospy.get_param("~deterministic", True))
        self.policy_debug = _param_bool(rospy.get_param("~policy_debug", False))

        self.control_freq = float(rospy.get_param("~control_freq", 20.0))
        self.height_control = _param_bool(rospy.get_param("~height_control", False))
        self.lock_z_control = _param_bool(rospy.get_param("~lock_z_control", True))
        self.takeoff_height = float(rospy.get_param("~takeoff_height", 1.2))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.3))
        self.odom_topic = rospy.get_param("~odom_topic", "/nokov/local_position/odom")
        self.setpoint_raw_topic = rospy.get_param(
            "~setpoint_raw_topic", "/mavros/setpoint_raw/local"
        )
        self.require_offboard = _param_bool(rospy.get_param("~require_offboard", True))
        self.real_auto_takeoff_on_offboard = _param_bool(
            rospy.get_param("~real_auto_takeoff_on_offboard", True)
        )
        self.auto_arm_on_offboard = _param_bool(
            rospy.get_param("~auto_arm_on_offboard", True)
        )
        self.post_takeoff_mode = str(rospy.get_param("~post_takeoff_mode", "direct")).lower()
        self.post_takeoff_mode_delay = float(rospy.get_param("~post_takeoff_mode_delay", 2.0))
        self.takeoff_reached_tolerance = float(
            rospy.get_param("~takeoff_reached_tolerance", 0.1)
        )
        self.on_offboard_loss = str(rospy.get_param("~on_offboard_loss", "stream_hold")).lower()
        self.hold_on_stop = _param_bool(rospy.get_param("~hold_on_stop", False))
        self.estop_hold_mode = str(rospy.get_param("~estop_hold_mode", "AUTO.LOITER"))
        self.estop_fallback_mode = str(rospy.get_param("~estop_fallback_mode", "POSCTL"))
        self.hold_verify_timeout = float(rospy.get_param("~hold_verify_timeout", 1.0))
        self.max_xy_speed_real = float(rospy.get_param("~max_xy_speed_real", 0.5))
        self.max_z_speed_real = float(rospy.get_param("~max_z_speed_real", 0.3))
        self.min_altitude = float(rospy.get_param("~min_altitude", 0.5))
        self.max_altitude = float(rospy.get_param("~max_altitude", 3.0))
        self.geofence_x = [float(v) for v in rospy.get_param("~geofence_x", [])]
        self.geofence_y = [float(v) for v in rospy.get_param("~geofence_y", [])]

        self.human_action_topic = rospy.get_param("~human_action_topic", "/srlc/human_action")
        self.human_action_timeout = float(rospy.get_param("~human_action_timeout", 0.3))
        self.assist_input_deadzone_norm = float(
            rospy.get_param("~assist_input_deadzone_norm", 0.05)
        )

        self.enable_safety_stop = _param_bool(rospy.get_param("~enable_safety_stop", True))
        self.enable_collision_detection = _param_bool(
            rospy.get_param("~enable_collision_detection", True)
        )
        self.safety_min_dist = float(rospy.get_param("~safety_min_dist", 0.25))
        self.collision_dist = float(rospy.get_param("~collision_dist", 0.15))
        self.safety_start_takeoff_delta = float(
            rospy.get_param("~safety_start_takeoff_delta", 0.3)
        )

    @staticmethod
    def _resolve_device(requested_device):
        device = str(requested_device).strip()
        if not device.lower().startswith("cuda"):
            return device
        if not torch.cuda.is_available():
            rospy.logwarn(
                "[SRLCReal] Requested device %s, but CUDA is unavailable. Falling back to cpu.",
                device,
            )
            return "cpu"
        return device


class RealNavigator:
    def __init__(self):
        rospy.init_node("srlc_real_navigator", anonymous=False)
        self.cfg = RealConfig()
        self._validate_config()

        rospy.loginfo("[SRLCReal] Loading checkpoint: %s", self.cfg.checkpoint_path)
        self.policy = TunnelPolicyNet.from_checkpoint(
            self.cfg.checkpoint_path,
            action_limit=self.cfg.action_limit,
            min_concentration=2.0,
            device=self.cfg.device,
            policy_mode=self.cfg.policy_mode,
        )
        self.policy.eval()
        self.policy.debug = bool(self.cfg.policy_debug)
        rospy.loginfo("[SRLCReal] Policy loaded on %s.", self.cfg.device)

        self.odom = None
        self.odom_received = False
        self.last_odom_time = None
        self.mavros_state = None
        self.topic_lidar_np = None
        self.last_lidar_time = None
        self.last_min_dist_time = None
        self.min_dist = float("inf")
        self._topic_human_cmd = np.zeros(3, dtype=np.float32)
        self._topic_human_lock = threading.Lock()
        self.last_human_action_time = None

        self.control_mode = self.cfg.post_takeoff_mode.upper()
        self.assist_enabled = self.control_mode == "ASSIST"
        self.real_lifecycle_state = "WAIT_OFFBOARD"
        self._takeoff_hold_z = None
        self._takeoff_xy = None
        self._takeoff_settle_start = None
        self._last_arm_request = rospy.Time(0)
        self._arm_request_failed = False
        self._last_hold_mode_request = rospy.Time(0)
        self._hold_mode_request_time = None
        self._hold_mode_requested = None
        self._hold_fallback_requested = False

        self.ready = False
        self.policy_active = False
        self.z_policy_active = False
        self.safety_stop = False
        self.collision = False
        self.initial_z = None
        self.safety_airborne = False

        self._setup_ros()
        self._publish_mode_state()
        self._log_config()

        self.safety_timer = rospy.Timer(rospy.Duration(0.1), self._safety_timer_cb)
        self.diagnostic_timer = rospy.Timer(rospy.Duration(0.5), self._diagnostic_timer_cb)
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.cfg.control_freq), self._control_callback
        )

    def _validate_config(self):
        if self.cfg.post_takeoff_mode not in {"direct", "assist"}:
            raise ValueError("post_takeoff_mode must be 'direct' or 'assist'")
        if self.cfg.on_offboard_loss != "stream_hold":
            raise ValueError("on_offboard_loss must be 'stream_hold'")
        if self.cfg.policy_mode not in {"residual", "direct"}:
            raise ValueError("policy_mode must be 'residual' or 'direct'")
        if not self.cfg.checkpoint_path:
            raise ValueError("~checkpoint_path is required")

    def _log_config(self):
        rospy.loginfo("[SRLCReal] odom=%s setpoint=%s", self.cfg.odom_topic, self.cfg.setpoint_raw_topic)
        rospy.loginfo(
            "[SRLCReal] mode=%s takeoff=%.2fm auto_arm=%s require_offboard=%s",
            self.control_mode,
            self.cfg.takeoff_height,
            self.cfg.auto_arm_on_offboard,
            self.cfg.require_offboard,
        )
        rospy.loginfo(
            "[SRLCReal] lidar=%dx%d range=%.2fm topics=(%s, %s)",
            self.cfg.lidar_hbeams,
            self.cfg.lidar_vbeams,
            self.cfg.lidar_range,
            self.cfg.lidar_range_image_topic,
            self.cfg.lidar_min_distance_topic,
        )

    def _setup_ros(self):
        self.odom_sub = rospy.Subscriber(self.cfg.odom_topic, Odometry, self._odom_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self._mavros_state_cb, queue_size=1)
        self.human_action_sub = rospy.Subscriber(
            self.cfg.human_action_topic, TwistStamped, self._human_action_cb, queue_size=1
        )
        self.lidar_range_sub = rospy.Subscriber(
            self.cfg.lidar_range_image_topic, Float32MultiArray, self._lidar_range_cb, queue_size=1
        )
        self.lidar_min_dist_sub = rospy.Subscriber(
            self.cfg.lidar_min_distance_topic, Float32, self._lidar_min_dist_cb, queue_size=1
        )

        self.action_pub = rospy.Publisher(self.cfg.setpoint_raw_topic, PositionTarget, queue_size=10)
        self.status_pub = rospy.Publisher("/tunnel_nav/status", String, queue_size=2)
        self.lifecycle_state_pub = rospy.Publisher(
            "/tunnel_nav/lifecycle_state", String, queue_size=2
        )
        self.control_mode_pub = rospy.Publisher("/tunnel_nav/control_mode", String, queue_size=2)
        self.collision_pub = rospy.Publisher(
            "/tunnel_nav/collision", Bool, queue_size=2, latch=True
        )
        self.policy_cmd_pub = rospy.Publisher("/tunnel_nav/policy_cmd", TwistStamped, queue_size=2)
        self.policy_active_pub = rospy.Publisher(
            "/tunnel_nav/policy_active", Bool, queue_size=2
        )
        self.z_policy_active_pub = rospy.Publisher(
            "/tunnel_nav/z_policy_active", Bool, queue_size=2
        )
        self.human_cmd_pub = rospy.Publisher(
            "/experiment_control/human_cmd", TwistStamped, queue_size=2
        )
        self.cmd_vis_pub = rospy.Publisher("/tunnel_nav/cmd_vel_vis", MarkerArray, queue_size=2)
        self.human_cmd_vis_pub = rospy.Publisher(
            "/tunnel_nav/human_cmd_vis", MarkerArray, queue_size=2
        )

        self.collision_pub.publish(Bool(data=False))
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    def _odom_cb(self, msg):
        self.odom = msg
        self.odom_received = True
        self.last_odom_time = rospy.Time.now()
        if self.initial_z is None:
            self.initial_z = float(msg.pose.pose.position.z)

    def _mavros_state_cb(self, msg):
        self.mavros_state = msg

    def _human_action_cb(self, msg):
        with self._topic_human_lock:
            self._topic_human_cmd[0] = msg.twist.linear.x
            self._topic_human_cmd[1] = msg.twist.linear.y
            self._topic_human_cmd[2] = msg.twist.linear.z
        self.last_human_action_time = rospy.Time.now()

    def _lidar_range_cb(self, msg):
        expected = self.cfg.lidar_hbeams * self.cfg.lidar_vbeams
        if len(msg.data) != expected:
            rospy.logwarn_throttle(
                2.0,
                "[SRLCReal] Ignoring LiDAR range image with %d values; expected %d",
                len(msg.data),
                expected,
            )
            return
        lidar_np = np.asarray(msg.data, dtype=np.float32).reshape(
            1, 1, self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
        )
        self.topic_lidar_np = np.clip(lidar_np, 0.0, 1.0)
        self.last_lidar_time = rospy.Time.now()
        self.ready = True

    def _lidar_min_dist_cb(self, msg):
        self.min_dist = float(msg.data)
        self.last_min_dist_time = rospy.Time.now()

    def _diagnostic_timer_cb(self, _event):
        self._publish_mode_state()
        if self.odom_received:
            return
        self.policy_active_pub.publish(Bool(data=False))
        self.status_pub.publish(
            String(data=f"cmd=[0,0,0] | {self.real_lifecycle_state} | NO_ODOM")
        )

    def _build_obs(self):
        odom = self.odom
        q_ros = odom.pose.pose.orientation
        quat_np = np.array([[q_ros.w, q_ros.x, q_ros.y, q_ros.z]], dtype=np.float32)
        quat = torch.from_numpy(quat_np).to(device=self.cfg.device)

        vel_b_np = np.array(
            [
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        ang_vel_b_np = np.array(
            [
                odom.twist.twist.angular.x,
                odom.twist.twist.angular.y,
                odom.twist.twist.angular.z,
            ],
            dtype=np.float32,
        )
        vel_b = torch.from_numpy(vel_b_np).unsqueeze(0).to(device=self.cfg.device)
        ang_vel_b = torch.from_numpy(ang_vel_b_np).unsqueeze(0).to(device=self.cfg.device)
        state = torch.cat([vel_b, ang_vel_b, quat], dim=-1)

        human_action_np = self._current_topic_human_cmd()
        human_action = torch.from_numpy(human_action_np).unsqueeze(0).to(device=self.cfg.device)

        if self.topic_lidar_np is None:
            lidar_scan = torch.zeros(
                1,
                1,
                self.cfg.lidar_hbeams,
                self.cfg.lidar_vbeams,
                device=self.cfg.device,
                dtype=torch.float32,
            )
        else:
            lidar_scan = torch.from_numpy(self.topic_lidar_np.copy()).to(device=self.cfg.device)

        return state, human_action, lidar_scan

    def _real_common_gate_ok(self):
        now = rospy.Time.now()
        if self.mavros_state is None or not self.mavros_state.connected:
            return False, "MAVROS_NOT_CONNECTED"
        if self.cfg.require_offboard and self.mavros_state.mode != "OFFBOARD":
            return False, "PX4_NOT_OFFBOARD"
        if self.last_odom_time is None:
            return False, "NO_ODOM"
        if (now - self.last_odom_time).to_sec() > self.cfg.odom_timeout:
            return False, "ODOM_TIMEOUT"
        if self.last_human_action_time is None:
            return False, "NO_RC_ACTION"
        if (now - self.last_human_action_time).to_sec() > self.cfg.human_action_timeout:
            return False, "RC_ACTION_TIMEOUT"
        if self.odom is not None:
            pos = self.odom.pose.pose.position
            if pos.z < self.cfg.min_altitude:
                return False, "LOW_ALTITUDE"
            if pos.z > self.cfg.max_altitude:
                return False, "HIGH_ALTITUDE"
            if len(self.cfg.geofence_x) == 2:
                if pos.x < self.cfg.geofence_x[0] or pos.x > self.cfg.geofence_x[1]:
                    return False, "GEOFENCE_X"
            if len(self.cfg.geofence_y) == 2:
                if pos.y < self.cfg.geofence_y[0] or pos.y > self.cfg.geofence_y[1]:
                    return False, "GEOFENCE_Y"
        return True, "OK"

    def _real_lidar_gate_ok(self):
        now = rospy.Time.now()
        if self.last_lidar_time is None:
            return False, "NO_LIDAR"
        if (now - self.last_lidar_time).to_sec() > self.cfg.lidar_timeout:
            return False, "LIDAR_TIMEOUT"
        if self.last_min_dist_time is None:
            return False, "NO_LIDAR_MIN_DISTANCE"
        if (now - self.last_min_dist_time).to_sec() > self.cfg.lidar_timeout:
            return False, "LIDAR_MIN_DISTANCE_TIMEOUT"
        if not self.ready:
            return False, "NOT_READY"
        return True, "OK"

    def _real_lifecycle_preconditions_ok(self):
        now = rospy.Time.now()
        if self.mavros_state is None or not self.mavros_state.connected:
            return False, "MAVROS_NOT_CONNECTED"
        if self.cfg.require_offboard and self.mavros_state.mode != "OFFBOARD":
            return False, "PX4_NOT_OFFBOARD"
        if self.last_odom_time is None:
            return False, "NO_ODOM"
        if (now - self.last_odom_time).to_sec() > self.cfg.odom_timeout:
            return False, "ODOM_TIMEOUT"
        if self.last_human_action_time is None:
            return False, "NO_RC_ACTION"
        if (now - self.last_human_action_time).to_sec() > self.cfg.human_action_timeout:
            return False, "RC_ACTION_TIMEOUT"
        if self.odom is not None:
            pos = self.odom.pose.pose.position
            if len(self.cfg.geofence_x) == 2:
                if pos.x < self.cfg.geofence_x[0] or pos.x > self.cfg.geofence_x[1]:
                    return False, "GEOFENCE_X"
            if len(self.cfg.geofence_y) == 2:
                if pos.y < self.cfg.geofence_y[0] or pos.y > self.cfg.geofence_y[1]:
                    return False, "GEOFENCE_Y"
        return True, "OK"

    def _publish_mode_state(self):
        self.lifecycle_state_pub.publish(String(data=str(self.real_lifecycle_state)))
        self.control_mode_pub.publish(String(data=str(self.control_mode)))

    def _set_control_mode(self, mode):
        self.control_mode = str(mode).upper()
        self.assist_enabled = self.control_mode == "ASSIST"
        self._publish_mode_state()

    def _request_arm_on_offboard(self):
        if not self.cfg.auto_arm_on_offboard:
            return
        if self.mavros_state is not None and self.mavros_state.armed:
            return
        now = rospy.Time.now()
        if (now - self._last_arm_request).to_sec() < 1.0:
            return
        self._last_arm_request = now
        try:
            resp = self.arming_client(CommandBoolRequest(value=True))
            self._arm_request_failed = not bool(resp.success)
            if not resp.success:
                rospy.logwarn_throttle(2.0, "[SRLCReal] PX4 arming request rejected")
        except rospy.ServiceException as exc:
            self._arm_request_failed = True
            rospy.logwarn_throttle(2.0, "[SRLCReal] PX4 arming request failed: %s", exc)

    def _start_real_takeoff(self):
        pos = self.odom.pose.pose.position
        self._takeoff_xy = (float(pos.x), float(pos.y))
        self._takeoff_hold_z = float(pos.z) + float(self.cfg.takeoff_height)
        self._takeoff_settle_start = None
        self.initial_z = float(pos.z)
        self.safety_airborne = False
        self.real_lifecycle_state = "TAKEOFF_CLIMB"
        self._publish_mode_state()
        rospy.loginfo(
            "[SRLCReal] Takeoff started at x=%.2f y=%.2f z=%.2f target_z=%.2f",
            pos.x,
            pos.y,
            pos.z,
            self._takeoff_hold_z,
        )

    def _resume_real_takeoff_target(self):
        if self._takeoff_xy is None or self._takeoff_hold_z is None:
            self._start_real_takeoff()
            return
        self._takeoff_settle_start = None
        self.real_lifecycle_state = "TAKEOFF_CLIMB"
        self._publish_mode_state()
        rospy.loginfo("[SRLCReal] Resuming takeoff target z=%.2f", self._takeoff_hold_z)

    def _px4_altitude_hold_target(self):
        if self._takeoff_hold_z is not None:
            return float(self._takeoff_hold_z)
        if self.odom is not None:
            return float(self.odom.pose.pose.position.z)
        return float(self.cfg.takeoff_height)

    def _takeoff_target_error(self, target_xy, target_z):
        if self.odom is None or target_xy is None or target_z is None:
            return float("inf"), float("inf"), float("inf")
        pos = self.odom.pose.pose.position
        dx = float(pos.x) - float(target_xy[0])
        dy = float(pos.y) - float(target_xy[1])
        dz = float(pos.z) - float(target_z)
        return math.sqrt(dx * dx + dy * dy + dz * dz), math.hypot(dx, dy), dz

    def _update_real_px4_lifecycle(self):
        self._publish_mode_state()
        if not self.cfg.real_auto_takeoff_on_offboard:
            gate_ok, gate_reason = self._real_common_gate_ok()
            if gate_ok:
                gate_ok, gate_reason = self._real_lidar_gate_ok()
            if not gate_ok:
                self.real_lifecycle_state = "WAIT_READY"
                self._publish_real_lifecycle_hold(gate_reason)
                return False
            if self.real_lifecycle_state != "ACTIVE":
                self.real_lifecycle_state = "ACTIVE"
                self._set_control_mode(self.cfg.post_takeoff_mode)
            return True

        hold_z = (
            self._takeoff_hold_z
            if self._takeoff_hold_z is not None
            else (float(self.odom.pose.pose.position.z) if self.odom is not None else None)
        )
        hold_xy = self._takeoff_xy
        if hold_xy is None and self.odom is not None:
            hold_xy = (
                float(self.odom.pose.pose.position.x),
                float(self.odom.pose.pose.position.y),
            )

        pre_ok, pre_reason = self._real_lifecycle_preconditions_ok()
        if not pre_ok:
            if pre_reason in {"MAVROS_NOT_CONNECTED", "PX4_NOT_OFFBOARD"}:
                if self.real_lifecycle_state == "ACTIVE":
                    rospy.logwarn("[SRLCReal] PX4 link/offboard lost; suppressing DIRECT/RL output")
                    self.real_lifecycle_state = "OFFBOARD_LOST_HOLD"
                elif self.real_lifecycle_state != "OFFBOARD_LOST_HOLD":
                    self.real_lifecycle_state = "WAIT_OFFBOARD"
                self._set_control_mode(self.cfg.post_takeoff_mode)
            self._publish_real_lifecycle_hold(pre_reason, target_xy=hold_xy, target_z=hold_z)
            return False

        if self.mavros_state is not None and not self.mavros_state.armed:
            self.real_lifecycle_state = "WAIT_ARMED"
            self._request_arm_on_offboard()
            reason = "ARMING_FAILED" if self._arm_request_failed else "WAIT_ARMED"
            self._publish_real_lifecycle_hold(reason, target_xy=hold_xy, target_z=hold_z)
            return False
        self._arm_request_failed = False

        if self.real_lifecycle_state in {"WAIT_OFFBOARD", "WAIT_ARMED"}:
            self._start_real_takeoff()
            hold_xy = self._takeoff_xy
            hold_z = self._takeoff_hold_z
        elif self.real_lifecycle_state == "OFFBOARD_LOST_HOLD":
            self._resume_real_takeoff_target()
            hold_xy = self._takeoff_xy
            hold_z = self._takeoff_hold_z

        if self.real_lifecycle_state == "TAKEOFF_CLIMB":
            target_z = float(hold_z if hold_z is not None else self.cfg.takeoff_height)
            self._publish_real_lifecycle_hold("TAKEOFF_CLIMB", target_xy=hold_xy, target_z=target_z)
            pos_err, xy_err, z_err = self._takeoff_target_error(hold_xy, target_z)
            if pos_err <= self.cfg.takeoff_reached_tolerance:
                self.real_lifecycle_state = "TAKEOFF_SETTLE"
                self._takeoff_settle_start = rospy.Time.now()
                rospy.loginfo(
                    "[SRLCReal] Takeoff target reached: err=%.3f xy=%.3f z=%.3f",
                    pos_err,
                    xy_err,
                    z_err,
                )
            return False

        if self.real_lifecycle_state == "TAKEOFF_SETTLE":
            target_z = float(hold_z if hold_z is not None else self.cfg.takeoff_height)
            self._publish_real_lifecycle_hold("TAKEOFF_SETTLE", target_xy=hold_xy, target_z=target_z)
            elapsed = (rospy.Time.now() - self._takeoff_settle_start).to_sec()
            if elapsed < self.cfg.post_takeoff_mode_delay:
                return False
            lidar_ok, lidar_reason = self._real_lidar_gate_ok()
            if not lidar_ok:
                self._publish_real_lifecycle_hold(lidar_reason, target_xy=hold_xy, target_z=target_z)
                return False
            self.real_lifecycle_state = "ACTIVE"
            self._set_control_mode(self.cfg.post_takeoff_mode)
            rospy.loginfo("[SRLCReal] Lifecycle ACTIVE: mode=%s", self.control_mode)

        return self.real_lifecycle_state == "ACTIVE"

    def _control_callback(self, _event):
        if not self.odom_received:
            return

        if not self._update_real_px4_lifecycle():
            return

        if self.collision:
            self._publish_stop(reason="COLLISION")
            self._publish_status("COLLISION", np.zeros(3, dtype=np.float32))
            return

        gate_ok, gate_reason = self._real_common_gate_ok()
        if gate_ok:
            gate_ok, gate_reason = self._real_lidar_gate_ok()
        if not gate_ok:
            request_hold = gate_reason not in {"NO_RC_ACTION"}
            self._publish_hover_cmd(request_hold=request_hold, reason=gate_reason)
            self._publish_status(gate_reason, np.zeros(3, dtype=np.float32))
            return

        if self.safety_stop:
            self._publish_stop(reason="SAFETY_STOP")
            self._publish_status("SAFETY_STOP", np.zeros(3, dtype=np.float32))
            return

        human_action_np = self._current_topic_human_cmd()
        self._publish_human_cmd(human_action_np)
        if (
            self.cfg.assist_input_deadzone_norm > 0.0
            and float(np.linalg.norm(human_action_np[:2])) < self.cfg.assist_input_deadzone_norm
        ):
            self._publish_hover_cmd(request_hold=False, reason="INPUT_DEADZONE")
            self._publish_status("INPUT_DEADZONE", np.zeros(3, dtype=np.float32))
            return

        if not self.assist_enabled:
            self._publish_direct_cmd(human_action_np)
            return

        state, human_action, lidar = self._build_obs()
        with torch.no_grad():
            action_world = self.policy(
                state,
                human_action,
                lidar,
                deterministic=self.cfg.deterministic,
            )
        cmd = action_world.squeeze(0).cpu().numpy().astype(np.float32)
        if self.cfg.lock_z_control:
            cmd[2] = 0.0

        self.policy_active = True
        self.policy_active_pub.publish(Bool(data=True))
        self._publish_policy_cmd(cmd)
        self._publish_cmd(cmd)
        self._publish_vis(cmd, human_action_np)
        self._publish_status("ASSIST", cmd)

    def _publish_status(self, reason, cmd):
        pos = self.odom.pose.pose.position if self.odom is not None else None
        if pos is None:
            msg = f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | {reason}"
        else:
            msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | "
                f"min_d={self.min_dist:.2f} | {reason}"
            )
        self.status_pub.publish(String(data=msg))

    def _publish_px4_hold_setpoint(self, target_xy=None, target_z=None, reason="HOLD"):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))

        if target_z is None:
            if self.odom is not None:
                target_z = float(self.odom.pose.pose.position.z)
            else:
                target_z = float(self.cfg.takeoff_height)
        if target_xy is None and self.odom is not None:
            target_xy = (
                float(self.odom.pose.pose.position.x),
                float(self.odom.pose.pose.position.y),
            )

        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        if target_xy is not None:
            msg.position.x = float(target_xy[0])
            msg.position.y = float(target_xy[1])
        msg.position.z = float(target_z)
        msg.yaw = self._current_yaw() if self.odom is not None else 0.0
        msg.type_mask = (
            PositionTarget.IGNORE_VX
            | PositionTarget.IGNORE_VY
            | PositionTarget.IGNORE_VZ
            | PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        self.action_pub.publish(msg)

    def _publish_real_lifecycle_hold(self, reason, target_xy=None, target_z=None):
        self._publish_px4_hold_setpoint(target_xy=target_xy, target_z=target_z, reason=reason)
        self._publish_mode_state()
        pos = self.odom.pose.pose.position if self.odom is not None else None
        if pos is None:
            status_msg = f"cmd=[0,0,0] | {self.real_lifecycle_state} | {reason}"
        else:
            z_cmd = float(target_z) if target_z is not None else float(pos.z)
            if target_xy is None:
                target_xy = (float(pos.x), float(pos.y))
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"hold=[{target_xy[0]:.2f},{target_xy[1]:.2f},{z_cmd:.2f}] | "
                f"min_d={self.min_dist:.2f} | {self.real_lifecycle_state} | {reason}"
            )
        self.status_pub.publish(String(data=status_msg))

    def _request_px4_hold_mode(self, reason="STOP"):
        if not self.cfg.hold_on_stop:
            return
        now = rospy.Time.now()
        if (now - self._last_hold_mode_request).to_sec() < 1.0:
            self._verify_px4_hold_mode(reason=reason)
            return
        self._last_hold_mode_request = now
        try:
            resp = self.set_mode_client(SetModeRequest(custom_mode=self.cfg.estop_hold_mode))
            if not resp.mode_sent:
                rospy.logwarn_throttle(
                    2.0,
                    "[SRLCReal] PX4 hold mode request rejected: mode=%s reason=%s",
                    self.cfg.estop_hold_mode,
                    reason,
                )
                self._request_px4_fallback_mode(reason=reason)
            else:
                self._hold_mode_request_time = now
                self._hold_mode_requested = self.cfg.estop_hold_mode
                self._hold_fallback_requested = False
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "[SRLCReal] PX4 hold mode request failed: %s", exc)
            self._request_px4_fallback_mode(reason=reason)
        self._verify_px4_hold_mode(reason=reason)

    def _request_px4_fallback_mode(self, reason="STOP"):
        if not self.cfg.estop_fallback_mode or self._hold_fallback_requested:
            return
        try:
            resp = self.set_mode_client(SetModeRequest(custom_mode=self.cfg.estop_fallback_mode))
            self._hold_fallback_requested = True
            if resp.mode_sent:
                self._hold_mode_request_time = rospy.Time.now()
                self._hold_mode_requested = self.cfg.estop_fallback_mode
                rospy.logwarn(
                    "[SRLCReal] Requested PX4 fallback mode %s after stop reason=%s",
                    self.cfg.estop_fallback_mode,
                    reason,
                )
            else:
                rospy.logerr(
                    "[SRLCReal] PX4 rejected fallback mode %s after stop reason=%s",
                    self.cfg.estop_fallback_mode,
                    reason,
                )
        except rospy.ServiceException as exc:
            rospy.logerr("[SRLCReal] PX4 fallback mode request failed: %s", exc)

    def _verify_px4_hold_mode(self, reason="STOP"):
        if (
            self._hold_mode_request_time is None
            or self.mavros_state is None
            or not self._hold_mode_requested
        ):
            return
        elapsed = (rospy.Time.now() - self._hold_mode_request_time).to_sec()
        if elapsed < self.cfg.hold_verify_timeout:
            return
        if self.mavros_state.mode != self._hold_mode_requested:
            rospy.logerr_throttle(
                2.0,
                "[SRLCReal] PX4 stop mode not confirmed: requested=%s current=%s reason=%s",
                self._hold_mode_requested,
                self.mavros_state.mode,
                reason,
            )
            self._request_px4_fallback_mode(reason=reason)

    def _publish_stop(self, reason="STOP"):
        self._publish_hover_cmd(request_hold=True, reason=reason)

    def _publish_hover_cmd(self, request_hold=False, reason="HOVER"):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))
        if request_hold:
            self._request_px4_hold_mode(reason=reason)

        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.velocity.x = 0.0
        msg.velocity.y = 0.0
        msg.yaw = self._current_yaw() if self.odom is not None else 0.0
        if self.cfg.height_control and not self.cfg.lock_z_control:
            msg.velocity.z = 0.0
            msg.type_mask = (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_PZ
                | PositionTarget.IGNORE_AFX
                | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ
                | PositionTarget.IGNORE_YAW_RATE
            )
        else:
            msg.position.z = self._px4_altitude_hold_target()
            msg.type_mask = (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_VZ
                | PositionTarget.IGNORE_AFX
                | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ
                | PositionTarget.IGNORE_YAW_RATE
            )
        self.action_pub.publish(msg)

    def _publish_direct_cmd(self, human_cmd_body):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))

        cmd = self._body_cmd_to_world(human_cmd_body)
        if self.cfg.lock_z_control:
            cmd[2] = 0.0
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))
        self._publish_cmd(cmd)
        self._publish_vis(cmd, human_cmd_body)
        self._publish_status("DIRECT", cmd)

    def _clamp_px4_cmd(self, cmd_vel_world):
        cmd = np.asarray(cmd_vel_world, dtype=np.float32).copy()
        hspeed = math.hypot(float(cmd[0]), float(cmd[1]))
        if hspeed > self.cfg.max_xy_speed_real:
            scale = self.cfg.max_xy_speed_real / hspeed
            cmd[0] *= scale
            cmd[1] *= scale
        cmd[2] = np.clip(cmd[2], -self.cfg.max_z_speed_real, self.cfg.max_z_speed_real)
        return cmd

    def _publish_cmd(self, cmd_vel_world):
        cmd_vel_world = self._clamp_px4_cmd(cmd_vel_world)
        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.velocity.x = float(cmd_vel_world[0])
        msg.velocity.y = float(cmd_vel_world[1])
        if self.cfg.height_control and not self.cfg.lock_z_control:
            msg.velocity.z = float(cmd_vel_world[2])
            msg.yaw = self._current_yaw()
            msg.type_mask = (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_PZ
                | PositionTarget.IGNORE_AFX
                | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ
                | PositionTarget.IGNORE_YAW_RATE
            )
        else:
            msg.position.z = self._px4_altitude_hold_target()
            msg.yaw = self._current_yaw()
            msg.type_mask = (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_VZ
                | PositionTarget.IGNORE_AFX
                | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ
                | PositionTarget.IGNORE_YAW_RATE
            )
        self.action_pub.publish(msg)

    def _current_topic_human_cmd(self):
        with self._topic_human_lock:
            cmd = self._topic_human_cmd.copy()
        if self.cfg.lock_z_control:
            cmd[2] = 0.0
        return cmd

    def _body_cmd_to_world(self, cmd_body):
        yaw = self._current_yaw()
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return np.array(
            [
                cy * cmd_body[0] - sy * cmd_body[1],
                sy * cmd_body[0] + cy * cmd_body[1],
                cmd_body[2],
            ],
            dtype=np.float32,
        )

    def _publish_human_cmd(self, cmd_body):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd_body[0])
        msg.twist.linear.y = float(cmd_body[1])
        msg.twist.linear.z = float(cmd_body[2])
        self.human_cmd_pub.publish(msg)

    def _publish_policy_cmd(self, cmd_world):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = float(cmd_world[0])
        msg.twist.linear.y = float(cmd_world[1])
        msg.twist.linear.z = float(cmd_world[2])
        self.policy_cmd_pub.publish(msg)

    def _apply_proximity_safety(self, min_dist):
        if not self.cfg.enable_safety_stop and not self.cfg.enable_collision_detection:
            self.safety_stop = False
            return
        if self.cfg.enable_collision_detection and min_dist < self.cfg.collision_dist:
            if not self.collision:
                rospy.logerr(
                    "[SRLCReal] COLLISION: min_dist=%.3f threshold=%.3f",
                    min_dist,
                    self.cfg.collision_dist,
                )
            self.collision = True
            self.collision_pub.publish(Bool(data=True))
            return
        if self.cfg.enable_safety_stop and min_dist < self.cfg.safety_min_dist:
            if not self.safety_stop:
                rospy.logwarn(
                    "[SRLCReal] SAFETY_STOP: min_dist=%.3f threshold=%.3f",
                    min_dist,
                    self.cfg.safety_min_dist,
                )
            self.safety_stop = True
        else:
            if self.safety_stop:
                rospy.loginfo("[SRLCReal] Safety cleared: min_dist=%.3f", min_dist)
            self.safety_stop = False

    def _safety_timer_cb(self, _event):
        if self.collision or not self.odom_received:
            return
        if self.last_min_dist_time is None:
            return
        if (rospy.Time.now() - self.last_min_dist_time).to_sec() > self.cfg.lidar_timeout:
            if self.cfg.enable_safety_stop or self.cfg.enable_collision_detection:
                if not self.safety_stop:
                    rospy.logwarn_throttle(2.0, "[SRLCReal] SAFETY_STOP: min_distance timeout")
                self.safety_stop = True
            return

        pos_z = float(self.odom.pose.pose.position.z)
        if self.initial_z is None:
            self.initial_z = pos_z
        if not self.safety_airborne:
            if pos_z >= self.initial_z + self.cfg.safety_start_takeoff_delta:
                self.safety_airborne = True
                rospy.loginfo(
                    "[SRLCReal] Safety monitor airborne at z=%.2f baseline=%.2f",
                    pos_z,
                    self.initial_z,
                )
            else:
                self.safety_stop = False
                return
        self._apply_proximity_safety(self.min_dist)

    @staticmethod
    def _quat_to_rot(q):
        return np.array(tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3])

    def _current_yaw(self):
        if self.odom is None:
            return 0.0
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def _publish_vis(self, cmd_world, human_cmd_body):
        if self.odom is None:
            return
        pos = self.odom.pose.pose.position

        rl_marker = Marker()
        rl_marker.header.frame_id = "map"
        rl_marker.header.stamp = rospy.Time.now()
        rl_marker.ns = "rl_cmd"
        rl_marker.id = 0
        rl_marker.type = Marker.ARROW
        rl_marker.action = Marker.ADD
        rl_marker.scale.x = 0.05
        rl_marker.scale.y = 0.1
        rl_marker.scale.z = 0.1
        rl_marker.color.g = 1.0
        rl_marker.color.a = 1.0
        rl_marker.points.append(Point(x=pos.x, y=pos.y, z=pos.z))
        rl_marker.points.append(
            Point(
                x=pos.x + float(cmd_world[0]),
                y=pos.y + float(cmd_world[1]),
                z=pos.z + float(cmd_world[2]),
            )
        )
        self.cmd_vis_pub.publish(MarkerArray(markers=[rl_marker]))

        rot = self._quat_to_rot(self.odom.pose.pose.orientation)
        human_world = rot @ np.asarray(human_cmd_body, dtype=np.float32)
        human_marker = Marker()
        human_marker.header.frame_id = "map"
        human_marker.header.stamp = rospy.Time.now()
        human_marker.ns = "human_cmd"
        human_marker.id = 1
        human_marker.type = Marker.ARROW
        human_marker.action = Marker.ADD
        human_marker.scale.x = 0.05
        human_marker.scale.y = 0.1
        human_marker.scale.z = 0.1
        human_marker.color.b = 1.0
        human_marker.color.a = 1.0
        human_marker.points.append(Point(x=pos.x, y=pos.y, z=pos.z))
        human_marker.points.append(
            Point(
                x=pos.x + float(human_world[0]),
                y=pos.y + float(human_world[1]),
                z=pos.z + float(human_world[2]),
            )
        )
        self.human_cmd_vis_pub.publish(MarkerArray(markers=[human_marker]))


def main():
    try:
        RealNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
