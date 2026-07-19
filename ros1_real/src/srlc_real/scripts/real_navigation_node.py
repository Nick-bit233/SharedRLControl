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

from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import SetMode, SetModeRequest
from srlc_real.msg import ObstacleClearance


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.policy_net import TunnelPolicyNet  # noqa: E402
from srlc_real_deployment.clearance_guard import (  # noqa: E402
    ClearanceGuard,
    ClearanceGuardConfig,
    ClearanceGuardResult,
    ClearanceState,
    project_velocity_away,
)
from srlc_real_deployment.config_migration import (  # noqa: E402
    find_legacy_safety_config,
    legacy_safety_migration_error,
)
from srlc_real_deployment.clearance_runtime import soft_guard_position  # noqa: E402
from srlc_real_deployment.one_shot_flight import (  # noqa: E402
    FlightAction,
    FlightSnapshot,
    LifecycleState,
    OneShotFlightConfig,
    OneShotFlightLifecycle,
)


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
        self.clearance_topic = rospy.get_param(
            "~clearance_topic", "/srlc/lidar/obstacle_clearance"
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
        self.takeoff_height = float(rospy.get_param("~takeoff_height", 1.0))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.3))
        self.odom_topic = rospy.get_param("~odom_topic", "/nokov/local_position/odom")
        self.px4_local_odom_topic = rospy.get_param(
            "~px4_local_odom_topic", "/mavros/local_position/odom"
        )
        self.px4_local_velocity_topic = rospy.get_param(
            "~px4_local_velocity_topic", "/mavros/local_position/velocity_local"
        )
        self.setpoint_raw_topic = rospy.get_param(
            "~setpoint_raw_topic", "/mavros/setpoint_raw/local"
        )
        self.require_offboard = _param_bool(rospy.get_param("~require_offboard", True))
        self.real_auto_takeoff_on_offboard = _param_bool(
            rospy.get_param("~real_auto_takeoff_on_offboard", True)
        )
        self.post_takeoff_mode = str(rospy.get_param("~post_takeoff_mode", "assist")).lower()
        self.takeoff_lower_margin = float(rospy.get_param("~takeoff_lower_margin", 0.2))
        self.takeoff_upper_margin = float(rospy.get_param("~takeoff_upper_margin", 0.2))
        self.takeoff_max_abs_vz = float(rospy.get_param("~takeoff_max_abs_vz", 0.25))
        self.takeoff_confirm_duration = float(
            rospy.get_param("~takeoff_confirm_duration", 0.5)
        )
        self.takeoff_timeout = float(rospy.get_param("~takeoff_timeout", 15.0))
        self.takeoff_max_overshoot = float(
            rospy.get_param("~takeoff_max_overshoot", 0.5)
        )
        self.takeoff_max_xy_drift = float(
            rospy.get_param("~takeoff_max_xy_drift", 0.5)
        )
        self.takeoff_max_climb_speed = float(
            rospy.get_param("~takeoff_max_climb_speed", 0.4)
        )
        self.takeoff_max_vertical_accel = float(
            rospy.get_param("~takeoff_max_vertical_accel", 0.5)
        )
        self.takeoff_max_tracking_error = float(
            rospy.get_param("~takeoff_max_tracking_error", 0.25)
        )
        self.input_recovery_grace = float(
            rospy.get_param("~input_recovery_grace", 1.0)
        )
        self.fault_response = str(rospy.get_param("~fault_response", "hold")).lower()
        self.fault_land_mode = str(rospy.get_param("~fault_land_mode", "AUTO.LAND"))
        self.fault_land_confirm_timeout = float(
            rospy.get_param("~fault_land_confirm_timeout", 2.0)
        )
        self.fault_land_retry_interval = float(
            rospy.get_param("~fault_land_retry_interval", 0.5)
        )
        self.fault_land_max_attempts = int(
            rospy.get_param("~fault_land_max_attempts", 3)
        )
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

        self.enable_proximity_guard = _param_bool(
            rospy.get_param("~enable_proximity_guard", False)
        )
        self.proximity_enter_clearance = float(
            rospy.get_param("~proximity_enter_clearance", 0.10)
        )
        self.proximity_release_clearance = float(
            rospy.get_param("~proximity_release_clearance", 0.15)
        )
        self.proximity_release_duration = float(
            rospy.get_param("~proximity_release_duration", 0.20)
        )
        self.proximity_escape_min_speed = float(
            rospy.get_param("~proximity_escape_min_speed", 0.05)
        )
        self.collision_confirm_clearance = float(
            rospy.get_param("~collision_confirm_clearance", 0.02)
        )
        self.collision_confirm_frames = int(
            rospy.get_param("~collision_confirm_frames", 2)
        )
        self.collision_immediate_clearance = float(
            rospy.get_param("~collision_immediate_clearance", -0.03)
        )
        self.clearance_guard_mode = str(
            rospy.get_param("~clearance_guard_mode", "enforce")
        ).strip().lower()

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
        self._reject_legacy_safety_config()
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
        self.px4_local_odom = None
        self.last_px4_local_odom_time = None
        self.px4_local_velocity = None
        self.last_px4_local_velocity_time = None
        self.mavros_state = None
        self.extended_state = None
        self.topic_lidar_np = None
        self.last_lidar_time = None
        self.last_min_dist_time = None
        self.policy_dist = float("inf")
        self.raw_center_distance = float("inf")
        self.clearance = float("nan")
        self.clearance_center_distance = float("nan")
        self._topic_human_cmd = np.zeros(3, dtype=np.float32)
        self._topic_human_lock = threading.Lock()
        self.last_human_action_time = None

        self._clearance_guard_lock = threading.Lock()
        self.clearance_guard = ClearanceGuard(
            ClearanceGuardConfig(
                proximity_enabled=self.cfg.enable_proximity_guard,
                proximity_enter_clearance=self.cfg.proximity_enter_clearance,
                proximity_release_clearance=self.cfg.proximity_release_clearance,
                proximity_release_duration=self.cfg.proximity_release_duration,
                escape_dot_threshold=self.cfg.proximity_escape_min_speed,
                collision_clearance=self.cfg.collision_confirm_clearance,
                collision_confirm_samples=self.cfg.collision_confirm_frames,
                immediate_collision_clearance=self.cfg.collision_immediate_clearance,
                sample_timeout=self.cfg.lidar_timeout,
            )
        )
        self._latest_clearance_sample = None
        self._last_clearance_source_stamp = None
        self._clearance_sample_usable = False
        self._clearance_result = ClearanceGuardResult(
            ClearanceState.NORMAL,
            None,
            None,
        )
        self._collision_pending = False

        self.control_mode = self.cfg.post_takeoff_mode.upper()
        self.assist_enabled = self.control_mode == "ASSIST"
        self.effective_mode = "INACTIVE"
        self.fault_reason = ""
        self.lifecycle = OneShotFlightLifecycle(
            OneShotFlightConfig(
                enabled=self.cfg.real_auto_takeoff_on_offboard,
                takeoff_height=self.cfg.takeoff_height,
                takeoff_lower_margin=self.cfg.takeoff_lower_margin,
                takeoff_upper_margin=self.cfg.takeoff_upper_margin,
                takeoff_max_abs_vz=self.cfg.takeoff_max_abs_vz,
                takeoff_confirm_duration=self.cfg.takeoff_confirm_duration,
                takeoff_timeout=self.cfg.takeoff_timeout,
                takeoff_max_overshoot=self.cfg.takeoff_max_overshoot,
                takeoff_max_xy_drift=self.cfg.takeoff_max_xy_drift,
                takeoff_max_climb_speed=self.cfg.takeoff_max_climb_speed,
                takeoff_max_vertical_accel=self.cfg.takeoff_max_vertical_accel,
                takeoff_max_tracking_error=self.cfg.takeoff_max_tracking_error,
                input_recovery_grace=self.cfg.input_recovery_grace,
                fault_response=self.cfg.fault_response,
                fault_land_mode=self.cfg.fault_land_mode,
                fault_land_confirm_timeout=self.cfg.fault_land_confirm_timeout,
                fault_land_retry_interval=self.cfg.fault_land_retry_interval,
                fault_land_max_attempts=self.cfg.fault_land_max_attempts,
            )
        )
        self.real_lifecycle_state = self.lifecycle.state
        self._last_decision = None

        self.ready = False
        self.policy_active = False
        self.z_policy_active = False
        self.safety_stop = False
        self.collision = False

        self._setup_ros()
        self._publish_mode_state()
        self._log_config()

        self.diagnostic_timer = rospy.Timer(rospy.Duration(0.5), self._diagnostic_timer_cb)
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.cfg.control_freq), self._control_callback
        )

    def _validate_config(self):
        if self.cfg.post_takeoff_mode not in {"direct", "assist"}:
            raise ValueError("post_takeoff_mode must be 'direct' or 'assist'")
        if not self.cfg.require_offboard:
            raise ValueError("one-shot real flight requires ~require_offboard=true")
        if self.cfg.fault_response not in {"auto_land", "hold"}:
            raise ValueError("fault_response must be 'auto_land' or 'hold'")
        if self.cfg.policy_mode not in {"residual", "direct"}:
            raise ValueError("policy_mode must be 'residual' or 'direct'")
        if self.cfg.clearance_guard_mode not in {"enforce", "shadow"}:
            raise ValueError("clearance_guard_mode must be 'enforce' or 'shadow'")
        if not self.cfg.checkpoint_path:
            raise ValueError("~checkpoint_path is required")

    @staticmethod
    def _reject_legacy_safety_config():
        uses = find_legacy_safety_config(
            rospy.get_param_names(),
            os.environ,
        )
        if not uses.found:
            return
        message = legacy_safety_migration_error(uses)
        rospy.logfatal("[SRLCReal] %s", message)
        raise RuntimeError(message)

    def _log_config(self):
        rospy.loginfo(
            "[SRLCReal] model_odom=%s px4_local=(%s, %s) setpoint=%s",
            self.cfg.odom_topic,
            self.cfg.px4_local_odom_topic,
            self.cfg.px4_local_velocity_topic,
            self.cfg.setpoint_raw_topic,
        )
        rospy.loginfo(
            "[SRLCReal] requested_mode=%s one_shot_takeoff=%.2fm manual_arm=true",
            self.control_mode,
            self.cfg.takeoff_height,
        )
        rospy.loginfo(
            "[SRLCReal] lidar=%dx%d range=%.2fm topics=(%s, policy_diag=%s, clearance=%s)",
            self.cfg.lidar_hbeams,
            self.cfg.lidar_vbeams,
            self.cfg.lidar_range,
            self.cfg.lidar_range_image_topic,
            self.cfg.lidar_min_distance_topic,
            self.cfg.clearance_topic,
        )
        rospy.loginfo(
            "[SRLCReal] takeoff_band=[-%.2f,+%.2f] vz<=%.2f stable=%.2fs timeout=%.1fs fault=%s",
            self.cfg.takeoff_lower_margin,
            self.cfg.takeoff_upper_margin,
            self.cfg.takeoff_max_abs_vz,
            self.cfg.takeoff_confirm_duration,
            self.cfg.takeoff_timeout,
            self.cfg.fault_response,
        )
        rospy.loginfo(
            "[SRLCReal] takeoff_profile: climb<=%.2fm/s accel<=%.2fm/s^2 tracking_error<=%.2fm",
            self.cfg.takeoff_max_climb_speed,
            self.cfg.takeoff_max_vertical_accel,
            self.cfg.takeoff_max_tracking_error,
        )
        rospy.loginfo(
            "[SRLCReal] clearance_guard=%s proximity=%s enter=%.2f release=%.2f collision=(%.2f x%d, immediate=%.2f)",
            self.cfg.clearance_guard_mode,
            self.cfg.enable_proximity_guard,
            self.cfg.proximity_enter_clearance,
            self.cfg.proximity_release_clearance,
            self.cfg.collision_confirm_clearance,
            self.cfg.collision_confirm_frames,
            self.cfg.collision_immediate_clearance,
        )

    def _setup_ros(self):
        self.odom_sub = rospy.Subscriber(self.cfg.odom_topic, Odometry, self._odom_cb, queue_size=1)
        self.px4_local_odom_sub = rospy.Subscriber(
            self.cfg.px4_local_odom_topic,
            Odometry,
            self._px4_local_odom_cb,
            queue_size=1,
        )
        self.px4_local_velocity_sub = rospy.Subscriber(
            self.cfg.px4_local_velocity_topic,
            TwistStamped,
            self._px4_local_velocity_cb,
            queue_size=1,
        )
        self.state_sub = rospy.Subscriber("/mavros/state", State, self._mavros_state_cb, queue_size=1)
        self.extended_state_sub = rospy.Subscriber(
            "/mavros/extended_state", ExtendedState, self._extended_state_cb, queue_size=1
        )
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
        self.effective_mode_pub = rospy.Publisher(
            "/tunnel_nav/effective_mode", String, queue_size=2
        )
        self.session_consumed_pub = rospy.Publisher(
            "/tunnel_nav/session_consumed", Bool, queue_size=2, latch=True
        )
        self.fault_reason_pub = rospy.Publisher(
            "/tunnel_nav/fault_reason", String, queue_size=2, latch=True
        )
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
        self.clearance_guard_state_pub = rospy.Publisher(
            "/tunnel_nav/clearance_guard_state",
            String,
            queue_size=10,
        )
        self.clearance_guard_shadow_state_pub = rospy.Publisher(
            "/tunnel_nav/clearance_guard_shadow_state",
            String,
            queue_size=10,
        )

        self.clearance_sub = rospy.Subscriber(
            self.cfg.clearance_topic,
            ObstacleClearance,
            self._clearance_cb,
            queue_size=50,
        )

        self.collision_pub.publish(Bool(data=False))
        self.session_consumed_pub.publish(Bool(data=False))
        self.fault_reason_pub.publish(String(data=""))
        self._publish_guard_states(ClearanceState.NORMAL, ClearanceState.NORMAL)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    def _odom_cb(self, msg):
        self.odom = msg
        self.odom_received = True
        self.last_odom_time = rospy.Time.now()

    def _px4_local_odom_cb(self, msg):
        self.px4_local_odom = msg
        self.last_px4_local_odom_time = rospy.Time.now()

    def _px4_local_velocity_cb(self, msg):
        self.px4_local_velocity = msg
        self.last_px4_local_velocity_time = rospy.Time.now()

    def _mavros_state_cb(self, msg):
        self.mavros_state = msg

    def _extended_state_cb(self, msg):
        self.extended_state = msg

    def _human_action_cb(self, msg):
        with self._topic_human_lock:
            self._topic_human_cmd[0] = msg.twist.linear.x
            self._topic_human_cmd[1] = msg.twist.linear.y
            self._topic_human_cmd[2] = msg.twist.linear.z
        self.last_human_action_time = rospy.Time.now()

    def _lidar_range_cb(self, msg):
        expected = self.cfg.lidar_hbeams * self.cfg.lidar_vbeams
        if len(msg.data) != expected:
            self._invalidate_lidar_range()
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
        if not np.isfinite(lidar_np).all():
            self._invalidate_lidar_range()
            rospy.logwarn_throttle(
                2.0,
                "[SRLCReal] Ignoring non-finite LiDAR range image",
            )
            return
        self.topic_lidar_np = np.clip(lidar_np, 0.0, 1.0)
        self.policy_dist = float(
            self.cfg.lidar_range * (1.0 - np.max(self.topic_lidar_np))
        )
        self.last_lidar_time = rospy.Time.now()
        self.ready = True

    def _invalidate_lidar_range(self):
        self.topic_lidar_np = None
        self.last_lidar_time = None
        self.policy_dist = float("inf")
        self.ready = False

    def _lidar_min_dist_cb(self, msg):
        self.raw_center_distance = float(msg.data)
        self.last_min_dist_time = rospy.Time.now()

    def _clearance_cb(self, msg):
        now = float(rospy.Time.now().to_sec())
        source_stamp = float(msg.header.stamp.to_sec())
        surface_clearance = float(msg.surface_clearance)
        center_distance = float(msg.center_distance)
        nearest_point = (
            float(msg.nearest_obstacle_point.x),
            float(msg.nearest_obstacle_point.y),
            float(msg.nearest_obstacle_point.z),
        )
        escape_direction = (
            float(msg.escape_direction.x),
            float(msg.escape_direction.y),
            float(msg.escape_direction.z),
        )
        human_velocity_world = self._body_cmd_to_world(
            self._current_topic_human_cmd()
        )
        px4_local_position = self._clearance_guard_position(
            self._current_px4_local_position()
        )

        with self._clearance_guard_lock:
            usable = self._clearance_values_usable_locked(
                now=now,
                source_stamp=source_stamp,
                frame_id=msg.header.frame_id,
                valid=msg.valid,
                surface_clearance=surface_clearance,
                center_distance=center_distance,
                nearest_point=nearest_point,
                escape_direction=escape_direction,
            )
            if usable:
                if (
                    self._last_clearance_source_stamp is None
                    or source_stamp > self._last_clearance_source_stamp
                ):
                    self._last_clearance_source_stamp = source_stamp
                self._latest_clearance_sample = (
                    source_stamp,
                    surface_clearance,
                    escape_direction,
                )
                self._clearance_sample_usable = True
                self.clearance = surface_clearance
                self.clearance_center_distance = center_distance
            else:
                self._latest_clearance_sample = None
                self._clearance_sample_usable = False
                self.clearance = float("nan")
                self.clearance_center_distance = float("nan")

            result = self.clearance_guard.update(
                now=now,
                source_stamp=source_stamp,
                valid=usable,
                surface_clearance=surface_clearance,
                escape_direction=escape_direction,
                human_velocity_world=human_velocity_world,
                px4_local_position=px4_local_position,
            )
            raw_state, effective_state = self._record_guard_result_locked(result)

        self._publish_guard_states(raw_state, effective_state)

    def _clearance_values_usable_locked(
        self,
        now,
        source_stamp,
        frame_id,
        valid,
        surface_clearance,
        center_distance,
        nearest_point,
        escape_direction,
    ):
        numeric_values = (
            now,
            source_stamp,
            surface_clearance,
            center_distance,
            *nearest_point,
            *escape_direction,
        )
        if not bool(valid) or not str(frame_id).strip():
            return False
        if not all(math.isfinite(value) for value in numeric_values):
            return False
        if source_stamp > now or now - source_stamp > self.cfg.lidar_timeout:
            return False
        if (
            self._last_clearance_source_stamp is not None
            and source_stamp < self._last_clearance_source_stamp
        ):
            return False
        direction_norm = math.sqrt(
            sum(component * component for component in escape_direction)
        )
        return math.isfinite(direction_norm) and direction_norm > 0.0

    def _record_guard_result_locked(self, result):
        self._clearance_result = result
        raw_state = result.state
        effective_state = (
            raw_state
            if self.cfg.clearance_guard_mode == "enforce"
            else ClearanceState.NORMAL
        )
        if (
            self.cfg.clearance_guard_mode == "enforce"
            and raw_state == ClearanceState.COLLISION
        ):
            self._collision_pending = True
        return raw_state, effective_state

    def _refresh_clearance_guard(
        self,
        now,
        human_velocity_world,
        px4_local_position,
    ):
        with self._clearance_guard_lock:
            sample = self._latest_clearance_sample
            if sample is None:
                result = self._clearance_result
            else:
                source_stamp, surface_clearance, escape_direction = sample
                result = self.clearance_guard.update(
                    now=now,
                    source_stamp=source_stamp,
                    valid=self._clearance_sample_usable,
                    surface_clearance=surface_clearance,
                    escape_direction=escape_direction,
                    human_velocity_world=human_velocity_world,
                    px4_local_position=px4_local_position,
                )
                if source_stamp > now or now - source_stamp > self.cfg.lidar_timeout:
                    self._latest_clearance_sample = None
                    self._clearance_sample_usable = False
                    self.clearance = float("nan")
                    self.clearance_center_distance = float("nan")
                self._record_guard_result_locked(result)

            raw_state = self._clearance_result.state
            effective_state = raw_state if (
                self.cfg.clearance_guard_mode == "enforce"
            ) else ClearanceState.NORMAL
        self._publish_guard_states(raw_state, effective_state)
        return result

    def _clearance_is_fresh(self, now):
        with self._clearance_guard_lock:
            if not self._clearance_sample_usable:
                return False
            if self._latest_clearance_sample is None:
                return False
            source_stamp = self._latest_clearance_sample[0]
            return (
                math.isfinite(source_stamp)
                and source_stamp <= now
                and now - source_stamp <= self.cfg.lidar_timeout
            )

    def _current_px4_local_position(self):
        if self.px4_local_odom is None:
            return (float("nan"), float("nan"), float("nan"))
        position = self.px4_local_odom.pose.pose.position
        return (float(position.x), float(position.y), float(position.z))

    def _clearance_guard_position(self, current_position):
        return soft_guard_position(
            clearance_guard_mode=self.cfg.clearance_guard_mode,
            lifecycle_active=self.lifecycle.state == LifecycleState.ACTIVE,
            px4_local_position=current_position,
        )

    def _publish_guard_states(self, raw_state, effective_state):
        self.clearance_guard_shadow_state_pub.publish(
            String(data=str(raw_state))
        )
        self.clearance_guard_state_pub.publish(
            String(data=str(effective_state))
        )

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

    def _publish_mode_state(self):
        self.real_lifecycle_state = self.lifecycle.state
        self.lifecycle_state_pub.publish(String(data=str(self.real_lifecycle_state)))
        self.control_mode_pub.publish(String(data=str(self.control_mode)))
        self.effective_mode_pub.publish(String(data=str(self.effective_mode)))
        self.session_consumed_pub.publish(
            Bool(data=bool(self.lifecycle.session_consumed))
        )
        self.fault_reason_pub.publish(String(data=str(self.fault_reason)))

    def _set_control_mode(self, mode):
        self.control_mode = str(mode).upper()
        self.assist_enabled = self.control_mode == "ASSIST"
        self._publish_mode_state()

    def _px4_altitude_hold_target(self):
        if self.lifecycle.takeoff_target is not None:
            return float(self.lifecycle.takeoff_target[2])
        if self.px4_local_odom is not None:
            return float(self.px4_local_odom.pose.pose.position.z)
        return float(self.cfg.takeoff_height)

    @staticmethod
    def _is_fresh(now, stamp, timeout):
        return stamp is not None and (now - stamp).to_sec() <= timeout

    def _external_fault_reason(self):
        if self.odom is None:
            return None
        pos = self.odom.pose.pose.position
        if pos.z > self.cfg.max_altitude:
            return "HIGH_ALTITUDE"
        if (
            self.lifecycle.state == LifecycleState.ACTIVE
            and pos.z < self.cfg.min_altitude
        ):
            return "LOW_ALTITUDE"
        if len(self.cfg.geofence_x) == 2:
            if pos.x < self.cfg.geofence_x[0] or pos.x > self.cfg.geofence_x[1]:
                return "GEOFENCE_X"
        if len(self.cfg.geofence_y) == 2:
            if pos.y < self.cfg.geofence_y[0] or pos.y > self.cfg.geofence_y[1]:
                return "GEOFENCE_Y"
        return None

    def _flight_snapshot(self, now):
        connected = bool(self.mavros_state is not None and self.mavros_state.connected)
        armed = bool(self.mavros_state is not None and self.mavros_state.armed)
        mode = str(self.mavros_state.mode) if self.mavros_state is not None else ""

        position = None
        velocity = None
        if self.px4_local_odom is not None:
            pos = self.px4_local_odom.pose.pose.position
            position = (float(pos.x), float(pos.y), float(pos.z))
        if self.px4_local_velocity is not None:
            vel = self.px4_local_velocity.twist.linear
            velocity = (float(vel.x), float(vel.y), float(vel.z))

        nokov_odom_fresh = self._is_fresh(
            now, self.last_odom_time, self.cfg.odom_timeout
        )
        px4_odom_fresh = self._is_fresh(
            now, self.last_px4_local_odom_time, self.cfg.odom_timeout
        )
        px4_velocity_fresh = self._is_fresh(
            now, self.last_px4_local_velocity_time, self.cfg.odom_timeout
        )
        odom_fresh = bool(
            nokov_odom_fresh and px4_odom_fresh and px4_velocity_fresh
        )
        rc_fresh = self._is_fresh(
            now, self.last_human_action_time, self.cfg.human_action_timeout
        )
        range_fresh = self._is_fresh(
            now, self.last_lidar_time, self.cfg.lidar_timeout
        )
        clearance_fresh = self._clearance_is_fresh(now.to_sec())
        landed = bool(
            self.extended_state is not None
            and self.extended_state.landed_state
            == ExtendedState.LANDED_STATE_ON_GROUND
        )
        with self._clearance_guard_lock:
            collision_pending = self._collision_pending
        if collision_pending:
            external_fault = "COLLISION"
        else:
            external_fault = (
                self._external_fault_reason() if nokov_odom_fresh else None
            )
        return FlightSnapshot(
            now=now.to_sec(),
            connected=connected,
            armed=armed,
            mode=mode,
            position=position,
            velocity=velocity,
            odom_fresh=odom_fresh,
            rc_fresh=rc_fresh,
            lidar_fresh=bool(
                range_fresh and clearance_fresh and self.ready
            ),
            landed=landed,
            external_fault=external_fault,
        )

    def _deactivate_policy(self):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))

    def _request_fault_mode(self, mode, reason):
        try:
            resp = self.set_mode_client(SetModeRequest(custom_mode=str(mode)))
            if not resp.mode_sent:
                rospy.logerr_throttle(
                    1.0,
                    "[SRLCReal] Emergency mode request rejected: mode=%s reason=%s",
                    mode,
                    reason,
                )
        except rospy.ServiceException as exc:
            rospy.logerr_throttle(
                1.0,
                "[SRLCReal] Emergency mode request failed: mode=%s reason=%s error=%s",
                mode,
                reason,
                exc,
            )

    def _log_decision_transition(self, decision):
        if not decision.state_changed:
            if decision.state == LifecycleState.FAULT_HOLD:
                rospy.logerr_throttle(
                    2.0,
                    "[SRLCReal] FAULT_HOLD: reason=%s target=%s; manual takeover required",
                    decision.reason,
                    decision.target,
                )
            return
        if decision.state == LifecycleState.TAKEOFF:
            rospy.loginfo(
                "[SRLCReal] One-shot takeoff started: origin=%s target=%s",
                self.lifecycle.takeoff_origin,
                self.lifecycle.takeoff_target,
            )
        elif decision.state == LifecycleState.ACTIVE:
            rospy.loginfo(
                "[SRLCReal] Lifecycle ACTIVE: mode=%s target_z=%.2f",
                self.control_mode,
                self._px4_altitude_hold_target(),
            )
        elif decision.state in {LifecycleState.FAULT_LAND, LifecycleState.FAULT_HOLD}:
            pos = decision.target
            rospy.logerr(
                "[SRLCReal] FLIGHT FAULT: state=%s reason=%s hold=%s response=%s",
                decision.state,
                decision.reason,
                pos,
                self.cfg.fault_response,
            )
        elif decision.state == LifecycleState.TERMINATED:
            rospy.logwarn(
                "[SRLCReal] Lifecycle TERMINATED: reason=%s; restart navigator before another takeoff",
                decision.reason,
            )

    def _handle_lifecycle_decision(self, decision):
        self._last_decision = decision
        self.real_lifecycle_state = decision.state
        self.fault_reason = (
            decision.reason
            if decision.state in {
                LifecycleState.FAULT_LAND,
                LifecycleState.FAULT_HOLD,
            }
            else self.fault_reason
        )
        if decision.reason == "COLLISION":
            self.collision = True
            self.collision_pub.publish(Bool(data=True))
        self.safety_stop = decision.reason == "INPUT_RECOVERY_HOLD"

        if decision.action == FlightAction.ACTIVE_CONTROL:
            self.effective_mode = self.control_mode
        elif decision.action == FlightAction.TAKEOFF_HOLD:
            self.effective_mode = "TAKEOFF_HOLD"
        elif decision.action == FlightAction.PRESTREAM_HOLD:
            self.effective_mode = "INACTIVE"
        elif decision.action == FlightAction.REQUEST_MODE:
            self.effective_mode = "AUTO_LAND_REQUEST"
        elif decision.action == FlightAction.FAULT_HOLD:
            self.effective_mode = (
                "INPUT_HOLD"
                if decision.reason in {"INPUT_RECOVERY_HOLD", "PROXIMITY_HOLD"}
                else "FAULT_HOLD"
            )
        elif decision.state == LifecycleState.TERMINATED:
            self.effective_mode = "TERMINATED"
        else:
            self.effective_mode = "INACTIVE"

        self._log_decision_transition(decision)
        self._publish_mode_state()

        if decision.action == FlightAction.ACTIVE_CONTROL:
            return True

        self._deactivate_policy()
        if decision.action == FlightAction.STOP_STREAM:
            self._publish_status(decision.reason, np.zeros(3, dtype=np.float32))
            return False

        if decision.request_mode:
            self._request_fault_mode(decision.request_mode, decision.reason)

        if decision.target is not None:
            self._publish_real_lifecycle_hold(
                decision.reason,
                target_xy=(decision.target[0], decision.target[1]),
                target_z=decision.target[2],
                target_velocity=decision.target_velocity,
            )
        else:
            self._publish_status(decision.reason, np.zeros(3, dtype=np.float32))
        return False

    def _control_callback(self, _event):
        now = rospy.Time.now()
        human_action_np = self._current_topic_human_cmd()
        human_velocity_world = self._body_cmd_to_world(human_action_np)
        current_px4_position = self._current_px4_local_position()
        guard_result = self._refresh_clearance_guard(
            now=now.to_sec(),
            human_velocity_world=human_velocity_world,
            px4_local_position=self._clearance_guard_position(
                current_px4_position
            ),
        )

        snapshot = self._flight_snapshot(now)
        decision = self.lifecycle.update(snapshot)
        if decision.reason == "COLLISION":
            with self._clearance_guard_lock:
                self._collision_pending = False
        if not self._handle_lifecycle_decision(decision):
            return

        if decision.state_changed and decision.state == LifecycleState.ACTIVE:
            guard_result = self._refresh_clearance_guard(
                now=now.to_sec(),
                human_velocity_world=human_velocity_world,
                px4_local_position=self._clearance_guard_position(
                    current_px4_position
                ),
            )

        self._publish_human_cmd(human_action_np)
        proximity_escape = False
        if self.cfg.clearance_guard_mode == "enforce":
            if guard_result.state == ClearanceState.PROXIMITY_HOLD:
                self._publish_proximity_hold(
                    guard_result,
                    reason="PROXIMITY_HOLD",
                )
                return
            if guard_result.state == ClearanceState.PROXIMITY_ESCAPE:
                if not self.assist_enabled:
                    self._publish_proximity_hold(
                        guard_result,
                        reason="PROXIMITY_HOLD_NON_ASSIST",
                    )
                    return
                if guard_result.escape_direction is None:
                    self._publish_proximity_hold(
                        guard_result,
                        reason="PROXIMITY_HOLD_NO_ESCAPE",
                    )
                    return
                proximity_escape = True

        if (
            not proximity_escape
            and self.cfg.assist_input_deadzone_norm > 0.0
            and float(np.linalg.norm(human_action_np[:2])) < self.cfg.assist_input_deadzone_norm
        ):
            self.effective_mode = "ASSIST_IDLE"
            self._publish_mode_state()
            self._publish_hover_cmd(reason="INPUT_DEADZONE")
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
        if proximity_escape:
            try:
                cmd = np.asarray(
                    project_velocity_away(
                        cmd,
                        guard_result.escape_direction,
                    ),
                    dtype=np.float32,
                )
            except ValueError:
                self._publish_proximity_hold(
                    guard_result,
                    reason="PROXIMITY_HOLD_INVALID_ESCAPE",
                )
                return

        self.policy_active = True
        self.policy_active_pub.publish(Bool(data=True))
        self.effective_mode = (
            "PROXIMITY_ESCAPE" if proximity_escape else "ASSIST"
        )
        self._publish_mode_state()
        self._publish_policy_cmd(cmd)
        self._publish_cmd(cmd)
        self._publish_vis(cmd, human_action_np)
        self._publish_status(
            "PROXIMITY_ESCAPE" if proximity_escape else "ASSIST",
            cmd,
        )

    def _publish_proximity_hold(self, guard_result, reason):
        hold_position = guard_result.hold_position
        if hold_position is None:
            hold_position = self._current_px4_local_position()
        if not all(math.isfinite(value) for value in hold_position):
            rospy.logerr_throttle(
                1.0,
                "[SRLCReal] Proximity HOLD has no usable PX4-local point",
            )
            self._publish_hover_cmd(reason=reason)
            self._publish_status(reason, np.zeros(3, dtype=np.float32))
            return

        self.safety_stop = True
        self.effective_mode = "PROXIMITY_HOLD"
        self._publish_px4_hold_setpoint(
            target_xy=(hold_position[0], hold_position[1]),
            target_z=hold_position[2],
            reason=reason,
        )
        self._publish_mode_state()
        self._publish_status(reason, np.zeros(3, dtype=np.float32))

    def _publish_status(self, reason, cmd):
        pos = self.odom.pose.pose.position if self.odom is not None else None
        px4_z = (
            float(self.px4_local_odom.pose.pose.position.z)
            if self.px4_local_odom is not None
            else float("nan")
        )
        if pos is None:
            msg = f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | {reason}"
        else:
            msg = (
                f"nokov=[{pos.x:.1f},{pos.y:.1f},{pos.z:.1f}] px4_z={px4_z:.2f} | "
                f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | "
                f"policy_d={self.policy_dist:.2f} clearance={self.clearance:.2f} | "
                f"{self.real_lifecycle_state}/{self.effective_mode} | {reason}"
            )
        self.status_pub.publish(String(data=msg))

    def _publish_px4_hold_setpoint(
        self,
        target_xy=None,
        target_z=None,
        target_velocity=None,
        reason="HOLD",
    ):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))

        if target_z is None:
            if self.px4_local_odom is not None:
                target_z = float(self.px4_local_odom.pose.pose.position.z)
            else:
                target_z = float(self.cfg.takeoff_height)
        if target_xy is None and self.px4_local_odom is not None:
            target_xy = (
                float(self.px4_local_odom.pose.pose.position.x),
                float(self.px4_local_odom.pose.pose.position.y),
            )

        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        if target_xy is not None:
            msg.position.x = float(target_xy[0])
            msg.position.y = float(target_xy[1])
        msg.position.z = float(target_z)
        if target_velocity is not None:
            msg.velocity.z = float(target_velocity[2])
        msg.yaw = self._current_px4_yaw()
        msg.type_mask = (
            PositionTarget.IGNORE_VX
            | PositionTarget.IGNORE_VY
            | PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        if target_velocity is None:
            msg.type_mask |= PositionTarget.IGNORE_VZ
        self.action_pub.publish(msg)

    def _publish_real_lifecycle_hold(
        self,
        reason,
        target_xy=None,
        target_z=None,
        target_velocity=None,
    ):
        self._publish_px4_hold_setpoint(
            target_xy=target_xy,
            target_z=target_z,
            target_velocity=target_velocity,
            reason=reason,
        )
        self._publish_mode_state()
        pos = self.odom.pose.pose.position if self.odom is not None else None
        if pos is None:
            status_msg = f"cmd=[0,0,0] | {self.real_lifecycle_state} | {reason}"
        else:
            z_cmd = float(target_z) if target_z is not None else float(pos.z)
            if target_xy is None:
                target_xy = (float(pos.x), float(pos.y))
            px4_z = (
                float(self.px4_local_odom.pose.pose.position.z)
                if self.px4_local_odom is not None
                else float("nan")
            )
            status_msg = (
                f"nokov=[{pos.x:.1f},{pos.y:.1f},{pos.z:.1f}] px4_z={px4_z:.2f} | "
                f"hold_px4=[{target_xy[0]:.2f},{target_xy[1]:.2f},{z_cmd:.2f}] | "
                f"ff_vz={target_velocity[2] if target_velocity is not None else 0.0:.2f} | "
                f"policy_d={self.policy_dist:.2f} clearance={self.clearance:.2f} | "
                f"{self.real_lifecycle_state}/{self.effective_mode} | {reason}"
            )
        self.status_pub.publish(String(data=status_msg))

    def _publish_hover_cmd(self, reason="HOVER"):
        self.policy_active = False
        self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        self.z_policy_active_pub.publish(Bool(data=False))
        self._publish_policy_cmd(np.zeros(3, dtype=np.float32))

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
        self.effective_mode = "DIRECT"
        self._publish_mode_state()

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

    @staticmethod
    def _quat_to_rot(q):
        return np.array(tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3])

    def _current_yaw(self):
        if self.odom is None:
            return 0.0
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def _current_px4_yaw(self):
        if self.px4_local_odom is None:
            return 0.0
        q = self.px4_local_odom.pose.pose.orientation
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
