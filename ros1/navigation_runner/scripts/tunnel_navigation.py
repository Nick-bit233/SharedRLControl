#!/usr/bin/env python3
"""
ROS1 deployment node for the tunnel obstacle-avoidance RL policy.

Observation pipeline (mirrors IsaacSim training):
    state(10D):        vel_b[3] + ang_vel_b[3] + quat[4]  (body frame)
    human_action(3D):  body-frame velocity from UserModelTunnel
    lidar(1,36,4):     normalised range image from map_manager RayCast service

Action output:
    3D velocity command (world frame) → /CERLAB/quadcopter/cmd_vel  (Gazebo)
                                       or /mavros/setpoint_raw/local (PX4)

Author: Copilot (auto-generated from SharedRLControl training code)
"""
import os
import sys
import math
import time
import threading
import numpy as np

import rospy
import torch
import tf.transformations

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, Float64, String

# Conditional PX4 imports
try:
    from mavros_msgs.msg import PositionTarget, State
    from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
    HAS_MAVROS = True
except ImportError:
    HAS_MAVROS = False

# map_manager RayCast service (kept as fallback, prefer Python raycaster)
try:
    from map_manager.srv import RayCast
    HAS_RAYCAST_SRV = True
except ImportError:
    HAS_RAYCAST_SRV = False

# Add tunnel_deployment package to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.policy_net import TunnelPolicyNet
from tunnel_deployment.user_model import UserModelTunnel
from tunnel_deployment.quat_utils import quat_rotate_inverse
from tunnel_deployment.pcd_raycast import PcdRaycaster


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class TunnelConfig:
    """Configuration for tunnel deployment (mirrors training YAML)."""

    def __init__(self):
        # Sensor
        self.lidar_range = rospy.get_param("~lidar_range", 4.0)
        self.lidar_vfov = rospy.get_param("~lidar_vfov", [-10.0, 20.0])
        self.lidar_vbeams = rospy.get_param("~lidar_vbeams", 4)
        self.lidar_hres = rospy.get_param("~lidar_hres", 10.0)
        self.lidar_hbeams = int(360.0 / self.lidar_hres)
        # PCD-based Python raycaster (bypasses C++ occupancy_map service)
        self.pcd_file = rospy.get_param("~pcd_file", "")
        self.use_pcd_raycast = rospy.get_param("~use_pcd_raycast", True)
        self.lidar_source = str(rospy.get_param("~lidar_source", "internal_pcd")).lower()
        self.lidar_range_image_topic = rospy.get_param(
            "~lidar_range_image_topic", "/srlc/lidar/range_image"
        )
        self.lidar_min_distance_topic = rospy.get_param(
            "~lidar_min_distance_topic", "/srlc/lidar/min_distance"
        )
        self.lidar_timeout = float(rospy.get_param("~lidar_timeout", 0.3))

        # Policy
        self.action_limit = rospy.get_param("~action_limit", 2.0)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "")
        self.device = self._resolve_device(rospy.get_param("~device", "cpu"))

        # Control
        self.control_freq = rospy.get_param("~control_freq", 20.0)
        self.use_px4 = rospy.get_param("~use_px4", False)
        self.height_control = rospy.get_param("~height_control", True)
        self.deterministic = rospy.get_param("~deterministic", True)
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.3))
        self.auto_arm = _param_bool(rospy.get_param("~auto_arm", False))
        self.auto_offboard = _param_bool(rospy.get_param("~auto_offboard", False))
        self.estop_hold_mode = str(rospy.get_param("~estop_hold_mode", "AUTO.LOITER"))
        self.estop_fallback_mode = str(rospy.get_param("~estop_fallback_mode", "POSCTL"))
        self.hold_verify_timeout = float(rospy.get_param("~hold_verify_timeout", 1.0))
        self.hold_on_stop = _param_bool(rospy.get_param("~hold_on_stop", True))
        self.max_xy_speed_real = float(rospy.get_param("~max_xy_speed_real", 1.0))
        self.max_z_speed_real = float(rospy.get_param("~max_z_speed_real", 0.5))
        self.min_altitude = float(rospy.get_param("~min_altitude", 0.3))
        self.max_altitude = float(rospy.get_param("~max_altitude", 5.0))
        self.geofence_x = rospy.get_param("~geofence_x", [])
        self.geofence_y = rospy.get_param("~geofence_y", [])

        # User model / keyboard mode
        self.human_action_source = str(rospy.get_param("~human_action_source", "user_model")).lower()
        self.human_action_topic = rospy.get_param("~human_action_topic", "/srlc/human_action")
        self.assist_enable_topic = rospy.get_param("~assist_enable_topic", "/srlc/assist_enable")
        self.human_action_timeout = float(rospy.get_param("~human_action_timeout", 0.3))
        self.keyboard_mode = rospy.get_param("~keyboard_mode", False)
        self.user_model_simple = rospy.get_param("~user_model_simple", True)
        self.user_model_profile = rospy.get_param("~user_model_profile", "m3_diverse")
        self.user_model_speed = rospy.get_param("~user_model_speed", 2.0)
        self.user_model_freq_base = rospy.get_param("~user_model_freq_base", 0.1)
        self.user_model_freq_scale = rospy.get_param("~user_model_freq_scale", 0.2)
        self.user_model_vx_bias = rospy.get_param("~user_model_vx_bias", 1.5)
        self.user_model_vx_amp = rospy.get_param("~user_model_vx_amp", 0.5)
        self.user_model_vy_amp = rospy.get_param("~user_model_vy_amp", 2.0)
        self.user_model_vz_amp = rospy.get_param("~user_model_vz_amp", 0.2)
        self.user_model_smoothness_base = rospy.get_param("~user_model_smoothness_base", 0.4)
        self.user_model_smoothness_scale = rospy.get_param("~user_model_smoothness_scale", 0.5)
        self.user_model_laziness = rospy.get_param("~user_model_laziness", 0.3)
        self.user_model_seed = int(rospy.get_param("~user_model_seed", 42))
        # Keyboard RL-assist scaling target (fraction of action_limit).
        # Training always used ha_x = action_limit; below ~0.877 * limit the
        # learned residual bias reverses the forward direction.  1.0 = full
        # match (saturated, no safety modulation in X); 0.925 ≈ moderate
        # forward speed with some safety braking capability.
        self.kb_assist_scale = rospy.get_param("~kb_assist_scale", 1.0)

        # Safety
        self.use_safety_shield = rospy.get_param("~use_safety_shield", False)
        self.safety_min_dist = rospy.get_param("~safety_min_dist", 0.3)
        self.collision_dist = rospy.get_param("~collision_dist", 0.05)
        self.safety_start_takeoff_delta = rospy.get_param("~safety_start_takeoff_delta", 0.5)
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.0)
        # Gazebo altitude-hold P-gain. The CERLAB plugin has no built-in
        # altitude hold when receiving cmd_vel; we overlay a P-controller
        # so the drone tracks takeoff_height.
        self.alt_hold_kp = rospy.get_param("~alt_hold_kp", 1.5)
        self.alt_hold_max_vz = rospy.get_param("~alt_hold_max_vz", 2.0)
        self.gazebo_z_mode = str(rospy.get_param("~gazebo_z_mode", "alt_hold")).lower()
        self.gazebo_policy_z_max = float(rospy.get_param("~gazebo_policy_z_max", 2.0))
        self.gazebo_z_blend_alpha = float(rospy.get_param("~gazebo_z_blend_alpha", 0.5))
        self.gazebo_policy_z_takeoff_gate = _param_bool(
            rospy.get_param("~gazebo_policy_z_takeoff_gate", True)
        )
        self.gazebo_policy_z_gate_tolerance = float(
            rospy.get_param("~gazebo_policy_z_gate_tolerance", 0.5)
        )
        self.policy_takeoff_gate = _param_bool(
            rospy.get_param("~policy_takeoff_gate", True)
        )
        self.policy_takeoff_gate_tolerance = float(
            rospy.get_param("~policy_takeoff_gate_tolerance", 0.5)
        )
        # Gazebo max horizontal velocity clamp. The CERLAB velocity PID can
        # pitch excessively when the drone is physically blocked but still
        # receiving high-speed commands ("PID windup"). Clamping prevents
        # tumble from aggressive velocity targets.
        self.gazebo_max_hvel = rospy.get_param("~gazebo_max_hvel", 1.0)
        # Tumble detection: if |roll| or |pitch| exceeds this threshold,
        # the drone is considered tumbled and RL control is paused.
        self.tumble_deg = rospy.get_param("~tumble_deg", 45.0)
        # Tumble recovery: when tumble is detected, switch to setpoint_pose
        # (position hold at current XY, takeoff_height Z, level attitude) so
        # the CERLAB position controller can self-right the drone.
        # Recovery height adds extra clearance to pull the drone off obstacles.
        self.tumble_recover_height = rospy.get_param(
            "~tumble_recover_height", 1.0
        )  # metres above takeoff_height during recovery
        self.tumble_recover_timeout = rospy.get_param(
            "~tumble_recover_timeout", 10.0
        )  # seconds before declaring mission failure

        # Goal detection
        self.goal_x = rospy.get_param("~goal_x", 10.0)  # metres — success threshold

    @staticmethod
    def _resolve_device(requested_device):
        device = str(requested_device).strip()
        if not device.lower().startswith("cuda"):
            return device

        if not torch.cuda.is_available():
            rospy.logwarn(
                "[TunnelNav] Requested device %s, but CUDA is unavailable in this "
                "runtime. Falling back to cpu.",
                device,
            )
            return "cpu"

        return device


class TunnelNavigator:
    """
    Main ROS1 node for tunnel RL policy deployment.

    Adapted from SharedRLControl/ros1/navigation_runner/scripts/navigation.py
    with observation construction rewritten for the tunnel (residual) policy.
    """

    def __init__(self):
        rospy.init_node("tunnel_navigator", anonymous=False)
        self.cfg = TunnelConfig()
        valid_z_modes = {"alt_hold", "policy", "policy_clamped", "blend"}
        if self.cfg.gazebo_z_mode not in valid_z_modes:
            rospy.logfatal(
                "[TunnelNav] Invalid gazebo_z_mode=%s; expected one of %s",
                self.cfg.gazebo_z_mode,
                sorted(valid_z_modes),
            )
            rospy.signal_shutdown("Invalid gazebo_z_mode")
            raise ValueError(f"Invalid gazebo_z_mode: {self.cfg.gazebo_z_mode}")
        valid_lidar_sources = {"internal_pcd", "topic"}
        if self.cfg.lidar_source not in valid_lidar_sources:
            rospy.logfatal(
                "[TunnelNav] Invalid lidar_source=%s; expected one of %s",
                self.cfg.lidar_source,
                sorted(valid_lidar_sources),
            )
            rospy.signal_shutdown("Invalid lidar_source")
            raise ValueError(f"Invalid lidar_source: {self.cfg.lidar_source}")
        valid_human_sources = {"user_model", "keyboard", "rc_topic"}
        if self.cfg.human_action_source not in valid_human_sources:
            rospy.logfatal(
                "[TunnelNav] Invalid human_action_source=%s; expected one of %s",
                self.cfg.human_action_source,
                sorted(valid_human_sources),
            )
            rospy.signal_shutdown("Invalid human_action_source")
            raise ValueError(f"Invalid human_action_source: {self.cfg.human_action_source}")
        if self.cfg.human_action_source == "keyboard":
            self.cfg.keyboard_mode = True

        # ---- Load policy ----
        rospy.loginfo(f"[TunnelNav] Loading checkpoint: {self.cfg.checkpoint_path}")
        rospy.loginfo(f"[TunnelNav] Device: {self.cfg.device}")
        self.policy = TunnelPolicyNet.from_checkpoint(
            self.cfg.checkpoint_path,
            action_limit=self.cfg.action_limit,
            min_concentration=2.0,
            device=self.cfg.device,
        )
        self.policy.eval()
        rospy.loginfo("[TunnelNav] Policy loaded successfully.")

        # ---- PCD Raycaster (bypasses C++ occupancy_map service) ----
        self.pcd_raycaster = None
        if (
            self.cfg.lidar_source == "internal_pcd"
            and self.cfg.use_pcd_raycast
            and self.cfg.pcd_file
        ):
            try:
                self.pcd_raycaster = PcdRaycaster(
                    self.cfg.pcd_file,
                    resolution=0.1,
                    inflate=(0.15, 0.15, 0.05),
                )
                rospy.loginfo("[TunnelNav] Using Python PCD raycaster")
            except Exception as e:
                rospy.logwarn(f"[TunnelNav] PCD raycaster failed: {e}, falling back to C++ service")
                self.pcd_raycaster = None
        if self.cfg.lidar_source == "topic":
            rospy.loginfo(
                "[TunnelNav] Using LiDAR topic input: %s",
                self.cfg.lidar_range_image_topic,
            )
        elif self.pcd_raycaster is None:
            rospy.loginfo("[TunnelNav] Using C++ occupancy_map/raycast service")

        # Log all configuration parameters
        rospy.loginfo(f"[TunnelNav] === Configuration ===")
        rospy.loginfo(f"[TunnelNav]   keyboard_mode  : {self.cfg.keyboard_mode}")
        rospy.loginfo(f"[TunnelNav]   takeoff_height : {self.cfg.takeoff_height} m")
        rospy.loginfo(f"[TunnelNav]   control_freq   : {self.cfg.control_freq} Hz")
        rospy.loginfo(f"[TunnelNav]   action_limit   : {self.cfg.action_limit} m/s")
        rospy.loginfo(f"[TunnelNav]   safety_min_dist: {self.cfg.safety_min_dist} m")
        rospy.loginfo(f"[TunnelNav]   collision_dist : {self.cfg.collision_dist} m")
        if not self.cfg.keyboard_mode:
            rospy.loginfo(
                f"[TunnelNav]   user_model     : "
                f"{'simple' if self.cfg.user_model_simple else self.cfg.user_model_profile} "
                f"@ {self.cfg.user_model_speed} m/s (seed={self.cfg.user_model_seed})"
            )
        rospy.loginfo(f"[TunnelNav]   lidar          : {self.cfg.lidar_hbeams}h x {self.cfg.lidar_vbeams}v, range={self.cfg.lidar_range}m")
        rospy.loginfo(f"[TunnelNav]   deterministic  : {self.cfg.deterministic}")
        rospy.loginfo(f"[TunnelNav]   height_control : {self.cfg.height_control}")
        rospy.loginfo(
            f"[TunnelNav]   gazebo_z_mode : {self.cfg.gazebo_z_mode} "
            f"(policy_z_max={self.cfg.gazebo_policy_z_max}, "
            f"blend_alpha={self.cfg.gazebo_z_blend_alpha}, "
            f"takeoff_gate={self.cfg.gazebo_policy_z_takeoff_gate}, "
            f"gate_tol={self.cfg.gazebo_policy_z_gate_tolerance})"
        )
        rospy.loginfo(
            f"[TunnelNav]   policy_gate   : {self.cfg.policy_takeoff_gate} "
            f"(tol={self.cfg.policy_takeoff_gate_tolerance})"
        )
        rospy.loginfo(f"[TunnelNav]   lidar_source  : {self.cfg.lidar_source}")
        rospy.loginfo(f"[TunnelNav]   human_source  : {self.cfg.human_action_source}")
        if self.cfg.use_px4:
            rospy.loginfo(
                f"[TunnelNav]   PX4 safety    : auto_arm={self.cfg.auto_arm}, "
                f"auto_offboard={self.cfg.auto_offboard}, hold_mode={self.cfg.estop_hold_mode}"
            )
        rospy.loginfo(f"[TunnelNav] ===================")

        # ---- User model OR keyboard input ----
        if self.cfg.keyboard_mode:
            self.user_model = None
            # Latest keyboard command: full TwistStamped for direct relay,
            # and extracted (3,) velocity for RL human_action
            self._kb_twist_msg = TwistStamped()
            self._kb_cmd = np.zeros(3, dtype=np.float32)
            self._kb_cmd_lock = threading.Lock()
            # RL assist starts OFF in keyboard mode — user flies directly first
            self.rl_assist = False
            rospy.loginfo("[TunnelNav] KEYBOARD MODE (DIRECT): use CERLAB keyboard to fly")
            rospy.loginfo("[TunnelNav]   Toggle RL assist: rostopic pub /tunnel_nav/assist_toggle std_msgs/Empty")
        elif self.cfg.human_action_source == "rc_topic":
            self.user_model = None
            self.rl_assist = False
            rospy.loginfo(
                "[TunnelNav] RC TOPIC MODE: waiting for %s and %s",
                self.cfg.human_action_topic,
                self.cfg.assist_enable_topic,
            )
        else:
            self.user_model = UserModelTunnel(
                max_speed=self.cfg.user_model_speed,
                dt=1.0 / self.cfg.control_freq,
                buffer_size=128,
                simple_mode=self.cfg.user_model_simple,
                profile=self.cfg.user_model_profile,
                freq_base=self.cfg.user_model_freq_base,
                freq_scale=self.cfg.user_model_freq_scale,
                vx_bias=self.cfg.user_model_vx_bias,
                vx_amp=self.cfg.user_model_vx_amp,
                vy_amp=self.cfg.user_model_vy_amp,
                vz_amp=self.cfg.user_model_vz_amp,
                smoothness_base=self.cfg.user_model_smoothness_base,
                smoothness_scale=self.cfg.user_model_smoothness_scale,
                laziness=self.cfg.user_model_laziness,
                device=self.cfg.device,
            )
            self.user_model.reset(seed=self.cfg.user_model_seed)
            self.rl_assist = True

        # ---- State ----
        self.odom = None
        self.odom_received = False
        self.last_odom_time = None
        self.raypoints_np = np.empty((0, 3), dtype=np.float32)  # numpy array for speed
        self.topic_lidar_np = None
        self.last_lidar_time = None
        self.last_min_dist_time = None
        self._topic_human_cmd = np.zeros(3, dtype=np.float32)
        self._topic_human_lock = threading.Lock()
        self.last_human_action_time = None
        self.assist_enabled = self.cfg.human_action_source != "rc_topic"
        self._last_hold_mode_request = rospy.Time(0)
        self._hold_mode_request_time = None
        self._hold_mode_requested = None
        self._hold_fallback_requested = False
        self.ready = False  # set True after takeoff + first odom + first raycast
        self.safety_stop = False
        self.collision = False  # True = min_dist < collision_dist → task failed
        self.min_dist = float("inf")  # current minimum obstacle distance
        self.initial_z = None
        self.safety_airborne = False
        self.policy_active = False
        self.z_policy_active = False
        # Tumble recovery state
        self.in_tumble_recovery = False
        self.tumble_recovery_start = None  # rospy.Time when recovery started
        # Goal
        self.goal_reached = False
        self.external_stop = False

        # ---- ROS interfaces ----
        self._setup_ros()

        # ---- Safety thread ----
        safety_thread = threading.Thread(target=self._safety_check, daemon=True)
        safety_thread.start()

        # ---- Takeoff ----
        if self.cfg.keyboard_mode:
            # CERLAB keyboard handles takeoff (Z key) → Gazebo plugin directly
            rospy.loginfo("[TunnelNav] Takeoff: press Z in CERLAB keyboard window")
        else:
            self._takeoff()

        # ---- Raycast timer (runs slightly faster than control) ----
        self.raycast_timer = rospy.Timer(
            rospy.Duration(1.0 / (self.cfg.control_freq * 1.5)),
            self._raycast_callback,
        )

        # ---- Control timer ----
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.cfg.control_freq),
            self._control_callback,
        )

        rospy.loginfo(
            f"[TunnelNav] Running at {self.cfg.control_freq} Hz  "
            f"(PX4={self.cfg.use_px4}, keyboard={self.cfg.keyboard_mode})"
        )

    # ==================================================================
    # ROS Setup
    # ==================================================================
    def _setup_ros(self):
        if self.cfg.use_px4:
            if not HAS_MAVROS:
                rospy.logfatal("[TunnelNav] use_px4=True but mavros_msgs not found!")
                rospy.signal_shutdown("Missing mavros_msgs")
                return
            self.odom_sub = rospy.Subscriber(
                "/mavros/local_position/odom", Odometry, self._odom_cb
            )
            self.action_pub = rospy.Publisher(
                "/mavros/setpoint_raw/local", PositionTarget, queue_size=10
            )
            self.pose_pub = rospy.Publisher(
                "/mavros/setpoint_position/local", PoseStamped, queue_size=10
            )
            self.state_sub = rospy.Subscriber(
                "/mavros/state", State, self._mavros_state_cb
            )
            self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
            self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            self.mavros_state = None
        else:
            self.odom_sub = rospy.Subscriber(
                "/CERLAB/quadcopter/odom", Odometry, self._odom_cb
            )
            self.action_pub = rospy.Publisher(
                "/CERLAB/quadcopter/cmd_vel", TwistStamped, queue_size=10
            )
            self.pose_pub = rospy.Publisher(
                "/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10
            )

        # Visualisation
        self.raycast_vis_pub = rospy.Publisher(
            "/tunnel_nav/raycast_vis", MarkerArray, queue_size=2
        )
        self.lidar_cloud_pub = rospy.Publisher(
            "/tunnel_nav/lidar_cloud", PointCloud2, queue_size=2
        )
        self.cmd_vis_pub = rospy.Publisher(
            "/tunnel_nav/cmd_vel_vis", MarkerArray, queue_size=2
        )
        self.human_cmd_vis_pub = rospy.Publisher(
            "/tunnel_nav/human_cmd_vis", MarkerArray, queue_size=2
        )
        self.status_pub = rospy.Publisher(
            "/tunnel_nav/status", String, queue_size=2
        )
        self.collision_pub = rospy.Publisher(
            "/tunnel_nav/collision", Bool, queue_size=2, latch=True
        )
        self.collision_pub.publish(Bool(data=False))
        self.policy_cmd_pub = rospy.Publisher(
            "/tunnel_nav/policy_cmd", TwistStamped, queue_size=2
        )
        self.policy_active_pub = rospy.Publisher(
            "/tunnel_nav/policy_active", Bool, queue_size=2
        )
        self.z_policy_active_pub = rospy.Publisher(
            "/tunnel_nav/z_policy_active", Bool, queue_size=2
        )
        self.human_cmd_pub = rospy.Publisher(
            "/experiment_control/human_cmd", TwistStamped, queue_size=2
        )
        self.external_stop_sub = rospy.Subscriber(
            "/experiment_control/stop", Bool, self._external_stop_cb, queue_size=1
        )
        if self.cfg.lidar_source == "topic":
            self.lidar_range_sub = rospy.Subscriber(
                self.cfg.lidar_range_image_topic,
                Float32MultiArray,
                self._lidar_range_cb,
                queue_size=1,
            )
            self.lidar_min_dist_sub = rospy.Subscriber(
                self.cfg.lidar_min_distance_topic,
                Float32,
                self._lidar_min_dist_cb,
                queue_size=1,
            )
        if self.cfg.human_action_source == "rc_topic":
            self.human_action_sub = rospy.Subscriber(
                self.cfg.human_action_topic,
                TwistStamped,
                self._human_action_cb,
                queue_size=1,
            )
            self.assist_enable_sub = rospy.Subscriber(
                self.cfg.assist_enable_topic,
                Bool,
                self._assist_enable_cb,
                queue_size=1,
            )

        # Keyboard mode: subscribe to remapped CERLAB keyboard output + assist toggle
        if self.cfg.keyboard_mode:
            self.kb_cmd_sub = rospy.Subscriber(
                "/keyboard/cmd_vel", TwistStamped, self._kb_cmd_cb
            )
            self.assist_toggle_sub = rospy.Subscriber(
                "/tunnel_nav/assist_toggle", Empty, self._assist_toggle_cb
            )
            self.assist_pub = rospy.Publisher(
                "/tunnel_nav/assist_active", Bool, queue_size=2, latch=True
            )
            self.assist_pub.publish(Bool(data=False))

    # ==================================================================
    # Callbacks
    # ==================================================================
    def _odom_cb(self, msg: Odometry):
        self.odom = msg
        self.odom_received = True
        self.last_odom_time = rospy.Time.now()
        if self.initial_z is None:
            self.initial_z = float(msg.pose.pose.position.z)

    def _mavros_state_cb(self, msg):
        self.mavros_state = msg

    def _kb_cmd_cb(self, msg: TwistStamped):
        """Keyboard mode: receive CERLAB keyboard cmd_vel (remapped topic)."""
        with self._kb_cmd_lock:
            self._kb_twist_msg = msg
            self._kb_cmd[0] = msg.twist.linear.x
            self._kb_cmd[1] = msg.twist.linear.y
            self._kb_cmd[2] = msg.twist.linear.z

    def _assist_toggle_cb(self, msg):
        """Keyboard mode: toggle RL assist on/off."""
        self.rl_assist = not self.rl_assist
        self.assist_pub.publish(Bool(data=self.rl_assist))
        mode_str = "RL ASSIST" if self.rl_assist else "DIRECT"
        rospy.loginfo(f"[TunnelNav] Mode → {mode_str}")

    def _external_stop_cb(self, msg):
        self.external_stop = bool(msg.data)

    def _lidar_range_cb(self, msg):
        expected = self.cfg.lidar_hbeams * self.cfg.lidar_vbeams
        if len(msg.data) != expected:
            rospy.logwarn_throttle(
                2.0,
                "[TunnelNav] Ignoring LiDAR range image with %d values; expected %d",
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

    def _human_action_cb(self, msg):
        with self._topic_human_lock:
            self._topic_human_cmd[0] = msg.twist.linear.x
            self._topic_human_cmd[1] = msg.twist.linear.y
            self._topic_human_cmd[2] = msg.twist.linear.z
        self.last_human_action_time = rospy.Time.now()

    def _assist_enable_cb(self, msg):
        self.assist_enabled = bool(msg.data)

    # ==================================================================
    # Raycast (LiDAR simulation — Python PCD raycaster or C++ service)
    # ==================================================================
    def _raycast_callback(self, event):
        if self.cfg.lidar_source != "internal_pcd":
            return
        if not self.odom_received:
            return
        pos = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.pcd_raycaster is not None:
            # Python PCD raycaster — deterministic, no sensor-update corruption
            pts = self.pcd_raycaster.raycast(
                [pos.x, pos.y, pos.z],
                yaw,
                self.cfg.lidar_range,
                self.cfg.lidar_vfov[0],
                self.cfg.lidar_vfov[1],
                self.cfg.lidar_vbeams,
                self.cfg.lidar_hres,
            )
            self.raypoints_np = pts
            if not self.ready and pts.shape[0] > 0:
                self.ready = True
            self._publish_lidar_cloud(pts)
        elif HAS_RAYCAST_SRV:
            # Fallback: C++ occupancy_map/raycast service
            try:
                raycast = rospy.ServiceProxy("occupancy_map/raycast", RayCast)
                pos_msg = Point(x=pos.x, y=pos.y, z=pos.z)
                resp = raycast(
                    pos_msg,
                    yaw,
                    self.cfg.lidar_range,
                    self.cfg.lidar_vfov[0],
                    self.cfg.lidar_vfov[1],
                    self.cfg.lidar_vbeams,
                    self.cfg.lidar_hres,
                )
                pts_flat = np.array(resp.points, dtype=np.float32)
                n_pts = len(pts_flat) // 3
                if n_pts > 0:
                    self.raypoints_np = pts_flat.reshape(n_pts, 3)
                else:
                    self.raypoints_np = np.empty((0, 3), dtype=np.float32)
                if not self.ready and n_pts > 0:
                    self.ready = True
                self._publish_lidar_cloud(self.raypoints_np)
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(5.0, f"[TunnelNav] RayCast service error: {e}")

    # ==================================================================
    # Build Observation  (mirrors _compute_state_and_obs in env_tunnel.py)
    # ==================================================================
    def _build_obs(self):
        """
        Returns: (state, human_action, lidar) tensors on self.cfg.device
          state:        (1, 10)  [vel_b(3), ang_vel_b(3), quat(4)]
          human_action: (1, 3)   body-frame velocity
          lidar:        (1, 1, 36, 4) normalised
        """
        dev = self.cfg.device
        odom = self.odom

        # --- Quaternion [w, x, y, z] ---
        q_ros = odom.pose.pose.orientation
        quat_np = np.array([[q_ros.w, q_ros.x, q_ros.y, q_ros.z]], dtype=np.float32)
        quat = torch.from_numpy(quat_np).to(device=dev)

        # --- World-frame velocity ---
        rot = self._quat_to_rot(q_ros)
        vel_body_np = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ], dtype=np.float64)
        vel_world_np = (rot @ vel_body_np).astype(np.float32)
        vel_w = torch.from_numpy(vel_world_np).unsqueeze(0).to(device=dev)  # (1,3)

        ang_vel_np = np.array([
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ], dtype=np.float64)
        ang_vel_world_np = (rot @ ang_vel_np).astype(np.float32)
        ang_vel_w = torch.from_numpy(ang_vel_world_np).unsqueeze(0).to(device=dev)

        # Body-frame velocities
        vel_b = quat_rotate_inverse(quat, vel_w)        # (1, 3)
        ang_vel_b = quat_rotate_inverse(quat, ang_vel_w)  # (1, 3)

        # State: [vel_b(3), ang_vel_b(3), quat(4)]
        state = torch.cat([vel_b, ang_vel_b, quat], dim=-1)  # (1, 10)

        # --- Human action (from user model or keyboard) ---
        if self.cfg.keyboard_mode:
            with self._kb_cmd_lock:
                kb_np = self._kb_cmd.copy()
            if self.rl_assist:
                # With Beta distribution, the residual operates linearly in [0,1]
                # space — no crossover/reversal issue. Pass keyboard input as-is.
                pass
            human_action = torch.from_numpy(kb_np).unsqueeze(0).to(device=dev)
        elif self.cfg.human_action_source == "rc_topic":
            with self._topic_human_lock:
                cmd_np = self._topic_human_cmd.copy()
            human_action = torch.from_numpy(cmd_np).unsqueeze(0).to(device=dev)
        else:
            human_action = self.user_model.step()  # (1, 3) body-frame

        # --- LiDAR ---
        if self.cfg.lidar_source == "topic" and self.topic_lidar_np is not None:
            lidar_scan = torch.from_numpy(self.topic_lidar_np.copy()).to(device=dev)
        else:
            pos_np = np.array([[odom.pose.pose.position.x,
                                odom.pose.pose.position.y,
                                odom.pose.pose.position.z]], dtype=np.float32)

            expected = self.cfg.lidar_hbeams * self.cfg.lidar_vbeams
            ray_np = self.raypoints_np
            if ray_np.shape[0] == expected:
                dists = np.linalg.norm(ray_np - pos_np, axis=-1).clip(max=self.cfg.lidar_range)
                lidar_np = ((self.cfg.lidar_range - dists) / self.cfg.lidar_range).astype(np.float32)
                lidar_scan = torch.from_numpy(lidar_np).reshape(
                    1, 1, self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
                ).to(device=dev)
            else:
                lidar_scan = torch.zeros(
                    1, 1, self.cfg.lidar_hbeams, self.cfg.lidar_vbeams,
                    device=dev, dtype=torch.float32,
                )

        return state, human_action, lidar_scan

    # ==================================================================
    # Main Control Loop
    # ==================================================================
    def _real_policy_gate_ok(self):
        if not self.cfg.use_px4:
            return True, "OK"
        now = rospy.Time.now()
        if self.external_stop:
            return False, "EXTERNAL_STOP"
        if self.mavros_state is None or not self.mavros_state.connected:
            return False, "MAVROS_NOT_CONNECTED"
        if self.last_odom_time is None:
            return False, "NO_ODOM"
        if (now - self.last_odom_time).to_sec() > self.cfg.odom_timeout:
            return False, "ODOM_TIMEOUT"
        if self.cfg.lidar_source == "topic":
            if self.last_lidar_time is None:
                return False, "NO_LIDAR"
            if (now - self.last_lidar_time).to_sec() > self.cfg.lidar_timeout:
                return False, "LIDAR_TIMEOUT"
        if self.cfg.human_action_source == "rc_topic":
            if self.last_human_action_time is None:
                return False, "NO_RC_ACTION"
            if (now - self.last_human_action_time).to_sec() > self.cfg.human_action_timeout:
                return False, "RC_ACTION_TIMEOUT"
            if not self.assist_enabled:
                return False, "ASSIST_DISABLED"
        if not self.ready:
            return False, "NOT_READY"
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

    def _control_callback(self, event):
        if not self.odom_received:
            return

        if self.cfg.use_px4:
            gate_ok, gate_reason = self._real_policy_gate_ok()
            if not gate_ok:
                self._publish_stop(reason=gate_reason)
                pos = self.odom.pose.pose.position
                status_msg = (
                    f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                    f"cmd=[0,0,0] | min_d={self.min_dist:.2f} | {gate_reason}"
                )
                self.status_pub.publish(String(data=status_msg))
                return

        # --- Keyboard DIRECT mode: relay CERLAB keyboard → drone, no RL ---
        if self.cfg.keyboard_mode and not self.rl_assist:
            with self._kb_cmd_lock:
                relay_msg = TwistStamped()
                relay_msg.header.stamp = rospy.Time.now()
                relay_msg.twist = self._kb_twist_msg.twist
                kb_cmd = self._kb_cmd.copy()
            self.action_pub.publish(relay_msg)
            self._publish_human_cmd(kb_cmd)
            # Publish status for monitoring
            pos = self.odom.pose.pose.position
            v = relay_msg.twist.linear
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[{v.x:.2f},{v.y:.2f},{v.z:.2f}] | "
                f"min_d={self.min_dist:.2f} | DIRECT"
            )
            self.status_pub.publish(String(data=status_msg))
            return

        # --- RL mode (auto or keyboard+RL assist) ---
        if not self.ready:
            # Keep publishing takeoff pose while waiting for first raycast
            self.pose_pub.publish(self._make_takeoff_pose())
            self.policy_active = False
            self.policy_active_pub.publish(Bool(data=False))
            self.z_policy_active = False
            self.z_policy_active_pub.publish(Bool(data=False))
            return

        if self.external_stop:
            self._publish_stop()
            pos = self.odom.pose.pose.position
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[0,0,0] | min_d={self.min_dist:.2f} | EXTERNAL_STOP"
            )
            self.status_pub.publish(String(data=status_msg))
            return

        # --- Goal check (before collision — reaching goal takes priority) ---
        if not self.goal_reached and self.odom is not None:
            cur_x = self.odom.pose.pose.position.x
            if cur_x >= self.cfg.goal_x:
                self.goal_reached = True
                rospy.loginfo(
                    f"[TunnelNav] *** GOAL REACHED *** x={cur_x:.1f} >= "
                    f"{self.cfg.goal_x:.1f}"
                )

        if self.goal_reached:
            self._publish_stop()
            pos = self.odom.pose.pose.position
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[0,0,0] | min_d={self.min_dist:.2f} | GOAL_REACHED"
            )
            self.status_pub.publish(String(data=status_msg))
            return

        if self.collision:
            self._publish_stop()
            pos = self.odom.pose.pose.position
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[0,0,0] | min_d={self.min_dist:.2f} | COLLISION"
            )
            self.status_pub.publish(String(data=status_msg))
            return

        if self.safety_stop:
            # Reset tumble recovery — safety stop uses pose hold which
            # actively stabilises attitude.
            self.in_tumble_recovery = False
            self.tumble_recovery_start = None
            self._publish_stop()
            pos = self.odom.pose.pose.position
            status_msg = (
                f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                f"cmd=[0,0,0] | min_d={self.min_dist:.2f} | SAFETY_STOP"
            )
            self.status_pub.publish(String(data=status_msg))
            return

        if (
            (not self.cfg.keyboard_mode)
            and self.cfg.policy_takeoff_gate
            and not self.policy_active
        ):
            cur_z = float(self.odom.pose.pose.position.z)
            gate_height = self.cfg.takeoff_height - self.cfg.policy_takeoff_gate_tolerance
            if cur_z < gate_height:
                self.pose_pub.publish(self._make_takeoff_pose())
                self.policy_active_pub.publish(Bool(data=False))
                self.z_policy_active = False
                self.z_policy_active_pub.publish(Bool(data=False))
                rospy.loginfo_throttle(
                    2.0,
                    f"[TunnelNav] Holding takeoff pose until policy gate: "
                    f"z={cur_z:.2f} < {gate_height:.2f}",
                )
                pos = self.odom.pose.pose.position
                status_msg = (
                    f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
                    f"cmd=[pose_hold] | min_d={self.min_dist:.2f} | POLICY_GATE"
                )
                self.status_pub.publish(String(data=status_msg))
                return
            self.policy_active = True
            rospy.loginfo(
                f"[TunnelNav] Policy control enabled at z={cur_z:.2f} "
                f"(gate={gate_height:.2f})"
            )
        if not self.policy_active:
            self.policy_active = True
        self.policy_active_pub.publish(Bool(data=True))

        # 1. Build observation
        state, human_action, lidar = self._build_obs()
        human_action_np = human_action.squeeze(0).detach().cpu().numpy()
        self._publish_human_cmd(human_action_np)

        # 2. Policy inference
        with torch.no_grad():
            action_world = self.policy(
                state, human_action, lidar,
                deterministic=self.cfg.deterministic,
            )  # (1, 3) world-frame m/s
        cmd = action_world.squeeze(0).cpu().numpy()

        # Debug: log first 15 iterations + every 200th
        if not hasattr(self, '_ctrl_iter'):
            self._ctrl_iter = 0
        self._ctrl_iter += 1
        if self._ctrl_iter <= 15 or self._ctrl_iter % 200 == 0:
            ha = human_action_np
            s = state.squeeze(0).cpu().numpy()
            L = lidar.squeeze().cpu().numpy()  # (36, 4)
            # Directional lidar averages: F=forward(bins 0,1,34,35), L=left(8,9,10), R=right(26,27,28), B=back(17,18,19)
            fwd_m = L[[0,1,34,35], :].mean()
            left_m = L[[8,9,10], :].mean()
            right_m = L[[26,27,28], :].mean()
            back_m = L[[17,18,19], :].mean()
            rospy.loginfo(
                f"[TunnelNav] iter={self._ctrl_iter}: "
                f"ha=[{ha[0]:.2f},{ha[1]:.2f},{ha[2]:.2f}] "
                f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] "
                f"vel_b=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f}] "
                f"lidar F={fwd_m:.3f} L={left_m:.3f} R={right_m:.3f} B={back_m:.3f} "
                f"[min={L.min():.3f} max={L.max():.3f}]"
            )

        # 3. Publish
        self._publish_policy_cmd(cmd)
        self._publish_cmd(cmd)

        # 4. Visualise
        self._publish_vis(cmd, human_action_np)

        # 5. Publish status
        pos = self.odom.pose.pose.position
        mode_str = "RL" if not self.cfg.keyboard_mode else "KB+RL"
        status_msg = (
            f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
            f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | "
            f"min_d={self.min_dist:.2f} | {mode_str} | "
            f"safe={'NO' if self.safety_stop else 'OK'}"
        )
        self.status_pub.publish(String(data=status_msg))

    # ==================================================================
    # Command publishers
    # ==================================================================
    def _clamp_px4_cmd(self, cmd_vel_world: np.ndarray) -> np.ndarray:
        cmd = np.asarray(cmd_vel_world, dtype=np.float32).copy()
        hspeed = math.hypot(float(cmd[0]), float(cmd[1]))
        if hspeed > self.cfg.max_xy_speed_real:
            scale = self.cfg.max_xy_speed_real / hspeed
            cmd[0] *= scale
            cmd[1] *= scale
        cmd[2] = np.clip(cmd[2], -self.cfg.max_z_speed_real, self.cfg.max_z_speed_real)
        return cmd

    def _publish_cmd(self, cmd_vel_world: np.ndarray):
        if self.cfg.use_px4:
            cmd_vel_world = self._clamp_px4_cmd(cmd_vel_world)
            msg = PositionTarget()
            msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.velocity.x = float(cmd_vel_world[0])
            msg.velocity.y = float(cmd_vel_world[1])
            if self.cfg.height_control:
                msg.velocity.z = float(cmd_vel_world[2])
                msg.yaw = self._current_yaw()
                msg.type_mask = (
                    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                    | PositionTarget.IGNORE_PZ
                    | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                    | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
                )
            else:
                msg.position.z = self.cfg.takeoff_height
                msg.yaw = self._current_yaw()
                msg.type_mask = (
                    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                    | PositionTarget.IGNORE_VZ
                    | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                    | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
                )
            self.action_pub.publish(msg)
        else:
            # Gazebo: velocity in heading-aligned body frame
            q = self.odom.pose.pose.orientation
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )

            # Tumble detection & recovery: if roll or pitch exceed the
            # threshold the drone has hit an obstacle. Switch to setpoint_pose
            # (position-hold with level attitude) so the CERLAB position
            # controller can self-right the drone.
            tumble_rad = math.radians(self.cfg.tumble_deg)
            is_tumbled = abs(roll) > tumble_rad or abs(pitch) > tumble_rad

            if is_tumbled and not self.in_tumble_recovery:
                # Just entered tumble — start recovery
                self.in_tumble_recovery = True
                self.tumble_recovery_start = rospy.Time.now()
                rospy.logwarn(
                    f"[TunnelNav] TUMBLE DETECTED roll={math.degrees(roll):.1f}° "
                    f"pitch={math.degrees(pitch):.1f}° — starting pose recovery"
                )

            if self.in_tumble_recovery:
                # Check recovery timeout
                elapsed = (rospy.Time.now() - self.tumble_recovery_start).to_sec()
                if elapsed > self.cfg.tumble_recover_timeout:
                    rospy.logerr(
                        f"[TunnelNav] Tumble recovery timeout ({elapsed:.1f}s) "
                        "— declaring collision"
                    )
                    self.collision = True
                    self.collision_pub.publish(Bool(data=True))
                    self._publish_stop()
                    return

                # Send setpoint_pose: current XY, elevated Z, level attitude
                cur_pos = self.odom.pose.pose.position
                recover_z = self.cfg.takeoff_height + self.cfg.tumble_recover_height
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "world"
                pose_msg.pose.position.x = cur_pos.x
                pose_msg.pose.position.y = cur_pos.y
                pose_msg.pose.position.z = recover_z
                pose_msg.pose.orientation.w = 1.0  # level attitude
                self.pose_pub.publish(pose_msg)

                # Check if recovered (pitch & roll back within half threshold)
                recover_thresh = tumble_rad * 0.5
                if abs(roll) < recover_thresh and abs(pitch) < recover_thresh:
                    rospy.loginfo(
                        f"[TunnelNav] Tumble RECOVERED in {elapsed:.1f}s "
                        f"(roll={math.degrees(roll):.1f}°, "
                        f"pitch={math.degrees(pitch):.1f}°)"
                    )
                    self.in_tumble_recovery = False
                    self.tumble_recovery_start = None
                else:
                    rospy.logwarn_throttle(
                        2.0,
                        f"[TunnelNav] Recovering from tumble... "
                        f"roll={math.degrees(roll):.1f}° "
                        f"pitch={math.degrees(pitch):.1f}° "
                        f"({elapsed:.1f}s elapsed)"
                    )
                return

            rot = np.array([
                [math.cos(yaw), math.sin(yaw), 0],
                [-math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ])
            cmd_local = rot @ cmd_vel_world

            # Clamp horizontal velocity to prevent PID windup in Gazebo.
            max_h = self.cfg.gazebo_max_hvel
            hspeed = math.hypot(cmd_local[0], cmd_local[1])
            if hspeed > max_h:
                scale = max_h / hspeed
                cmd_local[0] *= scale
                cmd_local[1] *= scale

            # Altitude hold: P-controller tracks takeoff_height.
            cur_z = self.odom.pose.pose.position.z
            alt_err = self.cfg.takeoff_height - cur_z
            vz_hold = np.clip(
                self.cfg.alt_hold_kp * alt_err,
                -self.cfg.alt_hold_max_vz,
                self.cfg.alt_hold_max_vz,
            )
            policy_vz = float(cmd_local[2])
            policy_vz_clamped = np.clip(
                policy_vz,
                -self.cfg.gazebo_policy_z_max,
                self.cfg.gazebo_policy_z_max,
            )
            z_mode = str(self.cfg.gazebo_z_mode).lower()
            use_policy_z = True
            if z_mode == "alt_hold":
                use_policy_z = False
                self.z_policy_active = False
            elif not self.cfg.gazebo_policy_z_takeoff_gate:
                self.z_policy_active = True
            elif self.cfg.gazebo_policy_z_takeoff_gate and not self.z_policy_active:
                gate_height = self.cfg.takeoff_height - self.cfg.gazebo_policy_z_gate_tolerance
                if cur_z >= gate_height:
                    self.z_policy_active = True
                    rospy.loginfo(
                        f"[TunnelNav] Gazebo policy z enabled at z={cur_z:.2f} "
                        f"(gate={gate_height:.2f}, mode={z_mode})"
                    )
                else:
                    use_policy_z = False
                    rospy.loginfo_throttle(
                        2.0,
                        f"[TunnelNav] Holding altitude until z policy gate: "
                        f"z={cur_z:.2f} < {gate_height:.2f} (mode={z_mode})",
                    )

            if not use_policy_z:
                vz_cmd = vz_hold
            elif z_mode == "policy":
                vz_cmd = policy_vz
            elif z_mode == "policy_clamped":
                vz_cmd = policy_vz_clamped
            elif z_mode == "blend":
                alpha = float(np.clip(self.cfg.gazebo_z_blend_alpha, 0.0, 1.0))
                vz_cmd = alpha * policy_vz_clamped + (1.0 - alpha) * vz_hold
            elif z_mode == "alt_hold":
                vz_cmd = vz_hold
            else:
                rospy.logwarn_throttle(
                    5.0,
                    f"[TunnelNav] Unknown gazebo_z_mode={self.cfg.gazebo_z_mode}; using alt_hold",
                )
                vz_cmd = vz_hold
                self.z_policy_active = False

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x = float(cmd_local[0])
            msg.twist.linear.y = float(cmd_local[1])
            msg.twist.linear.z = float(vz_cmd)
            self.action_pub.publish(msg)
            self.z_policy_active_pub.publish(Bool(data=bool(self.z_policy_active)))

    def _request_px4_hold_mode(self, reason="STOP"):
        if not self.cfg.use_px4 or not self.cfg.hold_on_stop:
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
                    "[TunnelNav] PX4 hold mode request was rejected: mode=%s reason=%s",
                    self.cfg.estop_hold_mode,
                    reason,
                )
                self._request_px4_fallback_mode(reason=reason)
            else:
                self._hold_mode_request_time = now
                self._hold_mode_requested = self.cfg.estop_hold_mode
                self._hold_fallback_requested = False
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(
                2.0,
                "[TunnelNav] PX4 hold mode request failed: %s",
                exc,
            )
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
                    "[TunnelNav] Requested PX4 fallback mode %s after stop reason=%s",
                    self.cfg.estop_fallback_mode,
                    reason,
                )
            else:
                rospy.logerr(
                    "[TunnelNav] PX4 rejected fallback mode %s after stop reason=%s",
                    self.cfg.estop_fallback_mode,
                    reason,
                )
        except rospy.ServiceException as exc:
            rospy.logerr("[TunnelNav] PX4 fallback mode request failed: %s", exc)

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
                "[TunnelNav] PX4 stop mode not confirmed: requested=%s current=%s "
                "reason=%s; pilot should be ready for manual takeover.",
                self._hold_mode_requested,
                self.mavros_state.mode,
                reason,
            )
            self._request_px4_fallback_mode(reason=reason)

    def _publish_stop(self, reason="STOP"):
        """Hover in place with level attitude.

        For Gazebo (not PX4) we use setpoint_pose rather than cmd_vel
        so the CERLAB position controller actively stabilises both
        position and attitude — plain zero-velocity cmd_vel doesn't
        fight attitude drift as effectively.
        """
        self.policy_active = False
        if hasattr(self, "policy_active_pub"):
            self.policy_active_pub.publish(Bool(data=False))
        self.z_policy_active = False
        if hasattr(self, "z_policy_active_pub"):
            self.z_policy_active_pub.publish(Bool(data=False))
        if self.cfg.use_px4:
            self._request_px4_hold_mode(reason=reason)
            msg = PositionTarget()
            msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.velocity.z = 0.0
            msg.yaw = self._current_yaw() if self.odom is not None else 0.0
            msg.type_mask = (
                PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_PZ
                | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
            )
            self.action_pub.publish(msg)
            return
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        if self.odom is not None:
            pose_msg.pose.position.x = self.odom.pose.pose.position.x
            pose_msg.pose.position.y = self.odom.pose.pose.position.y
        pose_msg.pose.position.z = self.cfg.takeoff_height
        pose_msg.pose.orientation.w = 1.0  # level attitude
        self.pose_pub.publish(pose_msg)

    def _publish_human_cmd(self, cmd_body: np.ndarray):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd_body[0])
        msg.twist.linear.y = float(cmd_body[1])
        msg.twist.linear.z = float(cmd_body[2])
        self.human_cmd_pub.publish(msg)

    def _publish_policy_cmd(self, cmd_world: np.ndarray):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = float(cmd_world[0])
        msg.twist.linear.y = float(cmd_world[1])
        msg.twist.linear.z = float(cmd_world[2])
        self.policy_cmd_pub.publish(msg)

    # ==================================================================
    # Takeoff
    # ==================================================================
    def _takeoff(self):
        rospy.loginfo("[TunnelNav] Waiting for odometry...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.odom_received:
            rate.sleep()

        rospy.loginfo(f"[TunnelNav] Taking off to height {self.cfg.takeoff_height} m")
        if self.cfg.use_px4:
            # Wait for MAVROS connection
            while not rospy.is_shutdown() and self.mavros_state is None:
                rate.sleep()
            if not self.cfg.auto_arm and not self.cfg.auto_offboard:
                rospy.loginfo(
                    "[TunnelNav] PX4 real mode: auto_arm=false and auto_offboard=false; "
                    "waiting for pilot/GCS to manage arming and mode."
                )
                return
            # PX4 requires setpoints before OFFBOARD can be accepted.
            if self.cfg.auto_offboard:
                for _ in range(100):
                    self.pose_pub.publish(self._make_takeoff_pose())
                    rate.sleep()
            try:
                if self.cfg.auto_offboard:
                    resp = self.set_mode_client(SetModeRequest(custom_mode="OFFBOARD"))
                    if not resp.mode_sent:
                        rospy.logwarn("[TunnelNav] PX4 rejected OFFBOARD mode request")
                if self.cfg.auto_arm:
                    resp = self.arming_client(CommandBoolRequest(value=True))
                    if not resp.success:
                        rospy.logwarn("[TunnelNav] PX4 rejected arming request")
            except rospy.ServiceException as exc:
                rospy.logwarn("[TunnelNav] PX4 takeoff service call failed: %s", exc)
        else:
            # Gazebo: send takeoff command (std_msgs/Empty), then hover
            takeoff_pub = rospy.Publisher(
                "/CERLAB/quadcopter/takeoff", Empty, queue_size=1
            )
            rospy.sleep(0.5)
            takeoff_pub.publish(Empty())
            rospy.loginfo("[TunnelNav] Takeoff command sent, waiting 3s...")
            rospy.sleep(3.0)

    def _make_takeoff_pose(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        if self.odom:
            msg.pose.position.x = self.odom.pose.pose.position.x
            msg.pose.position.y = self.odom.pose.pose.position.y
        msg.pose.position.z = self.cfg.takeoff_height
        msg.pose.orientation.w = 1.0
        return msg

    # ==================================================================
    # Safety
    # ==================================================================
    def _apply_proximity_safety(self, min_dist):
        if min_dist < self.cfg.collision_dist:
            self.collision = True
            self.safety_stop = True
            self.collision_pub.publish(Bool(data=True))
            rospy.logerr(
                f"[TunnelNav] COLLISION! min_dist={min_dist:.3f} m "
                f"(threshold={self.cfg.collision_dist} m) — task failed"
            )
        elif min_dist < self.cfg.safety_min_dist:
            if not self.safety_stop:
                rospy.logwarn(f"[TunnelNav] SAFETY STOP! min_dist={min_dist:.2f} m")
            self.safety_stop = True
        else:
            if self.safety_stop:
                rospy.loginfo(f"[TunnelNav] Safety cleared, min_dist={min_dist:.2f} m")
            self.safety_stop = False

    def _safety_check(self):
        """Background thread: monitor obstacle proximity, detect collisions."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.collision:
                rate.sleep()
                continue

            ray_np = self.raypoints_np
            if self.odom_received and (self.pcd_raycaster is not None or ray_np.shape[0] > 0):
                pos = np.array([
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                ], dtype=np.float32)
                if self.initial_z is None:
                    self.initial_z = float(pos[2])
                if not self.safety_airborne:
                    if pos[2] >= self.initial_z + self.cfg.safety_start_takeoff_delta:
                        self.safety_airborne = True
                        rospy.loginfo(
                            f"[TunnelNav] Safety monitor airborne at z={pos[2]:.2f} "
                            f"(baseline={self.initial_z:.2f})"
                        )
                    else:
                        self.min_dist = float("inf")
                        self.safety_stop = False
                        rate.sleep()
                        continue
                if self.pcd_raycaster is not None:
                    min_dist = self.pcd_raycaster.nearest_distance(pos)
                else:
                    # Fallback for the C++ raycast service path. The preferred
                    # PCD path uses true map proximity to avoid stale ray hits.
                    dists = np.linalg.norm(ray_np - pos, axis=-1)
                    min_dist = float(dists.min())
                self.min_dist = min_dist
                self._apply_proximity_safety(min_dist)
            elif (
                self.cfg.lidar_source == "topic"
                and self.odom_received
                and self.last_min_dist_time is not None
            ):
                if (rospy.Time.now() - self.last_min_dist_time).to_sec() > self.cfg.lidar_timeout:
                    if not self.safety_stop:
                        rospy.logwarn("[TunnelNav] SAFETY STOP! min_distance topic timeout")
                    self.safety_stop = True
                    rate.sleep()
                    continue
                pos_z = float(self.odom.pose.pose.position.z)
                if self.initial_z is None:
                    self.initial_z = pos_z
                if not self.safety_airborne:
                    if pos_z >= self.initial_z + self.cfg.safety_start_takeoff_delta:
                        self.safety_airborne = True
                        rospy.loginfo(
                            f"[TunnelNav] Safety monitor airborne at z={pos_z:.2f} "
                            f"(baseline={self.initial_z:.2f})"
                        )
                    else:
                        rate.sleep()
                        continue
                self._apply_proximity_safety(self.min_dist)
            rate.sleep()

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _quat_to_rot(q) -> np.ndarray:
        """ROS Quaternion → 3×3 rotation matrix."""
        return np.array(
            tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        )

    def _current_yaw(self) -> float:
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    # ==================================================================
    # LiDAR PointCloud2 Visualisation
    # ==================================================================
    def _publish_lidar_cloud(self, pts_np: np.ndarray):
        """Publish raycast hit-points as PointCloud2 for RViz."""
        if pts_np.shape[0] == 0:
            return
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = pts_np.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pts_np.shape[0]
        msg.is_dense = True
        msg.data = pts_np.astype(np.float32).tobytes()
        self.lidar_cloud_pub.publish(msg)

    # ==================================================================
    # Visualisation
    # ==================================================================
    def _publish_vis(self, cmd_world: np.ndarray, human_cmd_body: np.ndarray):
        """Publish arrow markers for RL command and human command."""
        pos = self.odom.pose.pose.position

        # RL command (green)
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
        rl_marker.points.append(Point(
            x=pos.x + cmd_world[0],
            y=pos.y + cmd_world[1],
            z=pos.z + cmd_world[2],
        ))

        ma = MarkerArray()
        ma.markers.append(rl_marker)
        self.cmd_vis_pub.publish(ma)

        # Human command (blue) — convert body→world for vis
        q = self.odom.pose.pose.orientation
        rot = self._quat_to_rot(q)
        human_world = rot @ human_cmd_body
        h_marker = Marker()
        h_marker.header.frame_id = "map"
        h_marker.header.stamp = rospy.Time.now()
        h_marker.ns = "human_cmd"
        h_marker.id = 1
        h_marker.type = Marker.ARROW
        h_marker.action = Marker.ADD
        h_marker.scale.x = 0.05
        h_marker.scale.y = 0.1
        h_marker.scale.z = 0.1
        h_marker.color.b = 1.0
        h_marker.color.a = 1.0
        h_marker.points.append(Point(x=pos.x, y=pos.y, z=pos.z))
        h_marker.points.append(Point(
            x=pos.x + human_world[0],
            y=pos.y + human_world[1],
            z=pos.z + human_world[2],
        ))

        ma2 = MarkerArray()
        ma2.markers.append(h_marker)
        self.human_cmd_vis_pub.publish(ma2)


# ======================================================================
# Entry point
# ======================================================================
def main():
    try:
        nav = TunnelNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
