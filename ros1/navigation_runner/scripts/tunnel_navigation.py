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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Empty, Float64

# Conditional PX4 imports
try:
    from mavros_msgs.msg import PositionTarget, State
    from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
    HAS_MAVROS = True
except ImportError:
    HAS_MAVROS = False

# map_manager RayCast service
from map_manager.srv import RayCast

# Add tunnel_deployment package to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.policy_net import TunnelPolicyNet
from tunnel_deployment.user_model import UserModelTunnel
from tunnel_deployment.quat_utils import quat_rotate_inverse


class TunnelConfig:
    """Configuration for tunnel deployment (mirrors training YAML)."""

    def __init__(self):
        # Sensor
        self.lidar_range = rospy.get_param("~lidar_range", 4.0)
        self.lidar_vfov = rospy.get_param("~lidar_vfov", [-10.0, 20.0])
        self.lidar_vbeams = rospy.get_param("~lidar_vbeams", 4)
        self.lidar_hres = rospy.get_param("~lidar_hres", 10.0)
        self.lidar_hbeams = int(360.0 / self.lidar_hres)

        # Policy
        self.action_limit = rospy.get_param("~action_limit", 2.0)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "")
        self.device = rospy.get_param("~device", "cpu")

        # Control
        self.control_freq = rospy.get_param("~control_freq", 20.0)
        self.use_px4 = rospy.get_param("~use_px4", False)
        self.height_control = rospy.get_param("~height_control", True)
        self.deterministic = rospy.get_param("~deterministic", True)

        # User model
        self.user_model_simple = rospy.get_param("~user_model_simple", True)
        self.user_model_speed = rospy.get_param("~user_model_speed", 2.0)
        self.user_model_freq_base = rospy.get_param("~user_model_freq_base", 0.1)
        self.user_model_freq_scale = rospy.get_param("~user_model_freq_scale", 0.3)

        # Safety
        self.use_safety_shield = rospy.get_param("~use_safety_shield", False)
        self.safety_min_dist = rospy.get_param("~safety_min_dist", 0.3)
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.0)


class TunnelNavigator:
    """
    Main ROS1 node for tunnel RL policy deployment.

    Adapted from SharedRLControl/ros1/navigation_runner/scripts/navigation.py
    with observation construction rewritten for the tunnel (residual) policy.
    """

    def __init__(self):
        rospy.init_node("tunnel_navigator", anonymous=False)
        self.cfg = TunnelConfig()

        # ---- Load policy ----
        rospy.loginfo(f"[TunnelNav] Loading checkpoint: {self.cfg.checkpoint_path}")
        rospy.loginfo(f"[TunnelNav] Device: {self.cfg.device}")
        self.policy = TunnelPolicyNet.from_checkpoint(
            self.cfg.checkpoint_path,
            action_limit=self.cfg.action_limit,
            device=self.cfg.device,
        )
        self.policy.eval()
        rospy.loginfo("[TunnelNav] Policy loaded successfully.")

        # ---- User model (human-action generator) ----
        self.user_model = UserModelTunnel(
            max_speed=self.cfg.user_model_speed,
            dt=1.0 / self.cfg.control_freq,
            buffer_size=128,
            simple_mode=self.cfg.user_model_simple,
            freq_base=self.cfg.user_model_freq_base,
            freq_scale=self.cfg.user_model_freq_scale,
            device=self.cfg.device,
        )
        self.user_model.reset(seed=42)

        # ---- State ----
        self.odom = None
        self.odom_received = False
        self.raypoints = []
        self.ready = False  # set True after takeoff + first odom + first raycast
        self.safety_stop = False

        # ---- ROS interfaces ----
        self._setup_ros()

        # ---- Safety thread ----
        safety_thread = threading.Thread(target=self._safety_check, daemon=True)
        safety_thread.start()

        # ---- Takeoff ----
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
            f"(PX4={self.cfg.use_px4}, safety_shield={self.cfg.use_safety_shield})"
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
        self.cmd_vis_pub = rospy.Publisher(
            "/tunnel_nav/cmd_vel_vis", MarkerArray, queue_size=2
        )
        self.human_cmd_vis_pub = rospy.Publisher(
            "/tunnel_nav/human_cmd_vis", MarkerArray, queue_size=2
        )

    # ==================================================================
    # Callbacks
    # ==================================================================
    def _odom_cb(self, msg: Odometry):
        self.odom = msg
        self.odom_received = True

    def _mavros_state_cb(self, msg):
        self.mavros_state = msg

    # ==================================================================
    # Raycast (LiDAR simulation via map_manager)
    # ==================================================================
    def _raycast_callback(self, event):
        if not self.odom_received:
            return
        pos = self.odom.pose.pose.position
        # Compute start angle from current yaw
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        try:
            raycast = rospy.ServiceProxy("occupancy_map/raycast", RayCast)
            pos_msg = Point(x=pos.x, y=pos.y, z=pos.z)
            resp = raycast(
                pos_msg,
                yaw,  # start_angle: drone heading
                self.cfg.lidar_range,
                self.cfg.lidar_vfov[0],
                self.cfg.lidar_vfov[1],
                self.cfg.lidar_vbeams,
                self.cfg.lidar_hres,
            )
            n_pts = len(resp.points) // 3
            pts = []
            for i in range(n_pts):
                pts.append([
                    resp.points[3 * i],
                    resp.points[3 * i + 1],
                    resp.points[3 * i + 2],
                ])
            self.raypoints = pts
            if not self.ready and len(pts) > 0:
                self.ready = True
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
        quat = torch.tensor(
            [[q_ros.w, q_ros.x, q_ros.y, q_ros.z]], device=dev, dtype=torch.float32
        )

        # --- World-frame velocity ---
        # Gazebo quadcopterPlugin publishes body-frame twist directly.
        # For PX4/mavros, twist is in body frame too (local_position/odom).
        # But the original NavRL code does rot @ vel_body → vel_world → then
        # the training env does quat_rotate_inverse to get vel_b.
        #
        # Strategy: extract world-frame velocity, then rotate to body frame.
        rot = self._quat_to_rot(q_ros)
        vel_body_np = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ])
        vel_world_np = rot @ vel_body_np
        vel_w = torch.tensor([vel_world_np], device=dev, dtype=torch.float32)  # (1,3)

        ang_vel_np = np.array([
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ])
        ang_vel_world_np = rot @ ang_vel_np
        ang_vel_w = torch.tensor([ang_vel_world_np], device=dev, dtype=torch.float32)

        # Body-frame velocities
        vel_b = quat_rotate_inverse(quat, vel_w)        # (1, 3)
        ang_vel_b = quat_rotate_inverse(quat, ang_vel_w)  # (1, 3)

        # State: [vel_b(3), ang_vel_b(3), quat(4)]
        state = torch.cat([vel_b, ang_vel_b, quat], dim=-1)  # (1, 10)

        # --- Human action (from user model) ---
        human_action = self.user_model.step()  # (1, 3) body-frame

        # --- LiDAR ---
        pos = torch.tensor(
            [[odom.pose.pose.position.x,
              odom.pose.pose.position.y,
              odom.pose.pose.position.z]],
            device=dev, dtype=torch.float32,
        )

        if len(self.raypoints) == self.cfg.lidar_hbeams * self.cfg.lidar_vbeams:
            ray_pts = torch.tensor(self.raypoints, device=dev, dtype=torch.float32)
            distances = (ray_pts - pos).norm(dim=-1).clamp_max(self.cfg.lidar_range)
            lidar_scan = (self.cfg.lidar_range - distances) / self.cfg.lidar_range
            lidar_scan = lidar_scan.reshape(
                1, 1, self.cfg.lidar_hbeams, self.cfg.lidar_vbeams
            )
        else:
            # Fallback: empty scan (max range → all zeros after normalisation)
            lidar_scan = torch.zeros(
                1, 1, self.cfg.lidar_hbeams, self.cfg.lidar_vbeams,
                device=dev, dtype=torch.float32,
            )

        return state, human_action, lidar_scan

    # ==================================================================
    # Main Control Loop
    # ==================================================================
    def _control_callback(self, event):
        if not self.odom_received:
            return

        if not self.ready:
            # Keep publishing takeoff pose while waiting for first raycast
            self.pose_pub.publish(self._make_takeoff_pose())
            return

        if self.safety_stop:
            self._publish_stop()
            return

        # 1. Build observation
        state, human_action, lidar = self._build_obs()

        # 2. Policy inference
        with torch.no_grad():
            action_world = self.policy(
                state, human_action, lidar,
                deterministic=self.cfg.deterministic,
            )  # (1, 3) world-frame m/s

        cmd = action_world.squeeze(0).cpu().numpy()

        # 3. Publish
        self._publish_cmd(cmd)

        # 4. Visualise
        self._publish_vis(cmd, human_action.squeeze(0).cpu().numpy())

    # ==================================================================
    # Command publishers
    # ==================================================================
    def _publish_cmd(self, cmd_vel_world: np.ndarray):
        if self.cfg.use_px4:
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
            # Gazebo: velocity in body frame
            q = self.odom.pose.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            rot = np.array([
                [math.cos(yaw), math.sin(yaw), 0],
                [-math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ])
            cmd_local = rot @ cmd_vel_world

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x = float(cmd_local[0])
            msg.twist.linear.y = float(cmd_local[1])
            msg.twist.linear.z = float(cmd_vel_world[2]) if self.cfg.height_control else 0.0
            self.action_pub.publish(msg)

    def _publish_stop(self):
        if self.cfg.use_px4:
            self.pose_pub.publish(self._make_takeoff_pose())
        else:
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            self.action_pub.publish(msg)

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
            # Arm + OFFBOARD
            for _ in range(100):
                self.pose_pub.publish(self._make_takeoff_pose())
                rate.sleep()
            try:
                self.set_mode_client(SetModeRequest(custom_mode="OFFBOARD"))
                self.arming_client(CommandBoolRequest(value=True))
            except rospy.ServiceException:
                pass
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
    def _safety_check(self):
        """Background thread: emergency stop if too close to obstacles."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.odom_received and len(self.raypoints) > 0:
                pos = np.array([
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                ])
                min_dist = float("inf")
                for pt in self.raypoints:
                    d = np.linalg.norm(np.array(pt) - pos)
                    min_dist = min(min_dist, d)
                if min_dist < self.cfg.safety_min_dist:
                    if not self.safety_stop:
                        rospy.logwarn(
                            f"[TunnelNav] SAFETY STOP! min_dist={min_dist:.2f} m"
                        )
                    self.safety_stop = True
                else:
                    self.safety_stop = False
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
