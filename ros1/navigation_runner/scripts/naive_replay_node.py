#!/usr/bin/env python3
"""Direct user-command replay baseline for tunnel experiments."""

import math
import os
import sys

import numpy as np
import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.pcd_io import read_pcd_xyz  # noqa: E402
from tunnel_deployment.user_model import UserModelTunnel  # noqa: E402

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class NaiveReplayNode:
    def __init__(self):
        rospy.init_node("naive_replay_node", anonymous=False)

        self.method = str(rospy.get_param("~method", "naive_raw")).lower()
        self.enable_safety = _param_bool(rospy.get_param("~enable_safety", False))
        self.rate_hz = float(rospy.get_param("~user_model_rate", 20.0))
        self.device = rospy.get_param("~device", "cpu")
        self.pcd_file = rospy.get_param("~pcd_file", "")
        self.takeoff_height = float(rospy.get_param("~takeoff_height", 5.0))
        self.takeoff_wait = float(rospy.get_param("~takeoff_wait", 3.0))
        self.takeoff_gate_tolerance = float(
            rospy.get_param("~policy_takeoff_gate_tolerance", 0.5)
        )
        self.gazebo_max_hvel = float(rospy.get_param("~gazebo_max_hvel", 2.0))
        self.gazebo_z_mode = str(rospy.get_param("~gazebo_z_mode", "alt_hold")).lower()
        self.alt_hold_kp = float(rospy.get_param("~alt_hold_kp", 1.5))
        self.alt_hold_max_vz = float(rospy.get_param("~alt_hold_max_vz", 2.0))
        self.gazebo_policy_z_max = float(rospy.get_param("~gazebo_policy_z_max", 2.0))
        self.gazebo_z_blend_alpha = float(rospy.get_param("~gazebo_z_blend_alpha", 0.5))
        self.gazebo_policy_z_takeoff_gate = _param_bool(
            rospy.get_param("~gazebo_policy_z_takeoff_gate", True)
        )
        self.gazebo_policy_z_gate_tolerance = float(
            rospy.get_param("~gazebo_policy_z_gate_tolerance", 0.5)
        )
        self.safety_min_dist = float(rospy.get_param("~safety_min_dist", 0.35))
        self.collision_dist = float(rospy.get_param("~collision_dist", 0.20))
        self.safety_mode = str(rospy.get_param("~safety_mode", "recover")).lower()
        if self.method not in ("naive_raw", "naive_safe"):
            rospy.logfatal("[NaiveReplay] invalid method=%s", self.method)
            rospy.signal_shutdown("Invalid naive method")
            raise ValueError(f"Invalid naive method: {self.method}")
        if self.safety_mode not in ("hold", "recover"):
            rospy.logfatal("[NaiveReplay] invalid safety_mode=%s", self.safety_mode)
            rospy.signal_shutdown("Invalid safety mode")
            raise ValueError(f"Invalid safety mode: {self.safety_mode}")
        self.safety_recover_speed = float(rospy.get_param("~safety_recover_speed", 0.35))
        self.safety_recover_forward_speed = float(
            rospy.get_param("~safety_recover_forward_speed", 0.15)
        )
        self.safety_recover_centerline_gain = float(
            rospy.get_param("~safety_recover_centerline_gain", 0.4)
        )
        self.safety_start_takeoff_delta = float(
            rospy.get_param("~safety_start_takeoff_delta", 0.5)
        )

        self.user_model = self._build_user_model()
        self.obstacle_points = None
        self.obstacle_tree = None
        if self.enable_safety:
            self._load_safety_map()

        self.odom = None
        self.odom_received = False
        self.initial_z = None
        self.safety_airborne = False
        self.stop_requested = False
        self.collision = False
        self.z_policy_active = False
        self.min_dist = float("inf")
        self.nearest_point = None

        self.odom_sub = rospy.Subscriber(
            "/CERLAB/quadcopter/odom_raw", Odometry, self._odom_cb, queue_size=1
        )
        self.stop_sub = rospy.Subscriber(
            "/experiment_control/stop", Bool, self._stop_cb, queue_size=1
        )
        self.cmd_pub = rospy.Publisher(
            "/CERLAB/quadcopter/cmd_vel", TwistStamped, queue_size=10
        )
        self.pose_pub = rospy.Publisher(
            "/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10
        )
        self.takeoff_pub = rospy.Publisher(
            "/CERLAB/quadcopter/takeoff", Empty, queue_size=1
        )
        self.human_cmd_pub = rospy.Publisher(
            "/experiment_control/human_cmd", TwistStamped, queue_size=10
        )
        self.collision_pub = rospy.Publisher(
            "/naive_replay/collision", Bool, queue_size=2, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/naive_replay/status", String, queue_size=2
        )
        self.collision_pub.publish(Bool(data=False))

        rospy.loginfo(
            "[NaiveReplay] Ready. method=%s safety=%s mode=%s z_mode=%s rate=%.1fHz",
            self.method,
            self.enable_safety,
            self.safety_mode,
            self.gazebo_z_mode,
            self.rate_hz,
        )

    def _build_user_model(self):
        kwargs = dict(
            max_speed=float(rospy.get_param("~user_model_speed", 2.0)),
            dt=1.0 / max(self.rate_hz, 1e-6),
            buffer_size=128,
            simple_mode=_param_bool(rospy.get_param("~user_model_simple", False)),
            profile=rospy.get_param("~user_model_profile", "m3_diverse"),
            freq_base=float(rospy.get_param("~user_model_freq_base", 0.1)),
            freq_scale=float(rospy.get_param("~user_model_freq_scale", 0.2)),
            vx_bias=float(rospy.get_param("~user_model_vx_bias", 1.5)),
            vx_amp=float(rospy.get_param("~user_model_vx_amp", 0.5)),
            vy_amp=float(rospy.get_param("~user_model_vy_amp", 2.0)),
            vz_amp=float(rospy.get_param("~user_model_vz_amp", 0.2)),
            smoothness_base=float(rospy.get_param("~user_model_smoothness_base", 0.4)),
            smoothness_scale=float(rospy.get_param("~user_model_smoothness_scale", 0.5)),
            laziness=float(rospy.get_param("~user_model_laziness", 0.3)),
            input_source=str(rospy.get_param("~input_source", "offline")).lower(),
            replay_dataset_path=rospy.get_param("~replay_dataset_path", ""),
            replay_dataset_format=rospy.get_param("~replay_dataset_format", "hdf5"),
            replay_sampling_mode=rospy.get_param("~replay_sampling_mode", "raw"),
            replay_trajectory_index=int(rospy.get_param("~replay_trajectory_index", -1)),
            replay_start_offset=int(rospy.get_param("~replay_start_offset", -1)),
            replay_loop=_param_bool(rospy.get_param("~replay_loop", True)),
        )
        seed = int(rospy.get_param("~user_model_seed", 42))
        try:
            model = UserModelTunnel(device=self.device, **kwargs)
            model.reset(seed=seed)
        except Exception as exc:
            rospy.logwarn(
                "[NaiveReplay] UserModel init on %s failed (%s). Falling back to cpu.",
                self.device,
                exc,
            )
            self.device = "cpu"
            model = UserModelTunnel(device="cpu", **kwargs)
            model.reset(seed=seed)
        return model

    def _load_safety_map(self):
        if not self.pcd_file:
            rospy.logfatal("[NaiveReplay] naive_safe requires ~pcd_file")
            rospy.signal_shutdown("Missing safety map")
            return
        points = read_pcd_xyz(self.pcd_file)
        if len(points) == 0:
            rospy.logfatal("[NaiveReplay] Safety PCD is empty: %s", self.pcd_file)
            rospy.signal_shutdown("Empty safety map")
            return
        self.obstacle_points = np.asarray(points, dtype=np.float32)
        self.obstacle_tree = cKDTree(self.obstacle_points) if HAS_SCIPY else None
        rospy.loginfo(
            "[NaiveReplay] Loaded safety map %s (%d points, scipy=%s)",
            self.pcd_file,
            len(self.obstacle_points),
            HAS_SCIPY,
        )

    def _odom_cb(self, msg):
        self.odom = msg
        self.odom_received = True
        if self.initial_z is None:
            self.initial_z = float(msg.pose.pose.position.z)

    def _stop_cb(self, msg):
        self.stop_requested = bool(msg.data)

    def _current_pos(self):
        p = self.odom.pose.pose.position
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    def _current_yaw(self):
        q = self.odom.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def _wait_for_odom(self):
        rospy.loginfo("[NaiveReplay] Waiting for odometry...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.odom_received:
            rate.sleep()

    def _make_hold_pose(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        if self.odom is not None:
            pos = self.odom.pose.pose.position
            msg.pose.position.x = pos.x
            msg.pose.position.y = pos.y
        msg.pose.position.z = self.takeoff_height
        msg.pose.orientation.w = 1.0
        return msg

    def _takeoff(self):
        self._wait_for_odom()
        if rospy.is_shutdown():
            return

        rospy.sleep(0.5)
        self.takeoff_pub.publish(Empty())
        rospy.loginfo("[NaiveReplay] Takeoff command sent; holding pose before replay")

        rate = rospy.Rate(20)
        gate_z = self.takeoff_height - self.takeoff_gate_tolerance
        min_wait_until = rospy.Time.now() + rospy.Duration(max(0.0, self.takeoff_wait))
        while not rospy.is_shutdown():
            z = float(self.odom.pose.pose.position.z) if self.odom is not None else 0.0
            if rospy.Time.now() >= min_wait_until and z >= gate_z:
                break
            self.pose_pub.publish(self._make_hold_pose())
            rospy.loginfo_throttle(
                2.0,
                "[NaiveReplay] Waiting for replay gate: z=%.2f >= %.2f",
                z,
                gate_z,
            )
            rate.sleep()

    def _nearest_obstacle(self):
        if self.obstacle_points is None or self.odom is None:
            return None, float("inf")
        pos = self._current_pos()
        if self.obstacle_tree is not None:
            dist, idx = self.obstacle_tree.query(pos)
            return self.obstacle_points[int(idx)].copy(), float(dist)
        diffs = self.obstacle_points - pos.reshape(1, 3)
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists_sq))
        return self.obstacle_points[idx].copy(), float(math.sqrt(dists_sq[idx]))

    def _publish_human_cmd(self, cmd_body):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd_body[0])
        msg.twist.linear.y = float(cmd_body[1])
        msg.twist.linear.z = float(cmd_body[2])
        self.human_cmd_pub.publish(msg)

    def _publish_cmd_body(self, cmd_body):
        cmd = np.asarray(cmd_body, dtype=np.float32).copy()
        hspeed = math.hypot(float(cmd[0]), float(cmd[1]))
        if hspeed > self.gazebo_max_hvel:
            scale = self.gazebo_max_hvel / hspeed
            cmd[0] *= scale
            cmd[1] *= scale

        cur_z = float(self.odom.pose.pose.position.z) if self.odom is not None else self.takeoff_height
        alt_err = self.takeoff_height - cur_z
        vz_hold = float(np.clip(self.alt_hold_kp * alt_err, -self.alt_hold_max_vz, self.alt_hold_max_vz))
        replay_vz = float(cmd[2])
        replay_vz_clamped = float(np.clip(replay_vz, -self.gazebo_policy_z_max, self.gazebo_policy_z_max))
        z_mode = self.gazebo_z_mode
        use_replay_z = True

        if z_mode == "alt_hold":
            use_replay_z = False
            self.z_policy_active = False
        elif not self.gazebo_policy_z_takeoff_gate:
            self.z_policy_active = True
        elif not self.z_policy_active:
            gate_z = self.takeoff_height - self.gazebo_policy_z_gate_tolerance
            if cur_z >= gate_z:
                self.z_policy_active = True
            else:
                use_replay_z = False

        if not use_replay_z:
            cmd[2] = vz_hold
        elif z_mode == "policy":
            cmd[2] = replay_vz
        elif z_mode == "policy_clamped":
            cmd[2] = replay_vz_clamped
        elif z_mode == "blend":
            alpha = float(np.clip(self.gazebo_z_blend_alpha, 0.0, 1.0))
            cmd[2] = alpha * replay_vz_clamped + (1.0 - alpha) * vz_hold
        elif z_mode != "alt_hold":
            rospy.logwarn_throttle(
                5.0,
                "[NaiveReplay] Unknown gazebo_z_mode=%s; using alt_hold",
                self.gazebo_z_mode,
            )
            cmd[2] = vz_hold
            self.z_policy_active = False

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.linear.z = float(cmd[2])
        self.cmd_pub.publish(msg)
        return cmd

    def _compute_recover_cmd_body(self):
        if self.odom is None:
            return np.zeros(3, dtype=np.float32)

        pos = self._current_pos()
        world_xy = np.array([self.safety_recover_forward_speed, 0.0], dtype=np.float32)
        if self.nearest_point is not None:
            away = pos[:2] - np.asarray(self.nearest_point[:2], dtype=np.float32)
            norm = float(np.linalg.norm(away))
            if norm > 1e-4:
                world_xy += (away / norm) * self.safety_recover_speed
        world_xy[1] += -self.safety_recover_centerline_gain * float(pos[1])

        norm = float(np.linalg.norm(world_xy))
        if norm > self.safety_recover_speed:
            world_xy *= self.safety_recover_speed / norm

        yaw = self._current_yaw()
        world_to_body = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]],
            dtype=np.float32,
        )
        body_xy = world_to_body @ world_xy
        return np.array([body_xy[0], body_xy[1], 0.0], dtype=np.float32)

    def _apply_safety(self):
        if not self.enable_safety:
            return "raw", None

        if not self.safety_airborne:
            if self.initial_z is not None and self.odom is not None:
                z = float(self.odom.pose.pose.position.z)
                if z >= self.initial_z + self.safety_start_takeoff_delta:
                    self.safety_airborne = True
                else:
                    self.min_dist = float("inf")
                    return "raw", None

        nearest, min_dist = self._nearest_obstacle()
        self.nearest_point = nearest
        self.min_dist = min_dist

        if min_dist < self.collision_dist:
            if not self.collision:
                rospy.logerr(
                    "[NaiveReplay] COLLISION min_dist=%.3f threshold=%.3f",
                    min_dist,
                    self.collision_dist,
                )
            self.collision = True
            self.collision_pub.publish(Bool(data=True))
            return "collision", None
        if min_dist < self.safety_min_dist:
            if self.safety_mode == "recover":
                return "recover", self._compute_recover_cmd_body()
            return "hold", None

        self.collision_pub.publish(Bool(data=False))
        return "raw", None

    def _control_cb(self, event):
        if not self.odom_received:
            return
        if self.stop_requested or self.collision:
            self.pose_pub.publish(self._make_hold_pose())
            return

        raw_cmd = self.user_model.step().squeeze(0).detach().cpu().numpy()
        raw_cmd = np.asarray(raw_cmd, dtype=np.float32)
        self._publish_human_cmd(raw_cmd)

        safety_state, safety_cmd = self._apply_safety()
        if safety_state in ("collision", "hold"):
            self.pose_pub.publish(self._make_hold_pose())
            executed = np.zeros(3, dtype=np.float32)
        elif safety_state == "recover":
            executed = self._publish_cmd_body(safety_cmd)
        else:
            executed = self._publish_cmd_body(raw_cmd)

        pos = self.odom.pose.pose.position
        self.status_pub.publish(String(data=(
            f"x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} | "
            f"raw=[{raw_cmd[0]:.2f},{raw_cmd[1]:.2f},{raw_cmd[2]:.2f}] | "
            f"cmd=[{executed[0]:.2f},{executed[1]:.2f},{executed[2]:.2f}] | "
            f"min_d={self.min_dist:.2f} | {self.method}:{safety_state}"
        )))

    def run(self):
        self._takeoff()
        if rospy.is_shutdown():
            return
        rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1e-6)), self._control_cb)
        rospy.loginfo("[NaiveReplay] Replay started")
        rospy.spin()


if __name__ == "__main__":
    NaiveReplayNode().run()
