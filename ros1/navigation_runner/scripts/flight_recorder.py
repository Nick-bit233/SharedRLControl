#!/usr/bin/env python3
"""Flight data recorder with optional auto-termination supervision."""

import json
import math
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class FlightRecorder:
    def __init__(self):
        rospy.init_node('flight_recorder', anonymous=False)

        self.rate = rospy.get_param('~rate', 50.0)
        self.output_dir = rospy.get_param('~output_dir', '/tmp/flight_data')
        self.method = rospy.get_param('~method', 'unknown')
        self.trial_id = rospy.get_param('~trial_id', 0)
        self.auto_start = rospy.get_param('~auto_start', True)
        self.goal_x = rospy.get_param('~goal_x', 15.0)
        self.collision_dist = rospy.get_param('~collision_dist', 0.05)
        self.collision_topic = str(rospy.get_param('~collision_topic', '')).strip()
        self.pcd_file = rospy.get_param('~pcd_file', '')
        self.auto_terminate = rospy.get_param('~auto_terminate', True)
        self.shutdown_on_complete = rospy.get_param('~shutdown_on_complete', True)
        self.timeout_sec = float(rospy.get_param('~timeout_sec', 0.0))
        self.completion_grace_period = float(
            rospy.get_param('~completion_grace_period', 0.5)
        )
        self.auto_start_takeoff_delta = float(
            rospy.get_param('~auto_start_takeoff_delta', 0.5)
        )
        self.batch_idx = int(rospy.get_param('~batch_idx', -1))
        self.run_idx = int(rospy.get_param('~run_idx', -1))
        run_id_param = str(rospy.get_param('~run_id', '')).strip()
        self.run_id = run_id_param or f'{self.method}_trial{int(self.trial_id):03d}'
        self.map_seed = int(rospy.get_param('~map_seed', -1))
        self.user_model_seed = int(rospy.get_param('~user_model_seed', -1))
        self.tunnel_world = rospy.get_param('~tunnel_world', '')

        self._init_buffers()
        self.recording = False
        self.start_time = None
        self.latest_odom = None
        self.latest_cmd = None
        self.latest_human_cmd = None
        self.reached_goal = False
        self.collision = False
        self.data_saved = False
        self.termination_reason = ""
        self.termination_time = None
        self.finalize_timer = None
        self.shutdown_requested = False
        self.min_obstacle_dist = float('inf')
        self.initial_z = None
        self.airborne = False
        self.external_collision = False
        self.obstacle_tree = self._load_obstacle_tree()

        os.makedirs(self.output_dir, exist_ok=True)

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.cmd_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/cmd_vel', TwistStamped, self._cmd_cb, queue_size=1)
        self.human_cmd_sub = rospy.Subscriber(
            '/experiment_control/human_cmd', TwistStamped, self._human_cmd_cb, queue_size=1)
        self.collision_sub = None
        if self.collision_topic:
            self.collision_sub = rospy.Subscriber(
                self.collision_topic, Bool, self._collision_cb, queue_size=1
            )

        self.start_sub = rospy.Subscriber(
            '/flight_recorder/start', Bool, self._start_cb, queue_size=1)
        self.stop_sub = rospy.Subscriber(
            '/flight_recorder/stop', Bool, self._stop_cb, queue_size=1)
        self.stop_pub = rospy.Publisher(
            '/experiment_control/stop', Bool, queue_size=1, latch=True)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._record_cb)
        rospy.on_shutdown(self._shutdown_cb)

        if self.auto_start:
            rospy.Timer(rospy.Duration(2.0), self._auto_start_cb, oneshot=True)

        rospy.loginfo("[Recorder] Ready. method=%s, trial=%d, rate=%.0fHz",
                      self.method, self.trial_id, self.rate)

    def _init_buffers(self):
        self.buffers = {
            'timestamps': [], 'position': [], 'orientation': [],
            'velocity': [], 'cmd_vel': [], 'cmd_vel_world': [],
            'human_cmd_body': [], 'human_cmd_world': [],
            'angular_vel': [], 'yaw': [],
            'min_obstacle_dist': [], 'min_obstacle_dist_monitored': [],
            'collision_flags': [],
        }

    def _load_obstacle_tree(self):
        if not self.pcd_file:
            return None
        if not HAS_SCIPY:
            if self.auto_terminate:
                rospy.logfatal(
                    "[Recorder] scipy is required for collision monitoring when "
                    "auto_terminate=true"
                )
                rospy.signal_shutdown("Missing scipy for collision monitoring")
            else:
                rospy.logwarn("[Recorder] scipy unavailable; collision monitoring disabled")
            return None

        points = []
        header_done = False
        try:
            with open(self.pcd_file, 'r') as handle:
                for line in handle:
                    if not header_done:
                        if line.startswith('DATA'):
                            header_done = True
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except FileNotFoundError:
            rospy.logfatal("[Recorder] PCD file not found: %s", self.pcd_file)
            rospy.signal_shutdown("Missing collision map")
            return None

        if not points:
            rospy.logwarn("[Recorder] PCD file %s is empty; collision monitoring disabled", self.pcd_file)
            return None

        tree = cKDTree(np.asarray(points, dtype=np.float32))
        rospy.loginfo(
            "[Recorder] Loaded collision map: %s (%d points)",
            self.pcd_file,
            len(points),
        )
        return tree

    def _odom_cb(self, msg):
        self.latest_odom = msg
        if self.initial_z is None:
            self.initial_z = float(msg.pose.pose.position.z)

    def _cmd_cb(self, msg):
        self.latest_cmd = msg

    def _human_cmd_cb(self, msg):
        self.latest_human_cmd = msg

    def _collision_cb(self, msg):
        self.external_collision = bool(msg.data)
        if not self.external_collision or not self.recording or self.termination_reason:
            return

        t = (rospy.Time.now() - self.start_time).to_sec() if self.start_time is not None else 0.0
        min_dist = self.min_obstacle_dist if np.isfinite(self.min_obstacle_dist) else float('inf')
        rospy.logerr(
            "[Recorder] External collision topic triggered at t=%.2fs (%s)",
            t,
            self.collision_topic,
        )
        self._request_termination('collision', t, min_dist)

    def _start_cb(self, msg):
        if msg.data and not self.recording:
            self._start_recording()

    def _stop_cb(self, msg):
        if msg.data and self.recording:
            self._stop_recording()

    def _auto_start_cb(self, event):
        if not self.recording and self.latest_odom is not None:
            self._start_recording()
        elif not self.recording:
            rospy.Timer(rospy.Duration(0.5), self._auto_start_cb, oneshot=True)

    def _start_recording(self):
        self.recording = True
        self.start_time = rospy.Time.now()
        self._init_buffers()
        self.reached_goal = False
        self.collision = False
        self.data_saved = False
        self.termination_reason = ""
        self.termination_time = None
        self.shutdown_requested = False
        self.finalize_timer = None
        self.min_obstacle_dist = float('inf')
        self.airborne = False
        self.external_collision = False
        self.stop_pub.publish(Bool(data=False))
        rospy.loginfo("[Recorder] STARTED (%s trial %d)", self.method, self.trial_id)

    def _stop_recording(self):
        self.recording = False
        self._save_data()
        rospy.loginfo("[Recorder] STOPPED. %d samples", len(self.buffers['timestamps']))

    @staticmethod
    def _local_to_world(cmd_local, yaw):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return np.array([
            cos_yaw * cmd_local[0] - sin_yaw * cmd_local[1],
            sin_yaw * cmd_local[0] + cos_yaw * cmd_local[1],
            cmd_local[2],
        ], dtype=np.float32)

    def _request_termination(self, reason, t, min_dist):
        if self.termination_reason:
            return

        self.termination_reason = reason
        self.termination_time = float(t)
        self.reached_goal = reason == 'goal_reached'
        self.collision = reason == 'collision'
        self.stop_pub.publish(Bool(data=True))

        rospy.logwarn(
            "[Recorder] Termination requested: reason=%s t=%.2fs min_dist=%.3f",
            reason,
            t,
            min_dist,
        )

        if self.completion_grace_period > 0.0:
            self.finalize_timer = rospy.Timer(
                rospy.Duration(self.completion_grace_period),
                self._finalize_termination,
                oneshot=True,
            )
        else:
            self._finalize_termination(None)

    def _finalize_termination(self, _event):
        if self.recording:
            self._stop_recording()
        if self.shutdown_on_complete and not self.shutdown_requested:
            self.shutdown_requested = True
            rospy.signal_shutdown(
                f"experiment complete: {self.termination_reason or 'external_stop'}"
            )

    def _record_cb(self, event):
        if not self.recording or self.latest_odom is None:
            return

        odom = self.latest_odom
        t = (rospy.Time.now() - self.start_time).to_sec()

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        v = odom.twist.twist.linear
        w = odom.twist.twist.angular
        current_z = float(p.z)

        if self.initial_z is None:
            self.initial_z = current_z
        if (not self.airborne) and current_z >= self.initial_z + self.auto_start_takeoff_delta:
            self.airborne = True
            rospy.loginfo(
                "[Recorder] Airborne detected at z=%.2f (baseline=%.2f)",
                current_z,
                self.initial_z,
            )

        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        cmd_local = np.zeros(3, dtype=np.float32)
        if self.latest_cmd is not None:
            c = self.latest_cmd.twist.linear
            cmd_local[:] = (c.x, c.y, c.z)
        cmd_world = self._local_to_world(cmd_local, yaw)

        human_cmd_body = np.zeros(3, dtype=np.float32)
        if self.latest_human_cmd is not None:
            h = self.latest_human_cmd.twist.linear
            human_cmd_body[:] = (h.x, h.y, h.z)
        human_cmd_world = self._local_to_world(human_cmd_body, yaw)

        min_dist = float('inf')
        monitored_min_dist = float('inf')
        collision_flag = False
        if self.obstacle_tree is not None:
            min_dist = float(self.obstacle_tree.query([p.x, p.y, p.z])[0])
            monitored_min_dist = min_dist if self.airborne else float('inf')
            collision_flag = monitored_min_dist < self.collision_dist
            self.min_obstacle_dist = min(self.min_obstacle_dist, monitored_min_dist)

        self.buffers['timestamps'].append(t)
        self.buffers['position'].append([p.x, p.y, p.z])
        self.buffers['orientation'].append([q.x, q.y, q.z, q.w])
        self.buffers['velocity'].append([v.x, v.y, v.z])
        self.buffers['cmd_vel'].append(cmd_local.tolist())
        self.buffers['cmd_vel_world'].append(cmd_world.tolist())
        self.buffers['human_cmd_body'].append(human_cmd_body.tolist())
        self.buffers['human_cmd_world'].append(human_cmd_world.tolist())
        self.buffers['angular_vel'].append([w.x, w.y, w.z])
        self.buffers['yaw'].append(yaw)
        self.buffers['min_obstacle_dist'].append(min_dist)
        self.buffers['min_obstacle_dist_monitored'].append(monitored_min_dist)
        self.buffers['collision_flags'].append(collision_flag)

        if self.auto_terminate and not self.termination_reason:
            if p.x >= self.goal_x:
                rospy.loginfo("[Recorder] Goal reached at t=%.1fs, x=%.2f", t, p.x)
                self._request_termination('goal_reached', t, min_dist)
            elif collision_flag:
                rospy.logerr(
                    "[Recorder] Collision detected at t=%.2fs (min_dist=%.3f < %.3f)",
                    t,
                    min_dist,
                    self.collision_dist,
                )
                self._request_termination('collision', t, min_dist)
            elif self.timeout_sec > 0.0 and t >= self.timeout_sec:
                rospy.logerr(
                    "[Recorder] Timeout at t=%.2fs (timeout=%.2fs)",
                    t,
                    self.timeout_sec,
                )
                self._request_termination('timeout', t, min_dist)

    def _save_data(self):
        if self.data_saved:
            return

        ts = self.buffers['timestamps']
        file_stem = "{}_{}_trial{:03d}".format(
            self.method, time.strftime("%Y%m%d_%H%M%S"), self.trial_id)
        filepath = os.path.join(self.output_dir, file_stem + ".npz")

        position = np.array(self.buffers['position'], dtype=np.float32)
        cmd_world = np.array(self.buffers['cmd_vel_world'], dtype=np.float32)
        human_world = np.array(self.buffers['human_cmd_world'], dtype=np.float32)
        collision_flags = np.array(self.buffers['collision_flags'], dtype=bool)

        save_dict = {
            'method': self.method,
            'trial_id': self.trial_id,
            'goal_reached': self.reached_goal,
            'collision': self.collision,
            'run_id': self.run_id,
            'batch_idx': self.batch_idx,
            'run_idx': self.run_idx,
            'map_seed': self.map_seed,
            'user_model_seed': self.user_model_seed,
            'goal_x': self.goal_x,
            'collision_dist': self.collision_dist,
            'termination_reason': self.termination_reason,
            'termination_time': self.termination_time if self.termination_time is not None else (ts[-1] if ts else 0.0),
            'total_time': ts[-1] if ts else 0.0,
            'pcd_file': self.pcd_file,
            'tunnel_world': self.tunnel_world,
            'controller_type': str(self.method).upper(),
            'positions': position,
            'human_vels_w': human_world,
            'ctrl_vels_w': cmd_world,
            'collisions': collision_flags,
        }
        for key, val in self.buffers.items():
            save_dict[key] = np.array(val)

        np.savez_compressed(filepath, **save_dict)
        max_x = float(np.max(position[:, 0])) if position.size else float('-inf')
        summary = {
            'method': self.method,
            'trial_id': int(self.trial_id),
            'run_id': self.run_id,
            'batch_idx': int(self.batch_idx),
            'run_idx': int(self.run_idx),
            'map_seed': int(self.map_seed),
            'user_model_seed': int(self.user_model_seed),
            'goal_reached': bool(self.reached_goal),
            'collision': bool(self.collision),
            'termination_reason': self.termination_reason or 'manual_stop',
            'termination_time': float(
                self.termination_time if self.termination_time is not None else (ts[-1] if ts else 0.0)
            ),
            'total_time': float(ts[-1] if ts else 0.0),
            'samples': int(len(ts)),
            'max_x': max_x,
            'min_obstacle_dist': float(
                np.min(self.buffers['min_obstacle_dist_monitored'])
                if self.buffers['min_obstacle_dist_monitored']
                else float('inf')
            ),
            'min_obstacle_dist_raw': float(
                np.min(self.buffers['min_obstacle_dist'])
                if self.buffers['min_obstacle_dist']
                else float('inf')
            ),
            'goal_x': float(self.goal_x),
            'collision_dist': float(self.collision_dist),
            'data_file': os.path.basename(filepath),
            'pcd_file': self.pcd_file,
            'tunnel_world': self.tunnel_world,
        }
        summary_path = os.path.join(self.output_dir, file_stem + ".json")
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        latest_summary_path = os.path.join(self.output_dir, 'run_summary.json')
        with open(latest_summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        self.data_saved = True
        rospy.loginfo("[Recorder] Saved: %s (%d samples)", filepath, len(ts))

    def _shutdown_cb(self):
        self.stop_pub.publish(Bool(data=True))
        if self.recording:
            rospy.loginfo("[Recorder] Shutdown while recording — saving buffered data")
            self.recording = False
            if not self.termination_reason:
                self.termination_reason = 'shutdown'
                self.termination_time = self.buffers['timestamps'][-1] if self.buffers['timestamps'] else 0.0
            self._save_data()


if __name__ == '__main__':
    try:
        FlightRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
