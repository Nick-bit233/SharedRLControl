#!/usr/bin/env python3
"""Replay recorded real-flight poses as RViz-friendly ROS topics and TF."""

import json
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
    Vector3Stamped,
)
from nav_msgs.msg import Odometry, Path
from scipy.spatial import cKDTree
from std_msgs.msg import Bool, ColorRGBA, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.flight_replay import (  # noqa: E402
    FlightRecordingError,
    integrate_human_input_prediction,
    load_flight_timeline,
)
from srlc_real_deployment.pcd_io import read_pcd_xyz  # noqa: E402


def _quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class FlightReplayNode:
    """Wall-clock replay state machine with a progressive 3D trail."""

    READY = "READY"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

    def __init__(self):
        rospy.init_node("flight_replay_node", anonymous=False)

        recording_file = str(rospy.get_param("~recording_file", "")).strip()
        if not recording_file:
            raise FlightRecordingError("~recording_file is required")

        start_raw = float(rospy.get_param("~start_time", -1.0))
        end_raw = float(rospy.get_param("~end_time", -1.0))
        start_time = None if start_raw < 0.0 else start_raw
        end_time = None if end_raw < 0.0 else end_raw
        self.timeline = load_flight_timeline(
            recording_file,
            start_time=start_time,
            end_time=end_time,
            map_yaw_deg=float(rospy.get_param("~map_yaw_deg", 0.0)),
            map_origin_xyz=rospy.get_param("~map_origin_xyz", [0.0, 0.0, 0.0]),
            max_sample_gap=float(rospy.get_param("~max_sample_gap", 0.2)),
        )

        self.frame_id = str(rospy.get_param("~frame_id", "map"))
        self.child_frame_id = str(rospy.get_param("~child_frame_id", "replay_uav"))
        self.camera_target_frame_id = str(
            rospy.get_param("~camera_target_frame_id", "replay_camera_target")
        )
        if self.camera_target_frame_id == self.child_frame_id:
            raise FlightRecordingError(
                "~camera_target_frame_id must differ from ~child_frame_id"
            )
        self.input_prediction_enabled = bool(
            rospy.get_param("~input_prediction_enabled", True)
        )
        self.input_prediction_max_xy_speed = float(
            rospy.get_param("~input_prediction_max_xy_speed", 0.5)
        )
        self.prediction_collision_radius = float(
            rospy.get_param("~prediction_collision_radius", 0.25)
        )
        self.prediction_collision_min_z = float(
            rospy.get_param("~prediction_collision_min_z", 0.30)
        )
        self.ceiling_z = float(rospy.get_param("~ceiling_z", 2.8))
        self.pcd_file = str(rospy.get_param("~pcd_file", "")).strip()
        for name, value in (
            ("~input_prediction_max_xy_speed", self.input_prediction_max_xy_speed),
            ("~prediction_collision_radius", self.prediction_collision_radius),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise FlightRecordingError("%s must be finite and positive" % name)
        if (
            not math.isfinite(self.prediction_collision_min_z)
            or not math.isfinite(self.ceiling_z)
            or self.prediction_collision_min_z >= self.ceiling_z
        ):
            raise FlightRecordingError(
                "prediction collision height range must be finite and increasing"
            )

        self.prediction_positions = ()
        self.prediction_info = {
            "enabled": False,
            "reason": "disabled",
            "has_human_input": self.timeline.has_human_input,
        }
        self.prediction_contact = None
        if self.input_prediction_enabled and self.timeline.has_human_input:
            self.prediction_positions = integrate_human_input_prediction(
                self.timeline,
                max_xy_speed=self.input_prediction_max_xy_speed,
            )
            self.prediction_info = self._analyze_input_prediction()
        elif self.input_prediction_enabled:
            self.prediction_info["reason"] = "recording_has_no_human_action"
            rospy.logwarn(
                "[Flight Replay] Input prediction disabled: recording has no "
                "human_action values"
            )
        self.speed = float(rospy.get_param("~speed", 1.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 60.0))
        self.path_publish_rate = float(rospy.get_param("~path_publish_rate", 30.0))
        self.autostart = bool(rospy.get_param("~autostart", True))
        self.autostart_delay = float(rospy.get_param("~autostart_delay", 2.0))
        if not math.isfinite(self.speed) or self.speed <= 0.0:
            raise FlightRecordingError("~speed must be finite and positive")
        if self.publish_rate <= 0.0 or self.path_publish_rate <= 0.0:
            raise FlightRecordingError("publish rates must be positive")

        self.odom_topic = str(rospy.get_param("~odom_topic", "/srlc/replay/odom"))
        self.path_topic = str(rospy.get_param("~path_topic", "/srlc/replay/path"))
        self.prediction_path_topic = str(
            rospy.get_param(
                "~prediction_path_topic", "/srlc/replay/input_prediction_path"
            )
        )
        self.prediction_collision_path_topic = str(
            rospy.get_param(
                "~prediction_collision_path_topic",
                "/srlc/replay/input_prediction_collision_path",
            )
        )
        self.human_input_topic = str(
            rospy.get_param("~human_input_topic", "/srlc/replay/human_input")
        )
        self.prediction_info_topic = str(
            rospy.get_param(
                "~prediction_info_topic", "/srlc/replay/input_prediction_info"
            )
        )
        self.marker_topic = str(
            rospy.get_param("~marker_topic", "/srlc/replay/markers")
        )
        self.state_topic = str(rospy.get_param("~state_topic", "/srlc/replay/state"))
        self.complete_topic = str(
            rospy.get_param("~complete_topic", "/srlc/replay/complete")
        )
        self.play_service_name = str(
            rospy.get_param("~play_service", "/srlc/replay/play")
        )
        self.reset_service_name = str(
            rospy.get_param("~reset_service", "/srlc/replay/reset")
        )

        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=5)
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.prediction_path_pub = rospy.Publisher(
            self.prediction_path_topic, Path, queue_size=1, latch=True
        )
        self.prediction_collision_path_pub = rospy.Publisher(
            self.prediction_collision_path_topic, Path, queue_size=1, latch=True
        )
        self.human_input_pub = rospy.Publisher(
            self.human_input_topic, Vector3Stamped, queue_size=5
        )
        self.prediction_info_pub = rospy.Publisher(
            self.prediction_info_topic, String, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            self.state_topic, String, queue_size=1, latch=True
        )
        self.complete_pub = rospy.Publisher(
            self.complete_topic, Bool, queue_size=1, latch=True
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self._lock = threading.RLock()
        self._state = self.READY
        self._elapsed = 0.0
        self._anchor_elapsed = 0.0
        self._anchor_monotonic = None
        self._path_cursor = 0
        self._prediction_path_cursor = 0
        self._last_path_publish = 0.0

        self.play_service = rospy.Service(
            self.play_service_name, SetBool, self._play_cb
        )
        self.reset_service = rospy.Service(
            self.reset_service_name, Trigger, self._reset_cb
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb
        )

        self.state_pub.publish(String(data=self.READY))
        self.complete_pub.publish(Bool(data=False))
        self.prediction_info_pub.publish(
            String(data=json.dumps(self.prediction_info, sort_keys=True))
        )
        self._publish_frame(0.0, force_path=True)
        rospy.loginfo(
            "[Flight Replay] Ready: file=%s samples=%d source=[%.3f, %.3f] "
            "duration=%.3fs window=%s speed=%.3fx",
            self.timeline.source_path,
            len(self.timeline.samples),
            self.timeline.source_start_time,
            self.timeline.source_end_time,
            self.timeline.duration,
            self.timeline.window_reason,
            self.speed,
        )
        if self.timeline.window_reason.endswith("_to_recording_end"):
            rospy.logwarn(
                "[Flight Replay] No landing/disarm boundary found; using recording end"
            )
        if self.prediction_positions:
            if self.prediction_info.get("collision_check") != "complete":
                rospy.loginfo(
                    "[Flight Replay] Raw-input prediction available; "
                    "collision check=%s",
                    self.prediction_info.get("collision_check", "unavailable"),
                )
            elif self.prediction_contact is None:
                rospy.loginfo(
                    "[Flight Replay] Raw-input prediction: no contact within %.2fm",
                    self.prediction_collision_radius,
                )
            else:
                rospy.logwarn(
                    "[Flight Replay] Raw-input prediction first contact at %.3fs "
                    "position=[%.3f, %.3f, %.3f], clearance=%.3fm",
                    self.prediction_contact["elapsed"],
                    *self.prediction_contact["position"],
                    self.prediction_contact["nearest_distance"],
                )
        if self.autostart:
            self.autostart_timer = rospy.Timer(
                rospy.Duration(max(0.0, self.autostart_delay)),
                self._autostart_cb,
                oneshot=True,
            )

    def _analyze_input_prediction(self):
        info = {
            "enabled": True,
            "reason": "prediction_available",
            "has_human_input": True,
            "method": (
                "zero_order_hold_body_xy_rotated_by_recorded_yaw; "
                "recorded_altitude; no_policy_or_obstacle_response"
            ),
            "max_xy_speed": self.input_prediction_max_xy_speed,
            "collision_radius": self.prediction_collision_radius,
            "collision_min_z": self.prediction_collision_min_z,
            "ceiling_z": self.ceiling_z,
            "contact_predicted": False,
        }
        if not self.pcd_file:
            info["collision_check"] = "unavailable_no_pcd_file"
            return info
        try:
            points = read_pcd_xyz(self.pcd_file)
            points = points[
                (points[:, 2] >= self.prediction_collision_min_z)
                & (points[:, 2] < self.ceiling_z)
            ]
            if len(points) == 0:
                info["collision_check"] = "unavailable_empty_height_slice"
                return info

            positions = np.asarray(self.prediction_positions, dtype=float)
            distances, nearest_indices = cKDTree(points).query(positions)
            eligible = positions[:, 2] >= self.prediction_collision_min_z
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices) == 0:
                info["collision_check"] = "unavailable_prediction_below_min_z"
                return info

            minimum_index = int(
                eligible_indices[np.argmin(distances[eligible_indices])]
            )
            info.update(
                {
                    "collision_check": "complete",
                    "obstacle_point_count": int(len(points)),
                    "minimum_clearance": float(distances[minimum_index]),
                    "minimum_clearance_elapsed": float(
                        self.timeline.samples[minimum_index].elapsed
                    ),
                }
            )
            contacts = np.flatnonzero(
                eligible & (distances <= self.prediction_collision_radius)
            )
            if len(contacts) == 0:
                return info

            contact_index = int(contacts[0])
            sample = self.timeline.samples[contact_index]
            contact = {
                "sample_index": contact_index,
                "elapsed": float(sample.elapsed),
                "source_time": float(sample.source_time),
                "position": [
                    float(value) for value in self.prediction_positions[contact_index]
                ],
                "nearest_distance": float(distances[contact_index]),
                "nearest_obstacle_point": [
                    float(value) for value in points[int(nearest_indices[contact_index])]
                ],
            }
            self.prediction_contact = contact
            info["contact_predicted"] = True
            info["first_contact"] = contact
            return info
        except Exception as exc:
            rospy.logwarn(
                "[Flight Replay] Prediction collision check unavailable: %s", exc
            )
            info["collision_check"] = "error"
            info["collision_check_error"] = str(exc)
            return info

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self.state_pub.publish(String(data=state))
            rospy.loginfo("[Flight Replay] state=%s", state)

    def _current_elapsed(self, now_monotonic):
        if self._state != self.PLAYING or self._anchor_monotonic is None:
            return self._elapsed
        return min(
            self.timeline.duration,
            self._anchor_elapsed
            + (now_monotonic - self._anchor_monotonic) * self.speed,
        )

    def _play_cb(self, request):
        with self._lock:
            now = time.monotonic()
            if request.data:
                if self._state == self.COMPLETE:
                    return SetBoolResponse(
                        success=False,
                        message="replay is complete; call reset before playing again",
                    )
                if self._state != self.PLAYING:
                    self._anchor_elapsed = self._elapsed
                    self._anchor_monotonic = now
                    self._set_state(self.PLAYING)
                return SetBoolResponse(success=True, message="replay playing")

            if self._state == self.PLAYING:
                self._elapsed = self._current_elapsed(now)
                self._anchor_elapsed = self._elapsed
                self._anchor_monotonic = None
                self._set_state(self.PAUSED)
            return SetBoolResponse(success=True, message="replay paused")

    def _reset_cb(self, _request):
        with self._lock:
            self._elapsed = 0.0
            self._anchor_elapsed = 0.0
            self._anchor_monotonic = None
            self._path_cursor = 0
            self._prediction_path_cursor = 0
            self._last_path_publish = 0.0
            self.complete_pub.publish(Bool(data=False))
            self._set_state(self.READY)
            self._publish_frame(0.0, force_path=True)
        return TriggerResponse(success=True, message="replay reset")

    def _autostart_cb(self, _event):
        response = self._play_cb(type("PlayRequest", (), {"data": True})())
        if not response.success:
            rospy.logerr("[Flight Replay] autostart failed: %s", response.message)

    def _timer_cb(self, _event):
        with self._lock:
            now = time.monotonic()
            elapsed = self._current_elapsed(now)
            self._elapsed = elapsed
            reached_end = (
                self._state == self.PLAYING
                and elapsed >= self.timeline.duration
            )
            if reached_end:
                self._anchor_elapsed = elapsed
                self._anchor_monotonic = None
                self._set_state(self.COMPLETE)
            self._publish_frame(elapsed, force_path=reached_end)
            if reached_end:
                self.complete_pub.publish(Bool(data=True))

    def _publish_frame(self, elapsed, force_path=False):
        stamp = rospy.Time.now()
        pose = self.timeline.sample_at(elapsed)
        orientation = _quaternion_from_yaw(pose.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = pose.position[0]
        odom.pose.pose.position.y = pose.position[1]
        odom.pose.pose.position.z = pose.position[2]
        odom.pose.pose.orientation = orientation
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        odom.twist.twist.linear.x = (
            cos_yaw * pose.linear_velocity[0] + sin_yaw * pose.linear_velocity[1]
        )
        odom.twist.twist.linear.y = (
            -sin_yaw * pose.linear_velocity[0] + cos_yaw * pose.linear_velocity[1]
        )
        odom.twist.twist.linear.z = pose.linear_velocity[2]
        odom.twist.twist.angular.z = pose.yaw_rate
        self.odom_pub.publish(odom)

        human_input = self.timeline.human_input_at(elapsed)
        human_message = Vector3Stamped()
        human_message.header.stamp = stamp
        human_message.header.frame_id = self.child_frame_id
        human_message.vector.x = human_input[0]
        human_message.vector.y = human_input[1]
        human_message.vector.z = human_input[2]
        self.human_input_pub.publish(human_message)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = self.child_frame_id
        transform.transform.translation.x = pose.position[0]
        transform.transform.translation.y = pose.position[1]
        transform.transform.translation.z = pose.position[2]
        transform.transform.rotation = orientation
        self.tf_broadcaster.sendTransform(transform)

        camera_target = TransformStamped()
        camera_target.header.stamp = stamp
        camera_target.header.frame_id = self.frame_id
        camera_target.child_frame_id = self.camera_target_frame_id
        camera_target.transform.translation.x = pose.position[0]
        camera_target.transform.translation.y = pose.position[1]
        camera_target.transform.translation.z = pose.position[2]
        camera_target.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(camera_target)

        self.marker_pub.publish(self._marker_array(pose, stamp))

        now = time.monotonic()
        path_period = 1.0 / self.path_publish_rate
        if force_path or now - self._last_path_publish >= path_period:
            self.path_pub.publish(self._path_message(pose, stamp))
            if self.prediction_positions:
                prediction_path = self._prediction_path_message(pose, stamp)
                self.prediction_path_pub.publish(prediction_path)
                self.prediction_collision_path_pub.publish(
                    self._prediction_collision_path_message(
                        prediction_path, pose, stamp
                    )
                )
            self._last_path_publish = now

    def _path_message(self, pose, stamp):
        while (
            self._path_cursor < len(self.timeline.samples)
            and self.timeline.samples[self._path_cursor].elapsed
            <= pose.elapsed + 1.0e-9
        ):
            self._path_cursor += 1

        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        for sample in self.timeline.samples[: self._path_cursor]:
            message.poses.append(
                self._pose_stamped(sample.position, sample.yaw, stamp)
            )
        if (
            not message.poses
            or abs(
                self.timeline.samples[self._path_cursor - 1].elapsed - pose.elapsed
            )
            > 1.0e-9
        ):
            message.poses.append(
                self._pose_stamped(pose.position, pose.yaw, stamp)
            )
        return message

    def _pose_stamped(self, position, yaw, stamp):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.pose.position.x = position[0]
        message.pose.position.y = position[1]
        message.pose.position.z = position[2]
        message.pose.orientation = _quaternion_from_yaw(yaw)
        return message

    def _prediction_path_message(self, pose, stamp):
        while (
            self._prediction_path_cursor < len(self.timeline.samples)
            and self.timeline.samples[self._prediction_path_cursor].elapsed
            <= pose.elapsed + 1.0e-9
        ):
            self._prediction_path_cursor += 1

        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        for index in range(self._prediction_path_cursor):
            message.poses.append(
                self._pose_stamped(
                    self.prediction_positions[index],
                    self.timeline.samples[index].yaw,
                    stamp,
                )
            )

        left_index = pose.left_index
        right_index = pose.right_index
        left = self.timeline.samples[left_index]
        right = self.timeline.samples[right_index]
        alpha = (
            (pose.elapsed - left.elapsed) / (right.elapsed - left.elapsed)
            if right.elapsed > left.elapsed
            else 0.0
        )
        alpha = min(max(alpha, 0.0), 1.0)
        current = tuple(
            self.prediction_positions[left_index][axis]
            + alpha
            * (
                self.prediction_positions[right_index][axis]
                - self.prediction_positions[left_index][axis]
            )
            for axis in range(3)
        )
        last_elapsed = (
            self.timeline.samples[self._prediction_path_cursor - 1].elapsed
            if message.poses
            else -1.0
        )
        if not message.poses or abs(last_elapsed - pose.elapsed) > 1.0e-9:
            message.poses.append(self._pose_stamped(current, pose.yaw, stamp))
        return message

    def _prediction_position_at(self, pose):
        if not self.prediction_positions:
            return None
        left_index = pose.left_index
        right_index = pose.right_index
        left = self.timeline.samples[left_index]
        right = self.timeline.samples[right_index]
        alpha = (pose.elapsed - left.elapsed) / (right.elapsed - left.elapsed)
        alpha = min(max(alpha, 0.0), 1.0)
        return tuple(
            self.prediction_positions[left_index][axis]
            + alpha
            * (
                self.prediction_positions[right_index][axis]
                - self.prediction_positions[left_index][axis]
            )
            for axis in range(3)
        )

    def _prediction_collision_path_message(self, prediction_path, pose, stamp):
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        if (
            self.prediction_contact is None
            or pose.elapsed + 1.0e-9 < self.prediction_contact["elapsed"]
        ):
            return message
        contact_index = int(self.prediction_contact["sample_index"])
        message.poses = list(prediction_path.poses[contact_index:])
        return message

    def _base_marker(self, marker_id, marker_type, stamp, namespace="vehicle"):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _offset(position, yaw, local_x, local_y, local_z):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            position[0] + cos_yaw * local_x - sin_yaw * local_y,
            position[1] + sin_yaw * local_x + cos_yaw * local_y,
            position[2] + local_z,
        )

    @staticmethod
    def _set_marker_pose(marker, position, yaw):
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = _quaternion_from_yaw(yaw)

    def _marker_array(self, pose, stamp):
        markers = MarkerArray()
        body = self._base_marker(0, Marker.CUBE, stamp)
        self._set_marker_pose(body, pose.position, pose.yaw)
        body.scale = Vector3(x=0.22, y=0.16, z=0.08)
        body.color = ColorRGBA(r=1.0, g=0.42, b=0.05, a=1.0)
        markers.markers.append(body)

        for marker_id, angle in ((1, math.pi / 4.0), (2, -math.pi / 4.0)):
            arm = self._base_marker(marker_id, Marker.CUBE, stamp)
            arm_position = self._offset(pose.position, pose.yaw, 0.0, 0.0, 0.015)
            self._set_marker_pose(arm, arm_position, pose.yaw + angle)
            arm.scale = Vector3(x=0.46, y=0.025, z=0.025)
            arm.color = ColorRGBA(r=0.18, g=0.20, b=0.24, a=1.0)
            markers.markers.append(arm)

        rotor_offsets = (
            (0.16, 0.16),
            (0.16, -0.16),
            (-0.16, 0.16),
            (-0.16, -0.16),
        )
        for index, (local_x, local_y) in enumerate(rotor_offsets, start=3):
            rotor = self._base_marker(index, Marker.CYLINDER, stamp)
            rotor_position = self._offset(
                pose.position, pose.yaw, local_x, local_y, 0.035
            )
            self._set_marker_pose(rotor, rotor_position, pose.yaw)
            rotor.scale = Vector3(x=0.13, y=0.13, z=0.012)
            rotor.color = ColorRGBA(r=0.05, g=0.06, b=0.08, a=0.95)
            markers.markers.append(rotor)

        nose = self._base_marker(7, Marker.ARROW, stamp)
        nose.points = [
            Point(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
            Point(
                x=pose.position[0] + 0.36 * math.cos(pose.yaw),
                y=pose.position[1] + 0.36 * math.sin(pose.yaw),
                z=pose.position[2],
            ),
        ]
        nose.scale = Vector3(x=0.035, y=0.08, z=0.10)
        nose.color = ColorRGBA(r=1.0, g=0.92, b=0.12, a=1.0)
        markers.markers.append(nose)

        prediction_position = self._prediction_position_at(pose)
        prediction = self._base_marker(
            0, Marker.SPHERE, stamp, namespace="input_prediction"
        )
        if prediction_position is None:
            prediction.action = Marker.DELETE
        else:
            self._set_marker_pose(prediction, prediction_position, 0.0)
            prediction.scale = Vector3(x=0.16, y=0.16, z=0.16)
            prediction.color = ColorRGBA(r=1.0, g=0.18, b=0.82, a=0.72)
        markers.markers.append(prediction)

        contact = self._base_marker(
            1, Marker.SPHERE, stamp, namespace="input_prediction"
        )
        contact_label = self._base_marker(
            2, Marker.TEXT_VIEW_FACING, stamp, namespace="input_prediction"
        )
        contact_arrow = self._base_marker(
            3, Marker.ARROW, stamp, namespace="input_prediction"
        )
        contact_visible = (
            self.prediction_contact is not None
            and pose.elapsed + 1.0e-9 >= self.prediction_contact["elapsed"]
        )
        if contact_visible:
            contact_position = self.prediction_contact["position"]
            self._set_marker_pose(contact, contact_position, 0.0)
            diameter = max(0.24, 2.4 * self.prediction_collision_radius)
            contact.scale = Vector3(x=diameter, y=diameter, z=diameter)
            contact.color = ColorRGBA(r=1.0, g=0.02, b=0.02, a=0.92)

            contact_arrow.points = [
                Point(
                    x=contact_position[0],
                    y=contact_position[1],
                    z=contact_position[2],
                ),
                Point(
                    x=contact_position[0],
                    y=contact_position[1],
                    z=contact_position[2] + 0.70,
                ),
            ]
            contact_arrow.scale = Vector3(x=0.08, y=0.18, z=0.22)
            contact_arrow.color = ColorRGBA(r=1.0, g=0.05, b=0.02, a=1.0)

            label_position = (
                contact_position[0],
                contact_position[1],
                contact_position[2] + 0.88,
            )
            self._set_marker_pose(contact_label, label_position, 0.0)
            contact_label.scale.z = 0.22
            contact_label.color = ColorRGBA(r=1.0, g=0.35, b=0.30, a=1.0)
            contact_label.text = "ESTIMATED CONTACT"
        else:
            contact.action = Marker.DELETE
            contact_label.action = Marker.DELETE
            contact_arrow.action = Marker.DELETE
        markers.markers.extend((contact, contact_label, contact_arrow))

        start = self._base_marker(10, Marker.SPHERE, stamp, namespace="endpoints")
        self._set_marker_pose(start, self.timeline.samples[0].position, 0.0)
        start.scale = Vector3(x=0.14, y=0.14, z=0.14)
        start.color = ColorRGBA(r=0.1, g=0.95, b=0.25, a=0.95)
        markers.markers.append(start)

        end = self._base_marker(11, Marker.SPHERE, stamp, namespace="endpoints")
        if self._state == self.COMPLETE:
            self._set_marker_pose(end, self.timeline.samples[-1].position, 0.0)
            end.scale = Vector3(x=0.14, y=0.14, z=0.14)
            end.color = ColorRGBA(r=0.95, g=0.12, b=0.12, a=0.95)
        else:
            end.action = Marker.DELETE
        markers.markers.append(end)
        return markers


def main():
    try:
        FlightReplayNode()
        rospy.spin()
    except FlightRecordingError as exc:
        rospy.logfatal("[Flight Replay] %s", exc)
        raise SystemExit(2)
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
