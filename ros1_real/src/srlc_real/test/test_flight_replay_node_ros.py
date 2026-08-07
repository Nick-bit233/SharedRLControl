#!/usr/bin/env python3

import json
import time
import unittest

import rospy
import rostest
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import MarkerArray


class FlightReplayNodeRosTest(unittest.TestCase):
    def test_topics_tf_and_controls(self):
        state = rospy.wait_for_message("/srlc/replay/state", String, timeout=5.0)
        self.assertEqual(state.data, "READY")
        odom = rospy.wait_for_message("/srlc/replay/odom", Odometry, timeout=5.0)
        path = rospy.wait_for_message("/srlc/replay/path", Path, timeout=5.0)
        prediction_path = rospy.wait_for_message(
            "/srlc/replay/input_prediction_path", Path, timeout=5.0
        )
        human_input = rospy.wait_for_message(
            "/srlc/replay/human_input", Vector3Stamped, timeout=5.0
        )
        prediction_info = json.loads(
            rospy.wait_for_message(
                "/srlc/replay/input_prediction_info", String, timeout=5.0
            ).data
        )
        markers = rospy.wait_for_message(
            "/srlc/replay/markers", MarkerArray, timeout=5.0
        )
        self.assertEqual(odom.header.frame_id, "map")
        self.assertEqual(odom.child_frame_id, "replay_uav")
        self.assertEqual(path.header.frame_id, "map")
        self.assertGreaterEqual(len(path.poses), 1)
        self.assertEqual(prediction_path.header.frame_id, "map")
        self.assertGreaterEqual(len(prediction_path.poses), 1)
        self.assertEqual(human_input.header.frame_id, "replay_uav")
        self.assertTrue(prediction_info["enabled"])
        self.assertEqual(
            prediction_info["collision_check"], "unavailable_no_pcd_file"
        )
        self.assertGreaterEqual(len(markers.markers), 8)

        buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buffer)
        transform = buffer.lookup_transform(
            "map", "replay_uav", rospy.Time(0), rospy.Duration(5.0)
        )
        self.assertEqual(transform.child_frame_id, "replay_uav")
        camera_target = buffer.lookup_transform(
            "map", "replay_camera_target", rospy.Time(0), rospy.Duration(5.0)
        )
        self.assertEqual(camera_target.child_frame_id, "replay_camera_target")
        self.assertAlmostEqual(camera_target.transform.rotation.x, 0.0, places=6)
        self.assertAlmostEqual(camera_target.transform.rotation.y, 0.0, places=6)
        self.assertAlmostEqual(camera_target.transform.rotation.z, 0.0, places=6)
        self.assertAlmostEqual(camera_target.transform.rotation.w, 1.0, places=6)

        rospy.wait_for_service("/srlc/replay/play", timeout=5.0)
        play = rospy.ServiceProxy("/srlc/replay/play", SetBool)
        response = play(True)
        self.assertTrue(response.success)

        deadline = time.monotonic() + 5.0
        complete = False
        while time.monotonic() < deadline and not complete:
            complete = rospy.wait_for_message(
                "/srlc/replay/complete", Bool, timeout=1.0
            ).data
        self.assertTrue(complete)
        state = rospy.wait_for_message("/srlc/replay/state", String, timeout=2.0)
        self.assertEqual(state.data, "COMPLETE")
        path = rospy.wait_for_message("/srlc/replay/path", Path, timeout=2.0)
        prediction_path = rospy.wait_for_message(
            "/srlc/replay/input_prediction_path", Path, timeout=2.0
        )
        self.assertEqual(len(path.poses), 5)
        self.assertEqual(len(prediction_path.poses), 5)
        self.assertAlmostEqual(path.poses[-1].pose.position.x, 1.0, places=5)

        rospy.wait_for_service("/srlc/replay/reset", timeout=5.0)
        reset = rospy.ServiceProxy("/srlc/replay/reset", Trigger)
        self.assertTrue(reset().success)
        state = rospy.wait_for_message("/srlc/replay/state", String, timeout=2.0)
        self.assertEqual(state.data, "READY")
        path = rospy.wait_for_message("/srlc/replay/path", Path, timeout=2.0)
        prediction_path = rospy.wait_for_message(
            "/srlc/replay/input_prediction_path", Path, timeout=2.0
        )
        self.assertEqual(len(path.poses), 1)
        self.assertEqual(len(prediction_path.poses), 1)


if __name__ == "__main__":
    rospy.init_node("test_flight_replay_node_ros")
    rostest.rosrun(
        "srlc_real",
        "test_flight_replay_node_ros",
        FlightReplayNodeRosTest,
    )
