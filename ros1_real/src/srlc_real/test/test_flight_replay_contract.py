#!/usr/bin/env python3

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


class FlightReplayContractTest(unittest.TestCase):
    def test_launch_is_offline_only_and_exposes_replay_components(self):
        root = ET.parse(PACKAGE_ROOT / "launch" / "flight_replay.launch").getroot()
        node_types = [node.attrib.get("type", "") for node in root.iter("node")]

        self.assertIn("pcd_map_publisher.py", node_types)
        self.assertIn("flight_replay_node.py", node_types)
        self.assertIn("rviz", node_types)
        self.assertNotIn("real_navigation_node.py", node_types)
        self.assertFalse(any("mavros" in value.lower() for value in node_types))

    def test_rviz_tracks_replay_frame_and_renders_required_topics(self):
        config = (PACKAGE_ROOT / "cfg" / "real_map" / "flight_replay.rviz").read_text(
            encoding="utf-8"
        )
        self.assertIn("Fixed Frame: map", config)
        self.assertIn("Class: rviz/Orbit", config)
        self.assertIn("Target Frame: replay_camera_target", config)
        self.assertIn("Topic: /real_map/cloud", config)
        self.assertIn("Topic: /real_map/ceiling", config)
        self.assertIn("Alpha: 0.18", config)
        self.assertIn("Topic: /srlc/replay/path", config)
        self.assertIn("Topic: /srlc/replay/input_prediction_path", config)
        self.assertIn(
            "Topic: /srlc/replay/input_prediction_collision_path", config
        )
        self.assertIn("Marker Topic: /srlc/replay/markers", config)

    def test_container_has_headless_video_dependencies(self):
        dockerfile_path = REPO_ROOT / "Dockerfile"
        if not dockerfile_path.exists():
            self.skipTest("Dockerfile is outside the catkin source-only image")
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        for package in ("ffmpeg", "xvfb", "x11-utils", "libgl1-mesa-dri"):
            self.assertIn(package, dockerfile)
        self.assertIn("pbr==7.0.3", dockerfile)
        self.assertIn("testresources==2.0.1", dockerfile)


if __name__ == "__main__":
    unittest.main()
