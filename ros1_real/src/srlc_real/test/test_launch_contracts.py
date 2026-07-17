#!/usr/bin/env python3

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"
PACKAGE_DIR = LAUNCH_DIR.parent
REPO_DIR = PACKAGE_DIR.parents[1]
NOKOV_SOURCE = PACKAGE_DIR.parent / "nokov_uav" / "src" / "nokov.cpp"


def parse_launch(name):
    return ET.parse(str(LAUNCH_DIR / name)).getroot()


def launch_args(root):
    return {element.attrib["name"]: element.attrib.get("default", "") for element in root.findall("arg")}


class LaunchContractTest(unittest.TestCase):
    def test_real_stack_owns_guard_but_not_mavros_or_nokov(self):
        root = parse_launch("real_px4.launch")
        args = launch_args(root)

        self.assertIn("start_stream_guard", args)
        for removed_arg in (
            "start_mavros",
            "start_nokov",
            "fcu_url",
            "gcs_url",
            "nokov_server",
            "nokov_port",
            "nokov_tracker",
        ):
            self.assertNotIn(removed_arg, args)

        includes = [element.attrib["file"] for element in root.iter("include")]
        self.assertIn("$(find srlc_real)/launch/mavlink_stream_guard.launch", includes)
        node_packages = {element.attrib["pkg"] for element in root.iter("node")}
        self.assertNotIn("mavros", node_packages)
        self.assertNotIn("vrpn_client_ros", node_packages)
        self.assertNotIn("nokov_uav", node_packages)

    def test_mavros_wrapper_has_tested_udp_defaults_and_no_gcs_forwarding(self):
        args = launch_args(parse_launch("mavros_px4.launch"))

        self.assertEqual(
            args["fcu_url"],
            "$(optenv PX4_FCU_URL udp://:14551@192.168.31.201:14550)",
        )
        self.assertEqual(args["gcs_url"], "$(optenv PX4_GCS_URL)")

    def test_nokov_wrapper_has_container_verified_defaults(self):
        args = launch_args(parse_launch("nokov.launch"))

        self.assertEqual(args["server"], "$(optenv NOKOV_SERVER 192.168.31.192)")
        self.assertEqual(args["port"], "$(optenv NOKOV_PORT 3883)")
        self.assertEqual(args["tracker"], "$(optenv NOKOV_TRACKER uav_soccer)")

    def test_guard_launch_exposes_recovery_policy(self):
        args = launch_args(parse_launch("mavlink_stream_guard.launch"))

        self.assertEqual(args["local_position_rate_hz"], "$(optenv MAVLINK_LOCAL_POSITION_RATE_HZ 30.0)")
        self.assertEqual(args["attitude_rate_hz"], "$(optenv MAVLINK_ATTITUDE_RATE_HZ 20.0)")
        self.assertEqual(args["verify_timeout_sec"], "$(optenv MAVLINK_STREAM_VERIFY_TIMEOUT_SEC 3.0)")
        self.assertEqual(args["stale_timeout_sec"], "$(optenv MAVLINK_STREAM_STALE_TIMEOUT_SEC 2.0)")
        self.assertEqual(args["retry_interval_sec"], "$(optenv MAVLINK_STREAM_RETRY_INTERVAL_SEC 2.0)")
        self.assertEqual(args["max_attempts"], "$(optenv MAVLINK_STREAM_MAX_ATTEMPTS 3)")

    def test_dry_run_includes_guard(self):
        root = parse_launch("dry_run_px4.launch")
        includes = [element.attrib["file"] for element in root.iter("include")]

        self.assertIn("$(find srlc_real)/launch/mavlink_stream_guard.launch", includes)

    def test_dry_run_models_manual_arm_then_single_offboard_entry(self):
        root = parse_launch("dry_run_px4.launch")
        args = launch_args(root)
        fake = next(
            node
            for node in root.iter("node")
            if node.attrib.get("name") == "srlc_real_fake_runtime_node"
        )
        params = {
            param.attrib["name"]: param.attrib.get("value", "")
            for param in fake.findall("param")
        }

        self.assertEqual(params["mode"], "POSCTL")
        self.assertEqual(params["armed"], "true")
        self.assertEqual(params["mode_after_value"], "OFFBOARD")
        self.assertEqual(params["mode_after"], "$(arg fake_offboard_after)")
        self.assertGreater(float(args["fake_offboard_after"]), 0.0)

    def test_real_stack_defaults_to_manual_arm_one_shot_assist(self):
        root = parse_launch("real_px4.launch")
        args = launch_args(root)

        self.assertEqual(
            args["takeoff_height"],
            "$(optenv SRLC_TAKEOFF_HEIGHT 1.0)",
        )
        self.assertEqual(
            args["post_takeoff_mode"],
            "$(optenv SRLC_POST_TAKEOFF_MODE assist)",
        )
        self.assertNotIn("auto_arm_on_offboard", args)
        self.assertNotIn("post_takeoff_mode_delay", args)
        self.assertNotIn("takeoff_reached_tolerance", args)
        for required_arg in (
            "takeoff_lower_margin",
            "takeoff_upper_margin",
            "takeoff_max_abs_vz",
            "takeoff_confirm_duration",
            "takeoff_timeout",
            "takeoff_max_overshoot",
            "takeoff_max_xy_drift",
            "takeoff_max_climb_speed",
            "takeoff_max_vertical_accel",
            "takeoff_max_tracking_error",
            "px4_local_odom_topic",
            "px4_local_velocity_topic",
            "fault_response",
            "fault_land_mode",
            "fault_land_confirm_timeout",
        ):
            self.assertIn(required_arg, args)

        navigator = next(
            node
            for node in root.iter("node")
            if node.attrib.get("name") == "srlc_real_navigator"
        )
        navigator_params = {
            param.attrib["name"]: param.attrib.get("value", "")
            for param in navigator.findall("param")
        }
        self.assertNotIn("auto_arm_on_offboard", navigator_params)
        self.assertEqual(
            navigator_params["post_takeoff_mode"],
            "$(arg post_takeoff_mode)",
        )
        self.assertEqual(
            navigator_params["px4_local_odom_topic"],
            "$(arg px4_local_odom_topic)",
        )
        self.assertEqual(
            navigator_params["px4_local_velocity_topic"],
            "$(arg px4_local_velocity_topic)",
        )

    def test_docker_defaults_match_real_launch(self):
        compose_path = REPO_DIR / "docker-compose.real.yml"
        if not compose_path.exists():
            self.skipTest("docker-compose.real.yml is outside the catkin source image")
        compose = compose_path.read_text(encoding="utf-8")

        self.assertIn("SRLC_POST_TAKEOFF_MODE: ${SRLC_POST_TAKEOFF_MODE:-assist}", compose)
        self.assertIn("SRLC_TAKEOFF_HEIGHT: ${SRLC_TAKEOFF_HEIGHT:-1.0}", compose)
        self.assertIn("SRLC_TAKEOFF_MAX_CLIMB_SPEED: ${SRLC_TAKEOFF_MAX_CLIMB_SPEED:-0.4}", compose)
        self.assertIn("SRLC_TAKEOFF_MAX_VERTICAL_ACCEL: ${SRLC_TAKEOFF_MAX_VERTICAL_ACCEL:-0.5}", compose)
        self.assertIn("SRLC_TAKEOFF_MAX_TRACKING_ERROR: ${SRLC_TAKEOFF_MAX_TRACKING_ERROR:-0.25}", compose)

    def test_nokov_vision_height_correction_is_preserved(self):
        source = NOKOV_SOURCE.read_text(encoding="utf-8")

        self.assertIn("double vision_z_offset = -0.15;", source)
        self.assertIn(
            'pnh.param<double>("vision_z_offset", vision_z_offset, -0.15);',
            source,
        )


if __name__ == "__main__":
    unittest.main()
