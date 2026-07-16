#!/usr/bin/env python3

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"


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


if __name__ == "__main__":
    unittest.main()
