#!/usr/bin/env python3

import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
LAUNCH_DIR = PACKAGE_DIR / "launch"
CFG_DIR = PACKAGE_DIR / "cfg" / "tunnel"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from srlc_real_deployment.clearance_guard import (  # noqa: E402
    ClearanceGuard,
    ClearanceGuardConfig,
    ClearanceState,
)


MESSAGE_FIELDS = [
    "std_msgs/Header header",
    "bool valid",
    "float32 surface_clearance",
    "float32 center_distance",
    "geometry_msgs/Point nearest_obstacle_point",
    "geometry_msgs/Vector3 escape_direction",
]


def _launch_args(name):
    root = ET.parse(str(LAUNCH_DIR / name)).getroot()
    return {
        element.attrib["name"]: element.attrib.get("default", "")
        for element in root.findall("arg")
    }, root


def _load_migration_module():
    path = SCRIPT_DIR / "srlc_real_deployment" / "config_migration.py"
    assert path.exists(), "pure legacy-config rejection helper is missing"
    spec = importlib.util.spec_from_file_location("config_migration", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_clearance_runtime_module():
    path = SCRIPT_DIR / "srlc_real_deployment" / "clearance_runtime.py"
    assert path.exists(), "pure clearance lifecycle adapter is missing"
    spec = importlib.util.spec_from_file_location("clearance_runtime", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_obstacle_clearance_message_and_catkin_dependencies_are_exact():
    message_path = PACKAGE_DIR / "msg" / "ObstacleClearance.msg"
    assert message_path.exists()
    fields = [
        line.strip()
        for line in message_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert fields == MESSAGE_FIELDS

    cmake = (PACKAGE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_DIR / "package.xml").read_text(encoding="utf-8")
    assert "message_generation" in cmake
    assert "add_message_files" in cmake
    assert "ObstacleClearance.msg" in cmake
    assert "generate_messages" in cmake
    assert "message_runtime" in cmake
    assert "<build_depend>message_generation</build_depend>" in package_xml
    assert "<exec_depend>message_runtime</exec_depend>" in package_xml


def test_map_lidar_uses_raw_full_pose_channels_and_source_header():
    source = (SCRIPT_DIR / "map_lidar_node.py").read_text(encoding="utf-8")

    for required in (
        "from srlc_real.msg import ObstacleClearance",
        'rospy.get_param("~odom_topic", "/nokov/local_position/odom")',
        "PcdClearanceGeometry",
        "policy_surface_distances",
        ".raycast_raw(",
        "rotation_map_from_local @ rotation_local_from_body",
        "rotation_map_from_local.T @",
        "clearance_msg.header = odom.header",
        "clearance_result.center_distance",
        "np.min(policy_distances)",
    ):
        assert required in source
    assert "minimum_raycast_distance" not in source
    assert "nearest_distance(" not in source


def test_navigation_uses_clearance_guard_without_safety_alias_subscription():
    source = (SCRIPT_DIR / "real_navigation_node.py").read_text(encoding="utf-8")

    for required in (
        "from srlc_real.msg import ObstacleClearance",
        "from srlc_real_deployment.clearance_runtime import soft_guard_position",
        '"/srlc/lidar/obstacle_clearance"',
        "ClearanceGuard(",
        "ClearanceGuardConfig(",
        "project_velocity_away(",
        "self._clearance_guard_lock",
        '"/tunnel_nav/clearance_guard_state"',
        '"/tunnel_nav/clearance_guard_shadow_state"',
        'external_fault = "COLLISION"',
        "if not np.isfinite(lidar_np).all()",
        "self.policy_dist = float(",
        "policy_d=",
        "clearance=",
        "self._clearance_guard_position(",
        "decision.state_changed and decision.state == LifecycleState.ACTIVE",
    ):
        assert required in source
    assert "min_safety_distance" not in source
    assert "lidar_safety_distance" not in source
    assert "safety_d=" not in source
    assert "enable_collision_detection" not in source
    assert source.count("self._invalidate_lidar_range()") == 2


def test_enforce_soft_guard_captures_current_position_only_after_active():
    runtime = _load_clearance_runtime_module()
    config = ClearanceGuardConfig(
        proximity_enabled=True,
        proximity_enter_clearance=0.10,
        proximity_release_clearance=0.15,
        collision_clearance=0.02,
        collision_confirm_samples=2,
        immediate_collision_clearance=-0.03,
        sample_timeout=0.30,
    )

    guard = ClearanceGuard(config)
    preflight_position = runtime.soft_guard_position(
        clearance_guard_mode="enforce",
        lifecycle_active=False,
        px4_local_position=(0.0, 0.0, 0.0),
    )
    assert all(math.isnan(component) for component in preflight_position)
    preflight = guard.update(
        now=1.0,
        source_stamp=1.0,
        valid=True,
        surface_clearance=0.05,
        escape_direction=(1.0, 0.0, 0.0),
        human_velocity_world=(0.0, 0.0, 0.0),
        px4_local_position=preflight_position,
    )
    assert preflight.state == ClearanceState.NORMAL
    assert preflight.hold_position is None

    current_position = (0.25, -0.10, 1.05)
    active_position = runtime.soft_guard_position(
        clearance_guard_mode="enforce",
        lifecycle_active=True,
        px4_local_position=current_position,
    )
    active = guard.update(
        now=1.01,
        source_stamp=1.0,
        valid=True,
        surface_clearance=0.05,
        escape_direction=(1.0, 0.0, 0.0),
        human_velocity_world=(0.0, 0.0, 0.0),
        px4_local_position=active_position,
    )
    assert active.state == ClearanceState.PROXIMITY_HOLD
    assert active.hold_position == current_position

    hard_guard = ClearanceGuard(config)
    first_hard = hard_guard.update(
        now=2.0,
        source_stamp=2.0,
        valid=True,
        surface_clearance=0.01,
        escape_direction=(1.0, 0.0, 0.0),
        human_velocity_world=(0.0, 0.0, 0.0),
        px4_local_position=preflight_position,
    )
    second_hard = hard_guard.update(
        now=2.01,
        source_stamp=2.01,
        valid=True,
        surface_clearance=0.01,
        escape_direction=(1.0, 0.0, 0.0),
        human_velocity_world=(0.0, 0.0, 0.0),
        px4_local_position=preflight_position,
    )
    assert first_hard.state == ClearanceState.NORMAL
    assert second_hard.state == ClearanceState.COLLISION

    assert runtime.soft_guard_position(
        clearance_guard_mode="shadow",
        lifecycle_active=False,
        px4_local_position=current_position,
    ) == current_position


def test_legacy_config_detection_uses_ros_basenames_and_nonempty_env_values():
    migration = _load_migration_module()

    uses = migration.find_legacy_safety_config(
        ros_param_names=[
            "/unrelated/collision_dist_extra",
            "/robot/navigator/enable_safety_stop",
            "/global/safety_min_dist",
        ],
        environment={
            "collision_dist": "0.15",
            "SRLC_ENABLE_SAFETY_STOP": "false",
            "SRLC_SAFETY_MIN_DIST": "   ",
        },
    )

    assert uses.ros_params == (
        "/global/safety_min_dist",
        "/robot/navigator/enable_safety_stop",
    )
    assert uses.environment == (
        "SRLC_ENABLE_SAFETY_STOP",
        "collision_dist",
    )
    error = migration.legacy_safety_migration_error(uses)
    for replacement in (
        "enable_proximity_guard",
        "proximity_enter_clearance",
        "proximity_release_clearance",
        "collision_confirm_clearance",
        "collision_immediate_clearance",
    ):
        assert replacement in error


def test_exact_yaml_defaults_and_removed_controls():
    map_cfg = yaml.safe_load(
        (CFG_DIR / "map_lidar_real_px4.yaml").read_text(encoding="utf-8")
    )
    nav_cfg = yaml.safe_load(
        (CFG_DIR / "real_nav_px4.yaml").read_text(encoding="utf-8")
    )

    assert map_cfg["vehicle_half_extents"] == [0.15, 0.15, 0.05]
    assert map_cfg["policy_extra_margin"] == [0.05, 0.05, 0.0]
    assert map_cfg["clearance_cap"] == 1.0
    assert map_cfg["clearance_topic"] == "/srlc/lidar/obstacle_clearance"

    expected_nav = {
        "enable_proximity_guard": False,
        "proximity_enter_clearance": 0.10,
        "proximity_release_clearance": 0.15,
        "proximity_release_duration": 0.20,
        "proximity_escape_min_speed": 0.05,
        "collision_confirm_clearance": 0.02,
        "collision_confirm_frames": 2,
        "collision_immediate_clearance": -0.03,
        "clearance_guard_mode": "enforce",
        "fault_response": "hold",
    }
    for name, expected in expected_nav.items():
        assert nav_cfg[name] == expected

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            CFG_DIR / "map_lidar_real_px4.yaml",
            CFG_DIR / "real_nav_px4.yaml",
            LAUNCH_DIR / "real_px4.launch",
            LAUNCH_DIR / "dry_run_px4.launch",
            REPO_DIR / "docker-compose.real.yml",
        )
    )
    for removed in (
        "enable_safety_stop",
        "safety_min_dist",
        "collision_dist",
        "enable_collision_detection",
        "SRLC_ENABLE_SAFETY_STOP",
        "SRLC_SAFETY_MIN_DIST",
        "SRLC_COLLISION_DIST",
    ):
        assert removed not in combined


def test_both_launches_and_compose_expose_exact_new_defaults():
    real_args, real_root = _launch_args("real_px4.launch")
    dry_args, dry_root = _launch_args("dry_run_px4.launch")
    defaults = {
        "vehicle_half_extents": ("[0.15,0.15,0.05]", "SRLC_VEHICLE_HALF_EXTENTS"),
        "policy_extra_margin": ("[0.05,0.05,0.0]", "SRLC_POLICY_EXTRA_MARGIN"),
        "clearance_cap": ("1.0", "SRLC_CLEARANCE_CAP"),
        "enable_proximity_guard": ("false", "SRLC_ENABLE_PROXIMITY_GUARD"),
        "proximity_enter_clearance": ("0.10", "SRLC_PROXIMITY_ENTER_CLEARANCE"),
        "proximity_release_clearance": ("0.15", "SRLC_PROXIMITY_RELEASE_CLEARANCE"),
        "proximity_release_duration": ("0.20", "SRLC_PROXIMITY_RELEASE_DURATION"),
        "proximity_escape_min_speed": ("0.05", "SRLC_PROXIMITY_ESCAPE_MIN_SPEED"),
        "collision_confirm_clearance": ("0.02", "SRLC_COLLISION_CONFIRM_CLEARANCE"),
        "collision_confirm_frames": ("2", "SRLC_COLLISION_CONFIRM_FRAMES"),
        "collision_immediate_clearance": ("-0.03", "SRLC_COLLISION_IMMEDIATE_CLEARANCE"),
        "clearance_guard_mode": ("enforce", "SRLC_CLEARANCE_GUARD_MODE"),
        "fault_response": ("hold", "SRLC_FAULT_RESPONSE"),
    }

    compose = (REPO_DIR / "docker-compose.real.yml").read_text(encoding="utf-8")
    for name, (default, env_name) in defaults.items():
        assert name in real_args
        assert name in dry_args
        assert real_args[name] == f"$(optenv {env_name} {default})"
        assert dry_args[name] == default
        assert f"{env_name}: ${{{env_name}:-{default}}}" in compose

    for root in (real_root, dry_root):
        params = {
            element.attrib.get("name", element.attrib.get("param"))
            for node in root.iter("node")
            for element in list(node)
            if element.tag in {"param", "rosparam"}
            and ("name" in element.attrib or "param" in element.attrib)
        }
        for name in defaults:
            assert name in params
