#!/usr/bin/env python3
"""Batch runner for ROS1 tunnel experiments."""

import argparse
import glob
import importlib.util
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_CHECKPOINT = "$(find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt"
ALLOWED_METHODS = ("rl", "ipc", "naive_raw", "naive_safe")
NAIVE_METHODS = {"naive_raw", "naive_safe"}

FATAL_LOG_MARKERS = (
    "Call to publish() on an invalid Publisher",
    "boost::thread_resource_error",
    "what():  boost::thread_resource_error",
    "gzserver: Aborted",
    "gzclient: Aborted",
    "process has died",
    "RL node failed to initialize",
)

EXPERIMENT_PROCESS_PATTERNS = (
    "/opt/ros/noetic/bin/rosmaster --core",
    "/opt/ros/noetic/lib/rosout/rosout",
    "/opt/ros/noetic/bin/roslaunch navigation_runner tunnel_comparison.launch",
    "gzserver",
    "gzclient",
    "gazebo_gui",
    "rviz",
    "navigation_runner/scripts/flight_recorder.py",
    "navigation_runner/scripts/tunnel_navigation.py",
    "navigation_runner/scripts/lidar_sim_node.py",
    "navigation_runner/scripts/cmd_bridge_node.py",
    "navigation_runner/scripts/rc_sim_node.py",
    "/root/slope_ws/devel/lib/ipc/ipc_node",
    "occupancy_map_node",
    "spawn_model",
    "topic_tools/throttle",
    "quadcopterTFBroadcaster",
    "static_transform_publisher",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batched RL/IPC tunnel experiments on regenerated maps."
    )
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Number of map batches to generate (default: 1)")
    parser.add_argument("--runs-per-batch", type=int, default=None,
                        help="Paired RL/IPC runs per batch (default: 5)")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method order, e.g. rl,ipc,naive_raw,naive_safe (default: rl,ipc)")
    parser.add_argument("--master-seed", type=int, default=None,
                        help="Master RNG seed for map/user-model seeds (default: 42)")
    parser.add_argument("--output-dir", default=None,
                        help="Output root (default: auto timestamp under /root/catkin_ws/results)")
    parser.add_argument("--resume-from", default=None,
                        help="Resume an existing batch result root in-place; skips complete runs")
    parser.add_argument("--batch-index", type=int, default=None,
                        help="Run only one batch index from the full seed plan")
    parser.add_argument("--batch-start", type=int, default=None,
                        help="First batch index to execute from the full seed plan")
    parser.add_argument("--batch-end", type=int, default=None,
                        help="Last batch index to execute from the full seed plan, inclusive")
    parser.add_argument("--min-complete-samples", type=int, default=1,
                        help="Minimum samples for an existing trajectory to be considered complete")
    parser.add_argument("--run-retries", type=int, default=0,
                        help="Retry a failed/timed-out/fatal-log run this many times")
    parser.add_argument("--launch-timeout", type=float, default=80.0,
                        help="External watchdog timeout per run in seconds")
    parser.add_argument("--recorder-timeout", type=float, default=60.0,
                        help="Optional recorder-side timeout (0 disables)")
    parser.add_argument("--completion-grace-period", type=float, default=0.5,
                        help="Grace period between stop signal and roslaunch shutdown")
    parser.add_argument("--inter-run-delay", type=float, default=2.0,
                        help="Delay between sequential runs")
    parser.add_argument("--goal-x", type=float, default=10.0)
    parser.add_argument("--collision-dist", type=float, default=0.20)
    parser.add_argument("--safety-min-dist", type=float, default=None,
                        help="RL safety stop distance passed to tunnel_navigation.py (default: 0.35)")
    parser.add_argument("--safety-mode", default=None, choices=("hold", "recover"),
                        help="RL safety intervention mode: hold or recover (default: recover)")
    parser.add_argument("--safety-recover-speed", type=float, default=None,
                        help="Max horizontal escape speed for safety_mode=recover")
    parser.add_argument("--safety-recover-forward-speed", type=float, default=None,
                        help="Small forward bias for safety_mode=recover")
    parser.add_argument("--safety-recover-centerline-gain", type=float, default=None,
                        help="Centerline-return weight for safety_mode=recover")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="RL checkpoint passed to tunnel_comparison.launch")
    parser.add_argument("--policy-mode", default="residual", choices=("residual", "direct"),
                        help="RL policy architecture mode for checkpoint loading")
    parser.add_argument("--gazebo-z-mode", default="alt_hold",
                        choices=("alt_hold", "policy", "policy_clamped", "blend"),
                        help="Gazebo vertical command execution mode for RL cmd_vel")
    parser.add_argument("--gazebo-policy-z-max", type=float, default=2.0,
                        help="Max |vz| for policy_clamped/blend modes")
    parser.add_argument("--gazebo-z-blend-alpha", type=float, default=0.5,
                        help="Blend mode weight: alpha*policy_vz + (1-alpha)*alt_hold")
    parser.add_argument("--disable-gazebo-policy-z-takeoff-gate", action="store_true",
                        help="Let policy z control start immediately, even below takeoff_height")
    parser.add_argument("--gazebo-policy-z-gate-tolerance", type=float, default=0.5,
                        help="Enable policy z when z >= takeoff_height - tolerance")
    parser.add_argument("--disable-policy-takeoff-gate", action="store_true",
                        help="Run and execute RL policy immediately after raycast readiness")
    parser.add_argument("--policy-takeoff-gate-tolerance", type=float, default=0.5,
                        help="Enable full policy control when z >= takeoff_height - tolerance")
    parser.add_argument("--gui", action="store_true",
                        help="Show Gazebo GUI")
    parser.add_argument("--rviz", action="store_true",
                        help="Launch RViz")
    parser.add_argument("--spawn-x", type=float, default=-8.5)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.1)
    parser.add_argument("--user-model-simple", action="store_true",
                        help="Use simple user model instead of Perlin profile")
    parser.add_argument("--user-model-profile", default="m3_diverse",
                        choices=("m3_diverse", "legacy_perlin", "simple"),
                        help="Online pilot profile for RL and IPC")
    parser.add_argument("--user-model-speed", type=float, default=2.0)
    parser.add_argument("--user-model-freq-base", type=float, default=0.1)
    parser.add_argument("--user-model-freq-scale", type=float, default=0.2)
    parser.add_argument("--user-model-vx-bias", type=float, default=1.5)
    parser.add_argument("--user-model-vx-amp", type=float, default=0.5)
    parser.add_argument("--user-model-vy-amp", type=float, default=2.0)
    parser.add_argument("--user-model-vz-amp", type=float, default=0.2)
    parser.add_argument("--user-model-smoothness-base", type=float, default=0.4)
    parser.add_argument("--user-model-smoothness-scale", type=float, default=0.5)
    parser.add_argument("--user-model-laziness", type=float, default=0.3)
    parser.add_argument("--input-source", default=None, choices=("online", "offline"),
                        help="Pilot input source for RL and IPC (default: online)")
    parser.add_argument("--replay-dataset-path", default=None,
                        help="Offline replay dataset path, e.g. trajectories_tunnel.h5")
    parser.add_argument("--replay-dataset-format", default=None, choices=("hdf5", "h5"),
                        help="Offline replay dataset format (default: hdf5)")
    parser.add_argument("--replay-sampling-mode", default=None, choices=("raw",),
                        help="Replay sampling mode (default: raw)")
    parser.add_argument("--replay-trajectory-index", type=int, default=None,
                        help="Fixed replay trajectory index; -1 derives it from user_model_seed")
    parser.add_argument("--replay-start-offset", type=int, default=None,
                        help="Fixed replay start offset; -1 derives it from user_model_seed")
    parser.add_argument("--no-replay-loop", dest="replay_loop", action="store_false",
                        help="Hold the last replay command instead of looping at dataset end")
    parser.set_defaults(replay_loop=None)
    parser.add_argument("--num-obstacles", type=int, default=15)
    parser.add_argument("--cuboid-ratio", type=float, default=0.5)
    parser.add_argument("--map-resolution", type=float, default=0.1)
    parser.add_argument("--map-sampling-mode", default=None,
                        choices=("uniform", "constrained"),
                        help="Obstacle center sampler for generated PCD maps")
    parser.add_argument("--min-obstacle-spacing", type=float, default=None,
                        help="Minimum XY footprint spacing for constrained map sampling")
    parser.add_argument("--local-density-window", type=float, default=None,
                        help="Square XY window size for local density checks")
    parser.add_argument("--max-obstacles-per-window", type=int, default=None,
                        help="Maximum obstacle centers in any local density window")
    parser.add_argument("--max-local-area-fraction", type=float, default=None,
                        help="Maximum obstacle footprint area fraction per local density window")
    parser.add_argument("--require-connectivity", action="store_true", default=None,
                        help="Reject maps without approximate start-to-goal free-space connectivity")
    parser.add_argument("--min-bottleneck-width", type=float, default=None,
                        help="Approximate minimum required clearance along connected free-space")
    parser.add_argument("--max-resample-attempts", type=int, default=None,
                        help="Maximum deterministic resampling attempts for constrained maps")
    parser.add_argument("--analyze", dest="analyze", action="store_true")
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.set_defaults(analyze=True)
    return parser.parse_args()


def kill_process_group(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def find_experiment_processes():
    current_pid = os.getpid()
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    matches = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid = int(parts[0])
        cmd = parts[1]
        if pid == current_pid:
            continue
        if any(pattern in cmd for pattern in EXPERIMENT_PROCESS_PATTERNS):
            matches.append((pid, cmd))
    return matches


def cleanup_experiment_processes(context):
    matches = find_experiment_processes()
    if not matches:
        return

    pids = [pid for pid, _ in matches]
    print(
        f"[Batch] Cleaning stale processes ({context}): "
        + ", ".join(str(pid) for pid in pids)
    )

    for sig, wait_seconds in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            remaining = {pid for pid, _ in find_experiment_processes()}
            if not any(pid in remaining for pid in pids):
                return
            time.sleep(0.2)

    remaining = [pid for pid, _ in find_experiment_processes()]
    if remaining:
        print(f"[Batch] WARNING: some stale processes are still alive: {remaining}")


def build_run_env(run_dir, run_slot):
    env = os.environ.copy()
    ros_master_port = 11311 + run_slot
    gazebo_master_port = 11345 + run_slot
    ros_log_dir = os.path.join(run_dir, "ros_logs")
    os.makedirs(ros_log_dir, exist_ok=True)
    env["ROS_IP"] = "127.0.0.1"
    env["ROS_HOSTNAME"] = "127.0.0.1"
    env["ROS_MASTER_URI"] = f"http://127.0.0.1:{ros_master_port}"
    env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{gazebo_master_port}"
    env["ROS_LOG_DIR"] = ros_log_dir
    return env


def ensure_output_dir(args):
    if args.resume_from:
        root = os.path.abspath(args.resume_from)
        if args.output_dir and os.path.abspath(args.output_dir) != root:
            raise ValueError("--output-dir must match --resume-from when resuming")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Resume directory does not exist: {root}")
    elif args.output_dir:
        root = os.path.abspath(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join("/root/catkin_ws/results", f"batch_{timestamp}")
    os.makedirs(root, exist_ok=True)
    return root


def load_existing_batch_config(output_root):
    manifest_path = os.path.join(output_root, "batch_manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            config = manifest.get("batch_config")
            if config:
                return config
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Batch] WARNING: ignoring unreadable batch_manifest.json: {exc}")

    config_path = os.path.join(output_root, "batch_config.json")
    if os.path.exists(config_path):
        return load_json(config_path)

    raise FileNotFoundError(
        f"Cannot resume without batch_manifest.json or batch_config.json in {output_root}"
    )


def apply_default_args(args):
    if args.num_batches is None:
        args.num_batches = 1
    if args.runs_per_batch is None:
        args.runs_per_batch = 5
    if args.methods is None:
        args.methods = "rl,ipc"
    if args.master_seed is None:
        args.master_seed = 42
    if args.safety_min_dist is None:
        args.safety_min_dist = 0.35
    if args.safety_mode is None:
        args.safety_mode = "recover"
    if args.safety_recover_speed is None:
        args.safety_recover_speed = 0.35
    if args.safety_recover_forward_speed is None:
        args.safety_recover_forward_speed = 0.15
    if args.safety_recover_centerline_gain is None:
        args.safety_recover_centerline_gain = 0.4
    if args.input_source is None:
        args.input_source = "online"
    if args.replay_dataset_path is None:
        args.replay_dataset_path = ""
    if args.replay_dataset_format is None:
        args.replay_dataset_format = "hdf5"
    if args.replay_sampling_mode is None:
        args.replay_sampling_mode = "raw"
    if args.replay_trajectory_index is None:
        args.replay_trajectory_index = -1
    if args.replay_start_offset is None:
        args.replay_start_offset = -1
    if args.replay_loop is None:
        args.replay_loop = True
    if args.map_sampling_mode is None:
        args.map_sampling_mode = "uniform"
    if args.min_obstacle_spacing is None:
        args.min_obstacle_spacing = 0.0
    if args.local_density_window is None:
        args.local_density_window = 3.0
    if args.max_obstacles_per_window is None:
        args.max_obstacles_per_window = 4
    if args.max_local_area_fraction is None:
        args.max_local_area_fraction = 0.45
    if args.require_connectivity is None:
        args.require_connectivity = False
    if args.min_bottleneck_width is None:
        args.min_bottleneck_width = 0.4
    if args.max_resample_attempts is None:
        args.max_resample_attempts = 2000


def normalize_methods(methods_arg):
    methods = [m.strip().lower() for m in str(methods_arg).split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method")

    unknown = [method for method in methods if method not in ALLOWED_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {','.join(unknown)}. "
            f"Allowed methods: {','.join(ALLOWED_METHODS)}"
        )

    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(f"Duplicate method(s) in --methods: {','.join(duplicates)}")
    return methods


def validate_method_config(args):
    methods = normalize_methods(args.methods)
    args.methods = ",".join(methods)

    if any(method in NAIVE_METHODS for method in methods) and args.input_source != "offline":
        raise ValueError(
            "naive_raw/naive_safe are dataset replay baselines and require "
            "--input-source offline with --replay-dataset-path"
        )
    return methods


def apply_resume_config(args, output_root):
    config = load_existing_batch_config(output_root)

    requested_seed = args.master_seed
    requested_num_batches = args.num_batches
    requested_runs_per_batch = args.runs_per_batch
    requested_methods = args.methods
    requested_safety_min_dist = args.safety_min_dist
    requested_safety_mode = args.safety_mode
    requested_safety_recover_speed = args.safety_recover_speed
    requested_safety_recover_forward_speed = args.safety_recover_forward_speed
    requested_safety_recover_centerline_gain = args.safety_recover_centerline_gain
    requested_input_source = args.input_source
    requested_replay_dataset_path = args.replay_dataset_path
    requested_replay_dataset_format = args.replay_dataset_format
    requested_replay_sampling_mode = args.replay_sampling_mode
    requested_replay_trajectory_index = args.replay_trajectory_index
    requested_replay_start_offset = args.replay_start_offset
    requested_replay_loop = args.replay_loop
    requested_map_sampling_mode = args.map_sampling_mode
    requested_min_obstacle_spacing = args.min_obstacle_spacing
    requested_local_density_window = args.local_density_window
    requested_max_obstacles_per_window = args.max_obstacles_per_window
    requested_max_local_area_fraction = args.max_local_area_fraction
    requested_require_connectivity = args.require_connectivity
    requested_min_bottleneck_width = args.min_bottleneck_width
    requested_max_resample_attempts = args.max_resample_attempts

    config_methods = ",".join(config.get("methods", []))
    if requested_seed is not None and requested_seed != int(config["master_seed"]):
        raise ValueError(
            f"--master-seed {requested_seed} does not match existing "
            f"batch seed {config['master_seed']}"
        )
    if requested_num_batches is not None and requested_num_batches != int(config["num_batches"]):
        raise ValueError(
            f"--num-batches {requested_num_batches} does not match existing "
            f"batch count {config['num_batches']}"
        )
    if (
        requested_runs_per_batch is not None
        and requested_runs_per_batch != int(config["runs_per_batch"])
    ):
        raise ValueError(
            f"--runs-per-batch {requested_runs_per_batch} does not match existing "
            f"runs-per-batch {config['runs_per_batch']}"
        )
    if requested_methods is not None and requested_methods != config_methods:
        raise ValueError(
            f"--methods {requested_methods} does not match existing methods {config_methods}"
        )
    config_safety_min_dist = float(config.get("safety_min_dist", 0.35))
    config_safety_mode = str(config.get("safety_mode", "recover"))
    config_safety_recover_speed = float(config.get("safety_recover_speed", 0.35))
    config_safety_recover_forward_speed = float(
        config.get("safety_recover_forward_speed", 0.15)
    )
    config_safety_recover_centerline_gain = float(
        config.get("safety_recover_centerline_gain", 0.4)
    )
    if (
        requested_safety_min_dist is not None
        and abs(float(requested_safety_min_dist) - config_safety_min_dist) > 1e-9
    ):
        raise ValueError(
            f"--safety-min-dist {requested_safety_min_dist} does not match existing "
            f"safety_min_dist {config_safety_min_dist}"
        )
    if requested_safety_mode is not None and requested_safety_mode != config_safety_mode:
        raise ValueError(
            f"--safety-mode {requested_safety_mode} does not match existing "
            f"safety_mode {config_safety_mode}"
        )
    for flag, requested_value, config_value in (
        ("--safety-recover-speed", requested_safety_recover_speed, config_safety_recover_speed),
        (
            "--safety-recover-forward-speed",
            requested_safety_recover_forward_speed,
            config_safety_recover_forward_speed,
        ),
        (
            "--safety-recover-centerline-gain",
            requested_safety_recover_centerline_gain,
            config_safety_recover_centerline_gain,
        ),
    ):
        if requested_value is not None and abs(float(requested_value) - config_value) > 1e-9:
            raise ValueError(
                f"{flag} {requested_value} does not match existing value {config_value}"
            )

    replay_defaults = {
        "input_source": "online",
        "replay_dataset_path": "",
        "replay_dataset_format": "hdf5",
        "replay_sampling_mode": "raw",
        "replay_trajectory_index": -1,
        "replay_start_offset": -1,
        "replay_loop": True,
    }
    for key, requested_value in (
        ("input_source", requested_input_source),
        ("replay_dataset_path", requested_replay_dataset_path),
        ("replay_dataset_format", requested_replay_dataset_format),
        ("replay_sampling_mode", requested_replay_sampling_mode),
        ("replay_trajectory_index", requested_replay_trajectory_index),
        ("replay_start_offset", requested_replay_start_offset),
        ("replay_loop", requested_replay_loop),
    ):
        config_value = config.get(key, replay_defaults[key])
        if requested_value is not None and requested_value != config_value:
            raise ValueError(
                f"--{key.replace('_', '-')} {requested_value} does not match "
                f"existing value {config_value}"
            )

    map_defaults = {
        "map_sampling_mode": "uniform",
        "min_obstacle_spacing": 0.0,
        "local_density_window": 3.0,
        "max_obstacles_per_window": 4,
        "max_local_area_fraction": 0.45,
        "require_connectivity": False,
        "min_bottleneck_width": 0.4,
        "max_resample_attempts": 2000,
    }
    for key, requested_value in (
        ("map_sampling_mode", requested_map_sampling_mode),
        ("min_obstacle_spacing", requested_min_obstacle_spacing),
        ("local_density_window", requested_local_density_window),
        ("max_obstacles_per_window", requested_max_obstacles_per_window),
        ("max_local_area_fraction", requested_max_local_area_fraction),
        ("require_connectivity", requested_require_connectivity),
        ("min_bottleneck_width", requested_min_bottleneck_width),
        ("max_resample_attempts", requested_max_resample_attempts),
    ):
        config_value = config.get(key, map_defaults[key])
        if requested_value is not None and requested_value != config_value:
            raise ValueError(
                f"--{key.replace('_', '-')} {requested_value} does not match "
                f"existing value {config_value}"
            )

    args.num_batches = int(config["num_batches"])
    args.runs_per_batch = int(config["runs_per_batch"])
    args.methods = config_methods
    args.master_seed = int(config["master_seed"])
    args.goal_x = float(config.get("goal_x", args.goal_x))
    args.collision_dist = float(config.get("collision_dist", args.collision_dist))
    args.safety_min_dist = config_safety_min_dist
    args.safety_mode = config_safety_mode
    args.safety_recover_speed = config_safety_recover_speed
    args.safety_recover_forward_speed = config_safety_recover_forward_speed
    args.safety_recover_centerline_gain = config_safety_recover_centerline_gain
    args.device = config.get("device", args.device)
    args.checkpoint = config.get("checkpoint", args.checkpoint)
    args.gazebo_z_mode = config.get("gazebo_z_mode", args.gazebo_z_mode)
    args.gazebo_policy_z_max = float(
        config.get("gazebo_policy_z_max", args.gazebo_policy_z_max)
    )
    args.gazebo_z_blend_alpha = float(
        config.get("gazebo_z_blend_alpha", args.gazebo_z_blend_alpha)
    )
    args.disable_gazebo_policy_z_takeoff_gate = not bool(
        config.get(
            "gazebo_policy_z_takeoff_gate",
            not args.disable_gazebo_policy_z_takeoff_gate,
        )
    )
    args.gazebo_policy_z_gate_tolerance = float(
        config.get("gazebo_policy_z_gate_tolerance", args.gazebo_policy_z_gate_tolerance)
    )
    args.disable_policy_takeoff_gate = not bool(
        config.get("policy_takeoff_gate", not args.disable_policy_takeoff_gate)
    )
    args.policy_takeoff_gate_tolerance = float(
        config.get("policy_takeoff_gate_tolerance", args.policy_takeoff_gate_tolerance)
    )
    args.gui = bool(config.get("gui", args.gui))
    args.rviz = bool(config.get("rviz", args.rviz))
    args.user_model_simple = bool(config.get("user_model_simple", args.user_model_simple))
    args.user_model_profile = config.get("user_model_profile", args.user_model_profile)
    args.user_model_speed = float(config.get("user_model_speed", args.user_model_speed))
    args.user_model_freq_base = float(
        config.get("user_model_freq_base", args.user_model_freq_base)
    )
    args.user_model_freq_scale = float(
        config.get("user_model_freq_scale", args.user_model_freq_scale)
    )
    args.user_model_vx_bias = float(config.get("user_model_vx_bias", args.user_model_vx_bias))
    args.user_model_vx_amp = float(config.get("user_model_vx_amp", args.user_model_vx_amp))
    args.user_model_vy_amp = float(config.get("user_model_vy_amp", args.user_model_vy_amp))
    args.user_model_vz_amp = float(config.get("user_model_vz_amp", args.user_model_vz_amp))
    args.user_model_smoothness_base = float(
        config.get("user_model_smoothness_base", args.user_model_smoothness_base)
    )
    args.user_model_smoothness_scale = float(
        config.get("user_model_smoothness_scale", args.user_model_smoothness_scale)
    )
    args.user_model_laziness = float(
        config.get("user_model_laziness", args.user_model_laziness)
    )
    args.input_source = config.get("input_source", replay_defaults["input_source"])
    args.replay_dataset_path = config.get(
        "replay_dataset_path", replay_defaults["replay_dataset_path"]
    )
    args.replay_dataset_format = config.get(
        "replay_dataset_format", replay_defaults["replay_dataset_format"]
    )
    args.replay_sampling_mode = config.get(
        "replay_sampling_mode", replay_defaults["replay_sampling_mode"]
    )
    args.replay_trajectory_index = int(
        config.get("replay_trajectory_index", replay_defaults["replay_trajectory_index"])
    )
    args.replay_start_offset = int(
        config.get("replay_start_offset", replay_defaults["replay_start_offset"])
    )
    args.replay_loop = bool(config.get("replay_loop", replay_defaults["replay_loop"]))
    args.num_obstacles = int(config.get("num_obstacles", args.num_obstacles))
    args.cuboid_ratio = float(config.get("cuboid_ratio", args.cuboid_ratio))
    args.map_resolution = float(config.get("map_resolution", args.map_resolution))
    args.map_sampling_mode = config.get("map_sampling_mode", map_defaults["map_sampling_mode"])
    args.min_obstacle_spacing = float(
        config.get("min_obstacle_spacing", map_defaults["min_obstacle_spacing"])
    )
    args.local_density_window = float(
        config.get("local_density_window", map_defaults["local_density_window"])
    )
    args.max_obstacles_per_window = int(
        config.get("max_obstacles_per_window", map_defaults["max_obstacles_per_window"])
    )
    args.max_local_area_fraction = float(
        config.get("max_local_area_fraction", map_defaults["max_local_area_fraction"])
    )
    args.require_connectivity = bool(
        config.get("require_connectivity", map_defaults["require_connectivity"])
    )
    args.min_bottleneck_width = float(
        config.get("min_bottleneck_width", map_defaults["min_bottleneck_width"])
    )
    args.max_resample_attempts = int(
        config.get("max_resample_attempts", map_defaults["max_resample_attempts"])
    )
    args.launch_timeout = float(config.get("launch_timeout", args.launch_timeout))
    args.recorder_timeout = float(config.get("recorder_timeout", args.recorder_timeout))
    args.completion_grace_period = float(
        config.get("completion_grace_period", args.completion_grace_period)
    )
    spawn = config.get("spawn")
    if spawn and len(spawn) == 3:
        args.spawn_x, args.spawn_y, args.spawn_z = (
            float(spawn[0]),
            float(spawn[1]),
            float(spawn[2]),
        )

    return config


def validate_offline_replay_config(args):
    if args.input_source != "offline":
        return

    dataset_path = str(args.replay_dataset_path or "").strip()
    if not dataset_path:
        raise ValueError(
            "--input-source offline requires --replay-dataset-path to be set"
        )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Offline replay dataset not found: {dataset_path}"
        )
    if importlib.util.find_spec("h5py") is None:
        raise RuntimeError(
            "Offline replay requires the Python package 'h5py', but it is not "
            "installed in this runtime. Rebuild the tunnel comparison image after "
            "adding h5py, or install it in the container before launching the batch."
        )

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Offline replay requires the Python package 'h5py', but it failed to import."
        ) from exc

    try:
        with h5py.File(dataset_path, "r") as handle:
            if "velocities" not in handle:
                raise KeyError(f"HDF5 dataset has no 'velocities': {dataset_path}")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to open offline replay dataset: {dataset_path}"
        ) from exc


def resolve_device(requested_device):
    requested_device = str(requested_device).strip()
    if not requested_device.startswith("cuda"):
        return requested_device

    try:
        import torch
    except ImportError:
        print(f"[Batch] WARNING: torch is unavailable; falling back from {requested_device} to cpu")
        return "cpu"

    if not torch.cuda.is_available():
        print(
            f"[Batch] WARNING: requested device {requested_device}, but this runtime "
            f"has no CUDA-enabled torch. Falling back to cpu."
        )
        return "cpu"

    return requested_device


def generate_seed_plan(args):
    rng = random.Random(args.master_seed)
    plan = []
    for batch_idx in range(args.num_batches):
        map_seed = rng.randint(0, 2**31 - 1)
        run_seeds = [rng.randint(0, 2**31 - 1) for _ in range(args.runs_per_batch)]
        plan.append({
            "batch_idx": batch_idx,
            "map_seed": map_seed,
            "run_seeds": run_seeds,
        })
    return plan


def seed_plan_from_config(config):
    plan = []
    for batch in config.get("batches", []):
        plan.append({
            "batch_idx": int(batch["batch_idx"]),
            "map_seed": int(batch["map_seed"]),
            "run_seeds": [int(seed) for seed in batch["run_seeds"]],
        })
    return plan


def ensure_batch_assets(args, output_root, batch_dir, batch_idx, map_seed, resume):
    map_dir = os.path.join(batch_dir, "map")
    map_path = os.path.join(map_dir, "tunnel_map.pcd")
    world_path = os.path.join(map_dir, "tunnel.world")
    metadata_path = os.path.join(map_dir, "obstacles.json")
    root_metadata = os.path.join(output_root, f"b{batch_idx:03d}_obstacles.json")

    assets_exist = (
        os.path.exists(map_path)
        and os.path.exists(world_path)
        and os.path.exists(metadata_path)
    )
    if resume and assets_exist:
        if not os.path.exists(root_metadata):
            shutil.copyfile(metadata_path, root_metadata)
        return map_path, world_path, metadata_path

    return generate_batch_assets(args, batch_dir, batch_idx, map_seed)


def generate_batch_assets(args, batch_dir, batch_idx, map_seed):
    map_dir = os.path.join(batch_dir, "map")
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, "tunnel_map.pcd")
    world_path = os.path.join(map_dir, "tunnel.world")
    metadata_path = os.path.join(map_dir, "obstacles.json")

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tunnel_deployment",
        "generate_tunnel_map.py",
    )
    cmd = [
        sys.executable,
        script_path,
        "--output", map_path,
        "--world-output", world_path,
        "--metadata-output", metadata_path,
        "--seed", str(map_seed),
        "--num-obstacles", str(args.num_obstacles),
        "--cuboid-ratio", str(args.cuboid_ratio),
        "--resolution", str(args.map_resolution),
        "--sampling-mode", str(args.map_sampling_mode),
        "--min-obstacle-spacing", str(args.min_obstacle_spacing),
        "--local-density-window", str(args.local_density_window),
        "--max-obstacles-per-window", str(args.max_obstacles_per_window),
        "--max-local-area-fraction", str(args.max_local_area_fraction),
        "--min-bottleneck-width", str(args.min_bottleneck_width),
        "--max-resample-attempts", str(args.max_resample_attempts),
    ]
    if args.require_connectivity:
        cmd.append("--require-connectivity")
    subprocess.run(cmd, check=True)

    root_metadata = os.path.join(os.path.dirname(batch_dir), f"b{batch_idx:03d}_obstacles.json")
    shutil.copyfile(metadata_path, root_metadata)
    return map_path, world_path, metadata_path


def build_roslaunch_cmd(args, method, run_dir, trial_id, run_id, batch_idx,
                        run_idx, map_seed, user_model_seed, map_path, world_path):
    bool_str = lambda value: "true" if value else "false"
    return [
        "roslaunch",
        "navigation_runner",
        "tunnel_comparison.launch",
        f"method:={method}",
        f"gui:={bool_str(args.gui)}",
        f"rviz:={bool_str(args.rviz)}",
        "record:=true",
        "auto_terminate:=true",
        "shutdown_on_complete:=true",
        f"timeout_sec:={args.recorder_timeout}",
        f"completion_grace_period:={args.completion_grace_period}",
        f"tunnel_map:={map_path}",
        f"tunnel_world:={world_path}",
        f"output_dir:={run_dir}",
        f"trial_id:={trial_id}",
        f"run_id:={run_id}",
        f"batch_idx:={batch_idx}",
        f"run_idx:={run_idx}",
        f"map_seed:={map_seed}",
        f"user_model_seed:={user_model_seed}",
        f"goal_x:={args.goal_x}",
        f"collision_dist:={args.collision_dist}",
        f"safety_min_dist:={args.safety_min_dist}",
        f"safety_mode:={args.safety_mode}",
        f"safety_recover_speed:={args.safety_recover_speed}",
        f"safety_recover_forward_speed:={args.safety_recover_forward_speed}",
        f"safety_recover_centerline_gain:={args.safety_recover_centerline_gain}",
        f"device:={args.device}",
        f"checkpoint:={args.checkpoint}",
        f"policy_mode:={args.policy_mode}",
        f"gazebo_z_mode:={args.gazebo_z_mode}",
        f"gazebo_policy_z_max:={args.gazebo_policy_z_max}",
        f"gazebo_z_blend_alpha:={args.gazebo_z_blend_alpha}",
        f"gazebo_policy_z_takeoff_gate:={bool_str(not args.disable_gazebo_policy_z_takeoff_gate)}",
        f"gazebo_policy_z_gate_tolerance:={args.gazebo_policy_z_gate_tolerance}",
        f"policy_takeoff_gate:={bool_str(not args.disable_policy_takeoff_gate)}",
        f"policy_takeoff_gate_tolerance:={args.policy_takeoff_gate_tolerance}",
        f"user_model_simple:={bool_str(args.user_model_simple)}",
        f"user_model_profile:={args.user_model_profile}",
        f"user_model_speed:={args.user_model_speed}",
        f"user_model_freq_base:={args.user_model_freq_base}",
        f"user_model_freq_scale:={args.user_model_freq_scale}",
        f"user_model_vx_bias:={args.user_model_vx_bias}",
        f"user_model_vx_amp:={args.user_model_vx_amp}",
        f"user_model_vy_amp:={args.user_model_vy_amp}",
        f"user_model_vz_amp:={args.user_model_vz_amp}",
        f"user_model_smoothness_base:={args.user_model_smoothness_base}",
        f"user_model_smoothness_scale:={args.user_model_smoothness_scale}",
        f"user_model_laziness:={args.user_model_laziness}",
        f"input_source:={args.input_source}",
        f"replay_dataset_path:={args.replay_dataset_path}",
        f"replay_dataset_format:={args.replay_dataset_format}",
        f"replay_sampling_mode:={args.replay_sampling_mode}",
        f"replay_trajectory_index:={args.replay_trajectory_index}",
        f"replay_start_offset:={args.replay_start_offset}",
        f"replay_loop:={bool_str(args.replay_loop)}",
        f"spawn_x:={args.spawn_x}",
        f"spawn_y:={args.spawn_y}",
        f"spawn_z:={args.spawn_z}",
    ]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)


def write_manifest(output_root, manifest):
    write_json(os.path.join(output_root, "batch_manifest.json"), manifest)


def load_existing_manifest(output_root):
    manifest_path = os.path.join(output_root, "batch_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        return load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Batch] WARNING: ignoring unreadable existing manifest: {exc}")
        return None


def run_manifest_key(summary):
    return (
        int(summary.get("batch_idx", -1)),
        int(summary.get("run_idx", -1)),
        str(summary.get("method", "")).lower(),
    )


def upsert_manifest_run(manifest, summary):
    key = run_manifest_key(summary)
    runs = []
    replaced = False
    for existing in manifest.get("runs", []):
        if run_manifest_key(existing) == key:
            runs.append(summary)
            replaced = True
        else:
            runs.append(existing)
    if not replaced:
        runs.append(summary)
    manifest["runs"] = runs


def expected_seed_plan_matches(args, config_plan):
    generated_plan = generate_seed_plan(args)
    return generated_plan == config_plan


def select_seed_plan(args, seed_plan):
    if args.batch_index is not None:
        if args.batch_start is not None or args.batch_end is not None:
            raise ValueError("--batch-index cannot be combined with --batch-start/--batch-end")
        start = end = args.batch_index
    else:
        start = 0 if args.batch_start is None else args.batch_start
        end = len(seed_plan) - 1 if args.batch_end is None else args.batch_end

    if start < 0 or end < 0 or start > end:
        raise ValueError(f"Invalid batch selection: start={start} end={end}")
    max_batch = len(seed_plan) - 1
    if end > max_batch:
        raise ValueError(f"Batch selection ends at {end}, but max batch index is {max_batch}")

    selected = [
        batch
        for batch in seed_plan
        if start <= int(batch["batch_idx"]) <= end
    ]
    if not selected:
        raise ValueError(f"No batches selected for range {start}..{end}")
    return selected, [int(batch["batch_idx"]) for batch in selected]


def detect_log_failures(log_path):
    if not os.path.exists(log_path):
        return []
    hits = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for marker in FATAL_LOG_MARKERS:
                    if marker in line and marker not in hits:
                        hits.append(marker)
    except OSError as exc:
        hits.append(f"unreadable log: {exc}")
    return hits


def run_attempt_failed(timed_out, exit_code, fatal_errors):
    return bool(timed_out or fatal_errors or (exit_code not in (0, None)))


def data_path_for_summary(run_dir, summary):
    data_file = summary.get("data_file", "")
    if not data_file:
        return ""
    if os.path.isabs(data_file):
        return data_file
    return os.path.join(run_dir, data_file)


def load_existing_run_summary(run_dir):
    summary_path = os.path.join(run_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        return None, "missing run_summary.json"
    try:
        return load_json(summary_path), ""
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid run_summary.json: {exc}"


def summary_matches_expected(summary, method, trial_id, batch_idx, run_idx,
                             map_seed, user_model_seed):
    expected = {
        "method": method,
        "trial_id": int(trial_id),
        "batch_idx": int(batch_idx),
        "run_idx": int(run_idx),
        "map_seed": int(map_seed),
        "user_model_seed": int(user_model_seed),
    }
    actual_method = str(summary.get("method", "")).lower()
    if actual_method != expected["method"]:
        return False, f"method mismatch: {actual_method} != {expected['method']}"
    for key, expected_value in expected.items():
        if key == "method":
            continue
        try:
            actual_value = int(summary.get(key))
        except (TypeError, ValueError):
            return False, f"{key} missing or invalid"
        if actual_value != expected_value:
            return False, f"{key} mismatch: {actual_value} != {expected_value}"
    return True, ""


def trajectory_file_is_complete(data_path, min_samples):
    if not data_path:
        return False, "missing data_file"
    if not os.path.exists(data_path):
        return False, f"missing data file: {data_path}"
    if os.path.getsize(data_path) <= 0:
        return False, f"empty data file: {data_path}"

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to validate resume data files") from exc

    try:
        with np.load(data_path) as data:
            if "timestamps" not in data:
                return False, "data file has no timestamps array"
            sample_count = len(data["timestamps"])
    except Exception as exc:
        return False, f"unreadable data file: {exc}"

    if sample_count < min_samples:
        return False, f"data file samples {sample_count} < {min_samples}"
    return True, ""


def existing_run_is_complete(run_dir, method, trial_id, batch_idx, run_idx,
                             map_seed, user_model_seed, min_samples):
    summary, reason = load_existing_run_summary(run_dir)
    if summary is None:
        return None, reason

    matches, reason = summary_matches_expected(
        summary, method, trial_id, batch_idx, run_idx, map_seed, user_model_seed
    )
    if not matches:
        return None, reason
    if summary.get("timed_out"):
        return None, "previous run timed out"
    if summary.get("fatal_errors"):
        return None, "previous run recorded fatal log markers"
    try:
        exit_code = int(summary.get("exit_code", 0))
    except (TypeError, ValueError):
        return None, "previous run exit_code missing or invalid"
    if exit_code != 0:
        return None, f"previous run exit_code {exit_code}"

    try:
        summary_samples = int(summary.get("samples", 0))
    except (TypeError, ValueError):
        summary_samples = 0
    if summary_samples < min_samples:
        return None, f"summary samples {summary_samples} < {min_samples}"

    complete, reason = trajectory_file_is_complete(
        data_path_for_summary(run_dir, summary),
        min_samples,
    )
    if not complete:
        return None, reason

    return summary, ""


def normalize_existing_summary(summary, run_dir, method, trial_id, run_id, batch_idx,
                               run_idx, map_seed, user_model_seed, log_path,
                               map_path, world_path, checkpoint):
    summary = dict(summary)
    summary["method"] = method
    summary["trial_id"] = trial_id
    summary["run_id"] = run_id
    summary["batch_idx"] = batch_idx
    summary["run_idx"] = run_idx
    summary["map_seed"] = map_seed
    summary["user_model_seed"] = user_model_seed
    summary["log_file"] = os.path.relpath(log_path, os.path.dirname(run_dir))
    summary["pcd_file"] = map_path
    summary["tunnel_world"] = world_path
    summary["checkpoint"] = checkpoint
    summary["run_dir"] = run_dir
    write_json(os.path.join(run_dir, "run_summary.json"), summary)
    return summary


def collect_run_summary(run_dir, method, trial_id, run_id, batch_idx, run_idx,
                         map_seed, user_model_seed, timed_out, exit_code, log_path,
                         map_path, world_path, checkpoint, fatal_errors=None,
                         attempt=1, max_attempts=1):
    fatal_errors = fatal_errors or []
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        summary = load_json(summary_path)
    else:
        summary = {
            "method": method,
            "trial_id": trial_id,
            "run_id": run_id,
            "batch_idx": batch_idx,
            "run_idx": run_idx,
            "map_seed": map_seed,
            "user_model_seed": user_model_seed,
            "goal_reached": False,
            "collision": False,
            "termination_reason": "timeout" if timed_out else "missing_summary",
            "total_time": 0.0,
            "samples": 0,
            "max_x": float("-inf"),
            "min_obstacle_dist": float("inf"),
            "data_file": "",
            "pcd_file": map_path,
            "tunnel_world": world_path,
        }
        npz_files = sorted(glob.glob(os.path.join(run_dir, "*.npz")))
        if npz_files:
            summary["data_file"] = os.path.basename(npz_files[-1])

    summary["exit_code"] = exit_code
    summary["timed_out"] = timed_out
    summary["fatal_errors"] = fatal_errors
    summary["attempt"] = attempt
    summary["max_attempts"] = max_attempts
    if fatal_errors:
        summary["termination_reason"] = "fatal_log_error"
    elif timed_out:
        summary["termination_reason"] = "timeout"
    elif exit_code not in (0, None) and summary.get("termination_reason") == "missing_summary":
        summary["termination_reason"] = "roslaunch_exit_nonzero"
    summary["log_file"] = os.path.relpath(log_path, os.path.dirname(run_dir))
    summary["pcd_file"] = summary.get("pcd_file", map_path) or map_path
    summary["tunnel_world"] = summary.get("tunnel_world", world_path) or world_path
    summary["checkpoint"] = checkpoint
    summary["run_dir"] = run_dir

    write_json(summary_path, summary)
    return summary


def run_batch(args, output_root):
    resume = bool(args.resume_from)
    resume_config = load_existing_batch_config(output_root) if resume else None
    if resume:
        seed_plan = seed_plan_from_config(resume_config)
        if not expected_seed_plan_matches(args, seed_plan):
            raise ValueError(
                "Existing batch seed plan does not match the requested resume configuration"
            )
    else:
        seed_plan = generate_seed_plan(args)
    methods = normalize_methods(args.methods)
    args.methods = ",".join(methods)
    execution_plan, selected_batch_indices = select_seed_plan(args, seed_plan)
    cleanup_experiment_processes("batch start")
    existing_manifest = load_existing_manifest(output_root)
    manifest = {
        "batch_config": {
            "num_batches": args.num_batches,
            "runs_per_batch": args.runs_per_batch,
            "num_trials": args.num_batches * args.runs_per_batch,
            "methods": methods,
            "master_seed": args.master_seed,
            "goal_x": args.goal_x,
            "collision_dist": args.collision_dist,
            "safety_min_dist": args.safety_min_dist,
            "safety_mode": args.safety_mode,
            "safety_recover_speed": args.safety_recover_speed,
            "safety_recover_forward_speed": args.safety_recover_forward_speed,
            "safety_recover_centerline_gain": args.safety_recover_centerline_gain,
            "device": args.device,
            "checkpoint": args.checkpoint,
            "gazebo_z_mode": args.gazebo_z_mode,
            "gazebo_policy_z_max": args.gazebo_policy_z_max,
            "gazebo_z_blend_alpha": args.gazebo_z_blend_alpha,
            "gazebo_policy_z_takeoff_gate": not args.disable_gazebo_policy_z_takeoff_gate,
            "gazebo_policy_z_gate_tolerance": args.gazebo_policy_z_gate_tolerance,
            "policy_takeoff_gate": not args.disable_policy_takeoff_gate,
            "policy_takeoff_gate_tolerance": args.policy_takeoff_gate_tolerance,
            "gui": args.gui,
            "rviz": args.rviz,
            "user_model_simple": args.user_model_simple,
            "user_model_profile": args.user_model_profile,
            "user_model_speed": args.user_model_speed,
            "user_model_freq_base": args.user_model_freq_base,
            "user_model_freq_scale": args.user_model_freq_scale,
            "user_model_vx_bias": args.user_model_vx_bias,
            "user_model_vx_amp": args.user_model_vx_amp,
            "user_model_vy_amp": args.user_model_vy_amp,
            "user_model_vz_amp": args.user_model_vz_amp,
            "user_model_smoothness_base": args.user_model_smoothness_base,
            "user_model_smoothness_scale": args.user_model_smoothness_scale,
            "user_model_laziness": args.user_model_laziness,
            "input_source": args.input_source,
            "replay_dataset_path": args.replay_dataset_path,
            "replay_dataset_format": args.replay_dataset_format,
            "replay_sampling_mode": args.replay_sampling_mode,
            "replay_trajectory_index": args.replay_trajectory_index,
            "replay_start_offset": args.replay_start_offset,
            "replay_loop": args.replay_loop,
            "num_obstacles": args.num_obstacles,
            "cuboid_ratio": args.cuboid_ratio,
            "map_resolution": args.map_resolution,
            "map_sampling_mode": args.map_sampling_mode,
            "min_obstacle_spacing": args.min_obstacle_spacing,
            "local_density_window": args.local_density_window,
            "max_obstacles_per_window": args.max_obstacles_per_window,
            "max_local_area_fraction": args.max_local_area_fraction,
            "require_connectivity": args.require_connectivity,
            "min_bottleneck_width": args.min_bottleneck_width,
            "max_resample_attempts": args.max_resample_attempts,
            "launch_timeout": args.launch_timeout,
            "recorder_timeout": args.recorder_timeout,
            "completion_grace_period": args.completion_grace_period,
            "spawn": [args.spawn_x, args.spawn_y, args.spawn_z],
            "batches": seed_plan,
        },
        "selected_batches": selected_batch_indices,
        "runs": list((existing_manifest or {}).get("runs", [])),
    }
    if resume:
        manifest["resume"] = {
            "resumed_from": output_root,
            "min_complete_samples": args.min_complete_samples,
        }

    write_json(os.path.join(output_root, "batch_config.json"), manifest["batch_config"])

    total_runs = len(execution_plan) * args.runs_per_batch * len(methods)
    completed_runs = 0
    skipped_runs = 0
    rerun_runs = 0

    print(
        f"[Batch] Executing batch indices: "
        f"{','.join(str(index) for index in selected_batch_indices)}"
    )

    for batch_info in execution_plan:
        batch_idx = batch_info["batch_idx"]
        batch_dir = os.path.join(output_root, f"batch_{batch_idx:03d}")
        runs_dir = os.path.join(batch_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)

        map_path, world_path, metadata_path = ensure_batch_assets(
            args, output_root, batch_dir, batch_idx, batch_info["map_seed"], resume
        )
        batch_info["map_path"] = map_path
        batch_info["world_path"] = world_path
        batch_info["metadata_path"] = metadata_path

        for run_idx, user_model_seed in enumerate(batch_info["run_seeds"]):
            trial_id = batch_idx * args.runs_per_batch + run_idx
            for method in methods:
                run_dir = os.path.join(runs_dir, f"{method}_trial_{run_idx:03d}")
                os.makedirs(run_dir, exist_ok=True)
                run_id = f"b{batch_idx:03d}_{method}_r{run_idx:03d}_seed{user_model_seed}"
                log_path = os.path.join(run_dir, "launch.log")

                if resume:
                    existing_summary, incomplete_reason = existing_run_is_complete(
                        run_dir=run_dir,
                        method=method,
                        trial_id=trial_id,
                        batch_idx=batch_idx,
                        run_idx=run_idx,
                        map_seed=batch_info["map_seed"],
                        user_model_seed=user_model_seed,
                        min_samples=args.min_complete_samples,
                    )
                    if existing_summary is not None:
                        summary = normalize_existing_summary(
                            summary=existing_summary,
                            run_dir=run_dir,
                            method=method,
                            trial_id=trial_id,
                            run_id=run_id,
                            batch_idx=batch_idx,
                            run_idx=run_idx,
                            map_seed=batch_info["map_seed"],
                            user_model_seed=user_model_seed,
                            log_path=log_path,
                            map_path=map_path,
                            world_path=world_path,
                            checkpoint=args.checkpoint,
                        )
                        upsert_manifest_run(manifest, summary)
                        skipped_runs += 1
                        completed_runs += 1
                        print(
                            f"[Batch] Skip complete {completed_runs}/{total_runs} "
                            f"batch={batch_idx} run={run_idx} method={method}"
                        )
                        continue

                    print(
                        f"[Batch] Re-run incomplete batch={batch_idx} run={run_idx} "
                        f"method={method}: {incomplete_reason}"
                    )
                    if os.path.isdir(run_dir):
                        shutil.rmtree(run_dir)
                    os.makedirs(run_dir, exist_ok=True)
                    log_path = os.path.join(run_dir, "launch.log")
                    rerun_runs += 1

                max_attempts = args.run_retries + 1
                summary = None
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        print(
                            f"[Batch] Retry {attempt}/{max_attempts} for "
                            f"batch={batch_idx} run={run_idx} method={method}"
                        )
                        if os.path.isdir(run_dir):
                            shutil.rmtree(run_dir)
                        os.makedirs(run_dir, exist_ok=True)
                        log_path = os.path.join(run_dir, "launch.log")

                    cleanup_experiment_processes(f"before {run_id}")
                    run_env = build_run_env(run_dir, completed_runs)
                    cmd = build_roslaunch_cmd(
                        args=args,
                        method=method,
                        run_dir=run_dir,
                        trial_id=trial_id,
                        run_id=run_id,
                        batch_idx=batch_idx,
                        run_idx=run_idx,
                        map_seed=batch_info["map_seed"],
                        user_model_seed=user_model_seed,
                        map_path=map_path,
                        world_path=world_path,
                    )

                    print(
                        f"[Batch] {completed_runs + 1}/{total_runs} "
                        f"batch={batch_idx} run={run_idx} method={method} "
                        f"seed={user_model_seed} attempt={attempt}/{max_attempts}"
                    )

                    timed_out = False
                    exit_code = None
                    with open(log_path, "w", encoding="utf-8") as log_handle:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid,
                            env=run_env,
                        )
                        try:
                            exit_code = proc.wait(timeout=args.launch_timeout)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            kill_process_group(proc)
                            exit_code = -signal.SIGTERM

                    fatal_errors = detect_log_failures(log_path)
                    summary = collect_run_summary(
                        run_dir=run_dir,
                        method=method,
                        trial_id=trial_id,
                        run_id=run_id,
                        batch_idx=batch_idx,
                        run_idx=run_idx,
                        map_seed=batch_info["map_seed"],
                        user_model_seed=user_model_seed,
                        timed_out=timed_out,
                        exit_code=exit_code,
                        log_path=log_path,
                        map_path=map_path,
                        world_path=world_path,
                        checkpoint=args.checkpoint,
                        fatal_errors=fatal_errors,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    cleanup_experiment_processes(f"after {run_id}")
                    if not run_attempt_failed(timed_out, exit_code, fatal_errors):
                        break
                    if attempt < max_attempts:
                        print(
                            f"[Batch] Run failed; retrying after errors: "
                            f"{fatal_errors or ['timeout/nonzero exit']}"
                        )

                upsert_manifest_run(manifest, summary)
                write_manifest(output_root, manifest)

                completed_runs += 1
                time.sleep(args.inter_run_delay)

    write_manifest(output_root, manifest)
    if resume:
        print(f"[Batch] Resume summary: skipped={skipped_runs} rerun={rerun_runs}")
    return manifest


def maybe_run_analysis(args, output_root):
    if not args.analyze:
        return
    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_results.py")
    analysis_dir = os.path.join(output_root, "analysis")
    cmd = [
        sys.executable,
        analyze_script,
        "--data-dir", output_root,
        "--output-dir", analysis_dir,
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    if args.run_retries < 0:
        raise ValueError("--run-retries must be non-negative")
    output_root = ensure_output_dir(args)
    if args.resume_from:
        apply_resume_config(args, output_root)
    else:
        apply_default_args(args)
    if args.user_model_simple:
        args.user_model_profile = "simple"
    args.device = resolve_device(args.device)
    validate_method_config(args)
    validate_offline_replay_config(args)
    manifest = run_batch(args, output_root)
    maybe_run_analysis(args, output_root)
    print(f"[Batch] Finished. Output root: {output_root}")
    print(f"[Batch] Recorded runs: {len(manifest['runs'])}")


if __name__ == "__main__":
    main()
