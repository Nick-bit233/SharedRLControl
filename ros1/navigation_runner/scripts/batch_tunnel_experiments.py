#!/usr/bin/env python3
"""Batch runner for ROS1 tunnel experiments."""

import argparse
import glob
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
    parser.add_argument("--num-batches", type=int, default=1,
                        help="Number of map batches to generate")
    parser.add_argument("--runs-per-batch", type=int, default=5,
                        help="Paired RL/IPC runs per batch")
    parser.add_argument("--methods", default="rl,ipc",
                        help="Comma-separated method order, e.g. rl,ipc")
    parser.add_argument("--master-seed", type=int, default=42,
                        help="Master RNG seed for map/user-model seeds")
    parser.add_argument("--output-dir", default=None,
                        help="Output root (default: auto timestamp under ./batch_results)")
    parser.add_argument("--launch-timeout", type=float, default=80.0,
                        help="External watchdog timeout per run in seconds")
    parser.add_argument("--recorder-timeout", type=float, default=60.0,
                        help="Optional recorder-side timeout (0 disables)")
    parser.add_argument("--completion-grace-period", type=float, default=0.5,
                        help="Grace period between stop signal and roslaunch shutdown")
    parser.add_argument("--inter-run-delay", type=float, default=2.0,
                        help="Delay between sequential runs")
    parser.add_argument("--goal-x", type=float, default=10.0)
    parser.add_argument("--collision-dist", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="RL checkpoint passed to tunnel_comparison.launch")
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
    parser.add_argument("--user-model-vz-amp", type=float, default=0.0)
    parser.add_argument("--user-model-smoothness-base", type=float, default=0.4)
    parser.add_argument("--user-model-smoothness-scale", type=float, default=0.5)
    parser.add_argument("--user-model-laziness", type=float, default=0.3)
    parser.add_argument("--num-obstacles", type=int, default=15)
    parser.add_argument("--cuboid-ratio", type=float, default=0.5)
    parser.add_argument("--map-resolution", type=float, default=0.1)
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
    if args.output_dir:
        root = os.path.abspath(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join("/root/catkin_ws/results", f"batch_{timestamp}")
    os.makedirs(root, exist_ok=True)
    return root


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
    ]
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
        f"device:={args.device}",
        f"checkpoint:={args.checkpoint}",
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
        f"spawn_x:={args.spawn_x}",
        f"spawn_y:={args.spawn_y}",
        f"spawn_z:={args.spawn_z}",
    ]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_run_summary(run_dir, method, trial_id, run_id, batch_idx, run_idx,
                         map_seed, user_model_seed, timed_out, exit_code, log_path,
                         map_path, world_path, checkpoint):
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
    summary["log_file"] = os.path.relpath(log_path, os.path.dirname(run_dir))
    summary["pcd_file"] = summary.get("pcd_file", map_path) or map_path
    summary["tunnel_world"] = summary.get("tunnel_world", world_path) or world_path
    summary["checkpoint"] = checkpoint
    summary["run_dir"] = run_dir

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_batch(args, output_root):
    seed_plan = generate_seed_plan(args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cleanup_experiment_processes("batch start")
    manifest = {
        "batch_config": {
            "num_batches": args.num_batches,
            "runs_per_batch": args.runs_per_batch,
            "num_trials": args.num_batches * args.runs_per_batch,
            "methods": methods,
            "master_seed": args.master_seed,
            "goal_x": args.goal_x,
            "collision_dist": args.collision_dist,
            "device": args.device,
            "checkpoint": args.checkpoint,
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
            "num_obstacles": args.num_obstacles,
            "cuboid_ratio": args.cuboid_ratio,
            "map_resolution": args.map_resolution,
            "launch_timeout": args.launch_timeout,
            "recorder_timeout": args.recorder_timeout,
            "completion_grace_period": args.completion_grace_period,
            "spawn": [args.spawn_x, args.spawn_y, args.spawn_z],
            "batches": seed_plan,
        },
        "runs": [],
    }

    with open(os.path.join(output_root, "batch_config.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest["batch_config"], handle, indent=2)

    total_runs = args.num_batches * args.runs_per_batch * len(methods)
    completed_runs = 0

    for batch_info in seed_plan:
        batch_idx = batch_info["batch_idx"]
        batch_dir = os.path.join(output_root, f"batch_{batch_idx:03d}")
        runs_dir = os.path.join(batch_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)

        map_path, world_path, metadata_path = generate_batch_assets(
            args, batch_dir, batch_idx, batch_info["map_seed"]
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
                    f"batch={batch_idx} run={run_idx} method={method} seed={user_model_seed}"
                )

                proc = None
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
                )
                manifest["runs"].append(summary)
                with open(os.path.join(output_root, "batch_manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)

                cleanup_experiment_processes(f"after {run_id}")
                completed_runs += 1
                time.sleep(args.inter_run_delay)

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
    if args.user_model_simple:
        args.user_model_profile = "simple"
    args.device = resolve_device(args.device)
    output_root = ensure_output_dir(args)
    manifest = run_batch(args, output_root)
    maybe_run_analysis(args, output_root)
    print(f"[Batch] Finished. Output root: {output_root}")
    print(f"[Batch] Recorded runs: {len(manifest['runs'])}")


if __name__ == "__main__":
    main()
