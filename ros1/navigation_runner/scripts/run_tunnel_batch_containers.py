#!/usr/bin/env python3
"""Host-side launcher that runs one ROS1 tunnel batch per Docker container."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.tunnel.yml")
CONTAINER_BATCH_SCRIPT = "/root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py"
CONTAINER_ANALYZE_SCRIPT = "/root/catkin_ws/src/navigation_runner/scripts/analyze_results.py"
RESULT_PREFIXES = ("/root/results", "/root/catkin_ws/results")
SLOPE_BUILD_PACKAGES = "mars_quadrotor_msgs;mars_planning_utils;mars_base;rog_map;ipc"


def shell_join(argv):
    return " ".join(shlex.quote(str(part)) for part in argv)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run ROS1 tunnel comparison experiments from the host, launching "
            "one disposable headless container per selected batch. Unknown "
            "arguments are forwarded to batch_tunnel_experiments.py."
        )
    )
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--service", default="tunnel_batch")
    parser.add_argument("--project-name", default="tunnelbatch")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Container path under /root/results, or host path under ros1/results")
    parser.add_argument("--resume-from", default=None,
                        help="Resume an existing output root; accepts container or host path")
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int, default=None)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--run", action="store_true",
                        help="Execute commands. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only. This is the default unless --run is set.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-analyze", dest="analyze", action="store_false",
                        help="Skip final analysis container")
    parser.set_defaults(analyze=True)
    parser.add_argument("--rebuild-slope", dest="rebuild_slope", action="store_true",
                        help="Run catkin_make -C /root/slope_ws inside each container")
    parser.add_argument("--no-rebuild-slope", dest="rebuild_slope", action="store_false")
    parser.set_defaults(rebuild_slope=True)
    parser.add_argument("--log-dir", default="",
                        help="Host log directory; default: <output-root>/host_logs")
    args, batch_args = parser.parse_known_args(argv)

    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    if args.batch_index is not None and (args.start_batch != 0 or args.end_batch is not None):
        parser.error("--batch-index cannot be combined with --start-batch/--end-batch")
    if args.output_dir and args.resume_from:
        parser.error("--output-dir and --resume-from are mutually exclusive")
    if not args.output_dir and not args.resume_from:
        parser.error("one of --output-dir or --resume-from is required")
    if args.num_batches is not None and args.num_batches <= 0:
        parser.error("--num-batches must be positive")
    return args, batch_args


def host_results_root():
    return os.path.join(REPO_ROOT, "ros1", "results")


def normalize_output_paths(path):
    raw_path = os.path.normpath(path)
    for prefix in RESULT_PREFIXES:
        if raw_path == prefix:
            return host_results_root(), "/root/results"
        if raw_path.startswith(prefix + "/"):
            suffix = raw_path[len(prefix):].lstrip("/")
            return os.path.join(host_results_root(), suffix), f"/root/results/{suffix}"

    if os.path.isabs(raw_path):
        host_path = raw_path
    else:
        host_path = os.path.abspath(os.path.join(REPO_ROOT, raw_path))

    results_root = host_results_root()
    try:
        rel = os.path.relpath(host_path, results_root)
    except ValueError:
        rel = None
    if rel and not rel.startswith("..") and rel != ".":
        return host_path, f"/root/results/{rel}"
    if host_path == results_root:
        return host_path, "/root/results"
    raise ValueError(
        f"Output path must be under ros1/results or /root/results: {path}"
    )


def load_num_batches(host_output_dir):
    config_path = os.path.join(host_output_dir, "batch_config.json")
    manifest_path = os.path.join(host_output_dir, "batch_manifest.json")
    for path in (config_path, manifest_path):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        config = data.get("batch_config", data)
        if "num_batches" in config:
            return int(config["num_batches"])
    return None


def selected_indices(args, host_output_dir):
    num_batches = args.num_batches or load_num_batches(host_output_dir)
    if num_batches is None:
        raise ValueError("--num-batches is required when the output root has no batch_config.json")
    if args.batch_index is not None:
        indices = [args.batch_index]
    else:
        end = num_batches - 1 if args.end_batch is None else args.end_batch
        indices = list(range(args.start_batch, end + 1))
    if not indices or min(indices) < 0 or max(indices) >= num_batches:
        raise ValueError(f"Invalid batch selection {indices}; num_batches={num_batches}")
    return num_batches, indices


def compose_base(args):
    return [
        "docker",
        "compose",
        "-f",
        args.compose_file,
        "-p",
        args.project_name,
    ]


def container_shell(batch_argv, rebuild_slope):
    lines = [
        "set -eo pipefail",
        "source /opt/ros/noetic/setup.bash",
    ]
    if rebuild_slope:
        lines.extend(
            [
                "echo '[host-runner] Rebuilding mounted slope_ws overlay'",
                f"catkin_make -C /root/slope_ws -j2 -l2 -DCATKIN_WHITELIST_PACKAGES={shlex.quote(SLOPE_BUILD_PACKAGES)}",
                'if [ "$(readlink /root/slope_ws/src/CMakeLists.txt 2>/dev/null)" = "/opt/ros/noetic/share/catkin/cmake/toplevel.cmake" ]; then rm -f /root/slope_ws/src/CMakeLists.txt; fi',
            ]
        )
    lines.extend(
        [
            "source /root/slope_ws/devel/setup.bash",
            "source /root/catkin_ws/devel/setup.bash",
            "export ROS_IP=127.0.0.1",
            "export ROS_HOSTNAME=127.0.0.1",
            "cd /root/catkin_ws",
            shell_join(batch_argv),
        ]
    )
    return "\n".join(lines)


def docker_run_argv(args, container_name, inner_command):
    return compose_base(args) + [
        "run",
        "--rm",
        "--name",
        container_name,
        "-e",
        "TUNNEL_RENDER_MODE=headless",
        args.service,
        "bash",
        "-lc",
        inner_command,
    ]


def run_logged(argv, log_path, dry_run):
    print(shell_join(argv))
    if dry_run:
        return 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n# {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(shell_join(argv) + "\n")
        log.flush()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def write_status(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def batch_container_name(args, batch_idx):
    prefix = args.experiment_name or "tunnel_batch"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prefix)
    return f"{safe}_b{batch_idx:03d}"


def run_batch_container(args, batch_args, num_batches, batch_idx, host_output_dir, container_output_dir):
    if args.run:
        os.makedirs(host_output_dir, exist_ok=True)
    use_resume = bool(args.resume_from) or os.path.exists(os.path.join(host_output_dir, "batch_config.json"))
    batch_argv = [
        "python3",
        CONTAINER_BATCH_SCRIPT,
        "--num-batches",
        str(num_batches),
        "--batch-index",
        str(batch_idx),
        "--no-analyze",
    ]
    batch_argv.extend(batch_args)
    if use_resume:
        batch_argv.extend(["--resume-from", container_output_dir])
    else:
        batch_argv.extend(["--output-dir", container_output_dir])

    container_name = batch_container_name(args, batch_idx)
    log_dir = args.log_dir or os.path.join(host_output_dir, "host_logs")
    log_path = os.path.join(log_dir, f"batch_{batch_idx:03d}.docker.log")
    status_path = os.path.join(log_dir, f"batch_{batch_idx:03d}.status.json")
    argv = docker_run_argv(args, container_name, container_shell(batch_argv, args.rebuild_slope))

    start = time.time()
    returncode = run_logged(argv, log_path, dry_run=not args.run)
    if args.run:
        write_status(
            status_path,
            {
                "batch_idx": batch_idx,
                "container_name": container_name,
                "returncode": returncode,
                "duration_sec": time.time() - start,
                "log_path": log_path,
                "command": argv,
            },
        )
    return returncode


def run_analysis_container(args, host_output_dir, container_output_dir):
    log_dir = args.log_dir or os.path.join(host_output_dir, "host_logs")
    log_path = os.path.join(log_dir, "analysis.docker.log")
    inner = container_shell(
        [
            "python3",
            CONTAINER_ANALYZE_SCRIPT,
            "--data-dir",
            container_output_dir,
            "--output-dir",
            os.path.join(container_output_dir, "analysis"),
        ],
        args.rebuild_slope,
    )
    argv = docker_run_argv(args, "tunnel_batch_analysis", inner)
    return run_logged(argv, log_path, dry_run=not args.run)


def main(argv=None):
    args, batch_args = parse_args(argv if argv is not None else sys.argv[1:])
    requested_output = args.resume_from or args.output_dir
    host_output_dir, container_output_dir = normalize_output_paths(requested_output)
    num_batches, indices = selected_indices(args, host_output_dir)
    dry_run = not args.run

    print("Dry run only. Pass --run to launch containers." if dry_run else "Executing containers.")
    print(f"Output root: host={host_output_dir} container={container_output_dir}")
    print(f"Selected batches: {','.join(str(index) for index in indices)}")

    failures = []
    for batch_idx in indices:
        rc = run_batch_container(
            args,
            batch_args,
            num_batches,
            batch_idx,
            host_output_dir,
            container_output_dir,
        )
        if rc != 0:
            failures.append((batch_idx, rc))
            if not args.continue_on_error:
                break

    if not failures and args.analyze:
        rc = run_analysis_container(args, host_output_dir, container_output_dir)
        if rc != 0:
            failures.append(("analysis", rc))

    if failures:
        for batch_idx, rc in failures:
            print(f"FAILED: {batch_idx} exit={rc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
