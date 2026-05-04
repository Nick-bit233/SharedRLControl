#!/usr/bin/env python3
"""Sequentially run or resume iteration-2 targeted tunnel screening candidates.

This host-side helper intentionally defaults to dry-run mode.  Pass --run to
launch the long Gazebo batches in the tunnel_debug container.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


MASTER_SEED = 5716
DEFAULT_CONTAINER = "tunnel_debug"
CONTAINER_SCRIPT_DIR = "/root/catkin_ws/src/navigation_runner/scripts"
BATCH_SCRIPT = os.path.join(CONTAINER_SCRIPT_DIR, "batch_tunnel_experiments.py")
ANALYZE_SCRIPT = os.path.join(CONTAINER_SCRIPT_DIR, "analyze_results.py")

ROS_SETUP_LINES = (
    "set -eo pipefail",
    "export ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}",
    "source /opt/ros/noetic/setup.bash",
    "source /root/slope_ws/devel/setup.bash",
    "source /root/catkin_ws/devel/setup.bash",
    "export ROS_IP=127.0.0.1",
    "export ROS_HOSTNAME=127.0.0.1",
    "cd /root/catkin_ws",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    output_dir: str
    safety_min_dist: float
    safety_mode: str
    gazebo_z_mode: str
    gazebo_policy_z_max: float
    gazebo_z_blend_alpha: float


CANDIDATES = (
    Candidate(
        name="zfix_hold030_alt_hold",
        output_dir="/root/results/iter2_zfix_hold030_alt_hold_seed5716",
        safety_min_dist=0.30,
        safety_mode="hold",
        gazebo_z_mode="alt_hold",
        gazebo_policy_z_max=2.0,
        gazebo_z_blend_alpha=0.5,
    ),
    Candidate(
        name="target_hold025_alt_hold",
        output_dir="/root/results/iter2_target_hold025_alt_hold_seed5716",
        safety_min_dist=0.25,
        safety_mode="hold",
        gazebo_z_mode="alt_hold",
        gazebo_policy_z_max=2.0,
        gazebo_z_blend_alpha=0.5,
    ),
    Candidate(
        name="target_hold020_alt_hold",
        output_dir="/root/results/iter2_target_hold020_alt_hold_seed5716",
        safety_min_dist=0.20,
        safety_mode="hold",
        gazebo_z_mode="alt_hold",
        gazebo_policy_z_max=2.0,
        gazebo_z_blend_alpha=0.5,
    ),
    Candidate(
        name="target_hold025_clamp035",
        output_dir="/root/results/iter2_target_hold025_clamp035_seed5716",
        safety_min_dist=0.25,
        safety_mode="hold",
        gazebo_z_mode="policy_clamped",
        gazebo_policy_z_max=0.35,
        gazebo_z_blend_alpha=0.5,
    ),
    Candidate(
        name="target_hold025_blend025",
        output_dir="/root/results/iter2_target_hold025_blend025_seed5716",
        safety_min_dist=0.25,
        safety_mode="hold",
        gazebo_z_mode="blend",
        gazebo_policy_z_max=2.0,
        gazebo_z_blend_alpha=0.25,
    ),
)

SUMMARY_CODE = r"""
import csv
import json
import math
import os

candidates = json.loads(os.environ["ITER2_CANDIDATES_JSON"])


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def fmt_pct(value):
    return "n/a" if not math.isfinite(value) else f"{value * 100.0:.1f}%"


def fmt_float(value):
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def summarize_from_metrics(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rl_rows = [row for row in rows if row.get("method") == "rl"]
    ipc_rows = [row for row in rows if row.get("method") == "ipc"]
    successes = [parse_bool(row.get("goal_reached")) for row in rl_rows]
    collisions = [parse_bool(row.get("collision")) for row in rl_rows]
    traps = [parse_bool(row.get("likely_safety_hold_trap")) for row in rl_rows]
    success_times = [
        parse_float(row.get("total_time"))
        for row in rl_rows
        if parse_bool(row.get("goal_reached"))
    ]
    ipc_successes = [parse_bool(row.get("goal_reached")) for row in ipc_rows]
    ipc_collisions = [parse_bool(row.get("collision")) for row in ipc_rows]
    return {
        "rl_runs": len(rl_rows),
        "rl_success": mean([float(value) for value in successes]),
        "rl_collision": mean([float(value) for value in collisions]),
        "rl_tcr1": mean([parse_float(row.get("tcr_at_1")) for row in rl_rows]),
        "rl_tcr2": mean([parse_float(row.get("tcr_at_2")) for row in rl_rows]),
        "rl_tcr5": mean([parse_float(row.get("tcr_at_5")) for row in rl_rows]),
        "rl_success_time": mean(success_times),
        "rl_trap": mean([float(value) for value in traps]),
        "rl_exposure": mean([parse_float(row.get("pct_close_safety_min")) for row in rl_rows]),
        "ipc_success": mean([float(value) for value in ipc_successes]),
        "ipc_collision": mean([float(value) for value in ipc_collisions]),
    }


def summarize_from_json(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rl = data.get("methods", {}).get("rl", {})
    ipc = data.get("methods", {}).get("ipc", {})
    return {
        "rl_runs": int(rl.get("count", 0)),
        "rl_success": float(rl.get("success_rate", float("nan"))),
        "rl_collision": float(rl.get("collision_rate", float("nan"))),
        "rl_tcr1": float(rl.get("tcr_at_1_mean", float("nan"))),
        "rl_tcr2": float(rl.get("tcr_at_2_mean", float("nan"))),
        "rl_tcr5": float(rl.get("tcr_at_5_mean", float("nan"))),
        "rl_success_time": float("nan"),
        "rl_trap": float(rl.get("likely_safety_hold_trap_rate", float("nan"))),
        "rl_exposure": float(rl.get("pct_close_safety_min_mean", float("nan"))),
        "ipc_success": float(ipc.get("success_rate", float("nan"))),
        "ipc_collision": float(ipc.get("collision_rate", float("nan"))),
    }


print("\nIteration-2 targeted screening summary")
print("-" * 139)
print(
    f"{'candidate':<24} {'runs':>4} {'RL succ':>8} {'RL coll':>8} "
    f"{'TCR@1':>7} {'TCR@2':>7} {'TCR@5':>7} {'succ time':>9} "
    f"{'trap':>8} {'exposure':>9} {'IPC succ':>8} {'IPC coll':>8}"
)
print("-" * 139)

for candidate in candidates:
    analysis_dir = os.path.join(candidate["output_dir"], "analysis")
    metrics_path = os.path.join(analysis_dir, "metrics.csv")
    summary_path = os.path.join(analysis_dir, "summary.json")
    if os.path.exists(metrics_path):
        summary = summarize_from_metrics(metrics_path)
    elif os.path.exists(summary_path):
        summary = summarize_from_json(summary_path)
    else:
        print(f"{candidate['name']:<24} missing analysis: {analysis_dir}")
        continue
    success_time = summary["rl_success_time"]
    success_time_text = "n/a" if not math.isfinite(success_time) else f"{success_time:.1f}s"
    print(
        f"{candidate['name']:<24} {summary['rl_runs']:>4} "
        f"{fmt_pct(summary['rl_success']):>8} {fmt_pct(summary['rl_collision']):>8} "
        f"{fmt_float(summary['rl_tcr1']):>7} {fmt_float(summary['rl_tcr2']):>7} "
        f"{fmt_float(summary['rl_tcr5']):>7} {success_time_text:>9} "
        f"{fmt_pct(summary['rl_trap']):>8} {fmt_float(summary['rl_exposure']):>9} "
        f"{fmt_pct(summary['ipc_success']):>8} {fmt_pct(summary['ipc_collision']):>8}"
    )
print("-" * 139)
"""


def format_float(value):
    return f"{value:.12g}"


def shell_join(argv):
    return " ".join(shlex.quote(str(part)) for part in argv)


def with_ros_setup(command_lines):
    return "\n".join(list(ROS_SETUP_LINES) + list(command_lines))


def candidate_names():
    return [candidate.name for candidate in CANDIDATES]


def selected_candidates(name):
    if name == "all":
        return list(CANDIDATES)
    return [candidate for candidate in CANDIDATES if candidate.name == name]


def candidate_output_dir(candidate, args):
    if not args.output_tag:
        return candidate.output_dir
    basename = os.path.basename(candidate.output_dir)
    if basename.startswith("iter2_"):
        basename = basename[len("iter2_"):]
    return os.path.join(args.output_root, f"{args.output_tag}_{basename}")


def build_batch_argv(candidate, args, resume):
    output_dir = candidate_output_dir(candidate, args)
    argv = [
        "python3",
        BATCH_SCRIPT,
        "--num-batches",
        str(args.num_batches),
        "--runs-per-batch",
        str(args.runs_per_batch),
        "--methods",
        "rl,ipc",
        "--master-seed",
        str(MASTER_SEED),
        "--goal-x",
        "10.0",
        "--collision-dist",
        "0.05",
        "--num-obstacles",
        "15",
        "--cuboid-ratio",
        "0.5",
        "--recorder-timeout",
        "60",
        "--device",
        "cpu",
        "--gazebo-z-mode",
        candidate.gazebo_z_mode,
        "--gazebo-policy-z-max",
        format_float(candidate.gazebo_policy_z_max),
        "--gazebo-z-blend-alpha",
        format_float(candidate.gazebo_z_blend_alpha),
        "--safety-min-dist",
        format_float(candidate.safety_min_dist),
        "--safety-mode",
        candidate.safety_mode,
        "--launch-timeout",
        format_float(args.launch_timeout),
        "--inter-run-delay",
        format_float(args.inter_run_delay),
    ]
    if resume:
        argv.extend(["--resume-from", output_dir])
    else:
        argv.extend(["--output-dir", output_dir])
    return argv


def build_analyze_inner(candidate, args):
    candidate_dir = candidate_output_dir(candidate, args)
    output_dir = shlex.quote(candidate_dir)
    analyze_cmd = shell_join(
        [
            "python3",
            ANALYZE_SCRIPT,
            "--data-dir",
            candidate_dir,
            "--output-dir",
            os.path.join(candidate_dir, "analysis"),
        ]
    )
    return with_ros_setup(
        [
            f"if [ ! -d {output_dir} ]; then",
            f"  echo 'ERROR: missing output directory for analysis: {candidate_dir}' >&2",
            "  exit 21",
            "fi",
            analyze_cmd,
        ]
    )


def build_batch_inner(candidate, args):
    candidate_dir = candidate_output_dir(candidate, args)
    output_dir = shlex.quote(candidate_dir)
    fresh_cmd = shell_join(build_batch_argv(candidate, args, resume=False))
    resume_cmd = shell_join(build_batch_argv(candidate, args, resume=True))
    analyze_cmd = shell_join(
        [
            "python3",
            ANALYZE_SCRIPT,
            "--data-dir",
            candidate_dir,
            "--output-dir",
            os.path.join(candidate_dir, "analysis"),
        ]
    )

    lines = [
        f"echo '=== iteration-2 candidate: {candidate.name} ==='",
        f"if [ -e {output_dir} ] && [ ! -d {output_dir} ]; then",
        f"  echo 'ERROR: output path exists but is not a directory: {candidate_dir}' >&2",
        "  exit 22",
        "fi",
    ]
    if args.resume:
        lines.extend(
            [
                f"if [ -d {output_dir} ]; then",
                f"  echo 'Resuming existing output directory: {candidate_dir}'",
                f"  {resume_cmd}",
                "else",
                f"  echo 'Output directory is absent; starting fresh: {candidate_dir}'",
                f"  {fresh_cmd}",
                "fi",
            ]
        )
    elif args.skip_existing:
        lines.extend(
            [
                f"if [ -d {output_dir} ]; then",
                f"  echo 'Skipping existing output directory: {candidate_dir}'",
                f"  {analyze_cmd}",
                "else",
                f"  {fresh_cmd}",
                "fi",
            ]
        )
    else:
        lines.extend(
            [
                f"if [ -d {output_dir} ]; then",
                "  echo 'ERROR: output directory already exists. Use --resume or --skip-existing.' >&2",
                f"  echo 'Refusing to overwrite: {candidate_dir}' >&2",
                "  exit 23",
                "fi",
                fresh_cmd,
            ]
        )
    return with_ros_setup(lines)


def build_summary_inner(candidates, args):
    payload = json.dumps(
        [
            {"name": candidate.name, "output_dir": candidate_output_dir(candidate, args)}
            for candidate in candidates
        ],
        sort_keys=True,
    )
    return with_ros_setup(
        [
            f"ITER2_CANDIDATES_JSON={shlex.quote(payload)} python3 - <<'PY'",
            SUMMARY_CODE,
            "PY",
        ]
    )


def docker_argv(container, inner_command):
    return ["docker", "exec", container, "bash", "-lc", inner_command]


def run_inner(inner_command, args, label, allow_failure=False):
    if args.print_inner_command:
        print(f"\n# {label}")
        print(inner_command)
        return 0

    argv = docker_argv(args.container, inner_command)
    if not args.run:
        print(f"\n# DRY-RUN: {label}")
        print(shell_join(argv))
        return 0

    print(f"\n# RUN: {label}")
    completed = subprocess.run(argv, check=False)
    if completed.returncode and not allow_failure:
        print(f"ERROR: {label} failed with exit code {completed.returncode}", file=sys.stderr)
    return completed.returncode


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run/resume iteration-2 targeted tunnel screening candidates sequentially."
    )
    parser.add_argument("--candidate", default="all", choices=["all"] + candidate_names())
    parser.add_argument("--run", action="store_true", help="Actually execute docker commands")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only (default)")
    parser.add_argument("--resume", action="store_true", help="Resume existing candidate dirs in-place")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not launch a candidate whose output dir already exists; re-run analysis instead",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analyze_results.py for selected existing candidate dirs",
    )
    parser.add_argument(
        "--print-inner-command",
        action="store_true",
        help="Print the container-internal bash command instead of docker exec",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--output-root", default="/root/results")
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional output prefix, e.g. quick2 -> /root/results/quick2_target_hold025_alt_hold_seed5716",
    )
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--runs-per-batch", type=int, default=10)
    parser.add_argument("--launch-timeout", type=float, default=100.0)
    parser.add_argument("--inter-run-delay", type=float, default=2.0)
    parsed = parser.parse_args(argv)

    if parsed.run and parsed.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    if parsed.resume and parsed.skip_existing:
        parser.error("--resume and --skip-existing are mutually exclusive")
    if parsed.num_batches <= 0 or parsed.runs_per_batch <= 0:
        parser.error("--num-batches and --runs-per-batch must be positive")
    if parsed.launch_timeout <= 0.0:
        parser.error("--launch-timeout must be positive")
    if parsed.inter_run_delay < 0.0:
        parser.error("--inter-run-delay must be non-negative")
    return parsed


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    candidates = selected_candidates(args.candidate)
    failures = []

    if not args.run and not args.print_inner_command:
        print("Dry run only. Pass --run to launch experiments.")
    print(f"Master seed: {MASTER_SEED}")
    print("Selected candidates: " + ", ".join(candidate.name for candidate in candidates))

    for candidate in candidates:
        if args.analyze_only:
            rc = run_inner(build_analyze_inner(candidate, args), args, f"analyze {candidate.name}")
        else:
            rc = run_inner(build_batch_inner(candidate, args), args, f"batch {candidate.name}")
            if rc != 0 and args.run and not args.print_inner_command:
                print(f"Attempting standalone analysis for {candidate.name} after batch failure.")
                run_inner(
                    build_analyze_inner(candidate, args),
                    args,
                    f"fallback analyze {candidate.name}",
                    allow_failure=True,
                )
        if rc != 0:
            failures.append((candidate.name, rc))

    if not failures and (args.run or args.print_inner_command):
        run_inner(build_summary_inner(candidates, args), args, "summarize iteration-2", allow_failure=True)
    elif not failures:
        print("\n# DRY-RUN: summary command omitted; pass --print-inner-command to print it.")

    if failures:
        for name, rc in failures:
            print(f"FAILED: {name} exit={rc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
