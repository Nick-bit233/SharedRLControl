#!/usr/bin/env python3
"""Batch experiment runner for IPC vs RL comparison.

Runs compare_ipc_rl.py multiple times, each with a different terrain seed
(and optionally different user-model start_seed), then merges all results
into a single output directory with unified statistics.

Usage examples:
    # 10 batches × 5 trials = 50 trials per method, random seeds
    python batch_experiment.py --num_batches 10 --trials_per_batch 5

    # Fixed terrain seeds for reproducibility
    python batch_experiment.py --num_batches 3 --terrain_seeds 42 100 200

    # With IPC speed profile and custom checkpoint
    python batch_experiment.py --num_batches 5 --ipc_speed_profile balanced \
        --checkpoint /path/to/model.pt

    # Resume from a previous run (skips completed batches)
    python batch_experiment.py --resume /path/to/batch_output_dir
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


def _kill_proc_tree(pid: int):
    """Kill a process group (all children spawned by the subprocess)."""
    import signal
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # Give 5s for graceful shutdown, then SIGKILL
    time.sleep(5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner: compare_ipc_rl.py across multiple terrain maps")

    # Batch structure
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches (each = one simulator launch with a unique terrain)")
    parser.add_argument("--trials_per_batch", type=int, default=5,
                        help="Trials per batch per method (default: 5)")
    parser.add_argument("--num_frames", type=int, default=1600,
                        help="Frames per trial (default: 1600)")
    parser.add_argument("--timeout_per_trial", type=int, default=600,
                        help="Max seconds per trial before force-killing batch (default: 600 = 10 min)")

    # Seed control
    parser.add_argument("--terrain_seeds", type=int, nargs="*", default=None,
                        help="Explicit terrain seeds (one per batch). If not given, "
                             "random seeds are generated from --master_seed.")
    parser.add_argument("--master_seed", type=int, default=12345,
                        help="Master seed for generating terrain/trial seeds (default: 12345)")

    # Passthrough to compare_ipc_rl.py
    parser.add_argument("--num_obstacles", type=int, default=30)
    parser.add_argument("--success_x", type=float, default=12.0)
    parser.add_argument("--model_type", type=str, default="ConstrainedBeta",
                        choices=["Simple", "Residual", "Constrained", "ConstrainedBeta"])
    parser.add_argument("--checkpoint", type=str,
                        default="/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/"
                                "shared_demos/ckpts/260331/checkpoint_final.pt")
    parser.add_argument("--ipc_config", type=str, default=None)
    parser.add_argument("--no_sfc", action="store_true")
    parser.add_argument("--ipc_speed_profile", type=str, default=None,
                        choices=["fast", "balanced"])
    parser.add_argument("--tcr_spacing", type=float, default=0.2)
    parser.add_argument("--state_dim", type=int, default=10, choices=[10, 11])
    parser.add_argument("--debug", action="store_true",
                        help="Pass --debug to each compare_ipc_rl.py run")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated with timestamp)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing batch output directory (skip completed batches)")

    return parser.parse_args()


def generate_seeds(master_seed: int, num_batches: int):
    """Generate reproducible (terrain_seed, start_seed) pairs."""
    rng = random.Random(master_seed)
    pairs = []
    for _ in range(num_batches):
        terrain_seed = rng.randint(0, 2**31 - 1)
        start_seed = rng.randint(0, 2**31 - 1)
        pairs.append((terrain_seed, start_seed))
    return pairs


def build_command(args, terrain_seed: int, start_seed: int, batch_dir: str) -> list:
    """Build the compare_ipc_rl.py subprocess command."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, "compare_ipc_rl.py")
    python = sys.executable

    cmd = [
        python, script,
        "--terrain_seed", str(terrain_seed),
        "--start_seed", str(start_seed),
        "--num_trials", str(args.trials_per_batch),
        "--num_frames", str(args.num_frames),
        "--num_obstacles", str(args.num_obstacles),
        "--success_x", str(args.success_x),
        "--model_type", args.model_type,
        "--checkpoint", args.checkpoint,
        "--state_dim", str(args.state_dim),
        "--tcr_spacing", str(args.tcr_spacing),
        "--viz", "none",
        "--headless",
        "--skip_aggregate",
        "--log_dir", batch_dir,
    ]
    if args.no_sfc:
        cmd.append("--no_sfc")
    if args.ipc_speed_profile:
        cmd.extend(["--ipc_speed_profile", args.ipc_speed_profile])
    if args.ipc_config:
        cmd.extend(["--ipc_config", args.ipc_config])
    if args.debug:
        cmd.append("--debug")
    return cmd


def find_latest_results(log_base: str) -> tuple:
    """Find the latest compare_results JSON and flight data dir in log_base."""
    json_files = sorted(glob.glob(os.path.join(log_base, "compare_results_*.json")))
    if not json_files:
        return None, None
    latest_json = json_files[-1]
    data_dirs = sorted(glob.glob(os.path.join(log_base, "flight_data_*")))
    latest_data = data_dirs[-1] if data_dirs else None
    return latest_json, latest_data


def load_batch_results(json_path: str) -> dict | None:
    """Load and validate a batch results JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        if "per_trial" not in data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None


def aggregate_all(all_trials: dict) -> dict:
    """Compute aggregated statistics across all trials."""
    agg = {}
    for method in ["IPC", "RL"]:
        trials = all_trials.get(method, [])
        if not trials:
            agg[method] = {}
            continue
        keys = trials[0].keys()
        method_agg = {}
        for k in keys:
            vals = [t[k] for t in trials]
            if isinstance(vals[0], bool):
                method_agg[f"{k}_rate"] = sum(vals) / len(vals)
            elif isinstance(vals[0], (int, float)):
                arr = np.array(vals, dtype=float)
                method_agg[f"{k}_mean"] = float(np.mean(arr))
                method_agg[f"{k}_std"] = float(np.std(arr))
            else:
                method_agg[k] = vals
        agg[method] = method_agg
    return agg


def print_results_table(agg: dict, total_trials: int, num_batches: int):
    """Print formatted comparison table."""
    ipc_agg = agg.get("IPC", {})
    rl_agg = agg.get("RL", {})

    print(f"\n{'='*70}")
    print(f"BATCH RESULTS: IPC vs RL  ({total_trials} trials across {num_batches} terrains)")
    print(f"{'='*70}")

    key_metrics = [
        ("Success Rate",           "success_rate",              "{:.1%}"),
        ("Max X Reached (mean)",   "max_x_reached_mean",       "{:.2f}"),
        ("Max X Reached (std)",    "max_x_reached_std",        "{:.2f}"),
        ("TCR@1 (mean)",           "tcr_at_1_mean",             "{:.3f}"),
        ("TCR@1 (std)",            "tcr_at_1_std",              "{:.3f}"),
        ("TCR@2 (mean)",           "tcr_at_2_mean",             "{:.3f}"),
        ("TCR@2 (std)",            "tcr_at_2_std",              "{:.3f}"),
        ("TCR@5 (mean)",           "tcr_at_5_mean",             "{:.3f}"),
        ("TCR@5 (std)",            "tcr_at_5_std",              "{:.3f}"),
        ("Latency mean ms",       "inference_mean_ms_mean",    "{:.2f}"),
        ("Latency p50 ms",        "inference_p50_ms_mean",     "{:.2f}"),
        ("Latency p95 ms",        "inference_p95_ms_mean",     "{:.2f}"),
        ("Inf/Frame Ratio (mean)", "inference_frame_ratio_mean", "{:.3f}"),
        ("Ctrl FPS (mean)",        "ctrl_fps_mean",             "{:.1f}"),
        ("Path Length m (mean)",   "path_length_m_mean",        "{:.2f}"),
        ("Avg Speed (mean)",       "avg_speed_mean",            "{:.4f}"),
        ("SFC Success Rate",       "sfc_success_rate_mean",     "{:.1%}"),
    ]

    header = f"{'Metric':<30} {'IPC':>14} {'RL':>14}"
    print(header)
    print("-" * 70)
    for label, key, fmt in key_metrics:
        iv = ipc_agg.get(key, "N/A")
        rv = rl_agg.get(key, "N/A")
        iv_str = fmt.format(iv) if isinstance(iv, (int, float)) else str(iv)
        rv_str = fmt.format(rv) if isinstance(rv, (int, float)) else str(rv)
        print(f"{label:<30} {iv_str:>14} {rv_str:>14}")
    print("=" * 70)


def main():
    args = parse_args()

    # --- Output directory ---
    if args.resume:
        output_dir = args.resume
        if not os.path.isdir(output_dir):
            print(f"ERROR: Resume directory not found: {output_dir}")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs", f"batch_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # --- Seed generation ---
    if args.terrain_seeds:
        if len(args.terrain_seeds) < args.num_batches:
            print(f"WARNING: Only {len(args.terrain_seeds)} terrain seeds given for "
                  f"{args.num_batches} batches. Generating remaining from master_seed.")
            rng = random.Random(args.master_seed)
            extra = [(rng.randint(0, 2**31-1), rng.randint(0, 2**31-1))
                     for _ in range(args.num_batches - len(args.terrain_seeds))]
            seed_pairs = [(ts, rng.randint(0, 2**31-1)) for ts in args.terrain_seeds] + extra
        else:
            rng = random.Random(args.master_seed)
            seed_pairs = [(ts, rng.randint(0, 2**31-1)) for ts in args.terrain_seeds[:args.num_batches]]
    else:
        seed_pairs = generate_seeds(args.master_seed, args.num_batches)

    # --- Save batch config ---
    batch_config = {
        "num_batches": args.num_batches,
        "trials_per_batch": args.trials_per_batch,
        "num_frames": args.num_frames,
        "timeout_per_trial": args.timeout_per_trial,
        "master_seed": args.master_seed,
        "seed_pairs": [{"terrain_seed": ts, "start_seed": ss} for ts, ss in seed_pairs],
        "num_obstacles": args.num_obstacles,
        "success_x": args.success_x,
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "use_sfc": not args.no_sfc,
        "ipc_speed_profile": args.ipc_speed_profile,
        "tcr_spacing": args.tcr_spacing,
    }
    config_path = os.path.join(output_dir, "batch_config.json")
    with open(config_path, 'w') as f:
        json.dump(batch_config, f, indent=2)
    print(f"Batch config saved to: {config_path}")

    # --- Determine which batches are already done (for resume) ---
    completed_batches = set()
    if args.resume:
        for i in range(args.num_batches):
            batch_dir = os.path.join(output_dir, f"batch_{i:03d}")
            marker = os.path.join(batch_dir, ".done")
            if os.path.exists(marker):
                completed_batches.add(i)
        if completed_batches:
            print(f"  Already completed: {sorted(completed_batches)}")

    # --- Run batches ---
    batch_results = []  # list of (batch_idx, json_path, data_dir, status)
    total_start = time.time()

    for batch_idx in range(args.num_batches):
        terrain_seed, start_seed = seed_pairs[batch_idx]
        batch_dir = os.path.join(output_dir, f"batch_{batch_idx:03d}")

        if batch_idx in completed_batches:
            json_path, data_dir = find_latest_results(batch_dir)
            if json_path:
                batch_results.append((batch_idx, json_path, data_dir, "resumed"))
                continue

        os.makedirs(batch_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Batch {batch_idx+1}/{args.num_batches}: "
              f"terrain_seed={terrain_seed}, start_seed={start_seed}")
        print(f"{'='*60}")

        cmd = build_command(args, terrain_seed, start_seed, batch_dir)
        print(f"  CMD: {' '.join(cmd[:6])}...")

        # Timeout = startup overhead + (per-trial budget × num_trials × 2 methods)
        batch_timeout = 120 + args.timeout_per_trial * args.trials_per_batch * 2
        print(f"  Timeout: {batch_timeout}s "
              f"({args.timeout_per_trial}s/trial × {args.trials_per_batch} trials × 2 methods + 120s overhead)")

        t0 = time.time()
        proc = None
        try:
            proc = subprocess.Popen(cmd, start_new_session=True)
            returncode = proc.wait(timeout=batch_timeout)
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT after {elapsed:.0f}s — killing process tree...")
            _kill_proc_tree(proc.pid)
            returncode = -1
        except KeyboardInterrupt:
            if proc is not None:
                _kill_proc_tree(proc.pid)
            print(f"\n  Interrupted at batch {batch_idx+1}. Aggregating completed results...")
            break

        if returncode == 0:
            status = "ok"
            # Mark as done for resume
            with open(os.path.join(batch_dir, ".done"), 'w') as f:
                f.write(f"completed at {datetime.now().isoformat()}\n")
        else:
            status = f"exit_{returncode}"
            print(f"  WARNING: Batch {batch_idx+1} exited with code {returncode} "
                  f"after {elapsed:.0f}s — will use any partial results.")

        # Find results regardless of exit code (incremental save may have data)
        json_path, data_dir = find_latest_results(batch_dir)
        if json_path:
            batch_results.append((batch_idx, json_path, data_dir, status))
            print(f"  Batch {batch_idx+1} finished in {elapsed:.1f}s [{status}]")
        else:
            print(f"  Batch {batch_idx+1}: No results found (crash before first trial?)")

    total_elapsed = time.time() - total_start

    # --- Merge all results ---
    print(f"\n{'='*60}")
    print(f"MERGING RESULTS from {len(batch_results)} batches")
    print(f"{'='*60}")

    merged_trials = {"IPC": [], "RL": []}
    merged_data_dir = os.path.join(output_dir, "all_flight_data")
    os.makedirs(merged_data_dir, exist_ok=True)

    batch_summaries = []
    for batch_idx, json_path, data_dir, status in batch_results:
        data = load_batch_results(json_path)
        if data is None:
            print(f"  Batch {batch_idx}: invalid JSON, skipping")
            continue

        terrain_seed = data.get("config", {}).get("terrain_seed", "?")
        for method in ["IPC", "RL"]:
            trials = data.get("per_trial", {}).get(method, [])
            for t in trials:
                t["batch_idx"] = batch_idx
                t["terrain_seed"] = terrain_seed
                # Update data_file to match merged naming
                if "data_file" in t:
                    t["data_file"] = f"b{batch_idx:03d}_{t['data_file']}"
            merged_trials[method].extend(trials)

        # Copy flight data files to merged directory
        if data_dir and os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                src = os.path.join(data_dir, fname)
                # Prefix with batch index to avoid name collisions
                dst_name = f"b{batch_idx:03d}_{fname}"
                dst = os.path.join(merged_data_dir, dst_name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        n_ipc = len(data.get("per_trial", {}).get("IPC", []))
        n_rl = len(data.get("per_trial", {}).get("RL", []))
        batch_summaries.append({
            "batch_idx": batch_idx,
            "terrain_seed": terrain_seed,
            "status": status,
            "ipc_trials": n_ipc,
            "rl_trials": n_rl,
        })
        print(f"  Batch {batch_idx}: {n_ipc} IPC + {n_rl} RL trials [{status}]")

    total_ipc = len(merged_trials["IPC"])
    total_rl = len(merged_trials["RL"])
    print(f"\n  Total: {total_ipc} IPC trials, {total_rl} RL trials "
          f"from {len(batch_results)} batches in {total_elapsed:.0f}s")

    if total_ipc == 0 and total_rl == 0:
        print("  No valid trials found. Exiting.")
        sys.exit(1)

    # --- Aggregate ---
    agg = aggregate_all(merged_trials)

    # --- Print table ---
    total_trials = max(total_ipc, total_rl)
    print_results_table(agg, total_trials, len(batch_results))

    # --- Save merged output ---
    merged_output = {
        "batch_config": batch_config,
        "batch_summaries": batch_summaries,
        "total_batches_completed": len(batch_results),
        "total_ipc_trials": total_ipc,
        "total_rl_trials": total_rl,
        "total_time_s": total_elapsed,
        "data_dir": merged_data_dir,
        "per_trial": merged_trials,
        "aggregated": agg,
    }
    merged_path = os.path.join(output_dir, "batch_results.json")
    with open(merged_path, 'w') as f:
        json.dump(merged_output, f, indent=2, default=str)

    print(f"\nMerged results: {merged_path}")
    print(f"Flight data:    {merged_data_dir}")
    print(f"\nRender any trial:  python render_viz.py {merged_path} --trial 0")


if __name__ == "__main__":
    main()
