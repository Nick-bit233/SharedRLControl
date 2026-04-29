#!/usr/bin/env python3
"""
Offline trajectory visualization renderer.

Renders 2D trajectory animations/plots from saved flight data produced by
``compare_ipc_rl.py``.  Does NOT require Isaac Sim — only matplotlib + numpy.

Usage examples::

    # Render trial 0 as side-by-side IPC vs RL animation
    python render_viz.py results.json --trial 0

    # Render trials 0,3,5 as static plots
    python render_viz.py results.json --trial 0 3 5 --static

    # Render all trials
    python render_viz.py results.json --all

    # Custom output directory and fps
    python render_viz.py results.json --trial 0 --fps 15 --output ./my_viz/

    # List all trials with their results
    python render_viz.py results.json --list
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trajectory_visualizer import (
    FlightDataRecorder,
    TrajectoryVisualizer,
    obstacles_to_info,
)


def load_experiment(results_path: str) -> dict:
    """Load experiment results JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def list_trials(results: dict):
    """Print a summary table of all trials."""
    completed = results.get("config", {}).get("completed_trials", "?")
    planned = results['config']['num_trials']
    status = "COMPLETE" if completed == planned * 2 else f"PARTIAL ({completed}/{planned*2})"

    print(f"\n{'='*90}")
    print(f"Experiment: {planned} trials × "
          f"{results['config']['num_frames']} frames  [{status}]")
    print(f"Data dir:   {results.get('data_dir', 'N/A')}")
    print(f"{'='*90}")

    for method in ["IPC", "RL"]:
        trials = results["per_trial"].get(method, [])
        if not trials:
            continue
        print(f"\n  {method} Trials ({len(trials)} completed):")
        print(f"  {'ID':>4}  {'Seed':>6}  {'OK':>4}  {'MaxX':>7}  "
              f"{'TCR@1':>7}  {'TCR@5':>7}  {'FPS':>7}  {'Lat_ms':>7}  {'Data File'}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*4}  {'-'*7}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
        for t in trials:
            tid = t.get("trial_id", "?")
            seed = t.get("trial_seed", "?")
            succ = "✓" if t.get("success") else "✗"
            mx = t.get("max_x_reached", 0)
            tcr1 = t.get("tcr_at_1", 0)
            tcr5 = t.get("tcr_at_5", 0)
            fps = t.get("ctrl_fps", 0)
            lat = t.get("inference_mean_ms", 0)
            df = t.get("data_file", "N/A")
            crash = t.get("crash_reason", "")
            extra = f"  ({crash})" if crash and not t.get("success") else ""
            print(f"  {tid:>4}  {seed:>6}  {succ:>4}  {mx:>7.2f}  "
                  f"{tcr1:>7.3f}  {tcr5:>7.3f}  {fps:>7.0f}  {lat:>7.2f}  {df}{extra}")

    # Compute aggregated on the fly (handles partial results)
    print(f"\n  Aggregated:")
    for method in ["IPC", "RL"]:
        trials = results["per_trial"].get(method, [])
        if not trials:
            print(f"    {method}: no data")
            continue
        import numpy as _np
        sr = sum(1 for t in trials if t.get("success")) / len(trials)
        tcr1 = _np.mean([t.get("tcr_at_1", 0) for t in trials])
        tcr5 = _np.mean([t.get("tcr_at_5", 0) for t in trials])
        fps = _np.mean([t.get("ctrl_fps", 0) for t in trials])
        lat = _np.mean([t.get("inference_mean_ms", 0) for t in trials])
        print(f"    {method}: success={sr:.1%}, TCR@1={tcr1:.3f}, "
              f"TCR@5={tcr5:.3f}, ctrl_fps={fps:.0f}, latency={lat:.2f}ms")
    print()


def resolve_data_dir(results: dict, results_path: str) -> str:
    """Resolve the data directory from results JSON."""
    data_dir = results.get("data_dir", "")
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    # Try relative to results file
    base = os.path.dirname(os.path.abspath(results_path))
    rel = os.path.join(base, os.path.basename(data_dir))
    if os.path.isdir(rel):
        return rel
    raise FileNotFoundError(
        f"Flight data directory not found: {data_dir}\n"
        f"Also tried: {rel}"
    )


def load_obstacles(data_dir: str, batch_idx: int | None = None) -> list:
    """Load obstacle metadata from data directory.

    For merged batch results, pass batch_idx to load the batch-specific
    obstacles file (e.g., b000_obstacles.json).
    """
    if batch_idx is not None:
        meta_path = os.path.join(data_dir, f"b{batch_idx:03d}_obstacles.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(data_dir, "obstacles.json")
    else:
        meta_path = os.path.join(data_dir, "obstacles.json")
    if not os.path.exists(meta_path):
        return [], (24.0, 12.0)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    obstacles = meta.get("obstacles", [])
    terrain_size = tuple(meta.get("terrain_size", [24.0, 12.0]))
    return obstacles, terrain_size


def get_trial_data_files(results: dict, trial_idx: int, data_dir: str):
    """Get IPC and RL data file paths for a given trial index.

    For single-batch results, matches by trial_id field.
    For merged batch results (where trial_id repeats), uses list position.
    """
    ipc_file = None
    rl_file = None

    for method, key in [("IPC", "ipc"), ("RL", "rl")]:
        trials = results["per_trial"].get(method, [])
        # Try matching by trial_id first
        matched = None
        for t in trials:
            if t.get("trial_id") == trial_idx:
                matched = t
                break
        # Fallback: use list index (works for merged batch results)
        if matched is None and 0 <= trial_idx < len(trials):
            matched = trials[trial_idx]

        if matched:
            fname = matched.get("data_file")
            if fname:
                path = os.path.join(data_dir, fname)
                if os.path.exists(path):
                    if method == "IPC":
                        ipc_file = path
                    else:
                        rl_file = path

        # Fallback: try standard naming
        if method == "IPC" and ipc_file is None:
            fallback = os.path.join(data_dir, f"ipc_trial{trial_idx:04d}.npz")
            if os.path.exists(fallback):
                ipc_file = fallback
        if method == "RL" and rl_file is None:
            fallback = os.path.join(data_dir, f"rl_trial{trial_idx:04d}.npz")
            if os.path.exists(fallback):
                rl_file = fallback

    return ipc_file, rl_file


def render_trial(
    results: dict,
    results_path: str,
    trial_idx: int,
    output_dir: str,
    fps: int = 20,
    static: bool = False,
    subsample: int = 3,
):
    """Render a single trial comparison."""
    data_dir = resolve_data_dir(results, results_path)

    # For merged batch results, find batch_idx from trial metadata
    batch_idx = None
    for method in ["IPC", "RL"]:
        trials = results["per_trial"].get(method, [])
        if 0 <= trial_idx < len(trials):
            batch_idx = trials[trial_idx].get("batch_idx")
            break

    obstacles, terrain_size = load_obstacles(data_dir, batch_idx=batch_idx)
    obs_info = obstacles_to_info(obstacles) if obstacles else []

    tx_half = terrain_size[0] / 2.0
    ty_half = terrain_size[1] / 2.0
    wall_inner_y = ty_half - 1.0

    viz = TrajectoryVisualizer(
        obstacles=obs_info,
        tunnel_x_range=(-tx_half - 1, tx_half + 1),
        tunnel_y_range=(-ty_half - 0.5, ty_half + 0.5),
        z_range=(0.0, 8.0),
        wall_y=(-wall_inner_y, wall_inner_y),
    )

    ipc_file, rl_file = get_trial_data_files(results, trial_idx, data_dir)

    if ipc_file is None or rl_file is None:
        missing = []
        if ipc_file is None:
            missing.append("IPC")
        if rl_file is None:
            missing.append("RL")
        print(f"ERROR: Missing data files for trial {trial_idx}: {', '.join(missing)}")
        print(f"  Searched in: {data_dir}")
        return False

    # Get trial seed for label
    trial_seed = 42 + trial_idx
    for t in results["per_trial"].get("IPC", []):
        if t.get("trial_id") == trial_idx:
            trial_seed = t.get("trial_seed", trial_seed)
            break
    label = f"Trial {trial_idx} (seed={trial_seed})"

    os.makedirs(output_dir, exist_ok=True)

    if static:
        out_path = os.path.join(output_dir, f"trajectory_trial{trial_idx}.png")
        viz.render_static_comparison(ipc_file, rl_file, out_path, trial_label=label)
        print(f"  Static plot: {out_path}")
    else:
        out_path = os.path.join(output_dir, f"compare_trial{trial_idx}.mp4")
        viz.render_comparison(
            ipc_file, rl_file, out_path,
            fps=fps, subsample=subsample, trial_label=label,
        )
        print(f"  Animation:   {out_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Offline trajectory visualization for IPC vs RL comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("results_json", type=str,
                        help="Path to compare_results_*.json from compare_ipc_rl.py")
    parser.add_argument("--trial", type=int, nargs="+", default=None,
                        help="Trial index(es) to render (e.g., --trial 0 3 5)")
    parser.add_argument("--all", action="store_true",
                        help="Render all trials")
    parser.add_argument("--list", action="store_true",
                        help="List all trials and their results, then exit")
    parser.add_argument("--static", action="store_true",
                        help="Render static .png instead of animated .mp4")
    parser.add_argument("--fps", type=int, default=20,
                        help="Animation frame rate (default: 20)")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Frame subsample factor (default: 3, i.e. 60fps→20fps)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <data_dir>/viz/)")

    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"ERROR: Results file not found: {args.results_json}")
        sys.exit(1)

    results = load_experiment(args.results_json)

    if args.list:
        list_trials(results)
        return

    if args.trial is None and not args.all:
        print("ERROR: Specify --trial <N> or --all (use --list to see available trials)")
        sys.exit(1)

    # Determine which trials to render
    # Support both single-batch ("config") and merged-batch ("batch_config") formats
    cfg = results.get("config") or results.get("batch_config", {})
    ipc_trials = results.get("per_trial", {}).get("IPC", [])
    rl_trials = results.get("per_trial", {}).get("RL", [])
    num_trials = max(len(ipc_trials), len(rl_trials), cfg.get("num_trials", 0))

    if args.all:
        trial_indices = list(range(num_trials))
    else:
        trial_indices = args.trial
        for t in trial_indices:
            if t < 0 or t >= num_trials:
                print(f"WARNING: Trial {t} out of range [0, {num_trials-1}], skipping")
        trial_indices = [t for t in trial_indices if 0 <= t < num_trials]

    if not trial_indices:
        print("No valid trials to render.")
        return

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        try:
            data_dir = resolve_data_dir(results, args.results_json)
            output_dir = os.path.join(data_dir, "viz")
        except FileNotFoundError:
            output_dir = os.path.join(os.path.dirname(args.results_json), "viz")

    fmt = "static .png" if args.static else f"animated .mp4 ({args.fps}fps)"
    print(f"Rendering {len(trial_indices)} trial(s) as {fmt}")
    print(f"Output: {output_dir}\n")

    success = 0
    for tidx in trial_indices:
        print(f"Trial {tidx}:")
        try:
            ok = render_trial(
                results, args.results_json, tidx, output_dir,
                fps=args.fps, static=args.static, subsample=args.subsample,
            )
            if ok:
                success += 1
        except Exception as e:
            print(f"ERROR: Failed to render trial {tidx}: {e}")

    print(f"\nDone: {success}/{len(trial_indices)} trials rendered to {output_dir}")


if __name__ == "__main__":
    main()
