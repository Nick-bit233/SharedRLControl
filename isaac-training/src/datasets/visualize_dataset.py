"""
Inspect & visualise an offline pilot trajectory dataset (HDF5 produced by
`src/datasets/trajectory_generator.py`).

Three things we want to validate (per the M2 review request):

  1. Diversity        - trajectories should look meaningfully different from
                        each other (start/end positions, lateral styles, etc.).
  2. Human-likeness   - per-step velocity changes should be smooth (no jagged
                        per-step jumps) and within plausible operator range.
  3. Reachability     - given the tunnel start (Isaac x=-7) and goal (x>=10)
                        and `max_episode_length` (default 1500 steps,
                        dt=0.016 s), it must be theoretically possible to
                        reach the goal in an obstacle-free world. We measure
                        this by integrating raw vx within a 1500-step rolling
                        window and reporting the fraction of windows whose
                        net forward travel >= 17 m.

Usage (env activated, run from `isaac-training/`):

    python experiments/04_tunnel_task/visualize_dataset.py \
        --dataset data/trajectories_tunnel.h5 \
        --num 32 \
        --out-dir figures/dataset_inspection

Outputs a PNG grid + a JSON stats report in the chosen out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Tunnel task constants (Isaac frame).
TUNNEL_START_X = -7.0
TUNNEL_GOAL_X = 10.0
TUNNEL_FORWARD_DIST = TUNNEL_GOAL_X - TUNNEL_START_X  # 17 m
DEFAULT_MAX_EPISODE_LEN = 1500


def load_dataset(path: str, num: int, seed: int = 0):
    """Load metadata, sample `num` trajectories from `path`."""
    with h5py.File(path, "r") as f:
        meta_attrs = dict(f["metadata"].attrs)
        N = int(meta_attrs["num_trajectories"])
        T = int(meta_attrs["trajectory_length"])
        D = int(meta_attrs["action_dim"])
        dt = float(meta_attrs["dt"])
        ref_map = list(map(float, meta_attrs["reference_map_bounds"]))

        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, size=min(num, N), replace=False))

        velocities = f["velocities"][idx]   # (num, T, D)
        positions = f["positions"][idx]     # (num, T, 3)

    return {
        "indices": idx,
        "velocities": velocities,
        "positions": positions,
        "dt": dt,
        "T": T,
        "D": D,
        "N": N,
        "reference_map_bounds": ref_map,
    }


def compute_dataset_stats(path: str, max_chunk: int = 4096) -> dict:
    """Streaming stats over the full dataset (mean / std of velocities, plus
    rolling-window forward-travel feasibility)."""
    with h5py.File(path, "r") as f:
        N = int(f["metadata"].attrs["num_trajectories"])
        T = int(f["metadata"].attrs["trajectory_length"])
        D = int(f["metadata"].attrs["action_dim"])
        dt = float(f["metadata"].attrs["dt"])
        vel_ds = f["velocities"]
        pos_ds = f["positions"]

        sums = np.zeros(D, dtype=np.float64)
        sqs = np.zeros(D, dtype=np.float64)
        count = 0

        # Rolling window stats.
        win = min(DEFAULT_MAX_EPISODE_LEN, T)
        # net forward travel in the most-favourable window per trajectory
        max_window_dx = []
        # mean_vx over each trajectory's full length
        traj_mean_vx = []
        # per-step |Δv| (smoothness) on a sub-sample
        delta_v_samples = []

        for start in range(0, N, max_chunk):
            end = min(N, start + max_chunk)
            v = vel_ds[start:end]                      # (b, T, D)
            sums += v.reshape(-1, D).sum(axis=0)
            sqs += (v.reshape(-1, D) ** 2).sum(axis=0)
            count += v.shape[0] * v.shape[1]

            # mean_vx per traj
            traj_mean_vx.append(v[:, :, 0].mean(axis=1))

            # rolling window net forward displacement: integrate vx, take
            # max over starting offset of (cumsum[i+win] - cumsum[i]) * dt.
            cs = np.cumsum(v[:, :, 0], axis=1) * dt   # (b, T)
            if T > win:
                window_dx = cs[:, win - 1:] - np.concatenate(
                    [np.zeros((cs.shape[0], 1)), cs[:, :T - win]], axis=1
                )
            else:
                window_dx = cs[:, -1:]
            max_window_dx.append(window_dx.max(axis=1))

            # smoothness sample (first 10 trajs of each chunk)
            sub = v[: min(10, v.shape[0])]
            delta_v_samples.append(np.diff(sub, axis=1).reshape(-1, D))

        mean = sums / count
        var = sqs / count - mean ** 2
        std = np.sqrt(np.maximum(var, 0))

        traj_mean_vx = np.concatenate(traj_mean_vx)
        max_window_dx = np.concatenate(max_window_dx)
        delta_v = np.concatenate(delta_v_samples, axis=0)
        delta_v_mag_per_step = np.linalg.norm(delta_v, axis=1)

        # Reachability: fraction of trajectories whose best 1500-step window
        # forward travel >= 17 m.
        reach_frac = float((max_window_dx >= TUNNEL_FORWARD_DIST).mean())

    return {
        "num_trajectories": N,
        "trajectory_length": T,
        "action_dim": D,
        "dt": dt,
        "rolling_window_steps": win,
        "vel_mean_xyz": mean.tolist(),
        "vel_std_xyz": std.tolist(),
        "traj_mean_vx_avg": float(traj_mean_vx.mean()),
        "traj_mean_vx_std": float(traj_mean_vx.std()),
        "best_window_dx_median": float(np.median(max_window_dx)),
        "best_window_dx_p10": float(np.percentile(max_window_dx, 10)),
        "best_window_dx_p90": float(np.percentile(max_window_dx, 90)),
        "tunnel_forward_dist_m": TUNNEL_FORWARD_DIST,
        "reachable_fraction_at_1500_steps": reach_frac,
        "delta_v_per_step_mean": float(delta_v_mag_per_step.mean()),
        "delta_v_per_step_p99": float(np.percentile(delta_v_mag_per_step, 99)),
    }


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

def plot_xy_trajectories(positions, ref_map, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(positions.shape[0]):
        ax.plot(positions[i, :, 0], positions[i, :, 1], lw=0.6, alpha=0.6)
    # tunnel goal line and start point
    ax.axvline(TUNNEL_START_X, color="tab:green", lw=1.0, ls="--", label="tunnel start (x=-7)")
    ax.axvline(TUNNEL_GOAL_X, color="tab:red", lw=1.0, ls="--", label="tunnel goal (x=10)")
    rx, ry = ref_map[0], ref_map[1]
    ax.set_xlim(-rx, rx)
    ax.set_ylim(-ry, ry)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (forward)")
    ax.set_ylabel("y [m] (lateral)")
    ax.set_title(f"XY trajectories (n={positions.shape[0]})  - diversity check")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_velocity_timeseries(velocities, dt, out_path, n_show=8):
    n_show = min(n_show, velocities.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = np.arange(velocities.shape[1]) * dt
    labels = ["vx", "vy", "vz"]
    for ax, ax_idx, label in zip(axs, range(3), labels):
        for i in range(n_show):
            ax.plot(t, velocities[i, :, ax_idx], lw=0.6, alpha=0.7)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylabel(f"{label} [m/s]")
        ax.grid(True, alpha=0.3)
    axs[0].set_title(f"Velocity time series (first {n_show} trajs)  - smoothness / human-likeness")
    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_velocity_histograms(velocities, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = ["vx", "vy", "vz"]
    flat = velocities.reshape(-1, velocities.shape[-1])
    for ax, ax_idx, label in zip(axs, range(3), labels):
        ax.hist(flat[:, ax_idx], bins=80, density=True, alpha=0.8)
        ax.axvline(flat[:, ax_idx].mean(), color="r", ls="--", lw=1.0,
                   label=f"mean={flat[:, ax_idx].mean():.2f}")
        ax.set_xlabel(f"{label} [m/s]")
        ax.set_title(f"{label} distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-channel velocity distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_forward_progress(velocities, dt, out_path):
    """Cumulative forward (x) travel vs time - reachability check.
    Mark the 17 m goal threshold and 1500-step (24 s) episode horizon."""
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(velocities.shape[1]) * dt
    cumdx = np.cumsum(velocities[:, :, 0], axis=1) * dt
    for i in range(cumdx.shape[0]):
        ax.plot(t, cumdx[i], lw=0.6, alpha=0.6)
    ax.axhline(TUNNEL_FORWARD_DIST, color="tab:red", ls="--", lw=1.0,
               label=f"goal forward distance = {TUNNEL_FORWARD_DIST:.0f} m")
    ax.axvline(DEFAULT_MAX_EPISODE_LEN * dt, color="tab:purple", ls="--", lw=1.0,
               label=f"episode horizon = {DEFAULT_MAX_EPISODE_LEN * dt:.1f} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("cumulative Δx [m]")
    ax.set_title(f"Forward progress (n={cumdx.shape[0]})  - reachability check")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_xz_trajectories(positions, ref_map, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(positions.shape[0]):
        ax.plot(positions[i, :, 0], positions[i, :, 2], lw=0.6, alpha=0.6)
    ax.axvline(TUNNEL_START_X, color="tab:green", lw=1.0, ls="--")
    ax.axvline(TUNNEL_GOAL_X, color="tab:red", lw=1.0, ls="--")
    ax.set_ylim(0.5, 2 * ref_map[2] + 0.5)
    ax.set_xlim(-ref_map[0], ref_map[0])
    ax.set_xlabel("x [m] (forward)")
    ax.set_ylabel("z [m] (height)")
    ax.set_title("XZ trajectories  - vertical behaviour check")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="Path to .h5 produced by trajectory_generator.py")
    parser.add_argument("--num", type=int, default=32,
                        help="Number of trajectories to plot (default: 32)")
    parser.add_argument("--out-dir", default="figures/dataset_inspection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip full-dataset streaming stats (faster).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading {args.num} trajectories from {args.dataset} ...")
    sample = load_dataset(args.dataset, args.num, args.seed)

    print(f"[viz] sample: V {sample['velocities'].shape}  P {sample['positions'].shape}  "
          f"dt={sample['dt']}  ref_map={sample['reference_map_bounds']}")

    # Plots
    plot_xy_trajectories(sample["positions"], sample["reference_map_bounds"],
                         out_dir / "01_xy_trajectories.png")
    plot_xz_trajectories(sample["positions"], sample["reference_map_bounds"],
                         out_dir / "02_xz_trajectories.png")
    plot_velocity_timeseries(sample["velocities"], sample["dt"],
                             out_dir / "03_velocity_timeseries.png")
    plot_velocity_histograms(sample["velocities"],
                             out_dir / "04_velocity_histograms.png")
    plot_forward_progress(sample["velocities"], sample["dt"],
                          out_dir / "05_forward_progress.png")
    print(f"[viz] wrote 5 figures to {out_dir}/")

    if not args.no_stats:
        print("[viz] computing full-dataset streaming stats (this may take a moment) ...")
        stats = compute_dataset_stats(args.dataset)
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("[viz] === DATASET STATS ===")
        print(json.dumps(stats, indent=2))
        print(f"[viz] wrote stats to {out_dir / 'stats.json'}")

        # Concise pass/fail summary against tunnel feasibility expectations.
        print("\n[viz] === FEASIBILITY VERDICT ===")
        mean_vx = stats["vel_mean_xyz"][0]
        target_vx_low, target_vx_high = 1.0, 2.0
        ok_vx = target_vx_low <= mean_vx <= target_vx_high
        ok_reach = stats["reachable_fraction_at_1500_steps"] >= 0.9
        print(f"  vx mean        = {mean_vx:+.3f} m/s     "
              f"(expect {target_vx_low}..{target_vx_high})   "
              f"[{'OK' if ok_vx else 'FAIL'}]")
        print(f"  reachable @1500 = {stats['reachable_fraction_at_1500_steps']*100:5.1f}%      "
              f"(expect >=90%)              [{'OK' if ok_reach else 'FAIL'}]")
        if not (ok_vx and ok_reach):
            print("  -> dataset does NOT meet tunnel feasibility criteria; "
                  "regenerate after fixing trajectory_gen_tunnel.yaml / generator.")
        else:
            print("  -> dataset meets tunnel feasibility criteria.")


if __name__ == "__main__":
    main()
