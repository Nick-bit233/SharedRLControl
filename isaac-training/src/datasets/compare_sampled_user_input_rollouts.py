"""Visualize training-style sampled rollouts from offline user-input datasets.

This script differs from compare_user_input_datasets.py: it does not plot raw
reference traces stored in HDF5. Instead, it repeatedly samples random windows,
applies the same scale-to-fit idea used during training, integrates from a fixed
start position, and clips to explicit map bounds. The output is intended to make
v1/v2 tunnel command-prior differences visible in the region actually used by
training.

Example:
    python src/datasets/compare_sampled_user_input_rollouts.py \
      --dataset v1=data/user_inputs/tunnel_perlin_v1.h5 \
      --dataset v2=data/user_inputs/tunnel_perlin_bounded_v2.h5 \
      --bounds -8 10 -5 5 2.5 7.5 \
      --start -7 0 5 \
      --num-rollouts 64 \
      --horizon 1500 \
      --window-size 128 \
      --out-dir outputs/user_input_compare/tunnel_v1_v2_sampled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXIS_NAMES = ("x", "y", "z")


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--dataset must use label=path format, for example v1=data/tunnel.h5"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("dataset label must not be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help="Dataset in label=path format. Can be repeated.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=[-8.0, 10.0, -5.0, 5.0, 2.5, 7.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Explicit rollout bounds.",
    )
    parser.add_argument(
        "--start",
        nargs=3,
        type=float,
        default=[-7.0, 0.0, 5.0],
        metavar=("X", "Y", "Z"),
        help="Fixed rollout start position.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and summaries.")
    parser.add_argument("--num-rollouts", type=int, default=64, help="Sampled rollouts per dataset.")
    parser.add_argument("--horizon", type=int, default=1500, help="Maximum integration steps per rollout.")
    parser.add_argument("--window-size", type=int, default=128, help="Random HDF5 window size per sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--min-scale-factor", type=float, default=0.5, help="Lower clamp for scale-to-fit.")
    parser.add_argument("--max-scale-factor", type=float, default=2.0, help="Upper clamp for scale-to-fit.")
    parser.add_argument(
        "--no-stop-at-goal",
        action="store_true",
        help="Continue integration after x reaches the upper x bound.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def bounds_to_arrays(bounds: list[float]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    upper = np.asarray([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    if np.any(upper <= lower):
        raise ValueError(f"invalid bounds: lower={lower.tolist()} upper={upper.tolist()}")
    return lower, upper


def fit_scale_to_bounds(
    ref_positions: np.ndarray,
    current_pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_scale: float,
    max_scale: float,
) -> float:
    relative = ref_positions[:, :3] - ref_positions[0:1, :3]
    bbox_min = relative.min(axis=0)
    bbox_max = relative.max(axis=0)
    room_pos = upper - current_pos
    room_neg = current_pos - lower
    eps = 1e-2

    scale_fwd = np.full(3, np.inf, dtype=np.float32)
    scale_bwd = np.full(3, np.inf, dtype=np.float32)
    mask_fwd = bbox_max > eps
    mask_bwd = -bbox_min > eps
    scale_fwd[mask_fwd] = room_pos[mask_fwd] / bbox_max[mask_fwd]
    scale_bwd[mask_bwd] = room_neg[mask_bwd] / (-bbox_min[mask_bwd])
    scale = float(np.min(np.minimum(scale_fwd, scale_bwd)))
    if not np.isfinite(scale):
        scale = max_scale
    return float(np.clip(scale, min_scale, max_scale))


def load_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return {k: _jsonable(v) for k, v in f["metadata"].attrs.items()}


def sample_dataset_rollouts(
    label: str,
    path: Path,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    num_rollouts: int,
    horizon: int,
    window_size: int,
    min_scale: float,
    max_scale: float,
    stop_at_goal: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    traces: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    scale_values: list[np.ndarray] = []
    reached_steps: list[int | None] = []

    with h5py.File(path, "r") as f:
        vel_ds = f["velocities"]
        pos_ds = f["positions"]
        meta = {k: _jsonable(v) for k, v in f["metadata"].attrs.items()}
        num_trajs = int(meta["num_trajectories"])
        traj_len = int(meta["trajectory_length"])
        action_dim = int(meta["action_dim"])
        dt = float(meta["dt"])

        if window_size > traj_len:
            raise ValueError(f"window_size={window_size} exceeds trajectory_length={traj_len} for {path}")
        if np.any(start < lower) or np.any(start > upper):
            raise ValueError(f"start={start.tolist()} is outside bounds for {label}")

        for _ in range(num_rollouts):
            pos = start.astype(np.float32).copy()
            trace = [pos.copy()]
            applied_vels = []
            scales = []
            reached: int | None = None
            step = 0

            while step < horizon:
                chunk = min(window_size, horizon - step)
                traj_idx = int(rng.integers(0, num_trajs))
                offset = int(rng.integers(0, traj_len - chunk + 1))
                ref_pos = pos_ds[traj_idx, offset : offset + chunk, :3]
                ref_vel = vel_ds[traj_idx, offset : offset + chunk, : min(3, action_dim)]
                scale = fit_scale_to_bounds(ref_pos, pos, lower, upper, min_scale, max_scale)
                vel_xyz = ref_vel.astype(np.float32) * scale

                for local_step in range(chunk):
                    next_pos = np.clip(pos + vel_xyz[local_step] * dt, lower, upper)
                    effective_vel = (next_pos - pos) / dt
                    pos = next_pos.astype(np.float32)
                    trace.append(pos.copy())
                    applied_vels.append(effective_vel.astype(np.float32))
                    scales.append(scale)
                    step += 1
                    if stop_at_goal and pos[0] >= upper[0] - 1e-5:
                        reached = step
                        break

                if reached is not None:
                    break

            traces.append(np.asarray(trace, dtype=np.float32))
            if applied_vels:
                velocities.append(np.asarray(applied_vels, dtype=np.float32))
                scale_values.append(np.asarray(scales, dtype=np.float32))
            else:
                velocities.append(np.zeros((0, 3), dtype=np.float32))
                scale_values.append(np.zeros((0,), dtype=np.float32))
            reached_steps.append(reached)

    return {
        "label": label,
        "path": str(path),
        "metadata": meta,
        "dt": dt,
        "traces": traces,
        "velocities": velocities,
        "scales": scale_values,
        "reached_steps": reached_steps,
    }


def concatenate_nonempty(arrays: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
    nonempty = [arr for arr in arrays if arr.size > 0]
    if not nonempty:
        return np.zeros((0, *shape_tail), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)


def compute_summary(result: dict[str, Any]) -> dict[str, Any]:
    traces = result["traces"]
    velocities = concatenate_nonempty(result["velocities"], (3,))
    scales = concatenate_nonempty([s[:, None] for s in result["scales"]], (1,)).reshape(-1)
    reached = np.asarray([s if s is not None else -1 for s in result["reached_steps"]], dtype=np.int64)
    final_positions = np.asarray([trace[-1] for trace in traces], dtype=np.float32)
    all_positions = np.concatenate(traces, axis=0)
    speed = np.linalg.norm(velocities, axis=1) if len(velocities) else np.zeros((0,), dtype=np.float32)

    reached_mask = reached >= 0
    summary: dict[str, Any] = {
        "path": result["path"],
        "metadata": result["metadata"],
        "num_rollouts": len(traces),
        "reached_fraction": float(reached_mask.mean()) if len(reached_mask) else 0.0,
        "mean_reach_step": float(reached[reached_mask].mean()) if reached_mask.any() else None,
        "final_position_mean": final_positions.mean(axis=0).tolist(),
        "final_position_std": final_positions.std(axis=0).tolist(),
        "x_p05_p95": np.percentile(all_positions[:, 0], [5, 95]).tolist(),
        "y_abs_p95": float(np.percentile(np.abs(all_positions[:, 1]), 95)),
        "z_p05_p95": np.percentile(all_positions[:, 2], [5, 95]).tolist(),
        "scale_mean": float(scales.mean()) if scales.size else None,
        "scale_p05_p95": np.percentile(scales, [5, 95]).tolist() if scales.size else None,
        "speed_mean": float(speed.mean()) if speed.size else None,
        "speed_p95": float(np.percentile(speed, 95)) if speed.size else None,
    }
    return summary


def plot_rollouts(
    results: dict[str, dict[str, Any]],
    out_path: Path,
    axes: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    title: str,
) -> None:
    cols = min(3, len(results))
    rows = int(np.ceil(len(results) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.8 * rows), squeeze=False)
    x_axis, y_axis = axes

    for ax, (label, result) in zip(axs.reshape(-1), results.items()):
        for trace in result["traces"]:
            ax.plot(trace[:, x_axis], trace[:, y_axis], lw=0.8, alpha=0.38)
        ax.scatter([start[x_axis]], [start[y_axis]], s=28, c="tab:green", zorder=4, label="start")
        ax.axvline(lower[x_axis], color="0.45", ls=":", lw=1.0)
        ax.axvline(upper[x_axis], color="tab:red", ls="--", lw=1.1)
        ax.axhline(lower[y_axis], color="0.45", ls=":", lw=1.0)
        ax.axhline(upper[y_axis], color="0.45", ls=":", lw=1.0)
        ax.set_xlim(float(lower[x_axis]), float(upper[x_axis]))
        ax.set_ylim(float(lower[y_axis]), float(upper[y_axis]))
        ax.set_xlabel(AXIS_NAMES[x_axis])
        ax.set_ylabel(AXIS_NAMES[y_axis])
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axs.reshape(-1)[len(results) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_velocity_histograms(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis_idx, axis_label in enumerate(("vx", "vy", "vz")):
        ax = axs[axis_idx]
        for label, result in results.items():
            vel = concatenate_nonempty(result["velocities"], (3,))
            if len(vel):
                ax.hist(vel[:, axis_idx], bins=70, density=True, histtype="step", lw=1.5, label=label)
        ax.set_title(axis_label)
        ax.set_xlabel("m/s")
        ax.grid(True, alpha=0.25)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scale_distribution(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, result in results.items():
        scales = concatenate_nonempty([s[:, None] for s in result["scales"]], (1,)).reshape(-1)
        if scales.size:
            ax.hist(scales, bins=60, density=True, histtype="step", lw=1.5, label=label)
    ax.set_xlabel("sample scale factor")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_markdown(stats: dict[str, Any], out_path: Path) -> None:
    lines = ["# Sampled User Input Rollout Comparison", ""]
    for label, item in stats.items():
        meta = item["metadata"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- path: `{item['path']}`",
                f"- generator_kind: `{meta.get('generator_kind', 'unknown')}`",
                f"- reached_fraction: {item['reached_fraction']:.4f}",
                f"- mean_reach_step: {item['mean_reach_step']}",
                f"- final_position_mean: {[round(v, 4) for v in item['final_position_mean']]}",
                f"- y_abs_p95: {item['y_abs_p95']:.4f}",
                f"- z_p05_p95: {[round(v, 4) for v in item['z_p05_p95']]}",
                f"- scale_mean: {item['scale_mean']}",
                f"- scale_p05_p95: {item['scale_p05_p95']}",
                f"- speed_mean/p95: {item['speed_mean']} / {item['speed_p95']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    lower, upper = bounds_to_arrays(args.bounds)
    start = np.asarray(args.start, dtype=np.float32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for idx, (label, path) in enumerate(args.dataset):
        rng = np.random.default_rng(args.seed + idx * 1009)
        results[label] = sample_dataset_rollouts(
            label=label,
            path=path,
            lower=lower,
            upper=upper,
            start=start,
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            window_size=args.window_size,
            min_scale=args.min_scale_factor,
            max_scale=args.max_scale_factor,
            stop_at_goal=not args.no_stop_at_goal,
            rng=rng,
        )

    stats = {label: compute_summary(result) for label, result in results.items()}
    stats["_rollout_settings"] = {
        "bounds": args.bounds,
        "start": args.start,
        "num_rollouts": args.num_rollouts,
        "horizon": args.horizon,
        "window_size": args.window_size,
        "min_scale_factor": args.min_scale_factor,
        "max_scale_factor": args.max_scale_factor,
        "stop_at_goal": not args.no_stop_at_goal,
    }

    plot_rollouts(results, out_dir / "01_sampled_xy_rollouts.png", (0, 1), lower, upper, start, "Sampled XY rollouts")
    plot_rollouts(results, out_dir / "02_sampled_xz_rollouts.png", (0, 2), lower, upper, start, "Sampled XZ rollouts")
    plot_velocity_histograms(results, out_dir / "03_sampled_velocity_histograms.png")
    plot_scale_distribution(results, out_dir / "04_sample_scale_distribution.png")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    write_summary_markdown({k: v for k, v in stats.items() if not k.startswith("_")}, out_dir / "summary.md")

    print(f"[compare-sampled] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
