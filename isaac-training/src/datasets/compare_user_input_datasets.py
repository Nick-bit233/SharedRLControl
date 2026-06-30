"""Compare offline user-input HDF5 datasets.

Run from the isaac-training repository root:

    python src/datasets/compare_user_input_datasets.py \
      --dataset legacy=data/user_inputs/legacy_perlin_v1.h5 \
      --dataset tunnel=data/user_inputs/tunnel_perlin_v1.h5 \
      --dataset intent=data/user_inputs/intent_pilot_v1.h5 \
      --out-dir outputs/user_input_compare/tunnel_inputs_v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TUNNEL_START_X = -7.0
TUNNEL_GOAL_X = 10.0
TUNNEL_FORWARD_DIST = TUNNEL_GOAL_X - TUNNEL_START_X
DEFAULT_MAX_EPISODE_LEN = 1500


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--dataset must use label=path format, for example tunnel=data/tunnel.h5"
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
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and summaries.")
    parser.add_argument("--num", type=int, default=64, help="Sample trajectories per dataset for plots.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--max-chunk", type=int, default=1024, help="Streaming stats chunk size.")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_sample(path: Path, num: int, seed: int) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        meta = {k: _jsonable(v) for k, v in f["metadata"].attrs.items()}
        n = int(meta["num_trajectories"])
        t = int(meta["trajectory_length"])
        d = int(meta["action_dim"])
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(num, n), replace=False))
        sample = {
            "path": str(path),
            "metadata": meta,
            "indices": idx,
            "velocities": f["velocities"][idx],
            "positions": f["positions"][idx],
            "N": n,
            "T": t,
            "D": d,
            "dt": float(meta["dt"]),
        }
        if "intent" in f:
            sample["intent"] = {}
            for key in f["intent"].keys():
                data = f["intent"][key]
                if data.shape[:1] == (n,):
                    sample["intent"][key] = data[idx]
        return sample


def compute_stats(path: Path, max_chunk: int) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        meta = {k: _jsonable(v) for k, v in f["metadata"].attrs.items()}
        n = int(meta["num_trajectories"])
        t = int(meta["trajectory_length"])
        d = int(meta["action_dim"])
        dt = float(meta["dt"])
        vel_ds = f["velocities"]
        pos_ds = f["positions"]
        win = min(DEFAULT_MAX_EPISODE_LEN, t)

        sums = np.zeros(d, dtype=np.float64)
        sqs = np.zeros(d, dtype=np.float64)
        speed_sum = 0.0
        speed_sq = 0.0
        speed_count = 0
        speed_p95_samples = []
        delta_samples = []
        max_window_dx = []
        endpoint_xy = []

        for start in range(0, n, max_chunk):
            end = min(n, start + max_chunk)
            v = vel_ds[start:end]
            p = pos_ds[start:end]
            flat = v.reshape(-1, d)
            sums += flat.sum(axis=0)
            sqs += (flat ** 2).sum(axis=0)

            speeds = np.linalg.norm(v[..., :3], axis=-1)
            speed_sum += speeds.sum()
            speed_sq += (speeds ** 2).sum()
            speed_count += speeds.size
            speed_p95_samples.append(speeds.reshape(-1))

            delta = np.diff(v[..., :3], axis=1)
            delta_samples.append(np.linalg.norm(delta, axis=-1).reshape(-1))

            cs = np.cumsum(v[:, :, 0], axis=1) * dt
            if t > win:
                window_dx = cs[:, win - 1:] - np.concatenate(
                    [np.zeros((cs.shape[0], 1)), cs[:, : t - win]], axis=1
                )
            else:
                window_dx = cs[:, -1:]
            max_window_dx.append(window_dx.max(axis=1))
            endpoint_xy.append(p[:, -1, :2])

        count = n * t
        mean = sums / count
        std = np.sqrt(np.maximum(sqs / count - mean ** 2, 0.0))
        speeds_all = np.concatenate(speed_p95_samples)
        delta_all = np.concatenate(delta_samples)
        max_window_dx_all = np.concatenate(max_window_dx)
        endpoint_xy_all = np.concatenate(endpoint_xy)

        stats: dict[str, Any] = {
            "metadata": meta,
            "num_trajectories": n,
            "trajectory_length": t,
            "dt": dt,
            "vel_mean_xyz": mean[:3].tolist(),
            "vel_std_xyz": std[:3].tolist(),
            "speed_mean": float(speed_sum / speed_count),
            "speed_std": float(np.sqrt(max(speed_sq / speed_count - (speed_sum / speed_count) ** 2, 0.0))),
            "speed_p95": float(np.percentile(speeds_all, 95)),
            "delta_v_mean": float(delta_all.mean()),
            "delta_v_p95": float(np.percentile(delta_all, 95)),
            "delta_v_p99": float(np.percentile(delta_all, 99)),
            "best_window_dx_median": float(np.median(max_window_dx_all)),
            "best_window_dx_p10": float(np.percentile(max_window_dx_all, 10)),
            "best_window_dx_p90": float(np.percentile(max_window_dx_all, 90)),
            "reachable_fraction_at_1500_steps": float((max_window_dx_all >= TUNNEL_FORWARD_DIST).mean()),
            "endpoint_spread_xy": float(
                np.sqrt(np.var(endpoint_xy_all[:, 0]) + np.var(endpoint_xy_all[:, 1]))
            ),
        }

        if "intent" in f:
            intent_stats = {}
            for key in ("intent_mode", "react_mode"):
                if key in f["intent"]:
                    values = f["intent"][key][:]
                    hist = np.bincount(values.reshape(-1).astype(np.int64))
                    total = max(hist.sum(), 1)
                    intent_stats[f"{key}_fraction"] = {
                        str(i): float(v / total) for i, v in enumerate(hist)
                    }
            if "threat" in f["intent"]:
                threat = f["intent"]["threat"][:]
                intent_stats["threat_mean"] = float(threat.mean())
                intent_stats["threat_p95"] = float(np.percentile(threat, 95))
            stats["intent"] = intent_stats

        return stats


def _sample_bounds(samples: dict[str, dict[str, Any]], axes: tuple[int, int]) -> tuple[float, float, float, float]:
    points = []
    for sample in samples.values():
        p = sample["positions"][..., list(axes)].reshape(-1, 2)
        points.append(p)
    all_points = np.concatenate(points, axis=0)
    lo = np.percentile(all_points, 1, axis=0)
    hi = np.percentile(all_points, 99, axis=0)
    pad = np.maximum((hi - lo) * 0.08, 0.5)
    return float(lo[0] - pad[0]), float(hi[0] + pad[0]), float(lo[1] - pad[1]), float(hi[1] + pad[1])


def plot_trajectory_grid(samples: dict[str, dict[str, Any]], out_path: Path, axes: tuple[int, int], title: str):
    labels = ("x", "y", "z")
    cols = min(3, len(samples))
    rows = math.ceil(len(samples) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.6 * rows), squeeze=False)
    x0, x1, y0, y1 = _sample_bounds(samples, axes)

    for ax, (label, sample) in zip(axs.reshape(-1), samples.items()):
        p = sample["positions"]
        for i in range(p.shape[0]):
            ax.plot(p[i, :, axes[0]], p[i, :, axes[1]], lw=0.6, alpha=0.35)
        if axes[0] == 0:
            ax.axvline(TUNNEL_START_X, color="tab:green", ls="--", lw=1.0)
            ax.axvline(TUNNEL_GOAL_X, color="tab:red", ls="--", lw=1.0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel(labels[axes[0]])
        ax.set_ylabel(labels[axes[1]])
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axs.reshape(-1)[len(samples):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_velocity_histograms(samples: dict[str, dict[str, Any]], out_path: Path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis_idx, axis_label in enumerate(("vx", "vy", "vz")):
        ax = axs[axis_idx]
        for label, sample in samples.items():
            flat = sample["velocities"][..., axis_idx].reshape(-1)
            ax.hist(flat, bins=80, density=True, histtype="step", lw=1.5, label=label)
        ax.set_title(axis_label)
        ax.set_xlabel("m/s")
        ax.grid(True, alpha=0.25)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_forward_progress(samples: dict[str, dict[str, Any]], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, sample in samples.items():
        v = sample["velocities"]
        dt = sample["dt"]
        t = np.arange(v.shape[1]) * dt
        cumdx = np.cumsum(v[:, :, 0], axis=1) * dt
        median = np.median(cumdx, axis=0)
        p10 = np.percentile(cumdx, 10, axis=0)
        p90 = np.percentile(cumdx, 90, axis=0)
        ax.plot(t, median, lw=1.8, label=label)
        ax.fill_between(t, p10, p90, alpha=0.12)
    ax.axhline(TUNNEL_FORWARD_DIST, color="tab:red", ls="--", lw=1.0, label="tunnel goal dx")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("cumulative dx [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_scalar_distribution(samples: dict[str, dict[str, Any]], out_path: Path, kind: str):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, sample in samples.items():
        v = sample["velocities"][..., :3]
        if kind == "speed":
            values = np.linalg.norm(v, axis=-1).reshape(-1)
            xlabel = "speed norm [m/s]"
        elif kind == "delta_v":
            values = np.linalg.norm(np.diff(v, axis=1), axis=-1).reshape(-1)
            xlabel = "|delta v| per step [m/s]"
        else:
            raise ValueError(kind)
        ax.hist(values, bins=80, density=True, histtype="step", lw=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_endpoint_distribution(samples: dict[str, dict[str, Any]], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, sample in samples.items():
        endpoints = sample["positions"][:, -1, :2]
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=12, alpha=0.65, label=label)
    ax.axvline(TUNNEL_START_X, color="tab:green", ls="--", lw=1.0)
    ax.axvline(TUNNEL_GOAL_X, color="tab:red", ls="--", lw=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_intent_modes(samples: dict[str, dict[str, Any]], out_path: Path):
    intent_items = {
        label: sample["intent"]
        for label, sample in samples.items()
        if "intent" in sample and "intent_mode" in sample["intent"]
    }
    if not intent_items:
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, intent in intent_items.items():
        for ax, key in zip(axs, ("intent_mode", "react_mode")):
            if key not in intent:
                continue
            values = intent[key].reshape(-1).astype(np.int64)
            hist = np.bincount(values)
            frac = hist / max(hist.sum(), 1)
            ax.plot(np.arange(len(frac)), frac, marker="o", lw=1.5, label=label)
            ax.set_title(key)
            ax.set_xlabel("mode id")
            ax.set_ylabel("fraction")
            ax.grid(True, alpha=0.25)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_summary_markdown(stats: dict[str, Any], out_path: Path):
    lines = ["# User Input Dataset Comparison", ""]
    for label, item in stats.items():
        meta = item["metadata"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- path: `{item['path']}`",
                f"- generator_kind: `{meta.get('generator_kind', 'unknown')}`",
                f"- trajectories: {item['num_trajectories']}",
                f"- length: {item['trajectory_length']}",
                f"- vx mean/std: {item['vel_mean_xyz'][0]:.4f} / {item['vel_std_xyz'][0]:.4f}",
                f"- speed mean/p95: {item['speed_mean']:.4f} / {item['speed_p95']:.4f}",
                f"- delta_v mean/p99: {item['delta_v_mean']:.4f} / {item['delta_v_p99']:.4f}",
                f"- best dx median: {item['best_window_dx_median']:.4f}",
                f"- reachable@1500: {item['reachable_fraction_at_1500_steps']:.4f}",
                "",
            ]
        )
        if "intent" in item:
            lines.append("- intent diagnostics: present")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {}
    stats = {}
    for idx, (label, path) in enumerate(args.dataset):
        samples[label] = load_sample(path, args.num, args.seed + idx * 997)
        stats[label] = compute_stats(path, args.max_chunk)
        stats[label]["path"] = str(path)

    plot_trajectory_grid(samples, out_dir / "01_xy_trajectories_grid.png", (0, 1), "XY trajectories")
    plot_trajectory_grid(samples, out_dir / "02_xz_trajectories_grid.png", (0, 2), "XZ trajectories")
    plot_velocity_histograms(samples, out_dir / "03_velocity_histograms_overlay.png")
    plot_forward_progress(samples, out_dir / "04_forward_progress_overlay.png")
    plot_scalar_distribution(samples, out_dir / "05_delta_v_distribution.png", "delta_v")
    plot_endpoint_distribution(samples, out_dir / "06_endpoint_distribution.png")
    plot_scalar_distribution(samples, out_dir / "07_speed_distribution.png", "speed")
    plot_intent_modes(samples, out_dir / "08_intent_mode_distribution.png")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    write_summary_markdown(stats, out_dir / "summary.md")

    print(f"[compare] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
