"""Utilities for post-hoc dynamic-risk calibration analysis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RISK_BINS = np.asarray(
    [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0],
    dtype=np.float32,
)
DEFAULT_EXPOSURE_THRESHOLDS = np.asarray([0.30, 0.50, 0.70, 0.90], dtype=np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_time_env_array(value: Any, *, name: str) -> np.ndarray:
    """Convert rollout tensors shaped [env, time, ...] to [env, time]."""

    array = _as_numpy(value)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to [env, time], got shape {array.shape}")
    return array


def lidar_min_distance(lidar: Any, *, lidar_range: float) -> np.ndarray:
    """Reconstruct true nearest LiDAR distance from normalized tunnel observations."""

    array = _as_numpy(lidar).astype(np.float32, copy=False)
    if array.ndim < 3:
        raise ValueError(f"lidar must have shape [env, time, ...], got {array.shape}")
    max_scan = array.reshape(array.shape[0], array.shape[1], -1).max(axis=-1)
    distances = float(lidar_range) * (1.0 - max_scan)
    return np.clip(distances, 0.0, float(lidar_range))


def _first_episode_valid_mask(done: np.ndarray) -> np.ndarray:
    done_bool = done.astype(bool, copy=False)
    valid = np.zeros(done_bool.shape, dtype=bool)
    num_envs, num_steps = done_bool.shape
    for env_id in range(num_envs):
        done_steps = np.flatnonzero(done_bool[env_id])
        last_step = int(done_steps[0]) if done_steps.size else num_steps - 1
        valid[env_id, : last_step + 1] = True
    return valid


def compute_future_event_labels(
    *,
    collision: Any,
    min_lidar_distance: Any,
    done: Any,
    horizon_steps: int,
    near_miss_distance: float,
) -> dict[str, np.ndarray]:
    """Create first-episode future labels for collision and near-miss events.

    The future window is inclusive of the current rollout row and contains at
    most ``horizon_steps`` samples: ``[t, t + horizon_steps)``.
    """

    collision_arr = as_time_env_array(collision, name="collision").astype(bool, copy=False)
    min_dist_arr = as_time_env_array(min_lidar_distance, name="min_lidar_distance")
    done_arr = as_time_env_array(done, name="done").astype(bool, copy=False)

    if collision_arr.shape != min_dist_arr.shape or collision_arr.shape != done_arr.shape:
        raise ValueError(
            "collision, min_lidar_distance, and done must have matching shapes: "
            f"{collision_arr.shape}, {min_dist_arr.shape}, {done_arr.shape}"
        )
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    near_miss_arr = min_dist_arr < float(near_miss_distance)
    future_collision = np.zeros_like(collision_arr, dtype=bool)
    future_near_miss = np.zeros_like(near_miss_arr, dtype=bool)
    valid = _first_episode_valid_mask(done_arr)

    num_envs, num_steps = collision_arr.shape
    for env_id in range(num_envs):
        valid_steps = np.flatnonzero(valid[env_id])
        if not valid_steps.size:
            continue
        last_step = int(valid_steps[-1])
        episode_end = last_step + 1
        for step in range(episode_end):
            end = min(episode_end, step + horizon_steps)
            future_collision[env_id, step] = bool(collision_arr[env_id, step:end].any())
            future_near_miss[env_id, step] = bool(near_miss_arr[env_id, step:end].any())

    return {
        "valid": valid,
        "instant_collision": collision_arr,
        "instant_near_miss": near_miss_arr,
        "future_collision": future_collision,
        "future_near_miss": future_near_miss,
        "future_event": future_collision | future_near_miss,
    }


def build_calibration_trace(
    *,
    score: Any,
    collision: Any,
    min_lidar_distance: Any,
    done: Any,
    horizon_steps: int,
    near_miss_distance: float,
    extra_scores: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Flatten first-episode rollout tensors into calibration samples."""

    score_arr = as_time_env_array(score, name="score").astype(np.float32, copy=False)
    min_dist_arr = as_time_env_array(min_lidar_distance, name="min_lidar_distance").astype(
        np.float32,
        copy=False,
    )
    labels = compute_future_event_labels(
        collision=collision,
        min_lidar_distance=min_dist_arr,
        done=done,
        horizon_steps=horizon_steps,
        near_miss_distance=near_miss_distance,
    )
    valid = labels["valid"]
    if score_arr.shape != valid.shape:
        raise ValueError(f"score shape {score_arr.shape} does not match valid mask {valid.shape}")

    trace: dict[str, np.ndarray] = {
        "score": score_arr[valid].astype(np.float32, copy=False),
        "future_collision": labels["future_collision"][valid],
        "future_near_miss": labels["future_near_miss"][valid],
        "future_event": labels["future_event"][valid],
        "instant_collision": labels["instant_collision"][valid],
        "instant_near_miss": labels["instant_near_miss"][valid],
        "min_lidar_distance": min_dist_arr[valid].astype(np.float32, copy=False),
    }
    if extra_scores:
        for key, value in extra_scores.items():
            extra_arr = as_time_env_array(value, name=key).astype(np.float32, copy=False)
            if extra_arr.shape != valid.shape:
                raise ValueError(f"{key} shape {extra_arr.shape} does not match {valid.shape}")
            trace[key] = extra_arr[valid]
    return trace


def _rate(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.astype(np.float64).mean())


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.astype(np.float64).mean())


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(np.float64), q))


def bin_calibration(
    trace: dict[str, np.ndarray],
    *,
    bins: np.ndarray = DEFAULT_RISK_BINS,
) -> list[dict[str, float | int]]:
    """Compute empirical event rates per risk bin."""

    score = np.asarray(trace["score"], dtype=np.float32)
    bins = np.asarray(bins, dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    total = int(score.size)
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == len(bins) - 2:
            mask = (score >= low) & (score <= high)
        else:
            mask = (score >= low) & (score < high)
        count = int(mask.sum())
        denom = float(total) if total else float("nan")
        rows.append(
            {
                "bin_index": idx,
                "risk_low": float(low),
                "risk_high": float(high),
                "count": count,
                "sample_fraction": float(count / denom) if total else float("nan"),
                "score_mean": _mean(score[mask]),
                "future_collision_rate": _rate(trace["future_collision"][mask]),
                "future_near_miss_rate": _rate(trace["future_near_miss"][mask]),
                "future_event_rate": _rate(trace["future_event"][mask]),
                "instant_collision_rate": _rate(trace["instant_collision"][mask]),
                "instant_near_miss_rate": _rate(trace["instant_near_miss"][mask]),
                "mean_min_lidar_distance": _mean(trace["min_lidar_distance"][mask]),
            }
        )
    return rows


def threshold_exposure(
    trace: dict[str, np.ndarray],
    *,
    thresholds: np.ndarray = DEFAULT_EXPOSURE_THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Compute event rates and exposure fractions above risk thresholds."""

    score = np.asarray(trace["score"], dtype=np.float32)
    total = int(score.size)
    rows: list[dict[str, float | int]] = []
    for threshold in np.asarray(thresholds, dtype=np.float32):
        mask = score >= threshold
        count = int(mask.sum())
        rows.append(
            {
                "risk_threshold": float(threshold),
                "count": count,
                "sample_fraction": float(count / total) if total else float("nan"),
                "score_mean": _mean(score[mask]),
                "future_collision_rate": _rate(trace["future_collision"][mask]),
                "future_near_miss_rate": _rate(trace["future_near_miss"][mask]),
                "future_event_rate": _rate(trace["future_event"][mask]),
            }
        )
    return rows


def summarize_trace(trace: dict[str, np.ndarray]) -> dict[str, float | int]:
    score = np.asarray(trace["score"], dtype=np.float32)
    return {
        "count": int(score.size),
        "score_mean": _mean(score),
        "score_q50": _quantile(score, 0.50),
        "score_q90": _quantile(score, 0.90),
        "score_q95": _quantile(score, 0.95),
        "score_q99": _quantile(score, 0.99),
        "future_collision_rate": _rate(trace["future_collision"]),
        "future_near_miss_rate": _rate(trace["future_near_miss"]),
        "future_event_rate": _rate(trace["future_event"]),
        "instant_collision_rate": _rate(trace["instant_collision"]),
        "instant_near_miss_rate": _rate(trace["instant_near_miss"]),
        "mean_min_lidar_distance": _mean(trace["min_lidar_distance"]),
    }


def concat_traces(traces: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not traces:
        raise ValueError("expected at least one trace")
    keys = sorted(set.intersection(*(set(trace) for trace in traces)))
    return {key: np.concatenate([trace[key] for trace in traces]) for key in keys}


def save_trace_npz(path: Path, trace: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    payload = dict(trace)
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def load_trace_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    return trace, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "NaN"
    return f"{value:.4f}"


__all__ = [
    "DEFAULT_EXPOSURE_THRESHOLDS",
    "DEFAULT_RISK_BINS",
    "bin_calibration",
    "build_calibration_trace",
    "compute_future_event_labels",
    "concat_traces",
    "format_float",
    "lidar_min_distance",
    "load_trace_npz",
    "save_trace_npz",
    "summarize_trace",
    "threshold_exposure",
    "write_csv",
]
