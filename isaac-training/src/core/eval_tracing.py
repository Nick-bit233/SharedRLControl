"""Trajectory trace extraction and plotting for post-hoc evaluation."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np


SCALAR_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reward", ("next", "agents", "reward")),
    ("pilot_risk_dyn_post", ("next", "agents", "pilot_risk_dyn_post")),
    ("assist_risk_dyn_post", ("next", "agents", "assist_risk_dyn_post")),
    ("assist_risk_dyn_full", ("next", "agents", "assist_risk_dyn_full")),
    ("delay_risk", ("next", "agents", "delay_risk")),
    ("risk_reduction_dyn", ("next", "agents", "risk_reduction_dyn")),
    ("return", ("next", "stats", "return")),
    ("episode_len", ("next", "stats", "episode_len")),
    ("collision", ("next", "stats", "collision")),
    ("out_of_bounds", ("next", "stats", "out_of_bounds")),
    ("above_bound", ("next", "stats", "above_bound")),
    ("below_bound", ("next", "stats", "below_bound")),
    ("terminated", ("next", "stats", "terminated")),
    ("truncated", ("next", "stats", "truncated")),
    ("success", ("next", "stats", "success")),
    ("diag_reward", ("next", "stats", "diag_reward")),
    ("diag_reward_task", ("next", "stats", "diag_reward_task")),
    ("diag_pilot_risk_dyn_post", ("next", "stats", "diag_pilot_risk_dyn_post")),
    ("diag_assist_risk_dyn_post", ("next", "stats", "diag_assist_risk_dyn_post")),
    ("diag_assist_risk_dyn_full", ("next", "stats", "diag_assist_risk_dyn_full")),
    ("diag_delay_risk", ("next", "stats", "diag_delay_risk")),
    ("diag_risk_reduction_dyn", ("next", "stats", "diag_risk_reduction_dyn")),
    ("diag_min_clearance_pilot", ("next", "stats", "diag_min_clearance_pilot")),
    ("diag_min_clearance_assist", ("next", "stats", "diag_min_clearance_assist")),
    ("diag_risk_worsening_rate", ("next", "stats", "diag_risk_worsening_rate")),
    ("diag_intervention_rate", ("next", "stats", "diag_intervention_rate")),
    ("diag_unnecessary_intervention_rate", ("next", "stats", "diag_unnecessary_intervention_rate")),
    ("diag_unsafe_non_intervention_rate", ("next", "stats", "diag_unsafe_non_intervention_rate")),
    ("diag_modal_residual_norm", ("next", "stats", "diag_modal_residual_norm")),
    ("diag_penalty_smooth", ("next", "stats", "diag_penalty_smooth")),
    ("diag_penalty_height", ("next", "stats", "diag_penalty_height")),
)

VECTOR_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("position_w", ("next", "stats", "debug_pos_world")),
    ("velocity_w", ("next", "stats", "debug_vec_world")),
    ("policy_command_w", ("next", "stats", "debug_vec_policy")),
    ("human_command_w", ("next", "stats", "debug_vec_target")),
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(cfg, key, default)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_last_unit_dims(array: np.ndarray) -> np.ndarray:
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _get_nested(trajs: Any, key: tuple[str, ...]) -> Any:
    getter = getattr(trajs, "get", None)
    if getter is not None:
        try:
            value = getter(key)
        except Exception:
            value = None
        if value is not None:
            return value
    try:
        return trajs[key]
    except Exception:
        return None


def as_time_env_scalar(value: Any, *, name: str) -> np.ndarray:
    array = _squeeze_last_unit_dims(_as_numpy(value))
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to [env, time], got shape {array.shape}")
    return array


def as_time_env_vector(value: Any, *, name: str, width: int = 3) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 3 or array.shape[-1] != width:
        raise ValueError(f"{name} must have shape [env, time, {width}], got {array.shape}")
    return array


def first_done_indices(done: Any) -> np.ndarray:
    done_arr = as_time_env_scalar(done, name="done").astype(bool, copy=False)
    first = np.full(done_arr.shape[0], done_arr.shape[1] - 1, dtype=np.int64)
    for env_id in range(done_arr.shape[0]):
        done_steps = np.flatnonzero(done_arr[env_id])
        if done_steps.size:
            first[env_id] = int(done_steps[0])
    return first


def first_episode_valid_mask(done: Any) -> np.ndarray:
    done_arr = as_time_env_scalar(done, name="done").astype(bool, copy=False)
    first = first_done_indices(done_arr)
    valid = np.zeros(done_arr.shape, dtype=bool)
    for env_id, last_step in enumerate(first):
        valid[env_id, : int(last_step) + 1] = True
    return valid


def _take_final(array: np.ndarray, first_done: np.ndarray) -> np.ndarray:
    return array[np.arange(array.shape[0]), first_done]


def _safe_array(data: dict[str, np.ndarray], key: str, fallback: float = 0.0) -> np.ndarray:
    value = data.get(key)
    if value is None:
        first = next(iter(data.values()))
        return np.full(first.shape[:2], fallback, dtype=np.float32)
    return value


def _parse_env_ids(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        try:
            values = [int(item) for item in value]
        except TypeError:
            values = [int(value)]
    if not values:
        return None
    unique_values = list(dict.fromkeys(values))
    return np.asarray(unique_values, dtype=np.int64)


def _validate_env_ids(env_ids: np.ndarray, *, num_envs: int) -> np.ndarray:
    if ((env_ids < 0) | (env_ids >= int(num_envs))).any():
        bad = env_ids[(env_ids < 0) | (env_ids >= int(num_envs))]
        raise ValueError(f"trace env_ids out of range for num_envs={num_envs}: {bad.tolist()}")
    return env_ids


def _diagnostic_reasons_for_envs(
    trace: dict[str, np.ndarray],
    env_ids: np.ndarray,
    *,
    max_episodes: int,
) -> list[str]:
    diagnostic_ids, diagnostic_reasons = select_diagnostic_envs(
        trace,
        max_episodes=max(int(max_episodes), int(env_ids.size)),
    )
    reason_by_env = {
        int(env_id): str(reason)
        for env_id, reason in zip(diagnostic_ids, diagnostic_reasons)
    }
    return [reason_by_env.get(int(env_id), "forced_env_id") for env_id in env_ids]


def select_diagnostic_envs(
    trace: dict[str, np.ndarray],
    *,
    max_episodes: int,
) -> tuple[np.ndarray, list[str]]:
    """Choose representative env ids for failure and risk diagnosis."""

    done = trace["done"].astype(bool, copy=False)
    valid = first_episode_valid_mask(done)
    first_done = first_done_indices(done)
    final_return = _take_final(_safe_array(trace, "return"), first_done)
    final_success = _take_final(_safe_array(trace, "success"), first_done)
    collision = _safe_array(trace, "collision")
    height_penalty = _safe_array(trace, "diag_penalty_height")
    risk = _safe_array(trace, "assist_risk_dyn_full")
    risk_worse = _safe_array(trace, "diag_risk_worsening_rate")

    selected: list[int] = []
    tags: dict[int, list[str]] = {}

    def add(env_id: int | None, reason: str) -> None:
        if env_id is None:
            return
        env_id = int(env_id)
        tags.setdefault(env_id, []).append(reason)
        if env_id not in selected and len(selected) < max_episodes:
            selected.append(env_id)

    add(int(np.nanargmin(final_return)), "lowest_return")

    collision_any = (collision.astype(bool) & valid).any(axis=1)
    if collision_any.any():
        first_collision = np.full(collision.shape[0], collision.shape[1] + 1, dtype=np.int64)
        for env_id in np.flatnonzero(collision_any):
            first_collision[env_id] = int(np.flatnonzero(collision[env_id].astype(bool) & valid[env_id])[0])
        add(int(np.argmin(first_collision)), "earliest_collision")

    valid_counts = valid.sum(axis=1).clip(min=1)
    height_mean = np.where(valid, height_penalty, 0.0).sum(axis=1) / valid_counts
    add(int(np.nanargmax(height_mean)), "highest_height_penalty")

    max_risk = np.where(valid, risk, -np.inf).max(axis=1)
    add(int(np.nanargmax(max_risk)), "highest_max_risk")

    risk_worse_mean = np.where(valid, risk_worse, 0.0).sum(axis=1) / valid_counts
    add(int(np.nanargmax(risk_worse_mean)), "highest_risk_worsening")

    success_ids = np.flatnonzero(final_success > 0.5)
    if success_ids.size:
        success_returns = final_return[success_ids]
        median_return = float(np.median(success_returns))
        add(int(success_ids[np.argmin(np.abs(success_returns - median_return))]), "median_success_return")

    for env_id in np.argsort(final_return):
        if len(selected) >= max_episodes:
            break
        add(int(env_id), "fallback_low_return")

    return np.asarray(selected, dtype=np.int64), [";".join(tags.get(int(env_id), [])) for env_id in selected]


def _extract_lidar_min_distance(cfg: Any, trajs: Any) -> np.ndarray | None:
    lidar = _get_nested(trajs, ("next", "agents", "observation", "lidar"))
    if lidar is None:
        return None
    from src.core.risk_calibration import lidar_min_distance

    sensor_cfg = _cfg_get(cfg, "sensor", {})
    lidar_range = float(_cfg_get(sensor_cfg, "lidar_range", 4.0))
    return lidar_min_distance(lidar, lidar_range=lidar_range).astype(np.float32, copy=False)


def extract_eval_trace(
    cfg: Any,
    trajs: Any,
    *,
    max_episodes: int,
    include_lidar_min_distance: bool = True,
    env_ids: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Extract selected first-episode trajectories from an eval rollout."""

    done_raw = _get_nested(trajs, ("next", "done"))
    if done_raw is None:
        raise ValueError("rollout is missing ('next', 'done')")
    done = as_time_env_scalar(done_raw, name="done").astype(bool, copy=False)
    valid = first_episode_valid_mask(done)
    first_done = first_done_indices(done)

    full: dict[str, np.ndarray] = {
        "done": done.astype(bool, copy=False),
        "valid_mask": valid,
    }

    for name, key in SCALAR_SPECS:
        value = _get_nested(trajs, key)
        if value is None:
            continue
        full[name] = as_time_env_scalar(value, name=name).astype(np.float32, copy=False)

    for name, key in VECTOR_SPECS:
        value = _get_nested(trajs, key)
        if value is None:
            continue
        full[name] = as_time_env_vector(value, name=name).astype(np.float32, copy=False)

    if include_lidar_min_distance:
        min_lidar = _extract_lidar_min_distance(cfg, trajs)
        if min_lidar is not None:
            full["nearest_lidar_distance"] = min_lidar

    forced_env_ids = _parse_env_ids(env_ids)
    if forced_env_ids is None:
        selected_env_ids, reasons = select_diagnostic_envs(full, max_episodes=max(1, int(max_episodes)))
        selection = "diagnostic"
    else:
        selected_env_ids = _validate_env_ids(forced_env_ids, num_envs=done.shape[0])
        reasons = _diagnostic_reasons_for_envs(full, selected_env_ids, max_episodes=max(1, int(max_episodes)))
        selection = "forced_env_ids"
    selected: dict[str, np.ndarray] = {
        key: value[selected_env_ids].copy()
        for key, value in full.items()
    }
    selected["env_ids"] = selected_env_ids
    selected["lengths"] = (first_done[selected_env_ids] + 1).astype(np.int64)
    selected["selection_reason"] = np.asarray(reasons, dtype="U128")

    metadata = {
        "num_envs": int(done.shape[0]),
        "num_steps": int(done.shape[1]),
        "selected_count": int(selected_env_ids.size),
        "selection": selection,
        "selected_env_ids": selected_env_ids.tolist(),
        "available_keys": sorted(key for key in selected if key not in {"selection_reason"}),
    }

    index_rows = build_trace_index(selected)
    return selected, metadata, index_rows


def _final_for_selected(trace: dict[str, np.ndarray], key: str, row: int, length: int) -> float:
    value = trace.get(key)
    if value is None:
        return float("nan")
    return float(value[row, max(0, length - 1)])


def build_trace_index(trace: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_ids = trace["env_ids"]
    lengths = trace["lengths"]
    reasons = trace["selection_reason"]
    risk = trace.get("assist_risk_dyn_full", trace.get("assist_risk_dyn_post"))
    for row, env_id in enumerate(env_ids):
        length = int(lengths[row])
        valid_slice = slice(0, length)
        row_risk = risk[row, valid_slice] if risk is not None else np.asarray([], dtype=np.float32)
        rows.append(
            {
                "rank": row,
                "env_id": int(env_id),
                "selection_reason": str(reasons[row]),
                "length": length,
                "final_return": _final_for_selected(trace, "return", row, length),
                "final_success": _final_for_selected(trace, "success", row, length),
                "final_collision": _final_for_selected(trace, "collision", row, length),
                "final_out_of_bounds": _final_for_selected(trace, "out_of_bounds", row, length),
                "max_assist_risk_dyn_full": float(np.nanmax(row_risk)) if row_risk.size else float("nan"),
            }
        )
    return rows


def _write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_trace_outputs(
    output_dir: str | Path,
    trace: dict[str, np.ndarray],
    metadata: dict[str, Any],
    index_rows: list[dict[str, Any]],
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "trajectory_trace.npz"
    metadata_path = output / "trajectory_trace_metadata.json"
    index_path = output / "trajectory_trace_index.csv"

    np.savez_compressed(npz_path, **trace)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_jsonable))
    _write_index_csv(index_path, index_rows)
    return {
        "trace_npz": str(npz_path),
        "trace_metadata": str(metadata_path),
        "trace_index": str(index_path),
    }


def _trace_cfg(cfg: Any) -> Any:
    return _cfg_get(_cfg_get(cfg, "eval", {}), "trace", {})


def trace_enabled(cfg: Any) -> bool:
    return bool(_cfg_get(_trace_cfg(cfg), "enable", False))


def trace_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(_trace_cfg(cfg), key, default)


def _line_collection(ax: Any, x: np.ndarray, y: np.ndarray, color_values: np.ndarray, *, label: str) -> Any:
    from matplotlib.collections import LineCollection

    if x.size < 2:
        ax.scatter(x, y, c=color_values, cmap="viridis", label=label)
        return None
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap="viridis", linewidth=2.0)
    collection.set_array(color_values[:-1])
    collection.set_clim(0.0, 1.0)
    ax.add_collection(collection)
    ax.autoscale()
    ax.scatter(x[0], y[0], s=35, c="#2f6fbb", marker="o", label="start")
    ax.scatter(x[-1], y[-1], s=45, c="#c43d3d", marker="x", label="end")
    return collection


def _series(trace: dict[str, np.ndarray], key: str, row: int, length: int) -> np.ndarray | None:
    value = trace.get(key)
    if value is None:
        return None
    return np.asarray(value[row, :length], dtype=np.float32)


def _plot_series(ax: Any, t: np.ndarray, trace: dict[str, np.ndarray], row: int, length: int, specs: list[tuple[str, str]]) -> None:
    for key, label in specs:
        y = _series(trace, key, row, length)
        if y is not None:
            ax.plot(t, y, label=label, linewidth=1.4)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=7)


def _mark_events(ax: Any, t: np.ndarray, trace: dict[str, np.ndarray], row: int, length: int) -> None:
    for key, color, label in [
        ("collision", "#c43d3d", "collision"),
        ("out_of_bounds", "#9c5f1a", "oob"),
        ("success", "#2f7d32", "success"),
    ]:
        y = _series(trace, key, row, length)
        if y is None:
            continue
        idx = np.flatnonzero(y > 0.5)
        if idx.size:
            ax.axvline(t[int(idx[0])], color=color, linestyle="--", alpha=0.7, label=label)


def plot_trace_panels(
    output_dir: str | Path,
    trace: dict[str, np.ndarray],
    *,
    risk_key: str = "assist_risk_dyn_full",
    sim_dt: float = 1.0,
    safe_height: tuple[float, float] | None = None,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    positions = trace.get("position_w")
    if positions is None:
        return plot_paths
    risk_values = trace.get(risk_key, trace.get("assist_risk_dyn_full", trace.get("assist_risk_dyn_post")))
    if risk_values is None:
        risk_values = np.zeros(positions.shape[:2], dtype=np.float32)

    for row, env_id in enumerate(trace["env_ids"]):
        length = int(trace["lengths"][row])
        if length <= 0:
            continue
        pos = positions[row, :length]
        risk = np.nan_to_num(np.asarray(risk_values[row, :length], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        risk = np.clip(risk, 0.0, 1.0)
        t = np.arange(length, dtype=np.float32) * float(sim_dt)

        fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
        axes = axes.reshape(-1)
        collection = _line_collection(axes[0], pos[:, 0], pos[:, 1], risk, label=risk_key)
        axes[0].set_title("XY trajectory colored by risk")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)
        if collection is not None:
            fig.colorbar(collection, ax=axes[0], label=risk_key)

        collection = _line_collection(axes[1], pos[:, 0], pos[:, 2], risk, label=risk_key)
        axes[1].set_title("XZ trajectory and safe height band")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        if safe_height is not None:
            axes[1].axhspan(safe_height[0], safe_height[1], color="#b7dfb0", alpha=0.25, label="safe height")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7)
        if collection is not None:
            fig.colorbar(collection, ax=axes[1], label=risk_key)

        _plot_series(
            axes[2],
            t,
            trace,
            row,
            length,
            [
                ("pilot_risk_dyn_post", "pilot post"),
                ("assist_risk_dyn_post", "assist post"),
                ("assist_risk_dyn_full", "assist full"),
                ("delay_risk", "delay"),
            ],
        )
        axes[2].set_title("Risk time series")
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("risk")
        axes[2].set_ylim(bottom=0.0)
        _mark_events(axes[2], t, trace, row, length)

        _plot_series(
            axes[3],
            t,
            trace,
            row,
            length,
            [
                ("diag_min_clearance_assist", "min clearance"),
                ("nearest_lidar_distance", "nearest lidar"),
            ],
        )
        axes[3].set_title("Clearance and event markers")
        axes[3].set_xlabel("time")
        axes[3].set_ylabel("distance")
        _mark_events(axes[3], t, trace, row, length)

        policy = _series(trace, "policy_command_w", row, length)
        human = _series(trace, "human_command_w", row, length)
        if policy is not None:
            axes[4].plot(t, np.linalg.norm(policy, axis=-1), label="policy |cmd|")
        if human is not None:
            axes[4].plot(t, np.linalg.norm(human, axis=-1), label="human |cmd|")
        residual = _series(trace, "diag_modal_residual_norm", row, length)
        if residual is not None:
            axes[4].plot(t, residual, label="modal residual")
        axes[4].set_title("Command and residual magnitudes")
        axes[4].set_xlabel("time")
        axes[4].grid(True, alpha=0.3)
        if axes[4].lines:
            axes[4].legend(fontsize=7)

        _plot_series(
            axes[5],
            t,
            trace,
            row,
            length,
            [
                ("diag_reward", "reward"),
                ("diag_reward_task", "task"),
                ("diag_penalty_height", "height penalty"),
                ("diag_penalty_smooth", "smooth penalty"),
            ],
        )
        axes[5].set_title("Reward diagnostics")
        axes[5].set_xlabel("time")

        title = (
            f"env={int(env_id)} reason={trace['selection_reason'][row]} "
            f"return={_final_for_selected(trace, 'return', row, length):.2f}"
        )
        fig.suptitle(title, fontsize=11)
        path = output / f"trajectory_env{int(env_id)}_{row:02d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))
    return plot_paths


def save_eval_trace_artifacts(cfg: Any, trajs: Any, output_dir: str | Path, env: Any | None = None) -> dict[str, Any]:
    trace_cfg = _trace_cfg(cfg)
    max_episodes = int(_cfg_get(trace_cfg, "max_episodes", 6))
    include_lidar = bool(_cfg_get(trace_cfg, "include_lidar_min_distance", True))
    trace, metadata, index_rows = extract_eval_trace(
        cfg,
        trajs,
        max_episodes=max_episodes,
        include_lidar_min_distance=include_lidar,
        env_ids=_cfg_get(trace_cfg, "env_ids", None),
    )

    sim_cfg = _cfg_get(cfg, "sim", {})
    sim_dt = float(_cfg_get(sim_cfg, "dt", 1.0)) * float(_cfg_get(sim_cfg, "substeps", 1))
    env_cfg = _cfg_get(cfg, "env", {})
    map_range = _cfg_get(env_cfg, "map_range", None)
    safe_height: tuple[float, float] | None = None
    if map_range is not None and len(map_range) >= 3:
        z_half = float(map_range[2])
        safe_height = (0.4 * z_half, 1.6 * z_half)

    metadata.update(
        {
            "sim_dt": sim_dt,
            "safe_height": safe_height,
            "risk_key": str(_cfg_get(trace_cfg, "risk_key", "assist_risk_dyn_full")),
            "max_episodes": max_episodes,
        }
    )
    if env is not None:
        metadata["env_max_episode_length"] = int(getattr(env, "max_episode_length", metadata["num_steps"]))

    paths: dict[str, Any] = {}
    if bool(_cfg_get(trace_cfg, "save_npz", True)):
        paths.update(save_trace_outputs(output_dir, trace, metadata, index_rows))

    plot_paths: list[str] = []
    if bool(_cfg_get(trace_cfg, "plot", True)):
        plot_dir = Path(output_dir) / str(_cfg_get(trace_cfg, "plot_dir", "plots"))
        plot_paths = plot_trace_panels(
            plot_dir,
            trace,
            risk_key=str(_cfg_get(trace_cfg, "risk_key", "assist_risk_dyn_full")),
            sim_dt=sim_dt,
            safe_height=safe_height,
        )
        paths["trace_plot_dir"] = str(plot_dir)
        paths["trace_plot_count"] = len(plot_paths)

    paths["trace_selected_count"] = int(trace["env_ids"].size)
    return paths


__all__ = [
    "as_time_env_scalar",
    "as_time_env_vector",
    "build_trace_index",
    "extract_eval_trace",
    "first_done_indices",
    "first_episode_valid_mask",
    "plot_trace_panels",
    "save_eval_trace_artifacts",
    "save_trace_outputs",
    "select_diagnostic_envs",
    "trace_cfg_value",
    "trace_enabled",
]
