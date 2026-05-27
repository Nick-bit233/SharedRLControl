"""Visualize online-sampled pilot trajectory distributions without training.

This script instantiates the user / pilot models directly and rolls them out in a
lightweight kinematic tunnel world so we can inspect what the online RL inputs
look like before training any assistant policy.

Key properties:
1. No Isaac Sim startup
2. No training loop
3. No offline dataset required
4. Online sampling for baseline / tunnel / diverse / intent pilot models

Run from `SharedRLControl/isaac-training/`:

    /home/haoming/miniconda3/envs/env_isaaclab/bin/python \
        experiments/06_tunnel_intent_task/visualize_pilot_distributions.py \
        --experiment tunnel_intent \
        --models baseline tunnel diverse intent \
        --num-trajs 64 \
        --steps 600
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.simulated_users.pilot_modes import IntentMode, ReactMode
from src.simulated_users.user_model import UserModel
from src.simulated_users.user_model_diverse import UserModelDiverse
from src.simulated_users.user_model_intent import UserModelIntent
from src.simulated_users.user_model_tunnely import UserModelTunnel


TUNNEL_START_X = -7.0
TUNNEL_GOAL_X = 10.0
SUPPORTED_MODELS = ("baseline", "tunnel", "diverse", "intent")


@dataclass
class ObstacleField:
    centers_xy: torch.Tensor
    radii: torch.Tensor
    half_x: float
    half_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize online pilot trajectory distributions without training."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="tunnel_intent",
        help="Experiment config name under configs/experiment/ to define map and pilot priors.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
        help="Pilot models to compare.",
    )
    parser.add_argument("--num-trajs", type=int, default=64, help="Number of trajectories per model.")
    parser.add_argument("--steps", type=int, default=600, help="Number of rollout steps per trajectory.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for sampling.",
    )
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=None,
        help="Override tunnel obstacle count used by the lightweight geometry model.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.8,
        help="Clearance threshold used to mark risky trajectory segments.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for plots and JSON summary. Defaults to outputs/pilot_distribution_compare/<experiment>.",
    )
    return parser.parse_args()


def load_cfg(experiment: str, device: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    cfg = OmegaConf.create()
    for rel_path in [
        "train.yaml",
        "algo/ppo.yaml",
        "task/drone.yaml",
        "task/sim.yaml",
        "task/user_model.yaml",
    ]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_dir / rel_path))

    experiment_path = config_dir / "experiment" / f"{experiment}.yaml"
    if experiment_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(experiment_path))
    else:
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.headless = True
    cfg.record_video = False
    cfg.eval_visualization = False
    cfg.wandb.mode = "disabled"
    cfg.user_model.offline_mode = False
    cfg.user_model.dataset_path = None
    cfg.user_model.preload_data = False
    cfg.user_model.simple_mode = False
    cfg.user_model.enable_yaw_rate = False
    OmegaConf.set_struct(cfg, True)
    return cfg


def build_model(model_name: str, cfg, num_envs: int):
    if model_name == "baseline":
        return UserModel(num_envs=num_envs, cfg=cfg, offline_mode=False)
    if model_name == "tunnel":
        return UserModelTunnel(num_envs=num_envs, cfg=cfg, offline_mode=False)
    if model_name == "diverse":
        return UserModelDiverse(num_envs=num_envs, cfg=cfg)
    if model_name == "intent":
        return UserModelIntent(num_envs=num_envs, cfg=cfg)
    raise ValueError(f"Unsupported model: {model_name}")


def make_start_states(cfg, num_trajs: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    half_y = float(cfg.env.map_range[0])
    half_x = float(cfg.env.map_range[1])
    spawn_x = max(TUNNEL_START_X, -half_x + 1.0)
    y_span = min(half_y - 0.8, max(1.0, float(cfg.env.platform_width) * 0.4))

    x = torch.full((num_trajs,), spawn_x, device=device)
    y = (torch.rand((num_trajs,), generator=generator, device=device) * 2.0 - 1.0) * y_span
    z = torch.full((num_trajs,), 1.5, device=device)
    z += (torch.rand((num_trajs,), generator=generator, device=device) * 2.0 - 1.0) * 0.15

    pos = torch.stack([x, y, z], dim=-1)
    quat = torch.zeros(num_trajs, 4, device=device)
    quat[:, 0] = 1.0
    return pos, quat


def generate_obstacle_field(cfg, num_obstacles: int, seed: int, device: torch.device) -> ObstacleField:
    half_y = float(cfg.env.map_range[0])
    half_x = float(cfg.env.map_range[1])
    obstacle_width_range = cfg.env.get("obstacle_width_range", [0.4, 1.1])
    width_lo, width_hi = float(obstacle_width_range[0]), float(obstacle_width_range[1])

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x_lo = max(TUNNEL_START_X + float(cfg.env.platform_width) * 0.5 + 1.5, -half_x + 1.5)
    x_hi = min(TUNNEL_GOAL_X - 0.75, half_x - 1.0)
    y_margin = 0.75

    accepted_centers: list[torch.Tensor] = []
    accepted_radii: list[torch.Tensor] = []
    max_attempts = max(200, num_obstacles * 20)

    for _ in range(max_attempts):
        if len(accepted_centers) >= num_obstacles:
            break

        radius = (torch.rand((1,), generator=generator, device=device) * (width_hi - width_lo) + width_lo) * 0.5
        x = torch.rand((1,), generator=generator, device=device) * (x_hi - x_lo) + x_lo
        y_limit = max(0.5, half_y - y_margin - radius.item())
        y = (torch.rand((1,), generator=generator, device=device) * 2.0 - 1.0) * y_limit
        candidate = torch.stack([x.squeeze(0), y.squeeze(0)])

        if not accepted_centers:
            accepted_centers.append(candidate)
            accepted_radii.append(radius.squeeze(0))
            continue

        centers = torch.stack(accepted_centers)
        radii = torch.stack(accepted_radii)
        sep = torch.linalg.norm(centers - candidate.unsqueeze(0), dim=-1)
        min_sep = radii + radius + 0.3
        if torch.all(sep > min_sep):
            accepted_centers.append(candidate)
            accepted_radii.append(radius.squeeze(0))

    if accepted_centers:
        centers_xy = torch.stack(accepted_centers)
        radii = torch.stack(accepted_radii)
    else:
        centers_xy = torch.zeros(0, 2, device=device)
        radii = torch.zeros(0, device=device)

    return ObstacleField(
        centers_xy=centers_xy,
        radii=radii,
        half_x=half_x,
        half_y=half_y,
    )


def compute_nearest_geometry(positions: torch.Tensor, field: ObstacleField) -> dict[str, torch.Tensor]:
    num_envs = positions.shape[0]
    device = positions.device

    wall_clearance = field.half_y - positions[:, 1].abs()
    wall_normal = torch.zeros(num_envs, 3, device=device)
    wall_normal[:, 1] = torch.where(positions[:, 1] >= 0.0, -1.0, 1.0)

    nearest_dist = wall_clearance
    nearest_normal = wall_normal

    if field.centers_xy.numel() > 0:
        delta_xy = positions[:, None, :2] - field.centers_xy[None, :, :]
        center_dist = torch.linalg.norm(delta_xy, dim=-1)
        obstacle_clearance = center_dist - field.radii.unsqueeze(0)
        obstacle_idx = obstacle_clearance.argmin(dim=1)
        obstacle_best = obstacle_clearance[
            torch.arange(num_envs, device=device), obstacle_idx
        ]

        obstacle_normal_xy = delta_xy[torch.arange(num_envs, device=device), obstacle_idx]
        obstacle_norm = torch.linalg.norm(obstacle_normal_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        obstacle_normal = torch.zeros(num_envs, 3, device=device)
        obstacle_normal[:, :2] = obstacle_normal_xy / obstacle_norm

        use_obstacle = obstacle_best < nearest_dist
        nearest_dist = torch.where(use_obstacle, obstacle_best, nearest_dist)
        nearest_normal = torch.where(use_obstacle.unsqueeze(-1), obstacle_normal, nearest_normal)

    return {
        "nearest_obstacle_dist": nearest_dist,
        "nearest_obstacle_normal": nearest_normal,
    }


def rollout_model(
    model_name: str,
    cfg,
    num_trajs: int,
    steps: int,
    seed: int,
    field: ObstacleField,
):
    device = torch.device(cfg.device)
    dt = float(cfg.sim.dt)
    env_ids = torch.arange(num_trajs, device=device)

    model = build_model(model_name, cfg, num_trajs)
    positions, quats = make_start_states(cfg, num_trajs, seed, device)

    if model_name == "intent":
        mode_prior = cfg.user_model.intent.get("tunnel_mode_prior", None)
        model.reset(
            positions.clone(),
            quats.clone(),
            env_ids=env_ids,
            seed=seed,
            mode_prior_override=mode_prior,
        )
    else:
        model.reset(positions.clone(), quats.clone(), env_ids=env_ids, seed=seed)

    trajectories = torch.zeros(steps + 1, num_trajs, 3, device=device)
    actions = torch.zeros(steps, num_trajs, 3, device=device)
    clearances = torch.zeros(steps, num_trajs, device=device)
    trajectories[0] = positions

    drone_state = torch.zeros(num_trajs, 10, device=device)
    drone_state[:, 6] = 1.0
    assistant_action = torch.zeros(num_trajs, 3, device=device)

    intent_modes = []
    react_modes = []
    threats = []

    for step_idx in range(steps):
        geom = compute_nearest_geometry(positions, field)
        clearances[step_idx] = geom["nearest_obstacle_dist"]
        drone_state[:, :3] = actions[step_idx - 1] if step_idx > 0 else 0.0

        if model_name == "intent":
            action, _ = model.step(
                drone_state,
                positions,
                assistant_action=assistant_action,
                env_geom=geom,
            )
            debug = model.debug_state()
            intent_modes.append(debug["intent_mode"].detach().cpu())
            react_modes.append(debug["react_mode"].detach().cpu())
            threats.append(debug["threat"].detach().cpu())
        else:
            action, _ = model.step(drone_state, positions)

        actions[step_idx] = action
        positions = positions + action * dt
        trajectories[step_idx + 1] = positions

    result = {
        "trajectories": trajectories.detach().cpu().numpy(),
        "actions": actions.detach().cpu().numpy(),
        "clearances": clearances.detach().cpu().numpy(),
        "dt": dt,
        "model_name": model_name,
    }
    if intent_modes:
        result["intent_modes"] = torch.stack(intent_modes).numpy()
        result["react_modes"] = torch.stack(react_modes).numpy()
        result["threats"] = torch.stack(threats).numpy()
    return result


def summarize_result(result: dict, risk_threshold: float) -> dict:
    trajectories = result["trajectories"]
    actions = result["actions"]
    clearances = result["clearances"]

    deltas = np.diff(trajectories, axis=0)
    path_length = np.linalg.norm(deltas, axis=-1).sum(axis=0)
    speeds = np.linalg.norm(actions, axis=-1)
    endpoints = trajectories[-1, :, :2]

    summary = {
        "num_trajectories": int(trajectories.shape[1]),
        "steps": int(actions.shape[0]),
        "dt": float(result["dt"]),
        "path_length_mean": float(path_length.mean()),
        "path_length_std": float(path_length.std()),
        "speed_mean": float(speeds.mean()),
        "speed_p95": float(np.percentile(speeds, 95)),
        "final_x_mean": float(trajectories[-1, :, 0].mean()),
        "final_x_std": float(trajectories[-1, :, 0].std()),
        "final_y_std": float(trajectories[-1, :, 1].std()),
        "endpoint_spread_xy": float(np.sqrt(np.var(endpoints[:, 0]) + np.var(endpoints[:, 1]))),
        "clearance_min_mean": float(clearances.min(axis=0).mean()),
        "collision_fraction": float((clearances.min(axis=0) < 0.0).mean()),
        "risk_fraction": float((clearances < risk_threshold).mean()),
        "vx_mean": float(actions[..., 0].mean()),
        "vy_mean": float(actions[..., 1].mean()),
        "vz_mean": float(actions[..., 2].mean()),
        "vx_std": float(actions[..., 0].std()),
        "vy_std": float(actions[..., 1].std()),
        "vz_std": float(actions[..., 2].std()),
    }

    if "intent_modes" in result:
        intent_modes = result["intent_modes"].reshape(-1)
        react_modes = result["react_modes"].reshape(-1)
        threats = result["threats"].reshape(-1)

        intent_hist = np.bincount(intent_modes, minlength=IntentMode.count()).astype(np.float64)
        react_hist = np.bincount(react_modes, minlength=ReactMode.count()).astype(np.float64)
        intent_hist = intent_hist / max(intent_hist.sum(), 1.0)
        react_hist = react_hist / max(react_hist.sum(), 1.0)

        summary["intent_mode_fraction"] = {
            mode.name.lower(): float(intent_hist[int(mode)])
            for mode in IntentMode
        }
        summary["react_mode_fraction"] = {
            mode.name.lower(): float(react_hist[int(mode)])
            for mode in ReactMode
        }
        summary["threat_mean"] = float(threats.mean())
        summary["threat_p95"] = float(np.percentile(threats, 95))

    return summary


def draw_tunnel(ax, field: ObstacleField):
    ax.axhline(field.half_y, color="black", lw=1.0)
    ax.axhline(-field.half_y, color="black", lw=1.0)
    ax.axvline(TUNNEL_START_X, color="tab:green", ls="--", lw=1.0, label="start")
    ax.axvline(TUNNEL_GOAL_X, color="tab:red", ls="--", lw=1.0, label="goal")

    for center, radius in zip(field.centers_xy.cpu().numpy(), field.radii.cpu().numpy()):
        circle = plt.Circle(center, radius, color="0.75", alpha=0.6)
        ax.add_patch(circle)

    ax.set_xlim(-field.half_x, field.half_x)
    ax.set_ylim(-field.half_y - 0.5, field.half_y + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_trajectory_distribution(results: dict[str, dict], field: ObstacleField, risk_threshold: float, out_path: Path):
    num_models = len(results)
    cols = 2
    rows = math.ceil(num_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows), squeeze=False)

    for ax, (model_name, result) in zip(axes.reshape(-1), results.items()):
        draw_tunnel(ax, field)
        traj = result["trajectories"]
        clearances = result["clearances"]

        for traj_idx in range(traj.shape[1]):
            ax.plot(traj[:, traj_idx, 0], traj[:, traj_idx, 1], lw=0.7, alpha=0.25, color="tab:blue")

        risk_mask = clearances < risk_threshold
        if np.any(risk_mask):
            risk_points = traj[1:][risk_mask]
            sample_stride = max(1, len(risk_points) // 2000)
            sampled = risk_points[::sample_stride]
            ax.scatter(sampled[:, 0], sampled[:, 1], s=5, alpha=0.35, color="tab:red", label="risk")

        endpoints = traj[-1]
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=12, alpha=0.7, color="tab:orange", label="end")
        ax.set_title(model_name)

    for ax in axes.reshape(-1)[num_models:]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)))
    fig.suptitle("Online-sampled pilot trajectory distributions", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_velocity_histograms(results: dict[str, dict], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = ("vx", "vy", "vz")
    colors = {
        "baseline": "tab:blue",
        "tunnel": "tab:green",
        "diverse": "tab:purple",
        "intent": "tab:orange",
    }

    for axis_idx, axis_label in enumerate(labels):
        ax = axes[axis_idx]
        for model_name, result in results.items():
            flat = result["actions"][..., axis_idx].reshape(-1)
            ax.hist(
                flat,
                bins=80,
                density=True,
                histtype="step",
                lw=1.6,
                alpha=0.95,
                label=model_name,
                color=colors.get(model_name, None),
            )
        ax.set_title(f"{axis_label} distribution")
        ax.set_xlabel(f"{axis_label} [m/s]")
        ax.grid(True, alpha=0.25)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_intent_diagnostics(results: dict[str, dict], out_path: Path):
    intent_result = results.get("intent")
    if intent_result is None or "intent_modes" not in intent_result:
        return

    intent_modes = intent_result["intent_modes"].reshape(-1)
    react_modes = intent_result["react_modes"].reshape(-1)

    intent_hist = np.bincount(intent_modes, minlength=IntentMode.count()).astype(np.float64)
    react_hist = np.bincount(react_modes, minlength=ReactMode.count()).astype(np.float64)
    intent_hist /= max(intent_hist.sum(), 1.0)
    react_hist /= max(react_hist.sum(), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        [mode.name.lower() for mode in IntentMode],
        intent_hist,
        color="tab:orange",
        alpha=0.85,
    )
    axes[0].set_title("Intent mode fraction")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(
        [mode.name.lower() for mode in ReactMode],
        react_hist,
        color="tab:red",
        alpha=0.85,
    )
    axes[1].set_title("Reactive mode fraction")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Visualizer] Requested device {args.device}, but CUDA is unavailable. Falling back to cpu.")
        args.device = "cpu"

    cfg = load_cfg(args.experiment, args.device)
    device = torch.device(cfg.device)

    default_out_dir = (
        Path("outputs")
        / "pilot_distribution_compare"
        / f"{args.experiment}_seed{args.seed}"
    )
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    obstacle_count = args.num_obstacles
    if obstacle_count is None:
        obstacle_count = int(cfg.env.get("num_obstacles", 24))

    field = generate_obstacle_field(cfg, obstacle_count, args.seed + 1000, device)

    summaries = {
        "experiment": args.experiment,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "num_trajs": args.num_trajs,
        "online_only": True,
        "obstacle_count": int(field.centers_xy.shape[0]),
        "models": {},
    }

    results = {}
    for model_offset, model_name in enumerate(args.models):
        print(f"[Visualizer] Sampling {model_name} trajectories...")
        result = rollout_model(
            model_name=model_name,
            cfg=cfg,
            num_trajs=args.num_trajs,
            steps=args.steps,
            seed=args.seed + model_offset * 97,
            field=field,
        )
        results[model_name] = result
        summaries["models"][model_name] = summarize_result(result, args.risk_threshold)

    plot_trajectory_distribution(
        results=results,
        field=field,
        risk_threshold=args.risk_threshold,
        out_path=out_dir / "trajectory_distribution_xy.png",
    )
    plot_velocity_histograms(
        results=results,
        out_path=out_dir / "velocity_distribution.png",
    )
    plot_intent_diagnostics(
        results=results,
        out_path=out_dir / "intent_mode_distribution.png",
    )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    print(f"[Visualizer] Wrote summary: {summary_path}")
    print(f"[Visualizer] Wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()
