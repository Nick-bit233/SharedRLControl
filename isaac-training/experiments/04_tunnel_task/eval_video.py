"""Standalone evaluation + video recording for a trained tunnel checkpoint.

Loads any checkpoint produced by `train.py`, runs a single evaluation rollout
(with deterministic actions), and writes follow-camera + (optional)
global-camera mp4 videos to disk along with the eval-info JSON.

Why a separate script?
- The training loop only logs videos to wandb. Reviewers / debugging
  often need on-disk mp4s without wandb.
- Decouples eval from training so we can sweep over multiple checkpoints
  cheaply.

Usage (run from `SharedRLControl/isaac-training/`):

    # M1 reg=0.005 best ckpt with the same env config as M1 training
    python experiments/04_tunnel_task/eval_video.py \\
        experiment=tunnel \\
        +resume_checkpoint=outputs/tunnel_m1_noreg/reg_0.005/2026-04-22_03-42-12/wandb/run-20260422_034223-o0nzna3g/files/checkpoint_best.pt

    # M2 second-run checkpoint with M2 env config + global view + 1 env only
    python experiments/04_tunnel_task/eval_video.py \\
        experiment=tunnel_m2_diverse_pilot \\
        +resume_checkpoint=outputs/tunnel_m2_diverse_pilot/.../checkpoint_best.pt \\
        +global_view=true \\
        env.num_envs=4 \\
        +video_dir=./eval_videos/m2_secondrun

Outputs (under hydra run dir, or `+video_dir=` if set):
    eval_video_follow.mp4
    eval_video_global.mp4   (only if +global_view=true)
    eval_info.json          (averaged eval/* metrics for that rollout)

Notes:
- Forces `record_video=True` and `wandb.mode=disabled`.
- Defaults `env.num_envs` to 4 (overridable) for speed; the renderer only
  shows env_0 anyway.
- Uses `ExplorationType.DETERMINISTIC` (matches train.py eval).
"""
from __future__ import annotations

import os

# Match train.py CUDA / memory env vars BEFORE torch import
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import math

import hydra
import imageio
import numpy as np
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    # ---------------- force eval-friendly defaults ----------------
    OmegaConf.set_struct(cfg, False)
    cfg.headless = True
    cfg.record_video = True
    cfg.wandb.mode = "disabled"
    if "global_view" not in cfg:
        cfg.global_view = False
    if "video_dir" not in cfg or cfg.video_dir is None:
        cfg.video_dir = None  # filled in after hydra context
    # Smaller num_envs for fast video eval; user can override.
    if cfg.env.get("num_envs", 256) > 16 and not cfg.get("keep_num_envs", False):
        cfg.env.num_envs = 4
    # Single eval, no early termination.
    cfg.max_iterations = 1
    OmegaConf.set_struct(cfg, True)

    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt is None:
        raise ValueError("eval_video.py requires +resume_checkpoint=PATH")
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"checkpoint not found: {resume_ckpt}")

    # ---------------- start sim app FIRST ----------------
    sim_app = init_simulation_app(cfg)

    # Imports that depend on sim app
    from src.envs.env_tunnel import EnvTunnelResidual
    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    algo_policy_mode = cfg.algo.get("policy_mode", "residual")
    if algo_distribution == "beta":
        from src.algos.ppo_constrained_beta import (
            ConstrainedResidualPPO_Beta as ConstrainedResidualPPO,
        )
        print(f"[EvalVideo] Using Beta distribution PPO ({algo_policy_mode} policy mode)")
    else:
        from src.algos.ppo_constrained import ConstrainedResidualPPO

    # ---------------- output dir ----------------
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.runtime.output_dir
    runtime_cwd = hydra_cfg.runtime.cwd
    video_dir = cfg.get("video_dir", None) or hydra_run_dir
    if not os.path.isabs(video_dir):
        video_dir = os.path.abspath(os.path.join(runtime_cwd, video_dir))
    os.makedirs(video_dir, exist_ok=True)
    print(f"[EvalVideo] Output dir: {video_dir}")
    print(f"[EvalVideo] Checkpoint: {resume_ckpt}")
    print(f"[EvalVideo] num_envs={cfg.env.num_envs}, "
          f"max_episode_length={cfg.env.max_episode_length}")

    # ---------------- offline trajectory dataset (if M2-style) ----------------
    trajectory_dataset = None
    if cfg.user_model.get("offline_mode", False):
        from src.datasets.trajectory_dataset import TrajectoryDataset
        dataset_path = cfg.user_model.get("dataset_path", None)
        if dataset_path is None or not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"offline_mode=True but dataset missing: {dataset_path}")
        print(f"[EvalVideo] Loading trajectory dataset: {dataset_path}")
        trajectory_dataset = TrajectoryDataset(
            dataset_path=dataset_path,
            device=torch.device(cfg.device),
            gpu_cache_reserve_gb=cfg.user_model.get("gpu_cache_reserve_gb", 2.0),
            min_scale_factor=cfg.user_model.get("min_scale_factor", 0.5),
            preload_data=cfg.user_model.get("preload_data", True),
        )

    # ---------------- env + policy ----------------
    env = EnvTunnelResidual(cfg, trajectory_dataset=trajectory_dataset)
    policy = ConstrainedResidualPPO(
        cfg.algo, env.observation_spec, env.action_spec, cfg.device
    )

    loaded = torch.load(resume_ckpt, map_location=cfg.device)
    state_dict = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded
    policy.load_state_dict(state_dict)
    print("[EvalVideo] Checkpoint loaded.")

    # ---------------- run one evaluation pass ----------------
    info = run_single_eval(
        env=env,
        policy=policy,
        cfg=cfg,
        video_dir=video_dir,
        seed=cfg.get("eval_seed", 42),
    )

    # Drop any non-serialisable entries (e.g. tensors)
    serialisable = {}
    for k, v in info.items():
        try:
            json.dumps({k: v})
            serialisable[k] = v
        except TypeError:
            pass
    info_path = os.path.join(video_dir, "eval_info.json")
    with open(info_path, "w") as f:
        json.dump(serialisable, f, indent=2, sort_keys=True)
    print(f"[EvalVideo] eval_info.json written: {info_path}")

    sim_app.close()


@torch.no_grad()
def run_single_eval(env, policy, cfg, video_dir: str, seed: int = 42) -> dict:
    """Mirror of train.py::evaluate, but writes mp4 to disk and returns info."""
    env.eval()
    print("[EvalVideo] Enabling renderer and warming up...")
    env.enable_render(True)
    env.set_envs_visibility(visible_env_ids={0})
    for _ in range(10):
        env.sim.render()

    if cfg.get("eval_visualization", False):
        env.set_visualization(enabled=True)

    eval_max_steps = int(env.max_episode_length)
    exploration_type = ExplorationType.DETERMINISTIC
    env.set_seed(seed)

    def rollout_with_camera(camera_mode: str):
        env.set_camera_view_mode(camera_mode)
        cb = RenderCallback(interval=2)
        with set_exploration_type(exploration_type):
            try:
                trajs = env.rollout(
                    max_steps=eval_max_steps,
                    policy=policy,
                    callback=cb,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
            except Exception as e:
                print(f"[EvalVideo] Rendering error: {e}; retrying without render")
                trajs = env.rollout(
                    max_steps=eval_max_steps,
                    policy=policy,
                    callback=None,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
                cb = None
        env.reset()
        return cb, trajs

    print("[EvalVideo] Rollout with FOLLOW camera...")
    cb_follow, trajs = rollout_with_camera("follow")

    cb_global = None
    if cfg.get("global_view", False):
        print("[EvalVideo] Rollout with GLOBAL camera...")
        cb_global, _ = rollout_with_camera("global")

    # ---------------- aggregate eval/* stats ----------------
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(t: torch.Tensor):
        idx = first_done.reshape(first_done.shape + (1,) * (t.ndim - 2))
        return torch.take_along_dim(t, idx, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }
    info: dict = {}
    for k, v in traj_stats.items():
        v_mean = torch.mean(v.float(), dim=0)
        if v_mean.numel() == 1:
            info[f"eval/{k}"] = v_mean.item()
        else:
            for suf, val in zip(["x", "y", "z", "w"][:v_mean.numel()], v_mean.reshape(-1)):
                info[f"eval_debug/{k}/{suf}"] = val.item()

    info.update(compute_intent_safety_metrics(trajs, env, cfg))
    print("[EvalVideo] eval info (single rollout):")
    for k, v in sorted(info.items()):
        print(f"    {k}: {v}")

    # ---------------- save videos ----------------
    fps = max(1, int(round(0.5 / (cfg.sim.dt * cfg.sim.substeps))))
    if cb_follow is not None and len(cb_follow.frames) > 0:
        path = os.path.join(video_dir, "eval_video_follow.mp4")
        save_video(cb_follow.frames, path, fps)
        info["video/follow"] = path
    else:
        print("[EvalVideo] WARNING: no follow frames captured.")

    if cb_global is not None and len(cb_global.frames) > 0:
        path = os.path.join(video_dir, "eval_video_global.mp4")
        save_video(cb_global.frames, path, fps)
        info["video/global"] = path

    env.set_envs_visibility(visible_env_ids=None)
    env.set_visualization(enabled=False)
    env.enable_render(False)
    env.train()
    return info


def _first_done_indices(done: torch.Tensor) -> torch.Tensor:
    done = done.squeeze(-1).bool().cpu()
    has_done = done.any(dim=1)
    first_done = torch.argmax(done.long(), dim=1)
    fallback = torch.full_like(first_done, done.shape[1] - 1)
    return torch.where(has_done, first_done, fallback)


def _point_to_polyline_dist(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    polyline = np.asarray(polyline, dtype=np.float32)
    if len(points) == 0 or len(polyline) == 0:
        return np.full(len(points), np.nan, dtype=np.float32)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    seg_starts = polyline[:-1]
    seg_ends = polyline[1:]
    seg_vecs = seg_ends - seg_starts
    seg_len_sq = np.sum(seg_vecs * seg_vecs, axis=1)
    seg_len_sq[seg_len_sq == 0.0] = 1e-12
    min_dists = np.full(len(points), np.inf, dtype=np.float32)
    for start, vec, length_sq in zip(seg_starts, seg_vecs, seg_len_sq):
        rel = points - start
        t = np.clip(np.sum(rel * vec, axis=1) / length_sq, 0.0, 1.0)
        closest = start + t[:, None] * vec
        min_dists = np.minimum(min_dists, np.linalg.norm(points - closest, axis=1))
    return min_dists


def _resample_by_arclength(path: np.ndarray, spacing: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return path.copy()
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(np.sum(seg_lens))
    if not np.isfinite(total) or total <= 1e-6:
        return path[:1].copy()
    targets = np.arange(0.0, total + spacing, spacing, dtype=np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    samples = []
    seg_idx = 0
    for target in targets:
        target = min(float(target), total)
        while seg_idx < len(seg_lens) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        denom = max(float(seg_lens[seg_idx]), 1e-6)
        alpha = float((target - cumulative[seg_idx]) / denom)
        samples.append(path[seg_idx] + alpha * (path[seg_idx + 1] - path[seg_idx]))
    return np.asarray(samples, dtype=np.float32)


def compute_intent_safety_metrics(trajs, env, cfg) -> dict:
    done = trajs.get(("next", "done"))
    first_done = _first_done_indices(done)
    stats = trajs[("next", "stats")]
    positions = stats["debug_pos_world"].detach().cpu()
    target_vel = stats["debug_vec_target"].detach().cpu()
    success = stats["success"].detach().cpu().squeeze(-1)
    truncated = stats["truncated"].detach().cpu().squeeze(-1)
    dt = float(getattr(env, "dt", cfg.sim.dt * cfg.sim.substeps))
    tcr_thresholds = (1.0, 2.0, 5.0)
    tcr_values = {threshold: [] for threshold in tcr_thresholds}
    cte_values = []

    lidar = trajs.get(("next", "agents", "observation", "lidar"), None)
    dmin_values = []
    if lidar is not None:
        lidar_scan = lidar.detach().cpu()
        lidar_range = float(getattr(env, "lidar_range", cfg.sensor.lidar_range))
        dmin = lidar_range * (1.0 - lidar_scan.flatten(start_dim=2).amax(dim=-1))
    else:
        dmin = None

    timeout_values = []
    for env_idx, end_idx_tensor in enumerate(first_done):
        end_idx = int(end_idx_tensor.item())
        pos_ep = positions[env_idx, : end_idx + 1].numpy()
        vel_ep = target_vel[env_idx, : end_idx + 1].numpy()
        if len(pos_ep) >= 2:
            reference = np.empty_like(pos_ep, dtype=np.float32)
            reference[0] = pos_ep[0]
            for idx in range(1, len(pos_ep)):
                reference[idx] = reference[idx - 1] + vel_ep[idx - 1] * dt
            sampled_reference = _resample_by_arclength(reference, spacing=0.5)
            ref_to_actual = _point_to_polyline_dist(sampled_reference, pos_ep)
            for threshold in tcr_thresholds:
                tcr_values[threshold].append(float(np.nanmean(ref_to_actual < threshold)))
            actual_to_ref = _point_to_polyline_dist(pos_ep, reference)
            cte_values.append(float(np.nanmean(actual_to_ref)))
        if dmin is not None:
            dmin_values.append(dmin[env_idx, : end_idx + 1])
        timeout_values.append(
            bool(truncated[env_idx, end_idx]) and not bool(success[env_idx, end_idx])
        )

    info = {
        "eval/timeout": float(np.mean(timeout_values)) if timeout_values else math.nan,
        "eval/cte": float(np.nanmean(cte_values)) if cte_values else math.nan,
    }
    for threshold in tcr_thresholds:
        info[f"eval/tcr_at_{int(threshold)}"] = (
            float(np.nanmean(tcr_values[threshold])) if tcr_values[threshold] else math.nan
        )

    if dmin_values:
        dmin_all = torch.cat([value.reshape(-1) for value in dmin_values])
        info["eval/dmin_min"] = float(torch.min(dmin_all).item())
        info["eval/dmin_mean"] = float(torch.mean(dmin_all.float()).item())
        info["eval/crr_0.5m"] = float(torch.mean((dmin_all < 0.5).float()).item())
        info["eval/crr_1.0m"] = float(torch.mean((dmin_all < 1.0).float()).item())
    else:
        info["eval/dmin_min"] = math.nan
        info["eval/dmin_mean"] = math.nan
        info["eval/crr_0.5m"] = math.nan
        info["eval/crr_1.0m"] = math.nan
    return info


def save_video(frames: list, path: str, fps: int) -> None:
    arr = np.stack(frames)  # (T, H, W, C)
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[3] not in (1, 3):
        # got (T, C, H, W) -> rearrange
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    imageio.mimsave(path, arr, fps=fps, macro_block_size=1)
    print(f"[EvalVideo] Saved {arr.shape[0]} frames @ {fps} fps -> {path}")


if __name__ == "__main__":
    main()
