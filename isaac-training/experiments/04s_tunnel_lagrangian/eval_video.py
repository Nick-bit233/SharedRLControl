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

    # Lagrangian best checkpoint with the same env config as training
    python experiments/04s_tunnel_lagrangian/eval_video.py \\
        experiment=tunnel_lagrangian \\
        +resume_checkpoint=outputs/tunnel_lagrangian/.../checkpoint_best.pt

    # Stage checkpoint with matching stage config + global view + fewer envs
    python experiments/04s_tunnel_lagrangian/eval_video.py \\
        experiment=tunnel_lagrangian_stage2 \\
        +resume_checkpoint=outputs/lagrangian_curriculum_stage2/.../checkpoint_best.pt \\
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
    from src.envs.env_tunnel_lagrangian import EnvTunnelLagrangian
    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    if algo_distribution != "beta":
        raise ValueError("04s_tunnel_lagrangian eval requires algo.distribution=beta")
    from src.algos.ppo_constrained_beta_lagrangian import (
        ConstrainedResidualPPO_BetaLagrangian as ConstrainedResidualPPO,
    )

    # ---------------- output dir ----------------
    from hydra.core.hydra_config import HydraConfig
    hydra_run_dir = HydraConfig.get().runtime.output_dir
    video_dir = cfg.get("video_dir", None) or hydra_run_dir
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
        print("[EvalVideo] User model source: M2 offline tunnel dataset via UserModelTunnel")
        trajectory_dataset = TrajectoryDataset(
            dataset_path=dataset_path,
            device=torch.device(cfg.device),
            gpu_cache_reserve_gb=cfg.user_model.get("gpu_cache_reserve_gb", 2.0),
            min_scale_factor=cfg.user_model.get("min_scale_factor", 0.5),
            preload_data=cfg.user_model.get("preload_data", True),
        )
    else:
        print("[EvalVideo] User model source: legacy online UserModelTunnel")

    # ---------------- env + policy ----------------
    env = EnvTunnelLagrangian(cfg, trajectory_dataset=trajectory_dataset)
    policy = ConstrainedResidualPPO(
        cfg.algo, env.observation_spec, env.action_spec, cfg.device
    )

    loaded = torch.load(resume_ckpt, map_location=cfg.device)
    state_dict = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        policy_keys = set(policy.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = policy_keys - loaded_keys
        unexpected_keys = loaded_keys - policy_keys
        allowed_missing = {"lambda_lag"}
        if missing_keys <= allowed_missing and not unexpected_keys:
            policy.load_state_dict(state_dict, strict=False)
            print("[EvalVideo] Loaded baseline Beta checkpoint without lambda_lag; using configured lambda_init.")
        else:
            raise exc
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
