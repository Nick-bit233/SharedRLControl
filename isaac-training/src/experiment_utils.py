from __future__ import annotations

import os

import torch


def resolve_constrained_policy(cfg):
    """Resolve the active constrained residual PPO implementation."""
    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    if algo_distribution == "beta":
        from src.algos.ppo_constrained_beta import (
            ConstrainedResidualPPO_Beta as ConstrainedResidualPPO,
        )

        return ConstrainedResidualPPO, "Beta distribution PPO"

    from src.algos.ppo_constrained import ConstrainedResidualPPO

    return ConstrainedResidualPPO, "TanhNormal distribution PPO"


def load_trajectory_dataset(cfg):
    """Load the offline trajectory dataset when enabled by config."""
    if not cfg.user_model.get("offline_mode", False):
        return None

    from src.datasets.trajectory_dataset import TrajectoryDataset

    dataset_path = cfg.user_model.get("dataset_path", None)
    if dataset_path is None:
        raise ValueError("user_model.dataset_path must be set when offline_mode=True")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Trajectory dataset not found: {dataset_path}")

    print(f"[Train] Loading trajectory dataset from: {dataset_path}")
    dataset = TrajectoryDataset(
        dataset_path=dataset_path,
        device=torch.device(cfg.device),
        gpu_cache_reserve_gb=cfg.user_model.get("gpu_cache_reserve_gb", 2.0),
        min_scale_factor=cfg.user_model.get("min_scale_factor", 0.5),
        preload_data=cfg.user_model.get("preload_data", True),
    )
    print("[Train] Trajectory dataset loaded successfully")
    return dataset
