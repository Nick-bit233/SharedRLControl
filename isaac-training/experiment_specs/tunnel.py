"""Tunnel-style experiment spec."""

from __future__ import annotations

import os
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.core.curriculum import RegCoeffSchedulerHook
from src.core.spec import ExperimentSpec, RuntimeResources


class ResidualPolicySanityCheckHook:
    """Run the tunnel initialization check before training."""

    def on_before_training(self, context: dict[str, Any]) -> None:
        cfg = context["cfg"]
        env = context["env"]
        policy = context["policy"]

        print("[Sanity Check] Running Zero-Shot Verification...")
        env.eval()
        try:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = env.reset()
                policy(td)

                net_output_norm = td["agents", "action_normalized"]
                if cfg.algo.get("policy_mode", "residual") == "direct":
                    finite = torch.isfinite(net_output_norm).all()
                    within_bounds = (net_output_norm.abs() <= 1.0 + 1e-5).all()
                    print("[Sanity Check] Direct policy mode: identity mapping is not expected.")
                    print(
                        f"[Sanity Check] action_normalized finite={bool(finite)}, "
                        f"within_bounds={bool(within_bounds)}"
                    )
                    if finite and within_bounds:
                        print("[Sanity Check] Initialization SUCCESS: direct policy emits finite bounded actions.")
                    else:
                        print("[Sanity Check] Initialization WARNING: direct policy emitted invalid action sample.")
                        print(f"   Sample Net Out: {net_output_norm[0]}")
                    return

                human_input_phys = td["agents", "observation", "human_action"]
                human_input_norm = human_input_phys / cfg.algo.actor.action_limit
                diff = (net_output_norm - human_input_norm).norm(dim=-1).mean()

                print(f"[Sanity Check] Initial Mean Error (Norm Space): {diff.item():.6f}")
                if diff.item() < 1e-2:
                    print("[Sanity Check] Initialization SUCCESS: network starts as identity mapping.")
                else:
                    print(f"[Sanity Check] Initialization WARNING: initial error is large ({diff.item()}).")
                    print(f"   Sample Net Out: {net_output_norm[0]}")
                    print(f"   Sample Human In: {human_input_norm[0]}")
        finally:
            env.train()
            env.reset()


def load_trajectory_dataset(cfg: Any, _hydra_cfg: Any) -> Any | None:
    """Load the offline trajectory dataset used by tunnel-style user models."""

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


def make_env(cfg: Any, resources: RuntimeResources) -> Any:
    """Build the tunnel task environment."""

    env_name = cfg.env.get("name", "tunnel")
    if env_name == "real_room":
        from src.envs.env_real_room import EnvRealRoomResidual as EnvClass
    elif env_name == "tunnel":
        from src.envs.env_tunnel import EnvTunnelResidual as EnvClass
    else:
        raise ValueError(f"Unknown tunnel spec env.name: {env_name}")

    return EnvClass(cfg, trajectory_dataset=resources.dataset)


def make_policy(cfg: Any, env: Any) -> Any:
    """Build the constrained residual PPO policy."""

    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    algo_policy_mode = cfg.algo.get("policy_mode", "residual")
    if algo_distribution == "beta":
        from src.algos.ppo_constrained_beta import ConstrainedResidualPPO_Beta as ConstrainedResidualPPO

        print(f"[Train] Using Beta distribution PPO ({algo_policy_mode} policy mode)")
    else:
        from src.algos.ppo_constrained import ConstrainedResidualPPO

        print("[Train] Using TanhNormal distribution PPO")

    return ConstrainedResidualPPO(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        cfg.device,
    )


def build_spec(_cfg: Any | None = None) -> ExperimentSpec:
    """Build the experiment contract consumed by the shared runner."""

    return ExperimentSpec(
        name="tunnel",
        env_factory=make_env,
        policy_factory=make_policy,
        dataset_loader=load_trajectory_dataset,
        hooks=(
            ResidualPolicySanityCheckHook(),
            RegCoeffSchedulerHook(),
        ),
    )

