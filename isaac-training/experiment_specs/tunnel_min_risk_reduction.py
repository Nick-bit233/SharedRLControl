"""Tunnel minimum-risk-reduction experiment spec."""

from __future__ import annotations

from typing import Any

from src.core.spec import ExperimentSpec, RuntimeResources
from experiment_specs.tunnel import (
    ResidualPolicySanityCheckHook,
    load_trajectory_dataset,
    make_policy,
)


def make_env(cfg: Any, resources: RuntimeResources) -> Any:
    """Build the tunnel environment with dynamic risk enabled by config."""

    from src.envs.env_tunnel import EnvTunnelResidual

    return EnvTunnelResidual(cfg, trajectory_dataset=resources.dataset)


def build_spec(_cfg: Any | None = None) -> ExperimentSpec:
    """Build the minimum-risk-reduction experiment contract."""

    return ExperimentSpec(
        name="tunnel_min_risk_reduction",
        env_factory=make_env,
        policy_factory=make_policy,
        dataset_loader=load_trajectory_dataset,
        hooks=(ResidualPolicySanityCheckHook(),),
    )
