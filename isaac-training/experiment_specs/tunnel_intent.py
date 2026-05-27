"""Tunnel-intent experiment spec."""

from __future__ import annotations

from typing import Any

from src.core.curriculum import RegCoeffSchedulerHook
from src.core.spec import ExperimentSpec, RuntimeResources

from experiment_specs.tunnel import load_trajectory_dataset, make_policy


def make_env(cfg: Any, resources: RuntimeResources) -> Any:
    from src.envs.env_tunnel_intent import EnvTunnelIntent

    return EnvTunnelIntent(cfg, trajectory_dataset=resources.dataset)


def build_spec(_cfg: Any | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        name="tunnel_intent",
        env_factory=make_env,
        policy_factory=make_policy,
        dataset_loader=load_trajectory_dataset,
        hooks=(RegCoeffSchedulerHook(),),
    )
