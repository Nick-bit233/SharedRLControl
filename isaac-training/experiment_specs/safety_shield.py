"""Safety-shield experiment spec."""

from __future__ import annotations

from typing import Any

from src.core.curriculum import RegCoeffSchedulerHook
from src.core.spec import ExperimentSpec, RuntimeResources

from experiment_specs.tunnel import make_policy


def make_env(cfg: Any, _resources: RuntimeResources) -> Any:
    from src.envs.env_safety_shield import EnvSafetyShield

    return EnvSafetyShield(cfg)


def build_spec(_cfg: Any | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        name="safety_shield",
        env_factory=make_env,
        policy_factory=make_policy,
        hooks=(RegCoeffSchedulerHook(),),
    )
