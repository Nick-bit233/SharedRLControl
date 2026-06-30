"""Tunnel Lagrangian experiment spec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.core.spec import (
    CheckpointState,
    DefaultCheckpointAdapter,
    EvalSummary,
    ExperimentSpec,
    RuntimeResources,
    default_eval_summary,
    get_eval_metric,
)
from experiment_specs.tunnel import ResidualPolicySanityCheckHook, load_trajectory_dataset


class LagrangianCheckpointAdapter(DefaultCheckpointAdapter):
    """Checkpoint adapter for PPO-Lagrangian lambda optimizer state."""

    def load_policy_state(self, policy: Any, policy_state: Mapping[str, Any], cfg: Any) -> None:
        try:
            policy.load_state_dict(policy_state)
        except RuntimeError as exc:
            policy_keys = set(policy.state_dict().keys())
            loaded_keys = set(policy_state.keys())
            missing_keys = policy_keys - loaded_keys
            unexpected_keys = loaded_keys - policy_keys
            allowed_missing = {"lambda_lag"}
            if missing_keys <= allowed_missing and not unexpected_keys:
                policy.load_state_dict(policy_state, strict=False)
                print("[Train] Loaded baseline Beta checkpoint without lambda_lag; keeping configured lambda_init.")
            else:
                raise exc

    def restore_training_state(
        self,
        policy: Any,
        checkpoint: Mapping[str, Any],
        cfg: Any,
    ) -> CheckpointState:
        state = super().restore_training_state(policy, checkpoint, cfg)
        try:
            if "lambda_optimizer" in checkpoint and hasattr(policy, "lambda_optimizer"):
                policy.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer"])
        except Exception as exc:
            print(f"[Checkpoint] WARNING: lambda optimizer restore failed: {exc}; continuing fresh.")
        return state

    def snapshot_training_state(
        self,
        policy: Any,
        cfg: Any,
    ) -> dict[str, Any]:
        state = super().snapshot_training_state(policy, cfg)
        try:
            if hasattr(policy, "lambda_optimizer"):
                state["lambda_optimizer"] = policy.lambda_optimizer.state_dict()
            if hasattr(policy, "lambda_lag"):
                state["lambda_lag_value"] = float(policy.lambda_lag.detach().cpu().item())
        except Exception as exc:
            print(f"[Checkpoint] WARNING: failed to snapshot lambda state: {exc}")
        return state


def lagrangian_eval_summary(eval_info: Mapping[str, Any]) -> EvalSummary:
    """Rank Lagrangian checkpoints by constrained success/safety metrics."""

    success = float(get_eval_metric(eval_info, "success", 0.0))
    collision = float(get_eval_metric(eval_info, "collision", 1.0))
    above_bound = float(get_eval_metric(eval_info, "above_bound", 0.0))
    below_bound = float(get_eval_metric(eval_info, "below_bound", 0.0))
    safety_cost = float(get_eval_metric(eval_info, "diag_safety_cost", 1.0))
    task_reward = float(get_eval_metric(eval_info, "diag_reward_task", 0.0))
    eval_return = float(get_eval_metric(eval_info, "return", 0.0))

    score = success - 0.5 * collision - 0.2 * above_bound - 0.2 * below_bound
    return EvalSummary(
        score=score,
        rank=(score, success, -collision, -safety_cost, task_reward, eval_return),
        success=success,
        collision=collision,
        metrics={
            "above_bound": above_bound,
            "below_bound": below_bound,
            "safety_cost": safety_cost,
            "task_reward": task_reward,
            "return": eval_return,
        },
    )


def make_env(cfg: Any, resources: RuntimeResources) -> Any:
    """Build the Lagrangian tunnel task environment."""

    from src.envs.env_tunnel_lagrangian import EnvTunnelLagrangian

    return EnvTunnelLagrangian(cfg, trajectory_dataset=resources.dataset)


def make_policy(cfg: Any, env: Any) -> Any:
    """Build the Beta PPO-Lagrangian policy."""

    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    if algo_distribution != "beta":
        raise ValueError("tunnel_lagrangian requires algo.distribution=beta")

    from src.algos.ppo_constrained_beta_lagrangian import (
        ConstrainedResidualPPO_BetaLagrangian as ConstrainedResidualPPO,
    )

    print("[Train] Using Beta PPO-Lagrangian")
    return ConstrainedResidualPPO(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        cfg.device,
    )


def build_spec(_cfg: Any | None = None) -> ExperimentSpec:
    """Build the Lagrangian experiment contract consumed by the shared runner."""

    return ExperimentSpec(
        name="tunnel_lagrangian",
        env_factory=make_env,
        policy_factory=make_policy,
        dataset_loader=load_trajectory_dataset,
        eval_summary_fn=lagrangian_eval_summary,
        checkpoint_adapter=LagrangianCheckpointAdapter(),
        hooks=(ResidualPolicySanityCheckHook(),),
        metadata={"base_eval_summary": default_eval_summary.__name__},
        best_checkpoint_policy_only=False,
    )

