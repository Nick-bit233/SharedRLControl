"""
spec.py
Runtime extension points for experiment training entrypoints.

The core runner should know how to run a training loop, but it should not know
which environment, policy, checkpoint extras, or evaluation ranking rule belongs
to a specific experiment.  This module defines the small contract between thin
``experiments/*/train.py`` wrappers and ``src.core.runner``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


MetricValue = int | float | bool | str | None
InfoDict = dict[str, Any]
SerializableInfo = dict[str, MetricValue]


@dataclass
class RuntimeResources:
    """Objects prepared by the runner and shared with factories/hooks."""

    hydra_cfg: Any | None = None
    dataset: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSummary:
    """Comparable evaluation summary used for best-checkpoint selection."""

    score: float
    rank: tuple[float, ...]
    success: float = 0.0
    collision: float = 0.0
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass
class CheckpointState:
    """State restored from a checkpoint before the main training loop starts."""

    resume_state: Mapping[str, Any] | None = None
    start_iter: int = 0
    start_env_frames: int = 0


class EnvFactory(Protocol):
    """Build an environment after Isaac Sim has been initialized."""

    def __call__(self, cfg: Any, resources: RuntimeResources) -> Any:
        ...


class PolicyFactory(Protocol):
    """Build a policy from the config and constructed environment."""

    def __call__(self, cfg: Any, env: Any) -> Any:
        ...


class DatasetLoader(Protocol):
    """Load optional experiment resources, such as offline trajectories."""

    def __call__(self, cfg: Any, hydra_cfg: Any) -> Any | None:
        ...


class EvalSummaryFn(Protocol):
    """Convert raw eval info into a comparable best-checkpoint summary."""

    def __call__(self, eval_info: Mapping[str, Any]) -> EvalSummary:
        ...


class SanityCheckFn(Protocol):
    """Run experiment-specific initialization checks before training."""

    def __call__(self, cfg: Any, env: Any, policy: Any) -> None:
        ...


class CheckpointAdapter(Protocol):
    """Experiment-specific checkpoint state load/save extension points."""

    def load_policy_state(self, policy: Any, policy_state: Mapping[str, Any], cfg: Any) -> None:
        ...

    def restore_training_state(
        self,
        policy: Any,
        checkpoint: Mapping[str, Any],
        cfg: Any,
    ) -> CheckpointState:
        ...

    def snapshot_training_state(
        self,
        policy: Any,
        cfg: Any,
    ) -> dict[str, Any]:
        ...


class RuntimeHook(Protocol):
    """Optional lifecycle hooks called by the shared runner.

    Hook methods receive a mutable context dictionary so experiments can exchange
    small pieces of state with the runner without forcing the runner to know the
    experiment name.  Implement only the methods a hook needs.

    Recognized hook names:
    - ``on_after_setup``
    - ``on_before_training``
    - ``on_before_train_step``
    - ``on_after_train_step``
    - ``on_before_eval``
    - ``on_after_eval``
    - ``on_before_checkpoint``
    - ``on_after_checkpoint``
    - ``on_after_training``

    Hooks that need checkpoint persistence can write serializable values into
    ``context["checkpoint_extra_state"]`` before checkpoint hooks return.
    """


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of one experiment entrypoint.

    A train.py wrapper should select env/policy classes, create an
    ExperimentSpec, then call ``run_training(cfg, spec)``.  The shared runner
    owns the loop; this spec owns the experiment-specific choices.

    ``sanity_check_fn`` is accepted for compatibility with older wrappers, but
    the shared runner no longer invokes it directly. New experiment-specific
    checks should be implemented as lifecycle hooks.
    """

    name: str
    env_factory: EnvFactory
    policy_factory: PolicyFactory
    dataset_loader: DatasetLoader | None = None
    eval_summary_fn: EvalSummaryFn | None = None
    checkpoint_adapter: CheckpointAdapter | None = None
    sanity_check_fn: SanityCheckFn | None = None
    hooks: Sequence[RuntimeHook] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExperimentSpec.name must be non-empty")


def get_eval_metric(eval_info: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read current eval keys while remaining compatible with old stats-prefixed logs."""

    for key in (f"eval/{name}", f"eval/stats_{name}"):
        if key in eval_info:
            return eval_info[key]
    return default


def serializable_eval_info(eval_info: Mapping[str, Any]) -> SerializableInfo:
    """Keep scalar eval entries suitable for checkpoint metadata."""

    clean: SerializableInfo = {}
    for key, value in eval_info.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            clean[key] = value
    return clean


def default_eval_summary(eval_info: Mapping[str, Any]) -> EvalSummary:
    """Default best-model ranking used by the tunnel-style experiments."""

    success = float(get_eval_metric(eval_info, "success", 0.0))
    collision = float(get_eval_metric(eval_info, "collision", 1.0))
    timeout = float(get_eval_metric(eval_info, "timeout", 0.0))
    above_bound = float(get_eval_metric(eval_info, "above_bound", 0.0))
    below_bound = float(get_eval_metric(eval_info, "below_bound", 0.0))
    task_reward = float(get_eval_metric(eval_info, "diag_reward_task", 0.0))
    eval_return = float(get_eval_metric(eval_info, "return", 0.0))

    score = success - 0.5 * collision - 0.2 * timeout - 0.1 * above_bound - 0.1 * below_bound
    rank = (score, success, -collision, -timeout, task_reward, eval_return)

    return EvalSummary(
        score=score,
        rank=rank,
        success=success,
        collision=collision,
        metrics={
            "timeout": timeout,
            "above_bound": above_bound,
            "below_bound": below_bound,
            "task_reward": task_reward,
            "return": eval_return,
        },
    )


class DefaultCheckpointAdapter:
    """Default checkpoint behavior for policies with standard optimizers."""

    def load_policy_state(self, policy: Any, policy_state: Mapping[str, Any], cfg: Any) -> None:
        policy.load_state_dict(policy_state)

    def restore_training_state(
        self,
        policy: Any,
        checkpoint: Mapping[str, Any],
        cfg: Any,
    ) -> CheckpointState:
        try:
            if "actor_optim" in checkpoint:
                policy.actor_optim.load_state_dict(checkpoint["actor_optim"])
            if "critic_optim" in checkpoint:
                policy.critic_optim.load_state_dict(checkpoint["critic_optim"])
            if "feature_extractor_optim" in checkpoint and hasattr(policy, "feature_extractor_optim"):
                policy.feature_extractor_optim.load_state_dict(checkpoint["feature_extractor_optim"])
        except Exception as exc:
            print(f"[Checkpoint] WARNING: optimizer restore failed: {exc}; continuing with fresh optimizer.")

        last_completed_iter = int(checkpoint.get("last_completed_iter", checkpoint.get("iter", -1)))
        return CheckpointState(
            resume_state=checkpoint,
            start_iter=max(0, last_completed_iter + 1),
            start_env_frames=int(checkpoint.get("env_frames", 0)),
        )

    def snapshot_training_state(
        self,
        policy: Any,
        cfg: Any,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {}
        try:
            state["actor_optim"] = policy.actor_optim.state_dict()
            state["critic_optim"] = policy.critic_optim.state_dict()
            if hasattr(policy, "feature_extractor_optim"):
                state["feature_extractor_optim"] = policy.feature_extractor_optim.state_dict()
        except Exception as exc:
            print(f"[Checkpoint] WARNING: failed to snapshot optimizer state: {exc}")
        return state


def call_hook(hook: RuntimeHook, name: str, context: MutableMapping[str, Any]) -> None:
    """Call a hook method only when the object implements it."""

    method = getattr(hook, name, None)
    if method is not None:
        method(context)


def call_hooks(hooks: Sequence[RuntimeHook], name: str, context: MutableMapping[str, Any]) -> None:
    """Call a named lifecycle hook on all registered hook objects."""

    for hook in hooks:
        call_hook(hook, name, context)


def as_hooks(hooks: Sequence[RuntimeHook] | RuntimeHook | None) -> tuple[RuntimeHook, ...]:
    """Normalize optional hook configuration into an immutable tuple."""

    if hooks is None:
        return ()
    if isinstance(hooks, Sequence) and not isinstance(hooks, (str, bytes)):
        return tuple(hooks)
    return (hooks,)


__all__ = [
    "CheckpointAdapter",
    "CheckpointState",
    "DatasetLoader",
    "DefaultCheckpointAdapter",
    "EnvFactory",
    "EvalSummary",
    "EvalSummaryFn",
    "ExperimentSpec",
    "InfoDict",
    "MetricValue",
    "PolicyFactory",
    "RuntimeHook",
    "RuntimeResources",
    "SanityCheckFn",
    "SerializableInfo",
    "as_hooks",
    "call_hook",
    "call_hooks",
    "default_eval_summary",
    "get_eval_metric",
    "serializable_eval_info",
]
