"""Checkpoint helpers for the shared training runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.core.spec import CheckpointAdapter, CheckpointState, DefaultCheckpointAdapter
from src.core.wandb_utils import get_run_dir


RICH_CHECKPOINT_KEYS = (
    "actor_optim",
    "critic_optim",
    "feature_extractor_optim",
    "iter",
    "last_completed_iter",
    "env_frames",
)

RICH_PROGRESS_KEYS = ("iter", "last_completed_iter")

CHECKPOINT_FORMAT_RICH = "rich"
CHECKPOINT_FORMAT_POLICY_WRAPPED = "policy_wrapped"
CHECKPOINT_FORMAT_WEIGHTS_ONLY = "weights_only"


def _torch() -> Any:
    import torch

    return torch


@dataclass
class BestCheckpointState:
    """Mutable best-eval tracking state used by the runner."""

    eval_score: float = -float("inf")
    eval_success: float = -1.0
    eval_collision: float = 1.0
    eval_rank: tuple[float, ...] | None = None
    eval_info: dict[str, Any] | None = None
    policy_state: dict[str, Any] | None = None
    checkpoint_path: str | None = None


@dataclass
class CheckpointPayload:
    """Loaded checkpoint split into policy weights and optional rich state."""

    policy_state: Mapping[str, Any]
    rich_state: Mapping[str, Any] | None = None
    raw: Any = None
    format: str = CHECKPOINT_FORMAT_WEIGHTS_ONLY


@dataclass
class CheckpointPaths:
    """Paths written by checkpoint save helpers."""

    checkpoint_path: str
    marker_paths: dict[str, str] = field(default_factory=dict)


def resolve_runtime_path(path: str | None, hydra_cfg: Any | None = None) -> str | None:
    """Resolve a user path relative to Hydra's original working directory."""

    if path is None or os.path.isabs(path):
        return path
    if hydra_cfg is not None and hasattr(hydra_cfg, "runtime"):
        cwd = getattr(hydra_cfg.runtime, "cwd", None)
        if cwd:
            return os.path.abspath(os.path.join(cwd, path))
    return os.path.abspath(path)


def is_rich_checkpoint(loaded: Any) -> bool:
    """Return True when a checkpoint contains policy plus training state."""

    return (
        isinstance(loaded, Mapping)
        and "policy" in loaded
        and "env_frames" in loaded
        and any(key in loaded for key in RICH_PROGRESS_KEYS)
    )


def is_policy_wrapped_checkpoint(loaded: Any) -> bool:
    """Return True for ``{"policy": state_dict}`` without training progress state."""

    return isinstance(loaded, Mapping) and "policy" in loaded and not is_rich_checkpoint(loaded)


def describe_checkpoint_format(format_name: str) -> str:
    """Return a human-readable checkpoint format description."""

    descriptions = {
        CHECKPOINT_FORMAT_RICH: (
            "rich checkpoint: policy weights plus training progress state "
            "(optimizer, iter/last_completed_iter, env_frames, and optional extras)"
        ),
        CHECKPOINT_FORMAT_POLICY_WRAPPED: (
            "legacy policy-wrapped checkpoint: {'policy': state_dict} without optimizer/progress state"
        ),
        CHECKPOINT_FORMAT_WEIGHTS_ONLY: (
            "legacy weights-only checkpoint: a bare policy state_dict without optimizer/progress state"
        ),
    }
    return descriptions.get(format_name, f"unknown checkpoint format: {format_name}")


def split_loaded_checkpoint(loaded: Any) -> CheckpointPayload:
    """Split torch-loaded data into policy weights and optional rich state.

    Supported formats:
    - rich: ``{"policy": state_dict, "iter"/"last_completed_iter": ..., ...}``
    - legacy policy-wrapped: ``{"policy": state_dict}``
    - legacy weights-only: a bare ``policy.state_dict()``

    Legacy formats are accepted for ``init_checkpoint`` warm-starts only.  They
    are intentionally not considered valid ``resume_checkpoint`` inputs because
    they cannot restore optimizer state, iteration counters, curriculum state,
    or other experiment-specific training progress.
    """

    if is_rich_checkpoint(loaded):
        return CheckpointPayload(
            policy_state=loaded["policy"],
            rich_state=loaded,
            raw=loaded,
            format=CHECKPOINT_FORMAT_RICH,
        )
    if is_policy_wrapped_checkpoint(loaded):
        return CheckpointPayload(
            policy_state=loaded["policy"],
            rich_state=None,
            raw=loaded,
            format=CHECKPOINT_FORMAT_POLICY_WRAPPED,
        )
    return CheckpointPayload(
        policy_state=loaded,
        rich_state=None,
        raw=loaded,
        format=CHECKPOINT_FORMAT_WEIGHTS_ONLY,
    )


def load_checkpoint(path: str, *, map_location: Any = None) -> CheckpointPayload:
    """Load a checkpoint file and split it into policy/rich state."""

    torch = _torch()
    loaded = torch.load(path, map_location=map_location)
    return split_loaded_checkpoint(loaded)


def get_init_resume_paths(cfg: Any, hydra_cfg: Any | None = None) -> tuple[str | None, str | None]:
    """Return resolved ``init_checkpoint`` and ``resume_checkpoint`` paths."""

    init_ckpt = resolve_runtime_path(cfg.get("init_checkpoint", None), hydra_cfg)
    resume_ckpt = resolve_runtime_path(cfg.get("resume_checkpoint", None), hydra_cfg)
    if init_ckpt is not None and resume_ckpt is not None:
        raise ValueError("Use only one of resume_checkpoint or init_checkpoint")
    return init_ckpt, resume_ckpt


def load_init_or_resume_checkpoint(
    cfg: Any,
    policy: Any,
    *,
    hydra_cfg: Any | None = None,
    adapter: CheckpointAdapter | None = None,
) -> CheckpointState:
    """Apply init/resume checkpoint settings and return restored training state.

    ``init_checkpoint`` means warm-starting a new run from policy weights only.
    It accepts rich checkpoints and deprecated legacy formats, but ignores all
    optimizer/progress state.

    ``resume_checkpoint`` means continuing an interrupted run.  It requires a
    rich checkpoint because a true resume must restore policy weights,
    optimizers, iteration counters, env frame counters, and optional extra state.
    Deprecated legacy formats must be passed through ``init_checkpoint`` instead.
    """

    adapter = adapter or DefaultCheckpointAdapter()
    init_ckpt, resume_ckpt = get_init_resume_paths(cfg, hydra_cfg)

    if init_ckpt is not None:
        print(f"[Train] Initializing policy from checkpoint: {init_ckpt}")
        payload = load_checkpoint(init_ckpt, map_location=cfg.device)
        print(f"[Train] Detected {describe_checkpoint_format(payload.format)}.")
        if payload.format != CHECKPOINT_FORMAT_RICH:
            print(
                "[Train] WARNING: legacy checkpoint formats are deprecated. "
                "They remain supported for init_checkpoint warm-starts only."
            )
        adapter.load_policy_state(policy, payload.policy_state, cfg)
        print("[Train] Init checkpoint loaded as policy weights only.")
        return CheckpointState()

    if resume_ckpt is None:
        return CheckpointState()

    print(f"[Train] Loading checkpoint: {resume_ckpt}")
    payload = load_checkpoint(resume_ckpt, map_location=cfg.device)
    print(f"[Train] Detected {describe_checkpoint_format(payload.format)}.")
    if payload.format != CHECKPOINT_FORMAT_RICH:
        raise ValueError(
            "resume_checkpoint requires a rich checkpoint with optimizer/progress state. "
            f"Got {payload.format}. Use init_checkpoint for legacy policy warm-starts, "
            "or resume from a checkpoint produced by the shared runtime."
        )
    adapter.load_policy_state(policy, payload.policy_state, cfg)
    assert payload.rich_state is not None
    checkpoint_state = adapter.restore_training_state(policy, payload.rich_state, cfg)
    last_completed_iter = checkpoint_state.start_iter - 1
    print("[Train] Optimizer states restored from rich checkpoint.")
    print(
        f"[Train] Resuming after iter={last_completed_iter}; "
        f"next_iter={checkpoint_state.start_iter}, "
        f"env_frames={checkpoint_state.start_env_frames}"
    )
    print("[Train] Checkpoint loaded successfully.")
    return checkpoint_state


def load_policy_for_eval(
    path: str,
    policy: Any,
    cfg: Any,
    *,
    adapter: CheckpointAdapter | None = None,
    hydra_cfg: Any | None = None,
) -> CheckpointPayload:
    """Load policy weights for evaluation without restoring training state."""

    resolved_path = resolve_runtime_path(path, hydra_cfg)
    if resolved_path is None:
        raise ValueError("Evaluation checkpoint path must be set.")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found: {resolved_path}")

    adapter = adapter or DefaultCheckpointAdapter()
    print(f"[Eval] Loading checkpoint: {resolved_path}")
    payload = load_checkpoint(resolved_path, map_location=cfg.device)
    print(f"[Eval] Detected {describe_checkpoint_format(payload.format)}.")
    adapter.load_policy_state(policy, payload.policy_state, cfg)
    print("[Eval] Policy checkpoint loaded.")
    return payload


def remaining_iterations_after_resume(
    *,
    max_iterations: int,
    start_iter: int,
    profiling_mode: bool = False,
) -> int | None:
    """Return remaining iterations when resuming toward an absolute target."""

    if start_iter <= 0:
        return None
    if profiling_mode:
        return 0
    remaining = max(0, max_iterations - start_iter)
    if remaining <= 0:
        raise ValueError(
            f"resume_checkpoint already reached max_iterations: "
            f"next_iter={start_iter}, max_iterations={max_iterations}. "
            "Increase max_iterations or use init_checkpoint for a new stage."
        )
    return remaining


def apply_resume_frame_budget(
    cfg: Any,
    *,
    max_iterations: int,
    start_iter: int,
    profiling_mode: bool = False,
) -> int | None:
    """Return resumed collector frame budget, or None when unchanged."""

    remaining = remaining_iterations_after_resume(
        max_iterations=max_iterations,
        start_iter=start_iter,
        profiling_mode=profiling_mode,
    )
    if remaining is None or profiling_mode:
        return None
    total_frames = cfg.algo.training_frame_num * cfg.env.num_envs * remaining
    print(f"[Train] Resuming: shrinking collector budget to {remaining} more iters.")
    return total_frames


def clone_policy_state(policy: Any) -> dict[str, Any]:
    """Clone a policy state_dict for in-memory best-checkpoint tracking."""

    return {key: value.clone() for key, value in policy.state_dict().items()}


def build_rich_checkpoint(
    policy: Any,
    *,
    iter_value: int,
    env_frames: int,
    cfg: Any,
    adapter: CheckpointAdapter | None = None,
    policy_state: Mapping[str, Any] | None = None,
    best_state: BestCheckpointState | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a rich checkpoint dictionary from common and adapter state."""

    adapter = adapter or DefaultCheckpointAdapter()
    rich: dict[str, Any] = {
        "policy": policy.state_dict() if policy_state is None else policy_state,
        "iter": iter_value,
        "last_completed_iter": iter_value,
        "env_frames": env_frames,
    }

    if best_state is not None:
        rich.update(
            {
                "best_eval_score": best_state.eval_score,
                "best_eval_success": best_state.eval_success,
                "best_eval_collision": best_state.eval_collision,
            }
        )
        if best_state.eval_info is not None:
            rich["best_eval_info"] = best_state.eval_info

    rich.update(adapter.snapshot_training_state(policy, cfg))
    if extra_state:
        rich.update(dict(extra_state))
    return rich


def write_marker(marker_path: str, checkpoint_path: str) -> None:
    """Write a marker file containing a checkpoint path."""

    marker_dir = os.path.dirname(marker_path)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    with open(marker_path, "w") as file:
        file.write(checkpoint_path)


def save_torch_object(obj: Any, path: str) -> str:
    """Save an object with torch and return the path."""

    torch = _torch()
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, path)
    return path


def save_latest_checkpoint(
    run: Any,
    cfg: Any,
    policy: Any,
    *,
    iter_value: int,
    env_frames: int,
    adapter: CheckpointAdapter | None = None,
    best_state: BestCheckpointState | None = None,
    extra_state: Mapping[str, Any] | None = None,
    filename: str | None = None,
) -> CheckpointPaths:
    """Save a periodic rich checkpoint and write ``latest_checkpoint_path.txt``."""

    save_dir = get_run_dir(run, cfg.log_output_dir)
    checkpoint_path = os.path.join(save_dir, filename or f"checkpoint_{iter_value}.pt")
    rich = build_rich_checkpoint(
        policy,
        iter_value=iter_value,
        env_frames=env_frames,
        cfg=cfg,
        adapter=adapter,
        best_state=best_state,
        extra_state=extra_state,
    )
    save_torch_object(rich, checkpoint_path)
    marker_path = os.path.join(cfg.log_output_dir, "latest_checkpoint_path.txt")
    write_marker(marker_path, checkpoint_path)
    print("[RunnerSimple]: model saved at training step: ", iter_value)
    return CheckpointPaths(checkpoint_path=checkpoint_path, marker_paths={"latest": marker_path})


def save_final_checkpoint(
    run: Any,
    cfg: Any,
    policy: Any,
    *,
    iter_value: int,
    env_frames: int,
    adapter: CheckpointAdapter | None = None,
    best_state: BestCheckpointState | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> CheckpointPaths:
    """Save final rich checkpoint and write ``final_checkpoint_path.txt``."""

    save_dir = get_run_dir(run, cfg.log_output_dir)
    checkpoint_path = os.path.join(save_dir, "checkpoint_final.pt")
    rich = build_rich_checkpoint(
        policy,
        iter_value=iter_value,
        env_frames=env_frames,
        cfg=cfg,
        adapter=adapter,
        best_state=best_state,
        extra_state=extra_state,
    )
    save_torch_object(rich, checkpoint_path)
    marker_path = os.path.join(cfg.log_output_dir, "final_checkpoint_path.txt")
    write_marker(marker_path, checkpoint_path)
    print(f"[Train] Final checkpoint saved: {checkpoint_path}")
    return CheckpointPaths(checkpoint_path=checkpoint_path, marker_paths={"final": marker_path})


def save_best_checkpoint(
    run: Any,
    cfg: Any,
    *,
    best_state: BestCheckpointState,
    fallback_checkpoint_path: str,
    policy: Any | None = None,
    iter_value: int | None = None,
    env_frames: int | None = None,
    adapter: CheckpointAdapter | None = None,
    extra_state: Mapping[str, Any] | None = None,
    policy_only: bool = True,
) -> CheckpointPaths:
    """Save best checkpoint marker, falling back to final when no eval ran."""

    marker_path = os.path.join(cfg.log_output_dir, "best_checkpoint_path.txt")
    if best_state.policy_state is None:
        write_marker(marker_path, fallback_checkpoint_path)
        print("[Train] No eval run; best checkpoint marker points to final.")
        return CheckpointPaths(
            checkpoint_path=fallback_checkpoint_path,
            marker_paths={"best": marker_path},
        )

    save_dir = get_run_dir(run, cfg.log_output_dir)
    checkpoint_path = best_state.checkpoint_path or os.path.join(save_dir, "checkpoint_best.pt")
    if policy_only:
        save_torch_object(best_state.policy_state, checkpoint_path)
    else:
        if policy is None or iter_value is None or env_frames is None:
            raise ValueError("policy, iter_value, and env_frames are required for rich best checkpoints")
        rich = build_rich_checkpoint(
            policy,
            iter_value=iter_value,
            env_frames=env_frames,
            cfg=cfg,
            adapter=adapter,
            policy_state=best_state.policy_state,
            best_state=best_state,
            extra_state=extra_state,
        )
        save_torch_object(rich, checkpoint_path)

    best_state.checkpoint_path = checkpoint_path
    write_marker(marker_path, checkpoint_path)
    print(
        f"[Train] Best checkpoint saved: {checkpoint_path} "
        f"(score={best_state.eval_score:.3f}, success={best_state.eval_success:.3f}, "
        f"collision={best_state.eval_collision:.3f})"
    )
    return CheckpointPaths(checkpoint_path=checkpoint_path, marker_paths={"best": marker_path})


__all__ = [
    "BestCheckpointState",
    "CHECKPOINT_FORMAT_POLICY_WRAPPED",
    "CHECKPOINT_FORMAT_RICH",
    "CHECKPOINT_FORMAT_WEIGHTS_ONLY",
    "CheckpointPayload",
    "CheckpointPaths",
    "RICH_CHECKPOINT_KEYS",
    "RICH_PROGRESS_KEYS",
    "apply_resume_frame_budget",
    "build_rich_checkpoint",
    "clone_policy_state",
    "describe_checkpoint_format",
    "get_init_resume_paths",
    "is_policy_wrapped_checkpoint",
    "is_rich_checkpoint",
    "load_checkpoint",
    "load_init_or_resume_checkpoint",
    "load_policy_for_eval",
    "remaining_iterations_after_resume",
    "resolve_runtime_path",
    "save_best_checkpoint",
    "save_final_checkpoint",
    "save_latest_checkpoint",
    "save_torch_object",
    "split_loaded_checkpoint",
    "write_marker",
]
