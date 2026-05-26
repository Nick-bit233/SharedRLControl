"""Shared training runner for experiment entrypoints."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from src.core.checkpointing import (
    BestCheckpointState,
    apply_resume_frame_budget,
    clone_policy_state,
    load_init_or_resume_checkpoint,
    save_best_checkpoint,
    save_final_checkpoint,
    save_latest_checkpoint,
)
from src.core.collector import (
    collector_env_frames,
    collector_fps,
    make_collector,
    make_episode_stats,
    update_episode_stats,
)
from src.core.evaluation import evaluate_policy_no_grad
from src.core.spec import (
    CheckpointState,
    DefaultCheckpointAdapter,
    EvalSummary,
    ExperimentSpec,
    RuntimeResources,
    as_hooks,
    call_hooks,
    default_eval_summary,
    serializable_eval_info,
)
from src.core.wandb_utils import (
    disable_wandb_for_special_modes,
    finish_wandb,
    init_wandb_run,
    log_info,
)


@dataclass
class RuntimeSettings:
    """Resolved runtime settings derived from cfg."""

    profiling_mode: bool = False
    env_test_mode: bool = False
    profiling_batches: int = 10
    max_iterations: int = 20010
    max_frame_num: int = 0
    eval_interval: int = 500
    save_interval: int = 500
    profiler_log_file: str | None = None


@dataclass
class EarlyStoppingState:
    """Mutable early-stopping state."""

    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.10
    degradation_count: int = 0


@dataclass
class RunnerResult:
    """Summary returned by run_training."""

    final_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
    latest_checkpoint_path: str | None = None
    last_iter: int = -1
    env_frames: int = 0
    best_state: BestCheckpointState | None = None


def _resolve_settings(cfg: Any) -> RuntimeSettings:
    profiling_mode = cfg.get("profiling_mode", False)
    env_test_mode = cfg.get("env_test_mode", False)
    profiling_batches = cfg.get("profiling_batches", 10)

    disable_wandb_for_special_modes(
        cfg,
        profiling_mode=profiling_mode,
        env_test_mode=env_test_mode,
    )
    if env_test_mode:
        cfg.env.max_episode_length = 10

    if profiling_mode:
        print("[Train] === PROFILING MODE ENABLED ===")
        print(f"[Train] Will run {profiling_batches} batches for profiling analysis")
        max_iterations = profiling_batches
        max_frame_num = cfg.algo.training_frame_num * cfg.env.num_envs * profiling_batches
        eval_interval = 0
        save_interval = profiling_batches + 1
    else:
        max_iterations = cfg.get("max_iterations", 20010)
        max_frame_num = cfg.algo.training_frame_num * cfg.env.num_envs * max_iterations
        eval_interval = cfg.get("eval_interval", 500)
        save_interval = cfg.get("save_interval", 500)

    return RuntimeSettings(
        profiling_mode=profiling_mode,
        env_test_mode=env_test_mode,
        profiling_batches=profiling_batches,
        max_iterations=max_iterations,
        max_frame_num=max_frame_num,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )


def _configure_output_dir(cfg: Any) -> Any:
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    cfg.log_output_dir = hydra_cfg.runtime.output_dir
    return hydra_cfg


def _make_profiler(cfg: Any, settings: RuntimeSettings) -> Any:
    from src.core.profiler import get_profiler

    settings.profiler_log_file = (
        os.path.join(cfg.log_output_dir, "profiler.log")
        if settings.profiling_mode
        else None
    )
    return get_profiler(
        enabled=settings.profiling_mode,
        cuda_sync=True,
        device=cfg.device,
        log_file=settings.profiler_log_file,
    )


def _print_configuration(cfg: Any, settings: RuntimeSettings) -> None:
    if settings.env_test_mode:
        print("=== ENV TEST MODE ENABLED ===")
        print("This mode is for quick environment sanity checks. Training and evaluation will be skipped.")
    else:
        from omegaconf import OmegaConf

        print("=== CONFIGURATION ===")
        print(OmegaConf.to_yaml(cfg))


def _build_resources(cfg: Any, spec: ExperimentSpec, hydra_cfg: Any) -> RuntimeResources:
    dataset = spec.dataset_loader(cfg, hydra_cfg) if spec.dataset_loader is not None else None
    return RuntimeResources(hydra_cfg=hydra_cfg, dataset=dataset)


def _restore_best_state_from_resume(
    checkpoint_state: CheckpointState,
    eval_summary_fn: Any,
) -> BestCheckpointState:
    best_state = BestCheckpointState()
    resume_state = checkpoint_state.resume_state
    if resume_state is None:
        return best_state

    if "best_eval_score" in resume_state:
        best_state.eval_score = float(resume_state["best_eval_score"])
        print(f"[Train] best_eval_score carried over: {best_state.eval_score:.3f}")
    if "best_eval_success" in resume_state:
        best_state.eval_success = float(resume_state["best_eval_success"])
        print(f"[Train] best_eval_success carried over: {best_state.eval_success:.3f}")
    if "best_eval_collision" in resume_state:
        best_state.eval_collision = float(resume_state["best_eval_collision"])
        print(f"[Train] best_eval_collision carried over: {best_state.eval_collision:.3f}")
    if "best_eval_info" in resume_state:
        best_state.eval_info = dict(resume_state["best_eval_info"])
        try:
            summary = eval_summary_fn(best_state.eval_info)
            best_state.eval_rank = summary.rank
        except Exception as exc:
            print(f"[Train] WARNING: failed to rebuild best eval rank from checkpoint: {exc}")
    return best_state


def _make_early_stopping(cfg: Any) -> EarlyStoppingState:
    es_cfg = cfg.get("early_stopping", {})
    state = EarlyStoppingState(
        enabled=es_cfg.get("enable", False),
        patience=es_cfg.get("patience", 5),
        min_delta=es_cfg.get("min_delta", 0.10),
    )
    if state.enabled:
        print(
            f"[Train] Early stopping ENABLED: "
            f"patience={state.patience}, min_delta={state.min_delta}"
        )
    return state


def _make_curriculum_scheduler(cfg: Any, policy: Any, checkpoint_state: CheckpointState) -> Any | None:
    curriculum_enabled = cfg.get("curriculum", {}).get("enable", False)
    if not curriculum_enabled:
        return None

    from src.core.curriculum import RegCoeffScheduler

    reg_scheduler = RegCoeffScheduler(cfg.curriculum)
    print(
        f"[Train] Curriculum ENABLED: reg_coeff will ramp from "
        f"{cfg.curriculum.initial_reg_coeff} to {cfg.curriculum.max_reg_coeff}"
    )

    resume_state = checkpoint_state.resume_state
    if resume_state is not None and "reg_scheduler" in resume_state:
        try:
            reg_scheduler.load_state_dict(resume_state["reg_scheduler"])
            print(
                f"[Train] RegCoeffScheduler state restored: "
                f"reg_coeff={reg_scheduler.current_reg_coeff:.4f}, "
                f"ema_success={reg_scheduler.ema_success:.3f}"
            )
            policy.set_reg_coeff(reg_scheduler.current_reg_coeff)
        except Exception as exc:
            print(f"[Train] WARNING: curriculum restore failed: {exc}")
    return reg_scheduler


def _checkpoint_extra_state(reg_scheduler: Any | None) -> dict[str, Any]:
    if reg_scheduler is None:
        return {}
    return {"reg_scheduler": reg_scheduler.state_dict()}


def _update_best_state(
    best_state: BestCheckpointState,
    summary: EvalSummary,
    eval_info: dict[str, Any],
    policy: Any,
) -> bool:
    if best_state.eval_rank is not None and summary.rank <= best_state.eval_rank:
        return False

    best_state.eval_rank = summary.rank
    best_state.eval_score = summary.score
    best_state.eval_success = summary.success
    best_state.eval_collision = summary.collision
    best_state.eval_info = serializable_eval_info(eval_info)
    best_state.policy_state = clone_policy_state(policy)
    best_state.checkpoint_path = None
    return True


def _should_stop_for_early_stopping(
    early_stopping: EarlyStoppingState,
    best_state: BestCheckpointState,
    current_success: float,
    policy: Any,
) -> bool:
    if not early_stopping.enabled or best_state.eval_success <= 0:
        return False

    if current_success < best_state.eval_success - early_stopping.min_delta:
        early_stopping.degradation_count += 1
        print(
            f"[Train] Performance drop: {current_success:.3f} vs best {best_state.eval_success:.3f} "
            f"(degradation {early_stopping.degradation_count}/{early_stopping.patience})"
        )
        if early_stopping.degradation_count >= early_stopping.patience:
            print(
                f"[Train] Early stopping triggered! "
                f"Restoring best model (success={best_state.eval_success:.3f})"
            )
            if best_state.policy_state is not None:
                policy.load_state_dict(best_state.policy_state)
            else:
                print("[Train] WARNING: no in-memory best policy to restore after resume.")
            return True
    else:
        early_stopping.degradation_count = 0
    return False


def _run_sanity_check(cfg: Any, spec: ExperimentSpec, env: Any, policy: Any) -> None:
    if spec.sanity_check_fn is None:
        return
    spec.sanity_check_fn(cfg, env, policy)


def run_training(cfg: Any, spec: ExperimentSpec) -> RunnerResult:
    """Run the shared training loop for one experiment spec."""

    # [Cfg Setup]
    # Resolve runtime settings and tools defined in cfg, before wandb init.
    settings = _resolve_settings(cfg)
    run = init_wandb_run(cfg)
    hydra_cfg = _configure_output_dir(cfg)
    profiler = _make_profiler(cfg, settings)
    _print_configuration(cfg, settings)

    # [Spec Setup]
    # The spec is the only experiment-specific contract the runner consumes.
    # Everything below should stay independent from concrete experiment names.
    hooks = as_hooks(spec.hooks)
    eval_summary_fn = spec.eval_summary_fn or default_eval_summary
    checkpoint_adapter = spec.checkpoint_adapter or DefaultCheckpointAdapter()

    # [Build Experiment Resources]
    # Order: dataset(optional) -> env -> policy. 
    # Isaac Sim is expected to be initialized by train.py.
    resources = _build_resources(cfg, spec, hydra_cfg)
    env = spec.env_factory(cfg, resources)
    policy = spec.policy_factory(cfg, env)

    # [Read Checkpoint Setup]
    # - init_checkpoint: only warms policy weights;
    # - resume_checkpoint: restores rich training state and adjusts the frame budget.
    checkpoint_state = load_init_or_resume_checkpoint(
        cfg,
        policy,
        hydra_cfg=hydra_cfg,
        adapter=checkpoint_adapter,
    )
    resumed_frame_budget = apply_resume_frame_budget(
        cfg,
        max_iterations=settings.max_iterations,
        start_iter=checkpoint_state.start_iter,
        profiling_mode=settings.profiling_mode,
    )
    if resumed_frame_budget is not None:
        settings.max_frame_num = resumed_frame_budget

    # Resources Loading Completed, Checks.
    print("[Train] Environment structure.")
    print(env)
    print("[Train] Policy structure.")
    print(policy(env.reset()))

    # [Make OmniDrones Wrapper]
    # - Collector: for data collection and trajectory stats (e.g. success/collision rates).
    # - EpisodeStats: for accumulating episode-level info from envs and exposing it to hooks and logging.
    collector = make_collector(
        cfg,
        env,
        policy,
        total_frames=settings.max_frame_num,
    )
    episode_stats = make_episode_stats(env)

    # [Extra states initialization]
    best_state = _restore_best_state_from_resume(checkpoint_state, eval_summary_fn)
    reg_scheduler = _make_curriculum_scheduler(cfg, policy, checkpoint_state)
    early_stopping = _make_early_stopping(cfg)
    latest_eval_success: float | None = None

    # [Extra checks -> Hooks] TODO: move the sanity check into "on_before_training" hooks so
    # it can be decided by whether execute or not.
    _run_sanity_check(cfg, spec, env, policy)

    # [Lifecycle Hooks] Called before the whole training process.
    context: dict[str, Any] = {
        "cfg": cfg,
        "spec": spec,
        "run": run,
        "env": env,
        "policy": policy,
        "collector": collector,
        "episode_stats": episode_stats,
        "profiler": profiler,
        "checkpoint_state": checkpoint_state,
        "best_state": best_state,
        "settings": settings,
        "resources": resources,
        "reg_scheduler": reg_scheduler,
    }
    call_hooks(hooks, "on_before_training", context)

    batch_start_time = time.perf_counter()
    last_i = -1
    latest_checkpoint_path: str | None = None
    final_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None

    try:
        # Main training loop: collect one batch, update policy, gather stats,
        # optionally evaluate/checkpoint, then log a single info dictionary.
        for last_i, data in enumerate(collector):
            global_iter = checkpoint_state.start_iter + last_i

            # Time measured here is the elapsed wall-clock since the previous
            # collector yield, matching the old train.py profiler semantics.
            batch_elapsed = time.perf_counter() - batch_start_time
            profiler.record("batch_total", batch_elapsed)
            profiler.increment_batch()
            batch_start_time = time.perf_counter()

            # [Get Collector Stats]
            current_env_frames = collector_env_frames(
                collector,
                start_env_frames=checkpoint_state.start_env_frames,
            )
            info = {
                "batch": global_iter,
                "env_frames": current_env_frames,
                "rollout_fps": collector_fps(collector),
            }

            # [Lifecycle Hooks] TODO: add a hook point here for "on_before_train_step".

            # [Train Step]
            # Policy classes expose train_op(data) as the common optimization
            # boundary. Algorithm-specific details stay inside src/algos.
            with profiler.timer("ppo_train_op"):
                training_infos = policy.train_op(data.to_tensordict())
            info.update({f"ppo_train/{key}": value for key, value in training_infos.items()})

            # [Getting EpisodeStats]
            # EpisodeStats may or may not have enough completed episodes in a
            # given batch; update_episode_stats returns an empty dict when not.
            with profiler.timer("episode_stats"):
                info.update(update_episode_stats(episode_stats, data, env))

            context.update(
                {
                    "data": data,
                    "info": info,
                    "global_iter": global_iter,
                    "env_frames": current_env_frames,
                    "latest_eval_success": latest_eval_success,
                }
            )

            # [Evaluation Step]
            if settings.eval_interval > 0 and last_i % settings.eval_interval == 0:
                # Evaluation returns raw eval/* metrics. The spec-owned summary
                # function decides how to rank them for best-checkpoint tracking.
                logging.info(f"Eval at iter {global_iter} ({collector._frames} steps).")
                
                # [Lifecycle Hooks] TODO: add a hook point here for "on_before_evaluation_step".

                eval_info = evaluate_policy_no_grad(cfg, env, policy)
                info.update(eval_info)
                print(f"[Train] Eval info at iter {global_iter} ({collector._frames} steps): DONE")

                summary = eval_summary_fn(eval_info)
                info["eval/checkpoint_score"] = summary.score
                current_success = summary.success
                latest_eval_success = current_success

                # [Lifecycle Hooks] Called after each evaluation step.
                context.update(
                    {
                        "eval_info": eval_info,
                        "eval_summary": summary,
                        "latest_eval_success": latest_eval_success,
                    }
                )
                call_hooks(hooks, "on_after_eval", context)

                if settings.env_test_mode:
                    print("[Training Loop] env_test_mode activated, exiting after evaluation.")
                    break

                # Keep the best policy in memory and save a policy-only best
                # checkpoint immediately. 
                if _update_best_state(best_state, summary, eval_info, policy):
                    best_paths = save_best_checkpoint(
                        run,
                        cfg,
                        best_state=best_state,
                        fallback_checkpoint_path=cfg.log_output_dir,
                        policy_only=True,
                    )
                    best_checkpoint_path = best_paths.checkpoint_path
                    print(
                        f"[Train] New best model! score={best_state.eval_score:.3f} "
                        f"success={best_state.eval_success:.3f} collision={best_state.eval_collision:.3f} "
                        f"at iter {global_iter}"
                    )

                # TODO: add a trigger for whether enable early stopping.
                if _should_stop_for_early_stopping(
                    early_stopping,
                    best_state,
                    current_success,
                    policy,
                ):
                    break
            
            # [Algorithm scheduler -> Hooks] TODO: move policy schedulers into lifecycle hooks instead of here.
            if reg_scheduler is not None and last_i % reg_scheduler.check_interval == 0:
                # Curriculum uses the latest evaluation success; if no eval has
                # run yet this iteration, keep the current coefficient.
                if latest_eval_success is not None:
                    new_reg = reg_scheduler.update(latest_eval_success)
                    policy.set_reg_coeff(new_reg)
                    info["curriculum/reg_coeff"] = new_reg
                    info["curriculum/ema_success"] = reg_scheduler.ema_success

            # [Lifecycle Hooks] Called after train step, can be customized.
            context.update(
                {
                    "info": info,
                    "global_iter": global_iter,
                    "env_frames": current_env_frames,
                    "latest_eval_success": latest_eval_success,
                }
            )
            call_hooks(hooks, "on_after_train_step", context)

            # [WandB]
            # Log profiler logs if in profiling mode every 5 iterations .
            if settings.profiling_mode and last_i > 0 and last_i % 5 == 0:
                profiler.log_to_wandb(run)

            # All metrics from train/eval/curriculum/hooks are committed once
            # per iteration to WandB log_info. 
            log_info(run, info)

            # [Checkpoint Saving] 
            if last_i % settings.save_interval == 0 and not settings.profiling_mode:
                # Periodic checkpoints are always rich checkpoints so they can
                # be used with resume_checkpoint later.
                call_hooks(hooks, "on_before_checkpoint", context)
                latest_paths = save_latest_checkpoint(
                    run,
                    cfg,
                    policy,
                    iter_value=global_iter,
                    env_frames=current_env_frames,
                    adapter=checkpoint_adapter,
                    best_state=best_state,
                    extra_state=_checkpoint_extra_state(reg_scheduler),
                )
                latest_checkpoint_path = latest_paths.checkpoint_path
                context["latest_checkpoint_path"] = latest_checkpoint_path
                call_hooks(hooks, "on_after_checkpoint", context)

    finally:
        # Always attempt final/best checkpoint and wandb cleanup, even if the
        # loop breaks early due to env_test_mode or early stopping.
        final_iter = checkpoint_state.start_iter + last_i if last_i >= 0 else checkpoint_state.start_iter - 1
        final_env_frames = collector_env_frames(
            collector,
            start_env_frames=checkpoint_state.start_env_frames,
        )

        if not settings.profiling_mode:
            # Normal runs summarize with final checkpoint and best checkpoint paths. 
            context.update(
                {
                    "global_iter": final_iter,
                    "env_frames": final_env_frames,
                    "info": {},
                }
            )
            call_hooks(hooks, "on_before_checkpoint", context)
            final_paths = save_final_checkpoint(
                run,
                cfg,
                policy,
                iter_value=final_iter,
                env_frames=final_env_frames,
                adapter=checkpoint_adapter,
                best_state=best_state,
                extra_state=_checkpoint_extra_state(reg_scheduler),
            )
            final_checkpoint_path = final_paths.checkpoint_path
            best_paths = save_best_checkpoint(
                run,
                cfg,
                best_state=best_state,
                fallback_checkpoint_path=final_checkpoint_path,
                policy_only=True,
            )
            best_checkpoint_path = best_paths.checkpoint_path
            context.update(
                {
                    "final_checkpoint_path": final_checkpoint_path,
                    "best_checkpoint_path": best_checkpoint_path,
                }
            )
            call_hooks(hooks, "on_after_checkpoint", context)

        if settings.profiling_mode:
            # Profiling runs, skip checkpoints saving.
            profiler.print_summary()
            profiler.log_to_wandb(run)
            print(f"[Train] Profiling complete. Log saved to: {settings.profiler_log_file}")

        finish_wandb()

    return RunnerResult(
        final_checkpoint_path=final_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        last_iter=checkpoint_state.start_iter + last_i if last_i >= 0 else checkpoint_state.start_iter - 1,
        env_frames=collector_env_frames(collector, start_env_frames=checkpoint_state.start_env_frames),
        best_state=best_state,
    )


__all__ = [
    "EarlyStoppingState",
    "RunnerResult",
    "RuntimeSettings",
    "run_training",
]
