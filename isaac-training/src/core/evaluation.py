"""Evaluation helpers for the shared training runtime."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.core.wandb_utils import add_video, video_fps_from_cfg


DEBUG_VECTOR_SUFFIXES = ("x", "y", "z", "w")


def _torch() -> Any:
    import torch

    return torch


def _torchrl_eval_tools() -> tuple[Any, Any, Any]:
    from omni_drones.utils.torchrl import RenderCallback
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    return RenderCallback, ExplorationType, set_exploration_type


def _call_if_present(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    return method(*args, **kwargs)


def flatten_first_episode_stats(trajs: Any) -> dict[str, Any]:
    """Flatten rollout stats into the eval key format used by train.py."""

    torch = _torch()
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: Any) -> Any:
        indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info: dict[str, Any] = {}
    for key, value in traj_stats.items():
        value_mean = torch.mean(value.float(), dim=0)
        if value_mean.numel() == 1:
            info["eval/" + key] = value_mean.item()
        else:
            clean = key
            if clean.startswith("debug_"):
                clean = clean[len("debug_") :]
            for suffix, val in zip(DEBUG_VECTOR_SUFFIXES[: value_mean.numel()], value_mean.reshape(-1)):
                info[f"eval_debug/{clean}/{suffix}"] = val.item()
    return info


def run_eval_rollout(
    cfg: Any,
    env: Any,
    policy: Any,
    *,
    camera_mode: str = "follow",
    record_video: bool = False,
    eval_max_steps: int | None = None,
    render_interval: int = 2,
    exploration_type: Any | None = None,
) -> tuple[Any | None, Any]:
    """Run one evaluation rollout, optionally recording frames."""

    RenderCallback, ExplorationType, set_exploration_type = _torchrl_eval_tools()
    if exploration_type is None:
        exploration_type = ExplorationType.DETERMINISTIC

    render_callback = None
    if record_video:
        _call_if_present(env, "set_camera_view_mode", camera_mode)
        render_callback = RenderCallback(interval=render_interval)

    max_steps = int(eval_max_steps if eval_max_steps is not None else env.max_episode_length)
    with set_exploration_type(exploration_type):
        if record_video:
            try:
                trajs = env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    callback=render_callback,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
            except Exception as exc:
                print(f"[Eval] detailed rendering error: {exc}")
                trajs = env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    callback=None,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
        else:
            trajs = env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=None,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )

    env.reset()
    return render_callback, trajs


def prepare_eval_rendering(
    cfg: Any,
    env: Any,
    *,
    record_video: bool,
    warmup_frames: int = 10,
) -> None:
    """Configure rendering before evaluation."""

    if record_video:
        print("[Eval] Enabling renderer and warming up...")
        env.enable_render(True)
        _call_if_present(env, "set_envs_visibility", visible_env_ids={0})
        for _ in range(warmup_frames):
            env.sim.render()
    else:
        env.enable_render(False)

    if cfg.get("eval_visualization", False):
        _call_if_present(env, "set_visualization", enabled=True)


def restore_eval_rendering(env: Any, *, record_video: bool) -> None:
    """Restore rendering/visibility state after evaluation."""

    _call_if_present(env, "set_visualization", enabled=False)
    if record_video:
        print("[Eval] Evaluation done. Restoring visibility and disabling renderer.")
        _call_if_present(env, "set_envs_visibility", visible_env_ids=None)
        env.enable_render(False)
    _call_if_present(env, "set_visualization", enabled=False)


def attach_recorded_videos(
    cfg: Any,
    info: dict[str, Any],
    callbacks: Mapping[str, Any | None],
    *,
    axes: str = "t c h w",
) -> None:
    """Attach recorded rollout videos to an eval info dictionary."""

    fps = video_fps_from_cfg(cfg)
    for key, callback in callbacks.items():
        if callback is None:
            continue
        add_video(
            info,
            key,
            callback.get_video_array(axes=axes),
            fps=fps,
            format="mp4",
        )


def evaluate_policy(
    cfg: Any,
    env: Any,
    policy: Any,
    *,
    seed: int = 42,
    render_interval: int = 2,
) -> dict[str, Any]:
    """Evaluate a policy with the behavior used by the tunnel train entrypoint."""

    record_video = cfg.get("record_video", False)
    print(f"[Eval] Starting evaluation... Video recording: {record_video}")

    env.eval()
    try:
        prepare_eval_rendering(cfg, env, record_video=record_video)
        env.set_seed(seed)
        eval_max_steps = int(env.max_episode_length)

        logging.info("[Eval] Running rollout...")
        render_callback_follow, trajs = run_eval_rollout(
            cfg,
            env,
            policy,
            camera_mode="follow",
            record_video=record_video,
            eval_max_steps=eval_max_steps,
            render_interval=render_interval,
        )

        render_callback_global = None
        if record_video and cfg.get("global_view", False):
            logging.info("[Eval] Running rollout with global camera view...")
            render_callback_global, _ = run_eval_rollout(
                cfg,
                env,
                policy,
                camera_mode="global",
                record_video=True,
                eval_max_steps=eval_max_steps,
                render_interval=render_interval,
            )

        logging.info(f"[Eval] trajs keys: {trajs.keys()}")
        info = flatten_first_episode_stats(trajs)
        logging.info(f"[Eval] eval info: {info}")

        if record_video:
            attach_recorded_videos(
                cfg,
                info,
                {
                    "recording_follow": render_callback_follow,
                    "recording_global": render_callback_global,
                },
            )
            if render_callback_follow is not None:
                logging.info("[Eval] Follow camera video saved to wandb")
            if render_callback_global is not None:
                logging.info("[Eval] Global camera video saved to wandb")

        return info
    finally:
        restore_eval_rendering(env, record_video=record_video)
        env.train()


def evaluate_policy_no_grad(
    cfg: Any,
    env: Any,
    policy: Any,
    *,
    seed: int = 42,
    render_interval: int = 2,
) -> dict[str, Any]:
    """Evaluate a policy inside ``torch.no_grad()``."""

    torch = _torch()
    with torch.no_grad():
        return evaluate_policy(
            cfg,
            env,
            policy,
            seed=seed,
            render_interval=render_interval,
        )


__all__ = [
    "DEBUG_VECTOR_SUFFIXES",
    "attach_recorded_videos",
    "evaluate_policy",
    "evaluate_policy_no_grad",
    "flatten_first_episode_stats",
    "prepare_eval_rendering",
    "restore_eval_rendering",
    "run_eval_rollout",
]
