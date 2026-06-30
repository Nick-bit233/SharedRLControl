"""Weights & Biases helpers for the shared training runtime."""

from __future__ import annotations

import datetime
import os
from collections.abc import Mapping
from typing import Any


DEFAULT_RUN_TIME_FORMAT = "%m-%d_%H-%M"


def _wandb() -> Any:
    import wandb

    return wandb


def _omega_conf() -> Any:
    from omegaconf import OmegaConf

    return OmegaConf


def disable_wandb_for_special_modes(
    cfg: Any,
    *,
    profiling_mode: bool = False,
    env_test_mode: bool = False,
) -> None:
    """Disable wandb when running profiling or quick environment checks."""

    if profiling_mode or env_test_mode:
        cfg.wandb.mode = "disabled"


def wandb_config_from_cfg(cfg: Any) -> dict[str, Any]:
    """Convert an OmegaConf config into a plain container accepted by wandb."""

    OmegaConf = _omega_conf()
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def build_wandb_run_name(
    cfg: Any,
    *,
    timestamp: datetime.datetime | None = None,
    time_format: str = DEFAULT_RUN_TIME_FORMAT,
) -> str:
    """Build the run name used by the legacy train.py entrypoints."""

    timestamp = timestamp or datetime.datetime.now()
    return f"{cfg.wandb.name}/{timestamp.strftime(time_format)}"


def get_wandb_group(cfg: Any) -> Any | None:
    """Return the configured wandb group, preserving null as None."""

    return cfg.wandb.get("group", None)


def get_wandb_run_id(cfg: Any) -> str | None:
    """Return an existing run id when resuming, otherwise None."""

    return cfg.wandb.get("run_id", None)


def init_wandb_run(
    cfg: Any,
    *,
    run_name: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Initialize a wandb run using the current experiment config.

    Behavior intentionally matches the existing train.py scripts:
    - generate a new run id when ``cfg.wandb.run_id`` is null
    - resume with ``resume="must"`` when a run id is provided
    - append a ``MM-DD_HH-MM`` timestamp to ``cfg.wandb.name``
    """

    wandb = _wandb()
    run_id = get_wandb_run_id(cfg)
    init_kwargs = {
        "project": cfg.wandb.project,
        "name": run_name or build_wandb_run_name(cfg),
        "entity": cfg.wandb.entity,
        "group": get_wandb_group(cfg),
        "config": config if config is not None else wandb_config_from_cfg(cfg),
        "mode": cfg.wandb.mode,
        "id": wandb.util.generate_id() if run_id is None else run_id,
    }
    if run_id is not None:
        init_kwargs["resume"] = "must"

    return wandb.init(**init_kwargs)


def get_run_dir(run: Any, fallback_dir: str, *, create: bool = True) -> str:
    """Return ``run.dir`` when available, otherwise the Hydra output directory."""

    run_dir = getattr(run, "dir", None)
    if run_dir and os.path.exists(run_dir):
        save_dir = run_dir
    else:
        save_dir = fallback_dir

    if create:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def log_info(run: Any, info: Mapping[str, Any]) -> None:
    """Log one metrics dictionary to wandb."""

    run.log(dict(info))


def finish_wandb() -> None:
    """Finish the active wandb run."""

    wandb = _wandb()
    wandb.finish()


def make_video(
    frames: Any,
    *,
    fps: float,
    format: str = "mp4",
    **kwargs: Any,
) -> Any:
    """Create a wandb video object from an array of frames."""

    wandb = _wandb()
    return wandb.Video(frames, fps=fps, format=format, **kwargs)


def video_fps_from_cfg(cfg: Any, *, playback_scale: float = 0.5) -> float:
    """Return the fps formula used by the current evaluation code."""

    return playback_scale / (cfg.sim.dt * cfg.sim.substeps)


def add_video(
    info: dict[str, Any],
    key: str,
    frames: Any,
    *,
    fps: float,
    format: str = "mp4",
    **kwargs: Any,
) -> None:
    """Attach a video to a metrics dictionary in-place."""

    info[key] = make_video(frames, fps=fps, format=format, **kwargs)


__all__ = [
    "DEFAULT_RUN_TIME_FORMAT",
    "add_video",
    "build_wandb_run_name",
    "disable_wandb_for_special_modes",
    "finish_wandb",
    "get_run_dir",
    "get_wandb_group",
    "get_wandb_run_id",
    "init_wandb_run",
    "log_info",
    "make_video",
    "video_fps_from_cfg",
    "wandb_config_from_cfg",
]
