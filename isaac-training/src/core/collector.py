"""Collector and episode-stat helpers for the shared training runtime."""

from __future__ import annotations

from typing import Any


DEBUG_VECTOR_SUFFIXES = ("x", "y", "z", "w")


def _torch() -> Any:
    import torch

    return torch


def _collector_tools() -> tuple[Any, Any]:
    from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector

    return SyncDataCollector, EpisodeStats


def make_collector(
    cfg: Any,
    env: Any,
    policy: Any,
    *,
    total_frames: int,
    return_same_td: bool = True,
) -> Any:
    """Create the SyncDataCollector used by the train entrypoints."""

    SyncDataCollector, _ = _collector_tools()
    return SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
        total_frames=total_frames,
        return_same_td=return_same_td,
        device=cfg.device,
    )


def get_stats_keys(env: Any) -> list[Any]:
    """Return nested observation keys used by EpisodeStats."""

    return [
        key
        for key in env.observation_spec.keys(True, True)
        if isinstance(key, tuple) and key[0] == "stats"
    ]


def make_episode_stats(env: Any) -> Any:
    """Create an EpisodeStats accumulator for an environment."""

    _, EpisodeStats = _collector_tools()
    return EpisodeStats(in_keys=get_stats_keys(env))


def flatten_episode_stats(episode_stats: Any, env: Any) -> dict[str, Any]:
    """Pop and flatten EpisodeStats into the existing wandb key layout."""

    torch = _torch()
    stats: dict[str, Any] = {}
    if len(episode_stats) < env.num_envs:
        return stats

    for key, value in episode_stats.pop().items(include_nested=True, leaves_only=True):
        key_name = key if isinstance(key, str) else "_".join(key)
        value_mean = torch.mean(value.float(), dim=0)
        if value_mean.numel() == 1:
            stats[f"episode/{key_name}"] = value_mean.item()
        else:
            clean = key_name
            for prefix in ("stats_debug_", "stats_"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    break
            for suffix, val in zip(DEBUG_VECTOR_SUFFIXES[: value_mean.numel()], value_mean.reshape(-1)):
                stats[f"debug/{clean}/{suffix}"] = val.item()
    return stats


def update_episode_stats(episode_stats: Any, data: Any, env: Any) -> dict[str, Any]:
    """Add one collector batch and return flattened episode stats if available."""

    episode_stats.add(data.to_tensordict())
    return flatten_episode_stats(episode_stats, env)


def collector_env_frames(collector: Any, *, start_env_frames: int = 0) -> int:
    """Return absolute env frame count including resumed progress."""

    return int(getattr(collector, "_frames", 0)) + int(start_env_frames)


def collector_fps(collector: Any) -> Any:
    """Return rollout fps from a collector if available."""

    return getattr(collector, "_fps", None)


__all__ = [
    "DEBUG_VECTOR_SUFFIXES",
    "collector_env_frames",
    "collector_fps",
    "flatten_episode_stats",
    "get_stats_keys",
    "make_collector",
    "make_episode_stats",
    "update_episode_stats",
]
