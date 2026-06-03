"""Simple LiDAR-based command-conditioned risk utilities."""

from __future__ import annotations

import math
from typing import Any

import torch


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _directional_lidar_risk(
    *,
    direction_vec_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    ray_dists: torch.Tensor,
    lidar_range: float,
    d_safe: float,
    distance_temperature: float,
    cone_cos: float,
    ttc_safe: float,
    ttc_temperature: float,
    min_speed: float,
    hit_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speed = direction_vec_w.norm(dim=-1, keepdim=True)
    unit_dir = direction_vec_w / speed.clamp_min(1e-6)
    cos_sim = (ray_dirs_w * unit_dir.unsqueeze(1)).sum(dim=-1)
    cone_mask = cos_sim >= cone_cos

    fallback = torch.full_like(ray_dists, float(lidar_range))
    dist_in_cone = torch.where(cone_mask, ray_dists, fallback)
    dist_cone = dist_in_cone.min(dim=-1, keepdim=True).values
    has_hit = cone_mask.any(dim=-1, keepdim=True) & (dist_cone < float(lidar_range) - hit_epsilon)
    active = speed > min_speed

    dist_risk = torch.sigmoid((float(d_safe) - dist_cone) / distance_temperature)
    ttc = (dist_cone - float(d_safe)).clamp_min(0.0) / speed.clamp_min(1e-6)
    ttc_risk = torch.sigmoid((float(ttc_safe) - ttc) / ttc_temperature)
    risk = torch.maximum(dist_risk, ttc_risk)
    risk = torch.where(active & has_hit, risk, torch.zeros_like(risk))
    ttc_fallback = (float(lidar_range) - float(d_safe)) / max(float(min_speed), 1e-6)
    ttc = torch.where(active & has_hit, ttc, torch.full_like(ttc, ttc_fallback))
    return risk.clamp(0.0, 1.0), dist_cone, ttc


def compute_simple_lidar_command_risk(
    *,
    position_w: torch.Tensor,
    velocity_w: torch.Tensor,
    command_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    ray_dists: torch.Tensor,
    height_range: torch.Tensor | None,
    params: Any,
) -> dict[str, torch.Tensor]:
    """Estimate collision risk from current LiDAR, velocity, and command direction.

    The public output keys match ``compute_dynamic_command_risk`` so callers can
    swap the estimator without changing reward or PPO plumbing.
    """

    if position_w.ndim != 2 or velocity_w.ndim != 2 or command_w.ndim != 2:
        raise ValueError("position_w, velocity_w, and command_w must be [N, 3]")
    if ray_dirs_w.ndim != 3 or ray_dists.ndim != 2:
        raise ValueError("ray_dirs_w must be [N, R, 3] and ray_dists must be [N, R]")
    if position_w.shape != velocity_w.shape or velocity_w.shape != command_w.shape:
        raise ValueError("position_w, velocity_w, and command_w must have matching shapes")

    lidar_range = float(_cfg_get(params, "lidar_range", 4.0))
    d_safe = float(_cfg_get(params, "d_safe", 0.5))
    distance_temperature = max(float(_cfg_get(params, "distance_temperature", 0.12)), 1e-6)
    cone_half_angle_deg = float(_cfg_get(params, "cone_half_angle_deg", 35.0))
    cone_cos = math.cos(math.radians(cone_half_angle_deg))
    ttc_safe = float(_cfg_get(params, "ttc_safe", 1.5))
    ttc_temperature = max(float(_cfg_get(params, "ttc_temperature", 0.35)), 1e-6)
    min_speed = float(_cfg_get(params, "min_speed", 0.10))
    hit_epsilon = float(_cfg_get(params, "hit_epsilon", 0.02))

    ray_dists = ray_dists.clamp(0.0, lidar_range)
    dist_min = ray_dists.min(dim=-1, keepdim=True).values
    risk_proximity = torch.sigmoid((d_safe - dist_min) / distance_temperature).clamp(0.0, 1.0)

    risk_velocity, dist_velocity_cone, ttc_velocity = _directional_lidar_risk(
        direction_vec_w=velocity_w,
        ray_dirs_w=ray_dirs_w,
        ray_dists=ray_dists,
        lidar_range=lidar_range,
        d_safe=d_safe,
        distance_temperature=distance_temperature,
        cone_cos=cone_cos,
        ttc_safe=ttc_safe,
        ttc_temperature=ttc_temperature,
        min_speed=min_speed,
        hit_epsilon=hit_epsilon,
    )
    risk_command, dist_command_cone, ttc_command = _directional_lidar_risk(
        direction_vec_w=command_w,
        ray_dirs_w=ray_dirs_w,
        ray_dists=ray_dists,
        lidar_range=lidar_range,
        d_safe=d_safe,
        distance_temperature=distance_temperature,
        cone_cos=cone_cos,
        ttc_safe=ttc_safe,
        ttc_temperature=ttc_temperature,
        min_speed=min_speed,
        hit_epsilon=hit_epsilon,
    )

    rho_post = risk_command
    rho_delay = torch.maximum(risk_proximity, risk_velocity)
    rho_full = torch.maximum(rho_post, rho_delay)
    min_clearance = dist_min - d_safe

    return {
        "rho_full": rho_full.clamp(0.0, 1.0),
        "rho_post": rho_post.clamp(0.0, 1.0),
        "rho_delay": rho_delay.clamp(0.0, 1.0),
        "min_clearance": min_clearance,
        "risk_proximity": risk_proximity,
        "risk_velocity": risk_velocity,
        "risk_command": risk_command,
        "dist_min": dist_min,
        "dist_velocity_cone": dist_velocity_cone,
        "dist_command_cone": dist_command_cone,
        "ttc_velocity": ttc_velocity,
        "ttc_command": ttc_command,
    }


__all__ = ["compute_simple_lidar_command_risk"]
