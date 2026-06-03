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
    angular_power: float,
    ttc_safe: float,
    ttc_temperature: float,
    min_speed: float,
    min_closing_speed: float,
    hit_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speed = direction_vec_w.norm(dim=-1, keepdim=True)
    unit_dir = direction_vec_w / speed.clamp_min(1e-6)
    cos_sim = (ray_dirs_w * unit_dir.unsqueeze(1)).sum(dim=-1)
    closing_speed = speed * cos_sim
    hit_mask = ray_dists < float(lidar_range) - hit_epsilon
    cone_mask = (cos_sim >= cone_cos) & (closing_speed >= min_closing_speed) & hit_mask

    fallback = torch.full_like(ray_dists, float(lidar_range))
    dist_in_cone = torch.where(cone_mask, ray_dists, fallback)
    dist_cone = dist_in_cone.min(dim=-1, keepdim=True).values
    has_hit = cone_mask.any(dim=-1, keepdim=True)
    active = speed > min_speed

    angle_weight = ((cos_sim - cone_cos) / max(1.0 - cone_cos, 1e-6)).clamp(0.0, 1.0)
    angle_weight = angle_weight.pow(max(float(angular_power), 0.0))
    ttc_ray = (ray_dists - float(d_safe)).clamp_min(0.0) / closing_speed.clamp_min(1e-6)
    dist_risk = torch.sigmoid((float(d_safe) - ray_dists) / distance_temperature)
    ttc_risk = torch.sigmoid((float(ttc_safe) - ttc_ray) / ttc_temperature)
    ray_risk = angle_weight * torch.maximum(dist_risk, ttc_risk)
    ray_risk = torch.where(cone_mask, ray_risk, torch.zeros_like(ray_risk))
    risk = ray_risk.max(dim=-1, keepdim=True).values
    risk = torch.where(active & has_hit, risk, torch.zeros_like(risk))

    ttc_in_cone = torch.where(cone_mask, ttc_ray, fallback)
    ttc = ttc_in_cone.min(dim=-1, keepdim=True).values
    ttc_fallback = (float(lidar_range) - float(d_safe)) / max(float(min_speed), 1e-6)
    ttc = torch.where(active & has_hit, ttc, torch.full_like(ttc, ttc_fallback))
    return risk.clamp(0.0, 1.0), dist_cone, ttc


def _motion_gated_min_distance_risk(
    *,
    velocity_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    ray_dists: torch.Tensor,
    d_safe: float,
    distance_temperature: float,
    side_distance_trigger: float,
    side_distance_temperature: float,
    side_ttc_safe: float,
    side_ttc_temperature: float,
    min_closing_speed: float,
    closing_speed_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist_min, min_idx = ray_dists.min(dim=-1, keepdim=True)
    gather_idx = min_idx.unsqueeze(-1).expand(-1, -1, ray_dirs_w.shape[-1])
    min_ray_dir_w = ray_dirs_w.gather(dim=1, index=gather_idx).squeeze(1)

    clearance_risk = torch.sigmoid((float(d_safe) - dist_min) / distance_temperature)
    closing_min = (velocity_w * min_ray_dir_w).sum(dim=-1, keepdim=True)
    closing_active = closing_min >= min_closing_speed

    side_ttc = (dist_min - float(d_safe)).clamp_min(0.0) / closing_min.clamp_min(1e-6)
    side_ttc_risk = torch.sigmoid((float(side_ttc_safe) - side_ttc) / side_ttc_temperature)
    side_distance_risk = torch.sigmoid(
        (float(side_distance_trigger) - dist_min) / side_distance_temperature
    )
    motion_gate = torch.sigmoid(
        (closing_min - float(min_closing_speed)) / closing_speed_temperature
    )
    side_closing_risk = side_ttc_risk * side_distance_risk * motion_gate
    side_closing_risk = torch.where(closing_active, side_closing_risk, torch.zeros_like(side_closing_risk))

    risk = torch.maximum(clearance_risk, side_closing_risk).clamp(0.0, 1.0)
    return risk, dist_min, side_ttc, closing_min


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
    distance_temperature = max(float(_cfg_get(params, "distance_temperature", 0.10)), 1e-6)
    cone_half_angle_deg = float(_cfg_get(params, "cone_half_angle_deg", 25.0))
    cone_cos = math.cos(math.radians(cone_half_angle_deg))
    angular_power = float(_cfg_get(params, "angular_power", 2.0))
    ttc_safe = float(_cfg_get(params, "ttc_safe", 1.0))
    ttc_temperature = max(float(_cfg_get(params, "ttc_temperature", 0.25)), 1e-6)
    min_speed = float(_cfg_get(params, "min_speed", 0.10))
    min_closing_speed = float(_cfg_get(params, "min_closing_speed", 0.15))
    closing_speed_temperature = max(float(_cfg_get(params, "closing_speed_temperature", 0.08)), 1e-6)
    side_distance_trigger = float(_cfg_get(params, "side_distance_trigger", 1.0))
    side_distance_temperature = max(float(_cfg_get(params, "side_distance_temperature", 0.20)), 1e-6)
    side_ttc_safe = float(_cfg_get(params, "side_ttc_safe", 0.8))
    side_ttc_temperature = max(float(_cfg_get(params, "side_ttc_temperature", 0.25)), 1e-6)
    hit_epsilon = float(_cfg_get(params, "hit_epsilon", 0.02))

    ray_norm = ray_dirs_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ray_dirs_w = ray_dirs_w / ray_norm
    ray_dists = ray_dists.clamp(0.0, lidar_range)
    risk_proximity, dist_min, side_ttc, closing_min = _motion_gated_min_distance_risk(
        velocity_w=velocity_w,
        ray_dirs_w=ray_dirs_w,
        ray_dists=ray_dists,
        d_safe=d_safe,
        distance_temperature=distance_temperature,
        side_distance_trigger=side_distance_trigger,
        side_distance_temperature=side_distance_temperature,
        side_ttc_safe=side_ttc_safe,
        side_ttc_temperature=side_ttc_temperature,
        min_closing_speed=min_closing_speed,
        closing_speed_temperature=closing_speed_temperature,
    )

    risk_velocity, dist_velocity_cone, ttc_velocity = _directional_lidar_risk(
        direction_vec_w=velocity_w,
        ray_dirs_w=ray_dirs_w,
        ray_dists=ray_dists,
        lidar_range=lidar_range,
        d_safe=d_safe,
        distance_temperature=distance_temperature,
        cone_cos=cone_cos,
        angular_power=angular_power,
        ttc_safe=ttc_safe,
        ttc_temperature=ttc_temperature,
        min_speed=min_speed,
        min_closing_speed=min_closing_speed,
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
        angular_power=angular_power,
        ttc_safe=ttc_safe,
        ttc_temperature=ttc_temperature,
        min_speed=min_speed,
        min_closing_speed=min_closing_speed,
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
        "ttc_min_distance": side_ttc,
        "closing_min_distance": closing_min,
    }


__all__ = ["compute_simple_lidar_command_risk"]
