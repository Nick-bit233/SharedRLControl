"""Dynamics-aware command-conditioned risk utilities."""

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


def _clip_by_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vec)
    norm = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (float(max_norm) / norm).clamp(max=1.0)
    return vec * scale


def compute_dynamic_command_risk(
    *,
    velocity_w: torch.Tensor,
    command_w: torch.Tensor,
    hold_command_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    ray_dists: torch.Tensor,
    params: Any,
) -> dict[str, torch.Tensor]:
    """Estimate short-horizon collision risk for a candidate velocity command.

    All vectors must be in the same frame. The implementation is intentionally
    pure Torch so it can be unit-tested without Isaac Sim.
    """

    if velocity_w.ndim != 2 or command_w.ndim != 2 or hold_command_w.ndim != 2:
        raise ValueError("velocity_w, command_w, and hold_command_w must be [N, 3]")
    if ray_dirs_w.ndim != 3 or ray_dists.ndim != 2:
        raise ValueError("ray_dirs_w must be [N, R, 3] and ray_dists must be [N, R]")

    device = velocity_w.device
    dtype = velocity_w.dtype
    num_envs = velocity_w.shape[0]

    dt = float(_cfg_get(params, "dt_risk", 0.05))
    horizon = float(_cfg_get(params, "T_horizon", 1.5))
    tau_delay = float(_cfg_get(params, "tau_delay", 0.2))
    tau_v = max(float(_cfg_get(params, "tau_v", 0.4)), 1e-6)
    a_max = float(_cfg_get(params, "a_max", 1.5))
    v_max = float(_cfg_get(params, "v_max", 2.0))
    r_uav = float(_cfg_get(params, "r_uav", 0.3))
    margin_static = float(_cfg_get(params, "margin_static", 0.15))
    margin_speed = float(_cfg_get(params, "margin_speed", 0.08))
    margin_time = float(_cfg_get(params, "margin_time", 0.03))
    risk_temperature = max(float(_cfg_get(params, "risk_temperature", 0.12)), 1e-6)

    steps = max(1, int(math.ceil(horizon / dt)))
    obstacle_points = ray_dirs_w * ray_dists.unsqueeze(-1)

    v = velocity_w.clone()
    p = torch.zeros_like(v)
    rho_full = torch.zeros(num_envs, 1, device=device, dtype=dtype)
    rho_delay = torch.zeros_like(rho_full)
    rho_post = torch.zeros_like(rho_full)
    min_clearance = torch.full((num_envs, 1), float("inf"), device=device, dtype=dtype)
    post_seen = False

    for step in range(steps):
        t0 = step * dt
        t1 = (step + 1) * dt
        in_delay = t0 < tau_delay
        q_step = hold_command_w if in_delay else command_w

        accel = _clip_by_norm((q_step - v) / tau_v, a_max)
        v = _clip_by_norm(v + dt * accel, v_max)
        p = p + dt * v

        clearance_dist = (obstacle_points - p.unsqueeze(1)).norm(dim=-1).min(dim=-1, keepdim=True).values
        tube_radius = r_uav + margin_static + margin_speed * v.norm(dim=-1, keepdim=True) + margin_time * t1
        clearance = clearance_dist - tube_radius
        risk = torch.sigmoid(-clearance / risk_temperature)

        rho_full = torch.maximum(rho_full, risk)
        min_clearance = torch.minimum(min_clearance, clearance)
        if in_delay:
            rho_delay = torch.maximum(rho_delay, risk)
        else:
            post_seen = True
            rho_post = torch.maximum(rho_post, risk)

    if not post_seen:
        rho_post = rho_full.clone()

    return {
        "rho_full": rho_full.clamp(0.0, 1.0),
        "rho_post": rho_post.clamp(0.0, 1.0),
        "rho_delay": rho_delay.clamp(0.0, 1.0),
        "min_clearance": min_clearance,
    }


__all__ = ["compute_dynamic_command_risk"]
