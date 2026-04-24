from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PerceptionConfig:
    sim_dt: float
    max_delay_sec: float = 0.5


class PilotPerceptionModel:
    """Delayed/noisy obstacle perception for vectorized pilot models."""

    def __init__(self, num_envs: int, device: torch.device, cfg: PerceptionConfig):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.buffer_len = max(2, int(round(cfg.max_delay_sec / cfg.sim_dt)) + 2)
        self.write_idx = 0
        self.dist_buffer = torch.full(
            (self.buffer_len, num_envs), float("inf"), device=device
        )
        self.normal_buffer = torch.zeros(self.buffer_len, num_envs, 3, device=device)

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.dist_buffer[:, ids] = float("inf")
        self.normal_buffer[:, ids] = 0.0

    def update(self, true_distance: torch.Tensor, true_normal: torch.Tensor):
        self.dist_buffer[self.write_idx] = true_distance
        self.normal_buffer[self.write_idx] = true_normal
        self.write_idx = (self.write_idx + 1) % self.buffer_len

    def perceive(
        self,
        tau_perc: torch.Tensor,
        sigma_perc: torch.Tensor,
        d_react: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ):
        delay_steps = torch.round(tau_perc / self.cfg.sim_dt).long().clamp(
            min=0, max=self.buffer_len - 1
        )
        read_idx = (self.write_idx - 1 - delay_steps) % self.buffer_len
        env_ids = torch.arange(self.num_envs, device=self.device)

        delayed_dist = self.dist_buffer[read_idx, env_ids]
        delayed_normal = self.normal_buffer[read_idx, env_ids]

        noise = torch.randn(
            self.num_envs, device=self.device, generator=generator
        ) * sigma_perc
        perceived_dist = (delayed_dist + noise).clamp_min(0.0)
        threat = ((d_react - perceived_dist) / d_react.clamp_min(1e-3)).clamp(0.0, 1.0)
        return threat, perceived_dist, delayed_normal
