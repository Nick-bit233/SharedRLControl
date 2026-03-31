"""
Diverse Human Input Model for Safety Shield Training.

Generates multi-modal velocity commands covering the full 3D action space,
enabling direction-invariant safety shield training.

Supported modes:
  - perlin_3d: Independent 3-channel Perlin noise (vx, vy, vz all vary)
  - straight:  Random constant direction + speed (tests steady-state tracking)
  - arc:       Circular arc in XY plane (tests smooth turning)
  - hover:     Near-zero velocity (tests zero-input behavior)
"""

import torch
import math
import logging
from typing import Optional, Literal, TYPE_CHECKING

from src.core.profiler import get_profiler
from src.core.user_model import BatchedPerlinNoise, InterpType

if TYPE_CHECKING:
    from src.datasets.trajectory_dataset import TrajectoryDataset


def batched_perlin_noise(
    channels: int,
    time: torch.Tensor,
    seeds: torch.Tensor,
    freq: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate batched 1D Perlin-like Gradient Noise (multi-channel).

    Args:
        channels: number of independent noise channels
        time: (N, T)
        seeds: (N, channels)
        freq: (N, 1) frequency scaling
        device: torch device

    Returns:
        noise: (N, T, channels) in approx [-1, 1]
    """
    N, T = time.shape
    noise = torch.zeros(N, T, channels, device=device)

    for ch in range(channels):
        perlin = BatchedPerlinNoise(
            seeds=seeds[:, ch],
            amplitude=1.0,
            frequency=1.0,
            octaves=1,
            interp=InterpType.COSINE,
            use_fade=False,
            device=device,
        )
        scaled_time = time * freq
        noise[:, :, ch] = perlin.get(scaled_time)

    return noise


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
MODE_PERLIN_3D = 0
MODE_STRAIGHT = 1
MODE_ARC = 2
MODE_HOVER = 3
NUM_MODES = 4

# Default mode probabilities (can be overridden via config)
DEFAULT_MODE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]


class UserModelDiverse:
    """
    Multi-modal human input generator for direction-invariant safety shield training.

    Each episode, every env randomly selects a command mode from:
      perlin_3d | straight | arc | hover

    All outputs are body-frame velocity commands (vx, vy, vz).
    """

    def __init__(
        self,
        num_envs: int,
        cfg,
        logger=None,
    ):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = cfg.device
        self.dt = cfg.sim.dt

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("user_model_diverse_null")
            self.logger.addHandler(logging.NullHandler())

        # Action limits
        self.max_speed = cfg.algo.actor.action_limit        # xy max (m/s)
        self.max_speed_z = cfg.user_model.get("max_speed_z", self.max_speed * 0.5)
        self.z_tilt_compensation = cfg.user_model.get("z_tilt_compensation", 0.0)

        # Buffer
        self.buffer_size = cfg.algo.training_frame_num  # e.g. 128 frames

        # Style knobs (Perlin noise params)
        self.freq_base = cfg.user_model.style.frequency_base
        self.freq_scale = cfg.user_model.style.frequency_scale

        # Mode weights (probabilities for each mode)
        mode_weights = cfg.user_model.get("mode_weights", DEFAULT_MODE_WEIGHTS)
        self.mode_weights = torch.tensor(mode_weights, dtype=torch.float32, device=self.device)
        self.mode_weights = self.mode_weights / self.mode_weights.sum()

        # --------------- Per-env state ---------------
        self.action_buffer = torch.zeros(num_envs, self.buffer_size, 3, device=self.device)
        self.buffer_read_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.noise_time = torch.zeros(num_envs, device=self.device)
        self.env_mode = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Perlin seeds: 3 channels for vx, vy, vz
        self.noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=self.device)
        self.styles = {
            "noise_freq": torch.rand(num_envs, 1, device=self.device) * self.freq_scale + self.freq_base,
        }

        # Straight mode: random direction & speed
        self.straight_vel = torch.zeros(num_envs, 3, device=self.device)

        # Arc mode: radius, angular speed, phase
        self.arc_radius = torch.zeros(num_envs, device=self.device)
        self.arc_omega = torch.zeros(num_envs, device=self.device)
        self.arc_phase = torch.zeros(num_envs, device=self.device)
        self.arc_vz = torch.zeros(num_envs, device=self.device)

        # Previous action for continuity
        self.prev_filtered_action = torch.zeros(num_envs, 3, device=self.device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, pos, quat, env_ids, seed=None):
        """Reset state for given env_ids and refill command buffers."""
        if pos.ndim == 3:
            pos = pos.squeeze(1)
        if quat.ndim == 3:
            quat = quat.squeeze(1)

        K = len(env_ids)

        self.noise_time[env_ids] = 0
        self.buffer_read_idx[env_ids] = 0
        self.prev_filtered_action[env_ids] = 0.0

        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            self.noise_seeds[env_ids] = (
                torch.randint(0, 100000, (K, 3), generator=gen, device=self.device)
                + env_ids.unsqueeze(1)
            )
            self.styles["noise_freq"][env_ids] = (
                torch.rand(K, 1, generator=gen, device=self.device) * self.freq_scale + self.freq_base
            )
            mode_probs = self.mode_weights.unsqueeze(0).expand(K, -1)
            self.env_mode[env_ids] = torch.multinomial(mode_probs, 1, generator=gen).squeeze(-1)
        else:
            self.noise_seeds[env_ids] = torch.randint(0, 100000, (K, 3), device=self.device)
            self.styles["noise_freq"][env_ids] = (
                torch.rand(K, 1, device=self.device) * self.freq_scale + self.freq_base
            )
            mode_probs = self.mode_weights.unsqueeze(0).expand(K, -1)
            self.env_mode[env_ids] = torch.multinomial(mode_probs, 1).squeeze(-1)

        # Initialize mode-specific params
        self._init_mode_params(env_ids, seed)

        # Fill command buffer
        self._refill_buffer(env_ids)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, drone_state, drone_pos_w):
        """Read next action from buffer; refill when exhausted."""
        profiler = get_profiler()
        profiler.start("user_model/step")

        needs_refill = self.buffer_read_idx >= self.buffer_size
        if needs_refill.any():
            with profiler.timer("user_model/refill_buffer"):
                idxs = needs_refill.nonzero(as_tuple=False).squeeze(-1)
                self._refill_buffer(idxs)
                self.buffer_read_idx[idxs] = 0

        read_indices = self.buffer_read_idx.view(-1, 1, 1).expand(-1, 1, 3)
        action = torch.gather(self.action_buffer, 1, read_indices).squeeze(1)
        self.buffer_read_idx += 1

        profiler.stop("user_model/step")
        return action, needs_refill

    # ------------------------------------------------------------------
    # Mode parameter initialization
    # ------------------------------------------------------------------
    def _init_mode_params(self, env_ids, seed=None):
        K = len(env_ids)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed + 12345)

        def _rand(shape):
            if gen is not None:
                return torch.rand(shape, generator=gen, device=self.device)
            return torch.rand(shape, device=self.device)

        # Straight mode params
        mask_straight = self.env_mode[env_ids] == MODE_STRAIGHT
        if mask_straight.any():
            ids_s = mask_straight.nonzero(as_tuple=False).squeeze(-1)
            n_s = ids_s.shape[0]
            # Random direction on unit sphere
            theta = _rand(n_s) * 2 * math.pi
            phi = torch.acos(1 - 2 * _rand(n_s))  # uniform on sphere
            speed = (0.3 + _rand(n_s) * 0.7) * self.max_speed
            self.straight_vel[env_ids[ids_s], 0] = speed * torch.sin(phi) * torch.cos(theta)
            self.straight_vel[env_ids[ids_s], 1] = speed * torch.sin(phi) * torch.sin(theta)
            self.straight_vel[env_ids[ids_s], 2] = speed * torch.cos(phi) * (self.max_speed_z / self.max_speed)

        # Arc mode params
        mask_arc = self.env_mode[env_ids] == MODE_ARC
        if mask_arc.any():
            ids_a = mask_arc.nonzero(as_tuple=False).squeeze(-1)
            n_a = ids_a.shape[0]
            # Arc radius → tangential speed
            self.arc_radius[env_ids[ids_a]] = 1.0 + _rand(n_a) * 4.0  # 1-5m radius
            # Angular speed: v_tangential / r ∈ [0.3, 1.0] * max_speed
            tang_speed = (0.3 + _rand(n_a) * 0.7) * self.max_speed
            self.arc_omega[env_ids[ids_a]] = tang_speed / self.arc_radius[env_ids[ids_a]]
            # Random sign for CW/CCW
            sign = ((_rand(n_a) > 0.5).float() * 2 - 1)
            self.arc_omega[env_ids[ids_a]] *= sign
            self.arc_phase[env_ids[ids_a]] = _rand(n_a) * 2 * math.pi
            # Gentle z oscillation
            self.arc_vz[env_ids[ids_a]] = (_rand(n_a) - 0.5) * self.max_speed_z * 0.3

    # ------------------------------------------------------------------
    # Buffer filling (dispatches per mode)
    # ------------------------------------------------------------------
    def _refill_buffer(self, env_ids):
        K = len(env_ids)
        T = self.buffer_size
        dt = self.dt
        modes = self.env_mode[env_ids]

        # Allocate output
        vels = torch.zeros(K, T, 3, device=self.device)

        # --- perlin_3d ---
        mask_p = modes == MODE_PERLIN_3D
        if mask_p.any():
            ids_p = mask_p.nonzero(as_tuple=False).squeeze(-1)
            vels[ids_p] = self._gen_perlin_3d(env_ids[ids_p], T, dt)

        # --- straight ---
        mask_s = modes == MODE_STRAIGHT
        if mask_s.any():
            ids_s = mask_s.nonzero(as_tuple=False).squeeze(-1)
            vels[ids_s] = self._gen_straight(env_ids[ids_s], T)

        # --- arc ---
        mask_a = modes == MODE_ARC
        if mask_a.any():
            ids_a = mask_a.nonzero(as_tuple=False).squeeze(-1)
            vels[ids_a] = self._gen_arc(env_ids[ids_a], T, dt)

        # --- hover ---
        mask_h = modes == MODE_HOVER
        if mask_h.any():
            ids_h = mask_h.nonzero(as_tuple=False).squeeze(-1)
            vels[ids_h] = self._gen_hover(env_ids[ids_h], T)

        self.action_buffer[env_ids] = vels
        self.noise_time[env_ids] += T * dt

    # ------------------------------------------------------------------
    # Mode generators
    # ------------------------------------------------------------------
    def _gen_perlin_3d(self, global_ids, T, dt):
        """3-channel independent Perlin noise → (K, T, 3)."""
        K = len(global_ids)
        t_start = self.noise_time[global_ids].unsqueeze(1)
        t_steps = torch.arange(T, device=self.device).unsqueeze(0) * dt
        time_grid = t_start + t_steps

        seeds = self.noise_seeds[global_ids]  # (K, 3)
        freq = self.styles["noise_freq"][global_ids]  # (K, 1)

        noise = batched_perlin_noise(3, time_grid, seeds, freq, self.device)  # (K, T, 3)

        # Scale: xy by max_speed, z by max_speed_z
        scale = torch.tensor(
            [self.max_speed, self.max_speed, self.max_speed_z], device=self.device
        )
        vels = noise * scale

        if self.z_tilt_compensation:
            vels[:, :, 2] += self.z_tilt_compensation

        return vels

    def _gen_straight(self, global_ids, T):
        """Constant velocity in random direction → (K, T, 3)."""
        K = len(global_ids)
        v = self.straight_vel[global_ids]  # (K, 3)
        return v.unsqueeze(1).expand(K, T, 3).clone()

    def _gen_arc(self, global_ids, T, dt):
        """Circular arc in body frame → (K, T, 3)."""
        K = len(global_ids)
        omega = self.arc_omega[global_ids]  # (K,)
        phase = self.arc_phase[global_ids]
        radius = self.arc_radius[global_ids]
        vz_base = self.arc_vz[global_ids]

        t_start = self.noise_time[global_ids]
        t_steps = torch.arange(T, device=self.device).float()  # (T,)
        t = t_start.unsqueeze(1) + t_steps.unsqueeze(0) * dt  # (K, T)
        angle = phase.unsqueeze(1) + omega.unsqueeze(1) * t  # (K, T)

        # Tangential velocity in body frame
        tang_speed = (omega * radius).abs()  # (K,)
        vx = tang_speed.unsqueeze(1) * torch.cos(angle)
        vy = tang_speed.unsqueeze(1) * torch.sin(angle)
        vz = vz_base.unsqueeze(1).expand(K, T)

        # Clamp to action limits
        vx = vx.clamp(-self.max_speed, self.max_speed)
        vy = vy.clamp(-self.max_speed, self.max_speed)
        vz = vz.clamp(-self.max_speed_z, self.max_speed_z)

        return torch.stack([vx, vy, vz], dim=-1)

    def _gen_hover(self, global_ids, T):
        """Near-zero velocity with tiny noise → (K, T, 3)."""
        K = len(global_ids)
        noise = torch.randn(K, T, 3, device=self.device) * 0.05 * self.max_speed
        noise[:, :, 2] *= self.max_speed_z / (self.max_speed + 1e-6)
        return noise
