"""
Standalone UserModelTunnel for ROS1 deployment.
Generates forward velocity commands (body-frame) for the residual RL policy.

Two modes:
  - simple_mode: constant forward speed (vx=max_speed, vy=0, vz=0)
  - online_mode: Perlin-noise-driven speed with lateral variation

No Isaac Sim / OmniDrones dependency.
"""
import torch
import math
from enum import Enum


# =====================================================================
# Perlin noise (ported from src/core/user_model.py)
# =====================================================================
class InterpType(Enum):
    LINEAR = 1
    COSINE = 2
    CUBIC = 3


class BatchedPerlinNoise:
    """GPU-batched 1-D Perlin gradient noise."""

    def __init__(self, seeds, amplitude=1.0, frequency=1.0, octaves=1,
                 interp=InterpType.COSINE, use_fade=False, device=None):
        self.device = device if device is not None else seeds.device
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.interp = interp
        self.use_fade = use_fade
        self.seeds = seeds.to(self.device).float()

    def _noise(self, x):
        combined = self.seeds.unsqueeze(1) + x.float()
        h = torch.sin(combined * 12.9898 + 78.233) * 43758.5453
        h = h - h.floor()
        return h * 2.0 - 1.0

    def _fade(self, x):
        return (6 * x**5) - (15 * x**4) + (10 * x**3)

    def _cosine_interp(self, a, b, x):
        x2 = (1 - torch.cos(x * math.pi)) / 2
        return a * (1 - x2) + b * x2

    def _linear_interp(self, a, b, x):
        return a + x * (b - a)

    def _interpolated_noise(self, x):
        prev_x = x.floor().long()
        next_x = prev_x + 1
        frac_x = x - prev_x.float()
        if self.use_fade:
            frac_x = self._fade(frac_x)
        if self.interp == InterpType.LINEAR:
            return self._linear_interp(self._noise(prev_x), self._noise(next_x), frac_x)
        else:  # COSINE (default)
            return self._cosine_interp(self._noise(prev_x), self._noise(next_x), frac_x)

    def get(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(1)
            squeeze = True
        freq, amp = self.frequency, self.amplitude
        result = torch.zeros_like(x)
        for _ in range(self.octaves):
            result += self._interpolated_noise(x * freq) * amp
            freq *= 2
            amp /= 2
        if squeeze:
            result = result.squeeze(1)
        return result


def _batched_perlin_noise(channels, time, seeds, freq, device):
    N, T = time.shape
    noise = torch.zeros(N, T, channels, device=device)
    for ch in range(channels):
        perlin = BatchedPerlinNoise(
            seeds=seeds[:, ch], amplitude=1.0, frequency=1.0,
            interp=InterpType.COSINE, device=device,
        )
        noise[:, :, ch] = perlin.get(time * freq)
    return noise


# =====================================================================
# User model
# =====================================================================
class UserModelTunnel:
    """
    Generates forward velocity commands for tunnel deployment.

    Args:
        max_speed:      Action limit (m/s), default 2.0
        dt:             Time step (seconds), default 0.05 (20 Hz)
        buffer_size:    How many steps of action to pre-generate, default 128
        simple_mode:    If True, use constant-speed forward command
        freq_base/freq_scale: Perlin noise frequency params
        device:         torch device
    """

    def __init__(
        self,
        max_speed: float = 2.0,
        dt: float = 0.05,
        buffer_size: int = 128,
        simple_mode: bool = True,
        profile: str = "m3_diverse",
        freq_base: float = 0.1,
        freq_scale: float = 0.2,
        vx_bias: float = 1.5,
        vx_amp: float = 0.5,
        vy_amp: float = 2.0,
        vz_amp: float = 0.0,
        smoothness_base: float = 0.4,
        smoothness_scale: float = 0.5,
        laziness: float = 0.3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.max_speed = max_speed
        self.dt = dt
        self.buffer_size = buffer_size
        requested_profile = str(profile)
        self.simple_mode = simple_mode or requested_profile == "simple"
        self.profile = "simple" if self.simple_mode else requested_profile
        if self.profile == "perlin":
            self.profile = "legacy_perlin"
        if self.profile not in ("simple", "legacy_perlin", "m3_diverse"):
            raise ValueError(
                f"Unsupported UserModelTunnel profile '{self.profile}'. "
                "Expected simple, legacy_perlin, or m3_diverse."
            )
        self.num_channels = 1 if self.profile == "legacy_perlin" else 3

        # State (batch size = 1 for ROS single-drone)
        N = 1
        self.action_buffer = torch.zeros(N, buffer_size, 3, device=self.device)
        self.buffer_read_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self.noise_time = torch.zeros(N, device=self.device)
        self.noise_seeds = torch.randint(0, 100000, (N, self.num_channels), device=self.device)

        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.noise_freq = torch.rand(N, 1, device=self.device) * freq_scale + freq_base
        self.vx_bias = vx_bias
        self.vx_amp = vx_amp
        self.vy_amp = vy_amp
        self.vz_amp = vz_amp
        self.smoothness_base = smoothness_base
        self.smoothness_scale = smoothness_scale
        self.laziness_max = laziness
        self.smoothness = torch.zeros(N, 1, device=self.device)
        self.laziness = torch.zeros(N, 1, device=self.device)
        self.prev_action = torch.zeros(N, 3, device=self.device)

    def reset(self, seed: int = None):
        """Reset for a new episode."""
        N = 1
        self.buffer_read_idx.zero_()
        self.noise_time.zero_()
        self.prev_action.zero_()

        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            self.noise_seeds = torch.randint(
                0, 100000, (N, self.num_channels), generator=gen, device=self.device
            )
            self.noise_freq = (
                torch.rand(N, 1, generator=gen, device=self.device) * self.freq_scale
                + self.freq_base
            )
            self.smoothness = (
                torch.rand(N, 1, generator=gen, device=self.device) * self.smoothness_scale
                + self.smoothness_base
            )
            self.laziness = torch.rand(N, 1, generator=gen, device=self.device) * self.laziness_max
        else:
            self.noise_seeds = torch.randint(0, 100000, (N, self.num_channels), device=self.device)
            self.noise_freq = torch.rand(N, 1, device=self.device) * self.freq_scale + self.freq_base
            self.smoothness = torch.rand(N, 1, device=self.device) * self.smoothness_scale + self.smoothness_base
            self.laziness = torch.rand(N, 1, device=self.device) * self.laziness_max

        if not self.simple_mode:
            self._refill_buffer()

    def step(self) -> torch.Tensor:
        """
        Return one (1, 3) body-frame velocity command.
        """
        if self.simple_mode:
            return self._step_simple()
        return self._step_online()

    def _step_simple(self) -> torch.Tensor:
        """Constant forward speed: vx=max_speed, vy=0, vz=0."""
        return torch.tensor(
            [[self.max_speed, 0.0, 0.0]], device=self.device, dtype=torch.float32
        )

    def _step_online(self) -> torch.Tensor:
        """Buffer-based Perlin noise command generation."""
        if self.buffer_read_idx[0] >= self.buffer_size:
            self._refill_buffer()
            self.buffer_read_idx.zero_()

        idx = self.buffer_read_idx[0].item()
        action = self.action_buffer[:, idx, :]  # (1, 3)
        self.buffer_read_idx += 1
        return action

    def _refill_buffer(self):
        if self.profile == "legacy_perlin":
            self._refill_legacy_perlin()
            return

        T = self.buffer_size
        dt = self.dt
        N = 1

        t_start = self.noise_time.unsqueeze(1)
        t_steps = torch.arange(T, device=self.device).unsqueeze(0) * dt
        time_grid = t_start + t_steps

        channel_noise = _batched_perlin_noise(
            self.num_channels, time_grid, self.noise_seeds,
            self.noise_freq, self.device,
        )  # (1, T, 3)

        amps = torch.tensor(
            [self.vx_amp, self.vy_amp, self.vz_amp],
            device=self.device,
            dtype=torch.float32,
        )
        bias = torch.tensor(
            [self.vx_bias, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        raw = channel_noise * amps + bias
        raw = torch.where(raw.abs() < self.laziness.unsqueeze(1), torch.zeros_like(raw), raw)

        alpha = 1.0 - self.smoothness
        last = self.prev_action.clone()
        for i in range(T):
            last = alpha * raw[:, i, :] + (1.0 - alpha) * last
            self.action_buffer[:, i, :] = last

        self.action_buffer[..., 0] = self.action_buffer[..., 0].clamp(0.0, self.max_speed)
        self.action_buffer[..., 1] = self.action_buffer[..., 1].clamp(-self.max_speed, self.max_speed)
        self.action_buffer[..., 2] = self.action_buffer[..., 2].clamp(-self.max_speed, self.max_speed)
        self.prev_action[:] = self.action_buffer[:, -1, :]
        self.noise_time += T * dt

    def _refill_legacy_perlin(self):
        T = self.buffer_size
        dt = self.dt
        N = 1

        t_start = self.noise_time.unsqueeze(1)
        t_steps = torch.arange(T, device=self.device).unsqueeze(0) * dt
        time_grid = t_start + t_steps

        channel_noise = _batched_perlin_noise(
            self.num_channels, time_grid, self.noise_seeds,
            self.noise_freq, self.device,
        )  # (1, T, 1)

        scale = torch.tensor(
            [self.max_speed, self.max_speed, 0.0], device=self.device
        )

        noise_expanded = channel_noise.unsqueeze(-1)
        target_vels = torch.cat([
            torch.ones_like(noise_expanded),
            noise_expanded,
            torch.zeros_like(noise_expanded),
        ], dim=-1)  # (1, T, 1, 3)

        target_vels = (target_vels * scale).squeeze(2)  # (1, T, 3)
        self.action_buffer[:] = target_vels
        self.prev_action[:] = target_vels[:, -1, :]
        self.noise_time += T * dt
