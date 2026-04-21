#!/usr/bin/env python3
"""
Offline Trajectory Generator for Drone Action Trajectories.

This script generates pre-computed drone action trajectories using Perlin noise
and stores them in HDF5 format for efficient loading during training.

Features:
- GPU-batched Perlin noise generation (fast)
- CPU-based third-party library generation with multiprocessing (fallback)
- Full offline processing: Perlin → Deadband → LPF → Position integration → APF → Clamping
- Multi-GPU support via torch.distributed
- Stratified style parameter sampling for diversity

Usage:
    # GPU batched generation (fast)
    python trajectory_generator.py --config-name=trajectory_gen backend=batched
    
    # CPU library generation with multiprocessing
    python trajectory_generator.py --config-name=trajectory_gen backend=library num_workers=8
    
    # Multi-GPU generation
    torchrun --nproc_per_node=4 trajectory_generator.py --config-name=trajectory_gen backend=batched
"""

import os
import sys
import math
import argparse
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf, DictConfig

from trajectory_dataset import create_trajectory_dataset, TrajectoryMetadata


# ============================================================================
# Perlin Noise Implementations
# ============================================================================

class InterpType(Enum):
    LINEAR = 1
    COSINE = 2
    CUBIC = 3


class BatchedPerlinNoise:
    """
    GPU-batched Perlin Noise generator using PyTorch.
    Copied from user_model.py for standalone usage.
    """
    
    def __init__(
        self,
        seeds: torch.Tensor,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        octaves: int = 1,
        interp: InterpType = InterpType.COSINE,
        use_fade: bool = False,
        device: torch.device = None
    ):
        self.device = device if device is not None else seeds.device
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.interp = interp
        self.use_fade = use_fade
        self.seeds = seeds.to(self.device).float()
    
    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        combined = self.seeds.unsqueeze(1) + x.float()
        h = torch.sin(combined * 12.9898 + 78.233) * 43758.5453
        h = h - h.floor()
        return h * 2.0 - 1.0
    
    def _fade(self, x: torch.Tensor) -> torch.Tensor:
        return (6 * x**5) - (15 * x**4) + (10 * x**3)
    
    def _linear_interp(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return a + x * (b - a)
    
    def _cosine_interp(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x2 = (1 - torch.cos(x * math.pi)) / 2
        return a * (1 - x2) + b * x2
    
    def _cubic_interp(
        self, v0: torch.Tensor, v1: torch.Tensor,
        v2: torch.Tensor, v3: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        p = (v3 - v2) - (v0 - v1)
        q = (v0 - v1) - p
        r = v2 - v0
        s = v1
        return p * x**3 + q * x**2 + r * x + s
    
    def _interpolated_noise(self, x: torch.Tensor) -> torch.Tensor:
        prev_x = x.floor().long()
        next_x = prev_x + 1
        frac_x = x - prev_x.float()
        
        if self.use_fade:
            frac_x = self._fade(frac_x)
        
        if self.interp == InterpType.LINEAR:
            result = self._linear_interp(
                self._noise(prev_x), self._noise(next_x), frac_x
            )
        elif self.interp == InterpType.COSINE:
            result = self._cosine_interp(
                self._noise(prev_x), self._noise(next_x), frac_x
            )
        else:
            result = self._cubic_interp(
                self._noise(prev_x - 1), self._noise(prev_x),
                self._noise(next_x), self._noise(next_x + 1), frac_x
            )
        return result
    
    def get(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        frequency = self.frequency
        amplitude = self.amplitude
        result = torch.zeros_like(x)
        
        for _ in range(self.octaves):
            result = result + self._interpolated_noise(x * frequency) * amplitude
            frequency *= 2
            amplitude /= 2
        
        if squeeze_output:
            result = result.squeeze(1)
        return result


def batched_perlin_noise_gpu(
    time: torch.Tensor,
    seeds: torch.Tensor,
    freq: torch.Tensor,
    num_channels: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generate batched 1D Perlin noise on GPU.
    
    Args:
        time: (N, T) time tensor
        seeds: (N, num_channels) seeds for each channel
        freq: (N, 1) frequency scaling
        num_channels: number of output channels (3 or 4)
        device: torch device
        
    Returns:
        noise: (N, T, num_channels) values in [-1, 1]
    """
    N, T = time.shape
    noise = torch.zeros(N, T, num_channels, device=device)
    
    for ch in range(num_channels):
        channel_seeds = seeds[:, ch]
        perlin = BatchedPerlinNoise(
            seeds=channel_seeds,
            amplitude=1.0,
            frequency=1.0,
            octaves=1,
            interp=InterpType.COSINE,
            use_fade=False,
            device=device
        )
        scaled_time = time * freq
        noise[:, :, ch] = perlin.get(scaled_time)
    
    return noise


def generate_single_trajectory_cpu(
    args: Tuple[int, int, float, float, float, float, float, float, float, float, Tuple[float, ...], int]
) -> Dict[str, np.ndarray]:
    """
    Generate a single trajectory using CPU-based Perlin noise library.
    
    This function is designed to be called via multiprocessing.
    
    Args:
        args: Tuple of (traj_idx, T, dt, noise_freq, smoothness, laziness,
                        max_speed, max_speed_z, max_speed_yaw, repulsive_gain,
                        map_bounds, action_dim)
    
    Returns:
        dict with 'velocities', 'positions', 'bbox', 'styles'
    """
    # Import here to avoid issues with multiprocessing
    from src.third_party.perlin_noise import PerlinNoise, Interp  # type: ignore
    
    (traj_idx, T, dt, noise_freq, smoothness, laziness,
     max_speed, max_speed_z, max_speed_yaw, repulsive_gain,
     map_bounds, action_dim) = args
    
    # Create Perlin noise generators for each channel
    np.random.seed(traj_idx)
    seeds = [np.random.randint(0, 100000) for _ in range(action_dim)]
    noise_generators = [
        PerlinNoise(seed=s, amplitude=1.0, frequency=1.0, octaves=1, interp=Interp.COSINE)
        for s in seeds
    ]
    
    # Generate raw noise
    scale = [max_speed, max_speed, max_speed_z]
    if action_dim == 4:
        scale.append(max_speed_yaw)
    
    velocities = np.zeros((T, action_dim), dtype=np.float32)
    positions = np.zeros((T, 3), dtype=np.float32)
    
    # Apply filters and integrate
    alpha = 1.0 - smoothness
    curr_pos = np.array([0.0, 0.0, 2.0])  # Start at origin, z=2 for safety
    curr_yaw = 0.0
    last_filtered = np.zeros(action_dim)
    
    for t in range(T):
        time_val = t * dt * noise_freq
        
        # Get raw noise
        raw = np.array([ng.get(time_val) * s for ng, s in zip(noise_generators, scale)])
        
        # Deadband
        raw = np.where(np.abs(raw) < laziness, 0.0, raw)
        
        # Low-pass filter
        filtered = alpha * raw + (1 - alpha) * last_filtered
        last_filtered = filtered
        
        # APF repulsion
        force = np.zeros(3)
        margin = 2.0
        
        # X axis
        d_min_x = curr_pos[0] + map_bounds[0]
        if d_min_x < margin:
            force[0] += repulsive_gain * (margin - d_min_x)
        d_max_x = map_bounds[0] - curr_pos[0]
        if d_max_x < margin:
            force[0] -= repulsive_gain * (margin - d_max_x)
        
        # Y axis
        d_min_y = curr_pos[1] + map_bounds[1]
        if d_min_y < margin:
            force[1] += repulsive_gain * (margin - d_min_y)
        d_max_y = map_bounds[1] - curr_pos[1]
        if d_max_y < margin:
            force[1] -= repulsive_gain * (margin - d_max_y)
        
        # Z axis (floor at 1.0, ceiling at 2 * map_bounds[2])
        d_floor = curr_pos[2] - 1.0
        if d_floor < margin:
            force[2] += repulsive_gain * (margin - d_floor)
        d_ceil = 2 * map_bounds[2] - curr_pos[2]
        if d_ceil < margin:
            force[2] -= repulsive_gain * (margin - d_ceil)
        
        # Apply force to velocity (in local frame, simplified - no rotation for generation)
        vel_modified = filtered.copy()
        vel_modified[:3] += force
        
        # Integrate position
        next_pos = curr_pos + vel_modified[:3] * dt
        
        # Clamp position
        next_pos[0] = np.clip(next_pos[0], -map_bounds[0], map_bounds[0])
        next_pos[1] = np.clip(next_pos[1], -map_bounds[1], map_bounds[1])
        next_pos[2] = np.clip(next_pos[2], 1.0, 2 * map_bounds[2])
        
        # Effective velocity
        effective_vel = (next_pos - curr_pos) / dt
        effective_vel = np.clip(effective_vel, -max_speed * 2, max_speed * 2)
        
        # Store
        vel_modified[:3] = effective_vel
        velocities[t] = vel_modified
        positions[t] = next_pos
        
        curr_pos = next_pos
        if action_dim == 4:
            curr_yaw += vel_modified[3] * dt
    
    # Compute bbox
    bbox = np.array([
        positions[:, 0].min(), positions[:, 1].min(), positions[:, 2].min(),
        positions[:, 0].max(), positions[:, 1].max(), positions[:, 2].max()
    ], dtype=np.float32)
    
    return {
        'velocities': velocities,
        'positions': positions,
        'bbox': bbox,
        'styles': {
            'noise_freq': noise_freq,
            'smoothness': smoothness,
            'laziness': laziness,
        }
    }


# ============================================================================
# GPU-based Trajectory Generation
# ============================================================================

def generate_trajectories_gpu(
    num_trajectories: int,
    trajectory_length: int,
    action_dim: int,
    dt: float,
    max_speed: float,
    max_speed_z: float,
    max_speed_yaw: float,
    repulsive_gain: float,
    map_bounds: Tuple[float, float, float],
    style_config: DictConfig,
    device: torch.device,
    batch_size: int = 1024,
    rank: int = 0,
    world_size: int = 1,
    directional_bias: Optional[DictConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate trajectories using GPU-batched Perlin noise.
    
    Args:
        num_trajectories: Total number of trajectories to generate
        trajectory_length: Length of each trajectory
        action_dim: Action dimension (3 or 4)
        dt: Time step
        max_speed: Maximum XY speed
        max_speed_z: Maximum Z speed
        max_speed_yaw: Maximum yaw rate
        repulsive_gain: APF repulsive gain
        map_bounds: Map half-extents (x, y, z)
        style_config: Style parameter configuration
        device: Torch device
        batch_size: Batch size for generation
        rank: Distributed rank
        world_size: Total number of processes
        
    Returns:
        velocities: (N, T, D) numpy array
        positions: (N, T, 3) numpy array
        bboxes: (N, 6) numpy array
        styles: dict of numpy arrays
    """
    # Calculate this rank's share of trajectories
    trajs_per_rank = num_trajectories // world_size
    start_idx = rank * trajs_per_rank
    end_idx = start_idx + trajs_per_rank if rank < world_size - 1 else num_trajectories
    local_num_trajs = end_idx - start_idx
    
    T = trajectory_length
    D = action_dim
    
    # Pre-allocate output arrays
    all_velocities = np.zeros((local_num_trajs, T, D), dtype=np.float32)
    all_positions = np.zeros((local_num_trajs, T, 3), dtype=np.float32)
    all_bboxes = np.zeros((local_num_trajs, 6), dtype=np.float32)
    all_styles = {
        'noise_freq': np.zeros(local_num_trajs, dtype=np.float32),
        'smoothness': np.zeros(local_num_trajs, dtype=np.float32),
        'laziness': np.zeros(local_num_trajs, dtype=np.float32),
    }
    
    map_bounds_t = torch.tensor(map_bounds, device=device, dtype=torch.float32)
    scale = torch.tensor(
        [max_speed, max_speed, max_speed_z] + ([max_speed_yaw] if D == 4 else []),
        device=device
    )

    # === Optional directional bias (e.g., for tunnel-like tasks that
    # require sustained forward x progress). When `directional_bias` is
    # set, every channel can be optionally rescaled (`amp_scale`) and
    # additively biased (`bias`, in m/s). Lengths must be either D, or
    # padded with zeros. Default keeps the original zero-mean behavior. ===
    if directional_bias is not None:
        bias_list = list(directional_bias.get("bias", [0.0] * D))
        amp_list = list(directional_bias.get("amp_scale", [1.0] * D))
        # Pad / truncate to D
        bias_list = (bias_list + [0.0] * D)[:D]
        amp_list = (amp_list + [1.0] * D)[:D]
        bias_t = torch.tensor(bias_list, device=device, dtype=torch.float32)
        amp_t = torch.tensor(amp_list, device=device, dtype=torch.float32)
    else:
        bias_t = torch.zeros(D, device=device, dtype=torch.float32)
        amp_t = torch.ones(D, device=device, dtype=torch.float32)
    
    # Generate in batches
    num_batches = (local_num_trajs + batch_size - 1) // batch_size
    
    pbar = tqdm(range(num_batches), desc=f"[GPU {rank}] Generating trajectories", disable=rank != 0)
    
    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, local_num_trajs)
        B = batch_end - batch_start
        
        # Sample style parameters
        noise_freq = (torch.rand(B, 1, device=device) * style_config.frequency_scale 
                      + style_config.frequency_base)
        smoothness = (torch.rand(B, 1, device=device) * style_config.smoothness_scale 
                      + style_config.smoothness_base)
        laziness = torch.rand(B, 1, device=device) * style_config.laziness
        
        # Generate random seeds
        seeds = torch.randint(0, 100000, (B, D), device=device)
        
        # Time grid
        t_steps = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0) * dt  # (1, T)
        time_grid = t_steps.expand(B, -1)  # (B, T)
        
        # Generate raw Perlin noise
        raw_noise = batched_perlin_noise_gpu(time_grid, seeds, noise_freq, D, device)  # (B, T, D)
        # Per-channel amplitude scaling + additive directional bias (m/s)
        target_vels = raw_noise * scale * amp_t + bias_t  # Scale to physical units
        
        # Apply filters and integrate
        alpha = 1.0 - smoothness  # (B, 1)
        
        # Initialize state
        filtered_traj = torch.zeros(B, T, D, device=device)
        positions_traj = torch.zeros(B, T, 3, device=device)
        
        last_val = torch.zeros(B, D, device=device)
        curr_pos = torch.zeros(B, 3, device=device)
        curr_pos[:, 2] = 2.0  # Start at z=2 for safety
        
        # Sequential loop for integration (necessary for position tracking)
        for t in range(T):
            raw_v = target_vels[:, t]  # (B, D)
            
            # Deadband
            mask_dead = raw_v.abs() < laziness
            raw_v = torch.where(mask_dead, torch.zeros_like(raw_v), raw_v)
            
            # Low-pass filter
            curr_v = alpha * raw_v + (1.0 - alpha) * last_val
            last_val = curr_v
            
            # APF repulsion
            force = torch.zeros_like(curr_pos)
            margin = 2.0
            
            # X axis
            d_min_x = curr_pos[:, 0] + map_bounds_t[0]
            mask = d_min_x < margin
            force[mask, 0] += repulsive_gain * (margin - d_min_x[mask])
            
            d_max_x = map_bounds_t[0] - curr_pos[:, 0]
            mask = d_max_x < margin
            force[mask, 0] -= repulsive_gain * (margin - d_max_x[mask])
            
            # Y axis
            d_min_y = curr_pos[:, 1] + map_bounds_t[1]
            mask = d_min_y < margin
            force[mask, 1] += repulsive_gain * (margin - d_min_y[mask])
            
            d_max_y = map_bounds_t[1] - curr_pos[:, 1]
            mask = d_max_y < margin
            force[mask, 1] -= repulsive_gain * (margin - d_max_y[mask])
            
            # Z axis
            d_floor = curr_pos[:, 2] - 1.0
            mask = d_floor < margin
            force[mask, 2] += repulsive_gain * (margin - d_floor[mask])
            
            d_ceil = 2 * map_bounds_t[2] - curr_pos[:, 2]
            mask = d_ceil < margin
            force[mask, 2] -= repulsive_gain * (margin - d_ceil[mask])
            
            # Apply force
            curr_v_modified = curr_v.clone()
            curr_v_modified[:, :3] += force
            
            # Integrate position
            next_pos = curr_pos + curr_v_modified[:, :3] * dt
            
            # Clamp position
            next_pos[:, 0] = torch.clamp(next_pos[:, 0], -map_bounds_t[0], map_bounds_t[0])
            next_pos[:, 1] = torch.clamp(next_pos[:, 1], -map_bounds_t[1], map_bounds_t[1])
            next_pos[:, 2] = torch.clamp(next_pos[:, 2], 1.0, 2 * map_bounds_t[2])
            
            # Effective velocity
            effective_vel = (next_pos - curr_pos) / dt
            limit = max_speed * 2.0
            effective_vel = torch.clamp(effective_vel, -limit, limit)
            
            curr_v_modified[:, :3] = effective_vel
            
            # Store
            filtered_traj[:, t] = curr_v_modified
            positions_traj[:, t] = next_pos
            
            curr_pos = next_pos
        
        # Compute bboxes
        pos_min = positions_traj.min(dim=1).values  # (B, 3)
        pos_max = positions_traj.max(dim=1).values  # (B, 3)
        bboxes = torch.cat([pos_min, pos_max], dim=-1)  # (B, 6)
        
        # Store results
        all_velocities[batch_start:batch_end] = filtered_traj.cpu().numpy()
        all_positions[batch_start:batch_end] = positions_traj.cpu().numpy()
        all_bboxes[batch_start:batch_end] = bboxes.cpu().numpy()
        all_styles['noise_freq'][batch_start:batch_end] = noise_freq.squeeze(-1).cpu().numpy()
        all_styles['smoothness'][batch_start:batch_end] = smoothness.squeeze(-1).cpu().numpy()
        all_styles['laziness'][batch_start:batch_end] = laziness.squeeze(-1).cpu().numpy()
    
    return all_velocities, all_positions, all_bboxes, all_styles


# ============================================================================
# CPU-based Trajectory Generation with Multiprocessing
# ============================================================================

def generate_trajectories_cpu(
    num_trajectories: int,
    trajectory_length: int,
    action_dim: int,
    dt: float,
    max_speed: float,
    max_speed_z: float,
    max_speed_yaw: float,
    repulsive_gain: float,
    map_bounds: Tuple[float, float, float],
    style_config: DictConfig,
    num_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate trajectories using CPU-based Perlin noise with multiprocessing.
    """
    T = trajectory_length
    D = action_dim
    
    # Pre-allocate
    all_velocities = np.zeros((num_trajectories, T, D), dtype=np.float32)
    all_positions = np.zeros((num_trajectories, T, 3), dtype=np.float32)
    all_bboxes = np.zeros((num_trajectories, 6), dtype=np.float32)
    all_styles = {
        'noise_freq': np.zeros(num_trajectories, dtype=np.float32),
        'smoothness': np.zeros(num_trajectories, dtype=np.float32),
        'laziness': np.zeros(num_trajectories, dtype=np.float32),
    }
    
    # Prepare arguments for each trajectory
    np.random.seed(42)  # For reproducibility of style sampling
    args_list = []
    for i in range(num_trajectories):
        noise_freq = np.random.uniform(
            style_config.frequency_base,
            style_config.frequency_base + style_config.frequency_scale
        )
        smoothness = np.random.uniform(
            style_config.smoothness_base,
            style_config.smoothness_base + style_config.smoothness_scale
        )
        laziness = np.random.uniform(0, style_config.laziness)
        
        args_list.append((
            i, T, dt, noise_freq, smoothness, laziness,
            max_speed, max_speed_z, max_speed_yaw, repulsive_gain,
            tuple(map_bounds), action_dim
        ))
    
    # Generate with multiprocessing
    print(f"[CPU] Generating {num_trajectories} trajectories with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(generate_single_trajectory_cpu, args_list),
            total=num_trajectories,
            desc="Generating trajectories"
        ))
    
    # Collect results
    for i, result in enumerate(results):
        all_velocities[i] = result['velocities']
        all_positions[i] = result['positions']
        all_bboxes[i] = result['bbox']
        all_styles['noise_freq'][i] = result['styles']['noise_freq']
        all_styles['smoothness'][i] = result['styles']['smoothness']
        all_styles['laziness'][i] = result['styles']['laziness']
    
    return all_velocities, all_positions, all_bboxes, all_styles


# ============================================================================
# Main Entry Point
# ============================================================================

FILE_PATH = os.path.join(os.path.dirname(__file__), "../../configs")


@hydra.main(config_path=FILE_PATH, config_name="trajectory_gen", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for trajectory generation."""
    print("=" * 60)
    print("Offline Trajectory Generator")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # Check for distributed environment
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    if is_distributed and cfg.backend == 'batched':
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device(cfg.device if cfg.backend == 'batched' else 'cpu')
    
    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  - Backend: {cfg.backend}")
        print(f"  - Trajectories: {cfg.num_trajectories}")
        print(f"  - Length: {cfg.trajectory_length}")
        print(f"  - Action dim: {cfg.action_dim}")
        print(f"  - Output: {cfg.output_path}")
        if is_distributed:
            print(f"  - World size: {world_size}")
    
    # Generate trajectories
    if cfg.backend == 'batched':
        velocities, positions, bboxes, styles = generate_trajectories_gpu(
            num_trajectories=cfg.num_trajectories,
            trajectory_length=cfg.trajectory_length,
            action_dim=cfg.action_dim,
            dt=cfg.dt,
            max_speed=cfg.max_speed,
            max_speed_z=cfg.max_speed_z,
            max_speed_yaw=cfg.max_speed_yaw,
            repulsive_gain=cfg.repulsive_gain,
            map_bounds=tuple(cfg.reference_map_bounds),
            style_config=cfg.style,
            device=device,
            batch_size=cfg.batch_size,
            rank=rank,
            world_size=world_size,
            directional_bias=cfg.get("directional_bias", None),
        )
    else:  # library
        if cfg.get("directional_bias", None) is not None:
            print(
                "[Generator] WARNING: directional_bias is currently only "
                "implemented for backend=batched (GPU). It will be ignored "
                "by the CPU library backend."
            )
        velocities, positions, bboxes, styles = generate_trajectories_cpu(
            num_trajectories=cfg.num_trajectories,
            trajectory_length=cfg.trajectory_length,
            action_dim=cfg.action_dim,
            dt=cfg.dt,
            max_speed=cfg.max_speed,
            max_speed_z=cfg.max_speed_z,
            max_speed_yaw=cfg.max_speed_yaw,
            repulsive_gain=cfg.repulsive_gain,
            map_bounds=tuple(cfg.reference_map_bounds),
            style_config=cfg.style,
            num_workers=cfg.num_workers,
        )
    
    # For distributed: each rank saves to a temp file, rank 0 merges them
    if is_distributed and cfg.backend == 'batched':
        import time as time_module
        
        # Each rank saves its data to a temporary file
        temp_dir = os.path.dirname(cfg.output_path) or '.'
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f".temp_traj_rank{rank}.npz")
        done_file = os.path.join(temp_dir, f".temp_traj_rank{rank}.done")
        
        # Move data to CPU immediately to free GPU memory
        if isinstance(velocities, torch.Tensor):
            velocities = velocities.cpu().numpy()
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        
        # Save this rank's data to temp file
        np.savez_compressed(
            temp_file,
            velocities=velocities,
            positions=positions,
            bboxes=bboxes,
            noise_freq=styles['noise_freq'],
            smoothness=styles['smoothness'],
            laziness=styles['laziness'],
        )
        
        # Create a .done marker file to signal completion
        with open(done_file, 'w') as f:
            f.write('done')
        print(f"[GPU {rank}] Saved temp file: {temp_file}")
        
        # Destroy process group BEFORE file I/O to avoid NCCL timeout
        dist.destroy_process_group()
        print(f"[GPU {rank}] Process group destroyed, exiting...")
        
        # Only rank 0 continues to merge files
        if rank != 0:
            # Non-rank-0 processes exit here
            return
        
        # Rank 0: Wait for all ranks to finish saving (file-based sync)
        print("\n[GPU 0] Waiting for all ranks to finish saving...")
        for r in range(world_size):
            done_file_r = os.path.join(temp_dir, f".temp_traj_rank{r}.done")
            while not os.path.exists(done_file_r):
                time_module.sleep(1)
                print(f"  Waiting for rank {r}...")
        print("[GPU 0] All ranks finished saving.")
        
        # Rank 0 merges all temp files
        print("\n[GPU 0] Merging data from all ranks...")
        
        all_velocities = []
        all_positions = []
        all_bboxes = []
        all_noise_freq = []
        all_smoothness = []
        all_laziness = []
        
        for r in range(world_size):
            temp_file_r = os.path.join(temp_dir, f".temp_traj_rank{r}.npz")
            done_file_r = os.path.join(temp_dir, f".temp_traj_rank{r}.done")
            print(f"  Loading rank {r} data from {temp_file_r}...")
            data = np.load(temp_file_r)
            all_velocities.append(data['velocities'])
            all_positions.append(data['positions'])
            all_bboxes.append(data['bboxes'])
            all_noise_freq.append(data['noise_freq'])
            all_smoothness.append(data['smoothness'])
            all_laziness.append(data['laziness'])
            data.close()
            
            # Remove temp files after loading
            os.remove(temp_file_r)
            os.remove(done_file_r)
        
        # Concatenate all data
        velocities = np.concatenate(all_velocities, axis=0)
        positions = np.concatenate(all_positions, axis=0)
        bboxes = np.concatenate(all_bboxes, axis=0)
        styles = {
            'noise_freq': np.concatenate(all_noise_freq, axis=0),
            'smoothness': np.concatenate(all_smoothness, axis=0),
            'laziness': np.concatenate(all_laziness, axis=0),
        }
        
        # Free memory
        del all_velocities, all_positions, all_bboxes
        del all_noise_freq, all_smoothness, all_laziness
        
        print(f"  Merged {len(velocities)} trajectories")
        
        # Mark distributed as already cleaned up
        is_distributed = False
    
    # Save dataset (only rank 0)
    if rank == 0:
        metadata = TrajectoryMetadata(
            num_trajectories=len(velocities),
            trajectory_length=cfg.trajectory_length,
            action_dim=cfg.action_dim,
            dt=cfg.dt,
            max_speed=cfg.max_speed,
            max_speed_z=cfg.max_speed_z,
            max_speed_yaw=cfg.max_speed_yaw,
            reference_map_bounds=tuple(cfg.reference_map_bounds),
        )
        
        create_trajectory_dataset(
            output_path=cfg.output_path,
            velocities=velocities,
            positions=positions,
            bboxes=bboxes,
            styles=styles,
            metadata=metadata,
            compression=cfg.compression,
            compression_opts=cfg.compression_level,
        )
        
        print("\n" + "=" * 60)
        print("Generation complete!")
        print("=" * 60)
    
    # Cleanup distributed (only if not already cleaned up)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
