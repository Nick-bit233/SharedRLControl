"""
Trajectory Dataset Module for Offline Trajectory Sampling.

This module provides efficient storage and retrieval of pre-generated 
drone action trajectories for reinforcement learning training.

Features:
- HDF5-based storage with compression for efficient disk usage
- Auto-sized GPU cache based on available memory
- Two sampling modes: "raw" (no transforms) and "scaled" (boundary-aware)
- Support for random sub-trajectory sampling from long trajectories
"""

import torch
import numpy as np
import h5py
import os
from typing import Optional, Tuple, Dict, Union, Literal
from dataclasses import dataclass


@dataclass
class TrajectoryMetadata:
    """Metadata for trajectory dataset."""
    num_trajectories: int
    trajectory_length: int
    action_dim: int
    dt: float
    max_speed: float
    max_speed_z: float
    max_speed_yaw: float
    reference_map_bounds: Tuple[float, float, float]  # Half-extents used during generation


class TrajectoryDataset:
    """
    Dataset class for storing and sampling pre-generated trajectories.
    
    Stores fully processed trajectories (velocities + positions) offline,
    then samples random windows at runtime with optional scale-to-fit
    transformation for boundary handling.
    
    Args:
        dataset_path: Path to HDF5 file containing trajectories
        device: PyTorch device for GPU operations
        gpu_cache_reserve_gb: Amount of GPU memory to reserve for simulator/training (GB)
        min_scale_factor: Minimum allowed scale factor for scaled sampling mode
    """
    
    def __init__(
        self,
        dataset_path: str,
        device: torch.device,
        gpu_cache_reserve_gb: float = 2.0,
        min_scale_factor: float = 0.5,
    ):
        self.device = device
        self.dataset_path = dataset_path
        self.min_scale_factor = min_scale_factor
        self.gpu_cache_reserve_gb = gpu_cache_reserve_gb
        
        # Load metadata and setup cache
        self._load_metadata()
        self._setup_gpu_cache()
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_metadata(self):
        """Load dataset metadata from HDF5 file."""
        with h5py.File(self.dataset_path, 'r') as f:
            meta = f['metadata']
            self.metadata = TrajectoryMetadata(
                num_trajectories=int(meta.attrs['num_trajectories']),
                trajectory_length=int(meta.attrs['trajectory_length']),
                action_dim=int(meta.attrs['action_dim']),
                dt=float(meta.attrs['dt']),
                max_speed=float(meta.attrs['max_speed']),
                max_speed_z=float(meta.attrs['max_speed_z']),
                max_speed_yaw=float(meta.attrs['max_speed_yaw']),
                reference_map_bounds=tuple(meta.attrs['reference_map_bounds']),
            )
            
            # Load bounding boxes for all trajectories (needed for scaled sampling)
            self.bboxes = torch.from_numpy(f['bboxes'][:]).to(self.device)  # (N, 6)
            
            # Load style parameters
            self.styles = {
                'noise_freq': torch.from_numpy(f['styles/noise_freq'][:]).to(self.device),
                'smoothness': torch.from_numpy(f['styles/smoothness'][:]).to(self.device),
                'laziness': torch.from_numpy(f['styles/laziness'][:]).to(self.device),
            }
        
        print(f"[TrajectoryDataset] Loaded metadata from {self.dataset_path}")
        print(f"  - Trajectories: {self.metadata.num_trajectories}")
        print(f"  - Length: {self.metadata.trajectory_length}")
        print(f"  - Action dim: {self.metadata.action_dim}")
    
    def _setup_gpu_cache(self):
        """Setup GPU cache with auto-sized capacity based on available memory."""
        # Query available GPU memory
        if self.device.type == 'cuda':
            free_memory, total_memory = torch.cuda.mem_get_info(self.device)
            free_memory_gb = free_memory / (1024 ** 3)
            
            # Reserve memory for simulator and training
            available_for_cache_gb = max(0.5, free_memory_gb - self.gpu_cache_reserve_gb)
            
            # Calculate bytes per trajectory
            # velocities: (T, D) float32 + positions: (T, 3) float32
            T = self.metadata.trajectory_length
            D = self.metadata.action_dim
            bytes_per_traj = (T * D + T * 3) * 4  # float32 = 4 bytes
            
            # Calculate cache capacity
            cache_capacity = int(available_for_cache_gb * (1024 ** 3) / bytes_per_traj)
            cache_capacity = max(100, min(cache_capacity, self.metadata.num_trajectories))
        else:
            # CPU mode: use a reasonable default
            cache_capacity = min(1000, self.metadata.num_trajectories)
        
        self.cache_capacity = cache_capacity
        
        # Pre-allocate cache tensors
        T = self.metadata.trajectory_length
        D = self.metadata.action_dim
        
        self.cache_velocities = torch.zeros(
            cache_capacity, T, D, device=self.device, dtype=torch.float32
        )
        self.cache_positions = torch.zeros(
            cache_capacity, T, 3, device=self.device, dtype=torch.float32
        )
        self.cache_indices = torch.full(
            (cache_capacity,), -1, device=self.device, dtype=torch.long
        )
        self.cache_lru = torch.zeros(cache_capacity, device=self.device, dtype=torch.long)
        self.cache_tick = 0
        
        print(f"[TrajectoryDataset] GPU cache initialized:")
        print(f"  - Capacity: {cache_capacity} trajectories")
        print(f"  - Memory: {cache_capacity * bytes_per_traj / (1024**2):.1f} MB")
    
    def _get_from_cache_or_load(self, traj_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get trajectories from cache or load from disk.
        
        Args:
            traj_indices: (B,) trajectory indices to retrieve
            
        Returns:
            velocities: (B, T, D)
            positions: (B, T, 3)
        """
        B = len(traj_indices)
        T = self.metadata.trajectory_length
        D = self.metadata.action_dim
        
        # Output tensors
        velocities = torch.zeros(B, T, D, device=self.device)
        positions = torch.zeros(B, T, 3, device=self.device)
        
        # Check which indices are in cache
        # Expand for comparison: cache_indices (C,) vs traj_indices (B,)
        cache_matches = (self.cache_indices.unsqueeze(0) == traj_indices.unsqueeze(1))  # (B, C)
        in_cache = cache_matches.any(dim=1)  # (B,)
        cache_slots = cache_matches.float().argmax(dim=1)  # (B,) - slot indices for cached items
        
        # Update LRU for cache hits
        self.cache_tick += 1
        hit_slots = cache_slots[in_cache]
        if len(hit_slots) > 0:
            self.cache_lru[hit_slots] = self.cache_tick
            velocities[in_cache] = self.cache_velocities[hit_slots]
            positions[in_cache] = self.cache_positions[hit_slots]
            self.cache_hits += len(hit_slots)
        
        # Load missing trajectories from disk
        missing_mask = ~in_cache
        if missing_mask.any():
            missing_indices = traj_indices[missing_mask].cpu().numpy()
            self.cache_misses += len(missing_indices)
            
            with h5py.File(self.dataset_path, 'r') as f:
                # Load trajectories one by one (HDF5 doesn't support fancy indexing well)
                for i, (batch_idx, traj_idx) in enumerate(zip(
                    missing_mask.nonzero(as_tuple=False).squeeze(-1).tolist(),
                    missing_indices.tolist()
                )):
                    vel = torch.from_numpy(f['velocities'][traj_idx]).to(self.device)
                    pos = torch.from_numpy(f['positions'][traj_idx]).to(self.device)
                    velocities[batch_idx] = vel
                    positions[batch_idx] = pos
                    
                    # Add to cache (LRU eviction)
                    lru_slot = self.cache_lru.argmin().item()
                    self.cache_velocities[lru_slot] = vel
                    self.cache_positions[lru_slot] = pos
                    self.cache_indices[lru_slot] = traj_idx
                    self.cache_lru[lru_slot] = self.cache_tick
        
        return velocities, positions
    
    def sample_raw(
        self,
        batch_size: int,
        window_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample random trajectory windows without any transformations.
        
        This mode ignores boundary constraints and is intended for training
        with extreme/edge cases where trajectories may go out of bounds.
        
        Args:
            batch_size: Number of trajectory windows to sample
            window_size: Length of each window (must be <= trajectory_length)
            
        Returns:
            velocities: (B, window_size, D) velocity trajectories
            styles: dict of style parameters for sampled trajectories
        """
        assert window_size <= self.metadata.trajectory_length, \
            f"window_size ({window_size}) must be <= trajectory_length ({self.metadata.trajectory_length})"
        
        # Random trajectory indices
        traj_indices = torch.randint(
            0, self.metadata.num_trajectories, (batch_size,), device=self.device
        )
        
        # Random start offsets within each trajectory
        max_offset = self.metadata.trajectory_length - window_size
        start_offsets = torch.randint(0, max_offset + 1, (batch_size,), device=self.device)
        
        # Load full trajectories from cache/disk
        full_velocities, _ = self._get_from_cache_or_load(traj_indices)
        
        # Extract windows using advanced indexing
        # Create index tensor for gathering
        window_indices = start_offsets.unsqueeze(1) + torch.arange(
            window_size, device=self.device
        ).unsqueeze(0)  # (B, window_size)
        
        # Gather velocities
        velocities = torch.gather(
            full_velocities,
            dim=1,
            index=window_indices.unsqueeze(-1).expand(-1, -1, self.metadata.action_dim)
        )
        
        # Get styles for sampled trajectories
        styles = {
            'noise_freq': self.styles['noise_freq'][traj_indices],
            'smoothness': self.styles['smoothness'][traj_indices],
            'laziness': self.styles['laziness'][traj_indices],
        }
        
        return velocities, styles
    
    def sample_scaled(
        self,
        batch_size: int,
        window_size: int,
        start_pos: torch.Tensor,
        map_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample random trajectory windows with scale-to-fit transformation.
        
        Scales trajectories to fit within available space from start position
        to map boundaries, maintaining smoothness while ensuring no boundary
        violations.
        
        Args:
            batch_size: Number of trajectory windows to sample
            window_size: Length of each window
            start_pos: (B, 3) starting positions in world frame
            map_bounds: (3,) half-extents of the map [x, y, z]
            
        Returns:
            velocities: (B, window_size, D) scaled velocity trajectories
            scale_factors: (B,) applied scale factors
            styles: dict of style parameters
        """
        assert window_size <= self.metadata.trajectory_length
        
        # Random trajectory indices
        traj_indices = torch.randint(
            0, self.metadata.num_trajectories, (batch_size,), device=self.device
        )
        
        # Random start offsets
        max_offset = self.metadata.trajectory_length - window_size
        start_offsets = torch.randint(0, max_offset + 1, (batch_size,), device=self.device)
        
        # Load full trajectories
        full_velocities, full_positions = self._get_from_cache_or_load(traj_indices)
        
        # Extract position windows to compute window-specific bboxes
        window_indices = start_offsets.unsqueeze(1) + torch.arange(
            window_size, device=self.device
        ).unsqueeze(0)  # (B, window_size)
        
        positions_window = torch.gather(
            full_positions,
            dim=1,
            index=window_indices.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, window_size, 3)
        
        # Compute window bboxes (relative to window start, not trajectory start)
        # Shift positions so window starts at origin
        positions_relative = positions_window - positions_window[:, 0:1, :]  # (B, window_size, 3)
        
        bbox_min = positions_relative.min(dim=1).values  # (B, 3)
        bbox_max = positions_relative.max(dim=1).values  # (B, 3)
        bbox_extent = (bbox_max - bbox_min).abs()  # (B, 3)
        
        # Compute available space from start_pos to boundaries
        # Map bounds: [-bounds, bounds] for x,y; [1.0, 2*bounds_z] for z
        available_pos = map_bounds - start_pos.abs()  # Distance to positive boundary
        available_neg = map_bounds + start_pos  # Distance to negative boundary (x,y)
        
        # For z-axis, handle floor (1.0) and ceiling (2 * bounds_z)
        available_pos[:, 2] = 2 * map_bounds[2] - start_pos[:, 2]  # To ceiling
        available_neg[:, 2] = start_pos[:, 2] - 1.0  # To floor
        
        # Minimum available space in each direction
        available = torch.min(available_pos, available_neg)  # (B, 3)
        
        # Compute scale factor: min(available / bbox_extent) per trajectory
        # Avoid division by zero for trajectories with zero extent in some axis
        bbox_extent_safe = bbox_extent.clamp(min=0.01)
        scale_per_axis = available / bbox_extent_safe  # (B, 3)
        scale_factors = scale_per_axis.min(dim=1).values  # (B,)
        
        # Clamp scale factors to reasonable range
        scale_factors = scale_factors.clamp(min=self.min_scale_factor, max=2.0)
        
        # Extract velocity windows
        velocities = torch.gather(
            full_velocities,
            dim=1,
            index=window_indices.unsqueeze(-1).expand(-1, -1, self.metadata.action_dim)
        )
        
        # Scale velocities (linear velocity channels only, not yaw rate)
        velocities_scaled = velocities.clone()
        velocities_scaled[..., :3] = velocities[..., :3] * scale_factors.view(-1, 1, 1)
        
        # Get styles
        styles = {
            'noise_freq': self.styles['noise_freq'][traj_indices],
            'smoothness': self.styles['smoothness'][traj_indices],
            'laziness': self.styles['laziness'][traj_indices],
        }
        
        return velocities_scaled, scale_factors, styles
    
    def sample(
        self,
        batch_size: int,
        window_size: int,
        mode: Literal["raw", "scaled"] = "scaled",
        start_pos: Optional[torch.Tensor] = None,
        map_bounds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Unified sampling interface.
        
        Args:
            batch_size: Number of samples
            window_size: Window length
            mode: "raw" for no transforms, "scaled" for boundary-aware scaling
            start_pos: (B, 3) starting positions (required for "scaled" mode)
            map_bounds: (3,) map half-extents (required for "scaled" mode)
            
        Returns:
            For "raw" mode: (velocities, styles)
            For "scaled" mode: (velocities, scale_factors, styles)
        """
        if mode == "raw":
            return self.sample_raw(batch_size, window_size)
        elif mode == "scaled":
            assert start_pos is not None, "start_pos required for scaled mode"
            assert map_bounds is not None, "map_bounds required for scaled mode"
            return self.sample_scaled(batch_size, window_size, start_pos, map_bounds)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
        }
    
    def reset_cache_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0


def create_trajectory_dataset(
    output_path: str,
    velocities: np.ndarray,
    positions: np.ndarray,
    bboxes: np.ndarray,
    styles: Dict[str, np.ndarray],
    metadata: TrajectoryMetadata,
    compression: str = "gzip",
    compression_opts: int = 4,
):
    """
    Create an HDF5 trajectory dataset file.
    
    Args:
        output_path: Path to output HDF5 file
        velocities: (N, T, D) velocity trajectories
        positions: (N, T, 3) position trajectories
        bboxes: (N, 6) bounding boxes [min_x, min_y, min_z, max_x, max_y, max_z]
        styles: dict with 'noise_freq', 'smoothness', 'laziness' arrays
        metadata: TrajectoryMetadata object
        compression: HDF5 compression type
        compression_opts: Compression level
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Store trajectories with chunking for efficient partial reads
        chunk_size = min(100, velocities.shape[0])
        
        f.create_dataset(
            'velocities',
            data=velocities,
            dtype='float32',
            compression=compression,
            compression_opts=compression_opts,
            chunks=(chunk_size, velocities.shape[1], velocities.shape[2]),
        )
        
        f.create_dataset(
            'positions',
            data=positions,
            dtype='float32',
            compression=compression,
            compression_opts=compression_opts,
            chunks=(chunk_size, positions.shape[1], positions.shape[2]),
        )
        
        f.create_dataset(
            'bboxes',
            data=bboxes,
            dtype='float32',
            compression=compression,
            compression_opts=compression_opts,
        )
        
        # Store styles
        styles_grp = f.create_group('styles')
        for key, value in styles.items():
            styles_grp.create_dataset(key, data=value, dtype='float32')
        
        # Store metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['num_trajectories'] = metadata.num_trajectories
        meta_grp.attrs['trajectory_length'] = metadata.trajectory_length
        meta_grp.attrs['action_dim'] = metadata.action_dim
        meta_grp.attrs['dt'] = metadata.dt
        meta_grp.attrs['max_speed'] = metadata.max_speed
        meta_grp.attrs['max_speed_z'] = metadata.max_speed_z
        meta_grp.attrs['max_speed_yaw'] = metadata.max_speed_yaw
        meta_grp.attrs['reference_map_bounds'] = list(metadata.reference_map_bounds)
    
    print(f"[TrajectoryDataset] Created dataset at {output_path}")
    print(f"  - Trajectories: {metadata.num_trajectories}")
    print(f"  - Length: {metadata.trajectory_length}")
    print(f"  - File size: {os.path.getsize(output_path) / (1024**2):.1f} MB")
