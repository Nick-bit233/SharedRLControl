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
from typing import Any, Optional, Tuple, Dict, Union, Literal
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
    
    Stores fully processed trajectories (velocities + positions) in memory (VRAM/RAM),
    then samples random windows at runtime.
    
    Args:
        dataset_path: Path to HDF5 file containing trajectories
        device: PyTorch device for GPU operations
        gpu_cache_reserve_gb: (Deprecated) Kept for API compatibility
        min_scale_factor: Minimum allowed scale factor for scaled sampling mode
        preload_data: (Deprecated) Always True now.
    """
    
    def __init__(
        self,
        dataset_path: str,
        device: torch.device,
        gpu_cache_reserve_gb: float = 2.0,
        min_scale_factor: float = 0.5,
        preload_data: bool = True,
    ):
        self.device = device
        self.dataset_path = dataset_path
        self.min_scale_factor = min_scale_factor
        self.gpu_cache_reserve_gb = gpu_cache_reserve_gb
        
        # Load metadata and setup cache
        self._load_metadata()
        
        # Always preload entire dataset
        self.is_preloaded = False
        self._preload_all_data()

    def _preload_all_data(self):
        """Load entire dataset into memory (VRAM if possible, else CPU pinned memory)."""
        print(f"[TrajectoryDataset] Preloading entire dataset into memory...")
        try:
            # Check memory availability
            target_device = "cpu"
            if self.device.type == 'cuda':
                free_memory, total_memory = torch.cuda.mem_get_info(self.device)
                free_gb = free_memory / (1024**3)
                
                # Calculate required size
                T = self.metadata.trajectory_length
                D = self.metadata.action_dim
                N = self.metadata.num_trajectories
                size_gb = (N * T * (D + 3) * 4) / (1024**3)
                
                print(f"  - Dataset Size: {size_gb:.2f} GB")
                print(f"  - Free VRAM: {free_gb:.2f} GB")
                
                if free_gb > size_gb + self.gpu_cache_reserve_gb:
                    target_device = self.device
                    print("  - Loading to: GPU VRAM (Fastest)")
                else:
                    print("  - Loading to: CPU RAM (System memory)")
            else:
                print("  - Loading to: CPU RAM")

            with h5py.File(self.dataset_path, 'r') as f:
                # Load everything.
                print("  - Reading from disk...")
                # Read into numpy first
                vel_np = f['velocities'][:] 
                pos_np = f['positions'][:]
                
                print("  - Moving to device...")
                self.all_velocities = torch.from_numpy(vel_np).to(dtype=torch.float32, device=target_device)
                self.all_positions = torch.from_numpy(pos_np).to(dtype=torch.float32, device=target_device)
                
            self.is_preloaded = True
            print("[TrajectoryDataset] Preload complete.")
            
        except Exception as e:
            print(f"[TrajectoryDataset] Preload failed: {e}")
            print("[TrajectoryDataset] WARNING: Falling back to slow direct disk read!")
            self.is_preloaded = False
    
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
            
            # Load bounding boxes for all trajectories
            self.bboxes = torch.from_numpy(f['bboxes'][:]).to(self.device)
            
            # Load style parameters
            self.styles = {}
            for name in ['noise_freq', 'smoothness', 'laziness']:
                key = f'styles/{name}'
                data = torch.from_numpy(f[key][:]).to(self.device)
                self.styles[name] = data
                
            self.style_len = self.styles['noise_freq'].shape[0]
        
        print(f"[TrajectoryDataset] Loaded metadata from {self.dataset_path}")
        print(f"  - Trajectories: {self.metadata.num_trajectories}")
        print(f"  - Length: {self.metadata.trajectory_length}")
    
    def _get_from_cache_or_load(self, traj_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get trajectories from memory or load from disk (slow fallback).
        """
        if self.is_preloaded:
            # Fast path: Indexing from preloaded memory
            storage_device = self.all_velocities.device
            if traj_indices.device != storage_device:
                indices = traj_indices.to(storage_device)
            else:
                indices = traj_indices
                
            velocities = self.all_velocities[indices]
            positions = self.all_positions[indices]
            
            # Ensure output is on the requested computation device
            if velocities.device != self.device:
                velocities = velocities.to(self.device, non_blocking=True)
                positions = positions.to(self.device, non_blocking=True)
                
            return velocities, positions
        else:
            # Slow fallback: Direct disk read
            # This is extremely inefficient for training but prevents crash if preload fails
            # Note: numpy/h5py processing is done on CPU
            indices_np = traj_indices.cpu().numpy()
            B = len(indices_np)
            T = self.metadata.trajectory_length
            D = self.metadata.action_dim
            
            vel_np = np.zeros((B, T, D), dtype=np.float32)
            pos_np = np.zeros((B, T, 3), dtype=np.float32)
            
            with h5py.File(self.dataset_path, 'r') as f:
                d_vel = f['velocities']
                d_pos = f['positions']
                for i, idx in enumerate(indices_np):
                    vel_np[i] = d_vel[idx]
                    pos_np[i] = d_pos[idx]
            
            return (
                torch.from_numpy(vel_np).to(self.device),
                torch.from_numpy(pos_np).to(self.device)
            )
    
    def sample_raw(
        self,
        batch_size: int,
        window_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample random trajectory windows without any transformations.
        """
        assert window_size <= self.metadata.trajectory_length, \
            f"window_size ({window_size}) must be <= trajectory_length ({self.metadata.trajectory_length})"
        
        # Random trajectory indices
        traj_indices = torch.randint(
            0,
            self.metadata.num_trajectories,
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        
        # Random start offsets within each trajectory
        max_offset = self.metadata.trajectory_length - window_size
        start_offsets = torch.randint(
            0,
            max_offset + 1,
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        
        velocities, _ = self.get_windows(traj_indices, start_offsets, window_size)
        styles = self.get_styles(traj_indices)
        
        return velocities, styles

    def get_styles(self, traj_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return style metadata for the given trajectory indices."""
        traj_indices = traj_indices.to(device=self.device, dtype=torch.long)
        style_indices = traj_indices % self.style_len
        return {
            'noise_freq': self.styles['noise_freq'][style_indices],
            'smoothness': self.styles['smoothness'][style_indices],
            'laziness': self.styles['laziness'][style_indices],
        }

    def get_windows(
        self,
        traj_indices: torch.Tensor,
        start_offsets: torch.Tensor,
        window_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather fixed trajectory windows by index and offset.

        This is the deterministic counterpart to random sampling. It is used by
        episode-contiguous user playback while keeping the old random sampling
        methods available for existing callers.
        """
        assert window_size <= self.metadata.trajectory_length, \
            f"window_size ({window_size}) must be <= trajectory_length ({self.metadata.trajectory_length})"
        if traj_indices.ndim != 1 or start_offsets.ndim != 1:
            raise ValueError("traj_indices and start_offsets must be 1D tensors")
        if len(traj_indices) != len(start_offsets):
            raise ValueError("traj_indices and start_offsets must have the same length")
        if len(traj_indices) == 0:
            return (
                torch.zeros((0, window_size, self.metadata.action_dim), device=self.device),
                torch.zeros((0, window_size, 3), device=self.device),
            )

        traj_indices = traj_indices.to(device=self.device, dtype=torch.long)
        start_offsets = start_offsets.to(device=self.device, dtype=torch.long)
        max_offset = self.metadata.trajectory_length - window_size
        if ((start_offsets < 0) | (start_offsets > max_offset)).any():
            raise ValueError(
                f"start_offsets must be in [0, {max_offset}] for window_size={window_size}"
            )

        full_velocities, full_positions = self._get_from_cache_or_load(traj_indices)
        window_indices = start_offsets.unsqueeze(1) + torch.arange(
            window_size, device=self.device
        ).unsqueeze(0)

        velocities = torch.gather(
            full_velocities,
            dim=1,
            index=window_indices.unsqueeze(-1).expand(-1, -1, self.metadata.action_dim),
        )
        positions = torch.gather(
            full_positions,
            dim=1,
            index=window_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        return velocities, positions

    def compute_scale_factors(
        self,
        positions_window: torch.Tensor,
        start_pos: torch.Tensor,
        map_bounds: Optional[torch.Tensor] = None,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute boundary-aware velocity scale factors for position windows."""
        if lower_bounds is None or upper_bounds is None:
            assert map_bounds is not None, (
                "map_bounds is required unless explicit lower_bounds and "
                "upper_bounds are provided"
            )

        positions_relative = positions_window - positions_window[:, 0:1, :]
        bbox_min = positions_relative.min(dim=1).values
        bbox_max = positions_relative.max(dim=1).values

        if lower_bounds is None or upper_bounds is None:
            map_bounds = map_bounds.to(device=start_pos.device, dtype=start_pos.dtype)
            lower_vec = torch.stack(
                (
                    -map_bounds[0],
                    -map_bounds[1],
                    torch.tensor(1.0, device=start_pos.device, dtype=start_pos.dtype),
                )
            )
            upper_vec = torch.stack(
                (
                    map_bounds[0],
                    map_bounds[1],
                    2 * map_bounds[2],
                )
            )
            bounds_source = "legacy map_bounds"
        else:
            lower_vec = lower_bounds.to(device=start_pos.device, dtype=start_pos.dtype)
            upper_vec = upper_bounds.to(device=start_pos.device, dtype=start_pos.dtype)
            bounds_source = "explicit sampling_bounds"

        if lower_vec.ndim == 1:
            lower_vec = lower_vec.view(1, 3)
        if upper_vec.ndim == 1:
            upper_vec = upper_vec.view(1, 3)
        lower_vec = lower_vec.expand_as(start_pos)
        upper_vec = upper_vec.expand_as(start_pos)

        room_pos = upper_vec - start_pos
        room_neg = start_pos - lower_vec

        if ((room_pos < 0) | (room_neg < 0)).any():
            if not getattr(self, "_warned_negative_available", False):
                import logging
                logging.getLogger(__name__).warning(
                    "[trajectory_dataset.compute_scale_factors] start_pos outside "
                    "sampling bounds along some axis (room_pos or room_neg < 0). "
                    "bounds_source=%s start_pos[0]=%s lower_bounds[0]=%s "
                    "upper_bounds[0]=%s",
                    bounds_source,
                    start_pos[0].tolist(),
                    lower_vec[0].tolist(),
                    upper_vec[0].tolist(),
                )
                self._warned_negative_available = True
            inf = torch.full_like(room_pos, float("inf"))
            room_pos = torch.where(room_pos < 0, inf, room_pos)
            room_neg = torch.where(room_neg < 0, inf, room_neg)

        eps = 0.01
        scale_fwd = room_pos / bbox_max.clamp(min=eps)
        scale_bwd = room_neg / (-bbox_min).clamp(min=eps)
        scale_fwd = torch.where(bbox_max <= eps, torch.full_like(scale_fwd, float("inf")), scale_fwd)
        scale_bwd = torch.where((-bbox_min) <= eps, torch.full_like(scale_bwd, float("inf")), scale_bwd)
        scale_per_axis = torch.min(scale_fwd, scale_bwd)
        scale_factors = scale_per_axis.min(dim=1).values
        return scale_factors.clamp(min=self.min_scale_factor, max=2.0)

    def sample_scaled(
        self,
        batch_size: int,
        window_size: int,
        start_pos: torch.Tensor,
        map_bounds: Optional[torch.Tensor] = None,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample random trajectory windows with scale-to-fit transformation.
        """
        assert window_size <= self.metadata.trajectory_length
        if lower_bounds is None or upper_bounds is None:
            assert map_bounds is not None, (
                "map_bounds is required unless explicit lower_bounds and "
                "upper_bounds are provided"
            )

        # Random trajectory indices
        traj_indices = torch.randint(
            0,
            self.metadata.num_trajectories,
            (batch_size,),
            device=self.device,
            generator=generator,
        )

        # Random start offsets
        max_offset = self.metadata.trajectory_length - window_size
        start_offsets = torch.randint(
            0,
            max_offset + 1,
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        
        velocities, positions_window = self.get_windows(
            traj_indices, start_offsets, window_size
        )
        scale_factors = self.compute_scale_factors(
            positions_window,
            start_pos,
            map_bounds=map_bounds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        
        # Scale velocities
        velocities_scaled = velocities.clone()
        velocities_scaled[..., :3] = velocities[..., :3] * scale_factors.view(-1, 1, 1)
        
        styles = self.get_styles(traj_indices)
        
        return velocities_scaled, scale_factors, styles
    
    def sample(
        self,
        batch_size: int,
        window_size: int,
        mode: Literal["raw", "scaled"] = "scaled",
        start_pos: Optional[torch.Tensor] = None,
        map_bounds: Optional[torch.Tensor] = None,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Unified sampling interface."""
        if mode == "raw":
            return self.sample_raw(batch_size, window_size, generator=generator)
        elif mode == "scaled":
            assert start_pos is not None, "start_pos required for scaled mode"
            assert map_bounds is not None or (
                lower_bounds is not None and upper_bounds is not None
            ), "map_bounds or explicit lower_bounds/upper_bounds required for scaled mode"
            return self.sample_scaled(
                batch_size,
                window_size,
                start_pos,
                map_bounds=map_bounds,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                generator=generator,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

def create_trajectory_dataset(
    output_path: str,
    velocities: np.ndarray,
    positions: np.ndarray,
    bboxes: np.ndarray,
    styles: Dict[str, np.ndarray],
    metadata: TrajectoryMetadata,
    compression: str = "gzip",
    compression_opts: int = 4,
    metadata_attrs: Optional[Dict[str, Any]] = None,
    extra_groups: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
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
        metadata_attrs: Additional HDF5 metadata attributes.
        extra_groups: Optional nested groups such as intent diagnostics.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    def _write_attr(group, key: str, value: Any):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        group.attrs[key] = value

    def _write_group(parent, group_name: str, values: Dict[str, Any]):
        grp = parent.create_group(group_name)
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, dict):
                _write_group(grp, key, value)
                continue
            arr = np.asarray(value)
            kwargs = {}
            if arr.ndim > 0:
                kwargs = {
                    "compression": compression,
                    "compression_opts": compression_opts,
                }
            grp.create_dataset(key, data=arr, **kwargs)
    
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
        if metadata_attrs:
            for key, value in metadata_attrs.items():
                _write_attr(meta_grp, key, value)

        if extra_groups:
            for group_name, values in extra_groups.items():
                _write_group(f, group_name, values)
    
    print(f"[TrajectoryDataset] Created dataset at {output_path}")
    print(f"  - Trajectories: {metadata.num_trajectories}")
    print(f"  - Length: {metadata.trajectory_length}")
    print(f"  - File size: {os.path.getsize(output_path) / (1024**2):.1f} MB")
