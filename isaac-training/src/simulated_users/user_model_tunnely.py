import torch
import numpy as np
import math
import logging
from enum import Enum
from typing import Any, Optional, Literal, TYPE_CHECKING
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

from src.core.profiler import get_profiler
from src.simulated_users.user_model import BatchedPerlinNoise, InterpType
if TYPE_CHECKING:
    from src.datasets.trajectory_dataset import TrajectoryDataset

def batched_perlin_noise(
    channels: int,
    time: torch.Tensor, 
    seeds: torch.Tensor, 
    freq: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Generate batched 1D Perlin-like Gradient Noise.
    
    Args:
        time: (N, T) time tensor
        seeds: (N, ch) seeds for channels (like vx, vy, vz), each is independent 1D noise
        freq: (N, 1) frequency scaling
        device: torch device
        
    Returns:
        noise: (N, T, ch) values in approx [-1, 1]
    """
    N, T = time.shape
    num_channels = channels  # Number of channels for noise generation
    
    # Output tensor
    noise = torch.zeros(N, T, num_channels, device=device)
    
    # Sample each channel independently
    for ch in range(num_channels):
        # Create noise generator for this channel with per-env seeds
        channel_seeds = seeds[:, ch]  # (N,)
        
        # Create batched perlin noise generator
        perlin = BatchedPerlinNoise(
            seeds=channel_seeds,
            amplitude=1.0,
            frequency=1.0,  # Base frequency, actual freq applied via time scaling
            octaves=1,
            interp=InterpType.COSINE,
            use_fade=False,
            device=device
        )
        
        # Scale time by frequency (freq is per-env)
        # time: (N, T), freq: (N, 1)
        scaled_time = time * freq  # (N, T)
        
        # Get noise values
        noise[:, :, ch] = perlin.get(scaled_time)
    
    return noise

class UserModelTunnel:
    """
    User Model for tunnel task simulating.
    """
    
    def __init__(
        self, 
        num_envs, 
        cfg, 
        use_lib_noise=False, 
        logger=None,
        offline_mode: bool = False,
        dataset: Optional["TrajectoryDataset"] = None,
        sampling_mode: Literal["scaled", "raw"] = "scaled",
    ):
        self.num_envs = num_envs
        self.num_channels = 1  # only generate noise for linear velocity (vx or vy)
        self.cfg = cfg
        
        # Offline mode settings
        self.offline_mode = offline_mode
        self.dataset = dataset
        self.sampling_mode = sampling_mode
        
        if self.offline_mode:
            if self.dataset is None:
                raise ValueError("dataset must be provided when offline_mode=True")
            print(f"[UserModel] Offline mode enabled with sampling_mode='{sampling_mode}'")
        
        # Setup logger (use provided logger or create a null logger)
        if logger is not None:
            self.logger = logger
        else:
            # Create a null logger that discards all messages
            self.logger = logging.getLogger("user_model_null")
            self.logger.addHandler(logging.NullHandler())
        # self.use_lib_noise = use_lib_noise
        # if self.use_lib_noise:
        #     print("[Warning]: Using 'noise' library for Perlin noise generation. This runs on cpu and may be slower than the batched implementation.")
        self.device = cfg.device
        self.dt = cfg.sim.dt
        self._dataset_eval_generator: Optional[torch.Generator] = None
        self._dataset_eval_seed: Optional[int] = None
        self._use_eval_dataset_generator = False
        
        # Map boundaries for APF [x, y, z] in IsaacSim world frame.
        # NOTE: cfg.env.map_range is stored as [Isaac-y, Isaac-x, Isaac-z] half-extents
        # (see comment in configs/experiment/tunnel.yaml). The downstream consumers
        # (e.g. trajectory_dataset.sample_scaled) expect map_bounds aligned with
        # start_pos which is in IsaacSim (x, y, z) order. Swap the first two
        # elements so map_bounds[i] correctly pairs with pos[i].
        _raw_map_range = list(cfg.env.map_range)
        if len(_raw_map_range) >= 2:
            _raw_map_range[0], _raw_map_range[1] = _raw_map_range[1], _raw_map_range[0]
        self.env_map_range = torch.tensor(
            _raw_map_range, dtype=torch.float32, device=self.device
        )
        self.sampling_lower_bounds = None
        self.sampling_upper_bounds = None
        sampling_bounds_cfg = cfg.user_model.get("sampling_bounds", None)
        if sampling_bounds_cfg is not None:
            lower, upper = self._build_sampling_bounds(
                sampling_bounds_cfg,
                cfg.user_model.get("sampling_bounds_expansion", 0.0),
            )
            self.sampling_lower_bounds = torch.tensor(
                lower, dtype=torch.float32, device=self.device
            )
            self.sampling_upper_bounds = torch.tensor(
                upper, dtype=torch.float32, device=self.device
            )
            print(
                "[UserModel] Using explicit sampling_bounds "
                f"lower={lower} upper={upper}"
            )

        # Parameters
        self.buffer_size = cfg.algo.training_frame_num  # training frame num steps (e.g. 128 frames is about 2 seconds)
        self.repulsive_gain = 1.0  # Maxium repulsive force gain for APF
        self.max_speed = cfg.algo.actor.action_limit
        self.max_speed_z = 0
        self.forward_speed = cfg.user_model.get("forward_speed", self.max_speed)
        self.lateral_speed_limit = cfg.user_model.get("lateral_speed_limit", self.max_speed)
        self.boundary_aware_y = cfg.user_model.get("boundary_aware_y", False)
        self.y_boundary_margin = cfg.user_model.get("y_boundary_margin", 0.5)
        
        # 增加Z轴速度偏置 (Z-axis compensation for tilt-induced lift loss)
        # 【注意】在LeePositionController和模拟环境计算速度时均已考虑了对z轴速度的补偿，在多数情况下，此值应设为0
        self.z_tilt_compensation = cfg.user_model.get("z_tilt_compensation", 0.0)
        
        self.max_speed_yaw = 0.5  # Max yaw rate (rad/s)

        # Simple mode parameters
        self.simple_mode = cfg.user_model.simple_mode
        self.enable_yaw_rate = cfg.user_model.enable_yaw_rate
        if self.simple_mode:
            print("[UserModel] Using simple step function (linear velocity commands).")
            self.xy_speed = torch.rand(num_envs, device=self.device) * self.max_speed
            self.yaw_rate_speed = torch.rand(num_envs, device=self.device) * self.max_speed_yaw
            self.theta = torch.rand(num_envs, device=self.device) * 2.0 * math.pi

        # Online mode parameters
        self.online_sample_filter = cfg.user_model.online_sample_filter
        
        # State
        self.action_buffer = torch.zeros(num_envs, self.buffer_size, 3, device=self.device)  # 3D velocity only
        self.buffer_read_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.noise_time = torch.zeros(num_envs, device=self.device)
        
        # Random seeds for noise (N, ch)
        self.noise_seeds = torch.randint(0, 100000, (num_envs, self.num_channels), device=self.device)

        # Style parameters (randomized per env)
        self.freq_base = cfg.user_model.style.frequency_base
        self.freq_scale = cfg.user_model.style.frequency_scale
        self.smooth_base = cfg.user_model.style.smoothness_base
        self.smooth_scale = cfg.user_model.style.smoothness_scale
        self.laziness = cfg.user_model.style.laziness
        self.styles = {
            'noise_freq': torch.rand(num_envs, 1, device=self.device) * self.freq_scale + self.freq_base,
            'smoothness': torch.rand(num_envs, 1, device=self.device) * self.smooth_scale + self.smooth_base,
            'laziness': torch.rand(num_envs, 1, device=self.device) * self.laziness,
        }
        
        # Previous action for smoothing (Low Pass Filter)
        self.prev_filtered_action = torch.zeros(num_envs, 3, device=self.device)  # 3D velocity only

    def set_eval_seed(self, seed: int) -> None:
        """Reset the offline dataset sampler for deterministic evaluation."""

        self._dataset_eval_seed = int(seed)
        self._dataset_eval_generator = torch.Generator(device=self.device)
        self._dataset_eval_generator.manual_seed(self._dataset_eval_seed + 1_000_003)
        self._use_eval_dataset_generator = True

    def _ensure_eval_dataset_generator(self, seed: int) -> None:
        if self._dataset_eval_generator is None or self._dataset_eval_seed != int(seed):
            self.set_eval_seed(seed)

    @staticmethod
    def _axis_bounds(bounds_cfg: Any, axis: str) -> list[float]:
        values = bounds_cfg.get(axis, None)
        if values is None:
            raise ValueError(f"user_model.sampling_bounds.{axis} must be set")
        values = list(values)
        if len(values) != 2:
            raise ValueError(f"user_model.sampling_bounds.{axis} must have two values")
        lower, upper = float(values[0]), float(values[1])
        if upper <= lower:
            raise ValueError(
                f"user_model.sampling_bounds.{axis} upper must be > lower, got {values}"
            )
        return [lower, upper]

    @staticmethod
    def _parse_bounds_expansion(expansion_cfg: Any) -> list[float]:
        if expansion_cfg is None:
            return [0.0, 0.0, 0.0]
        if isinstance(expansion_cfg, (int, float)):
            value = float(expansion_cfg)
            return [value, value, value]
        if hasattr(expansion_cfg, "get"):
            return [float(expansion_cfg.get(axis, 0.0)) for axis in ("x", "y", "z")]
        values = list(expansion_cfg)
        if len(values) != 3:
            raise ValueError(
                "user_model.sampling_bounds_expansion must be a scalar, "
                "a 3-value list, or a mapping with x/y/z"
            )
        return [float(v) for v in values]

    @classmethod
    def _build_sampling_bounds(
        cls,
        bounds_cfg: Any,
        expansion_cfg: Any,
    ) -> tuple[list[float], list[float]]:
        lower = []
        upper = []
        expansion = cls._parse_bounds_expansion(expansion_cfg)
        for axis, pad in zip(("x", "y", "z"), expansion):
            axis_lower, axis_upper = cls._axis_bounds(bounds_cfg, axis)
            if pad < 0:
                raise ValueError("sampling_bounds_expansion values must be non-negative")
            lower.append(axis_lower - pad)
            upper.append(axis_upper + pad)
        return lower, upper

    def reset(self, pos, quat, env_ids, seed=None):
        """
        Reset state for env_ids
        Args:
            pos: (K, 3) start positions (world frame)
            quat: (K, 4) start orientations (world frame)
            seed: (int, optional) If provided, forces deterministic behavior for evaluation.
        """
        if pos.ndim == 3: pos = pos.squeeze(1)
        if quat.ndim == 3: quat = quat.squeeze(1)
        
        K = len(env_ids)
        
        # Reset time and indices
        self.noise_time[env_ids] = 0
        self.buffer_read_idx[env_ids] = 0
        
        # Handle Seeding
        if seed is not None:
            # Evaluation Mode: Deterministic
            if self.offline_mode:
                self._ensure_eval_dataset_generator(int(seed))

            # 使用 seed + env_id 确保每个环境不同，但每次运行一致
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            
            # 为每个 env 生成一个基于 base gen 的确定性唯一种子
            base_seeds = torch.randint(0, 100000, (K, self.num_channels), generator=gen, device=self.device)
            # 为了保证不同 env_id 即使在不同 batch 也有区别，可以加上 env_ids
            self.noise_seeds[env_ids] = base_seeds + env_ids.unsqueeze(1)
            
            # 确定性的风格参数
            self.styles['noise_freq'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.freq_scale + self.freq_base
            if self.cfg.user_model.get("max_noise_freq", None) is not None:
                self.styles['noise_freq'][env_ids].clamp_(max=float(self.cfg.user_model.max_noise_freq))
            self.styles['smoothness'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.smooth_scale + self.smooth_base
            self.styles['laziness'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.laziness

            if self.simple_mode:
                self.theta[env_ids] = torch.rand(K, device=self.device, generator=gen) * 2.0 * math.pi
                self.xy_speed[env_ids] = torch.rand(K, device=self.device, generator=gen) * self.max_speed
                self.yaw_rate_speed[env_ids] = torch.rand(K, device=self.device, generator=gen) * self.max_speed_yaw
            
        else:
            # Training Mode: Random generate seeds and styles
            self._use_eval_dataset_generator = False
            self.noise_seeds[env_ids] = torch.randint(0, 100000, (K, self.num_channels), device=self.device)  # 3 channels for vx, vy, vz
            self.styles['noise_freq'][env_ids] = torch.rand(K, 1, device=self.device) * self.freq_scale + self.freq_base
            if self.cfg.user_model.get("max_noise_freq", None) is not None:
                self.styles['noise_freq'][env_ids].clamp_(max=float(self.cfg.user_model.max_noise_freq))
            self.styles['smoothness'][env_ids] = torch.rand(K, 1, device=self.device) * self.smooth_scale + self.smooth_base
            self.styles['laziness'][env_ids] = torch.rand(K, 1, device=self.device) * self.laziness

            if self.simple_mode:
                self.theta[env_ids] = torch.rand(K, device=self.device) * 2.0 * math.pi
                self.xy_speed[env_ids] = torch.rand(K, device=self.device) * self.max_speed
                self.yaw_rate_speed[env_ids] = torch.rand(K, device=self.device) * self.max_speed_yaw

        self.prev_filtered_action[env_ids] = 0.0
        
        if not self.simple_mode:
            # Refill buffer using appropriate method
            if self.offline_mode:
                self._refill_from_dataset(env_ids, pos, quat)
            else:
                self._refill_buffer(env_ids, pos, quat, enable_filter=self.online_sample_filter)
        else:
            pass

    def step_simple(self, drone_state, theta):
        """
        Simple step function: outputs random XY velocity with Z=0.
        Velocity direction is fixed by theta.
        
        Returns:
            action: (N, 3) velocity commands [vx, vy, vz] in body frame
            needs_refill: (N,) boolean tensor (always False in simple mode)
        """
        N = drone_state.shape[0]

        # Compute XY velocities
        vx = torch.ones(N, device=self.device) * self.xy_speed
        vy = torch.zeros(N, device=self.device)
        
        # Z is zero (yaw_rate removed from action space)
        vz = torch.zeros(N, device=self.device)
        
        # Stack into action tensor - 3D velocity only
        action = torch.stack([vx, vy, vz], dim=-1)  # (N, 3)
        
        # No refill needed in simple mode
        needs_refill = torch.zeros(N, dtype=torch.bool, device=self.device)

        return action, needs_refill

    def step(self, drone_state, drone_pos_w):
        profiler = get_profiler()
        profiler.start("user_model/step")

        if self.simple_mode:
            action, needs_refill = self.step_simple(drone_state, self.theta)
            profiler.stop("user_model/step")
            return action, needs_refill
        
        # drone_state: (N, 10) -> vel, ang_vel, quat, in body frame
        # drone_pos_w: (N, 3) position in world frame (need to be passed from outside since drone_state is body frame)
        pos = drone_pos_w
        quat = drone_state[..., 6:10]
        
        if pos.ndim == 3: pos = pos.squeeze(1)
        if quat.ndim == 3: quat = quat.squeeze(1)
        
        # Check refill
        needs_refill = self.buffer_read_idx >= self.buffer_size
        if needs_refill.any():
            with profiler.timer("user_model/refill_buffer"):
                idxs = needs_refill.nonzero(as_tuple=False).squeeze(-1)
                # For refill, we need current pos/quat to start integration
                if self.offline_mode:
                    self._refill_from_dataset(idxs, pos[idxs], quat[idxs])
                else:
                    self._refill_buffer(idxs, pos[idxs], quat[idxs])
                self.buffer_read_idx[idxs] = 0
            
        # Read action from buffer
        # action_buffer: (N, T, 3) - 3D velocity only
        # We need to select [i, read_idx[i], :]
        # Use gather
        read_indices = self.buffer_read_idx.view(-1, 1, 1).expand(-1, 1, 3)
        action = torch.gather(self.action_buffer, 1, read_indices).squeeze(1)
        
        self.buffer_read_idx += 1
        
        profiler.stop("user_model/step")
        return action, needs_refill

    def _refill_from_dataset(self, env_ids, start_pos, start_quat):
        """
        Refill action buffer by sampling from pre-generated trajectory dataset.
        
        Args:
            env_ids: (K,) environment indices to refill
            start_pos: (K, 3) starting positions in world frame
            start_quat: (K, 4) starting orientations in world frame
        """
        profiler = get_profiler()
        
        K = len(env_ids)
        T = self.buffer_size
        dataset_generator = (
            self._dataset_eval_generator
            if self._use_eval_dataset_generator
            else None
        )
        
        with profiler.timer("user_model/dataset_sample"):
            if self.sampling_mode == "raw":
                # Raw sampling: no boundary handling
                velocities, styles = self.dataset.sample_raw(
                    K,
                    T,
                    generator=dataset_generator,
                )
                # velocities: (K, T, D)
            else:
                # Scaled sampling: boundary-aware
                velocities, scale_factors, styles = self.dataset.sample_scaled(
                    K,
                    T,
                    start_pos,
                    map_bounds=self.env_map_range,
                    lower_bounds=self.sampling_lower_bounds,
                    upper_bounds=self.sampling_upper_bounds,
                    generator=dataset_generator,
                )
        
        with profiler.timer("user_model/dataset_velocity_postprocess"):
            # Rotate velocities from identity frame to drone's current orientation
            # The dataset stores velocities in a neutral frame (identity quaternion)
            # We need to rotate them to match the current drone orientation
            
            # Extract linear velocity and yaw rate
            lin_vel = velocities[..., :3]  # (K, T, 3)
            
            # Reshape for batch quaternion rotation
            # quat_rotate expects (N, 3) vectors
            K, T_len, _ = lin_vel.shape
            lin_vel_flat = lin_vel.reshape(K * T_len, 3)  # (K*T, 3)
            
            # Expand quaternion to match flattened velocities
            # start_quat: (K, 4) -> repeat for each timestep
            quat_expanded = start_quat.unsqueeze(1).expand(-1, T_len, -1).reshape(K * T_len, 4)
            
            # Rotate: transform from identity frame to current body frame
            # Since we want velocity in body frame relative to current orientation,
            # we actually want to keep the velocity in body frame, just sample diversity
            # The rotation here accounts for trajectory orientation diversity
            # For simplicity, we can skip rotation if trajectories are generated in body frame
            # OR we apply inverse rotation to go from world to body
            
            # Actually, the generated trajectories are in a "neutral" body frame
            # At runtime, the velocity commands are interpreted in the drone's body frame
            # So no rotation is needed - the velocity is already in body frame
            
            # However, for diversity, we can apply a random rotation based on start_quat
            # This makes the same trajectory appear different when drone starts with different yaw
            
            # For now, skip rotation (velocities are already in body frame)
            lin_vel_rotated = lin_vel_flat.reshape(K, T_len, 3)
            
            # Reconstruct full action tensor - 3D velocity only (yaw_rate removed)
            # Only use first 3 dimensions regardless of dataset action_dim
            rotated_velocities = lin_vel_rotated  # (K, T, 3)
        
        # Store in action buffer
        self.action_buffer[env_ids] = rotated_velocities
        
        # Update styles from dataset (optional, for logging/debugging)
        # Note: We don't overwrite self.styles as it may be used elsewhere
        # If needed, uncomment:
        # self.styles['noise_freq'][env_ids] = styles['noise_freq'].unsqueeze(-1)
        # self.styles['smoothness'][env_ids] = styles['smoothness'].unsqueeze(-1)
        # self.styles['laziness'][env_ids] = styles['laziness'].unsqueeze(-1)

    def _refill_buffer(self, env_ids, start_pos, start_quat, enable_filter=False):
        """
        Generate a trajectory of actions using Perlin noise and APF (online mode).
        """
        profiler = get_profiler()
        
        K = len(env_ids)
        T = self.buffer_size
        dt = self.dt
        
        # 1. Generate Perlin Noise for the whole batch (K, T, 3)
        with profiler.timer("user_model/perlin_noise"):
            # Time vector
            t_start = self.noise_time[env_ids].unsqueeze(1) # (K, 1)
            t_steps = torch.arange(T, device=self.device).unsqueeze(0) * dt # (1, T)
            time_grid = t_start + t_steps # (K, T)
            
            seeds = self.noise_seeds[env_ids] # (K, self.num_channels)
            freq = self.styles['noise_freq'][env_ids] # (K, 1)
            
            # Generate raw noise (Target Velocities)
            channel_noise = batched_perlin_noise(self.num_channels, time_grid, seeds, freq, self.device) # (K, T, self.num_channels)

            # Check if output is within [-1, 1]
            if not torch.all((channel_noise >= -1.0) & (channel_noise <= 1.0)):
                self.logger.warning("Perlin noise output out of bounds [-1, 1]")
            
            # Scale noise to physical units
            scale = torch.tensor(
                [self.forward_speed, self.lateral_speed_limit, self.max_speed_z], device=self.device)
            
            noise_expanded = channel_noise.unsqueeze(-1) 
            # 拼接后形状变为 (K, T, C, 3)
            target_vels = torch.cat([
                torch.ones_like(noise_expanded),         # vx = 1x
                noise_expanded,                          # vy = (as noise)
                torch.zeros_like(noise_expanded)          # vz = 0x
            ], dim=-1)

            # (K, T, C, 3) * (3,) 符合广播机制
            target_vels = target_vels * scale  

            # 如果C=1，在最后去掉单维度：
            if self.num_channels == 1:
                target_vels = target_vels.squeeze(2) # 变回 (K, T, 3)

            # Apply Z-axis tilt compensation
            if self.z_tilt_compensation:
                target_vels[:, :, 2] += self.z_tilt_compensation
            # print(f"[UserModel] refill target vels mean z: {target_vels[:, :, 2].mean().item():.4f} m/s")##

        # No filtering, directly store target_vels
        if self.boundary_aware_y:
            room_size = self.cfg.env.get("room_size", None)
            if room_size is not None:
                y_half = float(room_size[1]) / 2.0
                y_limit = max(0.0, y_half - float(self.y_boundary_margin))
                start_y = start_pos[:, 1].reshape(K, 1)
                delta_y = torch.cumsum(target_vels[:, :, 1] * dt, dim=1)
                allowed_delta = (y_limit - start_y.abs()).clamp(min=0.0)
                max_delta = delta_y.abs().amax(dim=1, keepdim=True)
                scale_y = torch.minimum(
                    torch.ones_like(max_delta),
                    allowed_delta / (max_delta + 1e-6),
                )
                target_vels[:, :, 1] *= scale_y

        self.action_buffer[env_ids] = target_vels
        last_val = target_vels[:, -1, :]
        
        # Update state
        self.prev_filtered_action[env_ids] = last_val
        self.noise_time[env_ids] += T * dt
