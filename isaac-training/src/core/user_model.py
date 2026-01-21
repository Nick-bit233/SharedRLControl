import torch
import numpy as np
import math
import logging
from enum import Enum
from typing import Optional, Literal, TYPE_CHECKING
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

from src.core.profiler import get_profiler
if TYPE_CHECKING:
    from src.datasets.trajectory_dataset import TrajectoryDataset


class InterpType(Enum):
    LINEAR = 1
    COSINE = 2
    CUBIC = 3


class BatchedPerlinNoise:
    """
    GPU-batched Perlin Noise generator using PyTorch.
    Algorithmically identical to the PerlinNoise class in perlin_noise.py.
    
    Note: The original PerlinNoise uses `random.Random(self.seed + x).uniform(-1, 1)` 
    which is essentially a hash function mapping (seed, position) -> deterministic value.
    We replicate this behavior using a GPU-friendly deterministic hash.
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
        """
        Args:
            seeds: (N,) tensor of seeds for N independent noise generators
            amplitude: Base amplitude
            frequency: Base frequency
            octaves: Number of octaves for fractal noise
            interp: Interpolation type
            use_fade: Whether to use fade function (useful for linear interp)
            device: Torch device
        """
        self.device = device if device is not None else seeds.device
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.interp = interp
        self.use_fade = use_fade
        
        # Store original seeds for deterministic noise generation
        # Seed need random generated before initialization, or use torch.Generator to make it fixed when evaluating
        self.seeds = seeds.to(self.device).float()  # (N,)
    
    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate deterministic noise value for integer positions.
        Mimics: random.Random(self.seed + x).uniform(-1, 1)
        
        This is a hash function, not sequential RNG. Given the same (seed, position),
        it must always return the same value. torch.rand() cannot do this because
        it depends on generator state that changes with each call.
        
        Args:
            x: (N, T) integer positions
        Returns:
            (N, T) noise values in [-1, 1]
        """
        # Combine seed with position: seeds (N,) -> (N, 1), x (N, T)
        combined = self.seeds.unsqueeze(1) + x.float()  # (N, T)
        
        # Deterministic hash function (GPU-friendly)
        # This mimics random.Random(seed).uniform(-1, 1) behavior
        # Using sine-based hash for good distribution
        h = torch.sin(combined * 12.9898 + 78.233) * 43758.5453
        h = h - h.floor()  # Fractional part, now in [0, 1)
        
        # Convert to [-1, 1]
        return h * 2.0 - 1.0
    
    def _fade(self, x: torch.Tensor) -> torch.Tensor:
        """Fade function: 6x^5 - 15x^4 + 10x^3"""
        return (6 * x**5) - (15 * x**4) + (10 * x**3)
    
    def _linear_interp(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Linear interpolation: a + x * (b - a)"""
        return a + x * (b - a)
    
    def _cosine_interp(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Cosine interpolation"""
        x2 = (1 - torch.cos(x * math.pi)) / 2
        return a * (1 - x2) + b * x2
    
    def _cubic_interp(
        self, v0: torch.Tensor, v1: torch.Tensor, 
        v2: torch.Tensor, v3: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Cubic interpolation"""
        p = (v3 - v2) - (v0 - v1)
        q = (v0 - v1) - p
        r = v2 - v0
        s = v1
        return p * x**3 + q * x**2 + r * x + s
    
    def _interpolated_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get interpolated noise at continuous positions.
        
        Args:
            x: (N, T) continuous positions
        Returns:
            (N, T) interpolated noise values
        """
        prev_x = x.floor().long()  # Previous integer
        next_x = prev_x + 1  # Next integer
        frac_x = x - prev_x.float()  # Fractional part
        
        if self.use_fade:
            frac_x = self._fade(frac_x)
        
        if self.interp == InterpType.LINEAR:
            result = self._linear_interp(
                self._noise(prev_x),
                self._noise(next_x),
                frac_x
            )
        elif self.interp == InterpType.COSINE:
            result = self._cosine_interp(
                self._noise(prev_x),
                self._noise(next_x),
                frac_x
            )
        else:  # CUBIC
            result = self._cubic_interp(
                self._noise(prev_x - 1),
                self._noise(prev_x),
                self._noise(next_x),
                self._noise(next_x + 1),
                frac_x
            )
        
        return result
    
    def get(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Perlin noise value at positions x.
        
        Args:
            x: (N, T) or (N,) positions to sample
        Returns:
            Same shape as x, noise values
        """
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


def batched_perlin_noise(
    time: torch.Tensor, 
    seeds: torch.Tensor, 
    freq: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Generate batched 1D Perlin-like Gradient Noise.
    
    Args:
        time: (N, T) time tensor
        seeds: (N, 3) seeds for 3 channels (vx, vy, vz), each is independent 1D noise
        freq: (N, 1) frequency scaling
        device: torch device
        
    Returns:
        noise: (N, T, 3) values in approx [-1, 1]
    """
    N, T = time.shape
    num_channels = 3  # 3D velocity only (vx, vy, vz)
    
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

class UserModel:
    """
    User Model for simulating human user inputs during drone training.
    
    Supports two modes:
    - Online mode: Generate trajectories on-the-fly using Perlin noise (slower but flexible)
    - Offline mode: Sample from pre-generated trajectory dataset (faster)
    
    Args:
        num_envs: Number of parallel environments
        cfg: Configuration object
        use_lib_noise: Deprecated, kept for compatibility
        logger: Optional logger instance
        offline_mode: If True, use pre-generated trajectories from dataset
        dataset: TrajectoryDataset instance (required if offline_mode=True)
        sampling_mode: "scaled" (boundary-aware) or "raw" (no transforms)
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
        
        # Map boundaries for APF [x, y, z]
        # Assuming map_range is half-extents
        self.env_map_range = torch.tensor(
            cfg.env.map_range, dtype=torch.float32, device=self.device
        )

        # Parameters
        self.buffer_size = cfg.algo.training_frame_num # training frame num steps (e.g. 128 frames is about 2 seconds)
        self.repulsive_gain = 1.0  # Maxium repulsive force gain for APF
        self.max_speed = cfg.algo.actor.action_limit
        self.max_speed_z = self.max_speed
        
        # Z-axis compensation for tilt-induced lift loss
        # 增加Z轴正向偏置 (z_bias) TODO: check LeePositionController
        # When drone tilts to fly horizontally, vertical lift component is reduced
        # Typical compensation: ~0.02-0.05 m/s depending on average tilt angle
        # Formula: compensation ≈ mean_horizontal_speed * sin(mean_tilt_angle)
        # With max_speed=2.0 and ~1.5 deg tilt: 2.0 * 0.5 * sin(0.026) ≈ 0.026 m/s
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
        
        # Random seeds for noise (N, 3) - 3 channels for vx, vy, vz
        self.noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=self.device)

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

            # 使用 seed + env_id 确保每个环境不同，但每次运行一致
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            
            # 为每个 env 生成一个基于 base gen 的确定性唯一种子
            base_seeds = torch.randint(0, 100000, (K, 3), generator=gen, device=self.device)  # 3 channels for vx, vy, vz
            # 为了保证不同 env_id 即使在不同 batch 也有区别，可以加上 env_ids
            self.noise_seeds[env_ids] = base_seeds + env_ids.unsqueeze(1)
            
            # 确定性的风格参数
            self.styles['noise_freq'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.freq_scale + self.freq_base
            self.styles['smoothness'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.smooth_scale + self.smooth_base
            self.styles['laziness'][env_ids] = torch.rand(K, 1, generator=gen, device=self.device) * self.laziness

            if self.simple_mode:
                self.theta[env_ids] = torch.rand(K, device=self.device, generator=gen) * 2.0 * math.pi
                self.xy_speed[env_ids] = torch.rand(K, device=self.device, generator=gen) * self.max_speed
                self.yaw_rate_speed[env_ids] = torch.rand(K, device=self.device, generator=gen) * self.max_speed_yaw
            
        else:
            # Training Mode: Random generate seeds and styles
            self.noise_seeds[env_ids] = torch.randint(0, 100000, (K, 3), device=self.device)  # 3 channels for vx, vy, vz
            self.styles['noise_freq'][env_ids] = torch.rand(K, 1, device=self.device) * self.freq_scale + self.freq_base
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
        vx = self.xy_speed * torch.cos(theta)
        vy = self.xy_speed * torch.sin(theta)
        
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
        
        # Update intent goals (just for visualization/compatibility, set to current pos + action direction)
        # self.intent_goals = pos + quat_rotate(quat, action[..., :3])
        
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
        
        with profiler.timer("user_model/dataset_sample"):
            if self.sampling_mode == "raw":
                # Raw sampling: no boundary handling
                velocities, styles = self.dataset.sample_raw(K, T)
                # velocities: (K, T, D)
            else:
                # Scaled sampling: boundary-aware
                velocities, scale_factors, styles = self.dataset.sample_scaled(
                    K, T, start_pos, self.env_map_range
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
            
            seeds = self.noise_seeds[env_ids] # (K, 3)
            freq = self.styles['noise_freq'][env_ids] # (K, 1)
            
            # Generate raw noise (Target Velocities)
            raw_noise = batched_perlin_noise(time_grid, seeds, freq, self.device)

            # Check if output is within [-1, 1]
            if not torch.all((raw_noise >= -1.0) & (raw_noise <= 1.0)):
                self.logger.warning("Perlin noise output out of bounds [-1, 1]")
            
            # Scale noise to physical units
            # vx, vy, vz (yaw_rate removed from action space)
            scale = torch.tensor(
                [self.max_speed, self.max_speed, self.max_speed_z], device=self.device)
            target_vels = raw_noise * scale
            
            # Apply Z-axis tilt compensation
            # When drone flies horizontally, it must tilt, which reduces vertical lift
            # This adds a small positive bias to Z velocity to compensate
            if self.z_tilt_compensation > 0:
                target_vels[:, :, 2] += self.z_tilt_compensation

        
        # 2. Apply Human Filters (Low Pass & Deadband)
        # Low Pass: y[i] = alpha * x[i] + (1-alpha) * y[i-1]
        ##############################################################################
        # TODO：检查Filters（alpha和laziness）是否必要，因为柏林噪声已经是一个平滑的信号了，调整控制激进程度的参数只有采样频率
        # 反之，如果要模拟人类的不精确抖动和控制，应该叠加一个高频噪声而不是通过滤波平滑？
        # 目前在enable_filter=False时跳过滤波步骤
        ##############################################################################

        if enable_filter:
            # === PROFILING: Time the integration_loop ===
            profiler.start("user_model/integration_loop")

            alpha = 1.0 - self.styles['smoothness'][env_ids] # (K, 1)
            laziness = self.styles['laziness'][env_ids] # (K, 1)
            
            # Initialize filtered trajectory
            filtered_traj = torch.zeros_like(target_vels)
            last_val = self.prev_filtered_action[env_ids] # (K, 4)

            # 3. Integration loop for APF and Filter
            # We need to track position to apply APF
            curr_pos = start_pos.clone() # (K, 3)
            curr_quat = start_quat.clone() # (K, 4)

            # We assume simplified kinematics: pos += rotate(vel) * dt, yaw += yaw_rate * dt
            # TODO: simple integration may not be accurate for large dt or high speeds
            for t in range(T):
                # a. Get raw target for this step
                raw_v = target_vels[:, t] # (K, 4)
                
                # b. Apply Deadband (Laziness)
                # If raw input is small, set to 0
                mask_dead = raw_v.abs() < laziness
                raw_v = torch.where(mask_dead, torch.zeros_like(raw_v), raw_v)
                
                # c. Apply Low Pass Filter
                # val = alpha * raw + (1-alpha) * last
                # Note: alpha is (K, 1)
                curr_v = alpha * raw_v + (1.0 - alpha) * last_val
                last_val = curr_v
                
                # d. Apply APF (Artificial Potential Field)
                # Calculate repulsion based on curr_pos
                repulsion = self._calculate_apf(curr_pos) # (K, 3)
                
                # Add repulsion to linear velocity (local frame or world frame?)
                # APF is usually in World Frame.
                # curr_v is in Body Frame (Joystick input).
                # We need to project APF to Body Frame to modify joystick input, 
                # OR modify the result of integration.
                # The user wants "stick inputs" modified.
                # So we rotate APF (World) -> Body
                
                # Rotate repulsion to body frame
                # quat_rotate_inverse rotates World -> Body
                repulsion_b = quat_rotate_inverse(curr_quat, repulsion)
                
                # Add to linear parts (0:3)
                curr_v_modified = curr_v.clone()
                curr_v_modified[:, :3] += repulsion_b
                
                # Calculate proposed world velocity
                vel_world = quat_rotate(curr_quat, curr_v_modified[:, :3])
                
                # Propose next position
                next_pos = curr_pos + vel_world * dt
                
                # Clamp position to map boundaries to ensure trajectory validity
                # This prevents the generated trajectory from going through walls/floor
                # Note: Z-axis limit is 2 * map_range[2] consistent with APF calculation (assuming map_range is half-extents)
                next_pos[:, 0] = torch.clamp(next_pos[:, 0], -self.env_map_range[0], self.env_map_range[0])
                next_pos[:, 1] = torch.clamp(next_pos[:, 1], -self.env_map_range[1], self.env_map_range[1])
                next_pos[:, 2] = torch.clamp(next_pos[:, 2], 1.0, 2.0 * self.env_map_range[2])
                
                # Calculate effective world velocity based on clamped position
                effective_vel_world = (next_pos - curr_pos) / dt
                
                # --- DEBUG PROBE START ---
                # Check for velocity spikes
                # vel_norm = torch.norm(effective_vel_world, dim=-1)
                # if (vel_norm > self.max_speed * 4.0).any():
                #     idx = torch.argmax(vel_norm).item()
                #     self.logger.info(f"Spike detected at t={t} for env {idx}")
                #     self.logger.debug(f"  dt: {dt}")
                #     self.logger.debug(f"  target_v: {target_vels[idx, t].detach().cpu().numpy()}")
                #     self.logger.debug(f"  curr_pos: {curr_pos[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  next_pos (unclamped): {(curr_pos + vel_world * dt)[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  next_pos (clamped): {next_pos[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  effective_vel_world: {effective_vel_world[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  vel_world (pre-clamp): {vel_world[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  repulsion: {repulsion[idx].detach().cpu().numpy() if 'repulsion' in locals() else 'N/A'}")
                #     self.logger.debug(f"  curr_v: {curr_v[idx].detach().cpu().numpy()}")
                #     self.logger.debug(f"  curr_quat: {curr_quat[idx].detach().cpu().numpy()}")
                # --- DEBUG PROBE END ---

                # Clamp effective velocity to prevent explosions (e.g. if correcting from out-of-bounds)
                # Allow slightly higher than max_speed for correction, but not infinite
                limit = self.max_speed * 2.0
                effective_vel_world = torch.clamp(effective_vel_world, -limit, limit)
                
                # Convert effective velocity back to body frame for storage
                effective_vel_body = quat_rotate_inverse(curr_quat, effective_vel_world)
                
                # Update the linear part of the action to be stored
                curr_v_modified[:, :3] = effective_vel_body
                
                # e. Store in buffer
                filtered_traj[:, t] = curr_v_modified
                
                # Update curr_pos for next iteration
                curr_pos = next_pos
                
                # f. Integrate Yaw for next step
                # Update Yaw
                yaw_rate = curr_v_modified[:, 3]
                
                # Update yaw in quaternion (approximate)
                # Create a rotation quaternion for Z axis
                half_angle = yaw_rate * dt * 0.5
                s = torch.sin(half_angle)
                c = torch.cos(half_angle)
                
                qx, qy, qz, qw = curr_quat[:, 0], curr_quat[:, 1], curr_quat[:, 2], curr_quat[:, 3]
                
                nqx = qx * c + qy * s
                nqy = qy * c - qx * s
                nqz = qz * c + qw * s
                nqw = qw * c - qz * s
                curr_quat = torch.stack([nqx, nqy, nqz, nqw], dim=1)
                # Normalize quaternion to prevent drift
                curr_quat = curr_quat / torch.norm(curr_quat, dim=1, keepdim=True)
            
            profiler.stop("user_model/integration_loop")
        
            # Store trajectory
            self.action_buffer[env_ids] = filtered_traj
        else:
            # No filtering, directly store target_vels
            self.action_buffer[env_ids] = target_vels
            last_val = target_vels[:, -1, :]
        
        # Update state
        self.prev_filtered_action[env_ids] = last_val
        self.noise_time[env_ids] += T * dt

    def _calculate_apf(self, pos):
        """
        Calculate Artificial Potential Field force.
        pos: (K, 3)
        """
        force = torch.zeros_like(pos)
        margin = 2.0  # Distance threshold to start applying repulsion
        gain = self.repulsive_gain
        
        # Map limits
        # self.env_map_range is [Lx, Ly, Lz]
        # Bounds: [-Lx, Lx], [-Ly, Ly], [0.1, 2Lz] (z is half height)
        
        limits = self.env_map_range
        
        # X axis
        # if pos < -Lx + margin: force += gain * (margin - (pos - (-Lx)))
        # dist to min: pos - (-Lx) = pos + Lx
        d_min_x = pos[:, 0] + limits[0]
        mask_min_x = d_min_x < margin
        force[mask_min_x, 0] += gain * (margin - d_min_x[mask_min_x])
        
        # dist to max: Lx - pos
        d_max_x = limits[0] - pos[:, 0]
        mask_max_x = d_max_x < margin
        force[mask_max_x, 0] -= gain * (margin - d_max_x[mask_max_x])
        
        # Y axis
        d_min_y = pos[:, 1] + limits[1]
        mask_min_y = d_min_y < margin
        force[mask_min_y, 1] += gain * (margin - d_min_y[mask_min_y])
        
        d_max_y = limits[1] - pos[:, 1]
        mask_max_y = d_max_y < margin
        force[mask_max_y, 1] -= gain * (margin - d_max_y[mask_max_y])
        
        # Z axis (Floor and Ceiling)
        # Floor at 1.0 (to avoid ground collision)
        d_floor = pos[:, 2] - 1.0
        mask_floor = d_floor < margin
        force[mask_floor, 2] += gain * (margin - d_floor[mask_floor])
        
        # Ceiling at 2Lz
        d_ceil = 2 * limits[2] - pos[:, 2]
        mask_ceil = d_ceil < margin
        force[mask_ceil, 2] -= gain * (margin - d_ceil[mask_ceil])
        
        return force