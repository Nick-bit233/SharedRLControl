import torch
import numpy as np
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
from noise import pnoise1  # TODO: use perlin-1d lib instead for better performance

def batched_perlin_noise(
    time: torch.Tensor, 
    seeds: torch.Tensor, 
    freq: torch.Tensor,
    device: torch.device
):
    """
    Generate batched 1D Perlin-like Gradient Noise.
    Args:
        time: (N, T) time tensor
        seeds: (N, 4) seeds for 4 channels
        freq: (N, 1) frequency scaling
    Returns:
        noise: (N, T, 4) values in approx [-1, 1]
    """
    # Expand dims for broadcasting
    # time: (N, T) -> (N, T, 1)
    t = time.unsqueeze(-1) * freq.unsqueeze(1) # (N, T, 4)
    
    t_floor = t.floor().long()
    t_ceil = t_floor + 1
    
    # Hash function to get gradients
    # We use a simple pseudo-random hash based on seeds and integer time
    def get_grad(t_int):
        # (N, T, 4)
        # Mix seed and time
        h = (seeds.unsqueeze(1) + t_int * 1327217881) ^ 0x5DEECE66D
        h = (h ^ (h >> 13)) * 1274126177
        h = h ^ (h >> 16)
        # Map to [-1, 1]
        grad = ((h & 0x7FFFFFFF).float() / 0x7FFFFFFF) * 2.0 - 1.0
        return grad

    g0 = get_grad(t_floor)
    g1 = get_grad(t_ceil)
    
    dt = t - t_floor.float()
    
    # Fade function (smootherstep)
    u = dt * dt * dt * (dt * (dt * 6 - 15) + 10)
    
    # Interpolate
    # Dot product of gradient and distance vector (1D, so just mult)
    n0 = g0 * dt
    n1 = g1 * (dt - 1)
    
    noise = n0 + u * (n1 - n0)
    return noise

class UserModel:
    def __init__(self, num_envs, cfg, use_lib_noise=False):
        self.num_envs = num_envs
        self.cfg = cfg
        self.use_lib_noise = use_lib_noise
        if self.use_lib_noise:
            print("[Warning]: Using 'noise' library for Perlin noise generation. This runs on cpu and may be slower than the batched implementation.")
        self.device = cfg.device
        self.dt = cfg.sim.dt
        
        # Map boundaries for APF [x, y, z]
        # Assuming map_range is half-extents
        self.env_map_range = torch.tensor(
            cfg.env.map_range, dtype=torch.float32, device=self.device
        )
        
        # Parameters
        # training frame num or max_episode_length ?
        self.buffer_size = cfg.algo.training_frame_num # steps (e.g. 128 frames is about 2 seconds)
        self.repulsive_gain = 1.0
        self.max_speed = cfg.algo.actor.action_limit
        
        # State
        self.action_buffer = torch.zeros(num_envs, self.buffer_size, 4, device=self.device)
        self.buffer_read_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.noise_time = torch.zeros(num_envs, device=self.device)
        
        # Random seeds for noise (N, 4)
        self.noise_seeds = torch.randint(0, 100000, (num_envs, 4), device=self.device)
        
        # Style parameters (randomized per env)
        self.styles = {
            'noise_freq': torch.rand(num_envs, 1, device=self.device) * 0.05 + 0.05, # 0.05 - 0.1
            'smoothness': torch.rand(num_envs, 1, device=self.device) * 0.5 + 0.2, # 0.2 - 0.7
            'laziness': torch.rand(num_envs, 1, device=self.device) * 0.2, # Deadband threshold 0 - 0.2
        }
        
        # Previous action for smoothing (Low Pass Filter)
        self.prev_filtered_action = torch.zeros(num_envs, 4, device=self.device)
        
    def reset(self, pos, quat, env_ids):
        """
        Reset state for env_ids
        """
        if pos.ndim == 3: pos = pos.squeeze(1)
        if quat.ndim == 3: quat = quat.squeeze(1)
        
        K = len(env_ids)
        
        # Reset time and indices
        self.noise_time[env_ids] = 0
        self.buffer_read_idx[env_ids] = 0
        
        # New seeds
        self.noise_seeds[env_ids] = torch.randint(0, 100000, (K, 4), device=self.device)
        
        # Randomize styles
        # freq range: 0.05 - 0.1 for lower change rate
        self.styles['noise_freq'][env_ids] = torch.rand(K, 1, device=self.device) * 0.05 + 0.05
        self.styles['smoothness'][env_ids] = torch.rand(K, 1, device=self.device) * 0.5 + 0.2
        self.styles['laziness'][env_ids] = torch.rand(K, 1, device=self.device) * 0.2
        
        self.prev_filtered_action[env_ids] = 0.0
        
        # Refill buffer
        self._refill_buffer(env_ids, pos, quat)

    def step(self, drone_state, prev_agent_action):
        # drone_state: (N, 13) -> pos, vel, quat, ang_vel
        pos = drone_state[..., :3]
        quat = drone_state[..., 6:10]
        
        if pos.ndim == 3: pos = pos.squeeze(1)
        if quat.ndim == 3: quat = quat.squeeze(1)
        
        # Check refill
        needs_refill = self.buffer_read_idx >= self.buffer_size
        if needs_refill.any():
            idxs = needs_refill.nonzero(as_tuple=False).squeeze(-1)
            # For refill, we need current pos/quat to start integration
            self._refill_buffer(idxs, pos[idxs], quat[idxs])
            self.buffer_read_idx[idxs] = 0
            
        # Read action from buffer
        # action_buffer: (N, T, 4)
        # We need to select [i, read_idx[i], :]
        # Use gather
        read_indices = self.buffer_read_idx.view(-1, 1, 1).expand(-1, 1, 4)
        action = torch.gather(self.action_buffer, 1, read_indices).squeeze(1)
        
        self.buffer_read_idx += 1
        
        # Update intent goals (just for visualization/compatibility, set to current pos + action direction)
        # self.intent_goals = pos + quat_rotate(quat, action[..., :3])
        
        return action, needs_refill

    def _generate_lib_noise(self, time_grid, seeds, freq):
        K, T = time_grid.shape
        time_np = time_grid.cpu().numpy()
        seeds_np = seeds.cpu().numpy()
        freq_np = freq.cpu().numpy()
        
        noise_data = np.zeros((K, T, 4), dtype=np.float32)
        
        for k in range(K):
            f = freq_np[k, 0]
            for ch in range(4):
                base = int(seeds_np[k, ch] % 256)
                for t in range(T):
                    noise_data[k, t, ch] = pnoise1(time_np[k, t] * f, base=base)
                    
        return torch.from_numpy(noise_data).to(self.device)

    def _refill_buffer(self, env_ids, start_pos, start_quat):
        """
        Generate a trajectory of actions using Perlin noise and APF.
        """
        K = len(env_ids)
        T = self.buffer_size
        dt = self.dt
        
        # 1. Generate Perlin Noise for the whole batch (K, T, 4)
        # Time vector
        t_start = self.noise_time[env_ids].unsqueeze(1) # (K, 1)
        t_steps = torch.arange(T, device=self.device).unsqueeze(0) * dt # (1, T)
        time_grid = t_start + t_steps # (K, T)
        
        seeds = self.noise_seeds[env_ids] # (K, 4)
        freq = self.styles['noise_freq'][env_ids] # (K, 1)
        
        # Generate raw noise (Target Velocities)
        # (K, T, 4)
        if self.use_lib_noise:
            raw_noise = self._generate_lib_noise(time_grid, seeds, freq)
            # check if raw_noise is in [-1, 1]
            assert torch.all(raw_noise >= -1.0) and torch.all(raw_noise <= 1.0), "Raw noise out of expected range [-1, 1]"
        else:
            raw_noise = batched_perlin_noise(time_grid, seeds, freq, self.device)
        
        # Scale noise to physical units
        # vx, vy, vz, yaw_rate
        # Assume output is [-1, 1], scale to max_speed
        scale = torch.tensor([self.max_speed, self.max_speed, self.max_speed, 2.0], device=self.device)
        target_vels = raw_noise * scale
        
        # 2. Apply Human Filters (Low Pass & Deadband)
        # We can do this on the whole trajectory
        # Low Pass: y[i] = alpha * x[i] + (1-alpha) * y[i-1]
        # This is sequential, but T is small (100).
        
        alpha = 1.0 - self.styles['smoothness'][env_ids] # (K, 1)
        laziness = self.styles['laziness'][env_ids] # (K, 1)
        
        # Initialize filtered trajectory
        filtered_traj = torch.zeros_like(target_vels)
        last_val = self.prev_filtered_action[env_ids] # (K, 4)
        
        # 3. Integration loop for APF and Filter
        # We need to track position to apply APF
        curr_pos = start_pos.clone() # (K, 3)
        curr_quat = start_quat.clone() # (K, 4)
        
        # Pre-compute rotation for efficiency? No, quat changes with yaw rate.
        # We assume simplified kinematics: pos += rotate(vel) * dt, yaw += yaw_rate * dt
        
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
            #     print(f"[DEBUG] Spike detected at t={t} for env {idx}")
            #     print(f"  dt: {dt}")
            #     print(f"  curr_pos: {curr_pos[idx].detach().cpu().numpy()}")
            #     print(f"  next_pos (unclamped): {(curr_pos + vel_world * dt)[idx].detach().cpu().numpy()}")
            #     print(f"  next_pos (clamped): {next_pos[idx].detach().cpu().numpy()}")
            #     print(f"  effective_vel_world: {effective_vel_world[idx].detach().cpu().numpy()}")
            #     print(f"  vel_world (pre-clamp): {vel_world[idx].detach().cpu().numpy()}")
            #     print(f"  repulsion: {repulsion[idx].detach().cpu().numpy() if 'repulsion' in locals() else 'N/A'}")
            #     print(f"  curr_v: {curr_v[idx].detach().cpu().numpy()}")
            #     print(f"  curr_quat: {curr_quat[idx].detach().cpu().numpy()}")
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
            
        # Store trajectory
        self.action_buffer[env_ids] = filtered_traj
        
        # Update state
        self.prev_filtered_action[env_ids] = last_val
        self.noise_time[env_ids] += T * dt

    def _calculate_apf(self, pos):
        """
        Calculate Artificial Potential Field force.
        pos: (K, 3)
        """
        force = torch.zeros_like(pos)
        margin = 0.5
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

    @DeprecationWarning
    def get_height_range(self):
        return torch.zeros(self.num_envs, 1, 2, device=self.device) # Dummy