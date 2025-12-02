import torch
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

def generate_action_from_stick_profile_batched(
        dt: float, 
        total_steps: int, 
        action_scale: torch.Tensor,
        right_target: dict[str, torch.Tensor], 
        left_target: dict[str, torch.Tensor], 
        params: dict,
        device: torch.device
    ):
    """
    Generate batched action sequence (local, normalized) from joystick profile.
    Uses Minimum Jerk Trajectory for human-like smooth control.
    Args:
        dt: time step
        total_steps: total number of steps to generate
        action_scale: scaling factor for (vel_x,vel_y,vel_z,yaw) (4,)
        right_target: dict of right joystick target values (pitch, roll) (N,)
        left_target: dict of left joystick target values (throttle, yaw) (N,)
        params: dict of timing parameters (delay, rise, hold, fall times)
        device: torch device
    Returns:
        actions: (num_envs, max_steps, 4) tensor
    """
    num_envs = right_target['pitch'].shape[0]
    
    # Time vector (1, T)
    t = (torch.arange(total_steps, device=device, dtype=torch.float32) * dt).unsqueeze(0) # (1, T)
    
    # Helper for Minimum Jerk profile generation
    def get_minimum_jerk_profile(target_val, delay, rise, hold, fall):
        # target_val, delay, rise, hold, fall: (N, 1)
        # t: (1, T)
        
        profile = torch.zeros(num_envs, total_steps, device=device)
        
        # Rise phase: 0 -> target
        t_start_rise = delay
        t_end_rise = delay + rise
        
        # Mask (N, T)
        mask_rise = (t >= t_start_rise) & (t < t_end_rise)
        
        # Compute polynomial for rise
        rise_safe = torch.where(rise > 1e-6, rise, torch.ones_like(rise) * 1e-6)
        tau_rise = (t - t_start_rise) / rise_safe
        poly_rise = 10 * tau_rise**3 - 15 * tau_rise**4 + 6 * tau_rise**5
        val_rise = target_val * poly_rise
        
        # Hold phase: target
        t_start_hold = t_end_rise
        t_end_hold = t_start_hold + hold
        mask_hold = (t >= t_start_hold) & (t < t_end_hold)
        
        # Fall phase: target -> 0
        t_start_fall = t_end_hold
        t_end_fall = t_start_fall + fall
        mask_fall = (t >= t_start_fall) & (t < t_end_fall)
        
        fall_safe = torch.where(fall > 1e-6, fall, torch.ones_like(fall) * 1e-6)
        tau_fall = (t - t_start_fall) / fall_safe
        poly_fall = 10 * tau_fall**3 - 15 * tau_fall**4 + 6 * tau_fall**5
        val_fall = target_val * (1.0 - poly_fall)
        
        # Combine phases
        # Initialize with zeros (already done)
        # Apply masks
        profile = torch.where(mask_rise, val_rise, profile)
        profile = torch.where(mask_hold, target_val, profile)
        profile = torch.where(mask_fall, val_fall, profile)
            
        return profile

    # Extract params
    delay_L = params['delay_time_L']
    delay_R = params['delay_time_R']
    rise_L = params['rise_time_L']
    rise_R = params['rise_time_R']
    hold_L = params['hold_time_L']
    hold_R = params['hold_time_R']
    fall_L = params['fall_time_L']
    fall_R = params['fall_time_R']
    noise_hf = params['noise_hf']
    noise_lf = params['noise_lf']

    # Generate base profiles
    vx = get_minimum_jerk_profile(right_target.get('pitch', 0.0), delay_R, rise_R, hold_R, fall_R)
    vy = get_minimum_jerk_profile(right_target.get('roll', 0.0), delay_R, rise_R, hold_R, fall_R)
    vz = get_minimum_jerk_profile(left_target.get('throttle', 0.0), delay_L, rise_L, hold_L, fall_L)
    yaw = get_minimum_jerk_profile(left_target.get('yaw', 0.0), delay_L, rise_L, hold_L, fall_L)

    actions = torch.stack([vx, vy, vz, yaw], dim=2) # (N, T, 4)
    actions = actions * action_scale.unsqueeze(0).unsqueeze(0)  # scale to physical units

    # Add Human-like Noise
    # High Frequency Noise
    if (noise_hf > 0).any():
        actions += torch.randn_like(actions) * noise_hf.unsqueeze(2)

    # Low Frequency Noise
    if (noise_lf > 0).any():
        freq1, freq2 = 0.3, 0.7 # Hz
        phase = torch.rand(num_envs, 4, device=device) * 2 * torch.pi
        
        t_exp = t.unsqueeze(2) # (1, T, 1)
        phase_exp = phase.unsqueeze(1) # (N, 1, 4)
        
        drift = (torch.sin(2 * torch.pi * freq1 * t_exp + phase_exp) + 
                 0.5 * torch.sin(2 * torch.pi * freq2 * t_exp))
        actions += drift * noise_lf.unsqueeze(2)

    return actions
    

def sample_joystick_profile_batched(num_envs, style, noise_level, device, min_duration=3.0, max_duration=5.0):
    """
    Sample random joystick profile parameters based on human style (Batched).
    """
    # style parameters (N, 1)
    agg = style.get('aggressiveness', torch.full((num_envs, 1), 0.5, device=device))
    dex = style.get('dexterity', torch.full((num_envs, 1), 0.5, device=device))
    nl = noise_level

    # 1. Duration
    t_total = torch.rand(num_envs, 1, device=device) * (max_duration - min_duration) + min_duration

    # 2. Targets

    # fixed 50% probability of activation  # TODO: adjust based on style parameters
    prob_activation = torch.full((num_envs, 1), 0.5, device=device)
    
    def get_target_val_batched():
        # TODO：参考当前无人机的状态（世界坐标），防止生成的动作飞离边界（暂时考虑上下边界）
        # Activation mask
        active = torch.rand(num_envs, 1, device=device) <= prob_activation
        
        # Value in [0.2, 1] based on agg
        # val = random.uniform(0.2, 0.5 + 0.5 * agg)
        val_mag = 0.2 + torch.rand(num_envs, 1, device=device) * (0.3 + 0.5 * agg)
        
        # Direction
        sign = torch.where(torch.rand(num_envs, 1, device=device) > 0.5, 1.0, -1.0)
        
        # if active, set to signed value, else 0
        return torch.where(active, val_mag * sign, torch.zeros_like(val_mag))

    right_target = {
        'pitch': get_target_val_batched(),
        'roll': get_target_val_batched(),
    }
    left_target = {
        'throttle': get_target_val_batched(), 
        'yaw': get_target_val_batched()
    }

    # 3. Timing Params
    delay_base = 0.1 + 0.3 * (1.0 - agg) # range from [0.1, 0.4]
    rise_base = 0.2 + 0.6 * (1.0 - agg)  # range from [0.2, 0.8]
    
    def rand_range(low, high):
        return low + torch.rand(num_envs, 1, device=device) * (high - low)
    
    delay_time_L = rand_range(torch.zeros_like(delay_base), delay_base)
    delay_time_R = rand_range(torch.zeros_like(delay_base), delay_base)
    
    rise_time_L = rand_range(rise_base * 0.8, rise_base * 1.2)
    rise_time_R = rand_range(rise_base * 0.8, rise_base * 1.2)
    
    # Hold time limits
    # limit = max(0.5, t_total - rise_base - delay_base)
    limit_L = torch.clamp(t_total - rise_base - delay_base, min=0.5)
    limit_R = torch.clamp(t_total - rise_base - delay_base, min=0.5)
    # TODO: check the hold time model
    hold_time_L = rand_range(torch.zeros_like(limit_L), limit_L)
    hold_time_R = rand_range(torch.full_like(limit_R, 0.5), limit_R)
    
    fall_time_L = rand_range(torch.full_like(agg, 0.2), torch.full_like(agg, 0.5))
    fall_time_R = rand_range(torch.full_like(agg, 0.2), torch.full_like(agg, 0.5))
    
    params = {
        'delay_time_L': delay_time_L,
        'delay_time_R': delay_time_R,
        'rise_time_L': rise_time_L,
        'rise_time_R': rise_time_R,
        'hold_time_L': hold_time_L,
        'hold_time_R': hold_time_R,
        'fall_time_L': fall_time_L,
        'fall_time_R': fall_time_R,
        'noise_hf': nl * (1.0 - dex), 
        'noise_lf': nl * 4.0 * (1.0 - dex)
    }
    
    return t_total, right_target, left_target, params


# User Model to simulate human actions
# TODO: Gengerate action based on a trajectory pre-sampling
class UserModel:
    def __init__(self, num_envs, cfg, lidar, lidar_resolution):
        self.num_envs = num_envs
        self.cfg = cfg
        
        # Init cfg parameters
        self.env_map_range = torch.tensor(cfg.env.map_range, dtype=torch.float32, device=cfg.device)  # half extents of the env map
        self.device = cfg.device
        self.lidar_range = cfg.sensor.lidar_range
        self.max_speed = cfg.algo.actor.action_limit  # max speed limit during training

        self.dt = cfg.sim.dt
        self.min_duration = cfg.user_model.min_duration
        self.max_duration = cfg.user_model.max_duration

        # Get RayCaster Lidar object from env
        self.lidar = lidar
        self.lidar_resolution = lidar_resolution  # (hbeams, vbeams)

        # sample random style parameters for the user model
        self.human_style = {
            'conformance': torch.rand(self.num_envs, 1, device=self.device),  # alpha
            'aggressiveness': torch.rand(self.num_envs, 1, device=self.device) , # beta
            'dexterity': torch.rand(self.num_envs, 1, device=self.device) # gamma
        }
        # TODO: more style params ...
        self.speed_delta = torch.rand(self.num_envs, 1, device=self.device)  # delta
        self.noise_level = 0.05 * torch.rand(self.num_envs, 1, device=self.device)  # noise level range [0.00, 0.05]

        # batched buffers
        self.intent_goals = torch.zeros(self.num_envs, 3, device=self.device)
        # self.intent_timers = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # height range limit: [minz, maxz] * envs
        self.height_range = torch.zeros(self.num_envs, 1, 2, device=self.device)

        # self.prev_joystick_action = torch.zeros(self.num_envs, 4, device=self.device) # user action output (simulate joystick) last step
        self.prev_actual_action = torch.zeros(self.num_envs, 4, device=self.device) # actual action taken by the policy last step

        # Buffer for pre-sampled trajectories

        self.max_traj_steps = int(self.max_duration / self.dt)  # max steps per trajectory
        self.current_trajectory_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # total steps in current trajectory (<= max_traj_steps)

        # 初始化预采样轨迹的缓冲区，每组轨迹的最大步数为max_traj_steps，实际读取时，长度(即[1])依据current_trajectory_steps决定，剩下的部分以0填充
        self.action_trajectories = torch.zeros(self.num_envs, self.max_traj_steps, 4, device=self.device)  # pre-sampled action trajectories
        self.traj_step_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # current step index in trajectory

    # def _init_beam_angles(self):
    #     H = self.lidar_resolution[0]
    #     # evenly spread in [-pi, pi)
    #     self.beam_angles = torch.linspace(-torch.pi, torch.pi, steps=H+1, device=self.device)[:-1]

    def reset(self, pos, quat, env_ids):
        """
        Reset timers and batched buffers, 
        resample new intent trajectory for the given env ids
        called whenever the environment resets.

        Args:
            pos: (K,1,3) or (K,3) positions of the reset envs in world frame.
            quat: (K,1,4) or (K,4) orientations of the reset envs in world frame.
            env_ids: (K,) long tensor, indices in [0, num_envs).
            lidar_scan: (K,1,H,V) lidar scan for the reset envs, or None.
        """
        # ------- make sure get shape as (K,3)/(K,4) -------
        if pos.ndim == 3:
            pos_k = pos.squeeze(1)
        else:
            pos_k = pos

        if quat.ndim == 3:
            quat_k = quat.squeeze(1)
        else:
            quat_k = quat

        K = env_ids.numel()

        # Resample style parameters for these envs (optional)
        self.human_style['conformance'][env_ids] = torch.rand(K, 1, device=self.device)
        self.human_style['aggressiveness'][env_ids] = torch.rand(K, 1, device=self.device)
        self.human_style['dexterity'][env_ids] = torch.rand(K, 1, device=self.device)
        self.speed_delta[env_ids] = torch.rand(K, 1, device=self.device)
        self.noise_level[env_ids] = 0.05 + 0.1 * torch.rand(K, 1, device=self.device)

        # Sample the first batch of intent trajectories
        new_traj = self._sample_intent_traj(env_ids=env_ids)
        
        # Calculate approximate goal position from velocity trajectory
        # Integrate velocity considering yaw rotation
        pos_delta = self._integrate_trajectory(new_traj, quat_k)
        new_goals = pos_k + pos_delta

        self.intent_goals[env_ids] = new_goals

        # Update height range based on current pos and new goals
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], new_goals[:, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], new_goals[:, 2])

        # Reset previous actions buffers
        # self.prev_joystick_action[env_ids] = 0.0
        self.prev_actual_action[env_ids] = 0.0
        
        # Reset trajectory indices
        self.traj_step_indices[env_ids] = 0

    def _integrate_trajectory(self, actions, init_quat):
        """
        Integrate action trajectory (velocities in Heading Frame) to get world displacement.
        Args:
            actions: (K, T, 4) tensor [vx, vy, vz, yaw_rate]
            init_quat: (K, 4) tensor, initial orientation
        Returns:
            pos_delta: (K, 3) tensor, total displacement in world frame
        """
        dt = self.dt
        K = actions.shape[0]
        
        # 1. Get initial yaw from quaternion
        # Rotate X-axis (1,0,0) by init_quat to get heading vector
        unit_x = torch.zeros(K, 3, device=self.device)
        unit_x[:, 0] = 1.0
        heading_vec = quat_rotate(init_quat, unit_x)
        init_yaw = torch.atan2(heading_vec[:, 1], heading_vec[:, 0]) # (K,)
        
        # 2. Calculate yaw sequence
        # actions: (K, T, 4) -> [vx, vy, vz, yaw_rate]
        yaw_rates = actions[..., 3]
        # Cumulative yaw change: cumsum(yaw_rate * dt)
        yaw_deltas = torch.cumsum(yaw_rates * dt, dim=1) # (K, T)
        
        # Yaw at start of each step t (approximate for integration)
        # yaw_t[i] = init_yaw + sum(yaw_rate[0:i]) * dt
        # We shift yaw_deltas to get the yaw at the beginning of the interval
        yaw_t_end = init_yaw.unsqueeze(1) + yaw_deltas
        yaw_t_start = yaw_t_end - (yaw_rates * dt)
        
        # 3. Rotate velocities to world frame
        # Assuming vx, vy are in the Heading Frame (Level, Yaw-rotated)
        vx = actions[..., 0]
        vy = actions[..., 1]
        vz = actions[..., 2]
        
        cos_yaw = torch.cos(yaw_t_start)
        sin_yaw = torch.sin(yaw_t_start)
        
        # Rotate (vx, vy) in 2D plane
        vx_w = vx * cos_yaw - vy * sin_yaw
        vy_w = vx * sin_yaw + vy * cos_yaw
        
        # 4. Sum up displacements
        dx = torch.sum(vx_w, dim=1) * dt
        dy = torch.sum(vy_w, dim=1) * dt
        dz = torch.sum(vz, dim=1) * dt
        
        return torch.stack([dx, dy, dz], dim=1)

    def _sample_intent_traj(self, env_ids=None):
        """
        Sample human actions by simulate controlller input.
        Returns:
            trajectories: (K, MaxSteps, 4) tensor of actions
        """
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, int):
            # Fallback if int is passed
            ids = torch.tensor([env_ids], device=self.device)
        else:
            ids = env_ids
            
        K = ids.numel()
        if K == 0:
            return torch.empty(0, self.max_traj_steps, 4, device=self.device)
        
        # Get style for this env
        style = {k: v[ids] for k, v in self.human_style.items()}
        nl = self.noise_level[ids]
        
        # sample target action profile
        t_total, right_target, left_target, params = sample_joystick_profile_batched(K, style, nl, self.device)

        # Calculate actual duration for each env based on params
        duration_R = params['delay_time_R'] + params['rise_time_R'] + params['hold_time_R'] + params['fall_time_R']
        duration_L = params['delay_time_L'] + params['rise_time_L'] + params['hold_time_L'] + params['fall_time_L']
        actual_duration = torch.max(duration_R, duration_L) # (K, 1)
        
        # Convert to steps
        traj_steps = (actual_duration / self.dt).ceil().long().squeeze(1) # (K,)
        
        # Clamp to max_traj_steps
        batch_steps = torch.clamp(traj_steps, max=self.max_traj_steps)
        
        # get action trajectory from joystick profile
        # batch_trajs shape: (K, steps, 4)
        max_steps_in_batch = int(batch_steps.max().item())
        # make scale of only x,y to max_speed, z,yaw remain unchanged
        scale_factor = torch.tensor(
            [self.max_speed, self.max_speed, 1., 1.], 
            dtype=torch.float32, 
            device=self.device
        )
        batch_trajs = generate_action_from_stick_profile_batched(
            dt=self.dt,
            total_steps=max_steps_in_batch,
            action_scale=scale_factor,
            right_target=right_target,
            left_target=left_target,
            params=params,
            device=self.device
        )
        
        # Store in the main buffer
        self.current_trajectory_steps[ids] = batch_steps

        # Pad batch_trajs to max_traj_steps and store
        current_len = batch_trajs.shape[1]
        if current_len < self.max_traj_steps:
            padding = torch.zeros(K, self.max_traj_steps - current_len, 4, device=self.device)
            self.action_trajectories[ids] = torch.cat([batch_trajs, padding], dim=1)
        else:
            self.action_trajectories[ids] = batch_trajs[:, :self.max_traj_steps]

        self.traj_step_indices[ids] = 0 # Reset indices for these envs
        
        return batch_trajs

    def step(self, drone_state, prev_agent_action):
        # print(f"UserModel step called. Indices[0]: {self.traj_step_indices[0]}")
        
        # Get drone state from environment
        N = self.num_envs
        drone_pos_w = drone_state[:, 0:3]  # (N,3)
        drone_vel_w = drone_state[:, 3:6]  # (N,3)
        drone_orientation_q = drone_state[:, 6:10]  # (N,4)

        # get previous actual actions from environment history
        self.prev_actual_action = prev_agent_action.detach()

        # 1. Check if need new intent trajectory
        # Simple Condition: a trajectory finished
        traj_finished = self.traj_step_indices >= self.current_trajectory_steps
        
        # Filter out drones that have reached current goal and need new intent
        need_new_intent = traj_finished  # (N) boolean tensor
        if need_new_intent.any():
            # idx = if need_new_intent is True
            idx = need_new_intent.nonzero(as_tuple=False).squeeze(-1)
            
            # Resample trajectory for these envs
            new_traj = self._sample_intent_traj(env_ids=idx)
            
            # Update goals based on new trajectory
            # Use current orientation to integrate the new trajectory
            # TODO：计算intent goal时，是否考虑无人机当前的速度和位置（因为实际动作会有误差累积）
            current_quat = drone_orientation_q[idx]
            pos_delta = self._integrate_trajectory(new_traj, current_quat)
            
            self.intent_goals[idx] = drone_pos_w[idx] + pos_delta

        # 2. Get action from pre-sampled trajectory
        # indices: (N,)
        # trajectories: (N, MaxSteps, 4)
        # We need to gather: trajectories[i, indices[i], :]
        
        # Clamp indices to avoid out of bound (though we reset them)
        safe_indices = self.traj_step_indices.clamp(max=self.max_traj_steps - 1)
        
        # Gather actions
        # (N, 1, 4)
        action_local = torch.gather(
            self.action_trajectories, 
            1, 
            safe_indices.view(-1, 1, 1).expand(-1, 1, 4)
        ).squeeze(1)
        
        # Increment indices
        self.traj_step_indices += 1

        # Use action directly (noise is already in profile)
        # TODO: 添加alpha参数（conformance）对user动作的影响
        au_local_noisy = action_local

        # # 3. Convert local action to world frame
        # # Split linear and vertical components
        # vels_u_b_noisy = au_local_noisy[:, 0:3]
        # au_world_noisy = torch.cat([quat_rotate(drone_orientation_q, vels_u_b_noisy), au_local_noisy[:, 3:4]], dim=-1)

        # # Update previous joystick action based on current noisy action
        # self.prev_joystick_action = au_world_noisy.detach() # Store for next step
        
        return au_local_noisy, need_new_intent, self.intent_goals
    
    def get_height_range(self):
        """
        Get the height range [minz, maxz] for each env based on current pos and intent goal
        Returns:
            height_range: (num_envs, 2) tensor, minz and maxz for each env
        """
        return self.height_range
    
    # def _visualize_single_env(self, env_idx, pos, goal, action_b):
    #     """
    #     Visualize the drone position and intent goal in the environment using debug drawing
    #     Inputs: 
    #         env_idx: int, index of the environment to visualize
    #         pos: (3,) tensor, drone position in world frame
    #         goal: (3,) tensor, intent goal position in world frame
    #     """
    #     if self.debug_draw is None:
    #         return

    #     p0 = pos.reshape(1, 3)
    #     g = goal.reshape(1, 3)
    #     vel_b = action_b[0:3].reshape(1, 3)

    #     self.debug_draw.clear()
    #     g_top = g + torch.tensor([0, 0, 0.5], device=self.device) # 向上 0.5m
    #     line_points = torch.cat([g, g_top], dim=0) # (2, 3)
        
    #     # draw goal (red)
    #     self.debug_draw.plot(x=line_points, size=12.0, color=(1.0,0.0,0.0,1.0))

    #     # draw goal vector (green)
    #     self.debug_draw.vector(p0, (g - p0), size=2.0, color=(0.1,1.0,0.1,1.0))

    #     # draw current velocity vector (blue)
    #     self.debug_draw.vector(p0, vel_b, size=2.0, color=(0.1,0.1,1.0,1.0))