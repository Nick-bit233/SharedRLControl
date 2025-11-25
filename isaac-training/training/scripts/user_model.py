import torch
import random
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

def generate_action_from_stick_profile(
        dt=0.01, t_total=3.0,
        right_target={'pitch':1.0,'roll':0.0,'throttle':0.0,'yaw':0.0},
        left_target={'pitch':0.0,'roll':0.0,'throttle':0.0,'yaw':0.5},
        params=None
    ):
    """
    Generate action sequence (local, normalized) from joystick profile.
    Uses Minimum Jerk Trajectory for human-like smooth control.
    
    num_steps = int(t_total / dt)
    Params:
        dt: float, time step
        t_total: float, total time duration
        right_target: dict, target action values for right stick
        left_target: dict, target action values for left stick
        params: dict, profile parameters
            'delay_time_L': float, delay time before starting left stick
            'delay_time_R': float, delay time before starting right stick
            'rise_time_L': float, time to reach target from 0.0
            'rise_time_R': float, time to reach target from 0.0
            'hold_time_L': float, time to hold at target value
            'hold_time_R': float, time to hold at target value
            'fall_time_L': float, time to return to 0.0
            'fall_time_R': float, time to return to 0.0
            'noise_hf': float, high frequency noise level (jitter)
            'noise_lf': float, low frequency noise level (drift)
    Returns:
        actions: (num_steps, 4) tensor, action sequence over time [vx, vy, vz, yaw_rate]
    """
    if params is None:
        params = {}

    # 1. Parse Parameters with defaults
    delay_L = params.get('delay_time_L', 0.2)
    delay_R = params.get('delay_time_R', 0.8)
    rise_L = params.get('rise_time_L', 0.6) # Slightly slower rise for smooth control
    rise_R = params.get('rise_time_R', 0.3)
    hold_L = params.get('hold_time_L', 1.0)
    hold_R = params.get('hold_time_R', 0.0)
    fall_L = params.get('fall_time_L', 0.6)
    fall_R = params.get('fall_time_R', 0.3)
    noise_hf = params.get('noise_hf', 0.005)
    noise_lf = params.get('noise_lf', 0.2)

    num_steps = int(t_total / dt)
    t = torch.linspace(0, t_total, num_steps)

    # 2. Helper for Minimum Jerk profile generation
    # This creates a bell-shaped velocity profile (smooth acceleration)
    # Formula: x(t) = x0 + (xf - x0) * (10t^3 - 15t^4 + 6t^5) for t in [0, 1]
    def get_minimum_jerk_profile(target_val, delay, rise, hold, fall):
        profile = torch.zeros_like(t)
        
        # Rise phase: 0 -> target
        t_start_rise = delay
        t_end_rise = delay + rise
        mask_rise = (t >= t_start_rise) & (t < t_end_rise)
        if rise > 1e-6:
            tau = (t[mask_rise] - t_start_rise) / rise
            # Minimum Jerk Polynomial: 10t^3 - 15t^4 + 6t^5
            # This ensures zero velocity and acceleration at start and end of the transition
            poly = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            profile[mask_rise] = target_val * poly
        
        # Hold phase: target
        t_start_hold = t_end_rise
        t_end_hold = t_start_hold + hold
        mask_hold = (t >= t_start_hold) & (t < t_end_hold)
        profile[mask_hold] = target_val
        
        # Fall phase: target -> 0
        t_start_fall = t_end_hold
        t_end_fall = t_start_fall + fall
        mask_fall = (t >= t_start_fall) & (t < t_end_fall)
        if fall > 1e-6:
            tau = (t[mask_fall] - t_start_fall) / fall
            poly = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            profile[mask_fall] = target_val * (1.0 - poly)
            
        return profile

    # 3. Generate base profiles for each channel
    # Mapping (Mode 2):
    # Right Stick: Pitch -> vx, Roll -> vy
    # Left Stick: Throttle -> vz, Yaw -> yaw_rate
    
    vx = get_minimum_jerk_profile(right_target.get('pitch', 0.0), delay_R, rise_R, hold_R, fall_R)
    vy = get_minimum_jerk_profile(right_target.get('roll', 0.0), delay_R, rise_R, hold_R, fall_R)
    vz = get_minimum_jerk_profile(left_target.get('throttle', 0.0), delay_L, rise_L, hold_L, fall_L)
    yaw = get_minimum_jerk_profile(left_target.get('yaw', 0.0), delay_L, rise_L, hold_L, fall_L)

    actions = torch.stack([vx, vy, vz, yaw], dim=1) # (num_steps, 4)

    # 4. Add Human-like Noise
    # High Frequency Noise (Tremor/Jitter)
    if noise_hf > 0:
        actions += torch.randn_like(actions) * noise_hf

    # Low Frequency Noise (Drift / Correction)
    # Simulates the user slowly correcting or drifting around the target value
    if noise_lf > 0:
        # Sum of sines for organic drift
        freq1, freq2 = 0.3, 0.7 # Hz
        phase = torch.rand(4) * 2 * torch.pi
        drift = (torch.sin(2 * torch.pi * freq1 * t.unsqueeze(1) + phase.unsqueeze(0)) + 
                 0.5 * torch.sin(2 * torch.pi * freq2 * t.unsqueeze(1)))
        actions += drift * noise_lf

    return actions
    

def sample_joystick_profile(style, noise_level, min_duration=3.0, max_duration=5.0):
    """
    Sample random joystick profile parameters based on human style.
    Args:
        style: dict containing 'aggressiveness', 'dexterity' (scalars or 1-element tensors)
        noise_level: float or scalar tensor
        max_duration: float, max duration of the profile
    Returns:
        t_total: float
        right_target: dict
        left_target: dict
        params: dict
    """
    # Helper to convert tensor to float
    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return float(x)

    # style parameters
    # agg: aggressiveness [0, 1], dex: dexterity [0, 1]
    agg = to_float(style.get('aggressiveness', 0.5))
    dex = to_float(style.get('dexterity', 0.5))
    nl = to_float(noise_level)

    # 1. Duration
    # Random duration between 1.5 and max_duration
    t_total = random.uniform(min_duration, max_duration)

    # 2. Targets
    # Probability of activating an axis
    prob_activation = 0.3 + 0.4 * agg  # range from [0.3, 0.7]
    
    def get_target_val():
        """
        Sample target value in [-1, 1] or 0.0 based on activation probability.
        TODO：参考当前无人机的状态，防止生成的动作飞离边界（暂时考虑上下边界）
        """
        if random.random() > prob_activation:
            return 0.0
        # Value in [-1, 1]
        val = random.uniform(0.2, 0.5 + 0.5 * agg)  # random value, range from [0.2, 1]
        ret_val = val if random.random() > 0.5 else -val  # random direction
        return ret_val

    right_target = {
        'pitch': get_target_val(),
        'roll': get_target_val(),
    }
    left_target = {
        'throttle': get_target_val(), 
        'yaw': get_target_val()
    }

    # 3. Timing Params
    delay_base = 0.1 + 0.3 * (1.0 - agg) 
    rise_base = 0.2 + 0.6 * (1.0 - agg)
    
    params = {
        'delay_time_L': random.uniform(0.0, delay_base),
        'delay_time_R': random.uniform(0.0, delay_base),
        'rise_time_L': random.uniform(rise_base * 0.8, rise_base * 1.2),
        'rise_time_R': random.uniform(rise_base * 0.8, rise_base * 1.2),
        'hold_time_L': random.uniform(0.5, max(0.5, t_total - rise_base - delay_base)),
        'hold_time_R': random.uniform(0.5, max(0.5, t_total - rise_base - delay_base)),
        'fall_time_L': random.uniform(0.2, 0.5),
        'fall_time_R': random.uniform(0.2, 0.5),
        'noise_hf': nl * (1.0 - dex), 
        'noise_lf': nl * 2.0 * (1.0 - dex)
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

        self.dt = cfg.sim.sim_dt

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

        self.prev_joystick_action = torch.zeros(self.num_envs, 4, device=self.device) # user action output (simulate joystick) last step
        self.prev_actual_action = torch.zeros(self.num_envs, 4, device=self.device) # actual action taken by the policy last step

        # Buffer for pre-sampled trajectories
        # Max duration 4.0s (slightly larger than default 3.0 to be safe)
        self.max_traj_steps = int(4.0 / self.dt)
        self.action_trajectories = torch.zeros(self.num_envs, self.max_traj_steps, 4, device=self.device)
        self.traj_step_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # def _init_beam_angles(self):
    #     H = self.lidar_resolution[0]
    #     # evenly spread in [-pi, pi)
    #     self.beam_angles = torch.linspace(-torch.pi, torch.pi, steps=H+1, device=self.device)[:-1]

    def reset(self, pos, quat, env_ids, lidar_scan=None):
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
        self.human_style[env_ids] = {
            'conformance': torch.rand(K, 1, device=self.device),
            'aggressiveness': torch.rand(K, 1, device=self.device),
            'dexterity': torch.rand(K, 1, device=self.device)
        }
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
        self.prev_joystick_action[env_ids] = 0.0
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
            ids = torch.arange(env_ids, device=self.device)
        else:
            ids = env_ids
            
        K = ids.numel()
        trajs = []
        
        # Iterate over each environment to generate unique profile
        for i in range(K):
            env_idx = ids[i]
            
            # Get style for this env
            style = {k: v[env_idx] for k, v in self.human_style.items()}
            nl = self.noise_level[env_idx]
            
            # sample target action profile
            t_total, right_target, left_target, params = sample_joystick_profile(style, nl)

            # get action trajectory from joystick profile
            # actions: (steps, 4)
            actions = generate_action_from_stick_profile(
                dt=self.dt,
                t_total=t_total,
                right_target=right_target,
                left_target=left_target,
                params=params
            )
            
            # Pad or truncate to self.max_traj_steps
            L = actions.shape[0]
            if L < self.max_traj_steps:
                # Pad with zeros
                padding = torch.zeros(self.max_traj_steps - L, 4, device=actions.device)
                actions = torch.cat([actions, padding], dim=0)
            else:
                actions = actions[:self.max_traj_steps]
                
            trajs.append(actions)

        # Stack into (K, MaxSteps, 4)
        batch_trajs = torch.stack(trajs).to(self.device)
        
        # Store in the main buffer
        self.action_trajectories[ids] = batch_trajs
        self.traj_step_indices[ids] = 0 # Reset indices for these envs
        
        return batch_trajs

    def step(self, drone_state, lidar_scan, prev_agent_action):
        
        # Get drone state from environment
        N = self.num_envs
        drone_pos_w = drone_state[:, 0:3]  # (N,3)
        drone_vel_w = drone_state[:, 3:6]  # (N,3)
        drone_orientation_q = drone_state[:, 6:10]  # (N,4)

        # get previous actual actions from environment history
        self.prev_actual_action = prev_agent_action.detach()

        # 1. Check if need new intent trajectory
        # Simple Condition: a trajectory finished
        traj_finished = self.traj_step_indices >= self.max_traj_steps
        
        # Filter out drones that have reached current goal and need new intent
        need_new_intent = traj_finished
        if need_new_intent.any():
            # idx = if need_new_intent is True
            idx = need_new_intent.nonzero(as_tuple=False).squeeze(-1)
            
            # Resample trajectory for these envs
            new_traj = self._sample_intent_traj(env_ids=idx)
            
            # Update goals based on new trajectory
            # Use current orientation to integrate the new trajectory
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
        au_local_noisy = action_local

        # 3. Convert local action to world frame
        # Split linear and vertical components
        vels_u_b_noisy = au_local_noisy[:, 0:3]
        au_world_noisy = torch.cat([quat_rotate(drone_orientation_q, vels_u_b_noisy), au_local_noisy[:, 3:4]], dim=-1)

        # Update previous joystick action based on current noisy action
        self.prev_joystick_action = au_world_noisy.detach() # Store for next step
        
        return au_local_noisy, need_new_intent
    
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