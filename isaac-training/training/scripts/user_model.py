import torch
from omni_drones.envs.isaac_env import DebugDraw
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

# User Model to simulate human actions
# TODO: do not planning, and gengerate action based on a trajectory pre-sampling
class UserModel:
    def __init__(self, num_envs, cfg, lidar, lidar_resolution, debug_draw: DebugDraw=None):
        self.num_envs = num_envs
        self.cfg = cfg
        self.debug_draw = debug_draw

        # Init cfg parameters
        self.env_map_range = torch.tensor(cfg.env.map_range, dtype=torch.float32, device=cfg.device)  # half extents of the env map
        self.device = cfg.device
        self.lidar_range = cfg.sensor.lidar_range
        self.max_speed = cfg.algo.actor.action_limit  # max speed limit during training

        self.num_candidates = cfg.user_model.num_goal_candidates
        self.sample_candidates_range = cfg.user_model.sample_candidates_range
        self.fallback_goal_distance = cfg.user_model.fallback_goal_distance

        # Count min/max steps for intent duration
        intent_min, intent_max = self.cfg.user_model.intent_duration_range
        dt = cfg.sim.dt
        self.min_steps = int(intent_min / dt)
        self.max_steps = int(intent_max / dt)

        # Get RayCaster Lidar object from env
        self.lidar = lidar
        self.lidar_resolution = lidar_resolution  # (hbeams, vbeams)

        # sample random style parameters for the user model
        self.conformance = torch.rand(self.num_envs, 1, device=self.device)  # alpha
        self.aggressiveness = torch.rand(self.num_envs, 1, device=self.device)  # beta
        self.dexterity = torch.rand(self.num_envs, 1, device=self.device)  # gamma
        # TODO: more style params ...
        self.speed_delta = torch.rand(self.num_envs, 1, device=self.device)  # delta
        self.noise_level = 0.05 + 0.1 * torch.rand(self.num_envs, 1, device=self.device)  # noise level range [0.05, 0.15]

        # batched buffers
        self.intent_goals = torch.zeros(self.num_envs, 3, device=self.device)
        # self.intent_timers = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # height range limit: [minz, maxz] * envs
        self.height_range = torch.zeros(self.num_envs, 1, 2, device=self.device)

        self.prev_joystick_action = torch.zeros(self.num_envs, 4, device=self.device) # user action output (simulate joystick) last step
        self.prev_actual_action = torch.zeros(self.num_envs, 4, device=self.device) # actual action taken by the policy last step

    # def _init_beam_angles(self):
    #     H = self.lidar_resolution[0]
    #     # evenly spread in [-pi, pi)
    #     self.beam_angles = torch.linspace(-torch.pi, torch.pi, steps=H+1, device=self.device)[:-1]

    def reset(self, pos, quat, env_ids, lidar_scan=None):
        """
        Reset timers and batched buffers, resample new intent goals for the given env ids
        called whenever the environment resets.

        Args:
            pos: (K,1,3) or (K,3) positions of the reset envs in world frame.
            quat: (K,1,4) or (K,4) orientations of the reset envs in world frame.
            env_ids: (K,) long tensor, indices in [0, num_envs).
            lidar_scan: (K,1,H,V) lidar scan for the reset envs, or None.
        """
        # ------- 0. 规范形状，得到 (K,3)、(K,4) -------
        if pos.ndim == 3:
            pos_k = pos.squeeze(1)
        else:
            pos_k = pos

        if quat.ndim == 3:
            quat_k = quat.squeeze(1)
        else:
            quat_k = quat

        K = env_ids.numel()

        # 重置意图计时器
        # new_timers = torch.randint(
        #     self.min_steps,
        #     self.max_steps,
        #     (K,),
        #     device=self.device,
        # )
        # self.intent_timers[env_ids] = new_timers

        # 为这些 env 重新采样风格参数（可选）
        self.conformance[env_ids] = torch.rand(K, 1, device=self.device)
        self.aggressiveness[env_ids] = torch.rand(K, 1, device=self.device)
        self.dexterity[env_ids] = torch.rand(K, 1, device=self.device)
        self.speed_delta[env_ids] = torch.rand(K, 1, device=self.device)
        self.noise_level[env_ids] = 0.05 + 0.1 * torch.rand(K, 1, device=self.device)

        # 为这些 env 采样第一批 intent_goals
        if lidar_scan is not None:
            # lidar_scan 传进来是 (K,1,H,V)，直接用
            new_goals = self._sample_reachable_goals_vectorized(
                pos_k,
                quat_k,
                lidar_scan,
                light_of_sight_check=False, 
            )
        else:
            # 没有雷达信息，用一个退化的「在地图内随机采样」策略
            sx, sy, sz = (self.env_map_range * 1.2).tolist()
            rand = torch.rand(K, 3, device=self.device)
            new_goals = torch.empty_like(rand)
            new_goals[..., 0] = (rand[..., 0] * 2.0 - 1.0) * sx
            new_goals[..., 1] = (rand[..., 1] * 2.0 - 1.0) * sy
            new_goals[..., 2] = 0.5 + rand[..., 2] * max(sz - 0.5, 1e-3)

        self.intent_goals[env_ids] = new_goals
        # 计算height range
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], new_goals[:, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], new_goals[:, 2])

        # ------- 4. 重置动作缓存 -------
        self.prev_joystick_action[env_ids] = 0.0
        self.prev_actual_action[env_ids] = 0.0

    def _sample_reachable_goals_vectorized(self, pos, quat, lidar_scan, light_of_sight_check=False):
        """
        Sample reachable goal points around the drone position based on lidar scan
        Inputs:
            pos: (N,3) drone states in world frame
            quat: (N,4) drone orientations in world frame
            lidar_scan: (N,1,H,V) lidar scan data in lidar local frame
            light_of_sight_check: bool, whether to perform line-of-sight check, default False (quick check only)
            # N = number of envs that need resample
        Return: (N,3) sampled goal points in world frame
        """
        N = pos.shape[0]  # 使用传入函数的当前批次大小
        assert pos.shape[0] == quat.shape[0] == lidar_scan.shape[0], "Input batch size mismatch"
        H = self.lidar_resolution[0]
        V = self.lidar_resolution[1]

        # ----------------------------------------
        # Step 1: Sample candidate goals
        # ----------------------------------------
        sx, sy, sz = (self.env_map_range * 1.2).tolist()  # get env map size
        C = self.num_candidates  # number of candidates per environment
        min_r, max_r = self.sample_candidates_range

        # expand quat to match candidate dimension C
        quat_expanded = quat.unsqueeze(1).expand(-1, C, -1)  # (N,C,4)

        # get a set of random goal distances and directions
        r = torch.rand(N, C, device=self.device) * (max_r - min_r) + min_r
        dirs = torch.randn(N, C, 3, device=self.device)  
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # calculate goal positions (based on current pos)
        candidates = pos.unsqueeze(1) + dirs * r.unsqueeze(-1)

        # clamp candidate goal points within map bounds
        candidates[..., 0] = candidates[..., 0].clamp(-sx, sx)
        candidates[..., 1] = candidates[..., 1].clamp(-sy, sy)
        candidates[..., 2] = candidates[..., 2].clamp(0.5, sz)  # clamp height within [0.5, sz]

        # ----------------------------------------
        # Step 2: Vectorized beam mapping & single check
        # ----------------------------------------
        vec = candidates - pos.unsqueeze(1)  # vector from drone to candidate points, (N,C,3)
        dist = vec.norm(dim=-1)  # distance to candidate points, (N,C)
        vec_body = quat_rotate_inverse(quat_expanded, vec)  # transform to body frame, (N,C,3)

        # compute azimuth phi = atan2(y, x) in body frame; y left, x forward
        phis = torch.atan2(vec_body[..., 1], vec_body[..., 0])
        # map phi to lidar beam index
        idx = ((phis + torch.pi) / (2 * torch.pi) * H).long() % H  # idx = lidar beam index for each candidate, (N,C)
        env_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(N, C)  # env_idx = which env each candidate belongs to, (N,C)

        # quick point check: compare lidar distance vs candidate distance
        midv = V // 2  # use mid vertical beam
        scan_vals = lidar_scan[env_idx.reshape(-1), 0, idx.reshape(-1), midv].view(N, C)  # (N*C,) -> (N,C)
        
        obs_dist = self.lidar_range - scan_vals  # obstacle distances along those beams, (N,C)
        # candidate is free if obs_dists > dist + drone_radius (obstacle is farther than candidate)
        free_mask = obs_dist > (dist + 0.3)  # (N,C) boolean

        # ----------------------------------------
        # Step 3: line-of-sight checks (optional)
        # ----------------------------------------
        if light_of_sight_check:
            T = self.num_checks
            tvals = torch.linspace(0.0, 1.0, steps=T, device=self.device)

            inter = pos.unsqueeze(1).unsqueeze(2) + \
                    (candidates - pos.unsqueeze(1)).unsqueeze(2) * tvals.view(1, 1, -1, 1)

            inter_flat = inter.reshape(-1, 3)

            # flatten quat & pos for broadcasting
            quat_flat = quat.unsqueeze(1).unsqueeze(2).expand(-1, C, T, -1).reshape(-1, 4)
            pos_flat = pos.unsqueeze(1).unsqueeze(2).expand(-1, C, T, 3).reshape(-1, 3)

            # vector in world
            vec_inter = inter_flat - pos_flat
            dist_inter = vec_inter.norm(dim=-1)

            # convert to body frame
            vec_inter_body = quat_rotate_inverse(quat_flat.unsqueeze(1), vec_inter.unsqueeze(1)).squeeze(1)

            phi_inter = torch.atan2(vec_inter_body[:, 1], vec_inter_body[:, 0])
            idx_inter = ((phi_inter + torch.pi) / (2 * torch.pi) * H).long() % H

            env_idx = torch.arange(N, device=self.device).unsqueeze(1).unsqueeze(2).expand(N, C, T).reshape(-1)
            scan_inter_vals = lidar_scan[env_idx, 0, idx_inter, midv]
            obs_inter = self.lidar_range - scan_inter_vals

            inter_free = obs_inter > (dist_inter + 0.3)
            los_mask = inter_free.reshape(N, C, T).all(dim=-1)

            reachable_goal_mask = free_mask & los_mask
        else:
            reachable_goal_mask = free_mask

        # select first OK candidate or fallback
        good = reachable_goal_mask.any(dim=1)  # if there exist any good candidates, (N,) Boolean
        best_idx = reachable_goal_mask.float().argmax(dim=1)  # first good candidate index, (N,) int64
        arange_n = torch.arange(N, device=self.device)
        # # debug
        # print("Good goals:", good.shape, good.dtype)
        # print("Best idx:", best_idx.shape, best_idx.dtype)
        # print("arange_n:", arange_n.shape, arange_n.dtype)

        chosen = candidates[arange_n, best_idx]

        # print("chosen:", chosen.shape, chosen.dtype)
        # # try to print the content of chosen
        # print("chosen content:", chosen[:10])

        # fallback: short forward
        forward_body = torch.tensor([1.0, 0, 0], device=self.device).expand(N, 3)  # format a batch of forward vectors, (N,3)
        forward_w = quat_rotate(quat, forward_body)  # transform to world frame, (N,3)
        fallback = pos + forward_w * self.fallback_goal_distance  # get fallback goal points, (N,3)

        chosen = torch.where(good.unsqueeze(-1), chosen, fallback)
        return chosen

    def _potential_field_planner(self, pos, quat, lidar_scan):
        """
        Simple Potential Field based planner to compute desired velocity V_t
        for every drone of the env in the batch.
        Inputs:
            pos: (N,3,) drone position in world frame
            quat: (N,4,) drone orientation in world frame
            lidar_scan: (N,1,H,V) lidar scan data in lidar local frame
        Returns:
            vels_t: (N,3) desired velocity vector in world frame
        """
        N = self.num_envs
        H = lidar_scan.shape[2]
        mid = H // 2

        # 计算实际距离值 d_real from lidar scan
        # 检查前方的光束，垂直方向取中点，平均几个点以减少噪声
        d = lidar_scan[:,0, mid, 1:3].mean(dim=-1)  # (N,)
        d_real = self.lidar_range - d  # (N,)

        # 1. 吸引力 (Pull towards intent goal)
        attractive_forces = self.intent_goals - pos
        
        # 2. 排斥力 (Push from obstacles)
        #    将 Lidar (机体系) 转换为世界坐标系点云，然后计算排斥力
        repulsive_body = torch.zeros_like(attractive_forces)  # (N,3)
        near = d_real < (self.lidar_range - 0.5)  # 判断哪些无人机近障碍物
        # 对于近障碍物的无人机，计算排斥力
        repulsive_body[near, 0] = -1.0 * (self.lidar_range - d_real[near])
        # 将排斥力从机体系（local frame）转到世界系
        repulsive_forces = quat_rotate(quat, repulsive_body)
        
        # 3. 合成最终力 (世界系)
        # 参数： aggressiveness 规划能力：谨慎-激进（盲目） (β)
        #    beta=0 (激进): 完全忽略排斥力
        #    beta=1 (谨慎): 完全使用排斥力
        beta = self.aggressiveness
        total_forces = beta * attractive_forces + (1 - beta) * repulsive_forces

        # 4. 速度上限 
        # 参数：δ - speed_delta ∈ [0,1], 最大限速调节因子
        max_vel_limit = self.max_speed  # 配置的训练期间最大限制速度
        max_speed = self.speed_delta * max_vel_limit  # 计算最终的速度上限

        # 将势场计算的力归一化，转换为成比例的速度
        speed = torch.norm(total_forces, dim=-1, keepdim=True).clamp(min=1e-6) 
        vels_t = total_forces / speed * max_speed  # (N,3)
        return vels_t  # 为每个无人机返回期望速度向量，(N,3)


    def _traj_action_sampler(self):

        # dummy: just return a stright forward action
        N = self.num_envs
        au_local = torch.zeros(N, 4, device=self.device)
        au_local[:, 0] = self.max_speed * 0.5  # half max speed forward
        return au_local

    def step(self, drone_state, lidar_scan, prev_agent_action, visualize_env_idx=None):
        
        # 提取无人机状态
        N = self.num_envs
        drone_pos_w = drone_state[:, 0:3]  # (N,3)
        drone_vel_w = drone_state[:, 3:6]  # (N,3)
        drone_orientation_q = drone_state[:, 6:10]  # (N,4)

        # 提取上一步的用户动作和实际动作
        self.prev_actual_action = prev_agent_action.detach()

        # 1. 检查是否需要新意图
        # self.intent_timers -= 1
        dist_to_goal = torch.norm(drone_pos_w - self.intent_goals, dim=-1)  # (N,)
        goal_reached = dist_to_goal < 0.5  # 到达意图目标的条件
        
        # 筛选出已经到达当前目标，需要新意图的无人机
        # 测试：去除timer
        # need_new_intent = (self.intent_timers <= 0) | goal_reached
        need_new_intent = goal_reached
        if need_new_intent.any():
            # idx 是需要新意图的无人机索引
            idx = need_new_intent.nonzero(as_tuple=False).squeeze(-1)
            # 重新采样新目标点
            new_goals = self._sample_reachable_goals_vectorized(
                drone_pos_w[idx],
                drone_orientation_q[idx],
                lidar_scan[idx],
            )
            self.intent_goals[idx] = new_goals
            # 重置意图计时器
            # new_timers = torch.randint(
            #     self.min_steps, self.max_steps, (idx.numel(),), device=self.device
            # )
            # self.intent_timers[idx] = new_timers

        # 2. 规划器：计算期望速度 V_t (N, 3) (世界系)
        # vels_t_w = self._potential_field_planner(
        #     drone_pos_w,
        #     drone_orientation_q,
        #     lidar_scan
        # )
        action_local = self._traj_action_sampler()  # (N,4)

        # # 3. 将速度映射到控制（目前只映射线速度）
        # prev_j_vel_w = self.prev_joystick_action[:, 0:3]  # (N,3)
        # prev_a_vel_w = prev_agent_action[:, 0:3]  # (N,3)

        # # s_t = J_t-1 + (p_t - J_t-1) * Pgain
        # gamma = self.dexterity  
        # Pgain = 0.5 + gamma * 0.5  # Pgain ∝ gamma, 映射到 [0.5, 1.0] 之间
        # vels_smooth_w = prev_j_vel_w + (vels_t_w - prev_j_vel_w) * Pgain

        # # J_t = s_t + (J_t-1 - Aa_t-1)(1 - α)
        # alpha = self.conformance
        # action_diff = prev_j_vel_w - prev_a_vel_w
        # vels_u_w = vels_smooth_w + (action_diff) * (1.0 - alpha)
        
        # # (D) 添加抖动(模拟不精确的操作和控制信号噪声等)
        # noise = (torch.randn_like(vels_u_w) * self.noise_level)
        # vels_u_w_noisy = vels_u_w + noise

        # # TODO: 是否需要额外的积分平滑（类似PID控制器中的I项）
        # # self.It = self.It + (action_diff) * (1.0 - alpha)
        # # self.It = self.It * 0.95 # 积分衰减，防止无限累积
        # # Igain = 0.1 # 平滑参数，可调
        # # au_world = action_joystick_t + self.It * Igain
        
        # # --- 4. 转换回机体坐标系，以符合训练网络输入规范 ---
        # vels_u_b_noisy = quat_rotate_inverse(drone_orientation_q, vels_u_w_noisy)
        # # transfrorm to 4D action (keep yaw speed rate = 0)
        # au_world_noisy = torch.cat([vels_u_w_noisy, torch.zeros(N, 1, device=self.device)], dim=-1)
        # au_local_noisy = torch.cat([vels_u_b_noisy, torch.zeros(N, 1, device=self.device)], dim=-1)

        noise = (torch.randn_like(action_local) * self.noise_level)
        au_local_noisy = action_local + noise

        vels_u_b_noisy = au_local_noisy[:, 0:3]
        # 转换到世界系
        au_world_noisy = torch.cat([quat_rotate(drone_orientation_q, vels_u_b_noisy), au_local_noisy[:, 3:4]], dim=-1)

        # --- 5. 更新Joystick动作（保持世界系）并返回 ---
        self.prev_joystick_action = au_world_noisy.detach() # 存储下一步使用

        # 可视化选项
        if visualize_env_idx is not None and self.debug_draw is not None:
            self._visualize_single_env(
                env_idx=visualize_env_idx,
                pos=drone_pos_w[visualize_env_idx],
                goal=self.intent_goals[visualize_env_idx],
                action_b=au_local_noisy[visualize_env_idx]
            )
        
        return au_local_noisy, goal_reached
    
    def get_height_range(self):
        """
        Get the height range [minz, maxz] for each env based on current pos and intent goal
        Returns:
            height_range: (num_envs, 2) tensor, minz and maxz for each env
        """
        return self.height_range
    
    def _visualize_single_env(self, env_idx, pos, goal, action_b):
        """
        Visualize the drone position and intent goal in the environment using debug drawing
        Inputs: 
            env_idx: int, index of the environment to visualize
            pos: (3,) tensor, drone position in world frame
            goal: (3,) tensor, intent goal position in world frame
        """
        if self.debug_draw is None:
            return

        p0 = pos.reshape(1, 3)
        g = goal.reshape(1, 3)
        vel_b = action_b[0:3].reshape(1, 3)

        self.debug_draw.clear()
        g_top = g + torch.tensor([0, 0, 0.5], device=self.device) # 向上 0.5m
        line_points = torch.cat([g, g_top], dim=0) # (2, 3)
        
        # draw goal (red)
        self.debug_draw.plot(x=line_points, size=12.0, color=(1.0,0.0,0.0,1.0))

        # draw goal vector (green)
        self.debug_draw.vector(p0, (g - p0), size=2.0, color=(0.1,1.0,0.1,1.0))

        # draw current velocity vector (blue)
        self.debug_draw.vector(p0, vel_b, size=2.0, color=(0.1,0.1,1.0,1.0))