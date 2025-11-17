from omni_drones.utils import torch
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate, quat_rotate_inverse

# User Model to simulate human actions
# TODO: finish it by sampling from different style params and distributions
class UserModel:
    def __init__(self, id, cfg, lidar, lidar_resolution):
        self.id = id  # marks the env index of this user model
        self.cfg = cfg

        # Init cfg parameters
        self.env_map_range = cfg.env.map_range
        self.device = cfg.device
        self.lidar_range = cfg.sensor.lidar_range

        # Count min/max steps for intent duration
        self.min_steps, self.max_steps = self.cfg.user_model.intent_duration_range / cfg.sim.dt
        self.min_steps = int(self.min_steps)
        self.max_steps = int(self.max_steps)

        # Get RayCaster Lidar object from env

        self.lidar = lidar
        self.lidar_resolution = lidar_resolution  # (hbeams, vbeams)

        # sample random style parameters for the user model
        self.style_params = {
            "conformance": torch.rand(1).item(),
            "aggressiveness": torch.rand(1).item(),
            "dexterity": torch.rand(1).item(),
            "max_speed": torch.rand(1).item(),
        }
        
        self._sample_new_intent_goal() 
        self.intent_timer = 0

        self.prev_user_action = torch.zeros(4, device=self.device) # user action input to the policy last step
        self.prev_actual_action = torch.zeros(4, device=self.device) # actual action taken by the policy last step

    def _sample_new_intent_goal(self):
        # 在地图边界内随机采样一个新目标点 G_hat # TODO：修改为在无人机当前可达范围内采样
        sx, sy, sz = self.env_map_range * 1.6  # env_map_range is half extents, extentd it a bit 
        self.G_hat = (torch.rand(3, device=self.device) - 0.5) * 2.0
        self.G_hat[0] *= sx
        self.G_hat[1] *= sy
        self.G_hat[2] = min(2.0, torch.rand(1).item() * sz)

        # TODO: 保证采样点不在（静态）障碍物内 (使用lidar点云检查)
        
        # 重置意图持续时间
        self.intent_timer = torch.randint(self.min_steps, self.max_steps, (1,)).item()

    def _potential_field_planner(self, drone_pos_w, lidar_scan, drone_orientation_q):
        # 1. 吸引力 (Pull towards G_hat)
        attractive_force = self.G_hat - drone_pos_w
        
        # 2. 排斥力 (Push from obstacles)
        #    将 Lidar (机体系) 转换为世界坐标系点云，然后计算排斥力
        #    TODO：(复杂版本的实现) ...
        
        #    (这是一个简化的实现)
        #    直接在Lidar数据上操作 (机体系)
        #    如果"前方" (Lidar中心) 有障碍物，施加一个"向后"的力
        repulsive_force_b = torch.zeros(3, device=self.device)
        scan_dist = lidar_scan[0, 18, 1:3].mean() # 检查正前方光束
        if scan_dist < (self.lidar_range - 0.5): # 快撞到了
            # 施加一个与距离成反比的力
            repulsive_force_b[0] = -1.0 * (self.lidar_range - scan_dist)
        # TODO: 将排斥力从机体转到世界系
        repulsive_force_b = quat_rotate(drone_orientation_q, repulsive_force_b)
        
        # 3. “盲目/熟练度 dexterity” (β)
        #    beta=0 (盲目): 完全忽略排斥力
        #    beta=1 (熟练): 完全使用排斥力
        #    (需要将排斥力从机体转到世界系，或者将吸引力从世界转到机体系)
        #    (为了简单，我们假设吸引力占主导)
        
        # 合成力 (世界系)
        beta = self.style_params["dexterity"]
        total_force = beta * attractive_force + (1 - beta) * repulsive_force_b

        # 3. 速度上限 (δ - Daring)
        max_vel_limit = self.style_params["max_speed"]
        delta = self.style_params["conformance"]  # 与规划器的一致性，越大越快
        max_speed = delta * max_vel_limit

        # 归一化力，并乘以最大速度
        V_t = (total_force / total_force.norm().clamp(min=1e-6)) * max_speed
        
        return V_t # (B, 3) 世界系下的"期望速度"

    def step(self, drone_state, lidar_scan, prev_agent_action):
        
        # 提取无人机状态 drone_state is (pos_w[3], vel_b[3], orientation_q[4])
        drone_pos_w = drone_state[0:3]
        drone_vel_w = drone_state[3:6]
        drone_orientation_q = drone_state[6:10]

        # 提取上一步的用户动作和实际动作
        self.prev_actual_action = prev_agent_action.detach()

        # 1. 检查是否需要新意图
        self.intent_timer -= 1
        dist_to_goal = (drone_pos_w - self.G_hat).norm()
        goal_reached = dist_to_goal < 0.5 # 到达意图目标的条件
        if self.intent_timer <= 0 or goal_reached:
            self._sample_new_intent_goal()
            
        # 2. 规划器：计算期望速度 V_t (世界系)
        V_t_world = self._potential_field_planner(drone_pos_w, lidar_scan, drone_orientation_q)
    
        # 3. 速度映射控制
        # (A) 转换 V_t 到 4D (vel_w[3], yaw_rate_w[1])
        # (我们的简单规划器只输出了vel_w[3]，假设yaw_rate=0)
        V_t = torch.cat([V_t_world, torch.zeros(1, device=self.device)])
        
        # (B) Joystick Control (P-Controller)
        # Pgain ∝ gamma (Aggressiveness)
        gamma = self.style_params["aggressiveness"]
        Pgain = 0.1 + gamma * 0.4 # 映射到 [0.1, 0.5]
        self.Jt = self.Jt + (V_t - self.Jt) * Pgain

        # (C) Adaptability Control (I-Controller)， 为了更加平滑规划动作和上一个用户动作
        # It+1 = It + (au - aa)(1 - α)
        action_diff = self.prev_user_action - prev_agent_action
        self.It = self.It + (action_diff) * (1.0 - self.alpha)
        self.It = self.It * 0.95 # 积分衰减，防止无限累积

        # (D) Final Action
        Igain = 0.1 # 平滑参数，可调
        au_world = self.Jt + self.It * Igain
        
        # (E) 添加抖动
        noise = (torch.randn_like(au_world) * self.noise_level)
        au_world_noisy = au_world + noise
        
        # --- 4. 转换回机体坐标系 ---
        # (因为我们的PPO-RNN期望的输入是机体系)
        au_local_noisy = quat_rotate_inverse(drone_orientation_q, au_world_noisy)
    
        # --- 5. 更新历史动作（保持世界系）并返回 ---
        self.prev_user_action = au_world_noisy.detach() # 存储下一步使用
        
        return au_local_noisy, goal_reached