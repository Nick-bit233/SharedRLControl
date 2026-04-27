import torch
import einops
import numpy as np
from typing import Optional
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, Categorical
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import isaaclab.sim as sim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.robots.drone import MultirotorBase
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler, quat_axis, quat_rotate, quat_rotate_inverse
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view

from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh

from src.core.trainning_utils import vec_to_body, vec_to_world
from src.core.user_model_intent import UserModelIntent
from src.core.profiler import get_profiler
from src.datasets.trajectory_dataset import TrajectoryDataset

@height_field_to_mesh
def tunnel_obstacles_terrain(difficulty: float, cfg: HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """
        自定义地形mesh生成，覆盖HfDiscreteObstaclesTerrain默认高度场：
            先生成随机障碍物，再在四周建墙形成隧道/封闭区域
    """
    
    # 1. 获取底层官方生成的 NumPy 2D 高度矩阵
    # 使用 .__wrapped__ 可以绕过原函数的 @height_field_to_mesh 装饰器，直接拿到 ndarray
    hf_raw = hf_terrains.discrete_obstacles_terrain.__wrapped__(difficulty, cfg)
    
    # 2. 计算墙壁参数
    wall_thickness_meters = 1.0  # 墙壁厚度 1 米
    wall_height_meters = 10.0    # 墙壁高度 10 米
    wall_start_meters = 2.0      # 墙壁起始位置(从-x方向开始算起)
    clear_zone_meters = 4.0    # 出生点平台宽度
    
    # 将物理尺寸（米）转换为矩阵的像素步数
    wall_thickness_pixels = int(wall_thickness_meters / cfg.horizontal_scale)
    wall_height_steps = int(wall_height_meters / cfg.vertical_scale)
    wall_start_pixels = int(wall_start_meters / cfg.horizontal_scale)
    clear_zone_pixels = int(clear_zone_meters / cfg.horizontal_scale)
    
    # 3. 拦截并直接修改这个 NumPy 矩阵
    # 在底部的墙壁内侧清理出安全区域，范围：从底部墙壁内侧边缘，向内延伸 clear_zone_pixels 的距离
    hf_raw[0: wall_start_pixels + clear_zone_pixels, :] = 0

    # 底部建墙（顶部打开，形成隧道）
    hf_raw[wall_start_pixels: wall_start_pixels + wall_thickness_pixels, :] = wall_height_steps
    
    # 左侧和右侧建墙
    hf_raw[:, 0:wall_thickness_pixels] = wall_height_steps
    hf_raw[:, -wall_thickness_pixels:] = wall_height_steps
    
    # 4. 返回修改后的 NumPy 数组，装饰器会自动接管并把它渲染成带墙壁的物理网格
    return hf_raw


# 创建对应的配置类，绑定我们刚才写的自定义函数
@configclass
class HfTunnelObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = tunnel_obstacles_terrain


class EnvTunnelIntent(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg, trajectory_dataset: Optional[TrajectoryDataset] = None):
        print("[Navigation Environment]: Initializing Simple Shared Autonomy Env (No Obstacles)...")

        # Controller for the drone (Omnidrones Controller)
        self.controller = None  # will be set in _design_scene()
        
        # Store trajectory dataset (if has) for later use
        self._trajectory_dataset = trajectory_dataset

        # Train task configuration
        self.enable_yaw_control = cfg.get("enable_yaw_control", False)
        if self.enable_yaw_control:
            self.human_action_dim = 4  # (vel_b[3] + yaw_rate[1])
        else:
            self.human_action_dim = 3  # (vel_b[3]) - yaw_rate removed
        
        # Check if lidar is enabled
        self.enable_lidar = cfg.env.get("enable_lidar", True)

        self.randomize_max_episode_length = False

        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        # Env map params:
        self.map_range = cfg.env.map_range  # [x_range, y_range, z_range], half extents
        self.platform_width = cfg.env.platform_width  # square platform width at the center of the map
        self.intent_privileged_dim = UserModelIntent.PRIVILEGED_DIM

        super().__init__(cfg, cfg.headless)  # _design_scene() will be called here, so the self.drone instance is created there
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # Algo action params
        self.max_action_vel = cfg.algo.actor.action_limit

        # Reward Function Params
        reward_weights = cfg.env.get("reward_weights", {})
        self.w_intent = reward_weights.get("intent_match", 2.5)
        self.w_safety = reward_weights.get("safety", 4.0)
        self.w_intervention = reward_weights.get("minimal_intervention", 1.25)
        self.w_smooth = reward_weights.get("smoothness", 0.5)
        self.w_height = reward_weights.get("height", 10.0)
        self.w_survival = reward_weights.get("survival", 0.2)
        self.w_crash = reward_weights.get("crash", -10.0)
        self.intent_safe_dist = cfg.env.get("intent_safe_dist", 3.5)

        self.user_model = UserModelIntent(
            num_envs=self.num_envs,
            cfg=cfg,
        )
        self.seed = cfg.get("seed", 0)  # seed for evaluation mode
        
        # history action buffer for memory
        with torch.device(self.device):
            self.root_state = torch.zeros(self.num_envs, 1, 17)  # (num_envs, 1, state_dim), state_dim=17 for drone's root state in world frame
            self.start_pos = torch.zeros(self.num_envs, 3)
            # prev_drone_vel_w is used to compute acceleration-based penalty
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 3)
            # previous action taken by the agent (drone) (vel_b[3])
            # use this 3D velocity action in user model and GRU network.
            self.agent_action = torch.zeros(self.num_envs, 3)
            # [Added] Previous action command buffer for smoothness reward
            self.prev_action_command = torch.zeros(self.num_envs, 3)
            # [Added] Original agent action command buffer for debugging
            self.agent_action_original = torch.zeros(self.num_envs, 3)

            self.prev_human_action = torch.zeros(self.num_envs, 3)
            self.prev_intent_velocity = torch.zeros(self.num_envs, 3)
            self.prev_critic_privileged = torch.zeros(self.num_envs, self.intent_privileged_dim)
            self.prev_threat = torch.zeros(self.num_envs, 1)
            self.intent_complete_counts = torch.zeros(self.num_envs, 1)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            # Cumulative following error for early termination
            self.cumulative_error = torch.zeros(self.num_envs, 1)
            self.error_ema_alpha = 0.995  # Exponential moving average decay factor
            self.error_threshold_base = 1.0  # Base threshold (strict, for no-obstacle case)
            self.error_threshold_max = 10.0   # Max threshold (relaxed, when obstacles are very close)
            self.safety_margin = 1.5  # Distance within which we start relaxing the threshold        
        self.common_step_counter = 0
        
        # visualize options
        self.disable_visualization = False
        self.render_lidar = True
        
        # Visualization Trajectory Buffers
        self.viz_traj_human = []
        self.viz_traj_agent = []
        self.viz_human_pos = None

    def _design_scene(self):
        # Init DRONE and CONTROLLER here, default prim path: /World/envs/envs_0
        self.drone, self.controller = MultirotorBase.make(
            self.cfg.drone.model_name, self.cfg.drone.controller_name, self.device
        )
        drone_prim = self.drone.spawn(translations=[(-7.0, 0.0, 5.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Terrain generation, as static obstacles
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(24.0, 12.0), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                curriculum=False,
                difficulty_range=(0.0, 1.0),
                sub_terrains={
                    "obstacles": HfTunnelObstaclesTerrainCfg(
                        size=(1.0, 1.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="choice",
                        obstacle_width_range=tuple(self.cfg.env.get("obstacle_width_range", [0.4, 1.1])),
                        obstacle_height_range=tuple(self.cfg.env.get("obstacle_height_range", [8.0, 20.0])),
                        platform_width=0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        # LiDAR Intialization
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)
        if self.enable_lidar:
            ray_caster_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                # attach_yaw_only=True, # Deprecated, use ray_alignment="yaw"
                ray_alignment="yaw",
                pattern_cfg=patterns.BpearlPatternCfg(
                    horizontal_res=self.lidar_hres, # horizontal default is set to 10
                    vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
                ),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        else:
            self.lidar = None

        # No dynamic obstacles initialization
        if (self.cfg.env_dyn.num_obstacles == 0):
            return ["/World/ground"]

        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = 10  # (vel_b[3] + ang_vel_b[3] + orientation_q[4])

        # Observation Spec
        obs_dict = {
            "state": Unbounded((drone_state_dim,), device=self.device), 
            "human_action": Unbounded((self.human_action_dim,), device=self.device),
            "critic_privileged": Unbounded((self.intent_privileged_dim,), device=self.device),
        }
            
        if self.enable_lidar:
             obs_dict["lidar"] = Unbounded((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device)

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Composite(obs_dict),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = Composite({
            "agents": Composite({
                # "action": self.drone.action_spec, # number of motor
                "action": Unbounded((self.human_action_dim,), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = Composite({
            "done": Categorical(2, (1,), dtype=torch.bool),
            "terminated": Categorical(2, (1,), dtype=torch.bool),
            "truncated": Categorical(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "above_bound": Unbounded(1),
            "below_bound": Unbounded(1),
            "collision": Unbounded(1),
            "debug_vec_world": Unbounded(3),
            "debug_vec_policy": Unbounded(3),
            "debug_vec_target": Unbounded(3),
            "debug_pos_world": Unbounded(3),
            "diag_reward": Unbounded(1),
            "diag_reward_task": Unbounded(1),
            "diag_penalty_smooth": Unbounded(1),
            "diag_penalty_height": Unbounded(1),
            "terminated": Unbounded(1),
            "truncated": Unbounded(1),
            "success": Unbounded(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = Composite({
            "drone_state": Unbounded((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        # =================================
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()        

    def _reset_idx(self, env_ids: torch.Tensor):
        """reset drone to random position of the scene"""
        self.drone._reset_idx(env_ids, self.training)
        
        # Reset drone state (vel, rot and pos)
        sx, sy, sz = self.map_range  # half-extent in x, y, and z-max
        sx = sx * 0.8
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        if (self.training):  # get random start position
            # generate random start positions (within the platform area near the center of the map)

            pos[:, 0, 0] = -7.0  # y = -7
            pos[:, 0, 1] = torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * 2 * sx - sx  # x: random in [-sx, sx]

            # [ENV DEBUG] 初始高度固定在5.0m
            heights = 5.0 + torch.zeros(env_ids.size(0), dtype=torch.float, device=self.device)
            # heights = 2.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (5.0 - 2.5)
            pos[:, 0, 2] = heights  # pos z: 5.0
            # print(f"[Env Reset T] Randomized start positions for {len(env_ids.tolist())} envs ")
        else:
            # assign positions on the center of the map grid
            pos[:, 0, 0] = -7.0
            pos[:, 0, 1] = 0
            pos[:, 0, 2] = min(5.0, sz)
            # print(f"[Env Reset E] Assigned start positions for envs {env_ids.tolist()}")

        # Fix: Update start_pos with correct indexing (start_pos is (num_envs, 3))
        self.start_pos[env_ids] = pos[:, 0, :].clone()  # record start pos for debug
        
        # drone's facing direction default as towards positive x-axis, which is the forward direction of the tunnel
        rpy = torch.zeros(len(env_ids), 3, device=self.device)
        # rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * torch.pi - torch.pi

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        # Reset previous step variables
        # self.prev_drone_vel_w[env_ids] = 0.
        self.agent_action[env_ids] = 0.
        self.prev_action_command[env_ids] = 0.
        self.prev_human_action[env_ids] = 0.
        self.prev_intent_velocity[env_ids] = 0.
        self.prev_critic_privileged[env_ids] = 0.
        self.prev_threat[env_ids] = 0.
        # Reset cumulative following error
        self.cumulative_error[env_ids] = 0.

        mode_prior = torch.tensor(
            self.cfg.user_model.intent.tunnel_mode_prior,
            device=self.device,
            dtype=torch.float32,
        )
        if (self.training):
            self.user_model.reset(
                pos=pos,
                quat=rot,
                env_ids=env_ids,
                mode_prior_override=mode_prior,
            )
        else:
            # When evaluating, use fixed seed for reproducibility
            self.user_model.reset(
                pos=pos,
                quat=rot,
                env_ids=env_ids,
                seed=self.seed,
                mode_prior_override=mode_prior,
            )

        self.stats[env_ids] = 0.  

        # Randomize episode length for each env to break synchronization
        # This prevents all envs from truncating at the same time, which causes critic loss spikes
        self.max_episode_per_env = torch.full(
            (self.num_envs,), float(self.max_episode_length), 
            dtype=torch.float, device=self.device
        )

        # Set default height range for each env
        self.height_range[env_ids, 0, 0] = 0.4 * sz    # min height
        self.height_range[env_ids, 0, 1] = 1.6 * sz    # max height

        # Reset visualization buffers if env 0 is reset
        if 0 in env_ids:
            self.viz_traj_agent = []
            self.viz_traj_human = []
            # Find the index of env 0 in env_ids
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            self.viz_human_pos = pos[idx, 0].clone()

    def _pre_sim_step(self, tensordict: TensorDictBase):

        # Store last step action command for smoothness reward
        self.prev_action_command[:] = self.agent_action.clone()

        # Get new action command from policy (world frame)
        action_command = tensordict[("agents", "action")] 
        
        # Ensure actions shape is compatible
        if action_command.ndim == 3:
            action_command = action_command.squeeze(1)
        self.agent_action[:] = action_command.clone()
        # 添加一个探针来记录原始的agent action command，看看是否是action_command本身的z值导致了高度不稳定
        self.agent_action_original[:] = action_command.clone()

        # Retrieve drone state for controller input
        # state赋值确定与当前状态同步(√)
        drone_state = tensordict[("info", "drone_state")][..., :13]  # (num_envs, 1, 13)

        # 判断tensordict[("info", "drone_state")]是否与self.root_state同步
        if not torch.allclose(drone_state, self.root_state[..., :13], atol=1e-3):
            print(f"[Warning] Drone state in tensordict is not synchronized with self.root_state!")
            print(f"  - Drone state in tensordict: {drone_state.squeeze(1)[0].cpu().numpy()}")
            print(f"  - Drone state in self.root_state: {self.root_state[..., :13][0].cpu().numpy()}")

        # Apply action using the Omnidrones controller
        # Notice: Omnidrones requires input tensor be of shape (num_envs, M, ), where M=1 by default
        # need to squeeze the input tensors first
        if self.enable_yaw_control and action_command.shape[-1] == 4:
            target_vel = action_command[..., :3].unsqueeze(1)  # (num_envs, 1, 3)
            # yaw speed is scaled to [-pi, pi]
            target_yaw = action_command[..., 3:4].unsqueeze(1) * torch.pi  # (num_envs, 1, 1)
        else:
            target_vel = action_command[..., :3].unsqueeze(1)
            target_yaw = None
        # action_command(target vel) -> actions(thrusts)
        actions = self.controller(
            root_state=drone_state,
            target_vel=target_vel,
            target_yaw=target_yaw
        )
        # Check if nan happens in actions
        if torch.isnan(actions).any():
            print("[Warning] NaN detected in actions from controller!")
            actions = torch.nan_to_num(actions, nan=0.0)
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        profiler = get_profiler()
        with profiler.timer("env/_post_sim_step"):
            # Update LiDAR sensor
            if self.enable_lidar:
                self.lidar.update(self.dt)
        
        self.common_step_counter += 1
    
    # get current states/observation
    def _compute_state_and_obs(self):
        profiler = get_profiler()
        profiler.start("env/_compute_state_and_obs")
        
        self.root_state = self.drone.get_state(env_frame=False)  # get drone's root state in world frame
        # explaination of root state(17):  
        # (world_pos[3], orientation (quat)[4], world_vel_and_angular[3+3], heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        ray_dists = None
        ray_dirs_w = None
        if self.enable_lidar:
            ray_vecs_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
            raw_ray_dists = ray_vecs_w.norm(dim=-1)
            finite_ray_vecs = torch.isfinite(ray_vecs_w).all(dim=-1)
            finite_ray_dists = torch.isfinite(raw_ray_dists)
            hit_mask = (
                finite_ray_vecs
                & finite_ray_dists
                & (raw_ray_dists > 1e-6)
                & (raw_ray_dists < self.lidar_range)
            )
            ray_dists = torch.where(
                hit_mask,
                raw_ray_dists.clamp_max(self.lidar_range),
                torch.full_like(raw_ray_dists, self.lidar_range),
            )
            ray_dirs_w = torch.zeros_like(ray_vecs_w)
            ray_dirs_w[hit_mask] = ray_vecs_w[hit_mask] / raw_ray_dists[hit_mask].unsqueeze(-1)

            self.lidar_scan = (
                self.lidar_range
                - ray_dists.reshape(self.num_envs, 1, *self.lidar_resolution)
            ) # lidar scan store the data that is range - distance and it is in lidar's local frame
            # Normalize LiDAR to [0, 1] for better CNN training stability
            self.lidar_scan = self.lidar_scan / self.lidar_range
        else:
            self.lidar_scan = None

        # ---------Network Input II: Drone's internal states---------
        # (Changed: remove all tensors about target(goal) from internal states)

        # get drone's internal states with velocity and angular velocity in world frame
        drone_pos_w = self.root_state[..., :3].squeeze(1)   # (N, 3)
        drone_vel_w = self.root_state[..., 7:10].squeeze(1)     # (N, 3) world_vel
        drone_ang_vel_w = self.root_state[..., 10:13].squeeze(1) # (N, 3) world_angular
        drone_orientation_q = self.root_state[..., 3:7].squeeze(1) # (N, 4) orientation(quat)

        min_dist_to_obs = torch.full((self.num_envs, 1), self.lidar_range, device=self.device)
        nearest_obstacle_normal_b = torch.zeros(self.num_envs, 3, device=self.device)
        if self.enable_lidar:
            min_dist_to_obs, min_idx = ray_dists.min(dim=-1, keepdim=True)
            nearest_normal_w = -ray_dirs_w[
                torch.arange(self.num_envs, device=self.device),
                min_idx.squeeze(-1),
            ]
            nearest_obstacle_normal_b = quat_rotate_inverse(
                drone_orientation_q, nearest_normal_w
            )

        # calculate drone's velocity and angular velocity in body frame
        vel_b = quat_rotate_inverse(drone_orientation_q, drone_vel_w)
        ang_vel_b = quat_rotate_inverse(drone_orientation_q, drone_ang_vel_w)

        # use body frame velocities for better generalization
        # drone_state_b: (N, 10) -> [vel_b(3), ang_vel_b(3), orientation_q(4)]
        drone_state_b = torch.cat([vel_b, ang_vel_b, drone_orientation_q], dim=-1)

        # ---------Network Input III: Dynamic obstacle states (Removed)--------

        # ---------Network Input IV: Human control action--------
        user_input_drone_state = drone_state_b.clone()  # (N, 10)
        assistant_action_b = quat_rotate_inverse(drone_orientation_q, self.agent_action)

        human_actions_local = torch.zeros(self.num_envs, self.human_action_dim, device=self.device)  # (N, 3) or (N, 4)
        current_privileged = self.prev_critic_privileged.clone()
        current_intent_velocity_b = self.prev_intent_velocity.clone()
        current_threat = self.prev_threat.clone()
 
        if getattr(self, "manual_mode", False):
            # Use manual action input as human input
            human_actions_local = self.manual_action.clone()
        else:
            # Step the simulated user model to get human action input
            with profiler.timer("env/user_model_step"):
                human_actions_local, _ = self.user_model.step(
                    user_input_drone_state,
                    drone_pos_w,
                    assistant_action=assistant_action_b,
                    env_geom={
                        "nearest_obstacle_dist": min_dist_to_obs.squeeze(-1),
                        "nearest_obstacle_normal": nearest_obstacle_normal_b,
                    },
                )
            current_privileged = self.user_model.get_privileged_obs()
            user_debug = self.user_model.debug_state()
            current_intent_velocity_b = user_debug["intent_velocity_body"]
            current_threat = user_debug["threat"].unsqueeze(-1)
        
        # print("[EnvDebug] Human action (local frame) at Env0 step {}: {}".format(self.common_step_counter, human_actions_local[0].cpu().numpy()))

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state_b,
            "human_action": human_actions_local,
            "critic_privileged": current_privileged,
        }
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan
        # -----------------Reward Calculation-----------------
        profiler.start("env/reward_calculation")

        prev_human_action_b = self.prev_human_action.clone()
        prev_intent_velocity_b = self.prev_intent_velocity.clone()
        target_vel_w = vec_to_world(
            prev_human_action_b,
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True
        )
        intent_vel_w = vec_to_world(
            prev_intent_velocity_b,
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True
        )
        
        r_safety = torch.zeros(self.num_envs, 1, device=self.device)
        danger_margin = torch.ones(self.num_envs, 1, device=self.device)
        if self.enable_lidar:
            cone_threshold = 0.866
            cur_vel_norm = drone_vel_w.norm(dim=-1, keepdim=True)
            cur_vel_dir = drone_vel_w / (cur_vel_norm + 1e-6)
            cos_sim_vel = (ray_dirs_w * cur_vel_dir.unsqueeze(1)).sum(dim=-1)
            mask_vel = cos_sim_vel > cone_threshold
            dist_in_vel_cone = torch.where(mask_vel, ray_dists, torch.full_like(ray_dists, self.lidar_range))
            dist_to_cur_vel_dir, _ = dist_in_vel_cone.min(dim=-1, keepdim=True)
            is_moving = (cur_vel_norm > 0.1).float()
            dist_to_cur_vel_dir = is_moving * dist_to_cur_vel_dir + (1.0 - is_moving) * self.lidar_range

            target_vel_norm = target_vel_w.norm(dim=-1, keepdim=True)
            target_vel_dir = target_vel_w / (target_vel_norm + 1e-6)
            cos_sim_cmd = (ray_dirs_w * target_vel_dir.unsqueeze(1)).sum(dim=-1)
            mask_cmd = cos_sim_cmd > cone_threshold
            dist_in_cmd_cone = torch.where(mask_cmd, ray_dists, torch.full_like(ray_dists, self.lidar_range))
            dist_to_human_action_dir, _ = dist_in_cmd_cone.min(dim=-1, keepdim=True)
            has_cmd = (target_vel_norm > 0.1).float()
            dist_to_human_action_dir = has_cmd * dist_to_human_action_dir + (1.0 - has_cmd) * self.lidar_range

            r_safety_dist_scale = 1.0  # Wider gradient reach (was 0.5)
            p_min = torch.exp(-min_dist_to_obs / r_safety_dist_scale)
            p_vel = torch.exp(-dist_to_cur_vel_dir / r_safety_dist_scale)
            p_cmd = torch.exp(-dist_to_human_action_dir / r_safety_dist_scale)

            w_min = 0.2  # 基础安全意识
            w_vel = 0.4  # 物理惯性危险：撞墙风险
            w_cmd = 0.4  # 主动寻死危险：指令导致撞墙

            safe_zone = 4.0  # was 3.0
            mask_min = (min_dist_to_obs < safe_zone).float()
            mask_vel = (dist_to_cur_vel_dir < safe_zone).float()
            mask_cmd = (dist_to_human_action_dir < safe_zone).float()
            r_safety = - (
                w_min * (p_min * mask_min) + 
                w_vel * (p_vel * mask_vel) + 
                w_cmd * (p_cmd * mask_cmd)
            )
            danger_margin = (min_dist_to_obs / self.intent_safe_dist).clamp(0.0, 1.0)

        policy_vel_w = self.agent_action[:, :3]
        intent_error_sq = ((policy_vel_w - intent_vel_w) ** 2).sum(dim=-1, keepdim=True)
        intervention_error_sq = ((policy_vel_w - target_vel_w) ** 2).sum(dim=-1, keepdim=True)
        reward_intent = -intent_error_sq
        reward_intervention = -(intervention_error_sq * danger_margin)
        reward_task = reward_intent

        # d. Smoothness Penalty
        action_diff = (self.agent_action - self.prev_action_command).norm(dim=-1, keepdim=True)
        penalty_action_smoothness = (action_diff / self.max_action_vel) ** 2

        # Forward progress reward: incentivize moving along tunnel length axis (pos[0])
        # pos[0] starts at -7.0, increases toward +12.0
        # forward_vel = drone_vel_w[..., 0:1]  # velocity along tunnel length axis
        # reward_progress = 0.1 * forward_vel.clamp(min=0.0)  # only reward forward motion

        # height penalty reward for flying unnessarily high or low
        h_min, h_max = self.height_range[..., 0], self.height_range[..., 1]
        # penalty when z > h_max + 0.2 or z < h_min - 0.2
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        # z = self.drone.pos[..., 2]
        # penalty_height[z > (h_max + 0.2)] = ((z - h_max - 0.2)**2)[z > (h_max + 0.2)]
        # penalty_height[z < (h_min - 0.2)] = ((h_min - 0.2 - z)**2)[z < (h_min - 0.2)]
        z = self.drone.pos[..., 2:3].reshape(self.num_envs, 1)
        # Quadratic penalty for height violations (stronger gradient far from boundary)
        height_excess_up = (z - (h_max + 0.2)).clamp(min=0.0)
        height_excess_down = ((h_min - 0.2) - z).clamp(min=0.0)
        penalty_height = height_excess_up ** 2 + height_excess_down ** 2
        
        self.reward = (
            self.w_intent * reward_intent
            + self.w_safety * r_safety
            + self.w_intervention * reward_intervention
            + self.w_survival
            - self.w_height * penalty_height
            - self.w_smooth * penalty_action_smoothness
        )
        
        # Terminate Conditions & Terminal Penalty
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > self.map_range[2] * 2.0 + 1.0 
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (1.0 - 0.3 / self.lidar_range)
        collision = static_collision
        # success if drone traverses the tunnel length (pos[0] is the tunnel length axis)
        # Drone starts at pos[0] ≈ -7.0, tunnel ends at pos[0] ≈ +10.0
        success = self.drone.pos[..., 0] > 10.0
        
        self.terminated = below_bound | above_bound | collision
        timeout_truncate = (self.progress_buf >= self.max_episode_per_env).unsqueeze(-1)
        self.truncated = timeout_truncate | success

        # === Apply Terminal Reward & Penalty ===
        # If crashed (not timed out), give a massive penalty encoded into the reward of this step.
        # This ensures the Value Function learns to fear these states.
        crash_penalty = self.w_crash
        # Only apply to envs that just terminated due to crash (collision or out-of-bounds)
        crashed_mask = ((collision | above_bound | below_bound) & ~self.truncated)
        self.reward[crashed_mask] += crash_penalty

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_human_action = human_actions_local.clone()
        self.prev_intent_velocity = current_intent_velocity_b.clone()
        self.prev_critic_privileged = current_privileged.clone()
        self.prev_threat = current_threat.clone()
        self.prev_drone_vel_w = drone_vel_w.clone()

        # ----------------- Visualization and Debugging -----------------
        profiler.start("env/visualization")
        if self._should_render(0) and not self.disable_visualization:
            self.debug_draw.clear()
            viz_env_id = 0  # visualize only the first env
            VIZ_VEL_SCALE = 0.5  # scale for velocity vector visualization
            view_pos = drone_pos_w[viz_env_id]  # lidar/camera view position is same as drone position by default

            # set the camera to focus on the lidar position (which is the drone)
            camera_mode = getattr(self, '_camera_view_mode', 'follow')
            if camera_mode == 'global':
                set_camera_view(
                    eye=torch.tensor([-3.0, 0.0, 32.0]),  # global top-down view
                    target=torch.tensor([0.0, 0.0, 0.0])                        
                )
            else:
                # 'follow' mode: camera follows behind the drone in world velocity direction
                eye_vel_offset = -drone_vel_w[viz_env_id] * torch.tensor([2.0, 2.0, 0.0], device=self.device) 
                set_camera_view(
                    # use cfg viewer settings as offset
                    eye=view_pos.cpu() + torch.as_tensor(self.cfg.viewer.eye) + eye_vel_offset.cpu(),
                    target=view_pos.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
                )
            if self.render_lidar and self.enable_lidar:
                # rendering LiDAR rays (360 degrees horizontal)
                v = (self.lidar.data.ray_hits_w[viz_env_id] - view_pos).reshape(*self.lidar_resolution, 3)
                self.debug_draw.vector(view_pos.expand_as(v[:, 0]), v[:, 0], color=(1, 0, 1, 0.5), size=1.0)  # Purple for all rays
                self.debug_draw.vector(view_pos.expand_as(v[:, -1]), v[:, -1], color=(1, 0, 1, 0.5), size=1.0)

            # A. Draw drone velocity vector (red arrow)
            drone_vel_w_vec = drone_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=view_pos, 
                v=drone_vel_w_vec * VIZ_VEL_SCALE,
                color=(1, 0, 0, 1), # Red
                size=2.0
            )

            # B. Draw human desired velocity (yellow arrow)
            human_vel_w_vec = target_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=view_pos, 
                v=human_vel_w_vec * VIZ_VEL_SCALE, 
                color=(1, 1, 0, 1), # Yellow
                size=2.0
            )
            
            # C. Draw Trajectories
            # Update trajectory buffers
            # Agent pos
            curr_pos = view_pos.clone()
            self.viz_traj_agent.append(curr_pos)
            
            # Human pos (integrated)
            if self.viz_human_pos is None:
                 self.viz_human_pos = curr_pos.clone()
            
            # Integrate human velocity
            human_vel = human_vel_w_vec
            self.viz_human_pos += human_vel * self.dt
            self.viz_traj_human.append(self.viz_human_pos.clone())
            
            # Limit trajectory length to avoid memory issues (e.g. last 500 points)
            max_traj_len = 1000
            if len(self.viz_traj_agent) > max_traj_len:
                self.viz_traj_agent.pop(0)
                self.viz_traj_human.pop(0)

            # Draw lines
            if len(self.viz_traj_agent) > 1:
                # Draw agent path (Blue)
                points = torch.stack(self.viz_traj_agent)
                starts = points[:-1]
                ends = points[1:]
                self.debug_draw.vector(starts, ends - starts, color=(0,0,1,1), size=2.0)
                
                # Draw human path (Green)
                points_h = torch.stack(self.viz_traj_human)
                starts_h = points_h[:-1]
                ends_h = points_h[1:]
                self.debug_draw.vector(starts_h, ends_h - starts_h, color=(0,1,0,1), size=2.0)
        profiler.stop("env/visualization")

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["above_bound"] = above_bound.float()
        self.stats["below_bound"] = below_bound.float()
    
        # === DIAGNOSTIC: Log internal reward components to stats ===
        self.stats["diag_reward"] = self.reward
        self.stats["diag_reward_task"] = reward_task
        self.stats["diag_penalty_smooth"] = penalty_action_smoothness
        self.stats["diag_penalty_height"] = penalty_height
        
        self.stats["terminated"] = self.terminated.float()
        self.stats["collision"] = collision.float()
        self.stats["success"] = success.float()
        self.stats["truncated"] = self.truncated.float()

        self.stats["debug_vec_world"] = drone_vel_w
        self.stats["debug_vec_policy"] = self.agent_action_original
        # Convert human action from body frame to world frame for consistent comparison
        human_action_w = vec_to_world(
            current_intent_velocity_b[:, :3],
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True
        )
        self.stats["debug_vec_target"] = human_action_w
        self.stats["debug_pos_world"] = drone_pos_w
        # [Debug Print] Check if drone is falling despite 0 command
        # if self.common_step_counter % 10 == 0:
        #      print(f"[Env Debug Step {self.common_step_counter}] Mean Z: {drone_pos_w[..., 2].mean():.2f} | Agent action Vz: {self.agent_action[..., 2].mean():.4f} | Human action local Vz: {human_actions_local[..., 2].mean():.4f} | Mean target world Vz (ideal): {target_vel_w[..., 2].mean():.4f} | Mean world Vz (actual): {drone_vel_w[..., 2].mean():.4f}")
        
        profiler.stop("env/_compute_state_and_obs")

        # === Probe: Check for NaNs in Observation and Reward ===
        if torch.isnan(self.reward).any():
            print("[Env Probe] NaN detected in Reward!")
            print("Reward components:")
            print(f"  intent: {reward_task[torch.isnan(self.reward)]}")
            raise ValueError("NaN in reward")

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def set_visualization(self, enabled: bool):
        """
        Enable or disable visualization.
        Use this to temporarily enable visualization during evaluation.
        :param enabled: bool - True to enable visualization, False to disable
        """
        self.disable_visualization = not enabled
        if enabled:
            print("[Env] Visualization ENABLED")
        else:
            print("[Env] Visualization DISABLED")

    def set_manual_mode(self, enabled: bool):
        """
        If manual mode is enabled, the environment will use outside manual_action as the human input action.
        :param enabled: bool
        """
        self.manual_mode = enabled
        self.manual_action = torch.zeros(self.num_envs, 3, device=self.device)

    def set_manual_action(self, action: torch.Tensor):
        if self.manual_mode:
            self.manual_action[:] = action

    def set_camera_view_mode(self, mode: str):
        """
        Set the camera view mode for rendering.
        :param mode: 'global' for top-down global view, 'follow' for drone-following view
        """
        if mode not in ['global', 'follow']:
            raise ValueError(f"Invalid camera mode: {mode}. Must be 'global' or 'follow'")
        self._camera_view_mode = mode
        print(f"[FollowingEnvSimple] Camera view mode set to: {mode}")
    
    def get_camera_view_mode(self) -> str:
        """
        Get the current camera view mode.
        :return: 'global' or 'follow'
        """
        return getattr(self, '_camera_view_mode', 'follow')

    def set_envs_visibility(self, visible_env_ids=None):
        """
        Show only specified environment instances and hide the rest.
        Useful for recording clean evaluation videos with a single drone.
        Physics simulation is NOT affected — only rendering visibility changes.

        Args:
            visible_env_ids: list/set of env indices to keep visible.
                             If None, all envs are made visible (restore).
        """
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        for i in range(self.num_envs):
            prim_path = f"/World/envs/env_{i}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible_env_ids is None or i in visible_env_ids:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
