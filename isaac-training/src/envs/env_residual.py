import csv
from sympy import N
import torch
import einops
import numpy as np
from typing import Optional
from tensordict.tensordict import TensorDict, TensorDictBase
# === Fix: Update deprecated imports ===
from torchrl.data import Unbounded, Composite, Categorical
# from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
# ======================================
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
# [Update import name rules]:
# - change all "omni.isaac.orbit" to name "isaaclab"
# - change all "omni.isaac.core" to name "isaacsim.core"
import isaaclab.sim as sim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.robots.drone import MultirotorBase
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate, quat_rotate_inverse
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
import time

from src.core.trainning_utils import vec_to_body, vec_to_world
from src.core.user_model import UserModel
from src.core.profiler import get_profiler
from src.datasets.trajectory_dataset import TrajectoryDataset

class FollowingEnvResidual(IsaacEnv):

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

        super().__init__(cfg, cfg.headless)  # _design_scene() will be called here, so the self.drone instance is created there
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # Algo action params
        self.max_action_vel = cfg.algo.actor.action_limit

        # User Model Initialization
        # Check for offline mode configuration
        offline_mode = cfg.user_model.get("offline_mode", False)
        sampling_mode = cfg.user_model.get("sampling_mode", "scaled")
        
        self.user_model = UserModel(
            num_envs=self.num_envs,
            cfg=cfg,
            offline_mode=offline_mode,
            dataset=self._trajectory_dataset,
            sampling_mode=sampling_mode,
        ) 
        self.seed = cfg.get("seed", 0)  # seed for evaluation mode
        
        # history action buffer for memory
        with torch.device(self.device):
            self.start_pos = torch.zeros(self.num_envs, 3)
            # prev_drone_vel_w is used to compute acceleration-based penalty
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 3)
            # previous action taken by the agent (drone) (vel_b[3])
            # use this 3D velocity action in user model and GRU network.
            self.agent_action = torch.zeros(self.num_envs, 3)
            # [Added] Previous action command buffer for smoothness reward
            self.prev_action_command = torch.zeros(self.num_envs, 3)
            
            self.prev_human_action = torch.zeros(self.num_envs, 3)
            self.intent_complete_counts = torch.zeros(self.num_envs, 1)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            # Cumulative following error for early termination
            self.cumulative_error = torch.zeros(self.num_envs, 1)
            self.error_ema_alpha = 0.995  # Exponential moving average decay factor
            self.error_threshold_base = 1.0  # Base threshold (strict, for no-obstacle case)
            self.error_threshold_max = 10.0   # Max threshold (relaxed, when obstacles are very close)
            self.safety_margin = 1.5  # Distance within which we start relaxing the threshold

        # visualize options
        self.disable_visualization = True
        self.render_lidar = False
        
        # Visualization Trajectory Buffers
        self.viz_traj_human = []
        self.viz_traj_agent = []
        self.viz_human_pos = None

    def _design_scene(self):
        # Init DRONE and CONTROLLER here, default prim path: /World/envs/envs_0
        self.drone, self.controller = MultirotorBase.make(
            self.cfg.drone.model_name, self.cfg.drone.controller_name, self.device
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 4.0)], device=self.device)[0]

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
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=(4.0, 10.0), # TODO: pramaize it
                        platform_width=self.platform_width,
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
                "action": self.drone.action_spec, # number of motor
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
            "terminated": Unbounded(1),
            "truncated": Unbounded(1),
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
        if (self.training):  # get random start position
            s_radius = self.platform_width / 2 

            # generate random start positions (within the platform area near the center of the map)
            pos = s_radius * (torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) - 0.5)
            heights = 2.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (5.0 - 2.5)
            pos[:, 0, 2] = heights  # pos z: range from 2.5 to 5.0
        else:
            # assign positions on the center of the map grid
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = 0
            pos[:, 0, 1] = 0
            pos[:, 0, 2] = min(2.5, sz)

        # Fix: Update start_pos with correct indexing (start_pos is (num_envs, 3))
        self.start_pos[env_ids] = pos[:, 0, :].clone()  # record start pos for debug
        
        # (randomlize the drone's facing direction because there's no goal to face now)
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * torch.pi - torch.pi

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        # Reset previous step variables
        # self.prev_drone_vel_w[env_ids] = 0.
        self.agent_action[env_ids] = 0.
        self.prev_action_command[env_ids] = 0.
        self.prev_human_action[env_ids] = 0.
        # Reset cumulative following error
        self.cumulative_error[env_ids] = 0.

        if (self.training):
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids)
        else:
            # When evaluating, use fixed seed for reproducibility
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids, seed=self.seed)

        self.stats[env_ids] = 0.  

        # Randomize episode length for each env to break synchronization
        # This prevents all envs from truncating at the same time, which causes critic loss spikes
        self.max_episode_per_env = torch.full(
            (self.num_envs,), float(self.max_episode_length), 
            dtype=torch.float, device=self.device
        )
        if hasattr(self, 'randomize_max_episode_length') and self.randomize_max_episode_length:
            # Randomize episode length for each env to break synchronization
            # This prevents all envs from truncating at the same time, which causes critic loss spikes
            # Random scale: 70% ~ 100% of max_episode_length
            random_scale = 0.7 + 0.3 * torch.rand(len(env_ids), device=self.device)
            self.max_episode_per_env[env_ids] = (self.max_episode_length * random_scale).floor()

        # Set default height range for each env
        self.height_range[env_ids, 0, 0] = 1.0  # min height
        self.height_range[env_ids, 0, 1] = 2 * sz    # max height

        # Reset visualization buffers if env 0 is reset
        if 0 in env_ids:
            self.viz_traj_agent = []
            self.viz_traj_human = []
            # Find the index of env 0 in env_ids
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            self.viz_human_pos = pos[idx, 0].clone()

    # === Override _step to add profiling for sim.step() without modifying isaac_env.py ===
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Override parent _step to add profiling for Isaac Sim physics step.
        This avoids modifying the third-party isaac_env.py dependency.
        """
        profiler = get_profiler()
        
        for substep in range(self.substeps):
            with profiler.timer("env/_pre_sim_step"):
                self._pre_sim_step(tensordict)
            with profiler.timer("env/sim_step"):
                self.sim.step(self._should_render(substep))

        self._post_sim_step(tensordict)
        self.progress_buf += 1

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.update(self._compute_reward_and_done())
        return tensordict

    def _pre_sim_step(self, tensordict: TensorDictBase):

        # Store last step action command for smoothness reward
        self.prev_action_command[:] = self.agent_action.clone()

        # Get new action command from policy
        action_command = tensordict[("agents", "action")]
        # Ensure actions shape is compatible with drone.shape (num_envs, 1, 3)
        if action_command.ndim == 2:
            action_command = action_command.unsqueeze(1)  # (num_envs, 3) -> (num_envs, 1, 3)
        self.agent_action[:] = action_command.clone()

        if self.enable_yaw_control:
            target_vel = action_command[..., :3]
            target_yaw = action_command[..., 3:4]
        else:
            target_vel = action_command[..., :3]
            target_yaw = None

        # Retrieve drone state for controller input
        # TODO: 调整这里的state赋值，确定与当前状态同步
        drone_state = tensordict[("info", "drone_state")][..., :13]
        
        # Apply action using the Omnidrones controller
        # action_command(target vel) -> actions(thrusts)
        actions = self.controller(
            root_state=drone_state,
            target_vel=target_vel,
            target_yaw=target_yaw
        )
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        profiler = get_profiler()
        with profiler.timer("env/_post_sim_step"):
            # Update LiDAR sensor
            if self.enable_lidar:
                self.lidar.update(self.dt)
    
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
        if self.enable_lidar:
            self.lidar_scan = self.lidar_range - (
                (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
                .norm(dim=-1)
                .clamp_max(self.lidar_range)
                .reshape(self.num_envs, 1, *self.lidar_resolution)
            ) # lidar scan store the data that is range - distance and it is in lidar's local frame
        else:
            self.lidar_scan = None

        # ---------Network Input II: Drone's internal states---------
        # (Changed: remove all tensors about target(goal) from internal states)

        # get drone's internal states with velocity and angular velocity in world frame
        drone_pos_w = self.root_state[..., :3].squeeze(1)   # (N, 3)
        drone_vel_w = self.root_state[..., 7:10].squeeze(1)     # (N, 3) world_vel
        drone_ang_vel_w = self.root_state[..., 10:13].squeeze(1) # (N, 3) world_angular
        drone_orientation_q = self.root_state[..., 3:7].squeeze(1) # (N, 4) orientation(quat)

        # calculate drone's velocity and angular velocity in body frame
        vel_b = quat_rotate_inverse(drone_orientation_q, drone_vel_w)
        ang_vel_b = quat_rotate_inverse(drone_orientation_q, drone_ang_vel_w)

        # use body frame velocities for better generalization
        # drone_state_b: (N, 10) -> [vel_b(3), ang_vel_b(3), orientation_q(4)]
        drone_state_b = torch.cat([vel_b, ang_vel_b, drone_orientation_q], dim=-1)

        # ---------Network Input III: Dynamic obstacle states (Removed)--------

        # ---------Network Input IV: Human control action--------
        user_input_drone_state = drone_state_b.clone()  # (N, 10)

        human_actions_local = torch.zeros(self.num_envs, self.human_action_dim, device=self.device)  # (N, 3) or (N, 4)
 
        if getattr(self, "manual_mode", False):
            # Use manual action input as human input
            human_actions_local = self.manual_action.clone()
        else:
            # Step the simulated user model to get human action input
            with profiler.timer("env/user_model_step"):
                human_actions_local, need_refill = self.user_model.step(
                    user_input_drone_state,
                    drone_pos_w
                )

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state_b,
            "human_action": human_actions_local,
        }
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan
        # -----------------Reward Calculation-----------------
        profiler.start("env/reward_calculation")
        
        # a. safety reward for static obstacles
        if self.enable_lidar:
            # lidar_scan: (N, 1, H, W) 或 (N, 1, Rays)
            # 找到最危险的射线（即 scan 值最大的点）
            # max_proximity_val 代表探测范围内离障碍物最近点的 "lidar_scan值"
            max_proximity_val, _ = self.lidar_scan.reshape(self.num_envs, -1).max(dim=-1, keepdim=True)
            
            # 将其还原为物理距离 (Physical Distance to closest obstacle)
            # dist = range - scan
            min_dist_to_obs = self.lidar_range - max_proximity_val

            # --------- (Positive Reward) ---------
            # 与最近障碍物的距离的对数奖励（越远离障碍物奖励越高）
            reward_safety_static = torch.log((min_dist_to_obs).clamp(min=1e-6, max=self.lidar_range))
            # [原始函数] 与所有障碍物距离的对数奖励平均值
            # reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))
            
            # --------- (Negative Penalty) ---------    
            # # 设定参数 # TODO: parametrize these value later
            # k_collision = 1.0
            # sigma = 0.4
            # safe_distance = 0.8
            
            # # 指数惩罚 (Bounded Exponential Penalty): P = k * exp(-d / sigma)
            # # 计算 safe_dist 处的指数基准值 (常数)
            # cutoff_val = torch.exp(torch.tensor(-safe_distance / sigma))
            # # 减去基准值，使得在 safe_dist 处刚好为 0
            # # 使用 clamp 确保不会出现正数奖励
            # dist_penalty = k_collision * (torch.exp(-min_dist_to_obs / sigma) - cutoff_val).clamp(min=0.0)

            # mask_safe = (min_dist_to_obs < safe_distance).float()  # (N, 1), within safe distance mask
            # static_safety_penalty = dist_penalty * mask_safe  # only penalize when within safe distance
            # reward_safety_static = 0.0

        # b. safety reward for dynamic obstacles
        # if (self.cfg.env_dyn.num_obstacles != 0):
        #     reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)
        # else:
        #     reward_safety_dynamic = 0.0

        # c. Velocity following reward (Positive)
        # 速度方向奖励和速度大小奖励分开加权
        
        # 目标速度为上一个step的 human action (in world frame)
        prev_human_action_b = self.prev_human_action.clone()
        target_vel_w = quat_rotate(drone_orientation_q, prev_human_action_b)

        # c1. 方向奖励 (Alignment)
        # 计算余弦相似度
        cosine_sim = torch.cosine_similarity(target_vel_w, drone_vel_w, dim=-1).unsqueeze(-1)
        # 映射到 [0, 1]
        reward_direction = (cosine_sim + 1.0) / 2.0 

        # c2. 速度大小奖励 (Magnitude)
        vel_error = (target_vel_w - drone_vel_w).norm(dim=-1, keepdim=True)
        # 使用 exp(-k * error) 形式，保证范围 [0, 1]
        reward_speed_match = torch.exp(-2.0 * vel_error) 

        # 综合任务奖励
        reward_task = 0.5 * reward_direction + 1.0 * reward_speed_match + 1.0 * reward_safety_static

        # d. Smoothness and Effort Penalty

        # d1. Penalize the difference between consecutive action commands
        # high frequency change in command -> huge penalty
        # p_smooth = || a_t - a_(t-1) ||^2 / max_action_vel^2
        action_diff = (self.agent_action - self.prev_action_command).norm(dim=-1, keepdim=True)
        penalty_action_smoothness = (action_diff / self.max_action_vel) ** 2

        # d2. smoothness reward for effort smoothness
        #  Agent 应该倾向于选择容易执行的动作（即与当前速度矢量夹角小的动作）
        # p_effort = || a_t - v_t || / max_action_vel
        action_change_cost = (self.agent_action - drone_vel_w).norm(dim=-1, keepdim=True)
        penalty_effort = action_change_cost / self.max_action_vel

        # d3. Z-axis tracking Penalty (for Spiral Up behavior)
        # Penalize vertical velocity error more heavily to discourage spiraling up
        # We separate Z component from velocity matching to give it higher weight
        target_vel_z = target_vel_w[..., 2:3]
        current_vel_z = drone_vel_w[..., 2:3]
        penalty_z_tracking = (target_vel_z - current_vel_z).abs() / self.max_action_vel

        # e. height penalty reward for flying unnessarily high or low
        h_min, h_max = self.height_range[..., 0], self.height_range[..., 1]
        # penalty when z > h_max + 0.2 or z < h_min - 0.2
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        # z = self.drone.pos[..., 2]
        # penalty_height[z > (h_max + 0.2)] = ((z - h_max - 0.2)**2)[z > (h_max + 0.2)]
        # penalty_height[z < (h_min - 0.2)] = ((h_min - 0.2 - z)**2)[z < (h_min - 0.2)]
        z = self.drone.pos[..., 2:3].reshape(self.num_envs, 1)
        penalty_height = (z - (h_max + 0.2)).clamp(min=0.0) + ((h_min - 0.2) - z).clamp(min=0.0)
        
        # f. Survival reward (keep flying as long as possible)
        # 因为碰撞由安全奖励变为危险惩罚，需要额外的存活奖励来鼓励继续飞行
        reward_survival = 1.0
        
        if not self.enable_lidar:
            self.reward = 2.0 * reward_task - 0.2 * penalty_action_smoothness - 4.0 * penalty_height
        else:
            # Full reward calculation
            self.reward = (
                1.0 * reward_task
                + 0.1 * reward_survival
                # - 1.0 * static_safety_penalty  # disabled when using positive safety reward
                - 0.1 * penalty_effort
                - 0.2 * penalty_action_smoothness 
                - 0.4 * penalty_z_tracking        
                - 8.0 * penalty_height  # height penalty to prevent crashing into ground
            )
        profiler.stop("env/reward_calculation")

        # Terminate Conditions
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > self.map_range[2] * 2.0 + 1.0  # 2*sz + 1, where sz is half z range of the map
        
        # collision check 
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (self.lidar_range - 0.2)
        collision = static_collision
        
        # 总终止条件
        # self.terminated = below_bound | above_bound | collision | poor_following
        self.terminated = below_bound | above_bound | collision
        timeout_truncate = (self.progress_buf >= self.max_episode_per_env).unsqueeze(-1)
        self.truncated = timeout_truncate

        # update previous velocity for smoothness calculation in the next ieteration
        # self.prev_drone_vel_w = drone_vel_w.clone()
        self.prev_human_action = human_actions_local.clone()
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
                self.debug_draw.vector(view_pos.expand_as(v[:, 0]), v[:, 0])
                self.debug_draw.vector(view_pos.expand_as(v[:, -1]), v[:, -1])

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
        # (remove reach_goal flag as no goal target is provided)
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["above_bound"] = above_bound.float()
        self.stats["below_bound"] = below_bound.float()
        # self.stats["within_safe_distance"] = mask_safe
        self.stats["terminated"] = self.terminated.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        
        profiler.stop("env/_compute_state_and_obs")

        # === Probe: Check for NaNs in Observation and Reward ===
        if torch.isnan(self.reward).any():
            print("[Env Probe] NaN detected in Reward!")
            print("Reward components:")
            print(f"  vel: {reward_task[torch.isnan(self.reward)]}")
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
            print("[FollowingEnvSimple] Visualization ENABLED")
        else:
            print("[FollowingEnvSimple] Visualization DISABLED")

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
