import csv
import torch
import einops
import numpy as np
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
from omni_drones.robots.drone import MultirotorBase
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate, quat_rotate_inverse
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view
from trainning_utils import vec_to_body, vec_to_world
from user_model import UserModel
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
import time
from profiler import get_profiler

class FollowingEnvSimple(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Simple Shared Autonomy Env (No Obstacles)...")
        
        # Force remove obstacles
        cfg.env.num_obstacles = 0
        cfg.env_dyn.num_obstacles = 0

        # observation related configs
        self.obs_add_prev = cfg.algo.observation_cat_prev_action
        
        # Check if lidar is enabled (default to False if not specified)
        self.enable_lidar = cfg.env.get("enable_lidar", False)

        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        # Env map params:
        self.map_range = cfg.env.map_range  # [x_range, y_range, z_range], half extents
        self.platform_width = cfg.env.platform_width  # square platform width at the center of the map

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # LiDAR Intialization
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)
        if self.enable_lidar:
            ray_caster_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                attach_yaw_only=True,
                # attach_yaw_only=False,
                pattern_cfg=patterns.BpearlPatternCfg(
                    horizontal_res=self.lidar_hres, # horizontal default is set to 10
                    vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
                ),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
                # mesh_prim_paths=["/World"],
            )
            self.lidar = RayCaster(ray_caster_cfg)
            self.lidar._initialize_impl()
        else:
            self.lidar = None

        # User Model Initialization
        self.user_model = UserModel(
            num_envs=self.num_envs,
            cfg=cfg, 
        ) 
        self.seed = cfg.get("seed", 0)  # seed for evaluation mode
        
        # history action buffer for memory
        with torch.device(self.device):
            self.start_pos = torch.zeros(self.num_envs, 3)
            # prev_drone_vel_w is used to compute acceleration-based penalty
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 3)
            # previous action taken by the agent (drone) (vel_b[3], yaw_rate_b[1])
            # use this 4D action instaed of the prev_drone_vel_w in user model and GRU network.
            self.prev_agent_action = torch.zeros(self.num_envs, 4)
            self.intent_complete_counts = torch.zeros(self.num_envs, 1)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            # Cumulative following error for early termination
            self.cumulative_error = torch.zeros(self.num_envs, 1)
            self.error_ema_alpha = 0.995  # Exponential moving average decay factor
            self.error_threshold_base = 0.5  # Base threshold (strict, for no-obstacle case)
            self.error_threshold_max = 5.0   # Max threshold (relaxed, when obstacles are very close)
            self.safety_margin = 1.5  # Distance within which we start relaxing the threshold

        # visualize options
        self.disable_visualization = True
        self.render_lidar = False
        
        # Visualization Trajectory Buffers
        self.viz_traj_human = []
        self.viz_traj_agent = []
        self.viz_human_pos = None

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import isaacsim.core.utils.prims as prim_utils

        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls()
        print("[NavigationEnv]: Spawning Drone Model:", self.cfg.drone.model_name)
        print(f"[NavigationEnv]: Drone Model Config: {cfg}")
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ["/World/defaultGroundPlane"]
        # No Terrain and dynamic obstacles initialization

    def _set_specs(self):
        drone_state_dim = 10  # (vel_b[3] + ang_vel_b[3] + orientation_q[4])
        prev_action_dim = 4  # (vel_b[3] + yaw_rate_b[1])
        human_action_dim = 4  # (vel_b[3] + yaw_rate_b[1])

        # Observation Spec
        if self.obs_add_prev:
            obs_dict = {
                "state": Unbounded((drone_state_dim,), device=self.device), 
                "prev_action": Unbounded((prev_action_dim,), device=self.device),
                "human_action": Unbounded((human_action_dim,), device=self.device),
            }
        else:
            obs_dict = {
                "state": Unbounded((drone_state_dim,), device=self.device), 
                "human_action": Unbounded((human_action_dim,), device=self.device),
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
            "intent_completion": Unbounded(1),
            "collision": Unbounded(1),
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
        self.prev_drone_vel_w[env_ids] = 0.
        self.prev_agent_action[env_ids] = 0.
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
        if not hasattr(self, 'max_episode_per_env'):
            self.max_episode_per_env = torch.full(
                (self.num_envs,), float(self.max_episode_length), 
                dtype=torch.float, device=self.device
            )
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
        # 这里的action为最终传递给VelController的速度指令，默认为world frame
        # 这里不可做任何变换，因为torchrl会直接把VelController转换后的action（此时为推力指令）传递给drone.apply_action()
        # 对actions坐标系等的转换提前在ppo.__call__里完成
        actions = tensordict[("agents", "action")]

        # store applied action (TODO： 确定这里存储的action数值是速度指令还是推力指令，必要时做转换)
        # actions may be shape (num_envs, 1, 4) or (num_envs, 4).
        # but remember that drone.apply_action only accepts shape (num_envs, 1, 4)
        if actions.ndim > 2:
            actions_flat = actions.reshape(self.num_envs, -1)[..., :4]  # be careful: assume first 4 are vel+yaw
        else:
            actions_flat = actions
        self.prev_agent_action = actions_flat.clone()  # clone to avoid in-place aliasing

        # Apply rotor commands directly to the drone
        # drone.apply_action expects rotor throttle commands, not velocity
        # Ensure actions shape is compatible with drone.shape (num_envs, 1, 4)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)  # (num_envs, 4) -> (num_envs, 1, 4)
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        profiler = get_profiler()
        with profiler.timer("env/_post_sim_step"):
            # No dynamic obstacles
            if self.enable_lidar:
                self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        profiler = get_profiler()
        profiler.start("env/_compute_state_and_obs")
        
        self.root_state = self.drone.get_state(env_frame=False)  # get drone's root state in world frame
        # explaination of root state:  
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

        # ---------Network Input IV: Previous drone action--------
        
        prev_action_local = self.prev_agent_action  # shape: (N, 4)

        # ---------Network Input V: Human control action--------
        user_input_drone_state = drone_state_b.clone()  # (N, 10)

        human_actions_local = torch.zeros(self.num_envs, 4, device=self.device)  # (N, 4)
        need_refill = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # (N,) Boolean
 
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
        if self.obs_add_prev:
            obs["prev_action"] = prev_action_local
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan
        # -----------------Reward Calculation-----------------
        profiler.start("env/reward_calculation")
        # Only reward for correct following human input velocity
        
        current_vel_w = self.drone.vel_w[..., :3] # (N, 1, 3)
        # squeeze the drone velocity tensor if needed
        if current_vel_w.ndim == 3 and current_vel_w.shape[1] == 1:
            current_vel_w = current_vel_w.squeeze(1)  # (N, 3)
        current_yaw_rate = self.drone.vel_w[..., 5:6]  # (N, 1, 1) get wz as yaw rate
        if current_yaw_rate.ndim == 3 and current_yaw_rate.shape[1] == 1:
            current_yaw_rate = current_yaw_rate.squeeze(-1)  # (N, 1)

        # # Velocity following reward (Positive)
        # TODO: 奖励重视速度的方向和yaw角速度，而不是绝对值大小
        # human_action_vel_b = human_actions_local[..., :3] # (N, 3)
        # target_vel_w = quat_rotate(drone_orientation_q, human_action_vel_b)

        # vel_error_norm = (torch.norm(current_vel_w - target_vel_w, dim=-1, keepdim=True))
        # reward_vel = torch.exp(-vel_error_norm)  # Positive reward
        
        # # Yaw rate following reward (Positive)
        # target_yaw_rate = human_actions_local[..., 3:4]  # (N, 1)
        # yaw_rate_error_norm = (torch.norm(current_yaw_rate - target_yaw_rate, dim=-1, keepdim=True))
        # reward_vel += torch.exp(-yaw_rate_error_norm)  # Positive reward

        # # 4D Action difference reward (Positive)
        human_action_vel_b = human_actions_local[..., :3] # (N, 3)
        target_vel_w = quat_rotate(drone_orientation_q, human_action_vel_b)
        target_yaw_rate = human_actions_local[..., 3:4]  # (N, 1)

        target_action = torch.cat([target_vel_w, target_yaw_rate], dim=-1)  # (N, 4)
        current_action = torch.cat([current_vel_w, current_yaw_rate], dim=-1)  # (N, 4)

        action_diff = (current_action - target_action).norm(dim=-1, keepdim=True)
        reward_vel = torch.exp(-action_diff)  # (N, 1) Positive reward

        # d. smoothness reward for action smoothness
        penalty_smooth = (current_vel_w - self.prev_drone_vel_w).norm(dim=-1, keepdim=True)
        
        # e. height penalty reward for flying unnessarily high or low
        h_min, h_max = self.height_range[..., 0], self.height_range[..., 1]
        # penalty when z > h_max + 0.2 or z < h_min - 0.2
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        z = self.drone.pos[..., 2]
        penalty_height[z > (h_max + 0.2)] = ((z - h_max - 0.2)**2)[z > (h_max + 0.2)]
        penalty_height[z < (h_min - 0.2)] = ((h_min - 0.2 - z)**2)[z < (h_min - 0.2)]
        
        # Survival reward (keep flying as long as possible)
        reward_survival = 1.0
        
        # Final reward calculation
        self.reward = (
            2.0 * reward_vel +
            0.1 * reward_survival 
            - 0.05 * penalty_smooth 
            - 1.0 * penalty_height
        )
        profiler.stop("env/reward_calculation")

        # Terminate Conditions
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > self.map_range[2] * 2.0 + 1.0  # 2*sz + 1, where sz is half z range of the map
        
        # No collision check needed as there are no obstacles, but ground collision is below_bound
        
        # 添加累积跟随误差终止条件（支持动态阈值）
        # 使用指数移动平均更新累积误差
        self.cumulative_error = (
            self.error_ema_alpha * self.cumulative_error + 
            (1 - self.error_ema_alpha) * action_diff
        )
        
        # 动态误差阈值：根据障碍物距离调整
        # 在 env_simple（无障碍物）中，使用固定的基础阈值
        # if self.enable_lidar:
        #     # 有lidar时，根据最近障碍物距离计算动态阈值
        #     # lidar_scan: 值越大表示障碍物越近 (range - distance)
        #     min_obstacle_dist = self.lidar_range - self.lidar_scan.max(dim=(2, 3)).values  # (N, 1)
        #     # obstacle_proximity: 0=无障碍物/远, 1=非常近
        #     obstacle_proximity = torch.clamp(
        #         (self.safety_margin - min_obstacle_dist) / self.safety_margin, 
        #         min=0, max=1
        #     )
        #     # 动态阈值：障碍物越近，阈值越宽松
        #     dynamic_threshold = (
        #         self.error_threshold_base + 
        #         (self.error_threshold_max - self.error_threshold_base) * obstacle_proximity
        #     )
        # else:
        #     # 无lidar（无障碍物环境），使用固定的基础阈值
        #     dynamic_threshold = self.error_threshold_base
        
        # 如果累积误差持续过大，视为"跟随失败"
        poor_following = self.cumulative_error > self.error_threshold_base
        
        self.terminated = below_bound | above_bound | poor_following
        # progress_buf 会不断累积，达到 max_episode_per_env 时触发截断
        # 使用随机化的 episode 长度来打破 episode 同步结束的问题
        timeout_truncate = (self.progress_buf >= self.max_episode_per_env).unsqueeze(-1)
        self.truncated = timeout_truncate

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = current_vel_w.clone()

        # ----------------- Visualization and Debugging -----------------
        profiler.start("env/visualization")
        if self._should_render(0) and not self.disable_visualization:
            self.debug_draw.clear()
            
            # get the first (env_id=0) lidar position
            if self.enable_lidar:
                view_pos = self.lidar.data.pos_w[0]
            else:
                view_pos = drone_pos_w[0]

            # set the camera to focus on the lidar position (which is the drone)
            camera_mode = getattr(self, '_camera_view_mode', 'follow')
            if camera_mode == 'global':
                set_camera_view(
                    eye=torch.tensor([-3.0, 0.0, 20.0]),  # global top-down view
                    target=torch.tensor([0.0, 0.0, 0.0])                        
                )
            else:
                set_camera_view(
                    # use cfg viewer settings as offset
                    eye=view_pos.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                    target=view_pos.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
                )
            if self.render_lidar and self.enable_lidar:
                # rendering LiDAR rays (360 degrees horizontal)
                v = (self.lidar.data.ray_hits_w[0] - view_pos).reshape(*self.lidar_resolution, 3)
                self.debug_draw.vector(view_pos.expand_as(v[:, 0]), v[:, 0])
                self.debug_draw.vector(view_pos.expand_as(v[:, -1]), v[:, -1])

            viz_env_id = 0

            # 绘制向量 (转换回世界系以便绘制)
            root_pos = drone_pos_w[viz_env_id]  # root pos == view pos TODO: merge it

            # A. 绘制无人机实际速度 (红色箭头)
            drone_vel_w_vec = drone_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=root_pos, 
                v=drone_vel_w_vec * 1.0, # 长度缩放
                color=(1, 0, 0, 1), # Red
                size=2.0
            )

            # B. 绘制人类期望速度 (黄色箭头)
            human_vel_w_vec = target_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=root_pos, 
                v=human_vel_w_vec * 1.0, 
                color=(1, 1, 0, 1), # Yellow
                size=2.0
            )
            
            # C. Draw Trajectories
            # Update trajectory buffers
            # Agent pos
            curr_pos = root_pos.clone()
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
        self.stats["poor_following"] = poor_following.float()
        self.stats["terminated_total"] = self.terminated.float()
        self.stats["collision"] = torch.zeros_like(self.stats["collision"]) # No collision
        self.stats["truncated"] = self.truncated.float()
        
        profiler.stop("env/_compute_state_and_obs")

        # === Probe: Check for NaNs in Observation and Reward ===
        if torch.isnan(self.reward).any():
            print("[Env Probe] NaN detected in Reward!")
            print("Reward components:")
            print(f"  vel: {reward_vel[torch.isnan(self.reward)]}")
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
        self.manual_action = torch.zeros(self.num_envs, 4, device=self.device)

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
