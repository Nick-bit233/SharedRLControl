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

        # visualize options
        self.render_lidar = False
        
        # Visualization Trajectory Buffers
        self.viz_traj_human = []
        self.viz_traj_agent = []
        self.viz_human_pos = None

        # Debug mode
        self.debug_mode = cfg.get("debug_mode", False)
        # if self.debug_mode:
        #     import os
        #     log_output_dir = cfg.get("log_output_dir", os.path.join(os.getcwd(), "outputs"))
        #     print("[NavigationEnv]: Debug Mode is ON!")
        #     log_file_path = os.path.join(log_output_dir, "debug_log.csv")
        #     self.debug_log_file = open(log_file_path, "w", newline="")
        #     self.csv_writer = csv.writer(self.debug_log_file)
        #     # 写入表头
        #     self.csv_writer.writerow([
        #         "step", "env_id", "mode", "start_pos_x", "start_pos_y", "start_pos_z",
        #         "reward_total", "reward_vel", "reward_intent_complete", "reward_safe", "reward_penalty_smooth","reward_penalty_height", 
        #         "human_vel_x", "drone_vel_x",
        #     ])

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
        # === Fix: Use new Spec classes ===
        obs_dict = {
            "state": Unbounded((drone_state_dim,), device=self.device), 
            "prev_action": Unbounded((prev_action_dim,), device=self.device),
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

                # 临时插入：强制检查生成的 pos 是否合法
        
        # 如果 Z < 0.1 (考虑到地面是 0，留一点余量)，则打印警告并强制修正
        if (pos[..., 2] < 0.0).any():
            bad_indices = torch.nonzero(pos[..., 2] < 0.0).squeeze()
            print(f"\n[CRITICAL WARNING] Found {len(bad_indices)} drones spawned UNDERGROUND in _reset_idx!")
            print(f"  Bad Z values: {pos[bad_indices, 2]}")
            print(f"  Map Range (sz): {sz}")
            
            # 强制修正，防止报错，但这说明上面的生成逻辑有 bug
            print("  -> Force fixing spawn height to 1.0m")
            pos[..., 2] = torch.clamp(pos[..., 2], min=1.0)

        self.start_pos = pos.clone()  # record start pos for debug
        
        # (randomlize the drone's facing direction because there's no goal to face now)
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * torch.pi - torch.pi

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        # Reset previous step variables
        self.prev_drone_vel_w[env_ids] = 0.
        self.prev_agent_action[env_ids] = 0.

        if (self.training):
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids)
        else:
            # When evaluating, use fixed seed for reproducibility
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids, seed=self.seed)

        self.stats[env_ids] = 0.  

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
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        # TODO：修改了ppo模型的定义，要求action均输出为body frame，这里的转换需要修正。
        actions = tensordict[("agents", "action")]  # action in body frame

        # store applied action so that the subsequent observation (next step) sees it as prev_action
        # actions may be shape (num_envs, 1, 4) or (num_envs, 4).
        # but remember that drone.apply_action only accepts shape (num_envs, 1, 4)
        if actions.ndim > 2:
            actions_flat = actions.reshape(self.num_envs, -1)[..., :4]  # be careful: assume first 4 are vel+yaw
        else:
            actions_flat = actions

        # store as prev_action
        self.prev_agent_action = actions_flat.clone()  # clone to avoid in-place aliasing

        # transform action from body frame to world frame in order to apply to drone
        # get current drone orientation
        drone_orientation_q = self.root_state[..., 3:7].squeeze(1)  # TODO: chceck if need to call drone.get_state
        actions_world = vec_to_world(
            actions_flat, drone_orientation_q, orientation_only=True
        )  # shape: (N, 4), convert vel_b to vel_w
        # unsqueeze to shape (N, 1, 4) for drone apply_action
        actions_world = actions_world.unsqueeze(1)
        self.drone.apply_action(actions_world) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        # No dynamic obstacles
        if self.enable_lidar:
            self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
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
        intent_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # (N,) Boolean
 
        if getattr(self, "manual_mode", False):
            # Use manual action input as human input
            human_actions_local = self.manual_action.clone()
        else:
            # Step the simulated user model to get human action input
            human_actions_local, intent_completed = self.user_model.step(
                user_input_drone_state,
                drone_pos_w
            )

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state_b,
            "human_action": human_actions_local,
            "prev_action": prev_action_local
        }
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan

        # -----------------Reward Calculation-----------------
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

        # Terminate Conditions
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        
        # No collision check needed as there are no obstacles, but ground collision is below_bound
        
        self.terminated = below_bound | above_bound
        # progress_buf 会不断累积，达到 max_episode_length 时触发截断（每次取batch训练时不会重置）
        timeout_truncate = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        self.truncated = timeout_truncate

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = current_vel_w.clone()

        # ----------------- Visualization and Debugging -----------------
        if self._should_render(0):
            self.debug_draw.clear()
            
            # get the first (env_id=0) lidar position
            if self.enable_lidar:
                view_pos = self.lidar.data.pos_w[0]
            else:
                view_pos = drone_pos_w[0]

            # set the camera to focus on the lidar position (which is the drone)
            if self.cfg.global_view:
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

            # A. 绘制无人机实际速度 (蓝色箭头)
            drone_vel_w_vec = drone_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=root_pos, 
                v=drone_vel_w_vec * 1.0, # 长度缩放
                color=(0, 0, 1, 1), # Blue
                size=2.0
            )

            # B. 绘制人类期望速度 (绿色箭头)
            human_vel_w_vec = target_vel_w[viz_env_id]
            self.debug_draw.vector(
                x=root_pos, 
                v=human_vel_w_vec * 1.0, 
                color=(0, 1, 0, 1), # Green
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
                # self.debug_draw.lines(starts, ends, color=(0,0,1,1), width=2.0)
                
                # Draw human path (Green)
                points_h = torch.stack(self.viz_traj_human)
                starts_h = points_h[:-1]
                ends_h = points_h[1:]
                self.debug_draw.vector(starts_h, ends_h - starts_h, color=(0,1,0,1), size=2.0)

        # if self.debug_mode:
        #     # 写入 CSV 日志
        #     self.csv_writer.writerow([
        #         self.progress_buf[viz_env_id].item(),  # step
        #         viz_env_id, # env id
        #         "train" if self.training else "eval",  # mode,
        #         self.start_pos[viz_env_id, 0, 0].item(),  # start x
        #         self.start_pos[viz_env_id, 0, 1].item(),  # start y
        #         self.start_pos[viz_env_id, 0, 2].item(),  # start z
        #         self.reward[viz_env_id].item(),  # reward total
        #         reward_vel[viz_env_id].item(),  # velocity reward
        #         0.0,  # intent complete reward
        #         0.0, # static safety reward
        #         0.0, # smoothness penalty
        #         0.0, # height penalty
        #         human_actions_local[0, 0].item(), # human vel x
        #         vel_b[viz_env_id, 0].item(), # drone vel x
        #     ])
        #     self.debug_log_file.flush() # 强制写入硬盘

        # # -----------------Training Stats-----------------
        # (remove reach_goal flag as no goal target is provided)
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["intent_completion"] = self.intent_complete_counts.float()
        self.stats["collision"] = torch.zeros_like(self.stats["collision"]) # No collision
        self.stats["truncated"] = self.truncated.float()

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
