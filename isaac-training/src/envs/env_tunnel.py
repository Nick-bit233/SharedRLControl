import csv
from gettext import translation
from huggingface_hub import HFCacheInfo
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
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.curriculums import DifficultyScheduler
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.rewards import success_bonus
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
# [Update import name rules]:
# - change all "omni.isaac.orbit" to name "isaaclab"
# - change all "omni.isaac.core" to name "isaacsim.core"
import isaaclab.sim as sim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.robots.drone import MultirotorBase
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler, quat_axis, quat_rotate, quat_rotate_inverse
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleXFormPrim as XFormPrim

from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh

from src.core.trainning_utils import vec_to_body, vec_to_world
from src.simulated_users.user_model_tunnely import UserModelTunnel
from src.core.profiler import get_profiler
from src.datasets.trajectory_dataset import TrajectoryDataset
from src.envs.dynamic_risk import compute_dynamic_command_risk

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


@height_field_to_mesh
def real_room_obstacles_terrain(difficulty: float, cfg: HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Deterministic flat room terrain with fixed cylindrical obstacle footprints."""
    size_x, size_y = cfg.size
    horizontal_scale = cfg.horizontal_scale
    vertical_scale = cfg.vertical_scale
    rows = int(size_x / horizontal_scale)
    cols = int(size_y / horizontal_scale)
    hf_raw = np.zeros((rows, cols), dtype=np.int16)

    centers = getattr(cfg, "obstacle_centers", ())
    radius_range = tuple(getattr(cfg, "obstacle_radius_range", (0.1, 0.15)))
    height_range = tuple(getattr(cfg, "obstacle_height_range", (1.8, 2.4)))
    if not centers:
        return hf_raw

    xs = np.arange(rows, dtype=np.float32) * horizontal_scale - size_x / 2.0
    ys = np.arange(cols, dtype=np.float32) * horizontal_scale - size_y / 2.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

    radius_min, radius_max = radius_range
    height_min, height_max = height_range
    denom = max(len(centers) - 1, 1)
    for idx, center in enumerate(centers):
        cx, cy = float(center[0]), float(center[1])
        ratio = float(idx % (denom + 1)) / denom
        radius = radius_min + (radius_max - radius_min) * ratio
        height = height_min + (height_max - height_min) * ratio
        mask = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius ** 2
        hf_raw[mask] = np.maximum(hf_raw[mask], int(height / vertical_scale))

    return hf_raw


@configclass
class HfRealRoomObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = real_room_obstacles_terrain
    obstacle_centers: tuple = ()
    obstacle_radius_range: tuple = (0.1, 0.15)
    obstacle_height_range: tuple = (1.5, 2.4)


class EnvTunnelResidual(IsaacEnv):

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
        self.env_name = cfg.env.get("name", "tunnel")
        self.is_real_room = self.env_name == "real_room"
        self.room_size = list(cfg.env.get("room_size", [6.0, 5.0, 2.5]))
        self.start_pos_cfg = list(cfg.env.get("start_pos", [-7.0, 0.0, 5.0]))
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

        # Reward Function Params
        reward_cfg = cfg.env.get("reward", {})
        self.enable_following_reward = reward_cfg.get(
            "enable_following",
            cfg.env.get("enable_task_reward", True),
        )
        self.enable_safety_reward = reward_cfg.get("enable_safety", True)
        self.enable_survival_reward = reward_cfg.get("enable_survival", True)
        self.enable_smoothness_penalty = reward_cfg.get("enable_smoothness", True)
        # Backward-compatible alias used by older configs and analysis notes.
        self.enable_task_reward = self.enable_following_reward

        risk_cfg = cfg.env.get("dynamic_risk", {})
        self.dynamic_risk_mode = str(risk_cfg.get("mode", "off"))
        self.enable_dynamic_risk = self.dynamic_risk_mode != "off"
        self.enable_dynamic_risk_reward = self.dynamic_risk_mode in {
            "hybrid_reward",
            "dynamic_reward",
            "full",
        }
        self.dynamic_risk_model_cfg = risk_cfg.get("model", {})
        self.dynamic_risk_reward_cfg = risk_cfg.get("reward", {})
        self.dynamic_risk_metrics_cfg = risk_cfg.get("metrics", {})
        self.legacy_safety_scale = float(
            risk_cfg.get(
                "legacy_safety_scale",
                0.3 if self.dynamic_risk_mode == "hybrid_reward" else 0.0,
            )
        )

        # User Model Initialization
        # Check for offline mode configuration
        offline_mode = cfg.user_model.get("offline_mode", False)
        sampling_mode = cfg.user_model.get("sampling_mode", "scaled")
        
        self.user_model = UserModelTunnel(
            num_envs=self.num_envs,
            cfg=cfg,
            offline_mode=offline_mode,
            dataset=self._trajectory_dataset,
            sampling_mode=sampling_mode,
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
            self.intent_complete_counts = torch.zeros(self.num_envs, 1)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.issue_velocity_w = torch.zeros(self.num_envs, 3)
            self.issue_human_action_b = torch.zeros(self.num_envs, 3)
            self.issue_human_action_w = torch.zeros(self.num_envs, 3)
            self.issue_action_w = torch.zeros(self.num_envs, 3)
            self.issue_hold_command_w = torch.zeros(self.num_envs, 3)
            num_lidar_rays = self.lidar_hbeams * self.lidar_vbeams
            self.issue_ray_dirs_w = torch.zeros(self.num_envs, num_lidar_rays, 3)
            self.issue_ray_dists = torch.full((self.num_envs, num_lidar_rays), self.lidar_range)
            self.pilot_risk_dyn_post = torch.zeros(self.num_envs, 1)
            self.assist_risk_dyn_post = torch.zeros(self.num_envs, 1)
            self.assist_risk_dyn_full = torch.zeros(self.num_envs, 1)
            self.delay_risk = torch.zeros(self.num_envs, 1)
            self.risk_reduction_dyn = torch.zeros(self.num_envs, 1)
            self.min_clearance_pilot = torch.full((self.num_envs, 1), self.lidar_range)
            self.min_clearance_assist = torch.full((self.num_envs, 1), self.lidar_range)
            self.follow_gate = torch.ones(self.num_envs, 1)
            self.modal_residual_norm = torch.zeros(self.num_envs, 1)
            self.risk_worsening = torch.zeros(self.num_envs, 1)
            self.intervention = torch.zeros(self.num_envs, 1)
            self.unnecessary_intervention = torch.zeros(self.num_envs, 1)
            self.unsafe_non_intervention = torch.zeros(self.num_envs, 1)
            self.reward_risk_reduce = torch.zeros(self.num_envs, 1)
            self.reward_risk_worse = torch.zeros(self.num_envs, 1)
            self.reward_abs_risk = torch.zeros(self.num_envs, 1)
            self.reward_delay_risk = torch.zeros(self.num_envs, 1)
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

    def set_seed(self, seed: int):
        result = super().set_seed(seed)
        self.seed = int(seed)
        if hasattr(self.user_model, "set_eval_seed"):
            self.user_model.set_eval_seed(self.seed)
        return result

    def _design_scene(self):
        # Init DRONE and CONTROLLER here, default prim path: /World/envs/envs_0
        self.drone, self.controller = MultirotorBase.make(
            self.cfg.drone.model_name, self.cfg.drone.controller_name, self.device
        )
        initial_translation = tuple(self.start_pos_cfg) if self.is_real_room else (-7.0, 0.0, 5.0)
        drone_prim = self.drone.spawn(translations=[initial_translation])[0]

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

        if self.is_real_room:
            fixed_cfg = self.cfg.env.get("fixed_obstacles", {})
            centers = []
            for center in fixed_cfg.get("centers", []):
                centers.append(tuple(float(v) for v in center[:2]))
            for group in fixed_cfg.get("group_centers", []):
                for center in group:
                    centers.append(tuple(float(v) for v in center[:2]))
            if fixed_cfg.get("include_origin", True):
                origin = fixed_cfg.get("origin", [0.0, 0.0])
                centers.append(tuple(float(v) for v in origin[:2]))

            radius_range = tuple(fixed_cfg.get("radius_range", [0.1, 0.15]))
            height_range = tuple(fixed_cfg.get("height_range", [1.5, 2.4]))
            terrain_size = (float(self.room_size[0]), float(self.room_size[1]))
            sub_terrain_cfg = HfRealRoomObstaclesTerrainCfg(
                size=terrain_size,
                horizontal_scale=0.05,
                vertical_scale=0.05,
                border_width=0.0,
                num_obstacles=len(centers),
                obstacle_height_mode="choice",
                obstacle_width_range=(radius_range[0] * 2.0, radius_range[1] * 2.0),
                obstacle_height_range=height_range,
                platform_width=0,
                obstacle_centers=tuple(centers),
                obstacle_radius_range=radius_range,
            )
            terrain_generator_cfg = TerrainGeneratorCfg(
                seed=0,
                size=terrain_size,
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.05,
                vertical_scale=0.05,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                curriculum=False,
                difficulty_range=(0.0, 1.0),
                sub_terrains={"obstacles": sub_terrain_cfg},
            )
        else:
            terrain_generator_cfg = TerrainGeneratorCfg(
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
            )

        # Terrain generation, as static obstacles
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator_cfg,
            visual_material=None,
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
                # "action": self.drone.action_spec, # number of motor
                "action": Unbounded((self.human_action_dim,), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded((1,)),
                "pilot_risk_dyn_post": Unbounded((1,)),
                "assist_risk_dyn_post": Unbounded((1,)),
                "assist_risk_dyn_full": Unbounded((1,)),
                "delay_risk": Unbounded((1,)),
                "risk_reduction_dyn": Unbounded((1,)),
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
            "diag_pilot_risk_dyn_post": Unbounded(1),
            "diag_assist_risk_dyn_post": Unbounded(1),
            "diag_assist_risk_dyn_full": Unbounded(1),
            "diag_delay_risk": Unbounded(1),
            "diag_risk_reduction_dyn": Unbounded(1),
            "diag_min_clearance_pilot": Unbounded(1),
            "diag_min_clearance_assist": Unbounded(1),
            "diag_follow_gate": Unbounded(1),
            "diag_reward_risk_reduce": Unbounded(1),
            "diag_reward_risk_worse": Unbounded(1),
            "diag_reward_abs_risk": Unbounded(1),
            "diag_reward_delay_risk": Unbounded(1),
            "diag_risk_worsening_rate": Unbounded(1),
            "diag_intervention_rate": Unbounded(1),
            "diag_unnecessary_intervention_rate": Unbounded(1),
            "diag_unsafe_non_intervention_rate": Unbounded(1),
            "diag_modal_residual_norm": Unbounded(1),
            "diag_penalty_smooth": Unbounded(1),
            "diag_penalty_height": Unbounded(1),
            "terminated": Unbounded(1),
            "truncated": Unbounded(1),
            "success": Unbounded(1),
            "out_of_bounds": Unbounded(1),
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
        if self.is_real_room:
            start_pos = torch.as_tensor(self.start_pos_cfg, dtype=torch.float, device=self.device)
            pos[:, 0, :] = start_pos
            start_y_randomization = float(self.cfg.env.get("start_y_randomization", 0.0))
            if self.training and start_y_randomization > 0.0:
                pos[:, 0, 1] += (
                    torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * 2.0 - 1.0
                ) * start_y_randomization
        elif (self.training):  # get random start position
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
        self.issue_velocity_w[env_ids] = 0.
        self.issue_human_action_b[env_ids] = 0.
        self.issue_human_action_w[env_ids] = 0.
        self.issue_action_w[env_ids] = 0.
        self.issue_hold_command_w[env_ids] = 0.
        self.issue_ray_dirs_w[env_ids] = 0.
        self.issue_ray_dists[env_ids] = self.lidar_range
        self.pilot_risk_dyn_post[env_ids] = 0.
        self.assist_risk_dyn_post[env_ids] = 0.
        self.assist_risk_dyn_full[env_ids] = 0.
        self.delay_risk[env_ids] = 0.
        self.risk_reduction_dyn[env_ids] = 0.
        self.min_clearance_pilot[env_ids] = self.lidar_range
        self.min_clearance_assist[env_ids] = self.lidar_range
        self.follow_gate[env_ids] = 1.
        self.modal_residual_norm[env_ids] = 0.
        self.risk_worsening[env_ids] = 0.
        self.intervention[env_ids] = 0.
        self.unnecessary_intervention[env_ids] = 0.
        self.unsafe_non_intervention[env_ids] = 0.
        self.reward_risk_reduce[env_ids] = 0.
        self.reward_risk_worse[env_ids] = 0.
        self.reward_abs_risk[env_ids] = 0.
        self.reward_delay_risk[env_ids] = 0.
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

        # Set default height range for each env
        if self.is_real_room:
            reward_cfg = self.cfg.env.get("reward", {})
            target_height = float(self.cfg.env.get("target_height", self.start_pos_cfg[2]))
            height_tolerance = float(reward_cfg.get("height_tolerance", 0.25))
            self.height_range[env_ids, 0, 0] = max(0.2, target_height - height_tolerance)
            self.height_range[env_ids, 0, 1] = min(float(self.room_size[2]), target_height + height_tolerance)
        else:
            self.height_range[env_ids, 0, 0] = 0.4 * sz    # min height
            self.height_range[env_ids, 0, 1] = 1.6 * sz    # max height

        # Reset visualization buffers if env 0 is reset
        if 0 in env_ids:
            self.viz_traj_agent = []
            self.viz_traj_human = []
            # Find the index of env 0 in env_ids
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            self.viz_human_pos = pos[idx, 0].clone()

    def _cache_issue_time_inputs(self, tensordict: TensorDictBase) -> None:
        if not self.enable_dynamic_risk:
            return

        obs_state = tensordict[("agents", "observation", "state")]
        obs_human_action = tensordict[("agents", "observation", "human_action")]
        if obs_state.ndim == 3 and obs_state.shape[1] == 1:
            obs_state = obs_state.squeeze(1)
        if obs_human_action.ndim == 3 and obs_human_action.shape[1] == 1:
            obs_human_action = obs_human_action.squeeze(1)

        issue_quat = obs_state[..., 6:10]
        issue_vel_b = obs_state[..., :3]
        issue_human_b = obs_human_action[..., :3]

        self.issue_velocity_w[:] = vec_to_world(
            issue_vel_b,
            issue_quat,
            orientation_only=True,
            yaw_only=False,
        )
        self.issue_human_action_b[:] = issue_human_b
        self.issue_human_action_w[:] = vec_to_world(
            issue_human_b,
            issue_quat,
            orientation_only=True,
            yaw_only=True,
        )
        self.issue_hold_command_w[:] = self.prev_action_command[..., :3]

        if self.enable_lidar and self.lidar is not None:
            ray_vecs_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
            ray_dists = ray_vecs_w.norm(dim=-1).clamp_max(self.lidar_range)
            ray_dirs_w = ray_vecs_w / (ray_dists.unsqueeze(-1) + 1e-6)
            self.issue_ray_dirs_w[:] = ray_dirs_w
            self.issue_ray_dists[:] = ray_dists
        else:
            self.issue_ray_dirs_w[:] = 0.
            self.issue_ray_dists[:] = self.lidar_range

    def _compute_dynamic_risk_terms(self) -> None:
        if not (self.enable_dynamic_risk and self.enable_lidar and self.lidar is not None):
            self.pilot_risk_dyn_post.zero_()
            self.assist_risk_dyn_post.zero_()
            self.assist_risk_dyn_full.zero_()
            self.delay_risk.zero_()
            self.risk_reduction_dyn.zero_()
            self.min_clearance_pilot.fill_(self.lidar_range)
            self.min_clearance_assist.fill_(self.lidar_range)
            self.follow_gate.fill_(1.0)
            return

        pilot = compute_dynamic_command_risk(
            velocity_w=self.issue_velocity_w,
            command_w=self.issue_human_action_w,
            hold_command_w=self.issue_hold_command_w,
            ray_dirs_w=self.issue_ray_dirs_w,
            ray_dists=self.issue_ray_dists,
            params=self.dynamic_risk_model_cfg,
        )
        assist = compute_dynamic_command_risk(
            velocity_w=self.issue_velocity_w,
            command_w=self.issue_action_w,
            hold_command_w=self.issue_hold_command_w,
            ray_dirs_w=self.issue_ray_dirs_w,
            ray_dists=self.issue_ray_dists,
            params=self.dynamic_risk_model_cfg,
        )

        self.pilot_risk_dyn_post[:] = pilot["rho_post"]
        self.assist_risk_dyn_post[:] = assist["rho_post"]
        self.assist_risk_dyn_full[:] = assist["rho_full"]
        self.delay_risk[:] = assist["rho_delay"]
        self.risk_reduction_dyn[:] = self.pilot_risk_dyn_post - self.assist_risk_dyn_post
        self.min_clearance_pilot[:] = pilot["min_clearance"]
        self.min_clearance_assist[:] = assist["min_clearance"]

        alpha_f = float(self.dynamic_risk_reward_cfg.get("alpha_f", 0.5))
        g_min = float(self.dynamic_risk_reward_cfg.get("g_min", 0.35))
        self.follow_gate[:] = (1.0 - alpha_f * self.pilot_risk_dyn_post).clamp(g_min, 1.0)

    def _update_dynamic_risk_stats(self) -> None:
        step_count = self.progress_buf.unsqueeze(1).clamp_min(1).float()

        def update_mean(key: str, value: torch.Tensor) -> None:
            self.stats[key] = self.stats[key] + (value - self.stats[key]) / step_count

        update_mean("diag_pilot_risk_dyn_post", self.pilot_risk_dyn_post)
        update_mean("diag_assist_risk_dyn_post", self.assist_risk_dyn_post)
        update_mean("diag_assist_risk_dyn_full", self.assist_risk_dyn_full)
        update_mean("diag_delay_risk", self.delay_risk)
        update_mean("diag_risk_reduction_dyn", self.risk_reduction_dyn)
        update_mean("diag_min_clearance_pilot", self.min_clearance_pilot)
        update_mean("diag_min_clearance_assist", self.min_clearance_assist)
        update_mean("diag_follow_gate", self.follow_gate)
        update_mean("diag_reward_risk_reduce", self.reward_risk_reduce)
        update_mean("diag_reward_risk_worse", self.reward_risk_worse)
        update_mean("diag_reward_abs_risk", self.reward_abs_risk)
        update_mean("diag_reward_delay_risk", self.reward_delay_risk)
        update_mean("diag_risk_worsening_rate", self.risk_worsening)
        update_mean("diag_intervention_rate", self.intervention)
        update_mean("diag_unnecessary_intervention_rate", self.unnecessary_intervention)
        update_mean("diag_unsafe_non_intervention_rate", self.unsafe_non_intervention)
        update_mean("diag_modal_residual_norm", self.modal_residual_norm)

    def _pre_sim_step(self, tensordict: TensorDictBase):

        # Store last step action command for smoothness reward
        self.prev_action_command[:] = self.agent_action.clone()
        self._cache_issue_time_inputs(tensordict)

        # Get new action command from policy (world frame)
        action_command = tensordict[("agents", "action")] 
        
        # Ensure actions shape is compatible
        if action_command.ndim == 3:
            action_command = action_command.squeeze(1)
        if self.enable_dynamic_risk:
            self.issue_action_w[:] = action_command[..., :3]
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
        if self.enable_lidar:
            self.lidar_scan = self.lidar_range - (
                (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
                .norm(dim=-1)
                .clamp_max(self.lidar_range)
                .reshape(self.num_envs, 1, *self.lidar_resolution)
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
                human_actions_local, _ = self.user_model.step(
                    user_input_drone_state,
                    drone_pos_w
                )
        
        # print("[EnvDebug] Human action (local frame) at Env0 step {}: {}".format(self.common_step_counter, human_actions_local[0].cpu().numpy()))

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state_b,
            "human_action": human_actions_local,
        }
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan
        # -----------------Reward Calculation-----------------
        profiler.start("env/reward_calculation")

        prev_human_action_b = self.prev_human_action.clone()
        target_vel_w = vec_to_world(
            prev_human_action_b,
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True
        )
        if self.enable_dynamic_risk_reward:
            target_vel_w = self.issue_human_action_w.clone()
        self._compute_dynamic_risk_terms()
        
        # a. Safety Penalty (Barrier Function)
        # Instead of a positive reward for being far, we apply a negative penalty for being close.
        # This decouples safety from the task when the agent is in safe space.
        # 只关注human_action输入方向（以及当前速度方向）上的障碍物，其他方向的障碍物降低权重
        reward_params = self.cfg.env.get("reward", {})
        r_safety = 0.0
        if self.enable_lidar:
            # -------------------------------------------------------------------------
            # 基础准备：获取所有射线的距离和方向
            # -------------------------------------------------------------------------
            # 计算每条射线在世界坐标系下的向量 (N, num_rays, 3)
            ray_vecs_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
            # 计算每条射线的实际物理距离 (N, num_rays)
            ray_dists = ray_vecs_w.norm(dim=-1).clamp_max(self.lidar_range)
            # 归一化得到射线方向的单位向量 (N, num_rays, 3)
            ray_dirs_w = ray_vecs_w / (ray_dists.unsqueeze(-1) + 1e-6)

            # 1. 绝对最小距离 (全局最危险距离)
            min_dist_to_obs, _ = ray_dists.min(dim=-1, keepdim=True)

            # -------------------------------------------------------------------------
            # 定义“关注锥角” (Cone Threshold)
            # -------------------------------------------------------------------------
            # 设定我们只关心速度方向前方正负 30 度的障碍物
            # cos(30°) ≈ 0.866
            cone_threshold = 0.866

            # -------------------------------------------------------------------------
            # 2. 计算当前速度方向 (v_now) 上的最短距离
            # -------------------------------------------------------------------------
            cur_vel_norm = drone_vel_w.norm(dim=-1, keepdim=True)
            # 防止除零，获取速度单位向量
            cur_vel_dir = drone_vel_w / (cur_vel_norm + 1e-6) # (N, 3)

            # 计算所有射线方向与当前速度方向的余弦相似度 (N, num_rays)
            cos_sim_vel = (ray_dirs_w * cur_vel_dir.unsqueeze(1)).sum(dim=-1)

            # 选出落在圆锥视角内的射线，视角外的射线距离强制设为最大安全距离
            mask_vel = cos_sim_vel > cone_threshold
            dist_in_vel_cone = torch.where(mask_vel, ray_dists, torch.full_like(ray_dists, self.lidar_range))

            # 获取该方向上的最短距离
            dist_to_cur_vel_dir, _ = dist_in_vel_cone.min(dim=-1, keepdim=True)

            # 【关键处理】：如果无人机悬停（速度极小），则没有方向，不应产生前方碰撞惩罚
            is_moving = (cur_vel_norm > 0.1).float()
            dist_to_cur_vel_dir = is_moving * dist_to_cur_vel_dir + (1.0 - is_moving) * self.lidar_range

            # -------------------------------------------------------------------------
            # 3. 计算输入指令方向 (v_in) 上的最短距离
            # -------------------------------------------------------------------------
            target_vel_norm = target_vel_w.norm(dim=-1, keepdim=True)
            target_vel_dir = target_vel_w / (target_vel_norm + 1e-6)

            cos_sim_cmd = (ray_dirs_w * target_vel_dir.unsqueeze(1)).sum(dim=-1)
            mask_cmd = cos_sim_cmd > cone_threshold

            dist_in_cmd_cone = torch.where(mask_cmd, ray_dists, torch.full_like(ray_dists, self.lidar_range))
            dist_to_human_action_dir, _ = dist_in_cmd_cone.min(dim=-1, keepdim=True)

            has_cmd = (target_vel_norm > 0.1).float()
            dist_to_human_action_dir = has_cmd * dist_to_human_action_dir + (1.0 - has_cmd) * self.lidar_range
             
            # -------------------------------------------------------------------------
            # 融合指数惩罚
            # -------------------------------------------------------------------------
            r_safety_dist_scale = float(reward_params.get("safety_dist_scale", 1.0))  # Wider gradient reach (was 0.5)
            
            # 计算三个独立的指数障碍物惩罚项 (范围 0 到 1，越近越接近1)
            p_min = torch.exp(-min_dist_to_obs / r_safety_dist_scale)
            p_vel = torch.exp(-dist_to_cur_vel_dir / r_safety_dist_scale)
            p_cmd = torch.exp(-dist_to_human_action_dir / r_safety_dist_scale)
            
            # 分配权重
            w_min = 0.2  # 基础安全意识
            w_vel = 0.4  # 物理惯性危险：撞墙风险
            w_cmd = 0.4  # 主动寻死危险：指令导致撞墙
            
            # 安全区覆盖LiDAR 75%范围，让策略更早收到安全信号
            safe_zone = float(reward_params.get("safe_zone", 4.0))  # was 3.0
            mask_min = (min_dist_to_obs < safe_zone).float()
            mask_vel = (dist_to_cur_vel_dir < safe_zone).float()
            mask_cmd = (dist_to_human_action_dir < safe_zone).float()
            
            # 最终的 safety 奖励就是三者的负加权和
            r_safety = - (
                w_min * (p_min * mask_min) + 
                w_vel * (p_vel * mask_vel) + 
                w_cmd * (p_cmd * mask_cmd)
            )

        if not self.enable_safety_reward:
            r_safety = 0.0

        # c. Velocity following reward (Positive)
        # c1. 方向奖励 (Alignment)
        cosine_sim = torch.cosine_similarity(target_vel_w, drone_vel_w, dim=-1).unsqueeze(-1)
        reward_direction = (cosine_sim + 1.0) / 2.0 

        # c2. 速度大小奖励 (Magnitude)
        vel_error = (target_vel_w - drone_vel_w).norm(dim=-1, keepdim=True)
        reward_speed_match = torch.exp(-2.0 * vel_error) 
        
        # Combined Task Reward (calculated for diagnostics)
        reward_task = 1.0 * reward_speed_match + 0.5 * reward_direction

        residual_norm = (self.issue_action_w - self.issue_human_action_w).norm(dim=-1, keepdim=True) / max(self.max_action_vel, 1e-6)
        self.modal_residual_norm[:] = residual_norm
        epsilon_delta = float(self.dynamic_risk_metrics_cfg.get("epsilon_delta", 0.05))
        rho_safe = float(self.dynamic_risk_metrics_cfg.get("rho_safe", 0.2))
        rho_danger = float(self.dynamic_risk_metrics_cfg.get("rho_danger", 0.7))
        self.risk_worsening[:] = (self.assist_risk_dyn_post > self.pilot_risk_dyn_post).float()
        self.intervention[:] = (residual_norm > epsilon_delta).float()
        self.unnecessary_intervention[:] = (
            (residual_norm > epsilon_delta) & (self.pilot_risk_dyn_post < rho_safe)
        ).float()
        self.unsafe_non_intervention[:] = (
            (residual_norm <= epsilon_delta) & (self.pilot_risk_dyn_post > rho_danger)
        ).float()

        self.reward_risk_reduce.zero_()
        self.reward_risk_worse.zero_()
        self.reward_abs_risk.zero_()
        self.reward_delay_risk.zero_()
        if self.enable_dynamic_risk_reward and self.enable_safety_reward:
            self.reward_risk_reduce[:] = float(self.dynamic_risk_reward_cfg.get("w_delta_rho", 0.5)) * self.risk_reduction_dyn
            self.reward_risk_worse[:] = -float(self.dynamic_risk_reward_cfg.get("w_worse", 1.5)) * (
                self.assist_risk_dyn_post - self.pilot_risk_dyn_post
            ).clamp(min=0.0)
            self.reward_abs_risk[:] = -float(self.dynamic_risk_reward_cfg.get("w_abs", 0.4)) * self.assist_risk_dyn_full
            self.reward_delay_risk[:] = -float(self.dynamic_risk_reward_cfg.get("w_delay", 0.3)) * self.delay_risk

        # d. Smoothness Penalty
        action_diff = (self.agent_action - self.prev_action_command).norm(dim=-1, keepdim=True)
        penalty_action_smoothness = (action_diff / self.max_action_vel) ** 2
        
        # Total Step Reward
        # [Analysis]: If enable_task_reward is False, the agent acts as a pure "Safety Shield".
        # It has no incentive to follow velocity other than the Residual Regularization term in the loss.
        # This removes the conflict where deviating for safety penalizes task reward.
        if self.enable_following_reward:
            task_reward_term = reward_task * self.follow_gate if self.enable_dynamic_risk_reward else reward_task
        else:
            task_reward_term = 0.0

        # Survival reward: positive reward per step alive
        # Reduced to prevent hovering from being optimal (was 0.5)
        reward_survival = 0.2 if self.enable_survival_reward else 0.0

        # Forward progress reward: incentivize moving along tunnel length axis (pos[0])
        # pos[0] starts at -7.0, increases toward +12.0
        forward_vel = drone_vel_w[..., 0:1]  # velocity along tunnel length axis
        reward_progress = float(reward_params.get("progress_weight", 0.0)) * forward_vel.clamp(min=0.0)

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

        if self.dynamic_risk_mode == "hybrid_reward":
            legacy_safety_scale = self.legacy_safety_scale
        elif self.dynamic_risk_mode in {"dynamic_reward", "full"}:
            legacy_safety_scale = 0.0
        else:
            legacy_safety_scale = 1.0

        dynamic_risk_reward = (
            self.reward_risk_reduce
            + self.reward_risk_worse
            + self.reward_abs_risk
            + self.reward_delay_risk
        )
        
        self.reward = (
            task_reward_term
            + reward_survival
            + reward_progress
            + legacy_safety_scale * float(reward_params.get("safety_weight", 3.0)) * r_safety
            + dynamic_risk_reward
            - float(reward_params.get("height_penalty_weight", 10.0)) * penalty_height
            - (float(reward_params.get("smoothness_weight", 0.5)) * penalty_action_smoothness if self.enable_smoothness_penalty else 0.0)
        )
        
        # Terminate Conditions & Terminal Penalty
        if self.is_real_room:
            boundary_cfg = self.cfg.env.get("boundary_buffer", {})
            x_buffer = float(boundary_cfg.get("x", 1.0))
            y_buffer = float(boundary_cfg.get("y", 1.0))
            z_buffer = float(boundary_cfg.get("z", 0.5))
            half_x = float(self.room_size[0]) / 2.0
            half_y = float(self.room_size[1]) / 2.0
            below_bound = self.drone.pos[..., 2] < 0.2
            above_bound = self.drone.pos[..., 2] > float(self.room_size[2]) + z_buffer
            x_lower_oob = self.drone.pos[..., 0] < -half_x - x_buffer
            y_oob = self.drone.pos[..., 1].abs() > half_y + y_buffer
            out_of_bounds = below_bound | above_bound | x_lower_oob | y_oob
            success = self.drone.pos[..., 0] >= float(self.cfg.env.get("success_x", half_x))
        else:
            below_bound = self.drone.pos[..., 2] < 0.2
            above_bound = self.drone.pos[..., 2] > self.map_range[2] * 2.0 + 1.0
            out_of_bounds = below_bound | above_bound
            # success if drone traverses the tunnel length (pos[0] is the tunnel length axis)
            # Drone starts at pos[0] ≈ -7.0, tunnel ends at pos[0] ≈ +10.0
            success = self.drone.pos[..., 0] > 10.0

        collision_dist = float(reward_params.get("collision_distance", 0.3))
        if self.enable_lidar:
            static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (1.0 - collision_dist / self.lidar_range)
        else:
            static_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        collision = static_collision
        success_bonus = float(reward_params.get("success_bonus", 0.0))
        if success_bonus:
            self.reward[success] += success_bonus
        
        self.terminated = out_of_bounds | collision
        timeout_truncate = (self.progress_buf >= self.max_episode_per_env).unsqueeze(-1)
        self.truncated = timeout_truncate | success

        # === Apply Terminal Reward & Penalty ===
        # If crashed (not timed out), give a massive penalty encoded into the reward of this step.
        # This ensures the Value Function learns to fear these states.
        crash_penalty = -10.0  # Reduced from -50 to lower return variance
        # Only apply to envs that just terminated due to crash (collision or out-of-bounds)
        crashed_mask = ((collision | out_of_bounds) & ~self.truncated)
        self.reward[crashed_mask] += crash_penalty

        # update previous velocity for smoothness calculation in the next ieteration
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
                eye_height = 10.0 if self.is_real_room else 32.0
                set_camera_view(
                    eye=torch.tensor([-3.0, 0.0, eye_height]),  # global top-down view
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
        self.stats["out_of_bounds"] = out_of_bounds.float()
    
        # === DIAGNOSTIC: Log internal reward components to stats ===
        self.stats["diag_reward"] = self.reward
        self.stats["diag_reward_task"] = reward_task
        self._update_dynamic_risk_stats()
        self.stats["diag_penalty_smooth"] = penalty_action_smoothness
        self.stats["diag_penalty_height"] = penalty_height
        
        self.stats["terminated"] = self.terminated.float()
        self.stats["collision"] = collision.float()
        self.stats["success"] = success.float()
        self.stats["truncated"] = self.truncated.float()

        self.stats["debug_vec_world"] = drone_vel_w
        self.stats["debug_vec_policy"] = self.agent_action_original
        # Convert human action from body frame to world frame for consistent comparison
        if self.enable_dynamic_risk:
            human_action_w = target_vel_w
        else:
            human_action_w = vec_to_world(
                human_actions_local[:, :3],
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
                    "reward": reward,
                    "pilot_risk_dyn_post": self.pilot_risk_dyn_post,
                    "assist_risk_dyn_post": self.assist_risk_dyn_post,
                    "assist_risk_dyn_full": self.assist_risk_dyn_full,
                    "delay_risk": self.delay_risk,
                    "risk_reduction_dyn": self.risk_reduction_dyn,
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
