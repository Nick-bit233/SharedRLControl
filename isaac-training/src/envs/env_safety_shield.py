"""
Safety Shield Environment — open obstacle field with danger-aware tracking reward.

Key differences from EnvTunnelResidual:
  1. No tunnel walls — open obstacle field (HfDiscreteObstaclesTerrainCfg)
  2. No directional "success" — episode ends only on collision or timeout
  3. Danger-aware tracking reward — follow human when safe, allow deviation when dangerous
  4. Diverse human inputs via UserModelDiverse (multi-modal: perlin3d, straight, arc, hover)
  5. Random start heading — drone starts at random yaw (not always facing +x)
  6. Pareto evaluation metrics — tracking RMSE, collision rate, intervention rate
"""

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
from isaaclab.terrains import (
    TerrainImporterCfg, TerrainImporter,
    TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg,
)
from omni_drones.utils.torch import (
    euler_to_quaternion, quaternion_to_euler,
    quat_axis, quat_rotate, quat_rotate_inverse,
)
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view

from src.core.trainning_utils import vec_to_body, vec_to_world
from src.core.user_model_diverse import UserModelDiverse
from src.core.profiler import get_profiler

import math


class EnvSafetyShield(IsaacEnv):
    """
    Open-field obstacle avoidance with danger-aware velocity tracking.

    The drone follows diverse human velocity commands while avoiding randomly
    placed obstacles.  There is **no navigation goal** — the objective is
    continuous high-fidelity velocity tracking with minimal safety interventions.
    """

    def __init__(self, cfg):
        print("[SafetyShield Env]: Initializing...")

        self.controller = None
        self.enable_yaw_control = cfg.get("enable_yaw_control", False)
        self.human_action_dim = 4 if self.enable_yaw_control else 3
        self.enable_lidar = cfg.env.get("enable_lidar", True)

        # LiDAR params
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (
            max(-89., cfg.sensor.lidar_vfov[0]),
            min(89., cfg.sensor.lidar_vfov[1]),
        )
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360 / self.lidar_hres)

        # Map params
        self.map_range = cfg.env.map_range  # [x, y, z] half-extents
        self.platform_width = cfg.env.get("platform_width", 6.0)

        super().__init__(cfg, cfg.headless)

        # Drone init
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # Action limits
        self.max_action_vel = cfg.algo.actor.action_limit

        # Danger-aware tracking params
        self.danger_safe_dist = cfg.env.get("danger_safe_dist", 2.0)
        self.danger_relax_factor = cfg.env.get("danger_relax_factor", 0.8)

        # Configurable reward weights
        rw = cfg.env.get("reward_weights", {})
        self.w_tracking = rw.get("tracking", 3.0)
        self.w_safety = rw.get("safety", 5.0)
        self.w_smooth = rw.get("smoothness", 0.1)
        self.w_height = rw.get("height", 1.0)
        self.w_crash = rw.get("crash", -10.0)
        self.height_margin = cfg.env.get("height_margin", 1.0)

        # User model
        self.user_model = UserModelDiverse(
            num_envs=self.num_envs,
            cfg=cfg,
        )
        self.seed = cfg.get("seed", 0)

        # State buffers
        with torch.device(self.device):
            self.root_state = torch.zeros(self.num_envs, 1, 17)
            self.start_pos = torch.zeros(self.num_envs, 3)
            self.agent_action = torch.zeros(self.num_envs, 3)
            self.prev_action_command = torch.zeros(self.num_envs, 3)
            self.agent_action_original = torch.zeros(self.num_envs, 3)
            self.prev_human_action = torch.zeros(self.num_envs, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)

        self.common_step_counter = 0
        self.disable_visualization = False
        self.render_lidar = True
        self.viz_traj_human = []
        self.viz_traj_agent = []
        self.viz_human_pos = None

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _design_scene(self):
        self.drone, self.controller = MultirotorBase.make(
            self.cfg.drone.model_name,
            self.cfg.drone.controller_name,
            self.device,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 5.0)])[0]

        # Lighting
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

        # Ground plane
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Open obstacle field (no tunnel walls)
        sx, sy = self.map_range[0], self.map_range[1]
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(2 * sx, 2 * sy),
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
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(1.0, 1.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="choice",
                        obstacle_width_range=tuple(
                            self.cfg.env.get("obstacle_width_range", [0.4, 1.1])
                        ),
                        obstacle_height_range=tuple(
                            self.cfg.env.get("obstacle_height_range", [8.0, 20.0])
                        ),
                        platform_width=self.platform_width,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        # LiDAR
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)
        if self.enable_lidar:
            ray_caster_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.BpearlPatternCfg(
                    horizontal_res=self.lidar_hres,
                    vertical_ray_angles=torch.linspace(
                        *self.lidar_vfov, self.lidar_vbeams
                    ),
                ),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        else:
            self.lidar = None

        return ["/World/ground"]

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------
    def _set_specs(self):
        drone_state_dim = 11  # vel_b(3) + ang_vel_b(3) + quat(4) + z_normalized(1)

        obs_dict = {
            "state": Unbounded((drone_state_dim,), device=self.device),
            "human_action": Unbounded((self.human_action_dim,), device=self.device),
        }
        if self.enable_lidar:
            obs_dict["lidar"] = Unbounded(
                (1, self.lidar_hbeams, self.lidar_vbeams), device=self.device
            )

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {"observation": Composite(obs_dict)}
                ).expand(self.num_envs)
            },
            shape=[self.num_envs],
            device=self.device,
        )

        self.action_spec = Composite(
            {
                "agents": Composite(
                    {"action": Unbounded((self.human_action_dim,), device=self.device)}
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite(
            {"agents": Composite({"reward": Unbounded((1,))})}
        ).expand(self.num_envs).to(self.device)

        self.done_spec = Composite(
            {
                "done": Categorical(2, (1,), dtype=torch.bool),
                "terminated": Categorical(2, (1,), dtype=torch.bool),
                "truncated": Categorical(2, (1,), dtype=torch.bool),
            }
        ).expand(self.num_envs).to(self.device)

        stats_spec = Composite(
            {
                "return": Unbounded(1),
                "episode_len": Unbounded(1),
                "collision": Unbounded(1),
                "above_bound": Unbounded(1),
                "below_bound": Unbounded(1),
                "terminated": Unbounded(1),
                "truncated": Unbounded(1),
                # Pareto metrics
                "tracking_error": Unbounded(1),
                "tracking_error_sum": Unbounded(1),
                "intervention_norm": Unbounded(1),
                "intervention_norm_sum": Unbounded(1),
                # Diagnostics
                "diag_reward": Unbounded(1),
                "diag_reward_tracking": Unbounded(1),
                "diag_reward_safety": Unbounded(1),
                "diag_penalty_smooth": Unbounded(1),
                "diag_penalty_height": Unbounded(1),
                "diag_danger_level": Unbounded(1),
                # Debug vectors
                "debug_vec_world": Unbounded(3),
                "debug_vec_policy": Unbounded(3),
                "debug_vec_target": Unbounded(3),
                "debug_pos_world": Unbounded(3),
            }
        ).expand(self.num_envs).to(self.device)

        info_spec = Composite(
            {
                "drone_state": Unbounded((self.drone.n, 13), device=self.device),
            }
        ).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        sx, sy, sz = self.map_range
        K = len(env_ids)
        pos = torch.zeros(K, 1, 3, device=self.device)

        if self.training:
            # Spawn within obstacle-free platform at map center
            spawn_radius = self.platform_width / 2.0
            pos[:, 0, 0] = (torch.rand(K, device=self.device) * 2 - 1) * spawn_radius
            pos[:, 0, 1] = (torch.rand(K, device=self.device) * 2 - 1) * spawn_radius
            pos[:, 0, 2] = 3.0 + torch.rand(K, device=self.device) * 4.0  # 3-7m height
        else:
            pos[:, 0, 0] = 0.0
            pos[:, 0, 1] = 0.0
            pos[:, 0, 2] = 5.0

        self.start_pos[env_ids] = pos[:, 0, :].clone()

        # Random yaw for direction invariance
        rpy = torch.zeros(K, 3, device=self.device)
        if self.training:
            rpy[:, 2] = torch.rand(K, device=self.device) * 2 * math.pi - math.pi
        rot = euler_to_quaternion(rpy)

        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.agent_action[env_ids] = 0.
        self.prev_action_command[env_ids] = 0.
        self.prev_human_action[env_ids] = 0.

        if self.training:
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids)
        else:
            self.user_model.reset(pos=pos, quat=rot, env_ids=env_ids, seed=self.seed)

        self.stats[env_ids] = 0.

        self.max_episode_per_env = torch.full(
            (self.num_envs,),
            float(self.max_episode_length),
            dtype=torch.float,
            device=self.device,
        )

        # Height constraints
        self.height_range[env_ids, 0, 0] = 0.5 * sz  # min
        self.height_range[env_ids, 0, 1] = 1.5 * sz  # max

        if 0 in env_ids:
            self.viz_traj_agent = []
            self.viz_traj_human = []
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            self.viz_human_pos = pos[idx, 0].clone()

    # ------------------------------------------------------------------
    # Pre/Post sim
    # ------------------------------------------------------------------
    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.prev_action_command[:] = self.agent_action.clone()

        action_command = tensordict[("agents", "action")]
        if action_command.ndim == 3:
            action_command = action_command.squeeze(1)
        self.agent_action[:] = action_command.clone()
        self.agent_action_original[:] = action_command.clone()

        drone_state = tensordict[("info", "drone_state")][..., :13]

        if self.enable_yaw_control and action_command.shape[-1] == 4:
            target_vel = action_command[..., :3].unsqueeze(1)
            target_yaw = action_command[..., 3:4].unsqueeze(1) * math.pi
        else:
            target_vel = action_command[..., :3].unsqueeze(1)
            target_yaw = None

        actions = self.controller(
            root_state=drone_state,
            target_vel=target_vel,
            target_yaw=target_yaw,
        )
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        profiler = get_profiler()
        with profiler.timer("env/_post_sim_step"):
            if self.enable_lidar:
                self.lidar.update(self.dt)
        self.common_step_counter += 1

    # ------------------------------------------------------------------
    # Observation & Reward
    # ------------------------------------------------------------------
    def _compute_state_and_obs(self):
        profiler = get_profiler()
        profiler.start("env/_compute_state_and_obs")

        self.root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = self.root_state[..., :13]

        # ---------- LiDAR ----------
        if self.enable_lidar:
            self.lidar_scan = self.lidar_range - (
                (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
                .norm(dim=-1)
                .clamp_max(self.lidar_range)
                .reshape(self.num_envs, 1, *self.lidar_resolution)
            )
            self.lidar_scan = self.lidar_scan / self.lidar_range
        else:
            self.lidar_scan = None

        # ---------- Drone state ----------
        drone_pos_w = self.root_state[..., :3].squeeze(1)
        drone_vel_w = self.root_state[..., 7:10].squeeze(1)
        drone_ang_vel_w = self.root_state[..., 10:13].squeeze(1)
        drone_orientation_q = self.root_state[..., 3:7].squeeze(1)

        vel_b = quat_rotate_inverse(drone_orientation_q, drone_vel_w)
        ang_vel_b = quat_rotate_inverse(drone_orientation_q, drone_ang_vel_w)

        # Normalized height: (z - h_mid) / (h_range/2), ~[-1, 1] in safe zone
        h_min_obs = self.height_range[:, 0, 0]
        h_max_obs = self.height_range[:, 0, 1]
        h_mid = (h_min_obs + h_max_obs) / 2.0
        h_half = (h_max_obs - h_min_obs) / 2.0
        z_normalized = ((drone_pos_w[:, 2] - h_mid) / h_half.clamp(min=0.1)).unsqueeze(-1)

        drone_state_b = torch.cat([vel_b, ang_vel_b, drone_orientation_q, z_normalized], dim=-1)

        # ---------- Human input ----------
        human_actions_local = torch.zeros(
            self.num_envs, self.human_action_dim, device=self.device
        )
        if getattr(self, "manual_mode", False):
            human_actions_local = self.manual_action.clone()
        else:
            with profiler.timer("env/user_model_step"):
                human_actions_local, _ = self.user_model.step(
                    drone_state_b, drone_pos_w
                )

        # ---------- Observation ----------
        obs = {
            "state": drone_state_b,
            "human_action": human_actions_local,
        }
        if self.enable_lidar:
            obs["lidar"] = self.lidar_scan

        # ================================================================
        #                      REWARD COMPUTATION
        # ================================================================
        profiler.start("env/reward_calculation")

        prev_human_action_b = self.prev_human_action.clone()
        target_vel_w = vec_to_world(
            prev_human_action_b,
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True,
        )

        # ---- (a) Safety barrier penalty ----
        r_safety = torch.zeros(self.num_envs, 1, device=self.device)
        danger_level = torch.zeros(self.num_envs, 1, device=self.device)

        if self.enable_lidar:
            ray_vecs_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
            ray_dists = ray_vecs_w.norm(dim=-1).clamp_max(self.lidar_range)
            ray_dirs_w = ray_vecs_w / (ray_dists.unsqueeze(-1) + 1e-6)

            min_dist_to_obs, _ = ray_dists.min(dim=-1, keepdim=True)

            # Danger level: smooth [0, 1] — 1 = very close to obstacle
            danger_level = torch.exp(-min_dist_to_obs / self.danger_safe_dist)

            # Directional cone: velocity direction
            cone_threshold = 0.866  # cos(30°)
            cur_vel_norm = drone_vel_w.norm(dim=-1, keepdim=True)
            cur_vel_dir = drone_vel_w / (cur_vel_norm + 1e-6)
            cos_sim_vel = (ray_dirs_w * cur_vel_dir.unsqueeze(1)).sum(dim=-1)
            mask_vel = cos_sim_vel > cone_threshold
            dist_in_vel_cone = torch.where(
                mask_vel, ray_dists, torch.full_like(ray_dists, self.lidar_range)
            )
            dist_to_cur_vel_dir, _ = dist_in_vel_cone.min(dim=-1, keepdim=True)
            is_moving = (cur_vel_norm > 0.1).float()
            dist_to_cur_vel_dir = (
                is_moving * dist_to_cur_vel_dir
                + (1.0 - is_moving) * self.lidar_range
            )

            # Directional cone: command direction
            target_vel_norm = target_vel_w.norm(dim=-1, keepdim=True)
            target_vel_dir = target_vel_w / (target_vel_norm + 1e-6)
            cos_sim_cmd = (ray_dirs_w * target_vel_dir.unsqueeze(1)).sum(dim=-1)
            mask_cmd = cos_sim_cmd > cone_threshold
            dist_in_cmd_cone = torch.where(
                mask_cmd, ray_dists, torch.full_like(ray_dists, self.lidar_range)
            )
            dist_to_cmd_dir, _ = dist_in_cmd_cone.min(dim=-1, keepdim=True)
            has_cmd = (target_vel_norm > 0.1).float()
            dist_to_cmd_dir = (
                has_cmd * dist_to_cmd_dir + (1.0 - has_cmd) * self.lidar_range
            )

            # Exponential barrier penalties
            safe_zone = self.lidar_range
            r_safety_scale = 1.0
            p_min = torch.exp(-min_dist_to_obs / r_safety_scale)
            p_vel = torch.exp(-dist_to_cur_vel_dir / r_safety_scale)
            p_cmd = torch.exp(-dist_to_cmd_dir / r_safety_scale)

            mask_min_zone = (min_dist_to_obs < safe_zone).float()
            mask_vel_zone = (dist_to_cur_vel_dir < safe_zone).float()
            mask_cmd_zone = (dist_to_cmd_dir < safe_zone).float()

            r_safety = -(
                0.2 * (p_min * mask_min_zone)
                + 0.4 * (p_vel * mask_vel_zone)
                + 0.4 * (p_cmd * mask_cmd_zone)
            )

        # ---- (b) Danger-aware tracking reward ----
        # Compare CURRENT policy output (world frame) vs human command (world frame)
        policy_vel_w = vec_to_world(
            self.agent_action[:, :3],
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True,
        )
        tracking_error_vec = policy_vel_w - target_vel_w
        tracking_error_sq = (tracking_error_vec ** 2).sum(dim=-1, keepdim=True)

        # Danger-aware weight: relax tracking when dangerous
        tracking_weight = 1.0 - self.danger_relax_factor * danger_level
        r_tracking = -tracking_error_sq * tracking_weight

        # ---- (c) Smoothness penalty ----
        action_diff = (
            self.agent_action - self.prev_action_command
        ).norm(dim=-1, keepdim=True)
        penalty_smoothness = (action_diff / self.max_action_vel) ** 2

        # ---- (d) Height penalty ----
        h_min, h_max = self.height_range[..., 0], self.height_range[..., 1]
        z = self.drone.pos[..., 2:3].reshape(self.num_envs, 1)
        height_excess_up = (z - (h_max + self.height_margin)).clamp(min=0.0)
        height_excess_down = ((h_min - self.height_margin) - z).clamp(min=0.0)
        penalty_height = height_excess_up ** 2 + height_excess_down ** 2

        # ---- Total reward (configurable weights) ----
        self.reward = (
            self.w_tracking * r_tracking
            + self.w_safety * r_safety
            - self.w_smooth * penalty_smoothness
            - self.w_height * penalty_height
        )

        profiler.stop("env/reward_calculation")

        # ================================================================
        #                    TERMINATION CONDITIONS
        # ================================================================
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > self.map_range[2] * 2.0 + 1.0

        if self.enable_lidar:
            static_collision = (
                einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max")
                > (1.0 - 0.3 / self.lidar_range)
            )
        else:
            static_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        collision = static_collision

        # Map boundary escape: drone too far from center
        oob_xy = (
            (drone_pos_w[:, 0:1].abs() > self.map_range[0] * 0.95)
            | (drone_pos_w[:, 1:2].abs() > self.map_range[1] * 0.95)
        )

        self.terminated = below_bound | above_bound | collision | oob_xy
        timeout_truncate = (self.progress_buf >= self.max_episode_per_env).unsqueeze(-1)
        self.truncated = timeout_truncate

        # Terminal penalty
        crashed_mask = (collision | above_bound | below_bound | oob_xy) & ~self.truncated
        self.reward[crashed_mask] += self.w_crash

        # NOTE: prev_human_action update moved AFTER stats block to avoid
        # stale reference in intervention_norm computation (L682).

        # ================================================================
        #                        VISUALIZATION
        # ================================================================
        profiler.start("env/visualization")
        if self._should_render(0) and not self.disable_visualization:
            self.debug_draw.clear()
            viz_env_id = 0
            VIZ_VEL_SCALE = 0.5
            view_pos = drone_pos_w[viz_env_id]

            camera_mode = getattr(self, '_camera_view_mode', 'follow')
            if camera_mode == 'global':
                set_camera_view(
                    eye=torch.tensor([-3.0, 0.0, 32.0]),
                    target=torch.tensor([0.0, 0.0, 0.0]),
                )
            else:
                eye_vel_offset = (
                    -drone_vel_w[viz_env_id]
                    * torch.tensor([2.0, 2.0, 0.0], device=self.device)
                )
                set_camera_view(
                    eye=view_pos.cpu()
                    + torch.as_tensor(self.cfg.viewer.eye)
                    + eye_vel_offset.cpu(),
                    target=view_pos.cpu() + torch.as_tensor(self.cfg.viewer.lookat),
                )

            if self.render_lidar and self.enable_lidar:
                v = (
                    self.lidar.data.ray_hits_w[viz_env_id] - view_pos
                ).reshape(*self.lidar_resolution, 3)
                self.debug_draw.vector(
                    view_pos.expand_as(v[:, 0]),
                    v[:, 0],
                    color=(1, 0, 1, 0.5),
                    size=1.0,
                )
                self.debug_draw.vector(
                    view_pos.expand_as(v[:, -1]),
                    v[:, -1],
                    color=(1, 0, 1, 0.5),
                    size=1.0,
                )

            # Drone velocity (red)
            self.debug_draw.vector(
                x=view_pos,
                v=drone_vel_w[viz_env_id] * VIZ_VEL_SCALE,
                color=(1, 0, 0, 1),
                size=2.0,
            )
            # Human target velocity (yellow)
            self.debug_draw.vector(
                x=view_pos,
                v=target_vel_w[viz_env_id] * VIZ_VEL_SCALE,
                color=(1, 1, 0, 1),
                size=2.0,
            )

            # Trajectories
            curr_pos = view_pos.clone()
            self.viz_traj_agent.append(curr_pos)
            if self.viz_human_pos is None:
                self.viz_human_pos = curr_pos.clone()
            self.viz_human_pos += target_vel_w[viz_env_id] * self.dt
            self.viz_traj_human.append(self.viz_human_pos.clone())

            max_traj_len = 1000
            if len(self.viz_traj_agent) > max_traj_len:
                self.viz_traj_agent.pop(0)
                self.viz_traj_human.pop(0)

            if len(self.viz_traj_agent) > 1:
                pts = torch.stack(self.viz_traj_agent)
                self.debug_draw.vector(
                    pts[:-1], pts[1:] - pts[:-1], color=(0, 0, 1, 1), size=2.0
                )
                pts_h = torch.stack(self.viz_traj_human)
                self.debug_draw.vector(
                    pts_h[:-1], pts_h[1:] - pts_h[:-1], color=(0, 1, 0, 1), size=2.0
                )
        profiler.stop("env/visualization")

        # ================================================================
        #                     STATISTICS / LOGGING
        # ================================================================
        # Tracking error: ||actual_vel - human_vel|| (world frame)
        tracking_err_norm = (drone_vel_w - target_vel_w).norm(dim=-1, keepdim=True)
        # Intervention: ||policy_output - human_input|| (normalized action space)
        intervention = (
            self.agent_action_original - self.prev_human_action
        ).norm(dim=-1, keepdim=True) / self.max_action_vel

        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["collision"] = collision.float()
        self.stats["above_bound"] = above_bound.float()
        self.stats["below_bound"] = below_bound.float()
        self.stats["terminated"] = self.terminated.float()
        self.stats["truncated"] = self.truncated.float()
        self.stats["tracking_error"] = tracking_err_norm
        self.stats["tracking_error_sum"] += tracking_err_norm
        self.stats["intervention_norm"] = intervention
        self.stats["intervention_norm_sum"] += intervention
        self.stats["diag_reward"] = self.reward
        self.stats["diag_reward_tracking"] = r_tracking
        self.stats["diag_reward_safety"] = r_safety
        self.stats["diag_penalty_smooth"] = penalty_smoothness
        self.stats["diag_penalty_height"] = penalty_height

        # Update prev_human_action AFTER stats (intervention_norm needs step-t value)
        self.prev_human_action = human_actions_local.clone()
        self.stats["diag_danger_level"] = danger_level
        self.stats["debug_vec_world"] = drone_vel_w
        self.stats["debug_vec_policy"] = self.agent_action_original
        human_action_w = vec_to_world(
            human_actions_local[:, :3],
            drone_orientation_q,
            orientation_only=True,
            yaw_only=True,
        )
        self.stats["debug_vec_target"] = human_action_w
        self.stats["debug_pos_world"] = drone_pos_w

        profiler.stop("env/_compute_state_and_obs")

        if torch.isnan(self.reward).any():
            raise ValueError("NaN in reward")

        return TensorDict(
            {
                "agents": TensorDict({"observation": obs}, [self.num_envs]),
                "stats": self.stats.clone(),
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        return TensorDict(
            {
                "agents": {"reward": self.reward},
                "done": self.terminated | self.truncated,
                "terminated": self.terminated,
                "truncated": self.truncated,
            },
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Utility methods (same interface as env_tunnel)
    # ------------------------------------------------------------------
    def set_visualization(self, enabled: bool):
        self.disable_visualization = not enabled

    def set_manual_mode(self, enabled: bool):
        self.manual_mode = enabled
        self.manual_action = torch.zeros(self.num_envs, 3, device=self.device)

    def set_manual_action(self, action: torch.Tensor):
        if self.manual_mode:
            self.manual_action[:] = action

    def set_camera_view_mode(self, mode: str):
        if mode not in ['global', 'follow']:
            raise ValueError(f"Invalid camera mode: {mode}")
        self._camera_view_mode = mode

    def get_camera_view_mode(self) -> str:
        return getattr(self, '_camera_view_mode', 'follow')

    def set_envs_visibility(self, visible_env_ids=None):
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
