# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Compare IPC vs RL in Tunnel Environment")
parser.add_argument("--model_type", type=str, default="ConstrainedBeta",
                    choices=["Simple", "Residual", "Constrained", "ConstrainedBeta"])
parser.add_argument("--checkpoint", type=str,
                    default="/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/shared_demos/ckpts/260331/checkpoint_final.pt")
parser.add_argument("--ipc_config", type=str, default=None,
                    help="Path to IPC config YAML (default: ipc_config.yaml in same dir)")
parser.add_argument("--no_sfc", action="store_true", help="Disable CIRI SFC for IPC")
parser.add_argument("--state_dim", type=int, default=10, choices=[10, 11])
parser.add_argument("--num_obstacles", type=int, default=60,
                    help="Number of obstacles in the tunnel terrain")
parser.add_argument("--num_trials", type=int, default=5,
                    help="Number of trials per controller for statistical significance")
parser.add_argument("--num_frames", type=int, default=1600,
                    help="Number of frames per trial")
parser.add_argument("--start_seed", type=int, default=42,
                    help="Starting seed for trials (trial i uses start_seed + i)")
parser.add_argument("--success_x", type=float, default=12.0,
                    help="X coordinate threshold for successful tunnel traversal")
parser.add_argument("--no_viz", action="store_true",
                    help="(Deprecated) Same as --viz none")
parser.add_argument("--viz", type=str, default="first",
                    choices=["none", "first", "all"],
                    help="Visualization mode: none/first/all (default: first)")
parser.add_argument("--viz_fps", type=int, default=20,
                    help="Visualization animation frame rate")
parser.add_argument("--debug", action="store_true",
                    help="Print per-trial/per-frame details to console (default: only final results)")
parser.add_argument("--ipc_speed_profile", type=str, default=None,
                    choices=["fast", "balanced"],
                    help="Apply IPC speed tuning preset: fast (~500+ FPS) or balanced (~100-300 FPS)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports (after SimulationApp) ---
import sys
import os
import json
import time
import torch
import numpy as np
import logging
from datetime import datetime
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaacsim.core.utils.viewports import set_camera_view

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate_inverse, quat_rotate
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(__file__))
from ipc.ipc_controller import IPCController, load_config
from trajectory_visualizer import (
    FlightDataRecorder, TrajectoryVisualizer, obstacles_to_info,
)

shared_demos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared_demos"))
sys.path.insert(0, shared_demos_dir)
from srlc_model import load_srlc_model, MockConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/src/core")))
from user_model_tunnely import UserModelTunnel

# Reuse terrain / obstacle extraction from shared module
from tunnel_terrain import (
    extract_obstacles_from_heightfield, quat_to_rotation_matrix,
    tunnel_obstacles_terrain, HfTunnelObstaclesTerrainCfg,
    INIT_POS, INIT_QUAT, TERRAIN_LEGACY_SEED,
    clear_captured_tiles, get_captured_heightfield,
)

drone_model_name = "Hummingbird"
drone_controller_name = "LeePositionController"


def setup_logger(log_dir="logs", verbose=False):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"compare_{timestamp}.log")
    logger = logging.getLogger("compare_ipc_rl")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


class MetricsCollector:
    """Collect per-trial metrics for IPC vs RL comparison."""

    # Tunnel map half-extents (from ipc_config.yaml map_size / 2)
    MAP_HALF_X = 17.0
    MAP_HALF_Y = 11.0
    MAP_Z_MAX = 12.0

    def __init__(self, success_x: float = 10.0):
        self.success_x = success_x
        self.positions = []
        self.velocities = []
        self.human_vels_w = []
        self.ctrl_vels_w = []
        self.collision_frames = 0
        self.total_frames = 0
        self.inference_times = []
        self._prev_vel = None
        self._prev_acc = None
        self._reached_goal = False
        self._crashed = False
        self._crash_reason = ""
        # CIRI SFC stats (only meaningful for IPC)
        self.sfc_attempts = 0
        self.sfc_successes = 0

    def update(self, pos, vel, human_vel_w, ctrl_vel_w, is_collision, dt, inference_time=0.0):
        self.positions.append(pos.copy())
        self.velocities.append(vel.copy())
        self.human_vels_w.append(human_vel_w.copy())
        self.ctrl_vels_w.append(ctrl_vel_w.copy())
        self.total_frames += 1

        if is_collision:
            self.collision_frames += 1

        if inference_time > 0:
            self.inference_times.append(inference_time)

        # Check success: drone reached end of tunnel
        if pos[0] > self.success_x:
            self._reached_goal = True

        # Check crash: multiple conditions (skip if already succeeded)
        if not self._crashed and not self._reached_goal:
            if pos[2] < 0.5:
                self._crashed = True
                self._crash_reason = f"fell below ground (z={pos[2]:.2f})"
            elif abs(pos[0]) > self.MAP_HALF_X:
                self._crashed = True
                self._crash_reason = f"out of bounds x (x={pos[0]:.2f})"
            elif abs(pos[1]) > self.MAP_HALF_Y:
                self._crashed = True
                self._crash_reason = f"out of bounds y (y={pos[1]:.2f})"
            elif pos[2] > self.MAP_Z_MAX:
                self._crashed = True
                self._crash_reason = f"out of bounds z (z={pos[2]:.2f})"

        self._prev_vel = vel.copy()

    def summary(self) -> dict:
        positions = np.array(self.positions)
        vels = np.array(self.velocities)
        human_vels = np.array(self.human_vels_w)
        ctrl_vels = np.array(self.ctrl_vels_w)

        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)) if len(positions) > 1 else 0.0

        # Max X reached (tunnel progress)
        max_x = float(np.max(positions[:, 0])) if len(positions) > 0 else 0.0

        # Success: reached goal and not crashed
        success = self._reached_goal and not self._crashed

        # Collision rate
        collision_rate = self.collision_frames / max(self.total_frames, 1)

        # Trajectory following: cosine similarity between user vel and ctrl vel
        h_norms = np.linalg.norm(human_vels, axis=1, keepdims=True)
        c_norms = np.linalg.norm(ctrl_vels, axis=1, keepdims=True)
        # Only compute when user is actually commanding movement
        active_mask = h_norms.squeeze() > 0.1
        if active_mask.any():
            h_active = human_vels[active_mask]
            c_active = ctrl_vels[active_mask]
            h_n = np.linalg.norm(h_active, axis=1, keepdims=True).clip(1e-8)
            c_n = np.linalg.norm(c_active, axis=1, keepdims=True).clip(1e-8)
            cos_sim = np.sum((h_active / h_n) * (c_active / c_n), axis=1)
            tracking_cosine_mean = float(np.mean(cos_sim))
            tracking_cosine_std = float(np.std(cos_sim))

            # Magnitude ratio: |ctrl| / |human|
            mag_ratio = np.linalg.norm(c_active, axis=1) / np.linalg.norm(h_active, axis=1).clip(1e-8)
            tracking_mag_mean = float(np.mean(mag_ratio))

            # Velocity error
            tracking_error = np.linalg.norm(c_active - h_active, axis=1)
            tracking_error_mean = float(np.mean(tracking_error))
        else:
            tracking_cosine_mean = 0.0
            tracking_cosine_std = 0.0
            tracking_mag_mean = 0.0
            tracking_error_mean = 0.0

        # Inference performance
        inf_times = np.array(self.inference_times) if self.inference_times else np.array([0.0])
        inference_mean_ms = float(np.mean(inf_times) * 1000)
        inference_std_ms = float(np.std(inf_times) * 1000)
        inference_max_ms = float(np.max(inf_times) * 1000)
        inference_p50_ms = float(np.percentile(inf_times, 50) * 1000)
        inference_p95_ms = float(np.percentile(inf_times, 95) * 1000)
        # Effective controller FPS (how fast the controller alone could run)
        ctrl_fps = float(1.0 / np.mean(inf_times)) if np.mean(inf_times) > 0 else 0.0

        return {
            "success": success,
            "crashed": self._crashed,
            "crash_reason": self._crash_reason,
            "reached_goal": self._reached_goal,
            "max_x_reached": max_x,
            "total_frames": self.total_frames,
            "collision_frames": self.collision_frames,
            "collision_rate": collision_rate,
            "path_length_m": float(path_length),
            "tracking_cosine_mean": tracking_cosine_mean,
            "tracking_cosine_std": tracking_cosine_std,
            "tracking_mag_ratio_mean": tracking_mag_mean,
            "tracking_error_mean": tracking_error_mean,
            "inference_mean_ms": inference_mean_ms,
            "inference_std_ms": inference_std_ms,
            "inference_max_ms": inference_max_ms,
            "inference_p50_ms": inference_p50_ms,
            "inference_p95_ms": inference_p95_ms,
            "ctrl_fps": ctrl_fps,
            "avg_speed": float(np.mean(np.linalg.norm(vels, axis=1))),
            "sfc_attempts": self.sfc_attempts,
            "sfc_successes": self.sfc_successes,
            "sfc_success_rate": self.sfc_successes / max(self.sfc_attempts, 1),
        }


def check_collision_lidar(lidar, lidar_range, threshold=0.3):
    """Collision detection via LiDAR minimum distance."""
    if lidar is None:
        return False
    ray_hits = lidar.data.ray_hits_w
    pos = lidar.data.pos_w
    dists = torch.norm(ray_hits - pos.unsqueeze(1), dim=-1)
    min_dist = dists.min().item()
    return min_dist < threshold


def main():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger, log_file = setup_logger(log_dir, verbose=args_cli.debug)
    logger.debug(f"Log file: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0
    headless = getattr(args_cli, 'headless', False)
    render = not headless

    logger.debug(f"Headless mode: {headless}")
    logger.debug(f"Num trials: {args_cli.num_trials}, Frames/trial: {args_cli.num_frames}")

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- Drone ---
    drone, controller = MultirotorBase.make(drone_model_name, drone_controller_name, device)
    drone.spawn(translations=torch.tensor([INIT_POS], device=device))

    # --- Lighting ---
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)
    sky_light_cfg = sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0)
    sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    # --- Terrain (Tunnel) ---
    terrain_cfg = TerrainImporterCfg(
        num_envs=1, env_spacing=0.0, prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0, size=(24.0, 12.0), border_width=5.0,
            num_rows=1, num_cols=1,
            horizontal_scale=0.1, vertical_scale=0.1,
            slope_threshold=0.75, use_cache=False, color_scheme="height",
            curriculum=False, difficulty_range=(0.0, 1.0),
            sub_terrains={
                "obstacles": HfTunnelObstaclesTerrainCfg(
                    size=(1.0, 1.0),
                    horizontal_scale=0.1, vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=args_cli.num_obstacles,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(8.0, 20.0),
                    platform_width=0,
                ),
            },
        ),
        visual_material=None, max_init_terrain_level=None,
        collision_group=-1, debug_vis=False,
    )
    # Clear capture buffer, then create terrain — the terrain function
    # records its actual heightfield so we get a guaranteed-exact match.
    clear_captured_tiles()
    np.random.seed(TERRAIN_LEGACY_SEED)
    terrain = terrain_cfg.class_type(terrain_cfg)

    # --- LiDAR ---
    LIDAR_RANGE = 4.0
    LIDAR_VFOV = (-10.0, 20.0)
    LIDAR_VBEAMS = 4
    LIDAR_HRES = 10.0
    drone_prim_path = f"/World/envs/env_0/{drone.name}_0/base_link"
    ray_caster_cfg = RayCasterCfg(
        prim_path=drone_prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_res=LIDAR_HRES,
            vertical_ray_angles=torch.linspace(LIDAR_VFOV[0], LIDAR_VFOV[1], LIDAR_VBEAMS),
        ),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
    )
    lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)

    sim.reset()
    drone.initialize()

    # --- UserModelTunnel ---
    mock_cfg = MockConfig(device)
    user_model = UserModelTunnel(num_envs=1, cfg=mock_cfg, logger=logger)
    logger.debug("UserModelTunnel enabled for reproducible comparison")

    # --- IPC Controller ---
    ipc_config_path = args_cli.ipc_config or os.path.join(os.path.dirname(__file__), "ipc_config.yaml")
    ipc_cfg = load_config(ipc_config_path)
    if args_cli.no_sfc:
        ipc_cfg.use_sfc = False

    # Apply speed profile overrides
    if args_cli.ipc_speed_profile == "fast":
        ipc_cfg.max_sfc_steps = 1
        ipc_cfg.mpc.horizon = 5
        ipc_cfg.replan_interval = 40
        ipc_cfg.max_obstacle_points = 80
        ipc_cfg.obstacle_query_radius = 3.0
        ipc_cfg.astar.resolution = 1.0
        ipc_cfg.astar.timeout = 0.02
        logger.info("IPC speed profile: FAST (horizon=5, sfc_steps=1)")
    elif args_cli.ipc_speed_profile == "balanced":
        ipc_cfg.max_sfc_steps = 2
        ipc_cfg.mpc.horizon = 8
        ipc_cfg.replan_interval = 30
        ipc_cfg.max_obstacle_points = 120
        ipc_cfg.obstacle_query_radius = 4.0
        logger.info("IPC speed profile: BALANCED (horizon=8, sfc_steps=2)")

    ipc = IPCController(cfg=ipc_cfg)

    hf, obstacles = extract_obstacles_from_heightfield(terrain_cfg.terrain_generator, tunnel_mode=True)
    # Prefer heightfield-based map for most accurate obstacle representation.
    gen_cfg = terrain_cfg.terrain_generator
    if hf is not None and hf.any():
        # Pixel 0 is at -size/2 (fence-post convention: N+1 pixels for N*h metres)
        origin = np.array([-gen_cfg.size[0] / 2.0, -gen_cfg.size[1] / 2.0, 0.0])
        ipc.build_map(heightfield=hf.astype(float),
                      horizontal_scale=gen_cfg.horizontal_scale,
                      vertical_scale=gen_cfg.vertical_scale,
                      origin=origin)
        logger.debug(f"IPC map built from heightfield ({len(obstacles)} obstacle clusters)")
    elif obstacles:
        ipc.build_map(obstacles=obstacles)
        logger.debug(f"IPC map built with {len(obstacles)} obstacle clusters")
    else:
        logger.warning("No obstacles extracted — IPC runs without occupancy grid")

    # --- RL Policy ---
    action_dim = 3
    state_dim = args_cli.state_dim
    policy = load_srlc_model(
        args_cli.model_type, args_cli.checkpoint, device,
        action_dim=action_dim, enable_lidar=True, state_dim=state_dim,
    )
    if policy is None:
        logger.error("Failed to load RL model!")
        simulation_app.close()
        return

    NUM_FRAMES = args_cli.num_frames
    NUM_TRIALS = args_cli.num_trials

    init_pos_t = torch.tensor([INIT_POS], device=device)
    init_quat_t = torch.tensor([INIT_QUAT], device=device)

    def reset_drone(trial_seed):
        """Reset drone + user model for a new trial."""
        env_ids = torch.tensor([0], device=device)
        drone._reset_idx(env_ids, train=False)
        drone.set_world_poses(
            init_pos_t.unsqueeze(1), init_quat_t.unsqueeze(1), env_ids
        )
        drone.set_velocities(torch.zeros(1, 1, 6, device=device), env_ids)

        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        user_model.reset(
            torch.tensor([[-7.0, 0.0, 5.0]], device=device),
            init_quat_t,
            env_ids,
        )

    # ===== Multi-trial runner =====
    all_results = {"IPC": [], "RL": []}
    # Flight data recorders for inline visualization (keyed by trial_idx)
    all_recorders: dict = {"IPC": {}, "RL": {}}
    viz_mode = "none" if args_cli.no_viz else args_cli.viz

    # Prepare obstacle info for visualizer
    obs_info = obstacles_to_info(obstacles) if obstacles else []

    # Data directory for flight recordings (always saved for offline rendering)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = os.path.join(log_dir, f"flight_data_{timestamp}")
    os.makedirs(data_dir, exist_ok=True)
    results_path = os.path.join(log_dir, f"compare_results_{timestamp}.json")
    # Save obstacle info for offline rendering
    obs_meta = {
        "obstacles": obstacles if obstacles else [],
        "terrain_size": list(terrain_cfg.terrain_generator.size),
    }
    with open(os.path.join(data_dir, "obstacles.json"), 'w') as f:
        json.dump(obs_meta, f, default=lambda o: list(o) if isinstance(o, tuple) else o)

    def _save_results_incremental():
        """Save current results to JSON (called after each trial for crash-safety)."""
        output = {
            "config": {
                "num_trials": NUM_TRIALS,
                "num_frames": NUM_FRAMES,
                "start_seed": args_cli.start_seed,
                "num_obstacles": args_cli.num_obstacles,
                "success_x": args_cli.success_x,
                "model_type": args_cli.model_type,
                "checkpoint": args_cli.checkpoint,
                "use_sfc": not args_cli.no_sfc,
                "ipc_speed_profile": args_cli.ipc_speed_profile,
                "completed_trials": sum(len(v) for v in all_results.values()),
            },
            "data_dir": data_dir,
            "per_trial": all_results,
        }
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

    for trial_idx in range(NUM_TRIALS):
        for trial_name, use_ipc in [("IPC", True), ("RL", False)]:
            if not simulation_app.is_running():
                break

            trial_seed = args_cli.start_seed + trial_idx
            logger.debug(f"\n{'='*50}")
            logger.debug(f"Trial {trial_idx+1}/{NUM_TRIALS} — {trial_name} (seed={trial_seed})")
            logger.debug(f"{'='*50}")
            # Compact progress to stdout
            print(f"\r  [{trial_name}] {trial_idx+1}/{NUM_TRIALS}", end="", flush=True)

            reset_drone(trial_seed)
            metrics = MetricsCollector(success_x=args_cli.success_x)
            # Always record flight data (saved to disk for offline rendering)
            recorder = FlightDataRecorder(controller_type=trial_name)

            # Attach occupancy grid slice to IPC recorder for diagnostic rendering
            if use_ipc and ipc is not None:
                occ_slice = ipc.get_occupancy_2d_slice(
                    z=ipc_cfg.altitude_hold, use_inflated=False)
                recorder.set_occupancy_grid(occ_slice)
            # Only keep in memory for inline visualization
            need_inline_viz = (viz_mode == "all") or (viz_mode == "first" and trial_idx == 0)
            prev_vel_w_np = np.zeros(3)
            prev_human_action = torch.zeros((1, action_dim), device=device)

            if use_ipc:
                ipc.reset()

            # Let physics stabilize
            for _ in range(10):
                sim.step(render=render)

            for frame in range(NUM_FRAMES):
                if not simulation_app.is_running():
                    break

                if sim.is_playing():
                    # 1. Drone state
                    root_state = drone.get_state()[..., :13]
                    current_pos = root_state[..., :3]
                    current_quat = root_state[..., 3:7]
                    current_vel_w = root_state[..., 7:10]
                    current_ang_vel_w = root_state[..., 10:13]

                    vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_vel_w.squeeze(1))
                    ang_vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_ang_vel_w.squeeze(1))
                    drone_state = torch.cat([vel_b, ang_vel_b, current_quat.squeeze(1)], dim=-1)

                    # 2. Human action (identical for both)
                    drone_pos_w = current_pos.squeeze(1)
                    human_action, _ = user_model.step(drone_state, drone_pos_w)
                    human_action_input = human_action[..., :action_dim]

                    # LiDAR
                    lidar.update(dt)

                    # 3. Controller step (timed)
                    t_start = time.perf_counter()

                    if use_ipc:
                        pos_np = current_pos.squeeze().cpu().numpy().astype(float)
                        vel_np = current_vel_w.squeeze().cpu().numpy().astype(float)
                        quat_np = current_quat.squeeze().cpu().numpy().astype(float)
                        acc_np = (vel_np - prev_vel_w_np) / dt
                        prev_vel_w_np = vel_np.copy()
                        rot_mat = quat_to_rotation_matrix(quat_np)
                        user_vel_body_np = human_action[..., :3].squeeze().cpu().numpy().astype(float)

                        ipc_vel = ipc.step(pos_np, vel_np, acc_np, user_vel_body_np, rot_mat)
                        ctrl_vel_w = ipc_vel
                        target_vel = torch.tensor(ipc_vel, dtype=torch.float32, device=device).reshape(1, 1, 3)
                    else:
                        ray_hits = lidar.data.ray_hits_w
                        pos_l = lidar.data.pos_w
                        dist = torch.norm(ray_hits - pos_l.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE)
                        scan = (LIDAR_RANGE - dist) / LIDAR_RANGE
                        lidar_obs = scan.reshape(1, 1, int(360 / LIDAR_HRES), LIDAR_VBEAMS)

                        obs = TensorDict({
                            "agents": TensorDict({
                                "observation": TensorDict({
                                    "state": drone_state,
                                    "human_action": prev_human_action,
                                    "lidar": lidar_obs,
                                }, batch_size=[1])
                            }, batch_size=[1])
                        }, batch_size=[1], device=device)

                        exploration_type = (ExplorationType.DETERMINISTIC
                                            if args_cli.model_type in ("Constrained", "ConstrainedBeta")
                                            else ExplorationType.MEAN)
                        with torch.no_grad(), set_exploration_type(exploration_type):
                            policy(obs)
                            model_action_w = obs["agents", "command"]

                        ctrl_vel_w = model_action_w.squeeze().cpu().numpy().astype(float)
                        target_vel = model_action_w.unsqueeze(1)

                    t_elapsed = time.perf_counter() - t_start

                    # Apply control (no yaw from user model tunnel)
                    action = controller(root_state=root_state, target_vel=target_vel, target_yaw=None)
                    drone.apply_action(action)

                    # Metrics
                    pos_np = current_pos.squeeze().cpu().numpy()
                    vel_np = current_vel_w.squeeze().cpu().numpy()
                    human_vel_w = quat_rotate(current_quat.squeeze(1), human_action[..., :3]).squeeze().cpu().numpy()
                    is_col = check_collision_lidar(lidar, LIDAR_RANGE, threshold=0.3)

                    metrics.update(pos_np, vel_np, human_vel_w, ctrl_vel_w, is_col, dt, t_elapsed)

                    # Record flight data for visualization
                    sfc_planes = ipc.get_last_sfc() if use_ipc else None
                    ref_path = ipc.get_ref_path() if use_ipc else None
                    lidar_hits_np = None
                    if not use_ipc:
                        lidar_hits_np = lidar.data.ray_hits_w.squeeze(0).cpu().numpy()
                    recorder.record_frame(
                        pos=pos_np, human_vel_w=human_vel_w,
                        ctrl_vel_w=ctrl_vel_w, is_collision=is_col,
                        sfc_planes=sfc_planes, ref_path=ref_path,
                        lidar_hits_w=lidar_hits_np,
                    )

                    prev_human_action = human_action_input.clone()

                    # Progress log (useful in headless mode)
                    if headless and frame % 100 == 0:
                        logger.debug(f"  [{trial_name}] frame {frame}/{NUM_FRAMES}, x={pos_np[0]:.2f}")

                    # Early termination on crash
                    if metrics._crashed:
                        logger.warning(f"  Crashed at frame {frame}: {metrics._crash_reason}")
                        break

                    # Early termination on success
                    if metrics._reached_goal:
                        logger.debug(f"  Reached goal at frame {frame}, x={pos_np[0]:.2f}")
                        break

                sim.step(render=render)

            # Record SFC stats for IPC trials
            if use_ipc:
                sfc_stats = ipc.get_sfc_stats()
                metrics.sfc_attempts = sfc_stats['attempts']
                metrics.sfc_successes = sfc_stats['successes']

            trial_summary = metrics.summary()
            all_results[trial_name].append(trial_summary)

            # Save flight data to disk (always)
            npz_filename = f"{trial_name.lower()}_trial{trial_idx:04d}.npz"
            npz_path = os.path.join(data_dir, npz_filename)
            recorder.save(npz_path)
            trial_summary["data_file"] = npz_filename
            trial_summary["trial_id"] = trial_idx
            trial_summary["trial_seed"] = trial_seed

            # Keep recorder in memory only if needed for inline visualization
            if need_inline_viz:
                all_recorders[trial_name][trial_idx] = recorder

            sfc_info = ""
            if use_ipc:
                sfc_info = f", sfc_rate={trial_summary['sfc_success_rate']:.1%}"
            logger.debug(f"  {trial_name} trial {trial_idx+1}: success={trial_summary['success']}, "
                        f"max_x={trial_summary['max_x_reached']:.2f}, "
                        f"col_rate={trial_summary['collision_rate']:.4f}, "
                        f"tracking_cos={trial_summary['tracking_cosine_mean']:.4f}, "
                        f"inference={trial_summary['inference_mean_ms']:.2f}ms{sfc_info}")

            # Incremental save — crash-safe, always has latest results
            _save_results_incremental()

    # ===== Aggregate and print results =====
    def aggregate(trials):
        """Compute mean ± std over trials."""
        if not trials:
            return {}
        keys = trials[0].keys()
        agg = {}
        for k in keys:
            vals = [t[k] for t in trials]
            if isinstance(vals[0], bool):
                agg[f"{k}_rate"] = sum(vals) / len(vals)
            elif isinstance(vals[0], (int, float)):
                arr = np.array(vals, dtype=float)
                agg[f"{k}_mean"] = float(np.mean(arr))
                agg[f"{k}_std"] = float(np.std(arr))
            else:
                agg[k] = vals
        return agg

    ipc_agg = aggregate(all_results["IPC"])
    rl_agg = aggregate(all_results["RL"])

    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS: IPC vs RL  ({NUM_TRIALS} trials, {NUM_FRAMES} frames/trial)")
    print(f"{'='*70}")

    # Key metrics table
    key_metrics = [
        ("Success Rate",           "success_rate",              "{:.1%}"),
        ("Collision Rate (mean)",  "collision_rate_mean",       "{:.4f}"),
        ("Collision Rate (std)",   "collision_rate_std",        "{:.4f}"),
        ("Max X Reached (mean)",   "max_x_reached_mean",       "{:.2f}"),
        ("Tracking Cosine (mean)", "tracking_cosine_mean_mean", "{:.4f}"),
        ("Tracking Cosine (std)",  "tracking_cosine_mean_std",  "{:.4f}"),
        ("Tracking Error (mean)",  "tracking_error_mean_mean",  "{:.4f}"),
        ("Mag Ratio (mean)",       "tracking_mag_ratio_mean_mean", "{:.4f}"),
        ("Ctrl FPS (mean)",        "ctrl_fps_mean",             "{:.1f}"),
        ("Ctrl FPS (std)",         "ctrl_fps_std",              "{:.1f}"),
        ("Latency mean ms",       "inference_mean_ms_mean",    "{:.2f}"),
        ("Latency p50 ms",        "inference_p50_ms_mean",     "{:.2f}"),
        ("Latency p95 ms",        "inference_p95_ms_mean",     "{:.2f}"),
        ("Latency max ms",        "inference_max_ms_mean",     "{:.2f}"),
        ("Path Length m (mean)",   "path_length_m_mean",        "{:.2f}"),
        ("Avg Speed (mean)",       "avg_speed_mean",            "{:.4f}"),
        ("SFC Success Rate (mean)","sfc_success_rate_mean",     "{:.1%}"),
        ("SFC Attempts (mean)",    "sfc_attempts_mean",         "{:.0f}"),
    ]

    header = f"{'Metric':<30} {'IPC':>14} {'RL':>14}"
    print(header)
    print("-" * 70)
    for label, key, fmt in key_metrics:
        iv = ipc_agg.get(key, "N/A")
        rv = rl_agg.get(key, "N/A")
        iv_str = fmt.format(iv) if isinstance(iv, (int, float)) else str(iv)
        rv_str = fmt.format(rv) if isinstance(rv, (int, float)) else str(rv)
        print(f"{label:<30} {iv_str:>14} {rv_str:>14}")

    print("=" * 70)

    # Save final results (with aggregated stats — overwrites incremental)
    output = {
        "config": {
            "num_trials": NUM_TRIALS,
            "num_frames": NUM_FRAMES,
            "start_seed": args_cli.start_seed,
            "num_obstacles": args_cli.num_obstacles,
            "success_x": args_cli.success_x,
            "model_type": args_cli.model_type,
            "checkpoint": args_cli.checkpoint,
            "use_sfc": not args_cli.no_sfc,
            "ipc_speed_profile": args_cli.ipc_speed_profile,
            "completed_trials": sum(len(v) for v in all_results.values()),
        },
        "data_dir": data_dir,
        "per_trial": all_results,
        "aggregated": {"IPC": ipc_agg, "RL": rl_agg},
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_path}")
    print(f"Flight data saved to: {data_dir}")
    print(f"  Render any trial offline:  python render_viz.py {results_path} --trial 0")

    # ===== Trajectory Visualization =====
    if viz_mode != "none" and all_recorders["IPC"] and all_recorders["RL"]:
        logger.debug("\nRendering trajectory visualizations...")
        viz_dir = os.path.join(log_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)

        # Tunnel bounds from terrain config
        gen_cfg = terrain_cfg.terrain_generator
        t_size = gen_cfg.size  # (width, length) per tile
        tx_half = t_size[0] / 2.0
        ty_half = t_size[1] / 2.0
        wall_inner_y = ty_half - 1.0  # 1m wall thickness

        viz = TrajectoryVisualizer(
            obstacles=obs_info,
            tunnel_x_range=(-tx_half - 1, tx_half + 1),
            tunnel_y_range=(-ty_half - 0.5, ty_half + 0.5),
            z_range=(0.0, 8.0),
            wall_y=(-wall_inner_y, wall_inner_y),
        )

        for tidx in sorted(set(all_recorders["IPC"].keys()) & set(all_recorders["RL"].keys())):
            trial_seed = args_cli.start_seed + tidx
            label = f"Trial {tidx+1} (seed={trial_seed})"

            # Side-by-side animated comparison
            anim_path = os.path.join(viz_dir, f"compare_trial{tidx+1}.mp4")
            try:
                viz.render_comparison_from_recorders(
                    ipc_recorder=all_recorders["IPC"][tidx],
                    rl_recorder=all_recorders["RL"][tidx],
                    output_path=anim_path,
                    fps=args_cli.viz_fps,
                    subsample=3,
                    trial_label=label,
                )
                logger.debug(f"  Animation saved: {anim_path}")
            except Exception as e:
                logger.warning(f"  Animation failed for trial {tidx+1}: {e}")

            # Static trajectory plot
            static_path = os.path.join(viz_dir, f"trajectory_trial{tidx+1}.png")
            try:
                viz.render_static_comparison(
                    all_recorders["IPC"][tidx],
                    all_recorders["RL"][tidx],
                    output_path=static_path,
                    trial_label=label,
                )
                logger.debug(f"  Static plot saved: {static_path}")
            except Exception as e:
                logger.warning(f"  Static plot failed for trial {tidx+1}: {e}")

        logger.debug(f"Visualizations saved to: {viz_dir}")
    elif viz_mode != "none":
        logger.debug("Skipping visualization — incomplete trial data")

    simulation_app.close()


if __name__ == "__main__":
    main()
