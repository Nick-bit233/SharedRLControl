# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher
import argparse

# 添加命令行参数
parser = argparse.ArgumentParser(description="Verify SRLC Model (Tunnel Environment)")
parser.add_argument("--model_type", type=str, default="ConstrainedBeta",
                    choices=["Auto", "Simple", "Residual", "Constrained", "ConstrainedBeta", "ConstrainedBetaLagrangian"],
                    help="Type of SRLC model to load")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt file)")
parser.add_argument("--usermodel", action="store_true",
                    help="Use UserModelTunnel to generate human action instead of joystick")
parser.add_argument("--model_yaw_control", action="store_true",
                    help="Enable model yaw control (action_dim=4)")
parser.add_argument("--no_lidar", action="store_true",
                    help="Disable lidar input for the model")
parser.add_argument("--num_obstacles", type=int, default=60,
                    help="Number of obstacles in the tunnel terrain")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 导入依赖 (必须在 SimulationApp 启动后导入) ---
import sys
import os
import torch
import numpy as np
import logging
import pygame
from datetime import datetime
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.core.utils.viewports import set_camera_view

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate_inverse, quat_rotate
from torchrl.envs.utils import set_exploration_type, ExplorationType
from joystick_wrapper import JoystickInterface
from srlc_model import load_srlc_model, MockConfig

from tensordict import TensorDict

# Add path to user_model_tunnely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/src/core")))
from user_model_tunnely import UserModelTunnel

drone_model_name = "Hummingbird"
drone_controller_name = "LeePositionController"
checkpoint_path = args_cli.checkpoint

# ============== Tunnel Terrain (same as env_tunnel.py) ==============

@height_field_to_mesh
def tunnel_obstacles_terrain(difficulty: float, cfg: HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Custom terrain mesh with walls forming a tunnel."""
    hf_raw = hf_terrains.discrete_obstacles_terrain.__wrapped__(difficulty, cfg)
    
    wall_thickness_meters = 1.0
    wall_height_meters = 10.0
    wall_start_meters = 2.0
    clear_zone_meters = 4.0
    
    wall_thickness_pixels = int(wall_thickness_meters / cfg.horizontal_scale)
    wall_height_steps = int(wall_height_meters / cfg.vertical_scale)
    wall_start_pixels = int(wall_start_meters / cfg.horizontal_scale)
    clear_zone_pixels = int(clear_zone_meters / cfg.horizontal_scale)
    
    hf_raw[0: wall_start_pixels + clear_zone_pixels, :] = 0
    # hf_raw[wall_start_pixels: wall_start_pixels + wall_thickness_pixels, :] = wall_height_steps
    hf_raw[:, 0:wall_thickness_pixels] = wall_height_steps
    hf_raw[:, -wall_thickness_pixels:] = wall_height_steps
    
    return hf_raw


@configclass
class HfTunnelObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = tunnel_obstacles_terrain


# ============== Logger Setup ==============

def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"verify_tunnel_{timestamp}.log")
    
    logger = logging.getLogger("verify_tunnel")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


# ============== Visualizer ==============

class Visualizer:
    def __init__(self, max_steps=500, model_vel_smooth=0.3):
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.max_steps = max_steps
        self.history = []
        # EMA smoothing factor for model velocity (0~1, smaller = smoother)
        self._model_vel_alpha = model_vel_smooth
        self._model_vel_ema = None  # [x, y, z]

    def update(self, pos, human_vel_w, model_vel_w, human_yaw_rate=None, model_yaw_rate=None):
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().squeeze().tolist()
        
        self.history.append(pos)
        if len(self.history) > self.max_steps:
            self.history.pop(0)
        
        starts = []
        ends = []
        colors = []
        sizes = []

        # Trajectory (Red)
        if len(self.history) >= 2:
            starts.extend(self.history[:-1])
            ends.extend(self.history[1:])
            colors.extend([(1.0, 0.0, 0.0, 1.0)] * (len(self.history) - 1))
            sizes.extend([2.0] * (len(self.history) - 1))

        # Human velocity: Green
        if human_vel_w is not None:
            if isinstance(human_vel_w, torch.Tensor):
                human_vel_w = human_vel_w.detach().cpu().squeeze().tolist()
            end_h = [p + v for p, v in zip(pos, human_vel_w)]
            starts.append(pos)
            ends.append(end_h)
            colors.append((0.0, 1.0, 0.0, 1.0))
            sizes.append(4.0)

        # Human yaw rate: Cyan vertical line
        if human_yaw_rate is not None:
            if isinstance(human_yaw_rate, torch.Tensor):
                human_yaw_rate = human_yaw_rate.detach().cpu().item()
            end_yaw_h = [pos[0], pos[1], pos[2] + human_yaw_rate]
            starts.append(pos)
            ends.append(end_yaw_h)
            colors.append((0.0, 1.0, 1.0, 1.0))
            sizes.append(4.0)

        # Model velocity: Blue (EMA-smoothed)
        if model_vel_w is not None:
            if isinstance(model_vel_w, torch.Tensor):
                model_vel_w = model_vel_w.detach().cpu().squeeze().tolist()
            a = self._model_vel_alpha
            if self._model_vel_ema is None:
                self._model_vel_ema = list(model_vel_w)
            else:
                self._model_vel_ema = [a * v + (1 - a) * s for v, s in zip(model_vel_w, self._model_vel_ema)]
            end_m = [p + v for p, v in zip(pos, self._model_vel_ema)]
            starts.append(pos)
            ends.append(end_m)
            colors.append((0.0, 0.0, 1.0, 1.0))
            sizes.append(4.0)

        # Model yaw rate: Purple vertical line
        if model_yaw_rate is not None:
            if isinstance(model_yaw_rate, torch.Tensor):
                model_yaw_rate = model_yaw_rate.detach().cpu().item()
            end_yaw = [pos[0], pos[1], pos[2] + model_yaw_rate]
            starts.append(pos)
            ends.append(end_yaw)
            colors.append((1.0, 0.0, 1.0, 1.0))
            sizes.append(4.0)

        self._draw.clear_lines()
        if starts:
            self._draw.draw_lines(starts, ends, colors, sizes)


# ============== Main ==============

def main():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger, log_file = setup_logger(log_dir)
    logger.info(f"Log file: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0
    action_dim = 4 if args_cli.model_yaw_control else 3

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- 初始化无人机 ---
    drone, controller = MultirotorBase.make(
        drone_model_name, drone_controller_name, device
    )
    # Spawn at the tunnel start position (same as env_tunnel _reset_idx)
    INIT_POS = torch.tensor([[-8.0, 0.0, 4.0]], device=device)
    INIT_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    drone.spawn(translations=INIT_POS)

    # --- Lighting ---
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sky_light_cfg = sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0)
    sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    # --- Terrain (Tunnel with obstacles) ---
    # LiDAR params matching training defaults
    LIDAR_RANGE = 4.0
    LIDAR_VFOV = (-10.0, 20.0)
    LIDAR_VBEAMS = 4
    LIDAR_HRES = 10.0  # 36 beams

    if not args_cli.no_lidar:
        terrain_cfg = TerrainImporterCfg(
            num_envs=1,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(24.0, 12.0),  # Tunnel dimensions (same as training env)
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
                        num_obstacles=args_cli.num_obstacles,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=(8.0, 20.0),
                        platform_width=0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain = terrain_cfg.class_type(terrain_cfg)
    else:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

    # --- Setup LiDAR ---
    lidar = None
    if not args_cli.no_lidar:
        target_mesh = "/World/ground"
        drone_prim_path = f"/World/envs/env_0/{drone.name}_0/base_link"

        ray_caster_cfg = RayCasterCfg(
            prim_path=drone_prim_path,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=LIDAR_HRES,
                vertical_ray_angles=torch.linspace(LIDAR_VFOV[0], LIDAR_VFOV[1], LIDAR_VBEAMS)
            ),
            debug_vis=False,
            mesh_prim_paths=[target_mesh],
        )
        lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        logger.info(f"LiDAR initialized on {drone_prim_path}")

    sim.reset()
    drone.initialize()

    joystick = JoystickInterface()
    vis = Visualizer(max_steps=500, model_vel_smooth=0.1)

    # --- 初始化 UserModelTunnel (如果启用) ---
    user_model = None
    if args_cli.usermodel:
        mock_cfg = MockConfig(device)
        user_model = UserModelTunnel(num_envs=1, cfg=mock_cfg, logger=logger)
        logger.info("UserModelTunnel enabled for human action generation")

    # --- 加载模型 ---
    model_type = args_cli.model_type
    policy = load_srlc_model(
        model_type, checkpoint_path, device,
        action_dim=action_dim,
        enable_lidar=not args_cli.no_lidar
    )
    if policy is None:
        return

    # --- Reset UserModel (如果启用) ---
    if user_model is not None:
        init_pos = torch.tensor([[-7.0, 0.0, 5.0]], device=device)
        init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        user_model.reset(init_pos, init_quat, torch.tensor([0], device=device))

    max_speed_xy = 2.0
    max_speed_z = 1.0
    max_yaw_rate = 0.5

    # 死区阈值: 手柄输入速度都很小时，忽略模型输出
    DEADZONE_HUMAN_VEL = 0.1   # human action body-frame speed threshold
    # DEADZONE_DRONE_VEL = 0.15  # drone world-frame speed threshold

    VIEWER_EYE_OFFSET = [-4.0, 0.0, 2.0]
    VIEWER_LOOKAT_OFFSET = [0.0, 0.0, 1.0]

    print("[INFO]: Setup complete")
    print(f"[INFO]: Joystick connected: {joystick.connected}")
    print(f"[INFO]: Model type: {model_type}")
    print(f"[INFO]: Action dim: {action_dim}")
    print(f"[INFO]: UserModel: {args_cli.usermodel}")
    print(f"[INFO]: LiDAR: {not args_cli.no_lidar}")
    print(f"[INFO]: Obstacles: {args_cli.num_obstacles}")
    print("[INFO]: Press joystick 'Start' (button 7) or keyboard 'R' to reset drone")

    def reset_drone():
        """Reset drone to initial pose with zero velocity."""
        env_ids = torch.tensor([0], device=device)
        drone._reset_idx(env_ids, train=False)
        pos = INIT_POS.unsqueeze(1)   # (1, 1, 3)
        quat = INIT_QUAT.unsqueeze(1) # (1, 1, 4)
        drone.set_world_poses(pos, quat, env_ids)
        zero_vel = torch.zeros(1, 1, 6, device=device)
        drone.set_velocities(zero_vel, env_ids)
        vis._model_vel_ema = None
        vis.history.clear()
        if user_model is not None:
            user_model.reset(INIT_POS, INIT_QUAT, env_ids)
        logger.info("Drone reset to initial position")

    # ===== 仿真循环 =====
    NUM_FRAMES = 20000
    frame_count = 0
    prev_human_action = torch.zeros((1, action_dim), device=device)

    while simulation_app.is_running():
        if frame_count >= NUM_FRAMES:
            logger.info("Simulation finished.")
            break

        if sim.is_playing():
            # 检查重置按钮 (joystick Start button 7 / keyboard R)
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN and event.button == 7:
                    reset_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_requested = True
            if reset_requested:
                reset_drone()
                prev_human_action = torch.zeros((1, action_dim), device=device)
                sim.step(render=True)
                frame_count += 1
                continue

            # 1. 获取无人机状态
            root_state = drone.get_state()[..., :13]  # (1, 1, 13)
            current_pos = root_state[..., :3]
            current_quat = root_state[..., 3:7]
            current_vel_w = root_state[..., 7:10]
            current_ang_vel_w = root_state[..., 10:13]

            # 计算 Body Frame 状态 (State: 10 dims)
            vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_vel_w.squeeze(1))
            ang_vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_ang_vel_w.squeeze(1))
            drone_state = torch.cat([vel_b, ang_vel_b, current_quat.squeeze(1)], dim=-1)  # (1, 10)

            # 2. 获取 Human Action (Body Frame)
            if user_model is not None:
                # UserModelTunnel 输出: (N, 3) 体坐标系线速度
                drone_pos_w = current_pos.squeeze(1)  # (1, 3)
                human_action_3d, _ = user_model.step(drone_state, drone_pos_w)  # (1, 3)
                
                if action_dim == 4:
                    # 如果模型需要4维动作，添加零yaw_rate
                    human_action = torch.cat([
                        human_action_3d,
                        torch.zeros(1, 1, device=device)
                    ], dim=-1)  # (1, 4)
                else:
                    human_action = human_action_3d  # (1, 3)
            else:
                # 从手柄输入获取 Human Action (Body Frame)
                stick_input = joystick.get_input().to(device)

                vel_cmd_x_body = stick_input[1] * max_speed_xy  # Pitch -> forward
                vel_cmd_y_body = -stick_input[0] * max_speed_xy   # Roll -> lateral
                vel_cmd_z = stick_input[2] * max_speed_z         # Throttle

                if action_dim == 4:
                    yaw_rate_cmd = -stick_input[3] * max_yaw_rate
                    human_action = torch.tensor(
                        [[vel_cmd_x_body, vel_cmd_y_body, vel_cmd_z, yaw_rate_cmd]],
                        device=device
                    )
                else:
                    human_action = torch.tensor(
                        [[vel_cmd_x_body, vel_cmd_y_body, vel_cmd_z]],
                        device=device
                    )

            human_action_input = human_action[..., :action_dim]

            # Compute LiDAR Observation
            lidar_hbeams = int(360 / LIDAR_HRES)
            lidar_obs = torch.zeros((1, 1, lidar_hbeams, LIDAR_VBEAMS), device=device)
            if lidar:
                lidar.update(dt)
                
                ray_hits = lidar.data.ray_hits_w
                pos_w = lidar.data.pos_w
                dist = torch.norm(ray_hits - pos_w.unsqueeze(1), dim=-1)
                dist = dist.clamp_max(LIDAR_RANGE)
                # Normalized scan: (range - distance) / range -> [0, 1]
                scan = (LIDAR_RANGE - dist) / LIDAR_RANGE
                lidar_obs = scan.reshape(1, 1, lidar_hbeams, LIDAR_VBEAMS)

            # 构建观察 TensorDict
            obs_dict = {
                "state": drone_state,                    # (1, 10)
                "human_action": prev_human_action,        # 使用上一步的用户动作输入观察
            }
            if not args_cli.no_lidar:
                obs_dict["lidar"] = lidar_obs             # (1, 1, 36, 4)
            
            obs = TensorDict({
                "agents": TensorDict({
                    "observation": TensorDict(obs_dict, batch_size=[1])
                }, batch_size=[1])
            }, batch_size=[1], device=device)

            # 3. 模型推理
            exploration_type = ExplorationType.DETERMINISTIC
            with torch.no_grad(), set_exploration_type(exploration_type):
                policy(obs)
                # 获取模型输出 (World Frame Velocity)
                model_action_w = obs["agents", "command"]  # (1, action_dim)

            # 死区: 手柄输入速度都很小时，忽略模型输出
            human_speed = torch.norm(human_action_input[..., :3], dim=-1)  # (1,)
            # drone_speed = torch.norm(current_vel_w.squeeze(1), dim=-1)     # (1,)
            in_deadzone = (human_speed < DEADZONE_HUMAN_VEL)
            if in_deadzone.any():
                model_action_w = torch.zeros_like(model_action_w)

            # 4. 应用控制
            if args_cli.model_yaw_control and action_dim == 4:
                model_vel_w = model_action_w[..., :3]
                model_yaw_rate = model_action_w[..., 3]
                target_yaw = model_yaw_rate.unsqueeze(1) * torch.pi
            else:
                model_vel_w = model_action_w[..., :3]
                model_yaw_rate = torch.zeros((1,), device=device)
                # 使用手柄的 yaw 控制 (如果连接了手柄)
                # if not args_cli.usermodel:
                #     stick_input = joystick.get_input().to(device)
                #     human_yaw_rate = -stick_input[3] * max_yaw_rate
                #     target_yaw = human_yaw_rate.unsqueeze(0).unsqueeze(0) * torch.pi
                # else:
                #     target_yaw = None
                target_yaw = None

            target_vel = model_vel_w.unsqueeze(1)  # (1, 1, 3)

            action = controller(
                root_state=root_state,
                target_vel=target_vel,
                target_yaw=target_yaw
            )
            drone.apply_action(action)

            # 可视化
            eye_vel_offset = -current_vel_w * torch.tensor([2.0, 2.0, 0.0], device=device)
            # set_camera_view(
            #     eye=current_pos.squeeze().cpu() + eye_vel_offset.squeeze().cpu() + torch.as_tensor(VIEWER_EYE_OFFSET).cpu(),
            #     target=current_pos.squeeze().cpu() + torch.as_tensor(VIEWER_LOOKAT_OFFSET).cpu()
            # )
            set_camera_view(
                eye=current_pos.squeeze().cpu() + torch.as_tensor(VIEWER_EYE_OFFSET).cpu(),
                target=current_pos.squeeze().cpu() + torch.as_tensor(VIEWER_LOOKAT_OFFSET).cpu()                      
            )

            # Update Visualizer
            human_vel_b = human_action[..., :3]
            human_vel_w = quat_rotate(current_quat.squeeze(1), human_vel_b)
            if not args_cli.model_yaw_control:
                vis.update(current_pos, human_vel_w, model_vel_w)
            else:
                human_yr = human_action[..., 3] if action_dim == 4 else None
                vis.update(current_pos, human_vel_w, model_vel_w, human_yr, model_yaw_rate)

            # 日志记录
            p = current_pos.squeeze().detach().cpu().numpy()
            dv = current_vel_w.squeeze().detach().cpu().numpy()
            av = model_vel_w.squeeze().detach().cpu().numpy()
            hv = human_vel_w.squeeze().detach().cpu().numpy()
            logger.debug(
                f"Frame {frame_count:06d} | Pos: [{p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f}] | "
                f"DroneVel: [{dv[0]:6.3f}, {dv[1]:6.3f}, {dv[2]:6.3f}] | "
                f"ModelVel: [{av[0]:6.3f}, {av[1]:6.3f}, {av[2]:6.3f}] | "
                f"HumanVel: [{hv[0]:6.3f}, {hv[1]:6.3f}, {hv[2]:6.3f}]"
            )

            # 更新 prev_human_action
            prev_human_action = human_action_input.clone()

        sim.step(render=True)
        frame_count += 1

    simulation_app.close()

if __name__ == "__main__":
    main()
