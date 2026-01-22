# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher
import argparse

# 添加命令行参数
parser = argparse.ArgumentParser(description="Verify SRLC Model")
parser.add_argument("--model_type", type=str, default="Residual", choices=["Simple", "Residual"], help="Type of SRLC model to load")
parser.add_argument("--checkpoint", type=str, default="/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/shared_demos/ckpts/0106-1123/checkpoint_1500.pt", help="Path to model checkpoint")
parser.add_argument("--usermodel", action="store_true", help="Use UserModel to generate human action instead of joystick")
parser.add_argument("--model_yaw_control", action="store_true", help="Enable model yaw control")
parser.add_argument("--no_lidar", action="store_true", help="Disable lidar input for the model")
# parser.add_argument("--use_obstacles", action="store_true", help="Enable obstacle forest environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 导入依赖 (必须在 SimulationApp 启动后导入) ---
import sys
import os
import torch
import numpy as np
import carb
import logging
from datetime import datetime
import omni.appwindow
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.core.utils.viewports import set_camera_view

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate_inverse, quat_rotate
from torchrl.envs.utils import set_exploration_type, ExplorationType
from joystick_wrapper import JoystickInterface
from srlc_model import load_srlc_model_simple, MockConfig

from tensordict import TensorDict

# Add path to user_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/src/core")))
from user_model import UserModel

drone_model_name = "Hummingbird" 
drone_controller_name = "LeePositionController"
checkpoint_path = args_cli.checkpoint

# --- 配置日志 ---
def setup_logger(log_dir="logs"):
    """Setup logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"verify_srlc_model_{timestamp}.log")
    
    logger = logging.getLogger("verify_srlc_model")
    logger.setLevel(logging.DEBUG)
    
    # File handler for all debug output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

class Visualizer:
    def __init__(self, max_steps=200):
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.max_steps = max_steps
        self.history = []

    def update(self, pos, human_vel_w, model_vel_w, human_yaw_rate=None, model_yaw_rate=None):
        # pos: (3,)
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
            colors.extend([(1.0, 0.0, 0.0, 1.0)] * (len(self.history)-1))
            sizes.extend([2.0] * (len(self.history)-1))

        # Vectors
        # Human: Green
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

        # Model: Blue
        if model_vel_w is not None:
            if isinstance(model_vel_w, torch.Tensor):
                model_vel_w = model_vel_w.detach().cpu().squeeze().tolist()
            end_m = [p + v for p, v in zip(pos, model_vel_w)]
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


def main():
    # 初始化日志
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger, log_file = setup_logger(log_dir)
    logger.info(f"Log file created at: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- 初始化无人机 ---
    drone, controller = MultirotorBase.make(
        drone_model_name, drone_controller_name, device
    )
    drone.spawn(translations=torch.tensor([[0.0, 0.0, 1.0]], device=device))

    # --- 构建环境 ---
    # Light
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Terrain / Ground
    terrain = None
    if not args_cli.no_lidar:
        # Create obstacle terrain
        terrain_cfg = TerrainImporterCfg(
            num_envs=1,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(40.0, 40.0), 
                border_width=5.0,
                num_rows=2, 
                num_cols=2, 
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
                        num_obstacles=100, 
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.8, 2.0),
                        obstacle_height_range=(8.0, 16.0),
                        platform_width=4.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain = terrain_cfg.class_type(terrain_cfg)
    else:
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/ground", cfg)
    
    # --- Setup Lidar ---
    lidar = None
    # Lidar Config Constants (match env_residual defaults)
    LIDAR_RANGE = 4.0
    LIDAR_VFOV = (-10.0, 20.0)
    LIDAR_VBEAMS = 4
    LIDAR_HRES = 10.0 # 36 beams
    
    if not args_cli.no_lidar:
        target_mesh = "/World/ground"
        # Default drone prim path when spawned
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
        logger.info(f"Lidar initialized attached to {drone_prim_path}")

    sim.reset()
    drone.initialize()

    joystick = JoystickInterface()
    vis = Visualizer(max_steps=500)

    # --- 初始化 UserModel (如果启用) ---
    user_model = None
    if args_cli.usermodel:
        mock_cfg = MockConfig(device)
        user_model = UserModel(num_envs=1, cfg=mock_cfg, logger=logger)
        logger.info("UserModel enabled for human action generation")

    # --- 加载模型 ---
    action_dim = 4 if args_cli.model_yaw_control else 3
    model_type = args_cli.model_type
    policy = load_srlc_model_simple(model_type, checkpoint_path, device, action_dim=action_dim, enable_lidar=not args_cli.no_lidar)
    if policy is None:
        return

    # --- Reset UserModel (如果启用) ---
    if user_model is not None:
        init_pos = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        user_model.reset(init_pos, init_quat, torch.tensor([0], device=device))

    max_speed_xy = 2.0 
    max_speed_z = 1.0  
    max_yaw_rate = 0.5 

    VIEWER_EYE_OFFSET = [5.0, -1.0, 2.0]
    VIEWER_LOOKAT_OFFSET = [0.0, 0.0, 1.0]

    print("[INFO]: Setup complete...")
    print(f"[INFO]: Joystick connnect status: {joystick.connected}")
    print(f"[INFO]: ActionModel type: {args_cli.model_type}")
    print(f"[INFO]: UserModel mode: {args_cli.usermodel}")
    # TODO: 添加使用手柄按键切换控制状态
    # print("[INFO]: Hold 'RT' to enable shared control, and 'LT' for pure Manual, otherwise Hover (default behavior).")

    # 初始目标位置 (用于积分)
    # target_pos = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    # target_yaw = torch.tensor([[0.0]], device=device, dtype=torch.float32)

    # ===== 仿真循环 =====
    NUM_FRAMES = 20000
    frame_count = 0
    human_action_input = torch.zeros((1, action_dim), device=device)
    prev_human_action = torch.zeros((1, action_dim), device=device)
    while simulation_app.is_running():
        if frame_count >= NUM_FRAMES:
            logger.info("Simulation finished.")
            break
        
        if sim.is_playing():
            # 1. 获取无人机状态
            root_state = drone.get_state()[..., :13] # (1, 1, 13)
            current_pos = root_state[..., :3]
            current_quat = root_state[..., 3:7]
            current_vel_w = root_state[..., 7:10]
            current_ang_vel_w = root_state[..., 10:13]

            # 计算 Body Frame 状态 (State: 10 dims)
            vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_vel_w.squeeze(1))
            ang_vel_b = quat_rotate_inverse(current_quat.squeeze(1), current_ang_vel_w.squeeze(1))
            drone_state = torch.cat([vel_b, ang_vel_b, current_quat.squeeze(1)], dim=-1) # (1, 10)

            # 2. 获取 Human Action (Body Frame)
            if user_model is not None:
                # 使用 UserModel 生成控制信号
                drone_pos_w = current_pos.squeeze(1)  # (1, 3)
                human_action, _ = user_model.step(drone_state, drone_pos_w)  # (1, 4)
            else:
                # 从手柄输入获取 Human Action
                stick_input = joystick.get_input().to(device)

                vel_cmd_x_body = -stick_input[1] * max_speed_xy # Pitch
                vel_cmd_y_body = stick_input[0] * max_speed_xy # Roll (Right is y)
                vel_cmd_z = stick_input[2] * max_speed_z
                yaw_rate_cmd = -stick_input[3] * max_yaw_rate

                human_action = torch.tensor([[vel_cmd_x_body, vel_cmd_y_body, vel_cmd_z, yaw_rate_cmd]], device=device) # (1, 4)

            # Ensure human_action passed to model matches the model's action_dim (e.g. 3 vs 4)
            human_action_input = human_action[..., :action_dim]
            human_yaw_rate = human_action[..., 3]

            # Compute Lidar Observation
            lidar_obs = torch.zeros((1, 1, 36, 4), device=device)
            if lidar:
                lidar.update(dt) # Update sensor
                
                # Calculate scan (Range - Distance)
                ray_hits = lidar.data.ray_hits_w
                pos = lidar.data.pos_w
                dist = torch.norm(ray_hits - pos.unsqueeze(1), dim=-1)
                dist = dist.clamp_max(LIDAR_RANGE)
                scan = LIDAR_RANGE - dist
                
                # Reshape to (N, 1, H, V) = (1, 1, 36, 4)
                lidar_obs = scan.reshape(1, 1, int(360/LIDAR_HRES), LIDAR_VBEAMS)

            obs = TensorDict({
                "agents": TensorDict({
                    "observation": TensorDict({
                        "state": drone_state,
                        "human_action": prev_human_action, # 使用上一步的用户动作输入观察
                        "lidar": lidar_obs,
                    }, batch_size=[1])
                }, batch_size=[1])
            }, batch_size=[1], device=device)

            # 3. 模型推理
            with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
                policy(obs)
                # 获取模型输出 (World Frame Velocity, as per user instruction)
                # (1, 4) if model_yaw_control else (1, 3)
                model_action_w = obs["agents", "command"]

            # 4. 应用控制
            if args_cli.model_yaw_control:
                model_vel_w = model_action_w[..., :3]  # (1, 3)
                model_yaw_rate = model_action_w[..., 3]  # (1, )
                target_yaw = model_yaw_rate.unsqueeze(1) * torch.pi # (1, 1, )
            else:
                model_vel_w = model_action_w
                model_yaw_rate = torch.zeros((1,), device=device)
                # 用户手柄的 Yaw 控制
                target_yaw = human_yaw_rate.unsqueeze(1) * torch.pi

            # adapt to VelController
            target_vel = model_vel_w.unsqueeze(1)  # (1, 1, 3)

            action = controller(
                root_state=root_state,
                target_vel=target_vel,
                target_yaw=target_yaw
            )

            drone.apply_action(action)

            # 可视化
            # update Camera
            set_camera_view(
                # use cfg viewer settings as offset
                eye=current_pos.squeeze().cpu() + torch.as_tensor(VIEWER_EYE_OFFSET),
                target=current_pos.squeeze().cpu() + torch.as_tensor(VIEWER_LOOKAT_OFFSET)                        
            )

            # Update Visualizer
            human_vel_b = human_action[..., :3]
            human_vel_w = quat_rotate(current_quat.squeeze(1), human_vel_b)
            if not args_cli.model_yaw_control:
                vis.update(current_pos, human_vel_w, model_vel_w)
            else:
                vis.update(current_pos, human_vel_w, model_vel_w, human_yaw_rate, model_yaw_rate)

            # 日志记录
            to_print_pos = current_pos.squeeze().detach().cpu().numpy()
            to_print_d_vel = current_vel_w.squeeze().detach().cpu().numpy()
            to_print_d_yaw_rate = current_ang_vel_w[..., 2].squeeze().detach().cpu().item()
            to_print_a_vel = model_vel_w.squeeze().detach().cpu().numpy()
            to_print_a_yaw_rate = model_yaw_rate.squeeze().detach().cpu().item()
            to_print_human_vel = human_vel_w.squeeze().detach().cpu().numpy()
            to_print_human_yaw_rate = human_yaw_rate.squeeze().detach().cpu().item()
            logger.debug(f"Frame {frame_count:06d} | Drone - Pos: [{to_print_pos[0]:7.3f}, {to_print_pos[1]:7.3f}, {to_print_pos[2]:7.3f}] | Vel: [{to_print_d_vel[0]:6.3f}, {to_print_d_vel[1]:6.3f}, {to_print_d_vel[2]:6.3f}] | Yaw Rate: {to_print_d_yaw_rate:6.3f}")
            logger.debug(f"Frame {frame_count:06d} | Model - Vel: [{to_print_a_vel[0]:6.3f}, {to_print_a_vel[1]:6.3f}, {to_print_a_vel[2]:6.3f}] | Yaw Rate: {to_print_a_yaw_rate:6.3f}")
            logger.debug(f"Frame {frame_count:06d} | Human - Vel: [{to_print_human_vel[0]:6.3f}, {to_print_human_vel[1]:6.3f}, {to_print_human_vel[2]:6.3f}] | Yaw Rate: {to_print_human_yaw_rate:6.3f}")

            # 最后更新实时用户动作
            prev_human_action = human_action_input.clone()

        sim.step(render=True)
        frame_count += 1

    simulation_app.close()

if __name__ == "__main__":
    main()
