# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# --- 导入依赖 (必须在 SimulationApp 启动后导入) ---
import torch
import numpy as np
import sys
import os
import logging
from datetime import datetime
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw
from srlc_model import MockConfig

# Add path to user_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))
from user_model import UserModel

from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

drone_model_name = "Hummingbird" 
drone_controller_name = "LeePositionController"

# --- 配置日志 ---
def setup_logger(log_dir="logs"):
    """Setup logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"verify_user_model_{timestamp}.log")
    
    logger = logging.getLogger("verify_user_model")
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

class TrajectoryVisualizer:
    def __init__(self, max_steps=200):
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.max_steps = max_steps
        self.history = []

    def update(self, pos, vel_world, yaw_rate):

        self._draw.clear_lines()
        # pos: squeeze to torch.Tensor (3,)
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().squeeze().tolist()
        if isinstance(vel_world, torch.Tensor):
            vel_world = vel_world.detach().cpu().squeeze().tolist()
        if isinstance(yaw_rate, torch.Tensor):
            yaw_rate = yaw_rate.detach().cpu().item()
        
        self.history.append(pos)
        if len(self.history) > self.max_steps:
            self.history.pop(0)
        
        # Draw Trajectory (Green)
        if len(self.history) >= 2:
            starts = self.history[:-1]
            ends = self.history[1:]
            colors = [(0.0, 1.0, 0.0, 1.0)] * len(starts) 
            sizes = [2.0] * len(starts)
            self._draw.draw_lines(starts, ends, colors, sizes)

        # Draw Velocity Vector (Red)
        vel_end = [pos[0] + vel_world[0], pos[1] + vel_world[1], pos[2] + vel_world[2]]
        self._draw.draw_lines([pos], [vel_end], [(1.0, 0.0, 0.0, 1.0)], [3.0])

        # Draw Yaw Rate Indicator (Blue Vertical Line)
        # Up for positive yaw rate, down for negative
        yaw_end = [pos[0], pos[1], pos[2] + yaw_rate]
        self._draw.draw_lines([pos], [yaw_end], [(1.0, 0.0, 1.0, 1.0)], [3.0])

def main():
    # 初始化日志
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger, log_file = setup_logger(log_dir)
    logger.info(f"Log file created at: {log_file}")
    
    # 初始化 PyTorch 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- 构建环境 ---
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Mock Config for User Model
    mock_cfg = MockConfig(device)
    
    # 从omnidrones构建无人机
    drone, controller = MultirotorBase.make(
        drone_model_name, drone_controller_name, device
    )
    z_spawn = mock_cfg.sim.z_spawn
    drone.spawn(translations=torch.tensor([[0.0, 0.0, z_spawn]], device=device))

    # 初始化 User Model (传入日志记录器)
    user_model = UserModel(num_envs=1, cfg=mock_cfg, logger=logger)
    
    traj_vis = TrajectoryVisualizer(max_steps=500)

    # Play the simulator
    sim.reset()
    drone.initialize()
    
    # Reset User Model
    # Attention: user model only considers single drone per env, so squeeze batch dim
    # pos: (1, 3), quat: (1, 4)
    # Ensure init_pos matches the spawn position to avoid initial drop
    init_pos = torch.tensor([[0.0, 0.0, z_spawn]], device=device)
    init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device) # w, x, y, z
    user_model.reset(init_pos, init_quat, torch.tensor([0], device=device))

    # 状态变量 (unsqueeze batch and env dims for controller)
    target_pos = init_pos.unsqueeze(1).clone()
    target_yaw = torch.zeros((1, 1, 1), device=device)
    
    NUM_FRAMES = 2000
    logger.info(f"Simulating for {NUM_FRAMES} frames...")
    # --- 仿真循环 ---
    frame_count = 0
    while simulation_app.is_running():
        if frame_count >= NUM_FRAMES:
            logger.info("Simulation finished.")
            break

        if sim.is_playing():
            # A. 获取无人机状态
            root_state = drone.get_state()[..., :13]  # (1, 1, 13)

            drone_pos_w = root_state[..., :3].squeeze(1)   # (N, 3)
            drone_vel_w = root_state[..., 7:10].squeeze(1)     # (N, 3) world_vel
            drone_ang_vel_w = root_state[..., 10:13].squeeze(1) # (N, 3) world_angular
            drone_orientation_q = root_state[..., 3:7].squeeze(1) # (N, 4) orientation(quat)

            # calculate drone's velocity and angular velocity in body frame
            vel_b = quat_rotate_inverse(drone_orientation_q, drone_vel_w)
            ang_vel_b = quat_rotate_inverse(drone_orientation_q, drone_ang_vel_w)

            drone_state_b = torch.cat([vel_b, ang_vel_b, drone_orientation_q], dim=-1)
            
            # B. 获取 User Model 生成的控制信号
            # action: (1, 4) -> [vx_b, vy_b, vz_b, yaw_rate]
            action, _ = user_model.step(drone_state_b, drone_pos_w)
            # unsqueeze actions to (1, 1, 4) for controller
            action = action.unsqueeze(1)
            prev_action = action
            
            # C. 将 Body Frame 速度转换为 World Frame
            
            # action[..., :3] 是 Body Frame 速度
            # action[..., 3] 是 Yaw Rate
            action_vel_b = action[..., :3]  # (1, 1, 3)
            action_yaw_rate = action[..., 3]  # (1, 1, )
            
            # Rotate velocity to world frame
            to_rotate_q = drone_orientation_q.unsqueeze(1)  # unsqueeze to (N, 1, 4) for using quat_rotate()
            action_vel_w = quat_rotate(to_rotate_q, action_vel_b) # (1, 1, 3)

            to_print_d_vel = drone_vel_w.squeeze().detach().cpu().numpy()
            to_print_a_vel = action_vel_w.squeeze().detach().cpu().numpy()
            to_print_pos = drone_pos_w.squeeze().detach().cpu().numpy()
            logger.debug(f"frame: {frame_count}, pos_w: {to_print_pos}, vel_w:{to_print_d_vel}, action_vel_w: {to_print_a_vel}")
            
            # 更新轨迹可视化
            traj_vis.update(drone_pos_w, action_vel_w, action_yaw_rate)
            
            # Integrate (pos control?)
            # target_pos += action_vel_w * dt
            # target_yaw += action_yaw_rate.unsqueeze(1) * dt

            # same implementation as VelController
            target_vel = action_vel_w
            target_yaw = action_yaw_rate * torch.pi

            # D. 计算控制指令
            control_action = controller(
                root_state=root_state,
                target_vel=target_vel,
                target_yaw=target_yaw
            )
            
            # E. 应用控制指令
            drone.apply_action(control_action)
            
            frame_count += 1

        # 步进物理世界
        sim.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
