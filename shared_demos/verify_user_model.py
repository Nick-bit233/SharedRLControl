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
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw

# Add path to user_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))
from user_model import UserModel

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate

drone_model_name = "Hummingbird" 
drone_controller_name = "LeePositionController"

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
        self._draw.draw_lines([pos], [yaw_end], [(0.0, 0.0, 1.0, 1.0)], [3.0])

class MockConfig:
    # Verification Settings
    use_lib_noise = False # Set to False to test GPU implementation
    num_frames = 20000

    class Sim:
        dt = 1.0 / 60.0
        z_spawn = 4.0
    class Env:
        map_range = [20.0, 20.0, 4.0] # Half extents [x, y, z]
        max_episode_length = 500
    class Algo:
        class actor:
            action_limit = 2.0 # m/s
    
    def __init__(self, device):
        self.device = device
        self.sim = self.Sim()
        self.env = self.Env()
        self.algo = self.Algo()

def main():
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

    # 初始化 User Model
    user_model = UserModel(num_envs=1, cfg=mock_cfg, use_lib_noise=mock_cfg.use_lib_noise)
    
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

    print(f"[INFO]: Setup complete. Using Library Noise: {mock_cfg.use_lib_noise}")
    print(f"[INFO]: Simulating for {mock_cfg.num_frames} frames...")

    # 状态变量 (unsqueeze batch and env dims for controller)
    target_pos = init_pos.unsqueeze(1).clone()
    target_yaw = torch.zeros((1, 1, 1), device=device)
    
    prev_action = torch.zeros((1, 1, 4), device=device)

    # --- 仿真循环 ---
    frame_count = 0
    while simulation_app.is_running():
        if frame_count >= mock_cfg.num_frames:
            print("[INFO]: Simulation finished.")
            break

        if sim.is_playing():
            # A. 获取无人机状态
            root_state = drone.get_state()[..., :13]  # (1, 1, 13)
            current_pos = root_state[..., :3]  # (1, 1, 3)
            current_quat = root_state[..., 3:7]  # (1, 1, 4)
            
            # B. 获取 User Model 生成的控制信号
            # action: (1, 4) -> [vx_b, vy_b, vz_b, yaw_rate]
            action, _ = user_model.step(root_state, prev_action)
            # unsqueeze actions to (1, 1, 4) for controller
            action = action.unsqueeze(1)
            prev_action = action
            
            # C. 将 Body Frame 速度转换为 World Frame 并积分得到位置目标
            # action[..., :3] 是 Body Frame 速度
            # action[..., 3] 是 Yaw Rate
            
            vel_body = action[..., :3]  # (1, 1, 3)
            yaw_rate = action[..., 3]  # (1, 1, )
            
            # Rotate velocity to world frame
            vel_world = quat_rotate(current_quat, vel_body)
            to_print_vel = vel_world.squeeze().detach().cpu().numpy()
            to_print_pos = current_pos.squeeze().detach().cpu().numpy()
            print(f"[DEBUG]: frame: {frame_count}, Pos: {to_print_pos}, Vel_world: {to_print_vel}")
            
            # 更新轨迹可视化
            traj_vis.update(current_pos, vel_world, yaw_rate)
            
            # Integrate
            target_pos += vel_world * dt
            target_yaw += yaw_rate.unsqueeze(1) * dt
            
            # 简单的边界限制 (Clamping target to map range to avoid drifting too far if APF fails)
            # map_range = mock_cfg.env.map_range
            # target_pos[..., 0] = torch.clamp(target_pos[..., 0], -map_range[0], map_range[0])
            # target_pos[..., 1] = torch.clamp(target_pos[..., 1], -map_range[1], map_range[1])
            # target_pos[..., 2] = torch.clamp(target_pos[..., 2], 0.1, map_range[2])

            # D. 计算控制指令
            control_action = controller.compute(
                root_state=root_state,
                target_pos=target_pos,
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
