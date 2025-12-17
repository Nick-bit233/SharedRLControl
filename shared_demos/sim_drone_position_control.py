# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher
# AppLauncher.add_app_launcher_args(parser) # 如果需要命令行参数
# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# --- 导入依赖 (必须在 SimulationApp 启动后导入) ---
import torch
import numpy as np
import carb
import omni.appwindow
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw

# 导入控制器和手柄接口
from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from joystick_wrapper import JoystickInterface

drone_model_name = "Hummingbird" 
drone_controller_name = "LeePositionController"

class TrajectoryVisualizer:
    def __init__(self, max_steps=200):
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.max_steps = max_steps
        self.history = []

    def update(self, pos):
        # pos: torch.Tensor (3,)
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().squeeze().tolist()
        
        self.history.append(pos)
        if len(self.history) > self.max_steps:
            self.history.pop(0)
        
        if len(self.history) < 2:
            return

        starts = self.history[:-1]
        ends = self.history[1:]
        colors = [(1.0, 0.0, 0.0, 1.0)] * len(starts)
        sizes = [2.0] * len(starts)
        
        self._draw.clear_lines()
        self._draw.draw_lines(starts, ends, colors, sizes)

class KeyboardInput:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )
        self.manual_mode = True # False = Hover, True = Manual
        self.world_frame = True # True = World Frame, False = Body Frame
        self.use_joystick = False # Default to Keyboard
        self.key_states = {
            "W": 0.0, "S": 0.0, # Pitch (Forward/Backward)
            "A": 0.0, "D": 0.0, # Roll (Left/Right)
            "I": 0.0, "K": 0.0, # Throttle (Up/Down)
            "J": 0.0, "L": 0.0  # Yaw (Left/Right)
        }

    def _on_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "M":
                self.manual_mode = True
                print("[INFO] Switched to Manual Mode")
            elif event.input.name == "H":
                self.manual_mode = False
                print("[INFO] Switched to Hover Mode (Position Hold)")
            elif event.input.name == "T":
                self.use_joystick = not self.use_joystick
                mode = "Joystick" if self.use_joystick else "Keyboard"
                print(f"[INFO] Input Source: {mode}")
            elif event.input.name == "B":
                self.world_frame = not self.world_frame
                frame = "World Frame" if self.world_frame else "Body Frame"
                print(f"[INFO] Control Frame: {frame}")
            
            if event.input.name in self.key_states:
                self.key_states[event.input.name] = 1.0
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self.key_states:
                self.key_states[event.input.name] = 0.0

    def get_input(self):
        # [roll, pitch, throttle, yaw_rate]
        roll = self.key_states["D"] - self.key_states["A"] # D=Right(+), A=Left(-)
        pitch = self.key_states["W"] - self.key_states["S"] # W=Forward(+), S=Backward(-)
        throttle = self.key_states["I"] - self.key_states["K"] # I=Up(+), K=Down(-)
        yaw = self.key_states["L"] - self.key_states["J"] # L=Right(+), J=Left(-)
        return torch.tensor([roll, pitch, throttle, yaw], dtype=torch.float32)

def main():
    # 初始化 PyTorch 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0  # 假设仿真频率 60Hz

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- 构建环境 ---
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # 从omnidrones构建无人机
    drone, controller = MultirotorBase.make(
        drone_model_name, drone_controller_name, device
    )
    # 注意：omni_drones 的 spawn 通常接受 translations 张量
    # 这里我们创建一个单独的无人机
    drone.spawn(translations=torch.tensor([[0.0, 0.0, 1.0]], device=device))

    # 初始化输入控制器
    joystick = JoystickInterface()
    keyboard_input = KeyboardInput()
    traj_vis = TrajectoryVisualizer(max_steps=500)

    # Play the simulator
    sim.reset()
    drone.initialize()

    # 控制增益 (将手柄的 -1~1 映射到物理单位)
    max_speed_xy = 2.0 # m/s
    max_speed_z = 1.0  # m/s
    max_yaw_rate = 1.5 # rad/s

    print("[INFO]: Setup complete...")
    print("[INFO]: Press 'M' for Manual, 'H' for Hover, 'T' to toggle Joystick/Keyboard.")
    print("[INFO]: Press 'B' to toggle Control Frame (World/Body).")
    print("[INFO]: Keyboard controls: WASD (Move), IK (Up/Down), JL (Yaw)")

    # 状态变量
    target_pos = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    target_yaw = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    
    was_manual = False

    # --- 仿真循环 ---
    while simulation_app.is_running():
        
        if sim.is_playing():
            # A. 获取无人机状态
            # get_state 返回 (num_envs, num_drones, state_dim) = (1, 1, 13)
            # 我们只需要前13维: pos(3), quat(4), lin_vel(3), ang_vel(3)
            root_state = drone.get_state()[..., :13]
            
            # 获取当前位置和姿态用于初始化或悬停逻辑
            current_pos = root_state[..., :3]
            current_quat = root_state[..., 3:7]
            
            # 更新轨迹可视化
            traj_vis.update(current_pos)
            
            # B. 状态机逻辑
            if keyboard_input.manual_mode:
                if not was_manual:
                    # 刚切换到手动模式，保持当前的 target_pos (平滑过渡)
                    # 或者也可以选择重置为当前位置，防止之前的 target_pos 偏差太大
                    target_pos = current_pos.clone()
                    # 计算当前 yaw 作为 target_yaw
                    r, p, y = quat_to_euler_angles(current_quat[0].cpu().squeeze())
                    target_yaw = torch.tensor([[y]], device=device, dtype=torch.float32)
                    was_manual = True
                
                # 获取输入
                if keyboard_input.use_joystick:
                    stick_input = joystick.get_input().to(device)
                else:
                    stick_input = keyboard_input.get_input().to(device)
                
                # 映射到速度指令 (Body Frame or World Frame?)
                # 通常手柄控制是相对于机身或世界坐标系的。

                # 世界坐标系：
                # 简单起见，假设 Roll/Pitch 控制世界坐标系的 XY 速度 (类似大疆模式2)
                if keyboard_input.world_frame:
                    vel_cmd_x = stick_input[1] * max_speed_xy # Pitch -> 前后 (X)
                    vel_cmd_y = -stick_input[0] * max_speed_xy # Roll -> 左右 (Y)
                    vel_cmd_z = stick_input[2] * max_speed_z  # Throttle -> 上下 (Z)
                    yaw_rate_cmd = -stick_input[3] * max_yaw_rate # Yaw
                else:
                    # 机身坐标系：
                    # 如果做机头朝向控制 (First Person View 风格)，需要旋转 vel_cmd_xy
                    vel_cmd_x_body = stick_input[1] * max_speed_xy # Pitch -> 前后 (X)
                    vel_cmd_y_body = -stick_input[0] * max_speed_xy # Roll -> 左右 (Y)
                    vel_cmd_z = stick_input[2] * max_speed_z  # Throttle -> 上下 (Z)
                    yaw_rate_cmd = -stick_input[3] * max_yaw_rate # Yaw

                    # 获取当前 Yaw, 旋转到世界坐标系
                    r, p, y = quat_to_euler_angles(current_quat[0].cpu().squeeze())
                    c = np.cos(y)
                    s = np.sin(y)
                    vel_cmd_x = vel_cmd_x_body * c - vel_cmd_y_body * s
                    vel_cmd_y = vel_cmd_x_body * s + vel_cmd_y_body * c

                
                # 积分速度得到位置目标
                target_pos[..., 0] += vel_cmd_x * dt
                target_pos[..., 1] += vel_cmd_y * dt
                target_pos[..., 2] += vel_cmd_z * dt
                target_yaw += yaw_rate_cmd * dt
                
            else:
                # 悬停模式
                if was_manual:
                    # 刚切换到悬停模式，锁定当前位置
                    target_pos = current_pos.clone()
                    r, p, y = quat_to_euler_angles(current_quat[0].cpu().squeeze())
                    target_yaw = torch.tensor([[y]], device=device, dtype=torch.float32)
                    was_manual = False
                
                # 保持 target_pos 不变
                pass

            # C. 计算控制指令
            # LeePositionController compute()
            # root_state, target_pos, target_vel, target_acc, target_yaw
            # action: (1, 1, 4)
            action = controller.compute(
                root_state=root_state,
                target_pos=target_pos,
                target_yaw=target_yaw
            )
            print(f"[DEBUG] Root state shape: {root_state.shape}, action shape: {action.shape}")
            
            # D. 应用控制指令
            drone.apply_action(action)

        # 步进物理世界
        sim.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()