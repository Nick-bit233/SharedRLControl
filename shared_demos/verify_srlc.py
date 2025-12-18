# --- 使用isaaclab的AppLauncher启动 Isaac Sim ---
from isaaclab.app import AppLauncher
import argparse

# 添加命令行参数
parser = argparse.ArgumentParser(description="Verify SRLC Model")
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
import omni.appwindow
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw

# 添加 ppo_simple.py 所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torch import quat_rotate_inverse, quat_rotate
from joystick_wrapper import JoystickInterface

from ppo_simple import SimplePPO
from tensordict import TensorDict
from torchrl.data import CompositeSpec, Unbounded

drone_model_name = "Hummingbird" 
drone_controller_name = "LeePositionController"
checkpoint_path = "/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/shared_demos/ckpts/checkpoint_1500.pt"

class Visualizer:
    def __init__(self, max_steps=200):
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.max_steps = max_steps
        self.history = []

    def update(self, pos, human_vel_w, model_vel_w):
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

        # Model: Blue
        if model_vel_w is not None:
            if isinstance(model_vel_w, torch.Tensor):
                model_vel_w = model_vel_w.detach().cpu().squeeze().tolist()
            end_m = [p + v for p, v in zip(pos, model_vel_w)]
            starts.append(pos)
            ends.append(end_m)
            colors.append((0.0, 0.0, 1.0, 1.0))
            sizes.append(4.0)

        self._draw.clear_lines()
        if starts:
            self._draw.draw_lines(starts, ends, colors, sizes)

class KeyboardInput:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )
        self.use_joystick = False 
        self.key_states = {
            "W": 0.0, "S": 0.0, # Pitch (Forward/Backward)
            "A": 0.0, "D": 0.0, # Roll (Left/Right)
            "I": 0.0, "K": 0.0, # Throttle (Up/Down)
            "J": 0.0, "L": 0.0  # Yaw (Left/Right)
        }

    def _on_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "T":
                self.use_joystick = not self.use_joystick
                mode = "Joystick" if self.use_joystick else "Keyboard"
                print(f"[INFO] Input Source: {mode}")
            
            if event.input.name in self.key_states:
                self.key_states[event.input.name] = 1.0
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self.key_states:
                self.key_states[event.input.name] = 0.0

    def get_input(self):
        # [roll, pitch, throttle, yaw_rate]
        roll = self.key_states["D"] - self.key_states["A"] 
        pitch = self.key_states["W"] - self.key_states["S"] 
        throttle = self.key_states["I"] - self.key_states["K"] 
        yaw = self.key_states["L"] - self.key_states["J"] 
        return torch.tensor([roll, pitch, throttle, yaw], dtype=torch.float32)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)

    # --- 构建环境 ---
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    drone, controller = MultirotorBase.make(
        drone_model_name, drone_controller_name, device
    )
    drone.spawn(translations=torch.tensor([[0.0, 0.0, 1.0]], device=device))

    joystick = JoystickInterface()
    keyboard_input = KeyboardInput()
    vis = Visualizer(max_steps=500)

    # --- 加载模型 ---
    class Config:
        class Algo:
            class Actor:
                action_limit = 2.0
                learning_rate = 1e-4
                clip_ratio = 0.1
            actor = Actor()
            class Critic:
                learning_rate = 1e-4
                clip_ratio = 0.1
            critic = Critic()
            class Rnn:
                enable = False
                gru_hidden_dim = 256
                gru_num_layers = 1
            rnn = Rnn()
            class FeatureExtractor:
                learning_rate = 1e-4
                dyn_obs_num = 5
            feature_extractor = FeatureExtractor()
            entropy_loss_coefficient = 1e-3
        algo = Algo()
        class Env:
            enable_lidar = False
        env = Env()
    
    cfg_model = Config()
    
    # 定义观察空间 (State: 10, Human Action: 4)
    observation_spec = CompositeSpec({
        "agents": CompositeSpec({
            "observation": CompositeSpec({
                "state": Unbounded((1, 10), device=device),
                "human_action": Unbounded((1, 4), device=device),
                "prev_action": Unbounded((1, 4), device=device),
            }, shape=(1,))
        }, shape=(1,))
    }, shape=(1,), device=device)
    
    # 定义动作空间 (Action: 4)
    action_spec = CompositeSpec({
        "agents": CompositeSpec({
            "action": Unbounded((1, 4), device=device)
        }, shape=(1,))
    }, shape=(1,), device=device)

    print(f"[INFO] Loading model from {checkpoint_path}")
    policy = SimplePPO(cfg_model.algo, observation_spec, action_spec, device)
    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    policy.eval()

    sim.reset()
    drone.initialize()

    max_speed_xy = 2.0 
    max_speed_z = 1.0  
    max_yaw_rate = 1.5 

    print("[INFO]: Setup complete...")
    print("[INFO]: Press 'T' to toggle Joystick/Keyboard.")
    print("[INFO]: Keyboard controls: WASD (Move), IK (Up/Down), JL (Yaw)")

    # 初始目标位置 (用于积分)
    target_pos = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    target_yaw = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    
    # 初始化 prev_action
    prev_action = torch.zeros((1, 4), device=device)

    while simulation_app.is_running():
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
            if keyboard_input.use_joystick:
                stick_input = joystick.get_input().to(device)
            else:
                stick_input = keyboard_input.get_input().to(device)
            
            vel_cmd_x_body = stick_input[1] * max_speed_xy # Pitch
            vel_cmd_y_body = -stick_input[0] * max_speed_xy # Roll (Right is -y)
            vel_cmd_z = stick_input[2] * max_speed_z
            yaw_rate_cmd = -stick_input[3] * max_yaw_rate

            human_action = torch.tensor([[vel_cmd_x_body, vel_cmd_y_body, vel_cmd_z, yaw_rate_cmd]], device=device) # (1, 4)

            # 3. 模型推理
            obs = TensorDict({
                "agents": TensorDict({
                    "observation": TensorDict({
                        "state": drone_state,
                        "human_action": human_action,
                        "prev_action": prev_action,
                    }, batch_size=[1])
                }, batch_size=[1])
            }, batch_size=[1], device=device)

            with torch.no_grad():
                policy(obs)
                # 获取模型输出 (Body Frame Velocity, as per user instruction)
                model_action_b = obs["agents", "action"] # (1, 4) [vx_b, vy_b, vz_b, yaw_rate]
                
                # Update prev_action for next step
                prev_action = model_action_b.clone()

            # 4. 应用控制
            # 模型输出的是 Body Frame 速度，我们需要将其转换为 World Frame
            model_vel_b = model_action_b[..., :3]
            model_yaw_rate = model_action_b[..., 3]
            
            # Rotate to World Frame
            model_vel_w = quat_rotate(current_quat.squeeze(1), model_vel_b)
            
            # Update target position
            target_pos = target_pos + model_vel_w * dt
            target_yaw = target_yaw + model_yaw_rate * dt

            # 限制 target_pos 不要离当前位置太远 (防止积分漂移过大)
            # error_pos = target_pos - current_pos
            # if error_pos.norm() > 1.0:
            #     target_pos = current_pos + error_pos / error_pos.norm() * 1.0

            action = controller.compute(
                root_state=root_state,
                target_pos=target_pos,
                target_yaw=target_yaw,
                target_vel=model_vel_w # Feed forward velocity
            )
            
            drone.apply_action(action)

            # 5. 可视化
            # Human Action (Body -> World)
            # human_action is [vx_b, vy_b, vz_b, yaw_rate]
            human_vel_b = human_action[..., :3]
            human_vel_w = quat_rotate(current_quat.squeeze(1), human_vel_b)
            
            vis.update(current_pos, human_vel_w, model_vel_w)

        sim.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
