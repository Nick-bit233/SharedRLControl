import os
import hydra
import torch
import wandb
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# 1. 启动 Isaac Sim (必须在其他 import 之前)
# 强制开启 GUI 渲染管线 (即使在 headless 模式下也能录像)
sim_app = SimulationApp({"headless": True, "anti_aliasing": 1, "renderer": "RayTracing"})

from env import NavigationEnv
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback 

FILE_PATH = os.path.join(os.path.dirname(__file__), "../cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    print("[DebugRunner] Starting Minimal Debug Environment...")

    # === 覆盖配置以进行 Minimal Debug ===
    cfg.env.num_envs = 1           # 只看 1 个无人机
    cfg.env.num_obstacles = 20     # 减少障碍物，看清场景
    cfg.env_dyn.num_obstacles = 0  # 减少动态障碍
    cfg.max_frame_num = 1000       # 只跑 1000 步
    cfg.debug_mode = True          # 开启我们在 env.py 写的调试模式
    
    # 打印配置确认
    print(OmegaConf.to_yaml(cfg))

    # === 初始化环境 ===
    env = NavigationEnv(cfg)
    
    # 启用渲染 (这对录像至关重要)
    env.enable_render(True)

    # === Transforms (保持与 train.py 一致) ===
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=True)
    transformed_env = TransformedEnv(env, Compose(vel_transform))
    
    # === 初始化 PPO (加载权重或随机) ===
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    
    # 如果有 checkpoint，可以在这里加载
    # checkpoint_path = "ckpts/checkpoint_final.pt"
    # if os.path.exists(checkpoint_path):
    #     print(f"Loading checkpoint: {checkpoint_path}")
    #     policy.load_state_dict(torch.load(checkpoint_path))

    # === 视频录制回调 ===
    # 这会自动抓取 viewport 并保存为 mp4
    render_callback = RenderCallback(interval=1) # 每一帧都录
    
    print("[DebugRunner] Starting Rollout...")
    
    # === 手动运行 Rollout (不使用 Collector，为了完全控制) 
    # TODO:不使用collector进行rollout会产生相关问题，主要在于受到torchrl管理的状态确实（如RNN的hidden state）无法被正确管理
    # ===
    transformed_env.train() # 使用 eval 模式 (或者 .train() 如果想看探索噪声)
    # 获得初始观测
    tensordict = transformed_env.reset()
    # 打印tendordict观测键值以确认
    print("Initial Tensordict Keys:", tensordict.keys())
    print("Initial Observation:", tensordict["agents"])


    # with set_exploration_type(ExplorationType.RANDOM): # MEAN or RANDOM
    #     for i in range(1000): # 运行 1000 步
    #         # 1. 获取观测
    #         tensordict = transformed_env.step(tensordict)
            
    #         # 2. 策略推理
    #         # (PPO-RNN 需要处理 hidden state)
    #         tensordict = policy(tensordict)
            
    #         # 3. 执行一步
    #         tensordict = transformed_env.step(tensordict)
            
    #         # 4. 录制视频帧
    #         render_callback(transformed_env, tensordict)
            
    #         if i % 100 == 0:
    #             print(f"Step {i}: Reward={tensordict['next', 'agents', 'reward'].mean().item():.4f}")

    #         # 检查是否结束
    #         if tensordict["next", "done"].any():
    #             print("Episode Done. Resetting...")
    #             transformed_env.reset()

    # === 保存视频 ===
    video_path = os.path.join(os.getcwd(), "outputs", "debug_rollout.mp4")
    print(f"[DebugRunner] Saving video to {video_path}...")
    # RenderCallback 通常将帧保存在内存中，我们需要手动保存
    # 注意：RenderCallback 的具体 API 可能有所不同，
    # 这里假设 omni_drones 的实现支持 export
    # 如果不支持，可以使用 wandb.Video(np.array(frames), fps=30)
    
    # 简单的保存逻辑 (如果 RenderCallback 没有直接保存方法)
    import imageio
    import numpy as np
    frames = render_callback.frames # 获取帧列表
    if len(frames) > 0:
        # 转换 tensor (C, H, W) -> numpy (H, W, C)
        video_frames = [f.permute(1, 2, 0).cpu().numpy() for f in frames]
        # 确保是 uint8
        video_frames = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in video_frames]
        imageio.mimsave(video_path, video_frames, fps=30)
        print("Video saved successfully.")
    else:
        print("No frames captured!")

    sim_app.close()

if __name__ == "__main__":
    main()