import os
import logging
import hydra
import torch
import imageio
import numpy as np
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp
from hydra.core.hydra_config import HydraConfig

# 1. 启动 Isaac Sim (必须在其他 import 之前)
# 强制开启 GUI 渲染管线 (即使在 headless 模式下也能录像)
sim_app = SimulationApp({"headless": True, "anti_aliasing": 1, "renderer": "RayTracing"})

from env import NavigationEnv
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback
from torchrl.data import UnboundedContinuousTensorSpec

FILE_PATH = os.path.join(os.path.dirname(__file__), "../cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    print("[DebugRunner] Starting Minimal Debug Environment...")

    # === 覆盖配置以进行 Debug ===
    cfg.env.num_envs = 100           # 无人机数量
    cfg.env.num_obstacles = 1024     # 静态障碍数量
    cfg.env_dyn.num_obstacles = 0   # 动态障碍数量
    cfg.algo.training_frame_num = 128  # 每个采集批次帧数
    cfg.max_frame_num = cfg.algo.training_frame_num * 101  # 最大采集帧数
    cfg.debug_mode = True          # 开启调试模式，以绘制辅助信息
    cfg.global_view = True       # 是否使用全局视角
    one_step_only = False         # 是否只跑一步
    eval_interval = 100            # 每 100 个 batch 评估一次

    hydra_cfg = HydraConfig.get()
    cfg.log_output_dir = hydra_cfg.runtime.output_dir  # 使用 Hydra 日志输出目录

    # 打印配置确认
    print(OmegaConf.to_yaml(cfg))

    # === 初始化环境 ===
    base_env = NavigationEnv(cfg)
    
    # 启用渲染 (这对录像至关重要)
    base_env.enable_render(True)

    # === Transforms (保持与 train.py 一致) ===
    controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=True)
    primers_dict = {
        # 给出一个key为recurrent_state的spec， primer根据此在 env.reset() 时创建对应的 tensordict 字段
        "recurrent_state": UnboundedContinuousTensorSpec(
                # shape=(batch, 1, hidden_dim),  # policy.gru_num_layers is set default to 1
                shape=(base_env.num_envs, 1, 256),
                device=cfg.device
            )
    }
    primer = TensorDictPrimer(primers=primers_dict, default_value=0.0)

    env = TransformedEnv(
        base_env, 
        Compose(
            InitTracker(),  # 跟踪初始化状态 (RNN)
            vel_transform,
            primer
        )
    ).train()
    
    # === 初始化 PPO (加载权重或随机) ===
    policy = PPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
    
    # primer = policy.get_recurrent_primer()
    # transformed_env.append_transform(primer)

    print("[DebugRunner] Environment structure.")
    print(env)

    print("[DebugRunner] Policy structure.")
    print(policy(env.reset()))

    def save_env_image(frame_idx: int):
        # === 保存帧用于检查 ===
        print("[DebugRunner] Capturing frame...")
        # 强制刷新一次渲染管线，确保画面是最新的
        base_env.sim.render() 
        # 获取 RGB 数据
        rgb_image = base_env.render(mode="rgb_array")
        
        if rgb_image is not None:
            # 检查维度，如果是 (3, H, W) 则转换为 (H, W, 3)
            if rgb_image.ndim == 3 and rgb_image.shape[0] == 3 and rgb_image.shape[2] != 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))
            
            # 使用 Hydra 的输出目录保存图片
            save_path = os.path.join(cfg.log_output_dir, f"debug_view_{frame_idx}.png")
            os.makedirs(cfg.log_output_dir, exist_ok=True)
            
            # 保存图片
            imageio.imwrite(save_path, rgb_image)
            print(f"[DebugRunner] Initialization frame saved to: {save_path}")
        else:
            print("[DebugRunner] Failed to capture frame. Check if renderer is enabled.")

    # === 同步数据采集器 ===
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
        total_frames=cfg.max_frame_num,
        return_same_td=True,
        device=cfg.device,
    )

    # === 数据统计器(torchrl) ===
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(in_keys=stats_keys)

    # === 评估函数 ===
    @torch.no_grad()
    def evaluate(seed: int=42):
        base_env.eval()
        env.eval()
        # 评估时，固定探索类型为确定性
        exploration_type = ExplorationType.MODE
        # 评估时，固定随机种子
        env.set_seed(seed)

        render_callback = RenderCallback(interval=1) # 每一帧记录渲染

        with set_exploration_type(exploration_type):
            # 手动进行一次完整的 rollout
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        save_env_image(collector._frames)
        env.reset()

        logging.info(f"[Eval] trajs keys: {trajs.keys()}")

        # 收集评估统计数据
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)
        
        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item() 
            for k, v in traj_stats.items()
        }
        logging.info(f"[Eval] eval info: {info}")

        # 保存评估视频
        video_path = os.path.join(cfg.log_output_dir, f"debug_eval_rollout_{collector._frames}_steps.mp4")
        logging.info(f"[Eval] Saving eval video to {video_path}")
        frames = render_callback.frames # 获取帧列表
        if len(frames) > 0:
            video_frames = []
            for f in frames:
                # Handle Torch Tensors
                if isinstance(f, torch.Tensor):
                    f = f.cpu().numpy()
                
                # Handle Numpy Arrays (frames: numpy.ndarray)
                # IsaacEnv render returns (H, W, 3), so we usually don't need to transpose.
                # Only transpose if we detect (3, H, W) structure.
                if f.ndim == 3 and f.shape[0] == 3 and f.shape[2] != 3:
                    f = np.transpose(f, (1, 2, 0))
                
                # Ensure uint8
                if f.dtype != np.uint8:
                    if f.max() <= 1.0:
                        f = (f * 255).astype(np.uint8)
                    else:
                        f = f.astype(np.uint8)
                
                video_frames.append(f)

            imageio.mimsave(video_path, video_frames, fps=30)
            logging.info("Video saved successfully.")
        else:
            logging.info("No frames captured!")
        
        return info

    # === 主训练循环 ===
    for i, data in enumerate(collector):
        # data: TensorDict 包含采集到的一个 batch 的数据
        info = {
            "batch": i,
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
        }

        if i == 0:  # test save image at second batch
            save_env_image(collector._frames)
            if one_step_only:
                print("[DebugRunner] One step only mode, exiting after first step.")
                break


        episode_stats.add(data.to_tensordict())
        # 每当有足够的 episode 结束时，计算并更新以此统计数据
        if len(episode_stats) >= base_env.num_envs:
            stats = {}
            for k, v in episode_stats.pop().items(include_nested=True, leaves_only=True):
                key_name = k if isinstance(k, str) else "_".join(k)  # key可能是str或tuple
                stats[f"episode/{key_name}"] = torch.mean(v.float()).item()
            info.update(stats)

        # 进行一次策略更新
        training_infos = policy.train(data.to_tensordict())
        # 将策略网络内部的训练信息添加到 info 中
        info.update({f"ppo_train/{k}": v for k, v in training_infos.items()})

        # 每隔 eval_interval 评估一次
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            # 改回训练模式
            env.train()
            base_env.train()
        
        # 记录当前信息到log
        # logging.info(f"[DebugRunner] Batch {i} info: \n {info}")

    sim_app.close()

if __name__ == "__main__":
    main()