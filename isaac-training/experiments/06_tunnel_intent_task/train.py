import os

# === Single GPU Mode ===
# Isaac Sim/PhysX requires all tensors on the same GPU device.
# For single-process training, we force a specific GPU.

# check gpu number on the machine
import torch
num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[Multi GPU Detected] CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0")
    else:
        print(f"[Multi GPU Detected] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("[Single GPU] Single GPU detected, no need to set CUDA_VISIBLE_DEVICES")

# Set memory allocation config to reduce fragmentation during rendering
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ========================

import logging
import hydra
import datetime
import wandb
import imageio
import numpy as np
from omegaconf import OmegaConf

# Updated imports to use 'src' package instead of sys.path hacks
# Ensure you have installed the project in editable mode: pip install -e .
from src.core.profiler import get_profiler, reset_profiler
from omni_drones import init_simulation_app 

from hydra.core.hydra_config import HydraConfig
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback
from torchrl.data import Unbounded
from src.experiment_utils import load_trajectory_dataset, resolve_constrained_policy

# Configs are now in the 'configs' directory
@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    # Start Simulation App
    sim_app = init_simulation_app(cfg)  # headless option is configurable via cfg

    # Import environment and algorithm (must after sim_app is instantiated)
    from src.envs.env_tunnel_intent import EnvTunnelIntent
    ConstrainedResidualPPO, algo_label = resolve_constrained_policy(cfg)
    print(f"[Train] Using {algo_label}")

    # === Special Mode Configuration ===
    profiling_mode = cfg.get("profiling_mode", False)  # Enable via CLI: profiling_mode=true
    env_test_mode = cfg.get("env_test_mode", False) # Enable via CLI: env_test_mode=true
    profiling_batches = cfg.get("profiling_batches", 10)  # Number of batches to profile
    if profiling_mode or env_test_mode:
        cfg.wandb.mode = "disabled"  # Disable wandb in profiling mode
    if env_test_mode:
        cfg.env.max_episode_length = 10  # Short episodes for quick testing

    # Use Wandb to monitor training
    # Convert OmegaConf to dict to avoid serialization errors with wandb/dataclasses
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb_group = cfg.wandb.get("group", None)

    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            group=wandb_group,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            group=wandb_group,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # === Profiling & Training Length Logic ===
    if profiling_mode:
        print("[Train] === PROFILING MODE ENABLED ===")
        print(f"[Train] Will run {profiling_batches} batches for profiling analysis")
        
        # In profiling mode, we override certain parameters
        MAX_FRAME_NUM = cfg.algo.training_frame_num * cfg.env.num_envs * profiling_batches
        eval_interval = 0  
        save_interval = profiling_batches + 1
        # warmup_iterations = 0 # Removed
    else:
        # Normal Training Mode: Load parameters from Hydra Config
        print("[Train] Starting Tunnel Intent Environment...")
        
        # Calculate total frames: frames_per_batch * num_envs * total_batches
        max_iterations = cfg.get("max_iterations", 20010)
        # warmup_iterations = cfg.algo.get("warmup_iterations", 0) # Removed in constrained approach

        MAX_FRAME_NUM = cfg.algo.training_frame_num * cfg.env.num_envs * max_iterations
        
        eval_interval = cfg.get("eval_interval", 500)
        save_interval = cfg.get("save_interval", 500)

    hydra_cfg = HydraConfig.get()
    cfg.log_output_dir = hydra_cfg.runtime.output_dir  # 使用 Hydra 日志输出目录

    # === Initialize Profiler ===
    profiler_log_file = os.path.join(cfg.log_output_dir, "profiler.log") if profiling_mode else None
    profiler = get_profiler(
        enabled=profiling_mode,
        cuda_sync=True,
        device=cfg.device,
        log_file=profiler_log_file
    )

    # 打印配置确认
    if env_test_mode:
        print("=== ENV TEST MODE ENABLED ===")
        print("This mode is for quick environment sanity checks. Training and evaluation will be skipped.")
    else:
        print("=== CONFIGURATION ===")
        print(OmegaConf.to_yaml(cfg))

    # === Load Trajectory Dataset (if offline mode enabled) ===
    trajectory_dataset = load_trajectory_dataset(cfg)

    # === 初始化环境 ===
    env = EnvTunnelIntent(cfg, trajectory_dataset=trajectory_dataset)
    # env.enable_render(True)
    # === 初始化 PPO ===
    # === CHANGED: Use ConstrainedResidualPPO ===
    policy = ConstrainedResidualPPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)

    # === Resume from checkpoint (for multi-stage curriculum) ===
    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt is not None:
        print(f"[Train] Loading checkpoint: {resume_ckpt}")
        state_dict = torch.load(resume_ckpt, map_location=cfg.device)
        policy.load_state_dict(state_dict)
        print(f"[Train] Checkpoint loaded successfully.")

    print("[Train] Environment structure.")
    print(env)

    print("[Train] Policy structure.")
    print(policy(env.reset()))

    def save_env_image(frame_idx: int):
        # === 保存帧用于检查 ===
        if cfg.get("headless", False):
            print("[Train] Skipping frame capture in headless mode.")
            return

        print("[Train] Capturing frame...")
        try:
            # 强制刷新一次渲染管线，确保画面是最新的
            env.sim.render()
            # TODO: 此函数仅在evaluate时至少渲染过一遍视频后起效，否则抛出无效的rgb数据错误，原因未知（
            rgb_image = env.render(mode="rgb_array")
        except Exception as exc:
            print(f"[Train] Failed to capture frame: {exc}")
            return

        if rgb_image is None or getattr(rgb_image, "ndim", 0) != 3:
            print("[Train] Renderer did not return a valid RGB frame; skipping save.")
            return

        # 检查维度，如果是 (3, H, W) 则转换为 (H, W, 3)
        if rgb_image.shape[0] == 3 and rgb_image.shape[2] != 3:
            rgb_image = np.transpose(rgb_image, (1, 2, 0))

        # 使用 Hydra 的输出目录保存图片
        save_path = os.path.join(cfg.log_output_dir, f"debug_view_{frame_idx}.png")
        os.makedirs(cfg.log_output_dir, exist_ok=True)

        # 保存图片
        imageio.imwrite(save_path, rgb_image)
        print(f"[Train] Initialization frame saved to: {save_path}")

    # === 同步数据采集器 ===
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
        total_frames=MAX_FRAME_NUM,
        return_same_td=True,
        device=cfg.device,
    )

    # === 数据统计器(torchrl) ===
    stats_keys = [
        k for k in env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(in_keys=stats_keys)

    # === 评估函数 ===
    @torch.no_grad()
    def evaluate(seed: int=42):
        # Configurable video recording
        record_video = cfg.get("record_video", False)
        print(f"[Eval] Starting evaluation... Video recording: {record_video}")
        
        env.eval()
        
        if record_video:
            print("[Eval] Enabling renderer and warming up...")
            # 1. 开启渲染
            env.enable_render(True)
            # 2. 只显示 env_0 的无人机，隐藏其余环境（物理不受影响）
            env.set_envs_visibility(visible_env_ids={0})
            # 3. 渲染预热
            for _ in range(10):
                env.sim.render()
        else:
            # 确保关闭渲染
            env.enable_render(False)
 
        # With TanhNormal, `mean`/`mode` may not be analytically available in TorchRL.
        # Use DETERMINISTIC interaction type to obtain a deterministic action.
        exploration_type = ExplorationType.DETERMINISTIC
        env.set_seed(seed)

        eval_max_steps = int(env.max_episode_length)
        
        if cfg.get("eval_visualization", False):
            env.set_visualization(enabled=True)

        def run_rollout_with_camera(camera_mode: str):
            """Run a rollout with specified camera mode and return frames + trajs"""
            render_callback = None
            
            if record_video:
                env.set_camera_view_mode(camera_mode)
                # Record every r frame (r>1) to reduce gpu memory usage
                render_callback = RenderCallback(interval=2)

            with set_exploration_type(exploration_type):
                if record_video:
                    try:
                        trajs = env.rollout(
                            max_steps=eval_max_steps,
                            policy=policy,
                            callback=render_callback,
                            auto_reset=True,
                            break_when_any_done=False,
                            return_contiguous=False,
                        )
                    except Exception as e:
                        print(f"[Eval] detailed rendering error: {e}")
                        # Fallback if rendering fails
                        trajs = env.rollout(
                            max_steps=eval_max_steps,
                            policy=policy,
                            callback=None, 
                            auto_reset=True,
                            break_when_any_done=False,
                            return_contiguous=False,
                        )
                else:
                     # No video, fast rollout
                     trajs = env.rollout(
                        max_steps=eval_max_steps,
                        policy=policy,
                        callback=None,
                        auto_reset=True,
                        break_when_any_done=False,
                        return_contiguous=False,
                    )
            env.reset()
            return render_callback, trajs

        logging.info("[Eval] Running rollout...")
        render_callback_follow, trajs = run_rollout_with_camera('follow')

        # === Rollout 2: Global camera view (Only if recording is enabled) ===
        render_callback_global = None
        if record_video and cfg.get("global_view", False):
            logging.info("[Eval] Running rollout with global camera view...")
            render_callback_global, _ = run_rollout_with_camera('global')

        logging.info(f"[Eval] trajs keys: {trajs.keys()}")

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)
        
        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {}
        for k, v in traj_stats.items():
            v_mean = torch.mean(v.float(), dim=0)
            if v_mean.numel() == 1:
                info["eval/" + k] = v_mean.item()
            else:
                clean = k
                for p in ["debug_", ""]:
                    if clean.startswith(p) and p:
                        clean = clean[len(p):]
                        break
                for suffix, val in zip(["x", "y", "z", "w"][:v_mean.numel()], v_mean.reshape(-1)):
                    info[f"eval_debug/{clean}/{suffix}"] = val.item()
        logging.info(f"[Eval] eval info: {info}")
        
        env.set_visualization(enabled=False)

        if record_video and render_callback_follow is not None:
            video_fps = 0.5 / (cfg.sim.dt * cfg.sim.substeps)
            info["recording_follow"] = wandb.Video(
                render_callback_follow.get_video_array(axes="t c h w"), 
                fps=video_fps, 
                format="mp4"
            )
            logging.info("[Eval] Follow camera video saved to wandb")
            
            if render_callback_global is not None:
                info["recording_global"] = wandb.Video(
                    render_callback_global.get_video_array(axes="t c h w"), 
                    fps=video_fps, 
                    format="mp4"
                )
                logging.info("[Eval] Global camera video saved to wandb")
        
        # 3. 评估结束，恢复所有环境可见性并关闭渲染
        if record_video:
            print("[Eval] Evaluation done. Restoring visibility and disabling renderer.")
            env.set_envs_visibility(visible_env_ids=None)  # 恢复所有环境可见
            env.enable_render(False)

        env.set_visualization(enabled=False)
        
        env.train()
        return info

    # === 初始化零点验证 (Sanity Check) ===
    # === Curriculum Scheduler (Phase 2: reg_coeff ramp) ===
    curriculum_enabled = cfg.get("curriculum", {}).get("enable", False)
    reg_scheduler = None
    if curriculum_enabled:
        from src.core.curriculum import RegCoeffScheduler
        reg_scheduler = RegCoeffScheduler(cfg.curriculum)
        print(f"[Train] Curriculum ENABLED: reg_coeff will ramp from "
              f"{cfg.curriculum.initial_reg_coeff} to {cfg.curriculum.max_reg_coeff}")

    # === Best Checkpoint Tracking ===
    best_eval_success = -1.0
    best_policy_state = None
    latest_eval_success = None  # Cached eval success for curriculum scheduler

    # === Early Stopping ===
    es_cfg = cfg.get("early_stopping", {})
    early_stopping_enabled = es_cfg.get("enable", False)
    es_patience = es_cfg.get("patience", 5)
    es_min_delta = es_cfg.get("min_delta", 0.10)
    es_degradation_count = 0
    if early_stopping_enabled:
        print(f"[Train] Early stopping ENABLED: patience={es_patience}, min_delta={es_min_delta}")

    print("[Sanity Check] Running Zero-Shot Verification...")
    env.eval() 
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        policy(td)
        
        net_output_norm = td["agents", "action_normalized"]
        human_input_phys = td["agents", "observation", "human_action"]
        human_input_norm = human_input_phys / cfg.algo.actor.action_limit

        # Diff between network output (normalized) and human input (normalized)
        # In this architecture, net_output_norm is the FINAL action.
        # Human input is injected via Residual Module.
        # If residuals are 0, then net_output_norm should be equal to human_input_norm
        # (assuming 1-to-1 mapping via residual scale=1.0)
        
        diff = (net_output_norm - human_input_norm).norm(dim=-1).mean()

        print(f"[Sanity Check] Initial Mean Error (Norm Space): {diff.item():.6f}")
        
        if diff.item() < 1e-2:
            print("✅ Initialization SUCCESS: Network starts as Identity Mapping.")
        else:
            print(f"❌ Initialization WARNING: Initial error is large ({diff.item()}).")
            print(f"   Sample Net Out: {net_output_norm[0]}")
            print(f"   Sample Human In: {human_input_norm[0]}")
    env.train()
    env.reset()

    # === 主训练循环 ===
    import time as time_module
    batch_start_time = time_module.perf_counter()
    
    for i, data in enumerate(collector):
        # === CHANGED: No Warmup Scale needed ===
        # Constrained Optimization handles the balance dynamically.
        # We implicitly assume residual_scale is fixed at 1.0 (default in init)
        
        # === Profiling: Measure batch timing ===
        batch_elapsed = time_module.perf_counter() - batch_start_time
        profiler.record("batch_total", batch_elapsed)
        profiler.increment_batch()
        batch_start_time = time_module.perf_counter()
        
        info = {
            "batch": i,
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
        }

        # 进行一次策略更新
        with profiler.timer("ppo_train_op"):
            training_infos = policy.train_op(data.to_tensordict())
        info.update({f"ppo_train/{k}": v for k, v in training_infos.items()})

        # 收集 episode 统计数据
        with profiler.timer("episode_stats"):
            episode_stats.add(data.to_tensordict())
            if len(episode_stats) >= env.num_envs:
                stats = {}
                for k, v in episode_stats.pop().items(include_nested=True, leaves_only=True):
                    key_name = k if isinstance(k, str) else "_".join(k)  # key可能是str或tuple
                    v_mean = torch.mean(v.float(), dim=0)  # mean across episodes, keep vector dims
                    if v_mean.numel() == 1:
                        stats[f"episode/{key_name}"] = v_mean.item()
                    else:
                        # Multi-dim stats -> group under debug/ section for cleaner wandb dashboard
                        clean = key_name
                        for p in ["stats_debug_", "stats_"]:
                            if clean.startswith(p):
                                clean = clean[len(p):]
                                break
                        for suffix, val in zip(["x", "y", "z", "w"][:v_mean.numel()], v_mean.reshape(-1)):
                            stats[f"debug/{clean}/{suffix}"] = val.item()
                info.update(stats)

        # 每隔 eval_interval 评估一次 (BEFORE curriculum update, so eval data is available)
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            eval_info = evaluate()
            info.update(eval_info)
            print(f"[Train] Eval info at step {collector._frames}: DONE")

            # Cache latest eval success rate for curriculum scheduler
            _latest_eval_success = eval_info.get("eval/stats_success", None)
            if _latest_eval_success is not None:
                latest_eval_success = _latest_eval_success

            if env_test_mode:
                save_env_image(i)
                print("[Training Loop] env_test_mode activated, exiting after evaluation.")
                break

            # === Best Checkpoint Tracking ===
            current_success = eval_info.get("eval/stats_success", -1.0)
            if current_success > best_eval_success:
                best_eval_success = current_success
                best_policy_state = {k: v.clone() for k, v in policy.state_dict().items()}
                # Save best checkpoint to disk
                best_save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
                os.makedirs(best_save_dir, exist_ok=True)
                best_ckpt_path = os.path.join(best_save_dir, "checkpoint_best.pt")
                torch.save(best_policy_state, best_ckpt_path)
                es_degradation_count = 0
                print(f"[Train] 🏆 New best model! success={current_success:.3f} at step {i}")
            elif early_stopping_enabled and best_eval_success > 0:
                # Check for performance degradation
                if current_success < best_eval_success - es_min_delta:
                    es_degradation_count += 1
                    print(f"[Train] ⚠️ Performance drop: {current_success:.3f} vs best {best_eval_success:.3f} "
                          f"(degradation {es_degradation_count}/{es_patience})")
                    if es_degradation_count >= es_patience:
                        print(f"[Train] 🛑 Early stopping triggered! Restoring best model (success={best_eval_success:.3f})")
                        policy.load_state_dict(best_policy_state)
                        break
                else:
                    es_degradation_count = 0

        # === Curriculum: update reg_coeff based on eval success_rate ===
        if reg_scheduler is not None and i % reg_scheduler.check_interval == 0:
            if latest_eval_success is not None:
                new_reg = reg_scheduler.update(latest_eval_success)
                policy.set_reg_coeff(new_reg)
                info["curriculum/reg_coeff"] = new_reg
                info["curriculum/ema_success"] = reg_scheduler.ema_success

        # === Profiling: Log timing stats to wandb ===
        if profiling_mode and i > 0 and i % 5 == 0:
            profiler.log_to_wandb(run)
        
        # 记录到 Wandb
        run.log(info)

        # Save Model (skip in profiling mode or if wandb is disabled)
        if i % save_interval == 0 and not profiling_mode:
            save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[RunnerSimple]: model saved at training step: ", i)

    # === Save final checkpoint ===
    if not profiling_mode:
        save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
        os.makedirs(save_dir, exist_ok=True)
        final_ckpt_path = os.path.join(save_dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), final_ckpt_path)
        print(f"[Train] Final checkpoint saved: {final_ckpt_path}")
        # Write path to a marker file so pipeline scripts can find it
        marker_path = os.path.join(cfg.log_output_dir, "final_checkpoint_path.txt")
        with open(marker_path, "w") as f:
            f.write(final_ckpt_path)

        # Write best checkpoint marker (preferred by curriculum pipeline)
        best_marker_path = os.path.join(cfg.log_output_dir, "best_checkpoint_path.txt")
        if best_policy_state is not None:
            best_save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
            best_ckpt_path = os.path.join(best_save_dir, "checkpoint_best.pt")
            torch.save(best_policy_state, best_ckpt_path)
            with open(best_marker_path, "w") as f:
                f.write(best_ckpt_path)
            print(f"[Train] Best checkpoint saved: {best_ckpt_path} (success={best_eval_success:.3f})")
        else:
            # No eval was run; fall back to final
            with open(best_marker_path, "w") as f:
                f.write(final_ckpt_path)
            print(f"[Train] No eval run; best checkpoint marker points to final.")

    if profiling_mode:
        profiler.print_summary()
        profiler.log_to_wandb(run)
        print(f"[Train] Profiling complete. Log saved to: {profiler_log_file}")

    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
