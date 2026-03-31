import os

# === Single GPU Mode ===
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ========================

import logging
import hydra
import datetime
import wandb
import imageio
import numpy as np
from omegaconf import OmegaConf

from src.core.profiler import get_profiler, reset_profiler
from omni_drones import init_simulation_app

from hydra.core.hydra_config import HydraConfig
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback
from torchrl.data import Unbounded


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    sim_app = init_simulation_app(cfg)

    # Import environment (must after sim_app is instantiated)
    from src.envs.env_safety_shield import EnvSafetyShield

    # Import PPO algorithm — select via cfg.algo.distribution
    algo_distribution = cfg.algo.get("distribution", "tanh_normal")
    if algo_distribution == "beta":
        from src.algos.ppo_constrained_beta import ConstrainedResidualPPO_Beta as ConstrainedResidualPPO
        print("[Train] Using Beta distribution PPO")
    else:
        from src.algos.ppo_constrained import ConstrainedResidualPPO
        print("[Train] Using TanhNormal distribution PPO")

    # === Special Mode Configuration ===
    profiling_mode = cfg.get("profiling_mode", False)
    env_test_mode = cfg.get("env_test_mode", False)
    profiling_batches = cfg.get("profiling_batches", 10)
    if profiling_mode or env_test_mode:
        cfg.wandb.mode = "disabled"
    if env_test_mode:
        cfg.env.max_episode_length = 10

    # wandb init
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_group = cfg.wandb.get("group", None)

    if cfg.wandb.run_id is None:
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

    # === Training Length ===
    if profiling_mode:
        print("[Train] === PROFILING MODE ENABLED ===")
        print(f"[Train] Will run {profiling_batches} batches for profiling analysis")
        MAX_FRAME_NUM = cfg.algo.training_frame_num * cfg.env.num_envs * profiling_batches
        eval_interval = 0
        save_interval = profiling_batches + 1
    else:
        print("[Train] Starting Safety Shield Environment...")
        max_iterations = cfg.get("max_iterations", 20010)
        MAX_FRAME_NUM = cfg.algo.training_frame_num * cfg.env.num_envs * max_iterations
        eval_interval = cfg.get("eval_interval", 500)
        save_interval = cfg.get("save_interval", 500)

    hydra_cfg = HydraConfig.get()
    cfg.log_output_dir = hydra_cfg.runtime.output_dir

    # === Initialize Profiler ===
    profiler_log_file = os.path.join(cfg.log_output_dir, "profiler.log") if profiling_mode else None
    profiler = get_profiler(
        enabled=profiling_mode,
        cuda_sync=True,
        device=cfg.device,
        log_file=profiler_log_file
    )

    if env_test_mode:
        print("=== ENV TEST MODE ENABLED ===")
    else:
        print("=== CONFIGURATION ===")
        print(OmegaConf.to_yaml(cfg))

    # === Initialize Environment ===
    env = EnvSafetyShield(cfg)

    # === Initialize PPO ===
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
        print("[Train] Capturing frame...")
        env.sim.render()
        rgb_image = env.render(mode="rgb_array")
        if rgb_image is not None:
            if rgb_image.ndim == 3 and rgb_image.shape[0] == 3 and rgb_image.shape[2] != 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))
            save_path = os.path.join(cfg.log_output_dir, f"debug_view_{frame_idx}.png")
            os.makedirs(cfg.log_output_dir, exist_ok=True)
            imageio.imwrite(save_path, rgb_image)
            print(f"[Train] Frame saved to: {save_path}")
        else:
            print("[Train] Failed to capture frame.")

    # === Data Collector ===
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
        total_frames=MAX_FRAME_NUM,
        return_same_td=True,
        device=cfg.device,
    )

    stats_keys = [
        k for k in env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(in_keys=stats_keys)

    # === Evaluate Function ===
    @torch.no_grad()
    def evaluate(seed: int = 42):
        record_video = cfg.get("record_video", False)
        print(f"[Eval] Starting evaluation... Video recording: {record_video}")

        env.eval()

        if record_video:
            print("[Eval] Enabling renderer and warming up...")
            env.enable_render(True)
            env.set_envs_visibility(visible_env_ids={0})
            for _ in range(10):
                env.sim.render()
        else:
            env.enable_render(False)

        exploration_type = ExplorationType.DETERMINISTIC
        env.set_seed(seed)

        eval_max_steps = int(env.max_episode_length)

        if cfg.get("eval_visualization", False):
            env.set_visualization(enabled=True)

        def run_rollout_with_camera(camera_mode: str):
            render_callback = None
            if record_video:
                env.set_camera_view_mode(camera_mode)
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
                        print(f"[Eval] rendering error: {e}")
                        trajs = env.rollout(
                            max_steps=eval_max_steps,
                            policy=policy,
                            callback=None,
                            auto_reset=True,
                            break_when_any_done=False,
                            return_contiguous=False,
                        )
                else:
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

        render_callback_global = None
        if record_video and cfg.get("global_view", False):
            logging.info("[Eval] Running rollout with global camera view...")
            render_callback_global, _ = run_rollout_with_camera('global')

        logging.info(f"[Eval] trajs keys: {trajs.keys()}")

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
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

        # Compute derived metrics for safety shield
        episode_len = info.get("eval/stats_episode_len", 1.0)
        if episode_len > 0:
            info["eval/tracking_rmse"] = info.get("eval/stats_tracking_error_sum", 0.0) / max(episode_len, 1.0)
            info["eval/intervention_mean"] = info.get("eval/stats_intervention_norm_sum", 0.0) / max(episode_len, 1.0)

        collision_rate = info.get("eval/stats_collision", 0.5)
        survival_rate = 1.0 - collision_rate
        info["eval/survival_rate"] = survival_rate
        info["eval/collision_rate"] = collision_rate

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

        if record_video:
            print("[Eval] Evaluation done. Restoring visibility and disabling renderer.")
            env.set_envs_visibility(visible_env_ids=None)
            env.enable_render(False)

        env.set_visualization(enabled=False)
        env.train()
        return info

    # === Curriculum Scheduler ===
    curriculum_enabled = cfg.get("curriculum", {}).get("enable", False)
    reg_scheduler = None
    if curriculum_enabled:
        from src.core.curriculum import RegCoeffScheduler
        reg_scheduler = RegCoeffScheduler(cfg.curriculum)
        print(f"[Train] Curriculum ENABLED: reg_coeff will ramp from "
              f"{cfg.curriculum.initial_reg_coeff} to {cfg.curriculum.max_reg_coeff}")

    # === Best Checkpoint Tracking (uses survival_rate as primary metric) ===
    best_eval_score = -1.0
    best_policy_state = None
    latest_eval_survival = None

    # === Early Stopping ===
    es_cfg = cfg.get("early_stopping", {})
    early_stopping_enabled = es_cfg.get("enable", False)
    es_patience = es_cfg.get("patience", 5)
    es_min_delta = es_cfg.get("min_delta", 0.10)
    es_degradation_count = 0
    if early_stopping_enabled:
        print(f"[Train] Early stopping ENABLED: patience={es_patience}, min_delta={es_min_delta}")

    # === Zero-Shot Sanity Check ===
    print("[Sanity Check] Running Zero-Shot Verification...")
    env.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        policy(td)

        net_output_norm = td["agents", "action_normalized"]
        human_input_phys = td["agents", "observation", "human_action"]
        human_input_norm = human_input_phys / cfg.algo.actor.action_limit

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

    # === Main Training Loop ===
    import time as time_module
    batch_start_time = time_module.perf_counter()

    for i, data in enumerate(collector):
        batch_elapsed = time_module.perf_counter() - batch_start_time
        profiler.record("batch_total", batch_elapsed)
        profiler.increment_batch()
        batch_start_time = time_module.perf_counter()

        info = {
            "batch": i,
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
        }

        # Policy update
        with profiler.timer("ppo_train_op"):
            training_infos = policy.train_op(data.to_tensordict())
        info.update({f"ppo_train/{k}": v for k, v in training_infos.items()})

        # Episode statistics
        with profiler.timer("episode_stats"):
            episode_stats.add(data.to_tensordict())
            if len(episode_stats) >= env.num_envs:
                stats = {}
                for k, v in episode_stats.pop().items(include_nested=True, leaves_only=True):
                    key_name = k if isinstance(k, str) else "_".join(k)
                    v_mean = torch.mean(v.float(), dim=0)
                    if v_mean.numel() == 1:
                        stats[f"episode/{key_name}"] = v_mean.item()
                    else:
                        clean = key_name
                        for p in ["stats_debug_", "stats_"]:
                            if clean.startswith(p):
                                clean = clean[len(p):]
                                break
                        for suffix, val in zip(["x", "y", "z", "w"][:v_mean.numel()], v_mean.reshape(-1)):
                            stats[f"debug/{clean}/{suffix}"] = val.item()
                info.update(stats)

        # Evaluation
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            eval_info = evaluate()
            info.update(eval_info)
            print(f"[Train] Eval at step {collector._frames}: "
                  f"survival={eval_info.get('eval/survival_rate', 0):.3f}, "
                  f"collision={eval_info.get('eval/collision_rate', 0):.3f}, "
                  f"tracking_rmse={eval_info.get('eval/tracking_rmse', -1):.4f}")

            # Cache latest survival rate for curriculum
            _latest_survival = eval_info.get("eval/survival_rate", None)
            if _latest_survival is not None:
                latest_eval_survival = _latest_survival

            if env_test_mode:
                save_env_image(i)
                print("[Training Loop] env_test_mode activated, exiting after evaluation.")
                break

            # === Best Checkpoint Tracking (primary: survival_rate) ===
            current_score = eval_info.get("eval/survival_rate", -1.0)
            if current_score > best_eval_score:
                best_eval_score = current_score
                best_policy_state = {k: v.clone() for k, v in policy.state_dict().items()}
                best_save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
                os.makedirs(best_save_dir, exist_ok=True)
                best_ckpt_path = os.path.join(best_save_dir, "checkpoint_best.pt")
                torch.save(best_policy_state, best_ckpt_path)
                es_degradation_count = 0
                print(f"[Train] 🏆 New best model! survival={current_score:.3f} at step {i}")
            elif early_stopping_enabled and best_eval_score > 0:
                if current_score < best_eval_score - es_min_delta:
                    es_degradation_count += 1
                    print(f"[Train] ⚠️ Performance drop: {current_score:.3f} vs best {best_eval_score:.3f} "
                          f"(degradation {es_degradation_count}/{es_patience})")
                    if es_degradation_count >= es_patience:
                        print(f"[Train] 🛑 Early stopping triggered! Restoring best model (survival={best_eval_score:.3f})")
                        policy.load_state_dict(best_policy_state)
                        break
                else:
                    es_degradation_count = 0

        # === Curriculum: update reg_coeff based on eval survival_rate ===
        if reg_scheduler is not None and i % reg_scheduler.check_interval == 0:
            if latest_eval_survival is not None:
                new_reg = reg_scheduler.update(latest_eval_survival)
                policy.set_reg_coeff(new_reg)
                info["curriculum/reg_coeff"] = new_reg
                info["curriculum/ema_survival"] = reg_scheduler.ema_success

        # Profiling stats
        if profiling_mode and i > 0 and i % 5 == 0:
            profiler.log_to_wandb(run)

        run.log(info)

        # Save checkpoint
        if i % save_interval == 0 and not profiling_mode:
            save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[Train]: model saved at training step: ", i)

    # === Save final checkpoint ===
    if not profiling_mode:
        save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
        os.makedirs(save_dir, exist_ok=True)
        final_ckpt_path = os.path.join(save_dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), final_ckpt_path)
        print(f"[Train] Final checkpoint saved: {final_ckpt_path}")
        marker_path = os.path.join(cfg.log_output_dir, "final_checkpoint_path.txt")
        with open(marker_path, "w") as f:
            f.write(final_ckpt_path)

        # Best checkpoint marker
        best_marker_path = os.path.join(cfg.log_output_dir, "best_checkpoint_path.txt")
        if best_policy_state is not None:
            best_save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
            best_ckpt_path = os.path.join(best_save_dir, "checkpoint_best.pt")
            torch.save(best_policy_state, best_ckpt_path)
            with open(best_marker_path, "w") as f:
                f.write(best_ckpt_path)
            print(f"[Train] Best checkpoint saved: {best_ckpt_path} (survival={best_eval_score:.3f})")
        else:
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
