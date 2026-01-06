"""
Multi-GPU Distributed Training Runner for Isaac Sim RL Training

Usage:
    # Single GPU (default):
    python runner_simple.py
    
    # Multi-GPU with torchrun (recommended):
    torchrun --nproc_per_node=4 runner_simple_distributed.py
    
    # Or with torch.distributed.run:
    python -m torch.distributed.run --nproc_per_node=4 runner_simple_distributed.py

Note: Each GPU runs its own Isaac Sim instance with separate environments.
      Gradients are synchronized across all GPUs after each training step.
"""

import os
import logging
import hydra
import datetime
import wandb
import torch
import torch.distributed as dist
import imageio
import numpy as np
from omegaconf import OmegaConf
from profiler import get_profiler, reset_profiler
from omni_drones import init_simulation_app

from hydra.core.hydra_config import HydraConfig
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback
from torchrl.data import Unbounded

FILE_PATH = os.path.join(os.path.dirname(__file__), "../cfg")


def setup_distributed():
    """Initialize distributed training environment."""
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set CUDA device for this process
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        print(f"[Distributed] Initialized rank {rank}/{world_size}, local_rank={local_rank}, device=cuda:{local_rank}")
        return True, rank, world_size, local_rank
    else:
        # Single GPU mode
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[Distributed] Running in single GPU mode")
        return False, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_gradients(model):
    """Synchronize gradients across all processes."""
    if not dist.is_initialized():
        return
    
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def sync_model_params(model):
    """Broadcast model parameters from rank 0 to all other ranks."""
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def reduce_dict(input_dict, average=True):
    """Reduce a dictionary of tensors across all processes."""
    if not dist.is_initialized():
        return input_dict
    
    world_size = dist.get_world_size()
    reduced_dict = {}
    
    for key, value in input_dict.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if average:
                tensor /= world_size
            reduced_dict[key] = tensor.item()
        else:
            reduced_dict[key] = value
    
    return reduced_dict


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Setup distributed training BEFORE any CUDA operations
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    # Update device in config based on local rank
    cfg.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cfg.sim.device = cfg.device
    
    # Start Simulation App (each process has its own)
    sim_app = init_simulation_app(cfg)

    # Import environment and algorithm (must after sim_app is instantiated)
    from env_simple import FollowingEnvSimple
    from ppo_simple import SimplePPO

    # Only main process logs to wandb to avoid duplicate logs
    if is_main_process:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_config["distributed"] = {
            "enabled": is_distributed,
            "world_size": world_size,
            "rank": rank
        }

        if cfg.wandb.run_id is None:
            run = wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.name}/distributed_{world_size}gpu/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                entity=cfg.wandb.entity,
                config=wandb_config,
                mode=cfg.wandb.mode,
                id=wandb.util.generate_id(),
            )
        else:
            run = wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                entity=cfg.wandb.entity,
                config=wandb_config,
                mode=cfg.wandb.mode,
                id=cfg.wandb.run_id,
                resume="must"
            )
    else:
        run = None  # Non-main processes don't log

    if is_main_process:
        print(f"[DistributedRunner] Starting with {world_size} GPUs...")

    # === Configuration Overrides ===
    # Scale down per-GPU environments to maintain total env count
    total_envs = 256
    cfg.env.num_envs = total_envs // world_size
    
    cfg.env.enable_lidar = False
    cfg.algo.rnn.enable = False
    cfg.algo.training_frame_num = 128
    cfg.max_frame_num = cfg.algo.training_frame_num * total_envs * 20000
    cfg.debug_mode = False
    cfg.global_view = True
    eval_interval = 500
    save_interval = 500

    # Profiling configuration
    profiling_mode = cfg.get("profiling_mode", False)
    profiling_batches = cfg.get("profiling_batches", 10)
    
    if profiling_mode:
        if is_main_process:
            print("[DistributedRunner] === PROFILING MODE ENABLED ===")
        cfg.max_frame_num = cfg.algo.training_frame_num * total_envs * profiling_batches
        eval_interval = 0
        save_interval = profiling_batches + 1

    hydra_cfg = HydraConfig.get()
    cfg.log_output_dir = hydra_cfg.runtime.output_dir

    profiler_log_file = os.path.join(cfg.log_output_dir, f"profiler_rank{rank}.log") if profiling_mode else None
    profiler = get_profiler(
        enabled=profiling_mode,
        cuda_sync=True,
        device=cfg.device,
        log_file=profiler_log_file
    )

    if is_main_process:
        print(OmegaConf.to_yaml(cfg))

    # === Initialize Environment ===
    base_env = FollowingEnvSimple(cfg)
    base_env.enable_render(is_main_process)  # Only main process renders

    # === Transforms ===
    controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=True)

    if cfg.algo.rnn.enable:
        primers_dict = {
            "recurrent_state": Unbounded(
                shape=(base_env.num_envs, 1, 256),
                device=cfg.device
            )
        }
        primer = TensorDictPrimer(primers=primers_dict, default_value=0.0)
        env = TransformedEnv(
            base_env,
            Compose(InitTracker(), vel_transform, primer)
        ).train()
    else:
        env = TransformedEnv(
            base_env,
            Compose(vel_transform)
        ).train()

    # === Initialize Policy ===
    policy = SimplePPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
    
    # Synchronize initial model parameters across all processes
    if is_distributed:
        sync_model_params(policy)
    
    if is_main_process:
        print("[DistributedRunner] Environment structure:")
        print(env)
        print("[DistributedRunner] Policy structure:")
        print(policy(env.reset()))

    # === Data Collector ===
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
        total_frames=cfg.max_frame_num // world_size,  # Divide total frames by world size
        return_same_td=True,
        device=cfg.device,
    )

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(in_keys=stats_keys)

    # === Evaluation Function (only on main process) ===
    @torch.no_grad()
    def evaluate(seed: int = 42):
        if not is_main_process:
            return {}
        
        base_env.eval()
        env.eval()
        exploration_type = ExplorationType.MODE
        env.set_seed(seed)
        render_callback = RenderCallback(interval=1)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )
        return info

    # === Main Training Loop ===
    import time as time_module
    batch_start_time = time_module.perf_counter()

    for i, data in enumerate(collector):
        batch_elapsed = time_module.perf_counter() - batch_start_time
        profiler.record("batch_total", batch_elapsed)
        profiler.increment_batch()
        batch_start_time = time_module.perf_counter()

        # Local stats
        info = {
            "batch": i,
            "env_frames": collector._frames * world_size,  # Total frames across all GPUs
            "rollout_fps": collector._fps * world_size,
        }

        with profiler.timer("episode_stats"):
            episode_stats.add(data.to_tensordict())
            if len(episode_stats) >= base_env.num_envs:
                stats = {}
                for k, v in episode_stats.pop().items(include_nested=True, leaves_only=True):
                    key_name = k if isinstance(k, str) else "_".join(k)
                    stats[f"episode/{key_name}"] = torch.mean(v.float()).item()
                info.update(stats)

        # Policy update with gradient synchronization
        with profiler.timer("ppo_train_op"):
            training_infos = policy.train_op(data.to_tensordict())
            
            # Synchronize gradients across all processes
            if is_distributed:
                sync_gradients(policy.feature_extractor)
                sync_gradients(policy.actor)
                sync_gradients(policy.critic)

        info.update({f"ppo_train/{k}": v for k, v in training_infos.items()})

        # Reduce stats across all processes (average)
        if is_distributed:
            info = reduce_dict(info, average=True)

        # Evaluation (only on main process)
        if eval_interval > 0 and i % eval_interval == 0:
            if is_distributed:
                dist.barrier()  # Sync before evaluation
            
            if is_main_process:
                logging.info(f"Eval at {collector._frames * world_size} steps.")
                eval_info = evaluate()
                info.update(eval_info)
                env.train()
                base_env.train()
                env.reset()
            
            if is_distributed:
                dist.barrier()  # Sync after evaluation

        # Logging (only main process)
        if is_main_process and run is not None:
            if profiling_mode and i > 0 and i % 5 == 0:
                profiler.log_to_wandb(run)
            run.log(info)

        # Save model (only main process)
        if i % save_interval == 0 and not profiling_mode and is_main_process:
            save_dir = run.dir if hasattr(run, 'dir') and run.dir and os.path.exists(run.dir) else cfg.log_output_dir
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[DistributedRunner] Model saved at training step: {i}")

    # Cleanup
    if profiling_mode and is_main_process:
        profiler.print_summary()
        if run is not None:
            profiler.log_to_wandb(run)

    if is_main_process and run is not None:
        wandb.finish()
    
    cleanup_distributed()
    sim_app.close()


if __name__ == "__main__":
    main()
