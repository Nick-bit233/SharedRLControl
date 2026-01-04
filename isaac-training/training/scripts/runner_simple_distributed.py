"""
Multi-GPU Distributed Training Runner for Isaac Sim RL Training

Usage:
    # Multi-GPU with the launcher script:
    ./run_distributed.sh 4
    
    # Or manually with environment variable per GPU:
    CUDA_VISIBLE_DEVICES=0 python runner_simple_distributed.py distributed.rank=0 distributed.world_size=4 &
    CUDA_VISIBLE_DEVICES=1 python runner_simple_distributed.py distributed.rank=1 distributed.world_size=4 &
    CUDA_VISIBLE_DEVICES=2 python runner_simple_distributed.py distributed.rank=2 distributed.world_size=4 &
    CUDA_VISIBLE_DEVICES=3 python runner_simple_distributed.py distributed.rank=3 distributed.world_size=4 &

Note: Isaac Sim requires each process to see only ONE GPU via CUDA_VISIBLE_DEVICES.
      We use manual process spawning instead of torchrun for better compatibility.
"""

import os
import sys
import logging
import hydra
import datetime
import wandb
import torch
import torch.distributed as dist
import imageio
import numpy as np
from omegaconf import OmegaConf, DictConfig
from profiler import get_profiler, reset_profiler

from hydra.core.hydra_config import HydraConfig
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.data import Unbounded

FILE_PATH = os.path.join(os.path.dirname(__file__), "../cfg")


def setup_logging(log_dir: str, rank: int):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_rank{rank}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (only for rank 0 or errors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


def setup_distributed(cfg: DictConfig):
    """Initialize distributed training environment using manual configuration."""
    # Check for manual distributed config (preferred for Isaac Sim)
    if hasattr(cfg, 'distributed') and cfg.distributed.get('enabled', False):
        rank = cfg.distributed.rank
        world_size = cfg.distributed.world_size
        master_addr = cfg.distributed.get('master_addr', 'localhost')
        master_port = cfg.distributed.get('master_port', 29500)
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Initialize process group
        dist.init_process_group(
            backend="gloo",  # Use gloo instead of nccl for better compatibility
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank
        )
        
        logging.info(f"Initialized distributed: rank {rank}/{world_size}")
        return True, rank, world_size
    
    # Check for torchrun environment
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group(backend="gloo", init_method="env://")
        logging.info(f"Initialized from torchrun: rank {rank}/{world_size}")
        return True, rank, world_size
    
    else:
        # Single process mode
        logging.info("Running in single GPU mode")
        return False, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value, op=dist.ReduceOp.SUM, average=True):
    """Reduce a scalar value across all processes."""
    if not dist.is_initialized():
        return value
    
    tensor = torch.tensor(value, dtype=torch.float32)
    dist.all_reduce(tensor, op=op)
    if average:
        tensor /= dist.get_world_size()
    return tensor.item()


def broadcast_state_dict(state_dict, src=0):
    """Broadcast model state dict from src rank to all other ranks."""
    if not dist.is_initialized():
        return state_dict
    
    # Serialize state dict on src rank, broadcast size, then broadcast data
    if dist.get_rank() == src:
        buffer = torch.ByteTensor(list(torch.save(state_dict, '/dev/null', _use_new_zipfile_serialization=False) or b''))
    
    # Simple approach: just sync parameters one by one
    for key in state_dict:
        dist.broadcast(state_dict[key], src=src)
    
    return state_dict


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Ensure CUDA_VISIBLE_DEVICES is set (should be done by launcher)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logging.warning("CUDA_VISIBLE_DEVICES not set! Defaulting to GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    # Get distributed config before any CUDA operations
    is_distributed = hasattr(cfg, 'distributed') and cfg.distributed.get('enabled', False)
    rank = cfg.distributed.rank if is_distributed else 0
    world_size = cfg.distributed.world_size if is_distributed else 1
    
    # Setup logging first
    hydra_cfg = HydraConfig.get()
    log_dir = hydra_cfg.runtime.output_dir
    log_file = setup_logging(log_dir, rank)
    
    logging.info(f"Starting process: rank={rank}, world_size={world_size}, GPU={gpu_id}")
    logging.info(f"Log file: {log_file}")
    
    # Force device to cuda:0 since each process only sees one GPU
    cfg.device = "cuda:0"
    cfg.sim.device = "cuda:0"
    
    # Import Isaac Sim AFTER setting CUDA_VISIBLE_DEVICES
    from omni_drones import init_simulation_app
    
    try:
        # Start Simulation App
        logging.info("Initializing Isaac Sim...")
        sim_app = init_simulation_app(cfg)
        logging.info("Isaac Sim initialized successfully")
        
        # Import after sim_app
        from env_simple import FollowingEnvSimple
        from ppo_simple import SimplePPO
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from omni_drones.utils.torchrl import RenderCallback, SyncDataCollector, EpisodeStats
        
        # Initialize distributed AFTER Isaac Sim
        is_distributed, rank, world_size = setup_distributed(cfg)
        is_main_process = (rank == 0)
        
        # Wandb (only main process)
        if is_main_process:
            wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            run = wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.name}/dist_{world_size}gpu/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                entity=cfg.wandb.entity,
                config=wandb_config,
                mode=cfg.wandb.mode,
                id=wandb.util.generate_id(),
            )
        else:
            run = None
        
        logging.info("Starting environment setup...")
        
        # === Configuration ===
        total_envs = 256
        cfg.env.num_envs = total_envs // world_size
        cfg.env.enable_lidar = False
        cfg.algo.rnn.enable = False
        cfg.algo.training_frame_num = 128
        cfg.max_frame_num = cfg.algo.training_frame_num * total_envs * 20000
        cfg.debug_mode = False
        cfg.global_view = True
        cfg.log_output_dir = log_dir
        eval_interval = 500
        save_interval = 500
        
        logging.info(f"Config: num_envs={cfg.env.num_envs}, total_envs={total_envs}")
        
        # Profiling
        profiling_mode = cfg.get("profiling_mode", False)
        profiler = get_profiler(enabled=profiling_mode, cuda_sync=True, device=cfg.device)
        
        if is_main_process:
            logging.info(OmegaConf.to_yaml(cfg))
        
        # === Initialize Environment ===
        logging.info("Creating environment...")
        base_env = FollowingEnvSimple(cfg)
        base_env.enable_render(is_main_process)
        logging.info("Environment created")
        
        # Transforms
        controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
        vel_transform = VelController(controller, yaw_control=True)
        
        env = TransformedEnv(
            base_env,
            Compose(vel_transform)
        ).train()
        
        # === Initialize Policy ===
        logging.info("Creating policy...")
        policy = SimplePPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
        logging.info("Policy created")
        
        # Sync initial parameters from rank 0
        if is_distributed:
            logging.info("Synchronizing model parameters...")
            for param in policy.parameters():
                dist.broadcast(param.data, src=0)
            logging.info("Parameters synchronized")
        
        # === Data Collector ===
        collector = SyncDataCollector(
            env,
            policy=policy,
            frames_per_batch=cfg.algo.training_frame_num * cfg.env.num_envs,
            total_frames=cfg.max_frame_num // world_size,
            return_same_td=True,
            device=cfg.device,
        )
        
        stats_keys = [
            k for k in base_env.observation_spec.keys(True, True)
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(in_keys=stats_keys)
        
        logging.info("Starting training loop...")
        
        # === Training Loop ===
        import time as time_module
        batch_start_time = time_module.perf_counter()
        
        for i, data in enumerate(collector):
            batch_elapsed = time_module.perf_counter() - batch_start_time
            profiler.record("batch_total", batch_elapsed)
            profiler.increment_batch()
            batch_start_time = time_module.perf_counter()
            
            info = {
                "batch": i,
                "env_frames": collector._frames * world_size,
                "rollout_fps": collector._fps * world_size,
            }
            
            # Episode stats
            episode_stats.add(data.to_tensordict())
            if len(episode_stats) >= base_env.num_envs:
                stats = {}
                for k, v in episode_stats.pop().items(include_nested=True, leaves_only=True):
                    key_name = k if isinstance(k, str) else "_".join(k)
                    val = torch.mean(v.float()).item()
                    if is_distributed:
                        val = reduce_scalar(val, average=True)
                    stats[f"episode/{key_name}"] = val
                info.update(stats)
            
            # Policy update
            training_infos = policy.train_op(data.to_tensordict())
            
            # Sync gradients (average across ranks)
            if is_distributed:
                for param in policy.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= world_size
            
            info.update({f"ppo_train/{k}": v for k, v in training_infos.items()})
            
            # Logging
            if is_main_process and run is not None:
                run.log(info)
            
            if i % 100 == 0:
                logging.info(f"Batch {i}, frames={collector._frames * world_size}")
            
            # Save model
            if i % save_interval == 0 and is_main_process:
                save_dir = log_dir
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"checkpoint_{i}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Model saved: {ckpt_path}")
        
        logging.info("Training completed!")
        
    except Exception as e:
        logging.exception(f"Error during training: {e}")
        raise
    finally:
        if is_main_process and run is not None:
            wandb.finish()
        cleanup_distributed()
        sim_app.close()


if __name__ == "__main__":
    main()

