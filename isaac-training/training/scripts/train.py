import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from utils import evaluate
from torchrl.envs.utils import ExplorationType
from torchrl.data import UnboundedContinuousTensorSpec


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # VelController transforms 4D force action space in omni_drones to desired 4D vel action space
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=True)
    
    # temp Transformed Env only used to init policy observation_spec / action_spec
    temp_transforms = []
    temp_transforms.append(InitTracker())  # InitTracker will add a "is_init" boolean mask in the TensorDict for tracking rnn states reset.
    # temp_transforms.append(vel_transform) # [DEBUG] as the VelController now has the same action spec size with base_env, no need to transform at ppo init.
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    temp_transformed_env = TransformedEnv(env, Compose(*temp_transforms)).train()
    temp_transformed_env.set_seed(cfg.seed)  

    # PPO Policy (with GRUModule as recurrent network)
    policy = PPO(cfg.algo, temp_transformed_env.observation_spec, temp_transformed_env.action_spec, cfg.device)
    # [DEBUG] print policy in TensorDict format
    print("\n" + "="*50)
    print("PPO Policy Network Structure:")
    print(policy(temp_transformed_env.reset()))
    print("="*50 + "\n")
    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))

    # Get GRU Primer Transform
    # Primer is a torchrl transform tells the environment to reset hidden states at the beginning of each episode
    # primer = policy.get_recurrent_primer()

    primers = {
        # 给出一个key为recurrent_state的spec， primer根据此在 env.reset() 时创建对应的 tensordict 字段
        "recurrent_state": UnboundedContinuousTensorSpec(
                # shape=(batch, 1, hidden_dim),  # policy.gru_num_layers is set default to 1
                shape=(env.num_envs, policy.gru_num_layers, policy.gru_hidden_dim),
                device=cfg.device
            )
    }
    # 创建 primer（显式）
    primer = TensorDictPrimer(primers=primers, default_value=0.0)

    # The actual Transformed Env used in training
    transforms = []
    transforms.append(InitTracker()) 
    transforms.append(primer)
    transforms.append(vel_transform)

    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # [DEBUG] print transformed_env info
    print("\n" + "="*50)
    print("VERIFYING TRANSFORMED ENVIRONMENT")
    td = transformed_env.reset()
    print("[DEBUG] transformed_env.batch_size:", transformed_env.batch_size)
    print("[DEBUG] keys:", td.keys(True, True))
    if "is_init" in td.keys(True, True):
        print("[DEBUG] is_init shape:", td.get("is_init").shape)
    if "recurrent_state" in td.keys(True, True):
        r = td.get("recurrent_state")
        print("[DEBUG] recurrent_state shape:", r.shape, " mean/std:", float(r.mean()), float(r.std()))
    print("="*50 + "\n")

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            env.enable_render(True)
            env.eval()
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.enable_render(not cfg.headless)
            env.train()
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    