
import os
import torch
import hydra
import logging
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Import utilities
from omni_drones import init_simulation_app
from src.core.profiler import get_profiler

# Note: We need to import environment and policy AFTER simulation app starts
# inside the main function usually, but for type hinting we might want them here.
# To be safe with Isaac Sim's python path handling, we import inside main.

@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    # 1. Start Simulation App
    # Ensure headless for speed unless debug set
    cfg.headless = True 
    sim_app = init_simulation_app(cfg)

    # 2. Import Environment & Policy
    from src.envs.env_residual import FollowingEnvResidual
    from src.algos.ppo_constrained import ConstrainedResidualPPO
    from omni_drones.utils.torchrl import SyncDataCollector

    print("\n" + "="*80)
    print(" 🧪 Reward Threshold Estimation Tool")
    print("="*80 + "\n")

    # 3. Initialize Environment
    # We turn off expensive things for estimation if possible, but keep dynamics same
    print(f"[Estimate] Initializing Environment...")
    
    # Load trajectory dataset if needed (same as train.py)
    trajectory_dataset = None
    if cfg.user_model.get("offline_mode", False):
        from src.datasets.trajectory_dataset import TrajectoryDataset
        dataset_path = cfg.user_model.get("dataset_path", None)
        if dataset_path and os.path.exists(dataset_path):
            trajectory_dataset = TrajectoryDataset(
                dataset_path=dataset_path,
                device=torch.device(cfg.device),
                preload_data=cfg.user_model.get("preload_data", True)
            )

    env = FollowingEnvResidual(cfg, trajectory_dataset=trajectory_dataset)
    
    # 4. Initialize Policy
    policy = ConstrainedResidualPPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
    
    # =========================================================================
    # Phase 1: Measure Baseline (User Policy / Residual = 0)
    # =========================================================================
    print("\n[Phase 1] Measuring Baseline Reward (Residual = 0)...")
    print("          (This simulates the unassisted User/Human performance)")
    
    # To simulate "Residual = 0", we force the residual scale to 0.0
    policy.set_residual_scale(0.0) 
    
    # We need to run evaluation loops
    num_eval_episodes = 20 # Collect enough data
    # Calculate frames needed: approx num_envs * episode_len * episodes
    # But SyncDataCollector works by batches.
    
    # Let's use the collector to gather data
    frames_per_batch = cfg.env.num_envs * cfg.algo.training_frame_num
    total_frames = cfg.env.num_envs * cfg.env.max_episode_length * 5 # Approx 5 full rollouts per env
    
    baseline_rewards = []
    
    # Use collector
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        return_same_td=True,
        device=cfg.device,
        split_trajs=False # We want raw steps
    )

    pbar = tqdm(total=total_frames, desc="Collecting Baseline Data")
    for i, data in enumerate(collector):
        # data is a TensorDict of shape [num_envs, frames_per_batch]
        # Get rewards: [next, agents, reward]
        rewards = data["next", "agents", "reward"] # [N, T, 1]
        baseline_rewards.append(rewards.mean().item())
        pbar.update(data.numel())
    pbar.close()
    collector.shutdown()

    avg_baseline_reward = np.mean(baseline_rewards)
    std_baseline_reward = np.std(baseline_rewards)

    print(f"\n✅ Baseline Measurement Complete:")
    print(f"   Average Reward per Step: {avg_baseline_reward:.5f} (± {std_baseline_reward:.5f})")

    # =========================================================================
    # Phase 2: Probe Potential (Pure PPO training)
    # =========================================================================
    print("\n[Phase 2] Probing Potential Reward (Pure PPO Training)...")
    print("          (Training for 20 iterations ignoring regularization to find 'Good' performance)")
    
    # Reset Environment for training (technically not needed but good for clean slate)
    env.reset()
    policy = ConstrainedResidualPPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
    
    # HACK: To make it "Pure PPO", we set lambda to a very large fixed value ??
    # No, in ConstrainedResidualPPO, if we want to ignore Regularization Loss and minimize Task Loss only:
    # loss_pi = (reg_loss + lambda * actor_loss) / (1 + lambda)
    # If lambda -> infinity, loss_pi -> actor_loss.
    # So we can manually set log_lambda to a large number.
    with torch.no_grad():
        policy.lambda_param.fill_(1000.0) # softplus(1000) ~= 1000. Reg_loss is negligible.
        # Also, we should disable the lambda optimizer to keep it fixed high?
        # Or just let it train. If we set reward_threshold very high (e.g. +100), lambda will naturally grow.
        policy.reward_threshold = 999.0 # Impossible reward -> Lambda will keep growing -> Pure PPO
    
    # Create new collector for training
    train_collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * 15, # Train for 15 batches
        return_same_td=True,
        device=cfg.device,
    )
    
    training_rewards = []
    pbar_train = tqdm(total=15, desc="Quick Training Probe")
    
    for i, data in enumerate(train_collector):
        # Train
        train_info = policy.train_op(data.to_tensordict())
        
        # Record the avg reward of this batch (before update)
        # Note: 'data' contains the experience collected with OLD policy
        current_reward = data["next", "agents", "reward"].mean().item()
        training_rewards.append(current_reward)
        
        pbar_train.set_postfix({"reward": f"{current_reward:.4f}"})
        pbar_train.update(1)
    
    pbar_train.close()
    train_collector.shutdown()

    peak_reward = np.max(training_rewards[-5:]) # Max of last 5 batches
    avg_peak_reward = np.mean(training_rewards[-5:]) # Mean of last 5 batches

    print(f"\n✅ Potential Probe Complete:")
    print(f"   Initial Reward: {training_rewards[0]:.5f}")
    print(f"   Final Reward (approx): {avg_peak_reward:.5f}")
    print(f"   Peak Reward observed: {peak_reward:.5f}")

    # =========================================================================
    # Recommendation
    # =========================================================================
    print("\n" + "="*80)
    print(" 📊 RECOMMENDATION ")
    print("="*80)
    
    print(f"1. Baseline Performance (User Only): {avg_baseline_reward:.4f}")
    print(f"2. Validated Standard RL Performance: {avg_peak_reward:.4f}")
    
    gap = avg_peak_reward - avg_baseline_reward
    
    if gap <= 0:
        print("\n⚠️  WARNING: Standard RL did not outperform Baseline in this short test.")
        print("   This might mean the task is very hard, or the User Policy is already optimal.")
        print("   Recommendation: Set Threshold slightly above Baseline to enforce safety constraints.")
        rec_threshold = avg_baseline_reward + 0.1 * abs(avg_baseline_reward) # Heuristic
    else:
        print(f"\n   Performance Gap detected: {gap:.4f}")
        print("   We want the Constrained Agent to perform better than Baseline, but we don't need it to be Perfect.")
        
        # Conservative (Start safe): Baseline + 20% of Gap
        rec_conservative = avg_baseline_reward + 0.2 * gap
        # Aggressive (Push for performance): Baseline + 50% of Gap
        rec_aggressive = avg_baseline_reward + 0.5 * gap
        
        print("\n   [Suggested Reward Thresholds]:")
        print(f"   🔹 Conservative (Easy): {rec_conservative:.4f} (Baseline + 20% Gap)")
        print(f"   🔹 Balanced (Recommended): {rec_conservative + 0.1 * gap :.4f} (Baseline + 30% Gap)")
        print(f"   🔹 Aggressive (Hard):   {rec_aggressive:.4f} (Baseline + 50% Gap)")
    
    print("\n   Check 'config/experiment/...' and update 'reward_threshold' accordingly!")
    print("="*80)

    sim_app.close()

if __name__ == "__main__":
    main()
