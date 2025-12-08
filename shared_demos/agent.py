import torch
import sys
import os
from omegaconf import OmegaConf
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

# Add training scripts to path to import SimplePPO
sys.path.append("/workspace/NavRL/isaac-training/training/scripts")
from ppo_simple import SimplePPO

class Agent:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.policy = self._init_model(checkpoint_path)
        self.policy.eval()

    def _init_model(self, checkpoint_path):
        # Mock configuration required by SimplePPO
        # Structure matches 'algo' section of ppo.yaml, as runner passes cfg.algo
        cfg = OmegaConf.create({
            "rnn": {
                "enable": False,
                "gru_hidden_dim": 256,
                "gru_num_layers": 1
            },
            # Add 'algo' key to satisfy potential buggy access in ppo_simple.py (cfg.algo.rnn...)
            "algo": {
                "rnn": {
                    "gru_hidden_dim": 256,
                    "gru_num_layers": 1
                }
            },
            "actor": {
                "action_limit": 2.0,  # Updated to 2.0 based on ppo.yaml
                "learning_rate": 1e-4,
                "clip_ratio": 0.2
            },
            "critic": {
                "clip_ratio": 0.2,
                "learning_rate": 1e-4
            },
            "feature_extractor": {
                "learning_rate": 1e-4
            },
            "entropy_loss_coefficient": 0.0,
            "training_epoch_num": 4,
            "num_minibatches": 4
        })

        # Define Observation Spec (Must match training)
        # Based on env_simple.py
        drone_state_dim = 10
        prev_action_dim = 4
        human_action_dim = 4

        obs_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((drone_state_dim,), device=self.device),
                    "prev_action": UnboundedContinuousTensorSpec((prev_action_dim,), device=self.device),
                    "human_action": UnboundedContinuousTensorSpec((human_action_dim,), device=self.device),
                })
            }).expand(1)
        }, shape=[1], device=self.device)

        # Define Action Spec
        action_dim = 4 # [vx, vy, vz, yaw_rate]
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device),
                "action_normalized": UnboundedContinuousTensorSpec((action_dim,), device=self.device) # PPO needs this key in spec sometimes? No, usually just action.
            })
        }).expand(1, action_dim).to(self.device)

        # Initialize Policy
        policy = SimplePPO(cfg, obs_spec, action_spec, self.device)

        # Load Checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            policy.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

        return policy

    def act(self, drone_state, human_action, prev_action):
        """
        Args:
            drone_state: (10,) tensor [vel_b(3), ang_vel_b(3), orientation(4)]
            human_action: (4,) tensor [vx, vy, vz, yaw_rate]
            prev_action: (4,) tensor [vx, vy, vz, yaw_rate]
        Returns:
            action: (4,) numpy array
        """
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state.unsqueeze(0),       # (1, 10)
                    "human_action": human_action.unsqueeze(0), # (1, 4)
                    "prev_action": prev_action.unsqueeze(0)    # (1, 4)
                }, batch_size=[1])
            }, batch_size=[1])
        }, batch_size=[1], device=self.device)

        with torch.no_grad():
            output = self.policy(obs)
            action = output["agents", "action"][0].cpu().numpy()
        
        return action
