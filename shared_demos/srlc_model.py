import os
import sys
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, Unbounded

# 添加 ppo_simple.py 所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))
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
        observation_cat_prev_action = False
        entropy_loss_coefficient = 1e-3
    algo = Algo()
    class Env:
        enable_lidar = False
    env = Env()

def load_srlc_model(checkpoint_path, device):
    """
    Load the SRLC model (SimplePPO) from the given checkpoint path.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str or torch.device): Device to load the model on.

    Returns:
        policy (SimplePPO): The loaded policy model, or None if loading failed.
    """
    try:
        from ppo_simple import SimplePPO
    except ImportError:
        raise ImportError("Could not import SimplePPO. Make sure the path to ppo_simple.py is in sys.path.")

    cfg_model = Config()
    
    # Define observation space (State: 10, Human Action: 4)
    observation_spec = CompositeSpec({
        "agents": CompositeSpec({
            "observation": CompositeSpec({
                "state": Unbounded((1, 10), device=device),
                "human_action": Unbounded((1, 4), device=device),
            }, shape=(1,))
        }, shape=(1,))
    }, shape=(1,), device=device)
    
    # Define action space (Action: 4)
    action_spec = CompositeSpec({
        "agents": CompositeSpec({
            "action": Unbounded((1, 4), device=device)
        }, shape=(1,))
    }, shape=(1,), device=device)

    print(f"[INFO] Loading SRLC model from {checkpoint_path}")
    policy = SimplePPO(cfg_model.algo, observation_spec, action_spec, device)
    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    policy.eval()
    return policy
