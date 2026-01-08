import os
import sys
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, Unbounded

# 添加 ppo_simple.py 所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))


class MockConfig:
    class Sim:
        dt = 1.0 / 60.0
        z_spawn = 4.0
    class Env:
        map_range = [20.0, 20.0, 10.0] # Half extents [x, y, z]
        max_episode_length = 500
        enable_lidar = False
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
        training_frame_num = 128
        feature_extractor = FeatureExtractor()
        observation_cat_prev_action = False
        entropy_loss_coefficient = 1e-3

    class UserModel:
        class style:
            frequency_base = 0.1  # base frequency for user command changes
            frequency_scale = 0.1  # scale for randomizing frequency
            smoothness_base = 0.5
            smoothness_scale = 0.45
            laziness = 0.2
        simple_mode = False
        enable_yaw_rate = True

    def __init__(self, device):
        self.device = device
        self.sim = self.Sim()
        self.env = self.Env()
        self.algo = self.Algo()
        self.user_model = self.UserModel()

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

    cfg_model = MockConfig(device=device)
    
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
