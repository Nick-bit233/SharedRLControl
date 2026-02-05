import os
import sys
import torch
from tensordict import TensorDict
from torchrl.data import Unbounded, Composite


class MockConfig:
    class Sim:
        dt = 1.0 / 60.0
        z_spawn = 5.0
    class Env:
        map_range = [20.0, 20.0, 10.0] # Half extents [x, y, z]
        max_episode_length = 500
        enable_lidar = False
    class Algo:
        def get(self, key, default=None):
            """Dict-like get method for compatibility."""
            return getattr(self, key, default)
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
        reward_threshold = 1.0

    class UserModel:
        class style:
            frequency_base = 0.1  # base frequency for user command changes
            frequency_scale = 0.1  # scale for randomizing frequency
            smoothness_base = 0.5
            smoothness_scale = 0.45
            laziness = 0.2
        simple_mode = False
        enable_yaw_rate = True
        offline_mode = False
        dataset_path = None   
        sampling_mode = "scaled" 
        gpu_cache_reserve_gb = 2.0 
        min_scale_factor = 0.5  
        preload_data = True  
        online_sample_filter = False  # If true, filter sampled trajectories based on velocity constraints
        
        # Z-axis tilt compensation (m/s)
        # Compensates for lift loss when drone tilts for horizontal flight
        # Recommended: 0.03-0.05 based on average tilt angle (~1.5 deg)
        z_tilt_compensation = 0.05  # Set to 0.04 to enable compensation
        
        def get(self, key, default=None):
            """Dict-like get method for compatibility."""
            return getattr(self, key, default)

    def __init__(self, device):
        self.device = device
        self.sim = self.Sim()
        self.env = self.Env()
        self.algo = self.Algo()
        self.user_model = self.UserModel()

def load_srlc_model_simple(model_type, checkpoint_path, device, action_dim=4, enable_lidar=True):
    """
    Load the SRLC model from the given checkpoint path. (no lidar feature)
    
    Args:
        model_type(str): Simple/Residual/Full
        checkpoint_path (str): Path to the model checkpoint.
        device (str or torch.device): Device to load the model on.
        action_dim (int): Action dimension, either 3 (no yaw control) or 4 (with yaw control). Default is 4.

    Returns:
        policy (SimplePPO): The loaded policy model, or None if loading failed.
    """
    assert action_dim in [3, 4], f"action_dim must be 3 or 4, got {action_dim}"

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/src/algos")))

    if model_type == "Simple":
        try:
            from ppo_simple import SimplePPO as ppo_model
        except ImportError:
            raise ImportError("Could not import SimplePPO. Make sure the path to ppo_simple.py is in sys.path.")
    elif model_type == "Residual":
        try:
            from ppo_residual import SimpleResidualPPO as ppo_model
        except ImportError:
            raise ImportError("Could not import SimpleResidualPPO. Make sure the path to ppo_simple.py is in sys.path.")
    elif model_type == "Constrained":
        try:
            from ppo_constrained import ConstrainedResidualPPO as ppo_model
        except ImportError:
            raise ImportError("Could not import ConstrainedResidualPPO. Make sure the path to ppo_constrained.py is in sys.path.")

    cfg_model = MockConfig(device=device)
    
    # Define observation space (State: 10, Human Action: 4)
    if enable_lidar:
        observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((10, ), device=device),
                    "human_action": Unbounded((action_dim, ), device=device),
                    "lidar": Unbounded((1, 36, 4), device=device),  # default lidar spec: vbeams=4, hbeams=36
                })
            }).expand(1)
        }, shape=[1], device=device)
    else:
        observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((10, ), device=device),
                    "human_action": Unbounded((action_dim, ), device=device),
                })
            }).expand(1)
        }, shape=[1], device=device)
    
    # Define action space (Action: 3 or 4 based on action_dim)
    action_spec = Composite({
        "agents": Composite({
            "action": Unbounded((action_dim, ), device=device)
        })
    }).expand(1).to(device)

    # print("--- SRLC Verify Model Space Shape ---")
    # print(f"Observation Spec: {observation_spec}")
    # print(f"Action Spec: {action_spec}")

    print(f"[INFO] Loading SRLC model from {checkpoint_path} (action_dim={action_dim})")
    policy = ppo_model(cfg_model.algo, observation_spec, action_spec, device)
    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] Model loaded successfully with action_dim={action_dim}.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    policy.eval()
    return policy
