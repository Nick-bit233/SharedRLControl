import os
import sys
import torch
from torchrl.data import Unbounded, Composite


SUPPORTED_MODEL_TYPES = [
    "Auto",
    "Simple",
    "Residual",
    "Constrained",
    "ConstrainedBeta",
    "ConstrainedBetaLagrangian",
]


class MockConfig:
    class Sim:
        dt = 1.0 / 60.0
        z_spawn = 5.0
    class Env:
        map_range = [6.0, 12.0, 5.0] # Half extents [x, y, z] (tunnel env default)
        max_episode_length = 1000
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
        min_concentration = 2.0  # Beta distribution parameter

    class UserModel:
        class style:
            frequency_base = 0.1  # base frequency for user command changes
            frequency_scale = 0.1  # scale for randomizing frequency
            smoothness_base = 0.5
            smoothness_scale = 0.45
            laziness = 0.2
        simple_mode = False
        enable_yaw_rate = False
        offline_mode = False
        dataset_path = None   
        sampling_mode = "scaled" 
        gpu_cache_reserve_gb = 2.0 
        min_scale_factor = 0.5  
        preload_data = True  
        online_sample_filter = False  # If true, filter sampled trajectories based on velocity constraints
        
        # Z-axis tilt compensation (m/s)
        z_tilt_compensation = 0.0
        
        def get(self, key, default=None):
            """Dict-like get method for compatibility."""
            return getattr(self, key, default)

    def __init__(self, device):
        self.device = device
        self.sim = self.Sim()
        self.env = self.Env()
        self.algo = self.Algo()
        self.user_model = self.UserModel()


def _ensure_training_import_paths():
    training_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training"))
    algos_root = os.path.join(training_root, "src", "algos")
    for path in (training_root, algos_root):
        if path not in sys.path:
            sys.path.append(path)


def _load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_policy_state(checkpoint):
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        metadata_keys = [key for key in checkpoint.keys() if key != "policy"]
        print(
            "[INFO] Detected rich checkpoint; loading checkpoint['policy']"
            + (f" and ignoring runtime metadata keys: {metadata_keys}" if metadata_keys else ".")
        )
        return checkpoint["policy"], True
    return checkpoint, False


def _state_dict_has_lagrangian_weights(state_dict):
    return isinstance(state_dict, dict) and "lambda_lag" in state_dict


def _select_model_class(model_type, state_dict):
    if model_type == "Auto":
        if _state_dict_has_lagrangian_weights(state_dict):
            model_type = "ConstrainedBetaLagrangian"
        else:
            model_type = "ConstrainedBeta"
        print(f"[INFO] Auto-selected SRLC model type: {model_type}")
    elif model_type == "ConstrainedBeta" and _state_dict_has_lagrangian_weights(state_dict):
        model_type = "ConstrainedBetaLagrangian"
        print("[INFO] Detected Lagrangian checkpoint weights; using ConstrainedBetaLagrangian loader.")

    if model_type == "Simple":
        from ppo_simple import SimplePPO as ppo_model
    elif model_type == "Residual":
        from ppo_residual import SimpleResidualPPO as ppo_model
    elif model_type == "Constrained":
        from ppo_constrained import ConstrainedResidualPPO as ppo_model
    elif model_type == "ConstrainedBeta":
        from ppo_constrained_beta import ConstrainedResidualPPO_Beta as ppo_model
    elif model_type == "ConstrainedBetaLagrangian":
        from ppo_constrained_beta_lagrangian import ConstrainedResidualPPO_BetaLagrangian as ppo_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: {', '.join(SUPPORTED_MODEL_TYPES)}")
    return model_type, ppo_model


def _load_policy_state(policy, state_dict, model_type):
    try:
        policy.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        if not isinstance(state_dict, dict):
            raise exc
        policy_keys = set(policy.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = policy_keys - loaded_keys
        unexpected_keys = loaded_keys - policy_keys

        # 04s Lagrangian checkpoints add lambda_lag. It is useful for resumed
        # training but does not affect forward inference, so allow old/new Beta
        # checkpoints to interoperate when this is the only state mismatch.
        if model_type == "ConstrainedBetaLagrangian" and missing_keys <= {"lambda_lag"} and not unexpected_keys:
            policy.load_state_dict(state_dict, strict=False)
            print("[INFO] Loaded non-Lagrangian Beta checkpoint into Lagrangian policy without lambda_lag.")
            return
        if model_type == "ConstrainedBeta" and unexpected_keys <= {"lambda_lag"} and not missing_keys:
            policy.load_state_dict(state_dict, strict=False)
            print("[INFO] Loaded Lagrangian Beta checkpoint into Beta policy; ignored inference-only lambda_lag.")
            return

        print(f"[ERROR] Missing checkpoint keys: {sorted(missing_keys)}")
        print(f"[ERROR] Unexpected checkpoint keys: {sorted(unexpected_keys)}")
        raise exc


def load_srlc_model(model_type, checkpoint_path, device, action_dim=3, enable_lidar=True, state_dim=10):
    """
    Load the SRLC model from the given checkpoint path.
    
    Args:
        model_type(str): Auto/Simple/Residual/Constrained/ConstrainedBeta/ConstrainedBetaLagrangian
        checkpoint_path (str): Path to the model checkpoint.
        device (str or torch.device): Device to load the model on.
        action_dim (int): Action dimension, either 3 (no yaw control) or 4 (with yaw control). Default is 3.
        enable_lidar (bool): Whether the model uses lidar input.
        state_dim (int): State observation dimension. 10 for standard envs, 11 for safety_shield (adds z_normalized).

    Returns:
        policy: The loaded policy model, or None if loading failed.
    """
    assert action_dim in [3, 4], f"action_dim must be 3 or 4, got {action_dim}"

    _ensure_training_import_paths()
    try:
        checkpoint = _load_checkpoint(checkpoint_path, device)
        state_dict, _ = _extract_policy_state(checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint policy payload must be a state_dict, got {type(state_dict).__name__}")
        model_type, ppo_model = _select_model_class(model_type, state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to read checkpoint: {e}")
        return None

    cfg_model = MockConfig(device=device)
    
    # Define observation space
    if enable_lidar:
        observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((state_dim, ), device=device),
                    "human_action": Unbounded((action_dim, ), device=device),
                    "lidar": Unbounded((1, 36, 4), device=device),  # default lidar spec: vbeams=4, hbeams=36
                })
            }).expand(1)
        }, shape=[1], device=device)
    else:
        observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((state_dim, ), device=device),
                    "human_action": Unbounded((action_dim, ), device=device),
                })
            }).expand(1)
        }, shape=[1], device=device)
    
    # Define action space
    action_spec = Composite({
        "agents": Composite({
            "action": Unbounded((action_dim, ), device=device)
        })
    }).expand(1).to(device)

    print(f"[INFO] Loading SRLC model: type={model_type}, checkpoint={checkpoint_path}, action_dim={action_dim}, state_dim={state_dim}, lidar={enable_lidar}")
    policy = ppo_model(cfg_model.algo, observation_spec, action_spec, device)
    try:
        _load_policy_state(policy, state_dict, model_type)
        print(f"[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    policy.eval()
    return policy


# Keep backward compatibility
load_srlc_model_simple = load_srlc_model
