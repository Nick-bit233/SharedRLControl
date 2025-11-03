from omni_drones.utils import torch

# User Model to simulate human actions
# TODO: finish it by sampling from different style params and distributions
class UserModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def predict(self, observation):
        # Dummy user model: return zero action
        batch_size = observation.shape[0]
        human_action_dim = 4  # (lin_vel x, y, z + ang_vel_yaw)
        return torch.zeros((batch_size, human_action_dim), device=observation.device)