import hydra
from omegaconf import DictConfig
import torch
from tensordict import TensorDict
from omni.isaac.orbit.app import AppLauncher

# Initialize AppLauncher
# We need to do this before importing anything that uses omni.isaac.core
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from env import NavigationEnv

@hydra.main(config_path="../cfg", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    # Set device
    device = cfg.device
    print(f"Device: {device}")

    # Create environment
    # We need to make sure the config structure matches what NavigationEnv expects
    # Usually cfg passed to main is the full config. NavigationEnv might expect cfg or cfg.task
    # Looking at env.py: __init__(self, cfg) -> super().__init__(cfg, cfg.headless)
    # It seems it expects the full config or a task config. 
    # In train.py, it usually does: env = NavigationEnv(cfg)
    
    env = NavigationEnv(cfg)

    # --- Test 1: Step Test ---
    print("\n--- Starting Step Test ---")
    
    # Reset environment
    print("Resetting environment...")
    tensordict = env.reset()
    
    print(f"Action Spec Type: {type(env.action_spec)}")
    
    num_steps = 50
    human_actions_history = []
    
    print(f"Running for {num_steps} steps...")
    for i in range(num_steps):
        # Sample random actions
        # Using .rand() instead of .sample() for TorchRL specs
        action_sample = env.action_spec.rand()
        
        # Handle case where action_spec is not CompositeSpec (returns Tensor instead of TensorDict)
        if isinstance(action_sample, torch.Tensor):
            # Check shape and unsqueeze if necessary
            # Expected shape might be [num_envs, num_agents, action_dim] -> [256, 1, 4]
            if action_sample.ndim == 2:
                action_sample = action_sample.unsqueeze(1)
                
            # Assuming the tensor corresponds to agents.action
            action_td = TensorDict({
                "agents": TensorDict({
                    "action": action_sample
                }, batch_size=env.num_envs)
            }, batch_size=env.num_envs)
        else:
            action_td = action_sample

        # Update tensordict with actions
        # Note: env.step expects the tensordict to contain the action
        tensordict.update(action_td)
        
        # Step environment
        tensordict = env.step(tensordict)
        
        # Check human action
        # Structure: tensordict["agents"]["observation"]["human_action"]
        human_action = tensordict["agents"]["observation"]["human_action"]
        human_actions_history.append(human_action.clone())
        
        if i % 10 == 0:
            print(f"Step {i}: Human Action Mean: {human_action.float().mean().item()}")

    # Check if human actions are changing/non-zero
    human_actions_stack = torch.stack(human_actions_history)
    print(f"Human Actions Mean: {human_actions_stack.float().mean()}")
    print(f"Human Actions Std: {human_actions_stack.float().std()}")
    
    # Save results
    with open("test_user_model_results.txt", "w") as f:
        f.write(f"Human Actions Mean: {human_actions_stack.float().mean()}\n")
        f.write(f"Human Actions Std: {human_actions_stack.float().std()}\n")

    # --- Test 2: Reset Test ---
    print("\n--- Starting Reset Test ---")
    print("Resetting environment again...")
    tensordict = env.reset()
    
    # Check if intent changed (indirectly via human action or internal state if accessible)
    # For now just run a few steps
    for i in range(10):
        action_sample = env.action_spec.rand()
        if isinstance(action_sample, torch.Tensor):
            if action_sample.ndim == 2:
                action_sample = action_sample.unsqueeze(1)
                
            action_td = TensorDict({
                "agents": TensorDict({
                    "action": action_sample
                }, batch_size=env.num_envs)
            }, batch_size=env.num_envs)
        else:
            action_td = action_sample
            
        tensordict.update(action_td)
        tensordict = env.step(tensordict)

    print("Test Complete.")
    simulation_app.close()

if __name__ == "__main__":
    main()
