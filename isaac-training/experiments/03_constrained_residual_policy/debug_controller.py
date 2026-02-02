import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from tensordict import TensorDict

# Ensure the root directory is in sys.path to import src
# script is in experiments/03_constrained_residual_policy/
# root is ../../
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Initialize SimulationApp
# We do this BEFORE importing any omni.* modules to avoid warnings/errors
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# Now import environment resources
from src.envs.env_residual import FollowingEnvResidual

@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("\n[Debug] Initializing Environment for Dynamics Diagnostics...")
    
    # 1. Force minimal debugging settings
    # Override number of environments to 1 for clear log
    cfg.env.num_envs = 1
    # Disable video recording if enabled
    if "record_video" in cfg:
        cfg.record_video = False
    
    # 2. Create Environment
    try:
        env = FollowingEnvResidual(cfg)
    except Exception as e:
        print(f"Error creating environment: {e}")
        exit(1)

    env.reset()
    
    # 3. Parameter Consistency Check
    print("\n" + "="*80)
    print(" DYNAMICS PARAMETER CHECK")
    print("="*80)
    
    drone = env.drone
    controller = env.controller
    
    # A. Mass Check
    drone_masses = drone.base_link.get_masses()
    phys_mass = drone_masses[0].item()
    print(f"1. Mass Consistency:")
    print(f"   - Drone Physical Mass (Sim):   {phys_mass:.6f} kg")  # 0.68
    
    if hasattr(controller, 'mass'):
        ctrl_mass = controller.mass.item()
        print(f"   - Controller Model Mass:       {ctrl_mass:.6f} kg")  # 0.716
        if abs(phys_mass - ctrl_mass) > 1e-4:
            print(f"   [WARNING] Mass mismatch! Controller thinks drone is {ctrl_mass} vs Real {phys_mass}")
    else:
        print("   - Controller has no 'mass' attribute.")

    # B. Gravity Check
    print(f"2. Gravity Consistency:")
    if hasattr(controller, 'g'):
        g_val = controller.g
        if g_val.numel() > 1:
            g_val = g_val[2] # Assume [0,0,g]
        print(f"   - Controller Gravity:          {g_val.item():.4f} m/s^2")
    
    # C. Thrust-to-Weight
    print(f"3. Thrust-to-Weight Ratio:")
    if hasattr(drone, 'THRUST2WEIGHT_0'):
        # THRUST2WEIGHT_0 is typically per-rotor [T2W_0, T2W_1, T2W_2, T2W_3]
        # We take the mean to get the average T2W per rotor
        t2w_per_rotor = drone.THRUST2WEIGHT_0.float().mean().item()
        num_rotors = drone.num_rotors
        total_t2w = t2w_per_rotor * num_rotors
        
        print(f"   - T2W Ratio (Per Rotor):       {t2w_per_rotor:.4f}")
        print(f"   - T2W Ratio (Total):           {total_t2w:.4f}")
        
        # Theoretical hover throttle (0-1)
        # Weight = Throttle * Total_Max_Thrust
        # Throttle = Weight / Total_Max_Thrust = 1 / Total_T2W
        hover_throttle = 1.0 / total_t2w
        
        # Hover command (-1 to 1) = (Throttle * 2) - 1
        hover_cmd = (hover_throttle * 2) - 1
        print(f"   - Theoretical Hover Throttle:  {hover_throttle:.4f} (range [0, 1])")
        print(f"   - Theoretical Hover Cmd:       {hover_cmd:.4f} (range [-1, 1])")
    
    # 4. Simulation Loop (Hover Test)
    print("\n" + "="*80)
    print(" SIMULATION LOOP: HOVER TEST (Target Vz = 0)")
    print("="*80)
    
    # Reset env first
    td = env.reset()
    
    # Warmup
    print("Warmup...")
    zeros_action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(10):  
        if "agents" not in td.keys():
             td["agents"] = TensorDict({}, batch_size=td.batch_size, device=td.device)
        td["agents", "action"] = zeros_action.clone()
        td = env.step(td)

    # Log Header
    print(f"{'Step':<6} | {'Z (m)':<10} | {'Vz Real':<10} | {'Vz Cmd':<10} | {'Thrust Cmd':<15} | {'Throttle':<10}")
    print("-" * 75)
    
    for i in range(50):
        # Update action in the EXISTING tensordict to preserve stats/info
        if "agents" not in td.keys():
             td["agents"] = TensorDict({}, batch_size=td.batch_size, device=td.device)
        td["agents", "action"] = zeros_action.clone()
        
        td = env.step(td)
        
        state = drone.get_state(env_frame=False)
        # Handle (num_envs, num_agents, state_dim) or (num_envs, state_dim)
        if state.ndim == 3:
            pos = state[0, 0, :3]
            vel = state[0, 0, 7:10]
        else:
            pos = state[0, :3]
            vel = state[0, 7:10]
        
        # Calculate what controller IS outputting given this state
        root_state = state[..., :13]
        target_vel = zeros_action.unsqueeze(1) # As per env logic
        computed_actions = controller(root_state=root_state, target_vel=target_vel, target_yaw=None)
        
        # Convert [-1, 1] cmd back to [0, 1] throttle for readability
        throttle = (computed_actions.mean().item() + 1) / 2
        
        print(f"{i:<6} | {pos[2].item():<10.4f} | {vel[2].item():<10.4f} | {0.0:<10.0f} | {computed_actions.mean().item():<15.4f} | {throttle:<10.4f}")

    print("="*80)
    simulation_app.close()

if __name__ == "__main__":
    main()
