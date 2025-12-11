import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from omni_drones import init_simulation_app
import sys
import os

# Ensure we can import user_model from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import user_model

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(version_base=None, config_path=FILE_PATH, config_name="sim_play")
def main(cfg):
    OmegaConf.resolve(cfg)
    # Force headless to False to ensure rendering if not specified
    # But init_simulation_app usually handles this based on cfg or args.
    # We want to save video, so we need rendering.
    
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    from isaacsim.core.utils.viewports import set_camera_view
    # from omni.isaac.debug_draw import _debug_draw  # @Deprecation - use isaacsim.util.debug_draw
    from isaacsim.util.debug_draw import _debug_draw
    import dataclasses

    hydra_cfg = HydraConfig.get()
    log_output_dir = hydra_cfg.runtime.output_dir  # 使用 Hydra 日志输出目录 

    sim_cfg = sim_utils.SimulationCfg(
        dt=cfg.sim.dt,
        device=cfg.sim.device,
    )
    sim = SimulationContext(sim_cfg)

    n = 1
    drone, controller = MultirotorBase.make(
        cfg.drone_model, "LeePositionController", cfg.sim.device
    )

    # Spawn drone
    translations = torch.zeros(n, 3)
    translations[:, 2] = 1.0
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    # Setup Camera
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb"],
    )
    camera_vis = Camera(camera_cfg)
    # Initialize to main viewport
    camera_vis.initialize("/OmniverseKit_Persp")
    
    sim.reset()
    drone.initialize()
    
    draw = _debug_draw.acquire_debug_draw_interface()

    # Generate Trajectory
    print("Generating trajectory...")
    dt = cfg.sim.dt

    # Mock style
    style = {
        'aggressiveness': torch.tensor([0.5], device=sim.device),
        'dexterity': torch.tensor([0.5], device=sim.device)
    }
    noise_level = torch.tensor([0.0], device=sim.device) # Clean trajectory for visualization

    batched_data = user_model.sample_joystick_profile_batched(
        num_envs=1,
        style=style,
        noise_level=noise_level,
        device=sim.device,
        min_duration=cfg.trajectory.min_duration,
        max_duration=cfg.trajectory.max_duration
    )
    t_total, right_target, left_target, params = batched_data
    print(f"Sampled target: left: {left_target}, right: {right_target}, duration: {t_total.item():.2f}s")
    
    # Generate actions (normalized)
    total_steps = int(t_total.item() / dt)
    action_scale = torch.ones(4, device=sim.device)

    actions = user_model.generate_action_from_stick_profile_batched(
        dt=dt, 
        total_steps=total_steps,
        action_scale=action_scale,
        right_target=right_target, 
        left_target=left_target,
        params=params,
        device=sim.device
    )
    # Squeeze batch dimension (1, T, 4) -> (T, 4)
    actions = actions.squeeze(0)

    print(f"Generated action shape: {actions.shape},  {actions.shape[0]} actions for {t_total.item():.2f} seconds.")
    
    # Scale actions to physical units (max velocities for drones)
    # Assuming:
    # Pitch -> vx (max 2.0)
    # Roll -> vy (max 2.0)
    # Throttle -> vz (max 1.0)
    # Yaw -> yaw_rate (max 1.0)
    scale_vel_xy = cfg.trajectory.scale_vel_xy
    scale_vel_z = cfg.trajectory.scale_vel_z
    scale_yaw = cfg.trajectory.scale_yaw
    
    actions_phys = actions.clone()
    actions_phys[:, 0] *= scale_vel_xy # vx
    actions_phys[:, 1] *= scale_vel_xy # vy
    actions_phys[:, 2] *= scale_vel_z  # vz
    actions_phys[:, 3] *= scale_yaw    # yaw_rate
    
    # Integrate to get reference trajectory
    # Initial state
    curr_pos = translations[0].clone().to(sim.device)
    curr_yaw = torch.tensor(0.0, device=sim.device)
    
    ref_positions = []
    ref_yaws = []
    
    T = actions_phys.shape[0]
    
    # calculate ref positions based on trajectory actions
    for t in range(T):
        act = actions_phys[t]
        # print(f"Step {t}: Action (vx, vy, vz, yaw_rate): {act.cpu().numpy()}")
        vx, vy, vz, yaw_rate = act[0], act[1], act[2], act[3]
        
        # Update yaw
        curr_yaw += yaw_rate * dt
        
        # Rotate velocity to world frame
        cy = torch.cos(curr_yaw)
        sy = torch.sin(curr_yaw)
        
        vel_world_x = vx * cy - vy * sy
        vel_world_y = vx * sy + vy * cy
        vel_world_z = vz
        
        curr_pos[0] += vel_world_x * dt
        curr_pos[1] += vel_world_y * dt
        curr_pos[2] += vel_world_z * dt
        
        ref_positions.append(curr_pos.clone())
        ref_yaws.append(curr_yaw.clone())
        
    ref_positions = torch.stack(ref_positions) # (T, 3)
    ref_yaws = torch.stack(ref_yaws) # (T,)
    
    # Draw trajectory
    points = ref_positions.cpu().tolist()
    if len(points) > 1:
        # Draw lines between consecutive points
        draw.draw_lines(points[:-1], points[1:], [(1, 0, 1, 1)] * (len(points)-1), [2.0] * (len(points)-1))
    
    frames = []
    drone_state = drone.get_state()[..., :13].squeeze(0)
    
    print(f"Simulating {T} steps...")
    from tqdm import tqdm
    for i in tqdm(range(T)):
        if sim.is_stopped():
            break
            
        # Control
        target_pos = ref_positions[i].unsqueeze(0) # (1, 3)
        target_yaw = ref_yaws[i].unsqueeze(0)      # (1,)
        
        action = controller.compute(drone_state, target_pos=target_pos, target_yaw=target_yaw)
        drone.apply_action(action)
        
        sim.step(render=True)
        
        # Update Camera
        # Follow drone 0
        pos = drone_state[0, :3]
        # Camera offset: behind and above
        # Simple offset in world frame
        cam_eye = pos + torch.tensor([-3.0, -2.0, 1.5], device=sim.device)
        cam_target = pos + torch.tensor([0.0, 0.0, 0.5], device=sim.device)
        
        set_camera_view(eye=cam_eye.cpu().numpy(), target=cam_target.cpu().numpy())
        
        if i % 2 == 0:
            # omni_drones camera using old omni.replicator.core interface, empty img tensors may occur in the first frame.
            images = camera_vis.get_images().get("rgb", None)
            if images is not None:
                frames.append(images.cpu())
            
        drone_state = drone.get_state()[..., :13].squeeze(0)
        
    # Save video
    import imageio
    if len(frames) > 0:
        print("Saving video...")
        # frames is a list of (1, C, H, W) tensors
        # cat gives (T, C, H, W)
        video_tensor = torch.cat(frames).permute(0, 2, 3, 1)[..., :3]
        
        # Ensure uint8
        if video_tensor.dtype != torch.uint8:
             if video_tensor.max() <= 1.0:
                 video_tensor = (video_tensor * 255).to(torch.uint8)
             else:
                 video_tensor = video_tensor.to(torch.uint8)
                 
        video_np = video_tensor.cpu().numpy()
        video_output_path = os.path.join(log_output_dir, "sim_play_trajectory.mp4")
        imageio.mimwrite(video_output_path, video_np, fps=30, quality=8)
        print("Video saved to sim_play_trajectory.mp4")
        
    simulation_app.close()

if __name__ == "__main__":
    main()
