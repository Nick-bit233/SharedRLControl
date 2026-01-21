# --- 轨迹漂移统计分析工具 ---
# 用于诊断在模拟器回放中观察到的Z轴向下漂移问题
# 
# 可能的漂移原因分析:
# 1. 控制器跟踪误差: LeePositionController可能在Z轴方向有系统性偏差
# 2. 速度命令解释: VelController中target_vel的解释方式
# 3. 重力补偿: 控制器的重力补偿可能不完美
# 4. body-to-world旋转: 当无人机倾斜时，body frame速度转换到world frame可能有偏差
# 5. 推力限制: 当需要较大向上推力时可能受到饱和限制
#
# 本脚本设计统计方法来量化这些潜在问题

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trajectory Drift Analysis Tool")
parser.add_argument("--num-trials", type=int, default=10, help="Number of independent trials")
parser.add_argument("--frames-per-trial", type=int, default=3000, help="Frames per trial (50 seconds at 60Hz)")
parser.add_argument("--output-dir", type=str, default="drift_analysis_results", help="Output directory for results")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()

app_launcher = AppLauncher(launcher_args={"headless": True})
simulation_app = app_launcher.app

import torch
import numpy as np
import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac-training/training/scripts")))
from user_model import UserModel
from srlc_model import MockConfig

from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse


@dataclass
class TrialStatistics:
    """Statistics collected from a single simulation trial."""
    trial_id: int
    duration_seconds: float
    num_frames: int
    
    # Position statistics
    start_z: float
    end_z: float
    z_drift_total: float  # end_z - start_z
    z_drift_rate: float   # drift per second
    
    # Velocity tracking statistics
    mean_cmd_vel_z: float      # Mean commanded Z velocity (world frame)
    mean_actual_vel_z: float   # Mean actual Z velocity
    mean_tracking_error_z: float  # cmd - actual
    std_tracking_error_z: float
    
    # Velocity command distribution (in world frame after rotation)
    mean_cmd_vel_x: float
    mean_cmd_vel_y: float
    std_cmd_vel_z: float
    
    # Body frame commands (before rotation)
    mean_cmd_vel_z_body: float  # Z command in body frame
    
    # Rotation-induced Z component: difference between world Z and body Z
    mean_rotation_z_contribution: float  # world_z - body_z (effect of tilting)
    
    # Thrust saturation analysis
    mean_thrust_magnitude: float
    max_thrust_used: float
    thrust_saturation_ratio: float  # Fraction of time thrust is near limits
    
    # Attitude effects
    mean_pitch: float  # Mean pitch angle
    mean_roll: float   # Mean roll angle
    mean_abs_pitch: float  # Mean absolute pitch
    mean_abs_roll: float   # Mean absolute roll
    
    # Position bounds
    min_z: float
    max_z: float
    z_range: float
    
    # Boundary violation detection
    hit_floor: bool  # Did the drone hit the floor?
    hit_ceiling: bool  # Did the drone hit the ceiling?
    time_at_floor: float  # Fraction of time at floor
    time_at_ceiling: float  # Fraction of time at ceiling


@dataclass
class AggregateStatistics:
    """Aggregate statistics across all trials."""
    num_trials: int
    total_frames: int
    
    # Drift analysis
    mean_z_drift_rate: float
    std_z_drift_rate: float
    min_z_drift_rate: float
    max_z_drift_rate: float
    
    # Is drift statistically significant?
    drift_significant: bool
    drift_t_statistic: float
    drift_p_value: float
    
    # Tracking error analysis
    mean_tracking_error_z: float
    std_tracking_error_z: float
    
    # Command bias analysis
    mean_cmd_vel_z_world: float  # Should be ~0 if unbiased
    mean_cmd_vel_z_body: float   # Body frame Z command
    mean_rotation_z_contribution: float  # Contribution from body rotation
    
    # Attitude analysis
    mean_abs_pitch: float
    mean_abs_roll: float
    
    # Thrust analysis
    mean_saturation_ratio: float


class DriftAnalyzer:
    """Analyzes trajectory drift in drone simulation."""
    
    def __init__(self, device: str, dt: float, output_dir: str):
        self.device = device
        self.dt = dt
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage for current trial
        self.reset_trial_data()
        
    def reset_trial_data(self):
        """Reset data storage for a new trial."""
        self.positions_z = []
        self.actual_vel_z = []
        self.cmd_vel_world = []  # (N, 3) commanded velocities in world frame
        self.cmd_vel_body = []   # (N, 3) commanded velocities in body frame
        self.thrust_cmds = []
        self.pitch_angles = []
        self.roll_angles = []
        self.tracking_errors_z = []
        
    def record_step(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat: torch.Tensor,
        cmd_vel_body: torch.Tensor,
        cmd_vel_world: torch.Tensor,
        control_cmds: torch.Tensor,
    ):
        """Record data from a single simulation step."""
        # Extract Z position and velocity
        self.positions_z.append(pos_w[0, 2].item())
        self.actual_vel_z.append(vel_w[0, 2].item())
        
        # Store commanded velocities
        self.cmd_vel_world.append(cmd_vel_world[0].detach().cpu().numpy())
        self.cmd_vel_body.append(cmd_vel_body[0].detach().cpu().numpy())
        
        # Calculate tracking error in Z
        cmd_z = cmd_vel_world[0, 2].item()
        actual_z = vel_w[0, 2].item()
        self.tracking_errors_z.append(cmd_z - actual_z)
        
        # Store control commands (thrust analysis)
        self.thrust_cmds.append(control_cmds[0, 0].detach().cpu().numpy())
        
        # Calculate pitch and roll from quaternion
        # Squeeze quat to (4,) if needed
        quat_flat = quat.squeeze()
        # quat is (w, x, y, z)
        w, x, y, z = quat_flat[0], quat_flat[1], quat_flat[2], quat_flat[3]
        
        # Roll (rotation around X)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp).item()
        
        # Pitch (rotation around Y)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0)).item()
        
        self.pitch_angles.append(pitch)
        self.roll_angles.append(roll)
        
    def compute_trial_statistics(self, trial_id: int, floor_z: float = 1.0, ceiling_z: float = 20.0) -> TrialStatistics:
        """Compute statistics for the current trial."""
        n = len(self.positions_z)
        duration = n * self.dt
        
        # Z drift
        z_drift_total = self.positions_z[-1] - self.positions_z[0]
        z_drift_rate = z_drift_total / duration if duration > 0 else 0
        
        # Velocity arrays
        cmd_vel_world = np.array(self.cmd_vel_world)
        cmd_vel_body = np.array(self.cmd_vel_body)
        positions_z = np.array(self.positions_z)
        
        # Rotation-induced Z contribution: world_z - body_z
        # This shows how much Z velocity is added/removed due to body rotation
        rotation_z_contribution = cmd_vel_world[:, 2] - cmd_vel_body[:, 2]
        
        # Thrust analysis (assuming control_cmds is normalized [-1, 1])
        thrust_cmds = np.array(self.thrust_cmds)
        thrust_magnitude = np.mean(np.abs(thrust_cmds))
        max_thrust = np.max(np.abs(thrust_cmds))
        saturation_ratio = np.mean(np.abs(thrust_cmds) > 0.9)
        
        # Boundary violation detection
        hit_floor = np.min(positions_z) < floor_z + 0.5
        hit_ceiling = np.max(positions_z) > ceiling_z - 0.5
        time_at_floor = np.mean(positions_z < floor_z + 0.5)
        time_at_ceiling = np.mean(positions_z > ceiling_z - 0.5)
        
        return TrialStatistics(
            trial_id=trial_id,
            duration_seconds=duration,
            num_frames=n,
            start_z=self.positions_z[0],
            end_z=self.positions_z[-1],
            z_drift_total=z_drift_total,
            z_drift_rate=z_drift_rate,
            mean_cmd_vel_z=np.mean(cmd_vel_world[:, 2]),
            mean_actual_vel_z=np.mean(self.actual_vel_z),
            mean_tracking_error_z=np.mean(self.tracking_errors_z),
            std_tracking_error_z=np.std(self.tracking_errors_z),
            mean_cmd_vel_x=np.mean(cmd_vel_world[:, 0]),
            mean_cmd_vel_y=np.mean(cmd_vel_world[:, 1]),
            std_cmd_vel_z=np.std(cmd_vel_world[:, 2]),
            mean_cmd_vel_z_body=np.mean(cmd_vel_body[:, 2]),
            mean_rotation_z_contribution=np.mean(rotation_z_contribution),
            mean_thrust_magnitude=thrust_magnitude,
            max_thrust_used=max_thrust,
            thrust_saturation_ratio=saturation_ratio,
            mean_pitch=np.mean(self.pitch_angles),
            mean_roll=np.mean(self.roll_angles),
            mean_abs_pitch=np.mean(np.abs(self.pitch_angles)),
            mean_abs_roll=np.mean(np.abs(self.roll_angles)),
            min_z=np.min(self.positions_z),
            max_z=np.max(self.positions_z),
            z_range=np.max(self.positions_z) - np.min(self.positions_z),
            hit_floor=hit_floor,
            hit_ceiling=hit_ceiling,
            time_at_floor=time_at_floor,
            time_at_ceiling=time_at_ceiling,
        )
    
    def compute_aggregate_statistics(self, trial_stats: List[TrialStatistics]) -> AggregateStatistics:
        """Compute aggregate statistics across all trials."""
        # Manual t-test implementation to avoid scipy dependency
        def ttest_1samp(data, expected_mean=0.0):
            """Manual implementation of one-sample t-test."""
            n = len(data)
            if n < 2:
                return 0.0, 1.0
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            if sample_std == 0:
                return float('inf') if sample_mean != expected_mean else 0.0, 0.0
            t_stat = (sample_mean - expected_mean) / (sample_std / np.sqrt(n))
            # Approximate p-value using normal distribution for large n
            # For small n, this is less accurate but sufficient for our purposes
            from math import erfc, sqrt
            p_value = erfc(abs(t_stat) / sqrt(2))
            return t_stat, p_value
        
        drift_rates = [t.z_drift_rate for t in trial_stats]
        tracking_errors = [t.mean_tracking_error_z for t in trial_stats]
        cmd_vel_z = [t.mean_cmd_vel_z for t in trial_stats]
        cmd_vel_z_body = [t.mean_cmd_vel_z_body for t in trial_stats]
        rotation_z_contrib = [t.mean_rotation_z_contribution for t in trial_stats]
        saturation_ratios = [t.thrust_saturation_ratio for t in trial_stats]
        abs_pitch = [t.mean_abs_pitch for t in trial_stats]
        abs_roll = [t.mean_abs_roll for t in trial_stats]
        
        # T-test: Is drift rate significantly different from 0?
        t_stat, p_value = ttest_1samp(drift_rates, 0.0)
        
        return AggregateStatistics(
            num_trials=len(trial_stats),
            total_frames=sum(t.num_frames for t in trial_stats),
            mean_z_drift_rate=np.mean(drift_rates),
            std_z_drift_rate=np.std(drift_rates),
            min_z_drift_rate=np.min(drift_rates),
            max_z_drift_rate=np.max(drift_rates),
            drift_significant=p_value < 0.05,
            drift_t_statistic=t_stat,
            drift_p_value=p_value,
            mean_tracking_error_z=np.mean(tracking_errors),
            std_tracking_error_z=np.std(tracking_errors),
            mean_cmd_vel_z_world=np.mean(cmd_vel_z),
            mean_cmd_vel_z_body=np.mean(cmd_vel_z_body),
            mean_rotation_z_contribution=np.mean(rotation_z_contrib),
            mean_abs_pitch=np.mean(abs_pitch),
            mean_abs_roll=np.mean(abs_roll),
            mean_saturation_ratio=np.mean(saturation_ratios),
        )
    
    def generate_report(
        self, 
        trial_stats: List[TrialStatistics], 
        agg_stats: AggregateStatistics
    ):
        """Generate analysis report with plots and summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Helper to convert numpy types to Python native types for JSON
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 1. Save raw statistics as JSON
        stats_file = os.path.join(self.output_dir, f"statistics_{timestamp}.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(to_serializable({
                    'trials': [asdict(t) for t in trial_stats],
                    'aggregate': asdict(agg_stats)
                }), f, indent=2)
            print(f"Statistics saved to: {stats_file}")
        except TypeError as e:
            print(f"Error saving statistics to JSON: {e}")
            
        # 2. Generate plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Z drift rate distribution
        ax = axes[0, 0]
        drift_rates = [t.z_drift_rate for t in trial_stats]
        ax.hist(drift_rates, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero drift')
        ax.axvline(x=agg_stats.mean_z_drift_rate, color='g', linestyle='-', label=f'Mean: {agg_stats.mean_z_drift_rate:.4f}')
        ax.set_xlabel('Z Drift Rate (m/s)')
        ax.set_ylabel('Count')
        ax.set_title('Z Drift Rate Distribution')
        ax.legend()
        
        # Plot 2: Tracking error distribution
        ax = axes[0, 1]
        tracking_errors = [t.mean_tracking_error_z for t in trial_stats]
        ax.hist(tracking_errors, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax.axvline(x=agg_stats.mean_tracking_error_z, color='g', linestyle='-', 
                   label=f'Mean: {agg_stats.mean_tracking_error_z:.4f}')
        ax.set_xlabel('Mean Z Tracking Error (cmd - actual) (m/s)')
        ax.set_ylabel('Count')
        ax.set_title('Z Velocity Tracking Error')
        ax.legend()
        
        # Plot 3: Commanded Z velocity (world frame)
        ax = axes[0, 2]
        cmd_vel_z = [t.mean_cmd_vel_z for t in trial_stats]
        ax.hist(cmd_vel_z, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero')
        ax.axvline(x=agg_stats.mean_cmd_vel_z_world, color='g', linestyle='-',
                   label=f'Mean: {agg_stats.mean_cmd_vel_z_world:.4f}')
        ax.set_xlabel('Mean Cmd Vel Z (world frame) (m/s)')
        ax.set_ylabel('Count')
        ax.set_title('Commanded Z Velocity (World Frame)')
        ax.legend()
        
        # Plot 4: Drift rate vs tracking error scatter
        ax = axes[1, 0]
        ax.scatter(tracking_errors, drift_rates, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Mean Tracking Error (m/s)')
        ax.set_ylabel('Drift Rate (m/s)')
        ax.set_title('Drift Rate vs Tracking Error')
        
        # Plot 5: Thrust saturation ratio
        ax = axes[1, 1]
        sat_ratios = [t.thrust_saturation_ratio for t in trial_stats]
        ax.hist(sat_ratios, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Thrust Saturation Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Thrust Saturation Analysis')
        
        # Plot 6: Drift vs mean pitch/roll
        ax = axes[1, 2]
        mean_pitch = [np.abs(t.mean_pitch) for t in trial_stats]
        mean_roll = [np.abs(t.mean_roll) for t in trial_stats]
        mean_tilt = [(p + r) / 2 for p, r in zip(mean_pitch, mean_roll)]
        ax.scatter(mean_tilt, drift_rates, alpha=0.7)
        ax.set_xlabel('Mean |Pitch| + |Roll| / 2 (rad)')
        ax.set_ylabel('Drift Rate (m/s)')
        ax.set_title('Drift Rate vs Mean Tilt Angle')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"drift_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_file}")
        
        # 3. Print summary
        print("\n" + "="*60)
        print("TRAJECTORY DRIFT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Number of trials: {agg_stats.num_trials}")
        print(f"Total frames analyzed: {agg_stats.total_frames}")
        print()
        print("Z-AXIS DRIFT ANALYSIS:")
        print(f"  Mean drift rate: {agg_stats.mean_z_drift_rate:.6f} m/s")
        print(f"  Std drift rate:  {agg_stats.std_z_drift_rate:.6f} m/s")
        print(f"  Min drift rate:  {agg_stats.min_z_drift_rate:.6f} m/s")
        print(f"  Max drift rate:  {agg_stats.max_z_drift_rate:.6f} m/s")
        print()
        print("STATISTICAL SIGNIFICANCE (H0: drift rate = 0):")
        print(f"  t-statistic: {agg_stats.drift_t_statistic:.4f}")
        print(f"  p-value:     {agg_stats.drift_p_value:.6f}")
        print(f"  Significant: {'YES' if agg_stats.drift_significant else 'NO'} (alpha=0.05)")
        print()
        
        # Boundary violation analysis
        trials_hitting_floor = sum(1 for t in trial_stats if t.hit_floor)
        trials_hitting_ceiling = sum(1 for t in trial_stats if t.hit_ceiling)
        mean_time_at_floor = np.mean([t.time_at_floor for t in trial_stats])
        mean_time_at_ceiling = np.mean([t.time_at_ceiling for t in trial_stats])
        
        print("BOUNDARY ANALYSIS:")
        print(f"  Trials hitting floor:   {trials_hitting_floor}/{len(trial_stats)}")
        print(f"  Trials hitting ceiling: {trials_hitting_ceiling}/{len(trial_stats)}")
        print(f"  Mean time at floor:     {mean_time_at_floor*100:.2f}%")
        print(f"  Mean time at ceiling:   {mean_time_at_ceiling*100:.2f}%")
        print()
        
        print("TRACKING ERROR ANALYSIS:")
        print(f"  Mean Z tracking error (cmd-actual): {agg_stats.mean_tracking_error_z:.6f} m/s")
        print(f"  Std Z tracking error:               {agg_stats.std_tracking_error_z:.6f} m/s")
        print()
        print("COMMAND ANALYSIS:")
        print(f"  Mean commanded Z vel (body frame):  {agg_stats.mean_cmd_vel_z_body:.6f} m/s")
        print(f"  Mean commanded Z vel (world frame): {agg_stats.mean_cmd_vel_z_world:.6f} m/s")
        print(f"  Mean rotation Z contribution:       {agg_stats.mean_rotation_z_contribution:.6f} m/s")
        print()
        print("ATTITUDE ANALYSIS:")
        print(f"  Mean |pitch|: {agg_stats.mean_abs_pitch:.4f} rad ({np.degrees(agg_stats.mean_abs_pitch):.2f} deg)")
        print(f"  Mean |roll|:  {agg_stats.mean_abs_roll:.4f} rad ({np.degrees(agg_stats.mean_abs_roll):.2f} deg)")
        print()
        print("THRUST ANALYSIS:")
        print(f"  Mean saturation ratio: {agg_stats.mean_saturation_ratio:.4f}")
        print()
        
        # Diagnose likely cause
        print("DIAGNOSIS:")
        
        # Check for random walk behavior (high variance in drift rates)
        if agg_stats.std_z_drift_rate > abs(agg_stats.mean_z_drift_rate):
            print("  [!] HIGH VARIANCE IN DRIFT RATES")
            print("      -> Drift direction varies significantly between trials")
            print("      -> This is consistent with RANDOM WALK behavior, not systematic bias")
            print("      -> Individual trajectories may drift up or down by chance")
        
        if trials_hitting_floor > 0 or trials_hitting_ceiling > 0:
            print(f"  [!] BOUNDARY VIOLATIONS DETECTED")
            print(f"      -> {trials_hitting_floor} trials hit floor, {trials_hitting_ceiling} hit ceiling")
            print("      -> Long trajectories without reset will exceed map bounds")
            print("      -> This is expected for unbounded random walks")
        
        if abs(agg_stats.mean_cmd_vel_z_body) > 0.01:
            print(f"  [!] Body frame Z command bias: {agg_stats.mean_cmd_vel_z_body:.4f} m/s")
            print("      -> This may be statistical noise from limited samples")
            print("      -> Check if bias is consistent across many more trials")
        
        if abs(agg_stats.mean_rotation_z_contribution) > 0.01:
            print(f"  [!] Rotation-induced Z contribution: {agg_stats.mean_rotation_z_contribution:.4f} m/s")
            print("      -> Body tilting causes Z velocity component when moving horizontally")
            print("      -> This is a PHYSICAL EFFECT, not a bug")
        
        if agg_stats.mean_tracking_error_z > 0.01:
            print("  [!] Positive tracking error: Controller under-tracks upward commands")
        elif agg_stats.mean_tracking_error_z < -0.01:
            print("  [!] Negative tracking error: Controller over-tracks (exceeds commanded velocity)")
        
        if agg_stats.mean_saturation_ratio > 0.1:
            print("  [!] High thrust saturation: Controller often hits thrust limits")
        
        if not agg_stats.drift_significant:
            print("  [OK] No statistically significant systematic drift detected")
        elif agg_stats.std_z_drift_rate > abs(agg_stats.mean_z_drift_rate):
            print("  [NOTE] Drift is statistically significant but highly variable")
            print("         This suggests random walk rather than systematic bias")
        
        print("="*60)
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Drift Analysis Summary\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Mean drift rate: {agg_stats.mean_z_drift_rate:.6f} m/s\n")
            f.write(f"Significant: {agg_stats.drift_significant}\n")
            f.write(f"p-value: {agg_stats.drift_p_value:.6f}\n")
        print(f"Summary saved to: {summary_file}")


def run_single_trial(
    trial_id: int,
    analyzer: DriftAnalyzer,
    sim: SimulationContext,
    drone,
    controller,
    user_model: UserModel,
    num_frames: int,
    device: str,
    dt: float,
    z_spawn: float,
):
    """Run a single simulation trial and collect statistics."""
    print(f"\n--- Starting Trial {trial_id + 1} ---")
    
    # Reset analyzer data
    analyzer.reset_trial_data()
    
    # Reset simulation
    sim.reset()
    drone.initialize()
    
    # Reset user model with new random trajectory
    init_pos = torch.tensor([[0.0, 0.0, z_spawn]], device=device)
    init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    user_model.reset(init_pos, init_quat, torch.tensor([0], device=device))
    
    # Simulation loop
    for frame in range(num_frames):
        if not simulation_app.is_running():
            break
            
        if sim.is_playing():
            # Get drone state
            root_state = drone.get_state()[..., :13]  # (1, 1, 13)
            
            drone_pos_w = root_state[..., :3].squeeze(1)        # (N, 3)
            drone_vel_w = root_state[..., 7:10].squeeze(1)      # (N, 3)
            drone_quat = root_state[..., 3:7].squeeze(1)        # (N, 4)
            drone_ang_vel_w = root_state[..., 10:13].squeeze(1) # (N, 3)
            
            # Calculate body frame velocities for user model
            vel_b = quat_rotate_inverse(drone_quat, drone_vel_w)
            ang_vel_b = quat_rotate_inverse(drone_quat, drone_ang_vel_w)
            drone_state_b = torch.cat([vel_b, ang_vel_b, drone_quat], dim=-1)
            
            # Get user model action
            action, _ = user_model.step(drone_state_b, drone_pos_w)
            action = action.unsqueeze(1)  # (1, 1, 3) or (1, 1, 4)
            
            # Extract velocity command (body frame)
            action_vel_b = action[..., :3]  # (1, 1, 3)
            if action.shape[-1] == 4:
                action_yaw_rate = action[..., 3]
            else:
                action_yaw_rate = torch.zeros((1, 1), device=device)
            
            # Rotate to world frame
            to_rotate_q = drone_quat.unsqueeze(1)
            action_vel_w = quat_rotate(to_rotate_q, action_vel_b)  # (1, 1, 3)
            
            # Apply to controller
            target_vel = action_vel_w
            target_yaw = action_yaw_rate * torch.pi
            
            control_action = controller(
                root_state=root_state,
                target_vel=target_vel,
                target_yaw=None,  # Not controlling yaw
            )
            
            # Record data
            analyzer.record_step(
                pos_w=drone_pos_w,
                vel_w=drone_vel_w,
                quat=drone_quat.unsqueeze(1),
                cmd_vel_body=action_vel_b.squeeze(1),
                cmd_vel_world=action_vel_w.squeeze(1),
                control_cmds=control_action,
            )
            
            # Apply control
            drone.apply_action(control_action)
        
        # Step physics
        sim.step(render=False)
    
    # Compute trial statistics
    stats = analyzer.compute_trial_statistics(trial_id)
    print(f"  Trial {trial_id + 1}: Z drift rate = {stats.z_drift_rate:.4f} m/s, "
          f"Tracking error = {stats.mean_tracking_error_z:.4f} m/s")
    
    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / 60.0
    
    print("="*60)
    print("TRAJECTORY DRIFT ANALYSIS TOOL")
    print("="*60)
    print(f"Device: {device}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Frames per trial: {args.frames_per_trial}")
    print(f"Duration per trial: {args.frames_per_trial * dt:.1f} seconds")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = SimulationContext(sim_cfg)
    
    # Build environment
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Mock config
    mock_cfg = MockConfig(device)
    
    # Build drone
    drone, controller = MultirotorBase.make(
        "Hummingbird", "LeePositionController", device
    )
    z_spawn = mock_cfg.sim.z_spawn
    drone.spawn(translations=torch.tensor([[0.0, 0.0, z_spawn]], device=device))
    
    # Initialize user model
    user_model = UserModel(
        num_envs=1,
        cfg=mock_cfg,
        logger=None,
        offline_mode=False,
        dataset=None,
        sampling_mode="raw",
    )
    
    # Initialize analyzer
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    analyzer = DriftAnalyzer(device=device, dt=dt, output_dir=output_dir)
    
    # Play simulator
    sim.reset()
    drone.initialize()
    
    # Run trials
    trial_stats = []
    for trial_id in range(args.num_trials):
        stats = run_single_trial(
            trial_id=trial_id,
            analyzer=analyzer,
            sim=sim,
            drone=drone,
            controller=controller,
            user_model=user_model,
            num_frames=args.frames_per_trial,
            device=device,
            dt=dt,
            z_spawn=z_spawn,
        )
        trial_stats.append(stats)
    
    # Compute aggregate statistics
    agg_stats = analyzer.compute_aggregate_statistics(trial_stats)
    
    # Generate report
    analyzer.generate_report(trial_stats, agg_stats)
    
    print("\nAnalysis complete!")
    simulation_app.close()


if __name__ == "__main__":
    main()
