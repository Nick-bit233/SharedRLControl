"""
UserModel Diagnostics Module

This module contains verification and diagnostic functions for analyzing
the UserModel's Perlin noise generation and trajectory drift behavior.

Usage:
    from src.simulated_users.user_model_diagnostics import run_full_diagnosis, diagnose_z_drift_issue
    
    # Quick diagnosis
    diagnose_z_drift_issue()
    
    # Complete diagnosis
    run_full_diagnosis()
"""

import torch
import math
from src.simulated_users.user_model import BatchedPerlinNoise, InterpType, batched_perlin_noise


def verify_noise_distribution(
    num_envs: int = 1000,
    buffer_size: int = 128,
    num_trials: int = 10,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify that the Perlin noise generation produces unbiased velocity distributions.
    
    This function tests whether vx, vy, vz have zero mean (unbiased) distribution.
    A systematic bias (especially in vz) would cause drones to drift in one direction.
    
    Args:
        num_envs: Number of simulated environments per trial
        buffer_size: Length of trajectory buffer (timesteps)
        num_trials: Number of independent trials to run
        device: Torch device
        verbose: Print detailed statistics
        
    Returns:
        dict: Statistics including mean, std, and bias test results for each axis
    """
    device = torch.device(device)
    dt = 0.02  # Typical simulation dt
    max_speed = 2.0
    max_speed_z = max_speed / 2.0
    
    # Collect statistics across trials
    all_means = []
    all_stds = []
    all_raw_noise_means = []
    
    for trial in range(num_trials):
        # Generate random seeds for this trial
        noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=device)
        noise_freq = torch.rand(num_envs, 1, device=device) * 0.5 + 0.5  # [0.5, 1.0]
        noise_time = torch.zeros(num_envs, device=device)
        
        # Generate time grid
        t_start = noise_time.unsqueeze(1)
        t_steps = torch.arange(buffer_size, device=device).unsqueeze(0) * dt
        time_grid = t_start + t_steps
        
        # Generate raw noise
        raw_noise = batched_perlin_noise(time_grid, noise_seeds, noise_freq, device)
        
        # Scale to velocities
        scale = torch.tensor([max_speed, max_speed, max_speed_z], device=device)
        velocities = raw_noise * scale
        
        # Collect statistics
        # Shape: (num_envs, buffer_size, 3)
        trial_mean = velocities.mean(dim=(0, 1))  # Mean across envs and time
        trial_std = velocities.std(dim=(0, 1))
        raw_mean = raw_noise.mean(dim=(0, 1))
        
        all_means.append(trial_mean)
        all_stds.append(trial_std)
        all_raw_noise_means.append(raw_mean)
    
    # Aggregate results
    means = torch.stack(all_means)  # (num_trials, 3)
    stds = torch.stack(all_stds)
    raw_means = torch.stack(all_raw_noise_means)
    
    overall_mean = means.mean(dim=0)
    overall_std = stds.mean(dim=0)
    mean_of_means = raw_means.mean(dim=0)
    std_of_means = raw_means.std(dim=0)
    
    # Statistical test: is the mean significantly different from zero?
    # Using t-test like approach: if |mean| > 2*std/sqrt(n), likely biased
    n_samples = num_envs * buffer_size * num_trials
    se = std_of_means / math.sqrt(num_trials)  # Standard error of the mean
    z_scores = mean_of_means.abs() / (se + 1e-8)
    
    results = {
        'velocity_mean': overall_mean.cpu().numpy(),
        'velocity_std': overall_std.cpu().numpy(),
        'raw_noise_mean': mean_of_means.cpu().numpy(),
        'raw_noise_std_of_mean': std_of_means.cpu().numpy(),
        'z_scores': z_scores.cpu().numpy(),
        'is_biased': (z_scores > 2.0).cpu().numpy(),  # 95% confidence
        'total_samples': n_samples,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("PERLIN NOISE DISTRIBUTION VERIFICATION")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {buffer_size} steps × {num_trials} trials")
        print(f"Total samples per axis: {n_samples:,}")
        print()
        print("Raw Noise Statistics (should be ~0 mean, ~0.5-0.6 std):")
        print(f"  vx: mean={mean_of_means[0]:.6f}, std_of_mean={std_of_means[0]:.6f}")
        print(f"  vy: mean={mean_of_means[1]:.6f}, std_of_mean={std_of_means[1]:.6f}")
        print(f"  vz: mean={mean_of_means[2]:.6f}, std_of_mean={std_of_means[2]:.6f}")
        print()
        print("Velocity Statistics (after scaling):")
        print(f"  vx: mean={overall_mean[0]:.4f} m/s, std={overall_std[0]:.4f}")
        print(f"  vy: mean={overall_mean[1]:.4f} m/s, std={overall_std[1]:.4f}")
        print(f"  vz: mean={overall_mean[2]:.4f} m/s, std={overall_std[2]:.4f}")
        print()
        print("Bias Test (z-score > 2.0 indicates significant bias):")
        for i, axis in enumerate(['vx', 'vy', 'vz']):
            status = "⚠️  BIASED" if results['is_biased'][i] else "✓ OK"
            print(f"  {axis}: z-score={z_scores[i]:.2f} {status}")
        print("="*60)
    
    return results


def verify_hash_function_distribution(
    num_samples: int = 1000000,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify that the hash function used in _noise() produces uniform distribution.
    
    The hash: h = sin(x * 12.9898 + 78.233) * 43758.5453
    Should produce values uniformly distributed in [-1, 1].
    
    Args:
        num_samples: Number of samples to test
        device: Torch device
        verbose: Print statistics
        
    Returns:
        dict: Distribution statistics
    """
    device = torch.device(device)
    
    # Generate random inputs (simulating seed + position combinations)
    x = torch.randint(0, 100000, (num_samples,), device=device).float()
    
    # Apply the hash function
    h = torch.sin(x * 12.9898 + 78.233) * 43758.5453
    h = h - h.floor()  # Fractional part [0, 1)
    h = h * 2.0 - 1.0  # Convert to [-1, 1]
    
    mean = h.mean().item()
    std = h.std().item()
    
    # Check distribution in bins
    num_bins = 20
    hist = torch.histc(h, bins=num_bins, min=-1.0, max=1.0)
    expected_per_bin = num_samples / num_bins
    chi_squared = ((hist - expected_per_bin) ** 2 / expected_per_bin).sum().item()
    
    # For uniform distribution, values should be evenly spread
    # Skewness check
    skewness = ((h - mean) ** 3).mean() / (std ** 3 + 1e-8)
    
    results = {
        'mean': mean,
        'std': std,
        'expected_std': 1.0 / math.sqrt(3),  # Uniform[-1,1] has std = 1/sqrt(3) ≈ 0.577
        'skewness': skewness.item(),
        'chi_squared': chi_squared,
        'histogram': hist.cpu().numpy(),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("HASH FUNCTION DISTRIBUTION VERIFICATION")
        print("="*60)
        print(f"Samples tested: {num_samples:,}")
        print(f"Mean: {mean:.6f} (expected: 0.0)")
        print(f"Std:  {std:.4f} (expected for uniform[-1,1]: {1/math.sqrt(3):.4f})")
        print(f"Skewness: {skewness.item():.6f} (expected: 0.0)")
        print()
        print("Histogram (should be roughly uniform):")
        normalized_hist = hist / hist.sum()
        for i in range(num_bins):
            bar_len = int(normalized_hist[i].item() * 100)
            print(f"  [{-1 + i*0.1:+.1f}, {-1 + (i+1)*0.1:+.1f}): {'█' * bar_len}")
        print("="*60)
    
    return results


def verify_interpolation_bias(
    num_envs: int = 1000,
    num_steps: int = 1000,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify that the interpolation process doesn't introduce systematic bias.
    
    Checks if the Perlin noise output has zero mean across many samples.
    """
    device = torch.device(device)
    
    # Create perlin noise generators with random seeds
    seeds = torch.randint(0, 100000, (num_envs,), device=device)
    
    perlin = BatchedPerlinNoise(
        seeds=seeds,
        amplitude=1.0,
        frequency=1.0,
        octaves=1,
        interp=InterpType.COSINE,
        use_fade=False,
        device=device
    )
    
    # Sample at many continuous positions
    x = torch.linspace(0, 100, num_steps, device=device).unsqueeze(0).expand(num_envs, -1)
    
    noise = perlin.get(x)  # (num_envs, num_steps)
    
    mean = noise.mean().item()
    std = noise.std().item()
    per_env_mean = noise.mean(dim=1)  # (num_envs,)
    
    results = {
        'overall_mean': mean,
        'overall_std': std,
        'per_env_mean_mean': per_env_mean.mean().item(),
        'per_env_mean_std': per_env_mean.std().item(),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("INTERPOLATION BIAS VERIFICATION")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {num_steps} steps")
        print(f"Overall mean: {mean:.6f} (expected: ~0)")
        print(f"Overall std:  {std:.4f}")
        print(f"Per-env mean distribution: μ={per_env_mean.mean():.6f}, σ={per_env_mean.std():.4f}")
        
        # Check if any individual env has strong bias
        biased_envs = (per_env_mean.abs() > 0.3).sum().item()
        print(f"Envs with |mean| > 0.3: {biased_envs} ({100*biased_envs/num_envs:.1f}%)")
        print("="*60)
    
    return results


def run_all_verifications(device: str = "cuda"):
    """
    Run all verification tests and provide a summary.
    """
    print("\n" + "🔍 "*20)
    print("RUNNING ALL NOISE DISTRIBUTION VERIFICATIONS")
    print("🔍 "*20)
    
    # Test 1: Hash function
    hash_results = verify_hash_function_distribution(device=device)
    
    # Test 2: Interpolation
    interp_results = verify_interpolation_bias(device=device)
    
    # Test 3: Full Perlin noise
    noise_results = verify_noise_distribution(device=device)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues_found = []
    
    if abs(hash_results['mean']) > 0.01:
        issues_found.append(f"Hash function has non-zero mean: {hash_results['mean']:.6f}")
    
    if abs(interp_results['overall_mean']) > 0.05:
        issues_found.append(f"Interpolation introduces bias: {interp_results['overall_mean']:.6f}")
    
    for i, axis in enumerate(['vx', 'vy', 'vz']):
        if noise_results['is_biased'][i]:
            issues_found.append(f"{axis} shows significant bias: mean={noise_results['raw_noise_mean'][i]:.6f}")
    
    if issues_found:
        print("⚠️  ISSUES DETECTED:")
        for issue in issues_found:
            print(f"   - {issue}")
        print("\nRECOMMENDATION: Consider using a better hash function or")
        print("                adding bias correction to the noise output.")
    else:
        print("✓ All tests passed - no systematic bias detected.")
    
    print("="*60)
    
    return {
        'hash': hash_results,
        'interpolation': interp_results,
        'noise': noise_results,
        'issues': issues_found
    }


def verify_trajectory_drift(
    num_envs: int = 500,
    num_episodes: int = 5,
    episode_length: int = 500,
    buffer_size: int = 128,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify trajectory drift over time by simulating actual trajectory integration.
    
    This tests whether the drone tends to drift in any direction (especially -Z)
    over multiple episodes when following the generated velocity commands.
    
    Args:
        num_envs: Number of environments
        num_episodes: Number of complete episodes to simulate
        episode_length: Steps per episode  
        buffer_size: UserModel buffer size
        device: Torch device
        verbose: Print detailed statistics
        
    Returns:
        dict: Drift statistics per axis
    """
    device = torch.device(device)
    dt = 0.02
    max_speed = 2.0
    max_speed_z = max_speed / 2.0
    
    # Track positions and velocities
    all_final_positions = []
    all_mean_velocities = []
    all_position_histories = []
    
    for episode in range(num_episodes):
        # Initialize position at origin
        pos = torch.zeros(num_envs, 3, device=device)
        pos[:, 2] = 5.0  # Start at z=5
        
        # Generate new seeds for each episode
        noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=device)
        noise_freq = torch.rand(num_envs, 1, device=device) * 0.5 + 0.5
        noise_time = torch.zeros(num_envs, device=device)
        
        episode_velocities = []
        position_history = [pos.clone()]
        
        # Simulate episode
        steps_done = 0
        while steps_done < episode_length:
            # Generate buffer of velocities
            t_start = noise_time.unsqueeze(1)
            t_steps = torch.arange(buffer_size, device=device).unsqueeze(0) * dt
            time_grid = t_start + t_steps
            
            raw_noise = batched_perlin_noise(time_grid, noise_seeds, noise_freq, device)
            scale = torch.tensor([max_speed, max_speed, max_speed_z], device=device)
            velocities = raw_noise * scale  # (num_envs, buffer_size, 3)
            
            # Integrate positions
            steps_to_use = min(buffer_size, episode_length - steps_done)
            for t in range(steps_to_use):
                vel = velocities[:, t, :]
                pos = pos + vel * dt
                episode_velocities.append(vel.clone())
                position_history.append(pos.clone())
            
            noise_time += buffer_size * dt
            steps_done += buffer_size
        
        # Record final position relative to start
        final_displacement = pos.clone()
        final_displacement[:, 2] -= 5.0  # Subtract starting height
        all_final_positions.append(final_displacement)
        
        # Record mean velocity
        mean_vel = torch.stack(episode_velocities, dim=1).mean(dim=1)
        all_mean_velocities.append(mean_vel)
        
        all_position_histories.append(torch.stack(position_history, dim=1))
    
    # Analyze results
    final_positions = torch.stack(all_final_positions)  # (num_episodes, num_envs, 3)
    mean_velocities = torch.stack(all_mean_velocities)  # (num_episodes, num_envs, 3)
    
    # Overall statistics
    mean_displacement = final_positions.mean(dim=(0, 1))
    std_displacement = final_positions.std(dim=(0, 1))
    mean_velocity = mean_velocities.mean(dim=(0, 1))
    std_velocity = mean_velocities.std(dim=(0, 1))
    
    # Per-axis drift rate (displacement per second)
    total_time = episode_length * dt
    drift_rate = mean_displacement / total_time
    
    results = {
        'mean_displacement': mean_displacement.cpu().numpy(),
        'std_displacement': std_displacement.cpu().numpy(),
        'mean_velocity': mean_velocity.cpu().numpy(),
        'std_velocity': std_velocity.cpu().numpy(),
        'drift_rate_per_sec': drift_rate.cpu().numpy(),
        'total_time': total_time,
        'episode_length': episode_length,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TRAJECTORY DRIFT VERIFICATION")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {num_episodes} episodes × {episode_length} steps")
        print(f"Episode duration: {total_time:.1f}s")
        print()
        print("Mean Final Displacement (from start):")
        print(f"  X: {mean_displacement[0]:+.3f} ± {std_displacement[0]:.3f} m")
        print(f"  Y: {mean_displacement[1]:+.3f} ± {std_displacement[1]:.3f} m")
        print(f"  Z: {mean_displacement[2]:+.3f} ± {std_displacement[2]:.3f} m")
        print()
        print("Mean Velocity over episodes:")
        print(f"  vx: {mean_velocity[0]:+.4f} ± {std_velocity[0]:.4f} m/s")
        print(f"  vy: {mean_velocity[1]:+.4f} ± {std_velocity[1]:.4f} m/s")
        print(f"  vz: {mean_velocity[2]:+.4f} ± {std_velocity[2]:.4f} m/s")
        print()
        print("Drift Rate (displacement per second):")
        print(f"  X: {drift_rate[0]:+.4f} m/s")
        print(f"  Y: {drift_rate[1]:+.4f} m/s")
        print(f"  Z: {drift_rate[2]:+.4f} m/s")
        print()
        
        # Statistical significance
        n = num_envs * num_episodes
        se = std_displacement / math.sqrt(n)
        z_scores = mean_displacement.abs() / (se + 1e-8)
        
        print("Bias Significance (|z-score| > 2 indicates real drift):")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            status = "⚠️  DRIFTING" if z_scores[i] > 2 else "✓ OK"
            print(f"  {axis}: z-score={z_scores[i]:.2f} {status}")
        print("="*60)
    
    return results


def verify_perlin_time_correlation(
    num_envs: int = 100,
    num_steps: int = 2000,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify if Perlin noise has any temporal correlation that could cause drift.
    
    Perlin noise is smooth and correlated, which means if it starts positive,
    it tends to stay positive for a while. This could cause position drift.
    """
    device = torch.device(device)
    dt = 0.02
    
    # Generate long trajectories
    noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=device)
    noise_freq = torch.rand(num_envs, 1, device=device) * 0.5 + 0.5
    
    t_start = torch.zeros(num_envs, 1, device=device)
    t_steps = torch.arange(num_steps, device=device).unsqueeze(0) * dt
    time_grid = t_start + t_steps
    
    raw_noise = batched_perlin_noise(time_grid, noise_seeds, noise_freq, device)
    
    # Analyze autocorrelation
    # Check mean of cumulative sum (this is what causes drift)
    cumsum = raw_noise.cumsum(dim=1)  # (num_envs, num_steps, 3)
    
    # Theoretical: for zero-mean noise, cumsum should be a random walk
    # The expected value of cumsum at step N should be 0
    # But variance grows with N
    
    final_cumsum = cumsum[:, -1, :]  # (num_envs, 3)
    mean_cumsum = final_cumsum.mean(dim=0)
    std_cumsum = final_cumsum.std(dim=0)
    
    # Expected std for random walk: sqrt(N) * noise_std
    noise_std = raw_noise.std(dim=(0, 1))
    expected_std = math.sqrt(num_steps) * noise_std
    
    results = {
        'mean_cumsum': mean_cumsum.cpu().numpy(),
        'std_cumsum': std_cumsum.cpu().numpy(),
        'expected_std': expected_std.cpu().numpy(),
        'noise_std': noise_std.cpu().numpy(),
        'num_steps': num_steps,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("PERLIN NOISE TEMPORAL CORRELATION ANALYSIS")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {num_steps} steps")
        print()
        print("Cumulative Sum at Final Step (should be ~0 mean):")
        print(f"  vx: mean={mean_cumsum[0]:+.2f}, std={std_cumsum[0]:.2f}")
        print(f"  vy: mean={mean_cumsum[1]:+.2f}, std={std_cumsum[1]:.2f}")
        print(f"  vz: mean={mean_cumsum[2]:+.2f}, std={std_cumsum[2]:.2f}")
        print()
        print("Expected std for random walk: sqrt(N) * noise_std")
        print(f"  vx: expected={expected_std[0]:.2f}, actual={std_cumsum[0]:.2f}")
        print(f"  vy: expected={expected_std[1]:.2f}, actual={std_cumsum[1]:.2f}")
        print(f"  vz: expected={expected_std[2]:.2f}, actual={std_cumsum[2]:.2f}")
        print()
        
        # Check if actual std is higher than expected (indicating positive correlation)
        correlation_factor = std_cumsum / expected_std
        print("Correlation Factor (>1 means positive autocorrelation):")
        print(f"  vx: {correlation_factor[0]:.2f}")
        print(f"  vy: {correlation_factor[1]:.2f}")
        print(f"  vz: {correlation_factor[2]:.2f}")
        print()
        print("Note: Perlin noise has positive autocorrelation by design.")
        print("This is expected but does NOT cause systematic drift if mean=0.")
        print("="*60)
    
    return results


def verify_usermodel_with_boundaries(
    num_envs: int = 200,
    episode_length: int = 1000,
    map_range: list = [10.0, 10.0, 5.0],
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Simulate UserModel with realistic boundary conditions.
    
    This tests whether boundary interactions cause asymmetric drift.
    Key insight: If drones are spawned at z=5 in a map with z_range=[0,10],
    they have more room to go up than down (floor at z=1).
    
    When drones hit the lower boundary more often, they get "stopped" there,
    which can make trajectories appear to have a downward bias even if the
    underlying noise is unbiased.
    """
    device = torch.device(device)
    dt = 0.02
    max_speed = 2.0
    max_speed_z = max_speed / 2.0
    buffer_size = 128
    
    map_range = torch.tensor(map_range, device=device)
    
    # Track statistics
    position_samples = []
    velocity_samples = []
    boundary_hits = {'floor': 0, 'ceiling': 0, 'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
    
    # Initialize at various positions
    pos = torch.zeros(num_envs, 3, device=device)
    pos[:, 0] = (torch.rand(num_envs, device=device) - 0.5) * 2 * map_range[0] * 0.5
    pos[:, 1] = (torch.rand(num_envs, device=device) - 0.5) * 2 * map_range[1] * 0.5
    pos[:, 2] = torch.rand(num_envs, device=device) * (2 * map_range[2] - 1.0) + 1.0  # [1, 2*Lz]
    
    initial_pos = pos.clone()
    
    # Generate noise parameters
    noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=device)
    noise_freq = torch.rand(num_envs, 1, device=device) * 0.5 + 0.5
    noise_time = torch.zeros(num_envs, device=device)
    
    # Simulate
    steps_done = 0
    while steps_done < episode_length:
        t_start = noise_time.unsqueeze(1)
        t_steps = torch.arange(buffer_size, device=device).unsqueeze(0) * dt
        time_grid = t_start + t_steps
        
        raw_noise = batched_perlin_noise(time_grid, noise_seeds, noise_freq, device)
        scale = torch.tensor([max_speed, max_speed, max_speed_z], device=device)
        velocities = raw_noise * scale
        
        steps_to_use = min(buffer_size, episode_length - steps_done)
        for t in range(steps_to_use):
            vel = velocities[:, t, :]
            velocity_samples.append(vel.mean(dim=0).clone())
            
            next_pos = pos + vel * dt
            
            # Apply boundary clamping and count hits
            # X boundaries
            x_min_hit = next_pos[:, 0] < -map_range[0]
            x_max_hit = next_pos[:, 0] > map_range[0]
            boundary_hits['x_min'] += x_min_hit.sum().item()
            boundary_hits['x_max'] += x_max_hit.sum().item()
            next_pos[:, 0] = torch.clamp(next_pos[:, 0], -map_range[0], map_range[0])
            
            # Y boundaries
            y_min_hit = next_pos[:, 1] < -map_range[1]
            y_max_hit = next_pos[:, 1] > map_range[1]
            boundary_hits['y_min'] += y_min_hit.sum().item()
            boundary_hits['y_max'] += y_max_hit.sum().item()
            next_pos[:, 1] = torch.clamp(next_pos[:, 1], -map_range[1], map_range[1])
            
            # Z boundaries (asymmetric: floor at 1.0, ceiling at 2*Lz)
            floor_hit = next_pos[:, 2] < 1.0
            ceiling_hit = next_pos[:, 2] > 2 * map_range[2]
            boundary_hits['floor'] += floor_hit.sum().item()
            boundary_hits['ceiling'] += ceiling_hit.sum().item()
            next_pos[:, 2] = torch.clamp(next_pos[:, 2], 1.0, 2 * map_range[2])
            
            pos = next_pos
            position_samples.append(pos.mean(dim=0).clone())
        
        noise_time += buffer_size * dt
        steps_done += buffer_size
    
    # Analyze
    positions = torch.stack(position_samples)  # (episode_length, 3)
    velocities_stacked = torch.stack(velocity_samples)
    
    final_displacement = pos - initial_pos
    mean_final_disp = final_displacement.mean(dim=0)
    std_final_disp = final_displacement.std(dim=0)
    
    mean_velocity = velocities_stacked.mean(dim=0)
    mean_position = positions.mean(dim=0)
    
    results = {
        'mean_final_displacement': mean_final_disp.cpu().numpy(),
        'std_final_displacement': std_final_disp.cpu().numpy(),
        'mean_velocity': mean_velocity.cpu().numpy(),
        'mean_position': mean_position.cpu().numpy(),
        'boundary_hits': boundary_hits,
        'initial_mean_z': initial_pos[:, 2].mean().item(),
        'final_mean_z': pos[:, 2].mean().item(),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("USERMODEL WITH BOUNDARIES VERIFICATION")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {episode_length} steps")
        print(f"Map range: X=[-{map_range[0]}, {map_range[0]}], Y=[-{map_range[1]}, {map_range[1]}], Z=[1.0, {2*map_range[2]}]")
        print()
        print("Boundary Collisions:")
        print(f"  Floor (z=1.0): {boundary_hits['floor']:,}")
        print(f"  Ceiling (z={2*map_range[2]:.0f}): {boundary_hits['ceiling']:,}")
        print(f"  X_min: {boundary_hits['x_min']:,}, X_max: {boundary_hits['x_max']:,}")
        print(f"  Y_min: {boundary_hits['y_min']:,}, Y_max: {boundary_hits['y_max']:,}")
        print()
        print("Position Statistics:")
        print(f"  Initial mean Z: {initial_pos[:, 2].mean():.2f}")
        print(f"  Final mean Z:   {pos[:, 2].mean():.2f}")
        print(f"  Mean Z over time: {mean_position[2]:.2f}")
        print()
        print("Mean Final Displacement:")
        print(f"  X: {mean_final_disp[0]:+.3f} ± {std_final_disp[0]:.3f} m")
        print(f"  Y: {mean_final_disp[1]:+.3f} ± {std_final_disp[1]:.3f} m")
        print(f"  Z: {mean_final_disp[2]:+.3f} ± {std_final_disp[2]:.3f} m")
        print()
        print("Mean Velocity (should be ~0):")
        print(f"  vx: {mean_velocity[0]:+.4f} m/s")
        print(f"  vy: {mean_velocity[1]:+.4f} m/s")
        print(f"  vz: {mean_velocity[2]:+.4f} m/s")
        print()
        
        if boundary_hits['floor'] > boundary_hits['ceiling'] * 1.5:
            print("⚠️  WARNING: Floor hits >> Ceiling hits")
            print("   This asymmetry may cause apparent downward drift!")
            print("   Possible causes:")
            print("   1. Drones spawn too high relative to ceiling")
            print("   2. Floor boundary (z=1.0) is too restrictive")
            print("   3. Z velocity scale is asymmetric or biased")
        elif abs(mean_final_disp[2]) > 1.0:
            print(f"⚠️  WARNING: Significant Z displacement detected!")
        else:
            print("✓ No significant drift detected in this simulation.")
        print("="*60)
    
    return results


def diagnose_z_drift_issue(device: str = "cuda"):
    """
    Comprehensive diagnosis of the Z-axis drift issue.
    
    Runs multiple tests to identify the root cause.
    """
    print("\n" + "🔍 "*20)
    print("Z-AXIS DRIFT DIAGNOSIS")
    print("🔍 "*20)
    
    # Test 1: Raw noise distribution
    print("\n[Test 1] Checking raw noise distribution...")
    noise_results = verify_noise_distribution(num_envs=2000, num_trials=20, device=device, verbose=False)
    z_bias = noise_results['raw_noise_mean'][2]
    print(f"  Z-axis noise mean: {z_bias:.6f}")
    if abs(z_bias) > 0.01:
        print(f"  ⚠️ Small bias detected in noise generation")
    else:
        print(f"  ✓ Noise appears unbiased")
    
    # Test 2: Trajectory integration without boundaries
    print("\n[Test 2] Checking trajectory drift WITHOUT boundaries...")
    drift_results = verify_trajectory_drift(num_envs=1000, num_episodes=10, device=device, verbose=False)
    z_drift = drift_results['drift_rate_per_sec'][2]
    print(f"  Z-axis drift rate: {z_drift:+.4f} m/s")
    if abs(z_drift) > 0.05:
        print(f"  ⚠️ Significant drift detected even without boundaries")
    else:
        print(f"  ✓ No significant drift without boundaries")
    
    # Test 3: With boundaries
    print("\n[Test 3] Checking trajectory WITH boundaries...")
    boundary_results = verify_usermodel_with_boundaries(num_envs=500, episode_length=2000, device=device, verbose=False)
    z_disp = boundary_results['mean_final_displacement'][2]
    floor_hits = boundary_results['boundary_hits']['floor']
    ceiling_hits = boundary_results['boundary_hits']['ceiling']
    print(f"  Mean Z displacement: {z_disp:+.3f} m")
    print(f"  Floor/Ceiling hit ratio: {floor_hits}/{ceiling_hits}")
    
    # Test 4: Check velocity scaling
    print("\n[Test 4] Checking velocity scaling...")
    max_speed = 2.0
    max_speed_z = max_speed / 2.0
    print(f"  max_speed (XY): {max_speed} m/s")
    print(f"  max_speed_z:    {max_speed_z} m/s")
    print(f"  Z is scaled to {max_speed_z/max_speed*100:.0f}% of XY speed")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    issues = []
    recommendations = []
    
    if abs(z_bias) > 0.01:
        issues.append("Noise generation has small Z-bias")
        recommendations.append("Apply bias correction to raw noise: raw_noise -= raw_noise.mean()")
    
    if abs(z_drift) > 0.05:
        issues.append("Trajectory integration shows Z-drift")
        recommendations.append("Check if drone controller interprets velocity commands correctly")
    
    if floor_hits > ceiling_hits * 2:
        issues.append("Floor boundary hit more often than ceiling")
        recommendations.append("Consider adjusting spawn height or Z velocity range")
    
    if not issues:
        issues.append("No issues detected in UserModel noise generation")
        recommendations.append("The drift may be caused by:")
        recommendations.append("  - Drone dynamics/controller behavior")
        recommendations.append("  - Gravity compensation issues")  
        recommendations.append("  - Observation/action frame misalignment")
        recommendations.append("  - External factors in your simulation environment")
    
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  → {rec}")
    
    print("="*60)
    
    return {
        'noise_bias': z_bias,
        'trajectory_drift': z_drift,
        'boundary_asymmetry': floor_hits / (ceiling_hits + 1),
        'issues': issues,
        'recommendations': recommendations
    }


def verify_body_to_world_transform(
    num_envs: int = 100,
    episode_length: int = 500,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Verify if body-to-world velocity transformation could cause Z-drift.
    
    When a drone is tilted (non-zero pitch/roll), body frame velocity 
    commands get transformed to world frame. If there's a systematic 
    tilt bias, this could cause apparent Z-drift.
    
    For example: if drone consistently pitches forward, a body-frame 
    forward velocity (vx_body) would have a downward component in world frame.
    """
    from omni_drones.utils.torch import quat_rotate, euler_to_quaternion
    
    device = torch.device(device)
    dt = 0.02
    max_speed = 2.0
    max_speed_z = max_speed / 2.0
    
    # Simulate with random pitch/roll tilts
    pitch_angles = torch.randn(num_envs, device=device) * 0.1  # ~6 degrees std
    roll_angles = torch.randn(num_envs, device=device) * 0.1
    yaw_angles = torch.rand(num_envs, device=device) * 2 * math.pi
    
    # Create quaternions from euler angles
    # euler_to_quaternion expects (N, 3) with [roll, pitch, yaw]
    euler = torch.stack([roll_angles, pitch_angles, yaw_angles], dim=-1)
    quats = euler_to_quaternion(euler)  # (N, 4)
    
    # Generate unbiased body velocities
    noise_seeds = torch.randint(0, 100000, (num_envs, 3), device=device)
    noise_freq = torch.ones(num_envs, 1, device=device) * 0.5
    noise_time = torch.zeros(num_envs, device=device)
    
    t_start = noise_time.unsqueeze(1)
    t_steps = torch.arange(episode_length, device=device).unsqueeze(0) * dt
    time_grid = t_start + t_steps
    
    raw_noise = batched_perlin_noise(time_grid, noise_seeds, noise_freq, device)
    scale = torch.tensor([max_speed, max_speed, max_speed_z], device=device)
    body_velocities = raw_noise * scale  # (N, T, 3)
    
    # Transform to world frame
    world_velocities = []
    for t in range(episode_length):
        vel_body = body_velocities[:, t, :]  # (N, 3)
        vel_world = quat_rotate(quats, vel_body)  # (N, 3)
        world_velocities.append(vel_world)
    
    world_velocities = torch.stack(world_velocities, dim=1)  # (N, T, 3)
    
    # Compare means
    body_mean = body_velocities.mean(dim=(0, 1))
    world_mean = world_velocities.mean(dim=(0, 1))
    
    # The difference shows how tilt affects velocity distribution
    transform_bias = world_mean - body_mean
    
    results = {
        'body_velocity_mean': body_mean.cpu().numpy(),
        'world_velocity_mean': world_mean.cpu().numpy(),
        'transform_bias': transform_bias.cpu().numpy(),
        'mean_pitch': pitch_angles.mean().item(),
        'mean_roll': roll_angles.mean().item(),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("BODY-TO-WORLD TRANSFORM VERIFICATION")
        print("="*60)
        print(f"Configuration: {num_envs} envs × {episode_length} steps")
        print(f"Pitch std: {pitch_angles.std().item()*180/math.pi:.1f}°")
        print(f"Roll std: {roll_angles.std().item()*180/math.pi:.1f}°")
        print()
        print("Body Frame Velocity Mean:")
        print(f"  vx: {body_mean[0]:+.4f} m/s")
        print(f"  vy: {body_mean[1]:+.4f} m/s")
        print(f"  vz: {body_mean[2]:+.4f} m/s")
        print()
        print("World Frame Velocity Mean (after rotation):")
        print(f"  vx: {world_mean[0]:+.4f} m/s")
        print(f"  vy: {world_mean[1]:+.4f} m/s")
        print(f"  vz: {world_mean[2]:+.4f} m/s")
        print()
        print("Transform-Induced Bias (World - Body):")
        print(f"  Δvx: {transform_bias[0]:+.4f} m/s")
        print(f"  Δvy: {transform_bias[1]:+.4f} m/s")
        print(f"  Δvz: {transform_bias[2]:+.4f} m/s")
        print()
        
        if abs(transform_bias[2]) > 0.05:
            print("⚠️  WARNING: Tilt causes significant Z-velocity bias!")
            print("   This could cause apparent downward drift if drone")
            print("   maintains consistent pitch during flight.")
        else:
            print("✓ Transform bias is negligible with random tilts.")
        print("="*60)
    
    return results


def run_full_diagnosis(device: str = "cuda"):
    """
    Run complete diagnosis including all verification tests.
    
    Usage:
        from src.simulated_users.user_model_diagnostics import run_full_diagnosis
        run_full_diagnosis()
    """
    print("\n" + "="*70)
    print("COMPLETE USERMODEL VELOCITY GENERATION DIAGNOSIS")
    print("="*70)
    
    # 1. Hash function
    print("\n[1/5] Hash Function Distribution...")
    verify_hash_function_distribution(device=device, verbose=True)
    
    # 2. Interpolation
    print("\n[2/5] Interpolation Bias...")
    verify_interpolation_bias(device=device, verbose=True)
    
    # 3. Full Perlin noise
    print("\n[3/5] Perlin Noise Distribution...")
    verify_noise_distribution(device=device, verbose=True)
    
    # 4. Trajectory drift
    print("\n[4/5] Trajectory Drift (without boundaries)...")
    verify_trajectory_drift(device=device, verbose=True)
    
    # 5. Body-to-world transform
    print("\n[5/5] Body-to-World Transform Bias...")
    verify_body_to_world_transform(device=device, verbose=True)
    
    # Final diagnosis
    print("\n" + "="*70)
    print("FINAL DIAGNOSIS")
    print("="*70)
    
    print("""
Based on the tests above:

1. If ALL tests show ✓ OK:
   → The UserModel noise generation is unbiased
   → The Z-drift you observe is likely caused by:
     a) Drone controller dynamics (gravity compensation, thrust limits)
     b) Simulation environment factors
     c) Actor network behavior during training
     d) Reward function incentivizing certain behaviors

2. Recommended next steps:
   → Log actual drone Z-velocity during training episodes
   → Compare commanded velocity vs actual achieved velocity
   → Check if actor network outputs biased actions
   → Verify gravity compensation in your velocity controller
""")
    print("="*70)


# Main entry point for command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UserModel Diagnostics")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnosis only")
    
    args = parser.parse_args()
    
    if args.quick:
        diagnose_z_drift_issue(device=args.device)
    else:
        run_full_diagnosis(device=args.device)
