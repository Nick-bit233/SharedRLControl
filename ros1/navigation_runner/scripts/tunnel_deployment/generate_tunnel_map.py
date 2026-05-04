#!/usr/bin/env python3
"""
Generate a tunnel obstacle map as PCD + matching Gazebo world.

Matches the IsaacSim training environment (EnvTunnelResidual).

In IsaacSim, the terrain is generated with:
    TerrainGeneratorCfg(size=(24.0, 12.0), ...)
    map_range = [6.0, 12.0, 5.0]  # config [x, y, z] half-extents
    -> [y, x, z] in isaacsim coordinates (axes swapped)

The FORWARD (travel) direction is 24m, LATERAL is 12m, HEIGHT is 10m.

In Gazebo (ROS), the drone faces +X, so:
    X = forward (24m) :  [-12, 12]
    Y = lateral (12m) :  [-6, 6]
    Z = vertical (10m):  [0, 10]

The terrain has a clear spawn zone at the -X end:
    X ∈ [-12, -6] : cleared (no obstacles), drone spawns at X≈-8
    X ∈ [-6, +12] : obstacle zone (18m of mixed cuboid/cylinder obstacles)
    Back wall at X ≈ -10 (behind drone)
    Side walls at Y = ±6

Obstacle types (mixed, similar to slope_inspection/mockamap):
    - Cylinders:  circular cross-section, random radius
    - Cuboids:    rectangular cross-section, random side lengths

A spawn protection zone (3m radius around spawn) is kept obstacle-free.

Usage:
    python3 generate_tunnel_map.py -o tunnel_map.pcd -w tunnel.world --seed 42 -n 60
"""
import argparse
import json
import numpy as np
from typing import Dict, List, NamedTuple, Optional, Tuple


class Obstacle(NamedTuple):
    """One obstacle in the tunnel."""
    cx: float          # centre X
    cy: float          # centre Y
    shape: str         # 'cylinder' or 'cuboid'
    # For cylinder: size_x = size_y = diameter; for cuboid: independent
    size_x: float      # full extent along X
    size_y: float      # full extent along Y
    height: float      # Z extent (from ground)


# ── Protection zone around spawn point ──────────────────────────────
SPAWN_X = -8.0
SPAWN_Y = 0.0
SPAWN_PROTECT_RADIUS = 3.0  # no obstacles within this radius of spawn


def _in_protection_zone(cx: float, cy: float, half_size: float) -> bool:
    """True if an obstacle (approximated as circle with given half_size) overlaps
    the spawn protection zone."""
    dx = cx - SPAWN_X
    dy = cy - SPAWN_Y
    dist = np.sqrt(dx * dx + dy * dy)
    return dist < (SPAWN_PROTECT_RADIUS + half_size)


def _footprint_radius(obs: Obstacle) -> float:
    return max(obs.size_x, obs.size_y) / 2.0


def _footprint_area(obs: Obstacle) -> float:
    if obs.shape == "cylinder":
        radius = obs.size_x / 2.0
        return float(np.pi * radius * radius)
    return float(obs.size_x * obs.size_y)


def _footprint_gap(a: Obstacle, b: Obstacle) -> float:
    center_dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
    return center_dist - _footprint_radius(a) - _footprint_radius(b)


def _local_density_stats(
    obstacles: List[Obstacle],
    window: float,
) -> Tuple[int, float]:
    if not obstacles or window <= 0.0:
        return 0, 0.0

    max_count = 0
    max_area_fraction = 0.0
    half_window = window / 2.0
    area = window * window
    for center in obstacles:
        included = [
            obs for obs in obstacles
            if abs(obs.cx - center.cx) <= half_window
            and abs(obs.cy - center.cy) <= half_window
        ]
        max_count = max(max_count, len(included))
        max_area_fraction = max(
            max_area_fraction,
            sum(_footprint_area(obs) for obs in included) / area,
        )
    return int(max_count), float(max_area_fraction)


def _candidate_rejection_reason(
    candidate: Obstacle,
    obstacles: List[Obstacle],
    min_obstacle_spacing: float,
    local_density_window: float,
    max_obstacles_per_window: int,
    max_local_area_fraction: float,
) -> Optional[str]:
    radius = _footprint_radius(candidate)
    if _in_protection_zone(candidate.cx, candidate.cy, radius):
        return "spawn_protection"

    if min_obstacle_spacing > 0.0:
        for obs in obstacles:
            if _footprint_gap(candidate, obs) < min_obstacle_spacing:
                return "min_spacing"

    trial = obstacles + [candidate]
    max_count, max_area_fraction = _local_density_stats(trial, local_density_window)
    if max_obstacles_per_window > 0 and max_count > max_obstacles_per_window:
        return "local_count"
    if max_local_area_fraction > 0.0 and max_area_fraction > max_local_area_fraction:
        return "local_area_fraction"
    return None


def _sample_obstacle(
    rng: np.random.RandomState,
    x_half: float,
    y_half: float,
    z_max: float,
    obstacle_width_range: tuple,
    obstacle_height_range: tuple,
    obstacle_zone_x_min: float,
    cuboid_ratio: float,
) -> Obstacle:
    margin = 1.0
    cx = rng.uniform(obstacle_zone_x_min + margin, x_half - margin)
    cy = rng.uniform(-y_half + margin, y_half - margin)
    height = min(rng.uniform(*obstacle_height_range), z_max)

    if rng.random() < cuboid_ratio:
        sx = rng.uniform(*obstacle_width_range)
        sy = rng.uniform(*obstacle_width_range)
        return Obstacle(cx, cy, "cuboid", sx, sy, height)

    diameter = rng.uniform(*obstacle_width_range)
    return Obstacle(cx, cy, "cylinder", diameter, diameter, height)


def _connectivity_metrics(
    obstacles: List[Obstacle],
    x_half: float,
    y_half: float,
    clearance: float = 0.2,
    grid_resolution: float = 0.25,
) -> Dict[str, object]:
    """Approximate 2-D traversability through obstacle footprints."""
    xs = np.arange(-x_half + grid_resolution / 2.0, x_half, grid_resolution)
    ys = np.arange(-y_half + grid_resolution / 2.0, y_half, grid_resolution)
    nx, ny = len(xs), len(ys)
    occupied = np.zeros((nx, ny), dtype=bool)
    clearance = max(0.0, float(clearance))

    for obs in obstacles:
        radius = _footprint_radius(obs) + clearance
        if obs.shape == "cylinder":
            dx = xs[:, None] - obs.cx
            dy = ys[None, :] - obs.cy
            occupied |= (dx * dx + dy * dy) <= radius * radius
        else:
            hx = obs.size_x / 2.0 + clearance
            hy = obs.size_y / 2.0 + clearance
            occupied |= (
                (np.abs(xs[:, None] - obs.cx) <= hx)
                & (np.abs(ys[None, :] - obs.cy) <= hy)
            )

    def nearest_cell(x: float, y: float) -> Tuple[int, int]:
        ix = int(np.clip(np.argmin(np.abs(xs - x)), 0, nx - 1))
        iy = int(np.clip(np.argmin(np.abs(ys - y)), 0, ny - 1))
        return ix, iy

    start = nearest_cell(SPAWN_X, SPAWN_Y)
    goal_cells = [(nx - 1, iy) for iy in range(ny) if not occupied[nx - 1, iy]]
    if occupied[start] or not goal_cells:
        return {
            "connected": False,
            "grid_resolution": float(grid_resolution),
            "clearance": float(clearance),
            "free_fraction": float(1.0 - np.mean(occupied)),
            "visited_cells": 0,
        }

    from collections import deque

    visited = np.zeros_like(occupied, dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True
    goal_set = set(goal_cells)
    reached = None
    while queue:
        cell = queue.popleft()
        if cell in goal_set:
            reached = cell
            break
        ix, iy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (ix + dx, iy + dy)
            if (
                0 <= nxt[0] < nx
                and 0 <= nxt[1] < ny
                and not occupied[nxt]
                and not visited[nxt]
            ):
                visited[nxt] = True
                parent[nxt] = cell
                queue.append(nxt)

    path_length = 0
    if reached is not None:
        cell = reached
        while cell != start:
            path_length += 1
            cell = parent[cell]

    return {
        "connected": reached is not None,
        "grid_resolution": float(grid_resolution),
        "clearance": float(clearance),
        "free_fraction": float(1.0 - np.mean(occupied)),
        "visited_cells": int(np.sum(visited)),
        "path_length_m": float(path_length * grid_resolution) if reached is not None else 0.0,
    }


def compute_feasibility_metrics(
    obstacles: List[Obstacle],
    x_half: float,
    y_half: float,
    local_density_window: float,
    min_bottleneck_width: float,
) -> Dict[str, object]:
    gaps = [
        _footprint_gap(a, b)
        for idx, a in enumerate(obstacles)
        for b in obstacles[idx + 1:]
    ]
    max_count, max_area_fraction = _local_density_stats(obstacles, local_density_window)
    connectivity = _connectivity_metrics(
        obstacles,
        x_half=x_half,
        y_half=y_half,
        clearance=max(0.0, min_bottleneck_width / 2.0),
    )
    return {
        "min_footprint_gap": float(np.min(gaps)) if gaps else float("inf"),
        "mean_footprint_gap": float(np.mean(gaps)) if gaps else float("inf"),
        "max_obstacles_per_local_window": int(max_count),
        "max_local_area_fraction": float(max_area_fraction),
        "connectivity": connectivity,
    }


def generate_obstacle_params(
    num_obstacles: int = 60,
    x_half: float = 12.0,
    y_half: float = 6.0,
    z_max: float = 10.0,
    obstacle_width_range: tuple = (0.4, 1.0),
    obstacle_height_range: tuple = (4.0, 12.0),
    obstacle_zone_x_min: float = -6.0,
    cuboid_ratio: float = 0.5,
    seed: int = 42,
    sampling_mode: str = "uniform",
    min_obstacle_spacing: float = 0.0,
    local_density_window: float = 3.0,
    max_obstacles_per_window: int = 4,
    max_local_area_fraction: float = 0.45,
    require_connectivity: bool = False,
    min_bottleneck_width: float = 0.4,
    max_resample_attempts: int = 2000,
    return_stats: bool = False,
) -> List[Obstacle]:
    """
    Generate a mixed list of cylinder and cuboid obstacle parameters.
    Obstacles are placed only in the obstacle zone (x >= obstacle_zone_x_min)
    and outside the spawn protection zone.
    Deterministic given the seed.
    """
    rng = np.random.RandomState(seed)
    obstacles: List[Obstacle] = []
    attempts = 0
    max_attempts = (
        num_obstacles * 10
        if sampling_mode == "uniform"
        else max(int(max_resample_attempts), num_obstacles)
    )
    rejections = {
        "spawn_protection": 0,
        "min_spacing": 0,
        "local_count": 0,
        "local_area_fraction": 0,
        "connectivity": 0,
    }

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        candidate = _sample_obstacle(
            rng,
            x_half,
            y_half,
            z_max,
            obstacle_width_range,
            obstacle_height_range,
            obstacle_zone_x_min,
            cuboid_ratio,
        )

        if sampling_mode == "uniform":
            if _in_protection_zone(candidate.cx, candidate.cy, _footprint_radius(candidate)):
                rejections["spawn_protection"] += 1
                continue
            obstacles.append(candidate)
        else:
            reason = _candidate_rejection_reason(
                candidate,
                obstacles,
                min_obstacle_spacing=min_obstacle_spacing,
                local_density_window=local_density_window,
                max_obstacles_per_window=max_obstacles_per_window,
                max_local_area_fraction=max_local_area_fraction,
            )
            if reason:
                rejections[reason] += 1
                continue
            obstacles.append(candidate)

    if len(obstacles) < num_obstacles:
        raise RuntimeError(
            f"Only sampled {len(obstacles)}/{num_obstacles} obstacles after "
            f"{attempts} attempts with sampling_mode={sampling_mode}"
        )

    feasibility = compute_feasibility_metrics(
        obstacles,
        x_half=x_half,
        y_half=y_half,
        local_density_window=local_density_window,
        min_bottleneck_width=min_bottleneck_width,
    )
    if require_connectivity and not feasibility["connectivity"]["connected"]:
        rejections["connectivity"] += 1
        raise RuntimeError(
            "Generated map failed approximate connectivity check; increase "
            "--max-resample-attempts or relax density/spacing constraints"
        )

    stats = {
        "sampling_mode": sampling_mode,
        "attempts": int(attempts),
        "rejections": rejections,
        "constraints": {
            "min_obstacle_spacing": float(min_obstacle_spacing),
            "local_density_window": float(local_density_window),
            "max_obstacles_per_window": int(max_obstacles_per_window),
            "max_local_area_fraction": float(max_local_area_fraction),
            "require_connectivity": bool(require_connectivity),
            "min_bottleneck_width": float(min_bottleneck_width),
            "max_resample_attempts": int(max_resample_attempts),
        },
        "feasibility": feasibility,
    }

    if return_stats:
        return obstacles, stats
    return obstacles


def generate_tunnel_map(
    num_obstacles: int = 60,
    x_half: float = 12.0,
    y_half: float = 6.0,
    z_half: float = 5.0,
    resolution: float = 0.05,
    obstacle_width_range: tuple = (0.4, 1.0),
    obstacle_height_range: tuple = (4.0, 12.0),
    obstacle_zone_x_min: float = -6.0,
    cuboid_ratio: float = 0.5,
    seed: int = 42,
    sampling_mode: str = "uniform",
    min_obstacle_spacing: float = 0.0,
    local_density_window: float = 3.0,
    max_obstacles_per_window: int = 4,
    max_local_area_fraction: float = 0.45,
    require_connectivity: bool = False,
    min_bottleneck_width: float = 0.4,
    max_resample_attempts: int = 2000,
) -> np.ndarray:
    """
    Generate obstacle points as (N, 3) array.

    Coordinate convention (Gazebo/ROS):
      X: forward (drone travels +X),  range [-x_half, x_half]
      Y: lateral,                     range [-y_half, y_half]
      Z: vertical (up),               range [0, z_max]
    """
    z_max = z_half * 2
    points = []

    # Ground plane
    xs = np.arange(-x_half, x_half, resolution)
    ys = np.arange(-y_half, y_half, resolution)
    xx, yy = np.meshgrid(xs, ys)
    ground = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    points.append(ground)

    # Ceiling
    ceiling = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_max)])
    points.append(ceiling)

    # Side walls (Y = ±y_half)
    xs_wall = np.arange(-x_half, x_half, resolution)
    zs_wall = np.arange(0, z_max, resolution)
    xw, zw = np.meshgrid(xs_wall, zs_wall)
    wall_left = np.column_stack([xw.ravel(), np.full(xw.size, -y_half), zw.ravel()])
    wall_right = np.column_stack([xw.ravel(), np.full(xw.size, y_half), zw.ravel()])
    points.append(wall_left)
    points.append(wall_right)

    # Back wall at X ≈ -x_half + 2 (behind spawn zone)
    back_wall_x = -x_half + 2.0
    ys_bw = np.arange(-y_half, y_half, resolution)
    zs_bw = np.arange(0, z_max, resolution)
    yw, zw2 = np.meshgrid(ys_bw, zs_bw)
    back_wall = np.column_stack([
        np.full(yw.size, back_wall_x), yw.ravel(), zw2.ravel()
    ])
    points.append(back_wall)

    # Mixed obstacles (cylinders + cuboids)
    obstacles = generate_obstacle_params(
        num_obstacles, x_half, y_half, z_max,
        obstacle_width_range, obstacle_height_range,
        obstacle_zone_x_min, cuboid_ratio, seed,
        sampling_mode=sampling_mode,
        min_obstacle_spacing=min_obstacle_spacing,
        local_density_window=local_density_window,
        max_obstacles_per_window=max_obstacles_per_window,
        max_local_area_fraction=max_local_area_fraction,
        require_connectivity=require_connectivity,
        min_bottleneck_width=min_bottleneck_width,
        max_resample_attempts=max_resample_attempts)

    for obs in obstacles:
        if obs.shape == 'cylinder':
            radius = obs.size_x / 2.0
            n_theta = max(8, int(2 * np.pi * radius / resolution))
            n_z = max(2, int(obs.height / resolution))
            thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            zs = np.linspace(0, obs.height, n_z)
            tt, zzc = np.meshgrid(thetas, zs)
            ox = obs.cx + radius * np.cos(tt.ravel())
            oy = obs.cy + radius * np.sin(tt.ravel())
            oz = zzc.ravel()
        else:
            # Cuboid: hollow shell (surface points only, like mockamap)
            hx, hy = obs.size_x / 2.0, obs.size_y / 2.0
            n_z = max(2, int(obs.height / resolution))
            n_x = max(2, int(obs.size_x / resolution))
            n_y = max(2, int(obs.size_y / resolution))
            zs = np.linspace(0, obs.height, n_z)

            face_pts = []
            # X-faces (front/back of cuboid)
            for sign in [-1, 1]:
                y_arr = np.linspace(-hy, hy, n_y)
                yf, zf = np.meshgrid(y_arr, zs)
                xf = np.full_like(yf, obs.cx + sign * hx)
                face_pts.append(np.column_stack([xf.ravel(),
                                                  (obs.cy + yf).ravel(),
                                                  zf.ravel()]))
            # Y-faces (left/right of cuboid)
            for sign in [-1, 1]:
                x_arr = np.linspace(-hx, hx, n_x)
                xf, zf = np.meshgrid(x_arr, zs)
                yf = np.full_like(xf, obs.cy + sign * hy)
                face_pts.append(np.column_stack([(obs.cx + xf).ravel(),
                                                  yf.ravel(),
                                                  zf.ravel()]))
            combined = np.vstack(face_pts)
            ox, oy, oz = combined[:, 0], combined[:, 1], combined[:, 2]

        mask = (
            (ox >= -x_half) & (ox <= x_half)
            & (oy >= -y_half) & (oy <= y_half)
            & (oz >= 0) & (oz <= z_max)
        )
        obs_pts = np.column_stack([ox[mask], oy[mask], oz[mask]])
        if obs_pts.shape[0] > 0:
            points.append(obs_pts)

    return np.vstack(points)


def write_pcd(filepath: str, points: np.ndarray):
    """Write points to ASCII PCD file."""
    n = points.shape[0]
    header = (
        f"# .PCD v0.7 - Point Cloud Data\n"
        f"VERSION 0.7\n"
        f"FIELDS x y z\n"
        f"SIZE 4 4 4\n"
        f"TYPE F F F\n"
        f"COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA ascii\n"
    )
    with open(filepath, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def write_gazebo_world(
    filepath: str,
    obstacles: List[Obstacle],
    x_half: float = 12.0,
    y_half: float = 6.0,
    z_max: float = 10.0,
):
    """
    Write a Gazebo SDF world file with mixed obstacles matching the PCD.

    Creates:
      - Ground plane
      - Side walls at Y = ±y_half
      - Back wall at X ≈ -x_half + 2
      - Cylinder and cuboid obstacles at the same positions as the PCD
      - No ceiling (better Gazebo visibility; lidar_sim handles ceiling via PCD)
    """
    wall_thickness = 0.3

    lines = []
    lines.append("""<?xml version="1.0" ?>
<sdf version='1.7'>
  <world name='tunnel_pcd_match'>
    <!-- Lighting -->
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 15 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Physics -->
    <gravity>0 0 -9.8</gravity>
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>

    <!-- Ground plane -->
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Grey</name>
          </script></material>
        </visual>
      </link>
    </model>
""")

    wall_len_x = x_half * 2
    wall_len_y = y_half * 2
    wall_h = z_max

    # Side walls (Y = ±y_half)
    for sign, name in [(-1, 'wall_left'), (1, 'wall_right')]:
        y_pos = sign * y_half
        lines.append(f"""
    <!-- {name} at Y={y_pos:.1f} -->
    <model name='{name}'>
      <static>1</static>
      <pose>0 {y_pos:.4f} {wall_h/2:.4f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{wall_len_x:.4f} {wall_thickness} {wall_h:.4f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{wall_len_x:.4f} {wall_thickness} {wall_h:.4f}</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 0.7</ambient></material>
        </visual>
      </link>
    </model>
""")

    # Back wall at X ≈ -x_half + 2.0
    back_wall_x = -x_half + 2.0
    lines.append(f"""
    <!-- back_wall at X={back_wall_x:.1f} -->
    <model name='back_wall'>
      <static>1</static>
      <pose>{back_wall_x:.4f} 0 {wall_h/2:.4f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{wall_thickness} {wall_len_y:.4f} {wall_h:.4f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{wall_thickness} {wall_len_y:.4f} {wall_h:.4f}</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 0.7</ambient></material>
        </visual>
      </link>
    </model>
""")

    # Mixed obstacles
    n_cyl = 0
    n_box = 0
    for obs in obstacles:
        if obs.shape == 'cylinder':
            radius = obs.size_x / 2.0
            lines.append(f"""
    <model name='cyl_{n_cyl}'>
      <static>1</static>
      <pose>{obs.cx:.6f} {obs.cy:.6f} {obs.height/2:.6f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><cylinder><radius>{radius:.6f}</radius><length>{obs.height:.6f}</length></cylinder></geometry>
        </collision>
        <visual name='visual'>
          <geometry><cylinder><radius>{radius:.6f}</radius><length>{obs.height:.6f}</length></cylinder></geometry>
          <material><ambient>0.6 0.3 0.1 1</ambient></material>
        </visual>
      </link>
    </model>
""")
            n_cyl += 1
        else:
            lines.append(f"""
    <model name='box_{n_box}'>
      <static>1</static>
      <pose>{obs.cx:.6f} {obs.cy:.6f} {obs.height/2:.6f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{obs.size_x:.6f} {obs.size_y:.6f} {obs.height:.6f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{obs.size_x:.6f} {obs.size_y:.6f} {obs.height:.6f}</size></box></geometry>
          <material><ambient>0.3 0.3 0.6 1</ambient></material>
        </visual>
      </link>
    </model>
""")
            n_box += 1

    lines.append("""
  </world>
</sdf>
""")

    with open(filepath, "w") as f:
        f.write("".join(lines))

    return n_cyl, n_box


def write_metadata(
    filepath: str,
    obstacles: List[Obstacle],
    seed: int,
    num_obstacles: int,
    cuboid_ratio: float,
    resolution: float,
    x_half: float,
    y_half: float,
    z_max: float,
    obstacle_zone_x_min: float,
    sampling_stats: Optional[Dict[str, object]] = None,
):
    payload = {
        "seed": int(seed),
        "num_obstacles": int(num_obstacles),
        "cuboid_ratio": float(cuboid_ratio),
        "resolution": float(resolution),
        "terrain_size": [float(x_half * 2.0), float(y_half * 2.0)],
        "z_max": float(z_max),
        "spawn": {
            "x": float(SPAWN_X),
            "y": float(SPAWN_Y),
            "protect_radius": float(SPAWN_PROTECT_RADIUS),
        },
        "obstacle_zone_x_min": float(obstacle_zone_x_min),
        "sampling": sampling_stats or {},
        "obstacles": [
            {
                "center": [float(obs.cx), float(obs.cy)],
                "radius": float(max(obs.size_x, obs.size_y) / 2.0),
                "shape": obs.shape,
                "size": [float(obs.size_x), float(obs.size_y), float(obs.height)],
                "height": float(obs.height),
            }
            for obs in obstacles
        ],
    }
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tunnel obstacle map (PCD + optional Gazebo world)")
    parser.add_argument("--output", "-o", default="tunnel_map.pcd",
                        help="Output PCD file path")
    parser.add_argument("--world-output", "-w", default=None,
                        help="Output Gazebo .world file (optional)")
    parser.add_argument("--metadata-output", default=None,
                        help="Output obstacle metadata JSON (optional)")
    parser.add_argument("--num-obstacles", "-n", type=int, default=60,
                        help="Number of obstacles (default=60)")
    parser.add_argument("--cuboid-ratio", type=float, default=0.5,
                        help="Fraction of cuboid obstacles (0.0=all cylinders, "
                             "1.0=all cuboids, default=0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Point spacing (metres)")
    parser.add_argument("--sampling-mode", choices=("uniform", "constrained"),
                        default="uniform",
                        help="Obstacle sampling mode")
    parser.add_argument("--min-obstacle-spacing", type=float, default=0.0,
                        help="Minimum XY footprint gap for constrained sampling")
    parser.add_argument("--local-density-window", type=float, default=3.0,
                        help="Square XY window size for local density checks")
    parser.add_argument("--max-obstacles-per-window", type=int, default=4,
                        help="Max obstacle centers in any local density window")
    parser.add_argument("--max-local-area-fraction", type=float, default=0.45,
                        help="Max footprint area fraction in any local density window")
    parser.add_argument("--require-connectivity", action="store_true",
                        help="Require approximate start-to-goal 2-D connectivity")
    parser.add_argument("--min-bottleneck-width", type=float, default=0.4,
                        help="Connectivity clearance width proxy in metres")
    parser.add_argument("--max-resample-attempts", type=int, default=2000,
                        help="Max candidate samples for constrained obstacle placement")
    args = parser.parse_args()

    # Fixed dimensions matching IsaacSim training environment
    # Gazebo: X=forward(24m), Y=lateral(12m), Z=vertical(10m)
    x_half = 12.0
    y_half = 6.0
    z_half = 5.0
    z_max = z_half * 2
    obstacle_zone_x_min = -6.0  # obstacles only in X ∈ [-6, +12]

    print(f"Tunnel: X∈[-{x_half},{x_half}] (24m fwd), "
          f"Y∈[-{y_half},{y_half}] (12m lat), Z∈[0,{z_max}] (10m)")
    print(f"Obstacle zone: X∈[{obstacle_zone_x_min},{x_half}] "
          f"({x_half - obstacle_zone_x_min:.0f}m × {y_half*2:.0f}m)")
    print(f"Spawn zone: X∈[-{x_half},{obstacle_zone_x_min}] (clear)")
    print(f"Spawn protection: {SPAWN_PROTECT_RADIUS}m radius around "
          f"({SPAWN_X}, {SPAWN_Y})")
    print(f"Obstacles: {args.num_obstacles} (cuboid ratio={args.cuboid_ratio}), "
          f"seed={args.seed}")
    print(
        f"Sampling: {args.sampling_mode} "
        f"(spacing={args.min_obstacle_spacing}, window={args.local_density_window}, "
        f"max_count={args.max_obstacles_per_window})"
    )

    # Generate PCD
    pts = generate_tunnel_map(
        num_obstacles=args.num_obstacles,
        x_half=x_half, y_half=y_half, z_half=z_half,
        resolution=args.resolution,
        obstacle_zone_x_min=obstacle_zone_x_min,
        cuboid_ratio=args.cuboid_ratio,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        min_obstacle_spacing=args.min_obstacle_spacing,
        local_density_window=args.local_density_window,
        max_obstacles_per_window=args.max_obstacles_per_window,
        max_local_area_fraction=args.max_local_area_fraction,
        require_connectivity=args.require_connectivity,
        min_bottleneck_width=args.min_bottleneck_width,
        max_resample_attempts=args.max_resample_attempts,
    )
    print(f"  Total points: {pts.shape[0]:,}")
    write_pcd(args.output, pts)
    print(f"  PCD written to: {args.output}")

    # Generate Gazebo world
    if args.world_output:
        obstacles = generate_obstacle_params(
            num_obstacles=args.num_obstacles,
            x_half=x_half, y_half=y_half, z_max=z_max,
            obstacle_zone_x_min=obstacle_zone_x_min,
            cuboid_ratio=args.cuboid_ratio,
            seed=args.seed,
            sampling_mode=args.sampling_mode,
            min_obstacle_spacing=args.min_obstacle_spacing,
            local_density_window=args.local_density_window,
            max_obstacles_per_window=args.max_obstacles_per_window,
            max_local_area_fraction=args.max_local_area_fraction,
            require_connectivity=args.require_connectivity,
            min_bottleneck_width=args.min_bottleneck_width,
            max_resample_attempts=args.max_resample_attempts,
        )
        n_cyl, n_box = write_gazebo_world(
            args.world_output, obstacles,
            x_half=x_half, y_half=y_half, z_max=z_max)
        print(f"  World written to: {args.world_output} "
              f"({n_cyl} cylinders + {n_box} cuboids + 3 walls)")

    if args.metadata_output:
        obstacles, sampling_stats = generate_obstacle_params(
            num_obstacles=args.num_obstacles,
            x_half=x_half, y_half=y_half, z_max=z_max,
            obstacle_zone_x_min=obstacle_zone_x_min,
            cuboid_ratio=args.cuboid_ratio,
            seed=args.seed,
            sampling_mode=args.sampling_mode,
            min_obstacle_spacing=args.min_obstacle_spacing,
            local_density_window=args.local_density_window,
            max_obstacles_per_window=args.max_obstacles_per_window,
            max_local_area_fraction=args.max_local_area_fraction,
            require_connectivity=args.require_connectivity,
            min_bottleneck_width=args.min_bottleneck_width,
            max_resample_attempts=args.max_resample_attempts,
            return_stats=True,
        )
        write_metadata(
            args.metadata_output,
            obstacles,
            seed=args.seed,
            num_obstacles=args.num_obstacles,
            cuboid_ratio=args.cuboid_ratio,
            resolution=args.resolution,
            x_half=x_half,
            y_half=y_half,
            z_max=z_max,
            obstacle_zone_x_min=obstacle_zone_x_min,
            sampling_stats=sampling_stats,
        )
        print(f"  Metadata written to: {args.metadata_output}")


if __name__ == "__main__":
    main()
