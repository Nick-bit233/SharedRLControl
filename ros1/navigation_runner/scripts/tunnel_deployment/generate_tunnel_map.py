#!/usr/bin/env python3
"""
Generate a tunnel obstacle map as PCD + matching Gazebo world.

Matches the IsaacSim training environment (EnvTunnelResidual):
  - Half-extents: map_range = [6.0, 12.0, 5.0]  (x, y, z)
  - Corridor: X ∈ [-6, 6], Y ∈ [-12, 12], Z ∈ [0, 10]
  - Random cylindrical obstacles scattered inside
  - Ground plane and ceiling
  - Side walls

Outputs:
  - ASCII PCD file for lidar_sim_node / map_manager
  - Gazebo SDF .world file with matching obstacles (--world-output)

Usage:
    python3 generate_tunnel_map.py -o tunnel_map.pcd --world-output tunnel.world
"""
import argparse
import numpy as np
from typing import List, Tuple


def generate_obstacle_params(
    num_obstacles: int = 170,
    map_range: tuple = (6.0, 12.0, 5.0),
    obstacle_width_range: tuple = (0.4, 1.1),
    obstacle_height_range: tuple = (4.0, 10.0),
    seed: int = 42,
) -> List[Tuple[float, float, float, float]]:
    """
    Generate obstacle parameters: list of (cx, cy, radius, height).
    Deterministic given the seed.
    """
    rng = np.random.RandomState(seed)
    x_half, y_half, z_half = map_range
    z_max = z_half * 2

    obstacles = []
    for _ in range(num_obstacles):
        cx = rng.uniform(-x_half + 1.0, x_half - 1.0)
        cy = rng.uniform(-y_half + 1.0, y_half - 1.0)
        radius = rng.uniform(*obstacle_width_range) / 2.0
        height = min(rng.uniform(*obstacle_height_range), z_max)
        obstacles.append((cx, cy, radius, height))
    return obstacles


def generate_tunnel_map(
    num_obstacles: int = 170,
    map_range: tuple = (6.0, 12.0, 5.0),
    resolution: float = 0.1,
    obstacle_width_range: tuple = (0.4, 1.1),
    obstacle_height_range: tuple = (4.0, 10.0),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate obstacle points as (N, 3) array.

    Coordinate convention (matching training):
      X: forward direction (drone travels +X)
      Y: lateral
      Z: vertical (up)
    """
    points = []
    x_half, y_half, z_half = map_range
    z_max = z_half * 2

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

    # Random cylindrical obstacles (same seed-based generation)
    obstacles = generate_obstacle_params(
        num_obstacles, map_range, obstacle_width_range,
        obstacle_height_range, seed)

    for cx, cy, radius, height in obstacles:
        n_theta = max(8, int(2 * np.pi * radius / resolution))
        n_z = max(2, int(height / resolution))
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        zs = np.linspace(0, height, n_z)
        tt, zzc = np.meshgrid(thetas, zs)
        cyl_x = cx + radius * np.cos(tt.ravel())
        cyl_y = cy + radius * np.sin(tt.ravel())
        cyl_z = zzc.ravel()

        mask = (
            (cyl_x >= -x_half) & (cyl_x <= x_half)
            & (cyl_y >= -y_half) & (cyl_y <= y_half)
            & (cyl_z >= 0) & (cyl_z <= z_max)
        )
        cyl = np.column_stack([cyl_x[mask], cyl_y[mask], cyl_z[mask]])
        if cyl.shape[0] > 0:
            points.append(cyl)

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
    obstacles: List[Tuple[float, float, float, float]],
    map_range: tuple = (6.0, 12.0, 5.0),
):
    """
    Write a Gazebo SDF world file with obstacles matching the PCD.

    Creates:
      - Ground plane
      - Two side walls at Y = ±y_half
      - Cylindrical obstacles at the same positions as the PCD
      - No ceiling (better visibility in Gazebo; lidar_sim handles ceiling)
    """
    x_half, y_half, z_half = map_range
    z_max = z_half * 2
    wall_thickness = 0.1

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

    # Side walls (Y = ±y_half)
    wall_len = x_half * 2
    wall_h = z_max
    for sign, name in [(-1, 'wall_left'), (1, 'wall_right')]:
        y_pos = sign * y_half
        lines.append(f"""
    <!-- {name} at Y={y_pos:.1f} -->
    <model name='{name}'>
      <static>1</static>
      <pose>0 {y_pos:.4f} {wall_h/2:.4f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{wall_len:.4f} {wall_thickness} {wall_h:.4f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{wall_len:.4f} {wall_thickness} {wall_h:.4f}</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 0.7</ambient></material>
        </visual>
      </link>
    </model>
""")

    # Cylindrical obstacles
    for i, (cx, cy, radius, height) in enumerate(obstacles):
        lines.append(f"""
    <model name='cyl_{i}'>
      <static>1</static>
      <pose>{cx:.6f} {cy:.6f} {height/2:.6f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><cylinder><radius>{radius:.6f}</radius><length>{height:.6f}</length></cylinder></geometry>
        </collision>
        <visual name='visual'>
          <geometry><cylinder><radius>{radius:.6f}</radius><length>{height:.6f}</length></cylinder></geometry>
          <material><ambient>0.6 0.3 0.1 1</ambient></material>
        </visual>
      </link>
    </model>
""")

    lines.append("""
  </world>
</sdf>
""")

    with open(filepath, "w") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate tunnel obstacle map (PCD + optional Gazebo world)")
    parser.add_argument("--output", "-o", default="tunnel_map.pcd",
                        help="Output PCD file path")
    parser.add_argument("--world-output", "-w", default=None,
                        help="Output Gazebo .world file (optional)")
    parser.add_argument("--num_obstacles", "-n", type=int, default=170,
                        help="Number of random cylindrical obstacles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Point spacing (metres)")
    parser.add_argument("--map_range", nargs=3, type=float,
                        default=[6.0, 12.0, 5.0],
                        help="Half-extents [x, y, z]")
    args = parser.parse_args()

    mr = tuple(args.map_range)
    print(f"Generating tunnel map: {args.num_obstacles} obstacles, "
          f"range={list(mr)}, res={args.resolution}m")

    # Generate PCD
    pts = generate_tunnel_map(
        num_obstacles=args.num_obstacles,
        map_range=mr,
        resolution=args.resolution,
        seed=args.seed,
    )
    print(f"  Total points: {pts.shape[0]:,}")
    write_pcd(args.output, pts)
    print(f"  PCD written to: {args.output}")

    # Generate Gazebo world
    if args.world_output:
        obstacles = generate_obstacle_params(
            num_obstacles=args.num_obstacles,
            map_range=mr,
            seed=args.seed,
        )
        write_gazebo_world(args.world_output, obstacles, map_range=mr)
        print(f"  World written to: {args.world_output} "
              f"({len(obstacles)} cylinders + 2 walls)")


if __name__ == "__main__":
    main()
