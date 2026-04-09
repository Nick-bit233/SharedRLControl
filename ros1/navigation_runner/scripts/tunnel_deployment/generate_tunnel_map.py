#!/usr/bin/env python3
"""
Generate a tunnel obstacle map as a PCD file for Gazebo / map_manager.

Matches the IsaacSim training environment (EnvTunnelResidual):
  - Half-extents: map_range = [6.0, 12.0, 5.0]  (x, y, z)
  - Corridor: X ∈ [-6, 6], Y ∈ [-12, 12], Z ∈ [0, 10]
  - Random cylindrical obstacles scattered inside
  - Ground plane and ceiling
  - Platform at the entrance side (width=4.0)

Output: ASCII PCD file loadable by map_manager's occupancy_map_node.

Usage:
    python3 generate_tunnel_map.py --output tunnel_map.pcd --num_obstacles 170
"""
import argparse
import numpy as np


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
    rng = np.random.RandomState(seed)
    points = []

    x_half, y_half, z_half = map_range
    z_max = z_half * 2  # Full height = 10.0

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

    # Random cylindrical obstacles
    for _ in range(num_obstacles):
        # Random position (avoid a clear channel near entrance)
        cx = rng.uniform(-x_half + 1.0, x_half - 1.0)
        cy = rng.uniform(-y_half + 1.0, y_half - 1.0)

        # Random radius and height
        radius = rng.uniform(*obstacle_width_range) / 2.0
        height = rng.uniform(*obstacle_height_range)
        height = min(height, z_max)

        # Generate cylinder surface points
        n_theta = max(8, int(2 * np.pi * radius / resolution))
        n_z = max(2, int(height / resolution))
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        zs = np.linspace(0, height, n_z)
        tt, zzc = np.meshgrid(thetas, zs)
        cyl_x = cx + radius * np.cos(tt.ravel())
        cyl_y = cy + radius * np.sin(tt.ravel())
        cyl_z = zzc.ravel()

        # Keep only points inside map bounds
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


def main():
    parser = argparse.ArgumentParser(description="Generate tunnel obstacle map (PCD)")
    parser.add_argument("--output", "-o", default="tunnel_map.pcd", help="Output PCD file path")
    parser.add_argument("--num_obstacles", "-n", type=int, default=170,
                        help="Number of random cylindrical obstacles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Point spacing (metres)")
    parser.add_argument("--map_range", nargs=3, type=float, default=[6.0, 12.0, 5.0],
                        help="Half-extents [x, y, z]")
    args = parser.parse_args()

    print(f"Generating tunnel map: {args.num_obstacles} obstacles, "
          f"range={args.map_range}, res={args.resolution}m")

    pts = generate_tunnel_map(
        num_obstacles=args.num_obstacles,
        map_range=tuple(args.map_range),
        resolution=args.resolution,
        seed=args.seed,
    )
    print(f"  Total points: {pts.shape[0]:,}")

    write_pcd(args.output, pts)
    print(f"  Written to: {args.output}")


if __name__ == "__main__":
    main()
