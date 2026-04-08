"""
Simplified 3D occupancy grid built from Isaac Sim terrain ground truth.

Replaces the probabilistic ROG-Map used in the original C++ slope_inspection
project. Since we operate in simulation with known obstacle geometry (from
HfDiscreteObstaclesTerrainCfg), we bypass LiDAR-based probabilistic updates
and directly mark cells as occupied from ground-truth positions, radii, and
heights.

Grid convention (matches ROG-Map):
  - The grid origin sits at the CENTER of the map volume.
  - Grid index [0, 0, 0] corresponds to world position (origin - map_size / 2).
  - Cell centres are at  origin - map_size/2 + (idx + 0.5) * resolution.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


@dataclass
class OccupancyGridConfig:
    resolution: float = 0.1
    inflation_steps: int = 3  # inflation radius in grid cells (0.3 m at 0.1 res)
    map_size: Tuple[float, float, float] = (40.0, 40.0, 15.0)
    map_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class OccupancyGrid:
    """Ground-truth 3D occupancy grid for Isaac Sim environments."""

    def __init__(self, cfg: OccupancyGridConfig = None):
        """Allocate an empty grid according to *cfg*."""
        if cfg is None:
            cfg = OccupancyGridConfig()
        self.cfg = cfg

        self.resolution = cfg.resolution
        self.inv_resolution = 1.0 / cfg.resolution

        self.map_size = np.asarray(cfg.map_size, dtype=np.float64)
        self.origin = np.asarray(cfg.map_origin, dtype=np.float64)

        # World-frame corner that maps to index [0, 0, 0].
        self.grid_origin = self.origin - self.map_size / 2.0

        self.grid_shape = np.ceil(self.map_size * self.inv_resolution).astype(int)

        self.raw_grid: np.ndarray = np.zeros(self.grid_shape, dtype=bool)
        self.inflated_grid: np.ndarray = np.zeros(self.grid_shape, dtype=bool)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def build_from_obstacles(self, obstacles: list) -> None:
        """Build occupancy from a list of cylindrical obstacles.

        Args:
            obstacles: list of dicts with keys
                ``'center'`` – (x, y) world position,
                ``'radius'`` – cylinder radius [m],
                ``'height'`` – cylinder height [m],
                ``'z_base'``  – z of the cylinder base (default 0).
        """
        self.raw_grid[:] = False

        for obs in obstacles:
            cx, cy = obs["center"]
            radius = obs["radius"]
            height = obs["height"]
            z_base = obs.get("z_base", 0.0)

            # Axis-aligned bounding box in world frame.
            bb_min = np.array([cx - radius, cy - radius, z_base])
            bb_max = np.array([cx + radius, cy + radius, z_base + height])

            idx_min = np.maximum(self._world_to_grid_floor(bb_min), 0)
            idx_max = np.minimum(
                self._world_to_grid_floor(bb_max) + 1, self.grid_shape
            )

            xs = np.arange(idx_min[0], idx_max[0])
            ys = np.arange(idx_min[1], idx_max[1])
            zs = np.arange(idx_min[2], idx_max[2])
            if xs.size == 0 or ys.size == 0 or zs.size == 0:
                continue

            # Cell centres in world frame.
            wx = self.grid_origin[0] + (xs + 0.5) * self.resolution
            wy = self.grid_origin[1] + (ys + 0.5) * self.resolution

            dx = wx - cx
            dy = wy - cy
            # Outer-product distance² for the XY disc test.
            dist2 = dx[:, None] ** 2 + dy[None, :] ** 2
            inside_xy = dist2 <= radius * radius  # (len_x, len_y)

            # Mark all z slices that fall inside the cylinder.
            inside_3d = np.repeat(
                inside_xy[:, :, np.newaxis], len(zs), axis=2
            )
            self.raw_grid[
                idx_min[0] : idx_max[0],
                idx_min[1] : idx_max[1],
                idx_min[2] : idx_max[2],
            ] |= inside_3d

        self.inflate()

    def build_from_heightfield(
        self,
        heightfield: np.ndarray,
        horizontal_scale: float,
        vertical_scale: float,
        origin: np.ndarray,
    ) -> None:
        """Build occupancy from a 2D heightfield array (vectorized).

        Any grid cell whose centre is **below** the terrain height at the
        corresponding (x, y) is marked occupied (solid ground / obstacle).

        The heightfield convention follows Isaac Lab:
        ``heightfield[i, j]`` where axis 0 = x and axis 1 = y.

        Args:
            heightfield: (W, L) height samples — axis 0 is x, axis 1 is y.
            horizontal_scale: metres per heightfield cell.
            vertical_scale: metres per height unit.
            origin: (3,) world position of the heightfield corner (min-x, min-y, z).
        """
        self.raw_grid[:] = False
        origin = np.asarray(origin, dtype=np.float64)

        hf_w, hf_l = heightfield.shape  # (x_size, y_size)

        # World-frame cell centres along each axis
        wx = self.grid_origin[0] + (np.arange(self.grid_shape[0]) + 0.5) * self.resolution
        wy = self.grid_origin[1] + (np.arange(self.grid_shape[1]) + 0.5) * self.resolution
        wz = self.grid_origin[2] + (np.arange(self.grid_shape[2]) + 0.5) * self.resolution

        # Map grid positions to heightfield indices (axis 0 = x, axis 1 = y)
        hf_xi = ((wx - origin[0]) / horizontal_scale).astype(int)
        hf_yj = ((wy - origin[1]) / horizontal_scale).astype(int)

        # Mask for valid index ranges
        valid_x = (hf_xi >= 0) & (hf_xi < hf_w)
        valid_y = (hf_yj >= 0) & (hf_yj < hf_l)

        # Clip to valid (for indexing; invalid cells stay at height 0)
        hf_xi_c = np.clip(hf_xi, 0, hf_w - 1)
        hf_yj_c = np.clip(hf_yj, 0, hf_l - 1)

        # 2D terrain height map: terrain_h[ix, iy]
        terrain_h = heightfield[hf_xi_c[:, None], hf_yj_c[None, :]] * vertical_scale + origin[2]
        # Zero out invalid columns
        terrain_h[~valid_x, :] = 0.0
        terrain_h[:, ~valid_y] = 0.0

        # Mark occupied: grid cell below terrain surface
        self.raw_grid = (wz[None, None, :] < terrain_h[:, :, None])

        self.inflate()

    def build_from_point_cloud(self, points: np.ndarray) -> None:
        """Build occupancy from a (N, 3) array of world-frame points."""
        self.raw_grid[:] = False
        if points.size == 0:
            self.inflate()
            return

        indices = self._world_to_grid_floor(points)

        # Discard out-of-bound points.
        valid = np.all(indices >= 0, axis=-1) & np.all(
            indices < self.grid_shape, axis=-1
        )
        indices = indices[valid]

        self.raw_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        self.inflate()

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------

    def inflate(self) -> None:
        """Dilate the raw grid by ``inflation_steps`` cells (ball element)."""
        steps = self.cfg.inflation_steps
        if steps <= 0:
            np.copyto(self.inflated_grid, self.raw_grid)
            return

        struct = _ball_structuring_element(steps)
        binary_dilation(
            self.raw_grid, structure=struct, output=self.inflated_grid
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_occupied(self, pos: np.ndarray) -> bool:
        """Return True if *pos* (world frame) lies in an occupied raw cell."""
        idx = self.world_to_grid(pos)
        if not self.is_valid_index(idx):
            return False
        return bool(self.raw_grid[idx[0], idx[1], idx[2]])

    def is_free_inflate(self, pos: np.ndarray) -> bool:
        """Return True if *pos* is free in the **inflated** grid."""
        idx = self.world_to_grid(pos)
        if not self.is_valid_index(idx):
            return False
        return not bool(self.inflated_grid[idx[0], idx[1], idx[2]])

    def is_valid_index(self, idx: np.ndarray) -> bool:
        """Return True if *idx* is inside the grid bounds."""
        idx = np.asarray(idx)
        return bool(np.all(idx >= 0) and np.all(idx < self.grid_shape))

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def world_to_grid(self, pos: np.ndarray) -> np.ndarray:
        """Convert world position(s) to integer grid indices."""
        pos = np.asarray(pos, dtype=np.float64)
        return np.floor((pos - self.grid_origin) * self.inv_resolution).astype(int)

    def grid_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Convert grid index/indices to world-frame cell centre(s)."""
        idx = np.asarray(idx, dtype=np.float64)
        return self.grid_origin + (idx + 0.5) * self.resolution

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def box_search(
        self,
        box_min: np.ndarray,
        box_max: np.ndarray,
        max_points: int = 500,
        downsample_resolution: float = 0.0,
    ) -> np.ndarray:
        """Return (N, 3) world-frame centres of occupied cells inside the AABB.

        Uses the **raw** (non-inflated) grid so that returned points represent
        actual obstacle surfaces — matching the interface expected by CIRI.

        Args:
            box_min: (3,) lower corner of query box (world frame).
            box_max: (3,) upper corner of query box (world frame).
            max_points: Maximum number of points to return.  When the raw
                result exceeds this, points are subsampled via a coarser
                voxel grid or uniformly at random.
            downsample_resolution: If > 0, subsample by snapping points to a
                coarser voxel grid of this resolution and keeping one per
                voxel.  Set to 0 (default) to use random subsampling.
        """
        idx_min = np.maximum(self._world_to_grid_floor(box_min), 0)
        idx_max = np.minimum(
            self._world_to_grid_floor(box_max) + 1, self.grid_shape
        )

        sub = self.raw_grid[
            idx_min[0] : idx_max[0],
            idx_min[1] : idx_max[1],
            idx_min[2] : idx_max[2],
        ]

        local_indices = np.argwhere(sub)  # (M, 3) relative to idx_min
        if local_indices.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        abs_indices = local_indices + idx_min
        points = self.grid_to_world(abs_indices)

        if len(points) <= max_points:
            return points

        # Downsample
        if downsample_resolution > 0:
            voxel_keys = np.floor(points / downsample_resolution).astype(np.int32)
            _, unique_idx = np.unique(
                voxel_keys, axis=0, return_index=True
            )
            points = points[unique_idx]
            if len(points) <= max_points:
                return points

        # Random subsampling as last resort
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), max_points, replace=False)
        return points[idx]

    def is_line_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check collision along a straight line using the inflated grid.

        Steps at half-resolution increments to avoid skipping cells.
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)

        diff = end - start
        length = np.linalg.norm(diff)
        if length < 1e-9:
            return self.is_free_inflate(start)

        step = self.resolution * 0.5
        n_steps = int(np.ceil(length / step))
        for i in range(n_steps + 1):
            t = i / max(n_steps, 1)
            pt = start + t * diff
            idx = self.world_to_grid(pt)
            if not self.is_valid_index(idx):
                return False
            if self.inflated_grid[idx[0], idx[1], idx[2]]:
                return False
        return True

    # ------------------------------------------------------------------
    # Export / visualisation helpers
    # ------------------------------------------------------------------

    def get_2d_slice(self, z: float, use_inflated: bool = False) -> dict:
        """Return a 2D occupancy slice at altitude *z* for visualisation.

        Returns a dict with:
            ``occupied_xy`` — (N, 2) world-frame cell centres that are occupied,
            ``resolution``  — cell size in metres,
            ``grid_origin``  — (2,) world XY of index [0, 0],
            ``grid_shape``   — (nx, ny) of the slice.
        """
        grid = self.inflated_grid if use_inflated else self.raw_grid
        z_idx = int(np.floor((z - self.grid_origin[2]) * self.inv_resolution))
        if z_idx < 0 or z_idx >= self.grid_shape[2]:
            return {
                "occupied_xy": np.empty((0, 2)),
                "resolution": self.resolution,
                "grid_origin": self.grid_origin[:2].copy(),
                "grid_shape": (self.grid_shape[0], self.grid_shape[1]),
            }
        slice_2d = grid[:, :, z_idx]  # (nx, ny) bool
        occ_ij = np.argwhere(slice_2d)  # (M, 2)
        if occ_ij.size == 0:
            occupied_xy = np.empty((0, 2))
        else:
            occupied_xy = self.grid_origin[:2] + (occ_ij + 0.5) * self.resolution
        return {
            "occupied_xy": occupied_xy,
            "resolution": self.resolution,
            "grid_origin": self.grid_origin[:2].copy(),
            "grid_shape": (self.grid_shape[0], self.grid_shape[1]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid_floor(self, pos: np.ndarray) -> np.ndarray:
        """Floor-based world→grid conversion (no rounding to int dtype)."""
        pos = np.asarray(pos, dtype=np.float64)
        return np.floor((pos - self.grid_origin) * self.inv_resolution).astype(int)


# ----------------------------------------------------------------------
# Module-level utilities
# ----------------------------------------------------------------------


def _ball_structuring_element(radius: int) -> np.ndarray:
    """Create a boolean ball structuring element of the given *radius*.

    The element has shape ``(2*radius+1,)³`` and a cell is True when its
    Euclidean distance from the centre is ≤ *radius*.
    """
    d = 2 * radius + 1
    ax = np.arange(d) - radius
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    return (xx * xx + yy * yy + zz * zz) <= radius * radius
