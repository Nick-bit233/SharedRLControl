"""
Python-based raycast engine using a voxelized PCD map.

This module loads the PCD file once, builds a voxel hash set, and
performs DDA (Digital Differential Analyzer) raycasting for each
LiDAR beam.
"""

import math
import numpy as np

try:
    from .pcd_io import read_pcd_xyz
except ImportError:
    from pcd_io import read_pcd_xyz  # type: ignore

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PcdRaycaster:
    """Voxel-based raycaster built from a static PCD file.

    Parameters
    ----------
    pcd_path : str
        Path to an ASCII or binary PCD file containing x/y/z fields.
    resolution : float
        Voxel size in metres (should match occupancy map resolution).
    inflate : tuple[float, float, float]
        Robot half-size for inflation in (x, y, z) metres.
    """

    def __init__(self, pcd_path: str, resolution: float = 0.1,
                 inflate: tuple = (0.15, 0.15, 0.05)):
        self.res = resolution
        self.inv_res = 1.0 / resolution

        # Load PCD
        points = self._load_pcd(pcd_path)
        if points is None or len(points) == 0:
            raise RuntimeError(f"Failed to load PCD: {pcd_path}")
        self.points = np.asarray(points, dtype=np.float32)
        self._tree = cKDTree(self.points) if HAS_SCIPY else None

        # Build voxel set (inflated)
        inflate_cells = (
            max(1, int(math.ceil(inflate[0] / resolution))),
            max(1, int(math.ceil(inflate[1] / resolution))),
            max(1, int(math.ceil(inflate[2] / resolution))),
        )
        self._occupied = set()
        for pt in points:
            cx = int(math.floor(pt[0] * self.inv_res))
            cy = int(math.floor(pt[1] * self.inv_res))
            cz = int(math.floor(pt[2] * self.inv_res))
            for dx in range(-inflate_cells[0], inflate_cells[0] + 1):
                for dy in range(-inflate_cells[1], inflate_cells[1] + 1):
                    for dz in range(-inflate_cells[2], inflate_cells[2] + 1):
                        self._occupied.add((cx + dx, cy + dy, cz + dz))

        print(f"[PcdRaycaster] Loaded {len(points)} points, "
              f"{len(self._occupied)} occupied voxels (res={resolution}m, "
              f"inflate={inflate})")
        if self._tree is None:
            print("[PcdRaycaster] scipy unavailable; nearest-distance queries "
                  "will use vectorized NumPy fallback")

    @staticmethod
    def _load_pcd(filepath: str):
        """Load ASCII or binary PCD file -> Nx3 float array."""
        return read_pcd_xyz(filepath)

    def raycast(self, position, yaw, range_m, vfov_min_deg, vfov_max_deg,
                vbeams, hres_deg):
        """Perform raycasting matching the C++ getRayCast API.

        Parameters
        ----------
        position : (3,) array-like — [x, y, z] in world frame
        yaw : float — heading angle in radians
        range_m : float — max ray length
        vfov_min_deg, vfov_max_deg : float — vertical FOV in degrees
        vbeams : int — number of vertical beams
        hres_deg : float — horizontal angular resolution in degrees

        Returns
        -------
        hit_points : np.ndarray, shape (n_hbeams * vbeams, 3)
            Ordered as [h0v0, h0v1, ..., h0v3, h1v0, ...] matching C++.
        """
        hres = math.radians(hres_deg)
        n_hbeams = int(360.0 / hres_deg)
        vfov_min = math.radians(vfov_min_deg)
        vfov_max = math.radians(vfov_max_deg)
        if vbeams > 1:
            vres = (vfov_max - vfov_min) / (vbeams - 1)
        else:
            vres = 0.0

        sx, sy, sz = float(position[0]), float(position[1]), float(position[2])
        n_total = n_hbeams * vbeams
        result = np.empty((n_total, 3), dtype=np.float32)

        step = self.res  # march step size
        n_steps = int(math.ceil(range_m / step))

        # IsaacLab BpearlPatternCfg with ray_alignment="yaw":
        #   h = arange(-180, 180, hres) then directions are NEGATED
        #   → beam 0 (h=-180°) has direction (+1,0,0) = world +X
        #   → beam 9 (h=-90°) has direction (0,+1,0) = world +Y
        #   AND ray_alignment="yaw" means directions are NOT rotated by drone yaw
        #   → lidar beams are WORLD-FRAME-FIXED, not body-frame
        # Our cos/sin convention naturally maps:
        #   h_angle=0 → (+1,0,0)=+X, h_angle=90° → (0,+1,0)=+Y
        # So start_h=0 exactly reproduces training beam ordering.
        start_h = 0.0

        idx = 0
        for h in range(n_hbeams):
            h_angle = start_h + h * hres
            cos_h = math.cos(h_angle)
            sin_h = math.sin(h_angle)
            for v in range(vbeams):
                v_angle = vfov_min + v * vres
                # Direction: match C++ convention
                vup = math.tan(v_angle)
                dx = cos_h
                dy = sin_h
                dz = vup
                norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx /= norm
                dy /= norm
                dz /= norm

                # March along ray
                hit = False
                for i in range(1, n_steps + 1):
                    px = sx + step * dx * i
                    py = sy + step * dy * i
                    pz = sz + step * dz * i
                    vx = int(math.floor(px * self.inv_res))
                    vy = int(math.floor(py * self.inv_res))
                    vz = int(math.floor(pz * self.inv_res))
                    if (vx, vy, vz) in self._occupied:
                        result[idx] = [px, py, pz]
                        hit = True
                        break
                if not hit:
                    # No hit → return max-range point (matches C++)
                    result[idx] = [
                        sx + range_m * dx,
                        sy + range_m * dy,
                        sz + range_m * dz,
                    ]
                idx += 1

        return result

    def nearest_point(self, position):
        """Return the nearest PCD point and its Euclidean distance."""
        pos = np.asarray(position, dtype=np.float32)
        if self._tree is not None:
            dist, idx = self._tree.query(pos)
            return self.points[int(idx)].copy(), float(dist)

        diffs = self.points - pos.reshape(1, 3)
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists_sq))
        return self.points[idx].copy(), float(math.sqrt(dists_sq[idx]))

    def nearest_distance(self, position) -> float:
        """Return Euclidean distance from position to the nearest PCD point.

        This measures true map proximity. It intentionally does not use the
        sparse raycast hit points because those points can be stale between
        raycast updates and cause false collision events during fast motion.
        """
        return self.nearest_point(position)[1]
