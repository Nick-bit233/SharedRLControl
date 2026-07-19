"""
Python-based raycast engine using a voxelized PCD map.

This module loads the PCD file once, builds a voxel hash set, and
performs DDA (Digital Differential Analyzer) raycasting for each
LiDAR beam.
"""

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RaycastResult:
    """Raw voxel-ray intersections in training beam order."""

    entry_distances: np.ndarray
    directions_world: np.ndarray
    hit_mask: np.ndarray
    points: np.ndarray

    @property
    def distances(self):
        """Compatibility alias for callers that do not need the entry qualifier."""
        return self.entry_distances

    @property
    def directions(self):
        """Compatibility alias; directions are always fixed in the world frame."""
        return self.directions_world


def box_radial_boundaries(directions_world, rotation_world_from_body,
                          half_extents):
    """Return the body-box boundary distance along world-frame unit rays.

    ``rotation_world_from_body`` is the full vehicle attitude. Directions are
    inverse-rotated into the body frame before intersecting the centered box.
    Components that are zero do not constrain the radial intersection.
    """
    directions = np.asarray(directions_world, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_body, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions_world must have shape (N, 3)")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_world_from_body must be a finite 3x3 matrix")
    if extents.shape != (3,) or not np.isfinite(extents).all():
        raise ValueError("half_extents must contain three finite values")
    if np.any(extents < 0.0):
        raise ValueError("half_extents must be non-negative")
    if not np.isfinite(directions).all():
        raise ValueError("directions_world must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=1e-7):
        raise ValueError("directions_world must contain unit directions")

    directions_body = directions @ rotation
    components = np.abs(directions_body)
    ratios = np.full(components.shape, np.inf, dtype=np.float64)
    np.divide(
        extents.reshape(1, 3),
        components,
        out=ratios,
        where=components > 1e-12,
    )
    return np.min(ratios, axis=1)


def policy_surface_distances(raw_entry_distances, directions_world, hit_mask,
                             rotation_world_from_body, half_extents,
                             max_range=4.0):
    """Convert raw voxel entries to distances from the policy vehicle box.

    Only real hits are shortened by the radial body-box boundary. Misses stay
    exactly at ``max_range`` so the observation contract is unchanged.
    """
    raw_distances = np.asarray(raw_entry_distances, dtype=np.float64)
    hits = np.asarray(hit_mask, dtype=bool)
    if raw_distances.ndim != 1:
        raise ValueError("raw_entry_distances must have shape (N,)")
    if hits.shape != raw_distances.shape:
        raise ValueError("hit_mask must match raw_entry_distances")
    if not np.isfinite(raw_distances).all() or np.any(raw_distances < 0.0):
        raise ValueError("raw_entry_distances must be finite and non-negative")
    max_range = float(max_range)
    if not math.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range must be a positive finite value")

    boundaries = box_radial_boundaries(
        directions_world,
        rotation_world_from_body,
        half_extents,
    )
    if boundaries.shape != raw_distances.shape:
        raise ValueError("directions_world must match raw_entry_distances")

    policy_distances = np.full(raw_distances.shape, max_range, dtype=np.float64)
    policy_distances[hits] = np.maximum(
        0.0,
        raw_distances[hits] - boundaries[hits],
    )
    return policy_distances


policy_clearance_distances = policy_surface_distances


def minimum_raycast_distance(hit_points, position, max_range):
    """Return the nearest finite hit from the current raycast frame.

    Unlike a nearest-neighbour query over the full PCD, this measurement only
    considers surfaces intersected by configured LiDAR rays.  That prevents a
    floor point directly below a landed vehicle from masquerading as a
    horizontal collision.
    """
    points = np.asarray(hit_points, dtype=np.float32)
    origin = np.asarray(position, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("hit_points must have shape (N, 3)")
    if origin.shape != (3,):
        raise ValueError("position must have shape (3,)")
    sensor_range = float(max_range)
    if sensor_range <= 0.0:
        raise ValueError("max_range must be positive")

    distances = np.linalg.norm(points - origin.reshape(1, 3), axis=-1)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return sensor_range
    return float(np.clip(np.min(finite), 0.0, sensor_range))


class PcdRaycaster:
    """Voxel-based raycaster built from a static PCD file.

    Parameters
    ----------
    pcd_path : str
        Path to an ASCII or binary PCD file containing x/y/z fields.
    resolution : float
        Voxel size in metres (should match occupancy map resolution).
    inflate : tuple[float, float, float]
        Optional occupancy inflation in (x, y, z) metres. Raw occupancy is the
        default; pass non-zero values only for compatibility uses.
    """

    def __init__(self, pcd_path: str, resolution: float = 0.1,
                 inflate: tuple = (0.0, 0.0, 0.0)):
        resolution = float(resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be a positive finite value")
        inflate_values = np.asarray(inflate, dtype=np.float64)
        if inflate_values.shape != (3,):
            raise ValueError("inflate must contain three values")
        if not np.isfinite(inflate_values).all() or np.any(inflate_values < 0.0):
            raise ValueError("inflate values must be finite and non-negative")

        self.res = resolution
        self.inv_res = 1.0 / resolution

        # Load PCD
        points = self._load_pcd(pcd_path)
        if points is None or len(points) == 0:
            raise RuntimeError(f"Failed to load PCD: {pcd_path}")
        self.points = np.asarray(points, dtype=np.float32)
        self._tree = cKDTree(self.points) if HAS_SCIPY else None

        # Build voxel set (inflated)
        inflate_cells = tuple(
            int(math.ceil(value / resolution)) for value in inflate_values
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
              f"inflate={tuple(float(value) for value in inflate_values)})")
        if self._tree is None:
            print("[PcdRaycaster] scipy unavailable; nearest-distance queries "
                  "will use vectorized NumPy fallback")

    @staticmethod
    def _load_pcd(filepath: str):
        """Load ASCII or binary PCD file -> Nx3 float array."""
        return read_pcd_xyz(filepath)

    def raycast_raw(self, position, yaw, range_m, vfov_min_deg, vfov_max_deg,
                    vbeams, hres_deg, direction_frame_yaw=0.0):
        """Return exact raw voxel entries and per-beam metadata.

        Parameters
        ----------
        position : (3,) array-like — [x, y, z] in world frame
        yaw : float — heading angle in radians
        range_m : float — max ray length
        vfov_min_deg, vfov_max_deg : float — vertical FOV in degrees
        vbeams : int — number of vertical beams
        hres_deg : float — horizontal angular resolution in degrees
        direction_frame_yaw : float — world-axis rotation in radians

        Beam directions are fixed in the selected world frame, so vehicle
        ``yaw`` is retained only for API compatibility. The optional direction
        frame rotation defaults to identity. Results are ordered
        ``[h0v0, h0v1, ..., h1v0, ...]`` exactly as the training input.
        """
        del yaw
        origin = np.asarray(position, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("position must contain three finite values")
        range_m = float(range_m)
        if not math.isfinite(range_m) or range_m <= 0.0:
            raise ValueError("range_m must be a positive finite value")
        vbeams = int(vbeams)
        hres_deg = float(hres_deg)
        if vbeams <= 0:
            raise ValueError("vbeams must be positive")
        if not math.isfinite(hres_deg) or hres_deg <= 0.0:
            raise ValueError("hres_deg must be a positive finite value")
        direction_frame_yaw = float(direction_frame_yaw)
        if not math.isfinite(direction_frame_yaw):
            raise ValueError("direction_frame_yaw must be finite")

        hres = math.radians(hres_deg)
        n_hbeams = int(360.0 / hres_deg)
        if n_hbeams <= 0:
            raise ValueError("hres_deg must produce at least one horizontal beam")
        vfov_min = math.radians(vfov_min_deg)
        vfov_max = math.radians(vfov_max_deg)
        if vbeams > 1:
            vres = (vfov_max - vfov_min) / (vbeams - 1)
        else:
            vres = 0.0

        n_total = n_hbeams * vbeams
        directions = np.empty((n_total, 3), dtype=np.float64)
        distances = np.full(n_total, range_m, dtype=np.float64)
        hit_mask = np.zeros(n_total, dtype=bool)

        # IsaacLab BpearlPatternCfg with ray_alignment="yaw":
        #   h = arange(-180, 180, hres) then directions are NEGATED
        #   → beam 0 (h=-180°) has direction (+1,0,0) = world +X
        #   → beam 9 (h=-90°) has direction (0,+1,0) = world +Y
        #   AND ray_alignment="yaw" means directions are NOT rotated by drone yaw
        #   → lidar beams are WORLD-FRAME-FIXED, not body-frame
        # Our cos/sin convention naturally maps:
        #   h_angle=0 → (+1,0,0)=+X, h_angle=90° → (0,+1,0)=+Y
        # So start_h=0 exactly reproduces training beam ordering. A static
        # world-frame transform may rotate that complete arrangement without
        # introducing any vehicle-yaw coupling.
        start_h = direction_frame_yaw

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
                direction = np.array([dx, dy, dz], dtype=np.float64)
                direction[np.abs(direction) < 1e-12] = 0.0
                direction /= np.linalg.norm(direction)
                directions[idx] = direction
                entry_distance = self._dda_entry_distance(origin, direction, range_m)
                if entry_distance is not None:
                    distances[idx] = entry_distance
                    hit_mask[idx] = True
                idx += 1

        points = origin.reshape(1, 3) + distances.reshape(-1, 1) * directions
        return RaycastResult(
            entry_distances=distances,
            directions_world=directions,
            hit_mask=hit_mask,
            points=points,
        )

    def _dda_entry_distance(self, origin, direction, max_range):
        """Return the exact entry parameter of the first occupied voxel."""
        voxel = np.floor(origin * self.inv_res).astype(np.int64)
        if tuple(voxel) in self._occupied:
            return 0.0

        step = np.sign(direction).astype(np.int64)
        t_max = np.full(3, np.inf, dtype=np.float64)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        for axis in range(3):
            component = direction[axis]
            if component > 0.0:
                boundary = (voxel[axis] + 1) * self.res
            elif component < 0.0:
                boundary = voxel[axis] * self.res
            else:
                continue
            t_max[axis] = (boundary - origin[axis]) / component
            t_delta[axis] = self.res / abs(component)

        while True:
            entry_distance = float(np.min(t_max))
            if not math.isfinite(entry_distance) or entry_distance > max_range:
                return None

            tolerance = 1e-12 * max(1.0, abs(entry_distance))
            crossed_axes = np.abs(t_max - entry_distance) <= tolerance
            voxel[crossed_axes] += step[crossed_axes]
            t_max[crossed_axes] += t_delta[crossed_axes]
            if tuple(voxel) in self._occupied:
                return max(0.0, entry_distance)

    def raycast(self, position, yaw, range_m, vfov_min_deg, vfov_max_deg,
                vbeams, hres_deg):
        """Return hit/max-range points using the legacy array-only API."""
        return self.raycast_raw(
            position,
            yaw,
            range_m,
            vfov_min_deg,
            vfov_max_deg,
            vbeams,
            hres_deg,
        ).points

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
