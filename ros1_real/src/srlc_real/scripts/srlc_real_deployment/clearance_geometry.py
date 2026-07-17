"""Continuous raw-point clearance for a fully oriented vehicle box."""

from dataclasses import dataclass
import math

import numpy as np

try:
    from .pcd_io import read_pcd_xyz
except ImportError:
    from pcd_io import read_pcd_xyz  # type: ignore

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised by forced fallback tests
    cKDTree = None


@dataclass(frozen=True)
class ClearanceResult:
    """Minimum raw-point clearance and its associated geometry."""

    surface_clearance: float
    center_distance: float
    nearest_point: np.ndarray
    escape_direction: np.ndarray


def oriented_box_sdf(points, position, rotation_world_from_body, half_extents):
    """Evaluate the signed distance from raw points to an oriented box."""
    points = np.asarray(points, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_body, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must contain three finite values")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_world_from_body must be a finite 3x3 matrix")
    if extents.shape != (3,) or not np.isfinite(extents).all():
        raise ValueError("half_extents must contain three finite values")
    if np.any(extents < 0.0):
        raise ValueError("half_extents must be non-negative")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")

    points_body = (points - position.reshape(1, 3)) @ rotation
    q = np.abs(points_body) - extents.reshape(1, 3)
    return (
        np.linalg.norm(np.maximum(q, 0.0), axis=1)
        + np.minimum(np.max(q, axis=1), 0.0)
    )


class PcdClearanceGeometry:
    """KD-tree accelerated OBB clearance queries over a raw PCD map."""

    def __init__(self, pcd_path, use_scipy=True):
        points = read_pcd_xyz(pcd_path)
        if points is None or len(points) == 0:
            raise RuntimeError(f"Failed to load PCD: {pcd_path}")
        self.points = np.asarray(points, dtype=np.float64)
        self._tree = (
            cKDTree(self.points)
            if bool(use_scipy) and cKDTree is not None
            else None
        )

    def query(self, position, rotation_world_from_body, half_extents,
              clearance_cap=1.0):
        """Return minimum continuous clearance for one vehicle pose.

        Candidate points are bounded by ``norm(half_extents) + clearance_cap``.
        If that radius is empty, clearance is known to be at least the cap; the
        globally nearest raw point is still returned for observability.
        """
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation_world_from_body, dtype=np.float64)
        extents = np.asarray(half_extents, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position must contain three finite values")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation_world_from_body must be a finite 3x3 matrix")
        if extents.shape != (3,) or not np.isfinite(extents).all():
            raise ValueError("half_extents must contain three finite values")
        if np.any(extents < 0.0):
            raise ValueError("half_extents must be non-negative")
        clearance_cap = float(clearance_cap)
        if not math.isfinite(clearance_cap) or clearance_cap <= 0.0:
            raise ValueError("clearance_cap must be a positive finite value")

        candidate_radius = float(np.linalg.norm(extents) + clearance_cap)
        candidate_indices = self._candidate_indices(position, candidate_radius)
        if candidate_indices.size == 0:
            selected_index = self._nearest_center_index(position)
            surface_clearance = clearance_cap
        else:
            candidates = self.points[candidate_indices]
            signed_distances = oriented_box_sdf(
                candidates,
                position,
                rotation,
                extents,
            )
            selected_local = int(np.argmin(signed_distances))
            selected_index = int(candidate_indices[selected_local])
            surface_clearance = min(
                float(signed_distances[selected_local]),
                clearance_cap,
            )

        nearest_point = self.points[selected_index].copy()
        relative_world = nearest_point - position
        center_distance = float(np.linalg.norm(relative_world))
        escape_direction = self._escape_direction(
            relative_world,
            rotation,
            extents,
        )
        return ClearanceResult(
            surface_clearance=surface_clearance,
            center_distance=center_distance,
            nearest_point=nearest_point,
            escape_direction=escape_direction,
        )

    def _candidate_indices(self, position, radius):
        if self._tree is not None:
            indices = self._tree.query_ball_point(position, radius)
            return np.asarray(sorted(indices), dtype=np.int64)

        relative = self.points - position.reshape(1, 3)
        distances_sq = np.einsum("ij,ij->i", relative, relative)
        return np.flatnonzero(distances_sq <= radius * radius)

    def _nearest_center_index(self, position):
        if self._tree is not None:
            _, index = self._tree.query(position)
            return int(index)

        relative = self.points - position.reshape(1, 3)
        distances_sq = np.einsum("ij,ij->i", relative, relative)
        return int(np.argmin(distances_sq))

    @staticmethod
    def _escape_direction(relative_world, rotation, extents):
        relative_body = rotation.T @ relative_world
        q = np.abs(relative_body) - extents
        outside_body = np.sign(relative_body) * np.maximum(q, 0.0)
        outside_norm = float(np.linalg.norm(outside_body))

        if outside_norm > 1e-12:
            escape_body = -outside_body / outside_norm
        else:
            closest_face_axis = int(np.argmax(q))
            face_sign = float(np.sign(relative_body[closest_face_axis]))
            if face_sign == 0.0:
                face_sign = 1.0
            escape_body = np.zeros(3, dtype=np.float64)
            escape_body[closest_face_axis] = -face_sign

        escape_world = rotation @ escape_body
        norm = float(np.linalg.norm(escape_world))
        if norm <= 0.0 or not math.isfinite(norm):
            raise ValueError("rotation_world_from_body produced an invalid direction")
        return escape_world / norm


ClearanceGeometry = PcdClearanceGeometry
PcdClearance = PcdClearanceGeometry
