#!/usr/bin/env python3

import importlib
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.pcd_io import write_pcd_ascii_xyz  # noqa: E402
import srlc_real_deployment.pcd_raycast as pcd_raycast  # noqa: E402
from srlc_real_deployment.pcd_raycast import PcdRaycaster  # noqa: E402


def _write_pcd(tmp_path, points):
    pcd_path = tmp_path / "synthetic.pcd"
    write_pcd_ascii_xyz(pcd_path, np.asarray(points, dtype=np.float32))
    return pcd_path


def _clearance_geometry():
    return importlib.import_module("srlc_real_deployment.clearance_geometry")


def test_raw_raycast_uses_uninflated_voxels_and_exact_entry_distance(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[2.25, 0.25, 0.25]])
    raycaster = PcdRaycaster(str(pcd_path), resolution=1.0)

    result = raycaster.raycast_raw(
        position=[0.25, 0.25, 0.25],
        yaw=np.pi / 3.0,
        range_m=3.75,
        vfov_min_deg=0.0,
        vfov_max_deg=0.0,
        vbeams=1,
        hres_deg=90.0,
    )

    np.testing.assert_array_equal(result.hit_mask, [True, False, False, False])
    np.testing.assert_allclose(
        result.entry_distances,
        [1.75, 3.75, 3.75, 3.75],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.points[0], [2.0, 0.25, 0.25], atol=1e-12)


def test_raw_raycast_preserves_world_fixed_beam_order_and_exact_misses(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[50.0, 50.0, 50.0]])
    raycaster = PcdRaycaster(str(pcd_path), resolution=0.5, inflate=(0.0, 0.0, 0.0))
    origin = np.array([0.25, -0.5, 1.0])
    requested_range = 3.7

    result = raycaster.raycast_raw(
        position=origin,
        yaw=-2.4,
        range_m=requested_range,
        vfov_min_deg=0.0,
        vfov_max_deg=0.0,
        vbeams=1,
        hres_deg=90.0,
    )

    expected_directions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    np.testing.assert_allclose(result.directions_world, expected_directions, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(result.directions_world, axis=1), 1.0)
    np.testing.assert_array_equal(result.hit_mask, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(
        result.entry_distances,
        np.full(4, requested_range, dtype=np.float64),
    )
    np.testing.assert_allclose(
        result.points,
        origin[None, :] + requested_range * expected_directions,
        atol=1e-14,
    )


def test_raw_raycast_orders_vertical_beams_inside_each_horizontal_beam(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[50.0, 50.0, 50.0]])
    raycaster = PcdRaycaster(str(pcd_path), resolution=0.5)

    result = raycaster.raycast_raw(
        position=[0.0, 0.0, 0.0],
        yaw=1.7,
        range_m=4.0,
        vfov_min_deg=-45.0,
        vfov_max_deg=45.0,
        vbeams=2,
        hres_deg=90.0,
    )

    root_half = np.sqrt(0.5)
    np.testing.assert_allclose(
        result.directions_world[:4],
        [
            [root_half, 0.0, -root_half],
            [root_half, 0.0, root_half],
            [0.0, root_half, -root_half],
            [0.0, root_half, root_half],
        ],
        atol=1e-15,
    )


def test_raw_raycast_dda_enters_diagonal_voxel_at_shared_boundary(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[1.25, 1.25, 0.25]])
    raycaster = PcdRaycaster(str(pcd_path), resolution=1.0, inflate=(0.0, 0.0, 0.0))

    result = raycaster.raycast_raw(
        position=[0.25, 0.25, 0.25],
        yaw=0.0,
        range_m=4.0,
        vfov_min_deg=0.0,
        vfov_max_deg=0.0,
        vbeams=1,
        hres_deg=45.0,
    )

    expected_entry = 0.75 / np.sqrt(0.5)
    assert result.hit_mask[1]
    np.testing.assert_allclose(result.entry_distances[1], expected_entry, atol=1e-12)
    np.testing.assert_allclose(result.points[1], [1.0, 1.0, 0.25], atol=1e-12)


def test_policy_surface_distances_apply_horizontal_buffer_to_hits_only():
    raw_distances = np.array([1.0, 4.0, 0.1, 1.0])
    directions_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
        ]
    )
    hit_mask = np.array([True, False, True, True])

    policy_distances = pcd_raycast.policy_surface_distances(
        raw_distances,
        directions_world,
        hit_mask,
        rotation_world_from_body=np.eye(3),
        half_extents=[0.20, 0.20, 0.05],
        max_range=4.0,
    )

    np.testing.assert_allclose(
        policy_distances,
        [0.8, 4.0, 0.0, 1.0 - 0.20 / np.sqrt(0.5)],
        atol=1e-12,
    )


def test_policy_surface_distances_use_full_attitude_for_vertical_extent():
    # A +90 degree body pitch maps body +Z onto world +X. A world +X beam
    # therefore exits through the 0.05 m body-Z half extent, not body-X.
    rotation_world_from_body = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )

    policy_distances = pcd_raycast.policy_surface_distances(
        raw_entry_distances=[0.5, 0.5],
        directions_world=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        hit_mask=[True, True],
        rotation_world_from_body=rotation_world_from_body,
        half_extents=[0.20, 0.20, 0.05],
        max_range=4.0,
    )

    np.testing.assert_allclose(policy_distances, [0.45, 0.30], atol=1e-12)


def test_policy_surface_distances_transform_world_beams_through_full_rpy():
    roll, pitch, yaw = 0.31, -0.47, 0.82
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotation_world_from_body = rz @ ry @ rx
    direction_body = np.array([1.0, -2.0, 3.0])
    direction_body /= np.linalg.norm(direction_body)
    direction_world = rotation_world_from_body @ direction_body
    half_extents = np.array([0.20, 0.20, 0.05])
    expected_boundary = np.min(half_extents / np.abs(direction_body))

    policy_distances = pcd_raycast.policy_surface_distances(
        raw_entry_distances=[1.0],
        directions_world=[direction_world],
        hit_mask=[True],
        rotation_world_from_body=rotation_world_from_body,
        half_extents=half_extents,
        max_range=4.0,
    )

    np.testing.assert_allclose(policy_distances, [1.0 - expected_boundary], atol=1e-12)


def test_obb_clearance_returns_face_distance_raw_point_and_escape(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[1.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=[0.0, 0.0, 0.0],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.20, 0.10, 0.05],
        clearance_cap=1.0,
    )

    assert result.surface_clearance == 0.8
    assert result.center_distance == 1.0
    np.testing.assert_array_equal(result.nearest_point, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.escape_direction, [-1.0, 0.0, 0.0])


def test_obb_clearance_uses_euclidean_corner_sdf(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[0.30, 0.40, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=[0.0, 0.0, 0.0],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.10, 0.20, 0.10],
        clearance_cap=1.0,
    )

    np.testing.assert_allclose(result.surface_clearance, np.sqrt(0.08), atol=1e-12)
    np.testing.assert_allclose(
        result.escape_direction,
        [-np.sqrt(0.5), -np.sqrt(0.5), 0.0],
        atol=1e-12,
    )


def test_obb_clearance_preserves_negative_inside_sdf(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[0.19, 0.0, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=[0.0, 0.0, 0.0],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.20, 0.20, 0.05],
        clearance_cap=1.0,
    )

    np.testing.assert_allclose(result.surface_clearance, -0.01, atol=1e-8)
    np.testing.assert_array_equal(result.escape_direction, [-1.0, 0.0, 0.0])


def test_obb_clearance_ground_point_directs_vehicle_upward(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[0.0, 0.0, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=[0.0, 0.0, 0.40],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.15, 0.15, 0.05],
        clearance_cap=1.0,
    )

    np.testing.assert_allclose(result.surface_clearance, 0.35, atol=1e-12)
    np.testing.assert_array_equal(result.escape_direction, [0.0, 0.0, 1.0])


def test_obb_clearance_uses_full_rpy_and_selects_minimum_sdf_point(tmp_path):
    roll, pitch, yaw = 0.23, -0.41, 0.67
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotation_world_from_body = rz @ ry @ rx
    position = np.array([1.0, -2.0, 0.5])
    farther_center_but_closer_surface = position + rotation_world_from_body @ [0.21, 0.0, 0.0]
    nearer_center_but_farther_surface = position + rotation_world_from_body @ [0.0, 0.0, 0.15]
    pcd_path = _write_pcd(
        tmp_path,
        [farther_center_but_closer_surface, nearer_center_but_farther_surface],
    )
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=position,
        rotation_world_from_body=rotation_world_from_body,
        half_extents=[0.15, 0.15, 0.05],
        clearance_cap=1.0,
    )

    np.testing.assert_allclose(result.surface_clearance, 0.06, atol=1e-6)
    np.testing.assert_allclose(result.center_distance, 0.21, atol=1e-6)
    np.testing.assert_allclose(
        result.nearest_point,
        farther_center_but_closer_surface,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.escape_direction,
        -rotation_world_from_body[:, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.norm(result.escape_direction), 1.0, atol=1e-12)


def test_obb_clearance_caps_empty_candidate_query_but_returns_global_raw_point(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[5.0, 0.0, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(str(pcd_path))

    result = geometry.query(
        position=[0.0, 0.0, 0.0],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.15, 0.15, 0.05],
        clearance_cap=1.0,
    )

    assert result.surface_clearance == 1.0
    assert result.center_distance == 5.0
    np.testing.assert_array_equal(result.nearest_point, [5.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.escape_direction, [-1.0, 0.0, 0.0])


def test_obb_clearance_numpy_fallback_matches_continuous_geometry(tmp_path):
    pcd_path = _write_pcd(tmp_path, [[0.40, 0.0, 0.0], [0.0, 0.30, 0.0]])
    geometry = _clearance_geometry().PcdClearanceGeometry(
        str(pcd_path),
        use_scipy=False,
    )

    result = geometry.query(
        position=[0.0, 0.0, 0.0],
        rotation_world_from_body=np.eye(3),
        half_extents=[0.20, 0.20, 0.05],
        clearance_cap=1.0,
    )

    np.testing.assert_allclose(result.surface_clearance, 0.10, atol=2e-8)
    np.testing.assert_allclose(result.center_distance, 0.30, atol=2e-8)
    np.testing.assert_allclose(result.nearest_point, [0.0, 0.30, 0.0], atol=2e-8)
    np.testing.assert_array_equal(result.escape_direction, [0.0, -1.0, 0.0])
