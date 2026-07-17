#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.pcd_raycast import minimum_raycast_distance  # noqa: E402


class RaycastSafetyDistanceTest(unittest.TestCase):
    def test_uses_current_ray_hits_instead_of_full_map_nearest_point(self):
        origin = np.array([-1.8, 0.0, 0.4], dtype=np.float32)
        hits = np.array(
            [
                [-0.8, 0.0, 0.4],
                [-1.8, 2.0, 0.4],
                [-1.8, 0.0, 4.4],
            ],
            dtype=np.float32,
        )

        self.assertAlmostEqual(
            minimum_raycast_distance(hits, origin, max_range=4.0),
            1.0,
            places=6,
        )

    def test_clamps_invalid_or_over_range_hits_to_sensor_range(self):
        origin = np.zeros(3, dtype=np.float32)
        hits = np.array(
            [
                [10.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(
            minimum_raycast_distance(hits, origin, max_range=4.0),
            4.0,
        )

    def test_rejects_wrong_point_shape(self):
        with self.assertRaises(ValueError):
            minimum_raycast_distance(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                max_range=4.0,
            )


if __name__ == "__main__":
    unittest.main()
