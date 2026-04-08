"""
IPC Controller — integrates MPC, A*, CIRI, and OccupancyGrid.

Mirrors the control pipeline from slope_inspection/IPC/src/ipc_fsm.cpp,
adapted for the SharedRLControl Isaac Sim environment.

Pipeline:
    1. User body-frame velocity → world-frame → integrate to short-term goal
    2. A* path from current position to goal (periodic replanning)
    3. CIRI safe flight corridor along MPC horizon
    4. MPC trajectory optimization with SFC constraints
    5. Extract v_opt (first-step velocity) as output
"""

import numpy as np
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional

from .mpc_solver import MPCSolver, MPCConfig
from .astar_planner import AStarPlanner, AStarConfig
from .ciri_corridor import CIRICorridor, CIRIConfig
from .occupancy_grid import OccupancyGrid, OccupancyGridConfig


@dataclass
class IPCControllerConfig:
    lookahead_time: float = 2.0
    replan_interval: int = 20
    use_sfc: bool = True
    fallback_to_simple: bool = True
    altitude_hold: float = 4.0
    max_speed: float = 2.0
    obstacle_query_radius: float = 6.0
    max_sfc_steps: int = 5
    max_obstacle_points: int = 200

    mpc: MPCConfig = field(default_factory=MPCConfig)
    astar: AStarConfig = field(default_factory=AStarConfig)
    ciri: CIRIConfig = field(default_factory=CIRIConfig)
    occupancy_grid: OccupancyGridConfig = field(default_factory=OccupancyGridConfig)


def load_config(yaml_path: str) -> IPCControllerConfig:
    """Load IPCControllerConfig from a YAML file."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = IPCControllerConfig()

    # Controller-level params
    ctrl = raw.get('controller', {})
    for key in ['lookahead_time', 'replan_interval', 'use_sfc',
                'fallback_to_simple', 'altitude_hold', 'max_speed',
                'obstacle_query_radius', 'max_sfc_steps', 'max_obstacle_points']:
        if key in ctrl:
            setattr(cfg, key, ctrl[key])

    # MPC
    mpc_raw = raw.get('mpc', {})
    mpc_cfg = MPCConfig()
    for key in ['horizon', 'step', 'R_p', 'R_v', 'R_a', 'R_u', 'R_u_con',
                'R_pN', 'R_vN', 'R_aN', 'D_x', 'D_y', 'D_z']:
        if key in mpc_raw:
            setattr(mpc_cfg, key, mpc_raw[key])
    for key in ['v_min', 'v_max', 'a_min', 'a_max', 'u_min', 'u_max']:
        if key in mpc_raw:
            setattr(mpc_cfg, key, np.array(mpc_raw[key], dtype=float))
    cfg.mpc = mpc_cfg

    # A*
    astar_raw = raw.get('astar', {})
    astar_cfg = AStarConfig()
    for key in ['resolution', 'timeout', 'tie_breaker']:
        if key in astar_raw:
            setattr(astar_cfg, key, astar_raw[key])
    cfg.astar = astar_cfg

    # CIRI
    ciri_raw = raw.get('ciri', {})
    ciri_cfg = CIRIConfig()
    for key in ['robot_radius', 'iterations', 'epsilon']:
        if key in ciri_raw:
            setattr(ciri_cfg, key, ciri_raw[key])
    cfg.ciri = ciri_cfg

    # Occupancy grid
    occ_raw = raw.get('occupancy_grid', {})
    occ_cfg = OccupancyGridConfig()
    for key in ['resolution', 'inflation_steps']:
        if key in occ_raw:
            setattr(occ_cfg, key, occ_raw[key])
    if 'map_size' in occ_raw:
        occ_cfg.map_size = tuple(occ_raw['map_size'])
    if 'map_origin' in occ_raw:
        occ_cfg.map_origin = tuple(occ_raw['map_origin'])
    cfg.occupancy_grid = occ_cfg

    return cfg


class IPCController:
    """
    Integrated Predictive Collision avoidance controller.

    Usage:
        ctrl = IPCController(config_path="ipc_config.yaml")
        ctrl.build_map(obstacles)   # one-time setup
        vel_w = ctrl.step(pos, vel, acc, user_vel_body, quat)
    """

    def __init__(self, cfg: Optional[IPCControllerConfig] = None,
                 config_path: Optional[str] = None):
        if cfg is None and config_path is not None:
            cfg = load_config(config_path)
        elif cfg is None:
            cfg = IPCControllerConfig()

        self.cfg = cfg
        self.mpc = MPCSolver(cfg.mpc)
        self.astar = AStarPlanner(cfg.astar)
        self.ciri = CIRICorridor(cfg.ciri)
        self.occ_grid = OccupancyGrid(cfg.occupancy_grid)

        # Internal state
        self._step_count = 0
        self._ref_path: Optional[np.ndarray] = None
        self._astar_index = 0
        self._last_goal = np.zeros(3)
        self._map_built = False

        # Visualization data (updated each step)
        self._last_sfc: dict = {}  # step_index → (K, 4) planes

        # CIRI success rate tracking
        self._sfc_attempts = 0
        self._sfc_successes = 0

    def reset(self):
        """Reset navigation state between trials (keeps map and config)."""
        self._step_count = 0
        self._ref_path = None
        self._astar_index = 0
        self._last_goal = np.zeros(3)
        self._last_sfc.clear()
        self._sfc_attempts = 0
        self._sfc_successes = 0

    def build_map(self, obstacles: list = None, heightfield=None,
                  horizontal_scale=None, vertical_scale=None, origin=None,
                  point_cloud=None):
        """Build the occupancy grid from scene geometry (call once)."""
        if obstacles is not None:
            self.occ_grid.build_from_obstacles(obstacles)
        elif heightfield is not None:
            self.occ_grid.build_from_heightfield(
                heightfield, horizontal_scale, vertical_scale, origin)
        elif point_cloud is not None:
            self.occ_grid.build_from_point_cloud(point_cloud)
        else:
            raise ValueError("Must provide obstacles, heightfield, or point_cloud")
        self._map_built = True

    def step(self, pos_w: np.ndarray, vel_w: np.ndarray, acc_w: np.ndarray,
             user_vel_body: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Run one IPC control step.

        Args:
            pos_w: (3,) current world position
            vel_w: (3,) current world velocity
            acc_w: (3,) estimated world acceleration
            user_vel_body: (3,) user velocity command in body frame
            rotation_matrix: (3,3) body-to-world rotation matrix

        Returns:
            (3,) target velocity in world frame
        """
        pos_w = np.asarray(pos_w, dtype=float).flatten()[:3]
        vel_w = np.asarray(vel_w, dtype=float).flatten()[:3]
        acc_w = np.asarray(acc_w, dtype=float).flatten()[:3]
        user_vel_body = np.asarray(user_vel_body, dtype=float).flatten()[:3]
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)

        # If critical state is non-finite (e.g. early sim frame), hold position
        if not (np.all(np.isfinite(pos_w)) and np.all(np.isfinite(rotation_matrix))):
            return np.zeros(3)

        # Convert user velocity to world frame.
        # Only use the yaw component of the rotation to avoid pitch/roll
        # coupling after collisions — extract yaw from rotation matrix and
        # build a pure yaw rotation so that body-x always maps to a
        # horizontal world direction.
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])
        user_vel_w = R_yaw @ user_vel_body

        # Clamp user velocity
        speed = np.linalg.norm(user_vel_w)
        if not np.isfinite(speed) or speed > self.cfg.max_speed:
            if np.isfinite(speed) and speed > 1e-9:
                user_vel_w = user_vel_w * (self.cfg.max_speed / speed)
            else:
                user_vel_w = np.zeros(3)

        # Compute goal: current position + velocity * lookahead_time
        goal = pos_w + user_vel_w * self.cfg.lookahead_time

        # Enforce altitude hold for z component
        goal[2] = self.cfg.altitude_hold

        # Clamp goal within map bounds to prevent runaway
        half = np.array(self.cfg.occupancy_grid.map_size) / 2.0
        origin = np.array(self.cfg.occupancy_grid.map_origin)
        map_lo = origin - half
        map_hi = origin + half
        goal = np.clip(goal, map_lo, map_hi)

        self._step_count += 1

        # --- A* global path planning (periodic) ---
        need_replan = (
            self._ref_path is None
            or self._step_count % self.cfg.replan_interval == 0
            or np.linalg.norm(goal - self._last_goal) > 1.0
        )

        if self._map_built and need_replan:
            path = self.astar.search(pos_w, goal, self.occ_grid)
            if path is not None:
                path = self.astar.simplify_path(path, self.occ_grid)
                self._ref_path = path
                self._astar_index = 0
            self._last_goal = goal.copy()

        # --- Set MPC goals ---
        self.mpc.clear_sfc()
        self._last_sfc.clear()
        N = self.cfg.mpc.horizon
        dt_mpc = self.cfg.mpc.step

        if self._ref_path is not None and len(self._ref_path) > 1:
            # Find nearest point on path
            dists = np.linalg.norm(self._ref_path - pos_w, axis=1)
            self._astar_index = int(np.argmin(dists))

            # Estimate path resolution
            path_segments = np.diff(self._ref_path, axis=0)
            avg_seg_len = np.mean(np.linalg.norm(path_segments, axis=1))
            if avg_seg_len < 1e-6:
                avg_seg_len = self.cfg.astar.resolution

            last_p_ref = pos_w.copy()
            for i in range(N):
                # Advance along path proportional to reference velocity
                vel_ref = np.linalg.norm(user_vel_w) + 0.5
                step_ratio = vel_ref * dt_mpc / avg_seg_len
                index_advance = int(step_ratio) if np.isfinite(step_ratio) else 1
                idx = min(self._astar_index + i * max(index_advance, 1),
                          len(self._ref_path) - 1)

                p_ref = self._ref_path[idx]

                # Velocity reference
                if i == 0:
                    v_ref = (p_ref - pos_w) / dt_mpc
                elif i == N - 1:
                    v_ref = np.zeros(3)
                else:
                    v_ref = (p_ref - last_p_ref) / dt_mpc

                last_p_ref = p_ref.copy()
                self.mpc.set_goal(p_ref, v_ref, np.zeros(3), i)

                # --- CIRI safe flight corridor (only first few steps) ---
                if self.cfg.use_sfc and self._map_built and i < self.cfg.max_sfc_steps:
                    self._add_sfc_for_step(pos_w, p_ref, i)
        else:
            # No A* path — use straight-line to goal but still apply SFC
            # to avoid flying blind through obstacles.
            for i in range(N):
                t = (i + 1) / N
                p_ref = pos_w + t * (goal - pos_w)
                if i < N - 1:
                    v_ref = (goal - pos_w) / (N * dt_mpc)
                else:
                    v_ref = np.zeros(3)
                self.mpc.set_goal(p_ref, v_ref, np.zeros(3), i)

                if self.cfg.use_sfc and self._map_built and i < self.cfg.max_sfc_steps:
                    self._add_sfc_for_step(pos_w, p_ref, i)

        # --- MPC solve ---
        self.mpc.set_status(pos_w, vel_w, acc_w)
        success = self.mpc.solve()

        if success:
            result = self.mpc.get_first_velocity()
            if np.all(np.isfinite(result)):
                # Clamp output speed to max_speed to prevent runaway
                out_speed = np.linalg.norm(result)
                if out_speed > self.cfg.max_speed:
                    result = result * (self.cfg.max_speed / out_speed)
                return result
        # Fallback: return clamped user velocity directly
        return user_vel_w

    def _add_sfc_for_step(self, pos_w: np.ndarray, goal_w: np.ndarray,
                          step: int):
        """Generate and add CIRI SFC for one MPC horizon step."""
        self._sfc_attempts += 1
        try:
            # Build boundary box that always contains both seed points
            margin = self.cfg.obstacle_query_radius
            box_min = np.minimum(pos_w, goal_w) - margin
            box_max = np.maximum(pos_w, goal_w) + margin

            obstacle_pts = self.occ_grid.box_search(
                box_min, box_max,
                max_points=self.cfg.max_obstacle_points,
                downsample_resolution=self.cfg.occupancy_grid.resolution * 3,
            )
            if obstacle_pts is None or len(obstacle_pts) == 0:
                return

            # Generate boundary planes
            boundary = self._make_boundary_planes(box_min, box_max)

            # CIRI corridor
            planes = self.ciri.generate(
                boundary=boundary,
                obstacles=obstacle_pts.T,  # (3, N) format
                seed_a=pos_w,
                seed_b=goal_w,
            )

            if planes is not None and len(planes) > 0:
                self.mpc.set_sfc(planes, step)
                self._last_sfc[step] = planes.copy()
                self._sfc_successes += 1
        except Exception:
            if not self.cfg.fallback_to_simple:
                raise

    @staticmethod
    def _make_boundary_planes(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
        """Create 6 halfplane constraints for a 3D bounding box.

        Each row [a, b, c, d] defines a*x + b*y + c*z + d <= 0.
        """
        planes = np.array([
            [ 1,  0,  0, -box_max[0]],  # x <= box_max[0]
            [-1,  0,  0,  box_min[0]],  # x >= box_min[0]
            [ 0,  1,  0, -box_max[1]],  # y <= box_max[1]
            [ 0, -1,  0,  box_min[1]],  # y >= box_min[1]
            [ 0,  0,  1, -box_max[2]],  # z <= box_max[2]
            [ 0,  0, -1,  box_min[2]],  # z >= box_min[2]
        ], dtype=float)
        return planes

    # -- Visualization accessors -----------------------------------------------

    def get_last_sfc(self) -> dict:
        """Return SFC planes from the most recent step.

        Returns:
            ``{step_index: (K, 4) ndarray}`` — halfplane constraints per
            MPC horizon step.  Empty dict if SFC was not active.
        """
        return self._last_sfc

    def get_ref_path(self) -> Optional[np.ndarray]:
        """Return the current A* reference path, or None."""
        return self._ref_path

    def get_sfc_stats(self) -> dict:
        """Return CIRI SFC success rate statistics.

        Returns:
            dict with 'attempts', 'successes', 'success_rate'.
        """
        rate = self._sfc_successes / max(self._sfc_attempts, 1)
        return {
            'attempts': self._sfc_attempts,
            'successes': self._sfc_successes,
            'success_rate': rate,
        }

    def reset_sfc_stats(self):
        """Reset SFC success counters (call between trials)."""
        self._sfc_attempts = 0
        self._sfc_successes = 0
