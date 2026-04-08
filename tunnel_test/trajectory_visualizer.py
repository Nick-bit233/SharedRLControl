"""
2D Trajectory Visualization for IPC vs RL Comparison.

Provides:
- ``FlightDataRecorder`` — lightweight per-frame data collector used during sim
- ``TrajectoryVisualizer`` — matplotlib offline renderer producing .mp4 animations

Usage::

    # During simulation
    recorder = FlightDataRecorder()
    for frame in range(N):
        recorder.record_frame(pos=..., vel=..., ...)
    recorder.save("logs/trial_ipc_0.npz")

    # After all trials
    viz = TrajectoryVisualizer(obstacles, tunnel_bounds)
    viz.render_comparison(
        ipc_data="logs/trial_ipc_0.npz",
        rl_data="logs/trial_rl_0.npz",
        output="logs/compare_trial0.mp4",
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

logger = logging.getLogger(__name__)


# =============================================================================
# FlightDataRecorder
# =============================================================================

class FlightDataRecorder:
    """Lightweight per-frame data collector for post-experiment visualization.

    All fields are append-only Python lists; overhead is negligible.
    Call :meth:`save` after the trial to write compressed ``.npz``.
    """

    def __init__(self, controller_type: str = "IPC"):
        """
        Args:
            controller_type: "IPC" or "RL" — determines which optional fields
                             are expected (SFC planes vs LiDAR rays).
        """
        self.controller_type = controller_type

        # Per-frame arrays (always recorded)
        self._positions: List[np.ndarray] = []       # (3,)
        self._human_vels_w: List[np.ndarray] = []    # (3,)
        self._ctrl_vels_w: List[np.ndarray] = []     # (3,)
        self._collisions: List[bool] = []

        # IPC-specific: SFC halfplanes per MPC step
        # Each entry: dict {step_idx: (K, 4) ndarray} or empty dict
        self._sfc_planes: List[Dict[int, np.ndarray]] = []
        # IPC-specific: A* reference path
        self._ref_paths: List[Optional[np.ndarray]] = []

        # RL-specific: LiDAR ray hit world positions
        # Each entry: (N_rays, 3) or None
        self._lidar_hits: List[Optional[np.ndarray]] = []

    def record_frame(
        self,
        pos: np.ndarray,
        human_vel_w: np.ndarray,
        ctrl_vel_w: np.ndarray,
        is_collision: bool = False,
        sfc_planes: Optional[Dict[int, np.ndarray]] = None,
        ref_path: Optional[np.ndarray] = None,
        lidar_hits_w: Optional[np.ndarray] = None,
    ):
        """Record one frame of flight data.

        Args:
            pos: (3,) world position.
            human_vel_w: (3,) user velocity in world frame.
            ctrl_vel_w: (3,) controller output velocity in world frame.
            is_collision: Whether this frame has a collision.
            sfc_planes: (IPC) SFC halfplane dict from ``ipc.get_last_sfc()``.
            ref_path: (IPC) A* reference path (M, 3).
            lidar_hits_w: (RL) LiDAR ray hit positions (N, 3).
        """
        self._positions.append(np.asarray(pos, dtype=np.float32).copy())
        self._human_vels_w.append(np.asarray(human_vel_w, dtype=np.float32).copy())
        self._ctrl_vels_w.append(np.asarray(ctrl_vel_w, dtype=np.float32).copy())
        self._collisions.append(bool(is_collision))

        # SFC: deep-copy planes dict (small data)
        if sfc_planes is not None:
            self._sfc_planes.append(
                {k: v.copy() for k, v in sfc_planes.items()}
            )
        else:
            self._sfc_planes.append({})

        self._ref_paths.append(ref_path.copy() if ref_path is not None else None)
        self._lidar_hits.append(
            lidar_hits_w.copy() if lidar_hits_w is not None else None
        )

    @property
    def num_frames(self) -> int:
        return len(self._positions)

    def save(self, path: str):
        """Save recorded data to compressed ``.npz`` file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        positions = np.array(self._positions)     # (T, 3)
        human_vels = np.array(self._human_vels_w) # (T, 3)
        ctrl_vels = np.array(self._ctrl_vels_w)   # (T, 3)
        collisions = np.array(self._collisions)   # (T,)

        save_dict = {
            "controller_type": self.controller_type,
            "positions": positions,
            "human_vels_w": human_vels,
            "ctrl_vels_w": ctrl_vels,
            "collisions": collisions,
        }

        # SFC planes: save as object array of dicts (variable-length)
        if any(len(d) > 0 for d in self._sfc_planes):
            sfc_list = []
            for d in self._sfc_planes:
                if d:
                    # Flatten to list of (step, planes) tuples
                    frame_data = {}
                    for step, planes in d.items():
                        frame_data[str(step)] = planes
                    sfc_list.append(frame_data)
                else:
                    sfc_list.append({})
            save_dict["sfc_planes"] = np.array(sfc_list, dtype=object)

        # Ref paths: save last non-None path for static overlay
        last_ref = None
        for rp in reversed(self._ref_paths):
            if rp is not None:
                last_ref = rp
                break
        if last_ref is not None:
            save_dict["ref_path"] = last_ref

        # LiDAR hits: save as object array (variable shape per frame)
        if any(h is not None for h in self._lidar_hits):
            save_dict["lidar_hits"] = np.array(self._lidar_hits, dtype=object)

        np.savez_compressed(path, **save_dict)
        logger.info(f"Flight data saved: {path} ({self.num_frames} frames)")

    @staticmethod
    def load(path: str) -> dict:
        """Load recorded data from ``.npz`` file.

        Returns:
            dict with keys: controller_type, positions, human_vels_w,
            ctrl_vels_w, collisions, and optionally sfc_planes, ref_path,
            lidar_hits.
        """
        data = np.load(path, allow_pickle=True)
        result = {
            "controller_type": str(data["controller_type"]),
            "positions": data["positions"],
            "human_vels_w": data["human_vels_w"],
            "ctrl_vels_w": data["ctrl_vels_w"],
            "collisions": data["collisions"],
        }
        if "sfc_planes" in data:
            result["sfc_planes"] = data["sfc_planes"]
        if "ref_path" in data:
            result["ref_path"] = data["ref_path"]
        if "lidar_hits" in data:
            result["lidar_hits"] = data["lidar_hits"]
        return result


# =============================================================================
# TrajectoryVisualizer
# =============================================================================

@dataclass
class ObstacleInfo:
    """Obstacle descriptor for 2D rendering."""
    center_x: float
    center_y: float
    half_width: float
    half_height: float
    z_height: float = 5.0  # for color intensity


def obstacles_to_info(obstacles: list) -> List[ObstacleInfo]:
    """Convert obstacle dicts from extract_obstacles_from_heightfield to ObstacleInfo."""
    result = []
    for obs in obstacles:
        cx, cy = obs["center"]
        r = obs["radius"]
        result.append(ObstacleInfo(
            center_x=cx, center_y=cy,
            half_width=r, half_height=r,
            z_height=obs.get("height", 5.0),
        ))
    return result


def _project_sfc_to_2d(
    planes_3d: np.ndarray, z0: float
) -> Optional[np.ndarray]:
    """Project 3D halfplane polytope to 2D polygon at altitude z0.

    Args:
        planes_3d: (K, 4) halfplanes ``ax + by + cz + d ≤ 0``.
        z0: Drone altitude for substitution.

    Returns:
        (N, 2) polygon vertices ordered CCW, or None if degenerate.
    """
    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection, ConvexHull

    # Substitute z = z0: ax + by + (c*z0 + d) ≤ 0
    planes_2d = []
    for a, b, c, d in planes_3d:
        ab_norm = np.sqrt(a * a + b * b)
        if ab_norm < 1e-10:
            # Pure z-constraint — check feasibility
            val = c * z0 + d
            if val > 1e-6:
                return None  # infeasible
            continue
        planes_2d.append([a, b, c * z0 + d])

    if len(planes_2d) < 3:
        return None

    planes_2d = np.array(planes_2d)  # (M, 3): ax + by + d' ≤ 0

    # Find interior point via Chebyshev centre LP
    normals = planes_2d[:, :2]
    offsets = planes_2d[:, 2]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-30)

    A_ub = np.hstack([normals, norms_safe])
    b_ub = -offsets
    c_obj = np.array([0.0, 0.0, -1.0])

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * 3, method="highs")
    if not res.success or res.x[2] < 1e-8:
        return None

    interior = res.x[:2]

    # HalfspaceIntersection expects format: [a, b, d] where ax+by+d ≤ 0
    try:
        hs = HalfspaceIntersection(planes_2d, interior)
    except Exception:
        return None

    vertices = hs.intersections
    if len(vertices) < 3:
        return None

    try:
        hull = ConvexHull(vertices)
    except Exception:
        return None

    return vertices[hull.vertices]


class TrajectoryVisualizer:
    """Matplotlib-based 2D flight trajectory animator."""

    def __init__(
        self,
        obstacles: List[ObstacleInfo],
        tunnel_x_range: Tuple[float, float] = (-12.0, 12.0),
        tunnel_y_range: Tuple[float, float] = (-6.0, 6.0),
        z_range: Tuple[float, float] = (0.0, 8.0),
        wall_y: Optional[Tuple[float, float]] = (-5.5, 5.5),
    ):
        """
        Args:
            obstacles: List of ObstacleInfo for static obstacle rendering.
            tunnel_x_range: X-axis display range.
            tunnel_y_range: Y-axis display range.
            z_range: Altitude range for color mapping.
            wall_y: Y-coordinates of tunnel side walls (inner edges).
        """
        self.obstacles = obstacles
        self.tunnel_x_range = tunnel_x_range
        self.tunnel_y_range = tunnel_y_range
        self.z_range = z_range
        self.wall_y = wall_y
        self._z_cmap = plt.cm.viridis
        self._z_norm = mcolors.Normalize(vmin=z_range[0], vmax=z_range[1])

    def _draw_obstacles(self, ax: plt.Axes):
        """Draw obstacles as gray rectangles."""
        for obs in self.obstacles:
            rect = mpatches.FancyBboxPatch(
                (obs.center_x - obs.half_width,
                 obs.center_y - obs.half_height),
                obs.half_width * 2, obs.half_height * 2,
                boxstyle="round,pad=0.02",
                facecolor="#8B7355", edgecolor="#5C4033",
                alpha=0.7, linewidth=0.5,
            )
            ax.add_patch(rect)

        # Tunnel walls
        if self.wall_y is not None:
            xl = self.tunnel_x_range
            for wy in self.wall_y:
                sign = 1 if wy > 0 else -1
                wall_rect = mpatches.Rectangle(
                    (xl[0], wy),
                    xl[1] - xl[0],
                    sign * 2.0,  # wall thickness
                    facecolor="#555555", edgecolor="#333333",
                    alpha=0.8, linewidth=0.5,
                )
                ax.add_patch(wall_rect)

    def _setup_axis(self, ax: plt.Axes, title: str = ""):
        """Configure axis appearance."""
        ax.set_xlim(self.tunnel_x_range)
        ax.set_ylim(self.tunnel_y_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        self._draw_obstacles(ax)

    def render_trial(
        self,
        data_path: str,
        output_path: str,
        fps: int = 20,
        subsample: int = 3,
        trail_length: int = 200,
    ):
        """Render a single trial as an animated .mp4.

        Args:
            data_path: Path to .npz file from FlightDataRecorder.
            output_path: Output .mp4 path.
            fps: Animation frame rate.
            subsample: Take every Nth sim frame.
            trail_length: How many past positions to show in the trail.
        """
        data = FlightDataRecorder.load(data_path)
        self._render_single(data, output_path, fps, subsample, trail_length)

    def render_comparison(
        self,
        ipc_data_path: str,
        rl_data_path: str,
        output_path: str,
        fps: int = 20,
        subsample: int = 3,
        trail_length: int = 200,
        trial_label: str = "",
    ):
        """Render side-by-side IPC vs RL animation.

        Args:
            ipc_data_path: Path to IPC trial .npz.
            rl_data_path: Path to RL trial .npz.
            output_path: Output .mp4 path.
            fps: Animation frame rate.
            subsample: Take every Nth sim frame.
            trail_length: Past trail length.
            trial_label: Label for the trial (e.g., "Trial 1, seed=42").
        """
        ipc_data = FlightDataRecorder.load(ipc_data_path)
        rl_data = FlightDataRecorder.load(rl_data_path)
        self._render_side_by_side(
            ipc_data, rl_data, output_path, fps, subsample, trail_length,
            trial_label,
        )

    def render_comparison_from_recorders(
        self,
        ipc_recorder: FlightDataRecorder,
        rl_recorder: FlightDataRecorder,
        output_path: str,
        fps: int = 20,
        subsample: int = 3,
        trail_length: int = 200,
        trial_label: str = "",
    ):
        """Render side-by-side from in-memory recorders (no file I/O)."""
        ipc_data = self._recorder_to_data(ipc_recorder)
        rl_data = self._recorder_to_data(rl_recorder)
        self._render_side_by_side(
            ipc_data, rl_data, output_path, fps, subsample, trail_length,
            trial_label,
        )

    @staticmethod
    def _recorder_to_data(recorder: FlightDataRecorder) -> dict:
        """Convert in-memory recorder to the same dict format as load()."""
        data = {
            "controller_type": recorder.controller_type,
            "positions": np.array(recorder._positions),
            "human_vels_w": np.array(recorder._human_vels_w),
            "ctrl_vels_w": np.array(recorder._ctrl_vels_w),
            "collisions": np.array(recorder._collisions),
        }
        if any(len(d) > 0 for d in recorder._sfc_planes):
            data["sfc_planes"] = recorder._sfc_planes  # keep as list of dicts
        last_ref = None
        for rp in reversed(recorder._ref_paths):
            if rp is not None:
                last_ref = rp
                break
        if last_ref is not None:
            data["ref_path"] = last_ref
        if any(h is not None for h in recorder._lidar_hits):
            data["lidar_hits"] = recorder._lidar_hits
        return data

    # ---- Internal rendering methods ----

    def _render_single(self, data, output_path, fps, subsample, trail_length):
        """Render one trial."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        ctrl_type = data["controller_type"]
        self._setup_axis(ax, f"{ctrl_type} Trajectory")

        positions = data["positions"]
        n_frames = len(positions)
        frame_indices = list(range(0, n_frames, subsample))

        # Static elements
        if "ref_path" in data:
            path_xy = data["ref_path"]
            ax.plot(path_xy[:, 0], path_xy[:, 1], '--', color='m',
                    alpha=0.5, linewidth=1.0, label="A* path", zorder=2)

        # Dynamic artists
        trail_line, = ax.plot([], [], '-', linewidth=1.5, alpha=0.7, zorder=5)
        drone_dot, = ax.plot([], [], 'o', markersize=8, zorder=10)
        arrow_human = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="->", color="royalblue",
                                                  lw=1.5),
                                  zorder=8)
        arrow_ctrl = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="->", color="red",
                                                 lw=1.5),
                                 zorder=8)
        col_scatter = ax.scatter([], [], c='red', s=30, marker='x',
                                 zorder=6, alpha=0.6)
        sfc_patches = []
        lidar_lines = []
        frame_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                             fontsize=8, va="top", fontfamily="monospace",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # Colorbar for altitude
        sm = plt.cm.ScalarMappable(cmap=self._z_cmap, norm=self._z_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Altitude (m)", shrink=0.6, pad=0.02)

        def _update(anim_frame):
            nonlocal sfc_patches, lidar_lines
            fi = frame_indices[anim_frame]

            # Trail
            start = max(0, fi - trail_length)
            trail_pos = positions[start:fi + 1]
            if len(trail_pos) > 1:
                # Color trail by altitude
                trail_colors = self._z_cmap(self._z_norm(trail_pos[:, 2]))
                trail_line.set_data(trail_pos[:, 0], trail_pos[:, 1])
                trail_line.set_color(trail_colors[-1])
            else:
                trail_line.set_data([], [])

            # Drone dot
            pos = positions[fi]
            color = self._z_cmap(self._z_norm(pos[2]))
            drone_dot.set_data([pos[0]], [pos[1]])
            drone_dot.set_color(color)

            # Velocity arrows
            scale = 0.5  # arrow length scale
            h_vel = data["human_vels_w"][fi]
            c_vel = data["ctrl_vels_w"][fi]
            arrow_human.xy = (pos[0] + h_vel[0] * scale, pos[1] + h_vel[1] * scale)
            arrow_human.set_position((pos[0], pos[1]))
            arrow_ctrl.xy = (pos[0] + c_vel[0] * scale, pos[1] + c_vel[1] * scale)
            arrow_ctrl.set_position((pos[0], pos[1]))

            # Collision markers (accumulated)
            col_mask = data["collisions"][:fi + 1]
            col_pos = positions[:fi + 1][col_mask]
            if len(col_pos) > 0:
                col_scatter.set_offsets(col_pos[:, :2])
            else:
                col_scatter.set_offsets(np.empty((0, 2)))

            # Remove old SFC patches
            for p in sfc_patches:
                p.remove()
            sfc_patches = []

            # Remove old LiDAR lines
            for ln in lidar_lines:
                ln.remove()
            lidar_lines = []

            # SFC polygons (IPC)
            if "sfc_planes" in data:
                sfc_data = data["sfc_planes"]
                if fi < len(sfc_data):
                    frame_sfc = sfc_data[fi]
                    if isinstance(frame_sfc, dict) and len(frame_sfc) > 0:
                        z0 = pos[2]
                        for step_key, planes in frame_sfc.items():
                            verts = _project_sfc_to_2d(planes, z0)
                            if verts is not None and len(verts) >= 3:
                                poly = mpatches.Polygon(
                                    verts, closed=True,
                                    facecolor="cyan", edgecolor="darkcyan",
                                    alpha=0.25, linewidth=1.0, zorder=3,
                                )
                                ax.add_patch(poly)
                                sfc_patches.append(poly)

            # LiDAR rays (RL)
            if "lidar_hits" in data:
                hits_data = data["lidar_hits"]
                if fi < len(hits_data):
                    hits = hits_data[fi]
                    if hits is not None and len(hits) > 0:
                        hits = np.asarray(hits)
                        # Subsample rays for clarity (every 2nd ray)
                        hits_sub = hits[::2]
                        dists = np.linalg.norm(
                            hits_sub[:, :2] - pos[:2], axis=1
                        )
                        max_d = 4.0  # LIDAR_RANGE
                        for j, (hit, d) in enumerate(zip(hits_sub, dists)):
                            ratio = np.clip(d / max_d, 0, 1)
                            # Green (far) → Red (close)
                            c_ray = (ratio, 0.0, 1.0 - ratio, 0.3)
                            ln, = ax.plot(
                                [pos[0], hit[0]], [pos[1], hit[1]],
                                '-', color=c_ray, linewidth=0.6, zorder=4,
                            )
                            lidar_lines.append(ln)

            # Frame info text
            frame_text.set_text(
                f"Frame {fi}/{n_frames-1}  "
                f"x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f}"
            )

            return [trail_line, drone_dot, col_scatter, frame_text]

        anim = FuncAnimation(
            fig, _update, frames=len(frame_indices),
            interval=1000 // fps, blit=False,
        )

        self._save_animation(anim, output_path, fps)
        plt.close(fig)
        logger.info(f"Animation saved: {output_path}")

    def _render_side_by_side(
        self, ipc_data, rl_data, output_path, fps, subsample,
        trail_length, trial_label,
    ):
        """Render side-by-side IPC vs RL comparison."""
        fig, (ax_ipc, ax_rl) = plt.subplots(1, 2, figsize=(22, 8))
        self._setup_axis(ax_ipc, "IPC (MPC + Safe Corridor)")
        self._setup_axis(ax_rl, "RL Policy")

        if trial_label:
            fig.suptitle(trial_label, fontsize=13, fontweight="bold", y=0.98)

        # Legend
        legend_elements = [
            mpatches.Patch(color="royalblue", alpha=0.6, label="User intent"),
            mpatches.Patch(color="red", alpha=0.6, label="Control output"),
            mpatches.Patch(color="cyan", alpha=0.4, label="Safe corridor (IPC)"),
            plt.Line2D([0], [0], color="m", linestyle="--", alpha=0.5, label="A* path (IPC)"),
            plt.Line2D([0], [0], color="green", alpha=0.5, label="LiDAR rays (RL)"),
            plt.Line2D([0], [0], marker="x", color="red", linestyle="None", label="Collision"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=6,
                   fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

        # Altitude colorbar (shared)
        sm = plt.cm.ScalarMappable(cmap=self._z_cmap, norm=self._z_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=[ax_ipc, ax_rl], label="Altitude (m)",
                     shrink=0.6, pad=0.02, location="right")

        n_ipc = len(ipc_data["positions"])
        n_rl = len(rl_data["positions"])
        n_max = max(n_ipc, n_rl)
        frame_indices = list(range(0, n_max, subsample))

        # Static: A* path on IPC axis
        if "ref_path" in ipc_data:
            rp = ipc_data["ref_path"]
            ax_ipc.plot(rp[:, 0], rp[:, 1], '--', color='m',
                        alpha=0.5, linewidth=1.0, zorder=2)

        # Create artist containers for each side
        def _make_artists(ax):
            trail, = ax.plot([], [], '-', linewidth=1.5, alpha=0.7, zorder=5)
            dot, = ax.plot([], [], 'o', markersize=8, zorder=10)
            arr_h = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->",
                                                color="royalblue", lw=1.5),
                                zorder=8)
            arr_c = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->",
                                                color="red", lw=1.5),
                                zorder=8)
            col_sc = ax.scatter([], [], c='red', s=30, marker='x',
                                zorder=6, alpha=0.6)
            info = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                           fontsize=8, va="top", fontfamily="monospace",
                           bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            return {
                "trail": trail, "dot": dot, "arr_h": arr_h, "arr_c": arr_c,
                "col_sc": col_sc, "info": info,
                "sfc_patches": [], "lidar_lines": [],
            }

        a_ipc = _make_artists(ax_ipc)
        a_rl = _make_artists(ax_rl)

        def _update_panel(ax, artists, data, fi, trail_length):
            positions = data["positions"]
            n = len(positions)
            if fi >= n:
                fi = n - 1

            # Trail
            start = max(0, fi - trail_length)
            trail_pos = positions[start:fi + 1]
            if len(trail_pos) > 1:
                color = self._z_cmap(self._z_norm(trail_pos[-1, 2]))
                artists["trail"].set_data(trail_pos[:, 0], trail_pos[:, 1])
                artists["trail"].set_color(color)
            else:
                artists["trail"].set_data([], [])

            # Drone dot
            pos = positions[fi]
            color = self._z_cmap(self._z_norm(pos[2]))
            artists["dot"].set_data([pos[0]], [pos[1]])
            artists["dot"].set_color(color)
            artists["dot"].set_markeredgecolor("black")
            artists["dot"].set_markeredgewidth(0.5)

            # Velocity arrows
            scale = 0.5
            hv = data["human_vels_w"][fi]
            cv = data["ctrl_vels_w"][fi]
            artists["arr_h"].xy = (pos[0] + hv[0] * scale, pos[1] + hv[1] * scale)
            artists["arr_h"].set_position((pos[0], pos[1]))
            artists["arr_c"].xy = (pos[0] + cv[0] * scale, pos[1] + cv[1] * scale)
            artists["arr_c"].set_position((pos[0], pos[1]))

            # Collision markers
            cm = data["collisions"][:fi + 1]
            cp = positions[:fi + 1][cm]
            if len(cp) > 0:
                artists["col_sc"].set_offsets(cp[:, :2])
            else:
                artists["col_sc"].set_offsets(np.empty((0, 2)))

            # Clear dynamic overlays
            for p in artists["sfc_patches"]:
                p.remove()
            artists["sfc_patches"] = []
            for ln in artists["lidar_lines"]:
                ln.remove()
            artists["lidar_lines"] = []

            # SFC polygons (IPC)
            if "sfc_planes" in data:
                sfc_list = data["sfc_planes"]
                if fi < len(sfc_list):
                    frame_sfc = sfc_list[fi]
                    if isinstance(frame_sfc, dict) and frame_sfc:
                        z0 = pos[2]
                        for step_key, planes in frame_sfc.items():
                            verts = _project_sfc_to_2d(planes, z0)
                            if verts is not None and len(verts) >= 3:
                                poly = mpatches.Polygon(
                                    verts, closed=True,
                                    facecolor="cyan", edgecolor="darkcyan",
                                    alpha=0.25, linewidth=1.0, zorder=3,
                                )
                                ax.add_patch(poly)
                                artists["sfc_patches"].append(poly)

            # LiDAR rays (RL)
            if "lidar_hits" in data:
                hits_list = data["lidar_hits"]
                if fi < len(hits_list):
                    hits = hits_list[fi]
                    if hits is not None and len(hits) > 0:
                        hits = np.asarray(hits)
                        hits_sub = hits[::2]  # subsample
                        for hit in hits_sub:
                            d = np.linalg.norm(hit[:2] - pos[:2])
                            ratio = np.clip(d / 4.0, 0, 1)
                            c_ray = (1.0 - ratio, ratio, 0.0, 0.3)
                            ln, = ax.plot(
                                [pos[0], hit[0]], [pos[1], hit[1]],
                                '-', color=c_ray, linewidth=0.5, zorder=4,
                            )
                            artists["lidar_lines"].append(ln)

            # Info text
            artists["info"].set_text(
                f"Frame {fi}  x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f}"
            )

        def _update(anim_frame):
            fi = frame_indices[anim_frame]
            _update_panel(ax_ipc, a_ipc, ipc_data, fi, trail_length)
            _update_panel(ax_rl, a_rl, rl_data, fi, trail_length)
            return []

        anim = FuncAnimation(
            fig, _update, frames=len(frame_indices),
            interval=1000 // fps, blit=False,
        )

        self._save_animation(anim, output_path, fps)
        plt.close(fig)
        logger.info(f"Comparison animation saved: {output_path}")

    @staticmethod
    def _save_animation(anim, output_path: str, fps: int):
        """Save animation as .mp4 (preferred) or .gif fallback."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if output_path.endswith(".mp4"):
            try:
                writer = FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
                return
            except Exception as e:
                logger.warning(f"FFMpeg failed ({e}), falling back to GIF")
                output_path = output_path.replace(".mp4", ".gif")

        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)

    def render_static_comparison(
        self,
        ipc_data_or_path,
        rl_data_or_path,
        output_path: str,
        trial_label: str = "",
    ):
        """Render static (non-animated) side-by-side trajectory plot.

        Useful for quick overview without animation overhead.
        """
        if isinstance(ipc_data_or_path, str):
            ipc_data = FlightDataRecorder.load(ipc_data_or_path)
        else:
            ipc_data = self._recorder_to_data(ipc_data_or_path)

        if isinstance(rl_data_or_path, str):
            rl_data = FlightDataRecorder.load(rl_data_or_path)
        else:
            rl_data = self._recorder_to_data(rl_data_or_path)

        fig, (ax_ipc, ax_rl) = plt.subplots(1, 2, figsize=(22, 8))
        self._setup_axis(ax_ipc, "IPC (MPC + Safe Corridor)")
        self._setup_axis(ax_rl, "RL Policy")

        if trial_label:
            fig.suptitle(trial_label, fontsize=13, fontweight="bold")

        for ax, data, label in [(ax_ipc, ipc_data, "IPC"), (ax_rl, rl_data, "RL")]:
            pos = data["positions"]
            cols = data["collisions"]

            # Full trajectory colored by altitude
            if len(pos) > 1:
                points = pos[:, :2].reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                z_colors = self._z_cmap(self._z_norm(pos[:-1, 2]))
                lc = mcoll.LineCollection(segments, colors=z_colors,
                                          linewidths=2.0, zorder=5)
                ax.add_collection(lc)

            # Collision points
            col_pos = pos[cols]
            if len(col_pos) > 0:
                ax.scatter(col_pos[:, 0], col_pos[:, 1], c='red', s=30,
                           marker='x', zorder=6, alpha=0.6, label="Collision")

            # Start / end markers
            ax.plot(pos[0, 0], pos[0, 1], 'gs', markersize=10, zorder=10,
                    label="Start")
            ax.plot(pos[-1, 0], pos[-1, 1], 'r^', markersize=10, zorder=10,
                    label="End")

            # A* path (IPC only)
            if "ref_path" in data:
                rp = data["ref_path"]
                ax.plot(rp[:, 0], rp[:, 1], '--', color='m', alpha=0.5,
                        linewidth=1.0, label="A* path")

            ax.legend(fontsize=8, loc="upper right")

        # Shared colorbar
        sm = plt.cm.ScalarMappable(cmap=self._z_cmap, norm=self._z_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=[ax_ipc, ax_rl], label="Altitude (m)",
                     shrink=0.6, pad=0.02, location="right")

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Static comparison saved: {output_path}")
