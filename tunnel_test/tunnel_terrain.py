"""
Shared tunnel terrain utilities.

Provides the tunnel terrain definition, obstacle extraction, and helper
functions used by both ``verify_ipc.py`` and ``compare_ipc_rl.py``.

This module deliberately does **not** create an ``AppLauncher`` or
``SimulationApp`` so it can be safely imported from any script that has
already initialised the simulator.
"""

from __future__ import annotations

import copy
import numpy as np
from isaaclab.terrains import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh

# ============== Tunnel Terrain (same as env_tunnel.py) ==============

# Match the training environment's tunnel reset pose.
INIT_POS = [-7.0, 0.0, 5.0]
INIT_QUAT = [1.0, 0.0, 0.0, 0.0]

# Fixed seed for legacy np.random so that the offline heightfield
# regeneration produces exactly the same obstacles as the simulator.
# Set np.random.seed(TERRAIN_LEGACY_SEED) BEFORE creating TerrainImporter,
# and regenerate_heightfield() will replay the same state.
TERRAIN_LEGACY_SEED = 42

# --------------- Heightfield Capture Mechanism ---------------
# The terrain function below saves its output so we can build
# the occupancy grid from the REAL heightfield, not a re-generated copy.
_captured_tiles: list[dict] = []


def clear_captured_tiles():
    """Clear the capture buffer.  Call BEFORE creating TerrainImporter."""
    global _captured_tiles
    _captured_tiles = []


def get_captured_heightfield(terrain_gen_cfg):
    """Assemble the full bordered heightfield from tiles captured during
    the actual terrain generation.

    Must be called AFTER ``TerrainImporter`` creation.

    Returns:
        Full heightfield as int16 ndarray, or *None* if nothing was captured.
    """
    if not _captured_tiles:
        return None

    h_scale = terrain_gen_cfg.horizontal_scale
    size = terrain_gen_cfg.size
    num_rows = terrain_gen_cfg.num_rows
    num_cols = terrain_gen_cfg.num_cols

    sub_w_px = int(size[0] / h_scale) + 1
    sub_l_px = int(size[1] / h_scale) + 1
    full_w = sub_w_px * num_cols
    full_l = sub_l_px * num_rows
    full_hf = np.zeros((full_w, full_l), dtype=np.int16)

    for index, tile in enumerate(_captured_tiles):
        sub_row, sub_col = np.unravel_index(index, (num_rows, num_cols))
        hf_inner = tile["hf_inner"]
        border_px = tile["border_px"]

        bordered = np.zeros((sub_w_px, sub_l_px), dtype=np.int16)
        bordered[border_px : border_px + hf_inner.shape[0],
                 border_px : border_px + hf_inner.shape[1]] = hf_inner

        x0 = sub_col * sub_w_px
        y0 = sub_row * sub_l_px
        full_hf[x0 : x0 + sub_w_px, y0 : y0 + sub_l_px] = bordered

    return full_hf


@height_field_to_mesh
def tunnel_obstacles_terrain(difficulty: float, cfg: HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Custom terrain with walls forming a tunnel + interior obstacles.

    Side-effect: appends the produced heightfield to ``_captured_tiles``
    so that :func:`get_captured_heightfield` can assemble the exact tile
    without PRNG replication.
    """
    hf_raw = hf_terrains.discrete_obstacles_terrain.__wrapped__(difficulty, cfg)

    # Convert pits (negative heights) to pillars of the same magnitude.
    # In "choice" mode the base function randomly assigns negative heights
    # to ~50% of obstacles, halving effective density.  Taking abs keeps
    # the height variety (half / full) while ensuring all obstacles are
    # physical pillars above ground.
    np.abs(hf_raw, out=hf_raw)

    wall_thickness_meters = 1.0
    wall_height_meters = 10.0
    wall_start_meters = 2.0
    clear_zone_meters = 4.0

    wall_thickness_pixels = int(wall_thickness_meters / cfg.horizontal_scale)
    wall_height_steps = int(wall_height_meters / cfg.vertical_scale)
    wall_start_pixels = int(wall_start_meters / cfg.horizontal_scale)
    clear_zone_pixels = int(clear_zone_meters / cfg.horizontal_scale)

    # Clear spawn zone
    hf_raw[0: wall_start_pixels + clear_zone_pixels, :] = 0
    # Side walls
    hf_raw[:, 0:wall_thickness_pixels] = wall_height_steps
    hf_raw[:, -wall_thickness_pixels:] = wall_height_steps

    # Capture for offline occupancy grid construction
    _captured_tiles.append({
        "hf_inner": hf_raw.copy(),
        "border_px": int(cfg.border_width / cfg.horizontal_scale) + 1,
    })

    return hf_raw


@configclass
class HfTunnelObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = tunnel_obstacles_terrain


# ============== Helpers ==============

def quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


# ============== Offline Heightfield Regeneration ==============

def regenerate_heightfield(terrain_gen_cfg, tunnel_mode=False):
    """
    Reproduce the exact heightfield generated by Isaac Sim's TerrainGenerator.

    Calls the **real** Isaac Lab ``discrete_obstacles_terrain`` function with
    matched PRNG state so that obstacle positions exactly match those in the
    running simulation.

    Prerequisites:
        ``np.random.seed(TERRAIN_LEGACY_SEED)`` must have been called
        **before** ``TerrainImporter`` created the terrain.  This function
        seeds with the same value to replay the identical legacy-random state.

    Args:
        terrain_gen_cfg: TerrainGeneratorCfg object.
        tunnel_mode: If True, apply tunnel wall modifications to each tile.

    Returns:
        Full heightfield as 2D numpy int16 array.
    """
    h_scale = terrain_gen_cfg.horizontal_scale
    v_scale = terrain_gen_cfg.vertical_scale
    num_rows = terrain_gen_cfg.num_rows
    num_cols = terrain_gen_cfg.num_cols
    size = terrain_gen_cfg.size

    # Match the decorator's total tile size: int(size/h) + 1 pixels
    # (fence-post: 241 posts for 240 intervals of 0.1m = 24.0m)
    sub_w_px = int(size[0] / h_scale) + 1
    sub_l_px = int(size[1] / h_scale) + 1
    full_w = sub_w_px * num_cols
    full_l = sub_l_px * num_rows
    full_hf = np.zeros((full_w, full_l), dtype=np.int16)

    seed = terrain_gen_cfg.seed if terrain_gen_cfg.seed is not None else 0

    # Reproduce TerrainGenerator's new-style RNG (for sub-index & difficulty).
    np_rng = np.random.default_rng(seed)
    sub_cfgs = list(terrain_gen_cfg.sub_terrains.values())
    proportions = np.array([c.proportion for c in sub_cfgs])
    proportions /= proportions.sum()

    # Seed legacy np.random to the same state that was set before terrain
    # creation in the simulation.  The TerrainGenerator never seeds legacy
    # random itself, so the state flows through unchanged.
    saved_state = np.random.get_state()
    np.random.seed(TERRAIN_LEGACY_SEED)

    for index in range(num_rows * num_cols):
        sub_row, sub_col = np.unravel_index(index, (num_rows, num_cols))
        sub_index = int(np_rng.choice(len(proportions), p=proportions))
        difficulty = float(np_rng.uniform(*terrain_gen_cfg.difficulty_range))
        sub_cfg = sub_cfgs[sub_index]

        # Replicate TerrainGenerator._get_terrain_mesh config setup
        sub_cfg_copy = copy.deepcopy(sub_cfg)
        sub_cfg_copy.size = size
        sub_cfg_copy.horizontal_scale = h_scale
        sub_cfg_copy.vertical_scale = v_scale
        sub_cfg_copy.slope_threshold = terrain_gen_cfg.slope_threshold

        # The @height_field_to_mesh decorator shrinks cfg.size by a 1-pixel
        # border on each side before calling the unwrapped function.
        border_px = int(sub_cfg_copy.border_width / h_scale) + 1
        inner_w = int(sub_cfg_copy.size[0] / h_scale) + 1 - 2 * border_px
        inner_l = int(sub_cfg_copy.size[1] / h_scale) + 1 - 2 * border_px
        sub_cfg_copy.size = (inner_w * h_scale, inner_l * h_scale)

        # Call the REAL Isaac Lab terrain function (not a reimplementation).
        hf_inner = hf_terrains.discrete_obstacles_terrain.__wrapped__(
            difficulty, sub_cfg_copy
        )

        # Re-wrap into the bordered tile (matching the decorator output exactly)
        bordered = np.zeros((sub_w_px, sub_l_px), dtype=np.int16)
        bordered[border_px:border_px + hf_inner.shape[0],
                 border_px:border_px + hf_inner.shape[1]] = hf_inner

        if tunnel_mode:
            wt_px = int(1.0 / h_scale)
            wh_steps = int(10.0 / v_scale)
            ws_px = int(2.0 / h_scale)
            cz_px = int(4.0 / h_scale)
            # Apply to bordered tile (offset by border)
            bordered[0: border_px + ws_px + cz_px, :] = 0
            bordered[:, 0:border_px + wt_px] = wh_steps
            bordered[:, -(border_px + wt_px):] = wh_steps

        x0 = sub_col * sub_w_px
        y0 = sub_row * sub_l_px
        full_hf[x0:x0 + sub_w_px, y0:y0 + sub_l_px] = bordered

    np.random.set_state(saved_state)
    return full_hf


def extract_obstacles_from_heightfield(terrain_gen_cfg, tunnel_mode=False,
                                       flight_z: float = 4.0) -> tuple:
    """Extract **collision-relevant** obstacles from the heightfield.

    Only POSITIVE obstacles (pillars extending upward above the ground)
    are returned.  Negative heightfield values represent pits below ground
    level and do not constitute collision hazards for a drone flying at
    *flight_z*.

    Prefers the **captured** heightfield (recorded during actual terrain
    generation) for exact accuracy.  Falls back to
    :func:`regenerate_heightfield` for offline use (e.g. ``render_viz.py``).

    Args:
        terrain_gen_cfg: TerrainGeneratorCfg object.
        tunnel_mode: If True, regenerate with tunnel wall modifications
            (only used in fallback path).
        flight_z: Nominal flight altitude.  Obstacles whose peak is below
            this altitude are excluded (they cannot cause collisions).

    Returns:
        (heightfield, obstacles_list)
        where obstacles_list has dicts with 'center', 'radius', 'height', 'z_base'.
    """
    from scipy import ndimage

    # Prefer captured heightfield (exact match with simulation)
    hf = get_captured_heightfield(terrain_gen_cfg)
    if hf is None:
        # Fallback for offline / no-sim usage
        hf = regenerate_heightfield(terrain_gen_cfg, tunnel_mode=tunnel_mode)
    h_scale = terrain_gen_cfg.horizontal_scale
    v_scale = terrain_gen_cfg.vertical_scale
    size = terrain_gen_cfg.size
    # World origin: pixel 0 is at -size/2  (fence-post: N+1 pixels for N*h_scale metres)
    origin_x = -size[0] / 2.0
    origin_y = -size[1] / 2.0

    # Only extract POSITIVE obstacles (pillars).  Negative heightfield
    # values are pits below ground — harmless at the flight altitude.
    # Threshold: height must exceed flight_z (with a small margin) to be
    # a collision hazard.  We also use a minimum of 2m to skip flat ground
    # and border noise.
    min_height_steps = max(2.0 / v_scale, flight_z / v_scale)
    obstacle_mask = hf >= min_height_steps

    if not obstacle_mask.any():
        print("[WARN] No positive obstacles found in heightfield")
        return hf, []

    labeled, num_features = ndimage.label(obstacle_mask)
    obstacles = []

    for label_id in range(1, num_features + 1):
        xs, ys = np.where(labeled == label_id)
        if len(xs) < 4:
            continue

        cx = np.mean(xs) * h_scale + origin_x
        cy = np.mean(ys) * h_scale + origin_y

        x_extent = (np.max(xs) - np.min(xs) + 1) * h_scale
        y_extent = (np.max(ys) - np.min(ys) + 1) * h_scale
        radius = max(x_extent, y_extent) / 2.0 + 0.1

        raw_height = float(np.max(hf[labeled == label_id])) * v_scale

        obstacles.append({
            'center': (cx, cy),
            'radius': max(radius, 0.2),
            'height': raw_height,
            'z_base': 0.0,
        })

    return hf, obstacles
