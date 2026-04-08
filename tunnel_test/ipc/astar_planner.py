"""
3D A* path planner — Python port.

Ported from the C++ implementation in:
    slope_inspection/IPC/include/astar_rm.cpp
    slope_inspection/IPC/include/astar_rm.h

Original authors used this planner for real-world LiDAR quadrotor obstacle
avoidance (IPC module).  This port preserves the same algorithmic behaviour:
26-connected grid search, diagonal heuristic with tie-breaker, 0.05 s timeout,
and Floyd-style path simplification via line-of-sight raycasting.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Pre-computed constants used by the diagonal heuristic.
_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)


@dataclass
class AStarConfig:
    """Configuration for the A* planner."""

    resolution: float = 0.1       # Grid resolution in metres
    timeout: float = 0.05         # Search timeout in seconds
    tie_breaker: float = 1.00001  # Heuristic tie-breaking multiplier


# ── internal node ────────────────────────────────────────────────────────────

@dataclass
class _Node:
    """Lightweight bookkeeping for a single grid cell during search."""

    index: Tuple[int, int, int]
    g: float = math.inf
    f: float = math.inf
    parent: Optional[_Node] = field(default=None, repr=False)
    # 0 = unseen, 1 = open, -1 = closed
    status: int = 0
    round: int = -1


# ── 26-connected neighbour offsets ───────────────────────────────────────────

def _build_neighbours() -> List[Tuple[Tuple[int, int, int], float]]:
    """Return (offset, edge_cost) for 26-connectivity."""
    nbrs: List[Tuple[Tuple[int, int, int], float]] = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                cost = math.sqrt(i * i + j * j + k * k)
                nbrs.append(((i, j, k), cost))
    return nbrs


_NEIGHBOURS = _build_neighbours()


# ── planner ──────────────────────────────────────────────────────────────────

class AStarPlanner:
    """Grid-based 3-D A* planner with path simplification.

    The occupancy grid is not imported directly; any object that exposes the
    four duck-typed methods listed in ``search()`` is accepted.
    """

    def __init__(self, cfg: AStarConfig | None = None) -> None:
        self.cfg = cfg or AStarConfig()
        self._round = 0
        # (ix, iy, iz) -> _Node  — lazily allocated per search round
        self._nodes: Dict[Tuple[int, int, int], _Node] = {}

    # ── heuristic ────────────────────────────────────────────────────────

    def _heuristic(
        self,
        idx_a: Tuple[int, int, int],
        idx_b: Tuple[int, int, int],
    ) -> float:
        """Diagonal heuristic (case 3 in original C++ with tie-breaker).

        h = (√3 − √2)·d0 + (√2 − 1)·d1 + d2   where d0 ≤ d1 ≤ d2
        """
        d = sorted(
            (
                abs(idx_a[0] - idx_b[0]),
                abs(idx_a[1] - idx_b[1]),
                abs(idx_a[2] - idx_b[2]),
            )
        )
        h = (_SQRT3 - _SQRT2) * d[0] + (_SQRT2 - 1.0) * d[1] + d[2]
        return h * self.cfg.tie_breaker

    # ── node access ──────────────────────────────────────────────────────

    def _get_node(self, idx: Tuple[int, int, int]) -> _Node:
        """Return the node for *idx*, creating it lazily if needed."""
        node = self._nodes.get(idx)
        if node is None:
            node = _Node(index=idx)
            self._nodes[idx] = node
        if node.round != self._round:
            # Reset for the current search round (avoids full map wipe).
            node.g = math.inf
            node.f = math.inf
            node.parent = None
            node.status = 0
            node.round = self._round
        return node

    # ── main search ──────────────────────────────────────────────────────

    def search(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        occ_grid,
    ) -> Optional[np.ndarray]:
        """Search for a path from *start* to *goal*.

        Args:
            start: ``(3,)`` start position in world coordinates.
            goal:  ``(3,)`` goal position in world coordinates.
            occ_grid: Occupancy-grid object exposing:
                - ``world_to_grid(pos) -> (3,) int`` grid indices
                - ``grid_to_world(idx) -> (3,) float`` world position
                - ``is_free_inflate(pos) -> bool``
                - ``is_valid_index(idx) -> bool``

        Returns:
            ``(N, 3)`` numpy array of waypoints in world coordinates, or
            ``None`` if no path is found.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Reject trivially close start/goal (mirrors C++ guard).
        if np.linalg.norm(start - goal) <= self.cfg.resolution * 1.1:
            return None

        # Reject start or goal inside obstacles.
        if not occ_grid.is_free_inflate(start):
            return None
        if not occ_grid.is_free_inflate(goal):
            return None

        self._round += 1

        start_idx = tuple(int(v) for v in occ_grid.world_to_grid(start))
        goal_idx = tuple(int(v) for v in occ_grid.world_to_grid(goal))

        start_node = self._get_node(start_idx)
        start_node.g = 0.0
        start_node.f = self._heuristic(start_idx, goal_idx)
        start_node.status = 1  # open
        start_node.parent = None

        # Counter for stable heap ordering when f-scores are equal.
        counter = 0
        open_heap: List[Tuple[float, int, _Node]] = []
        heapq.heappush(open_heap, (start_node.f, counter, start_node))

        t0 = time.monotonic()
        timeout = self.cfg.timeout

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            # Skip stale entries (node already closed or superseded).
            if current.status == -1:
                continue
            current.status = -1  # close

            if current.index == goal_idx:
                return self._reconstruct(current, occ_grid)

            cx, cy, cz = current.index

            for (di, dj, dk), edge_cost in _NEIGHBOURS:
                ni, nj, nk = cx + di, cy + dj, cz + dk
                n_idx = (ni, nj, nk)

                if not occ_grid.is_valid_index(np.array(n_idx, dtype=int)):
                    continue

                n_pos = occ_grid.grid_to_world(np.array(n_idx, dtype=int))
                if not occ_grid.is_free_inflate(n_pos):
                    continue

                neighbor = self._get_node(n_idx)

                if neighbor.status == -1:
                    continue

                tentative_g = current.g + edge_cost

                if neighbor.status == 0:
                    # First visit — add to open set.
                    neighbor.g = tentative_g
                    neighbor.f = tentative_g + self._heuristic(n_idx, goal_idx)
                    neighbor.parent = current
                    neighbor.status = 1
                    counter += 1
                    heapq.heappush(open_heap, (neighbor.f, counter, neighbor))
                elif tentative_g < neighbor.g:
                    # Better path found — re-insert (lazy deletion).
                    neighbor.g = tentative_g
                    neighbor.f = tentative_g + self._heuristic(n_idx, goal_idx)
                    neighbor.parent = current
                    counter += 1
                    heapq.heappush(open_heap, (neighbor.f, counter, neighbor))

            if time.monotonic() - t0 > timeout:
                break

        return None

    # ── path reconstruction ──────────────────────────────────────────────

    @staticmethod
    def _reconstruct(node: _Node, occ_grid) -> np.ndarray:
        """Walk parent pointers and convert to world coordinates."""
        indices: List[Tuple[int, int, int]] = []
        cur: Optional[_Node] = node
        while cur is not None:
            indices.append(cur.index)
            cur = cur.parent
        indices.reverse()

        path = np.empty((len(indices), 3), dtype=float)
        for i, idx in enumerate(indices):
            path[i] = occ_grid.grid_to_world(np.array(idx, dtype=int))
        return path

    # ── path simplification ──────────────────────────────────────────────

    def simplify_path(self, path: np.ndarray, occ_grid) -> np.ndarray:
        """Remove intermediate waypoints if direct line-of-sight exists.

        Mirrors ``SimplifyPath`` + ``FloydHandle`` from the C++ source:
        1. Remove collinear intermediate points.
        2. Three passes of Floyd-style shortcutting via raycasting.
        """
        if path.shape[0] <= 1:
            return path.copy()

        # Step 1 — collinear removal (SimplifyPath in C++)
        waypoints = [path[0]]
        if path.shape[0] == 2:
            waypoints.append(path[1])
        else:
            vec_last = path[1] - path[0]
            for i in range(2, len(path)):
                vec = path[i] - path[i - 1]
                norms_product = np.linalg.norm(vec) * np.linalg.norm(vec_last)
                dot = vec.dot(vec_last)
                if abs(norms_product - dot) >= 1e-5:
                    waypoints.append(path[i - 1])
                    vec_last = vec
                else:
                    vec_last = vec
            waypoints.append(path[-1])

        # Step 2 — Floyd-style line-of-sight shortcutting (3 passes)
        for _ in range(3):
            i = len(waypoints) - 1
            while i > 0:
                shortened = False
                for j in range(0, i - 1):
                    if self._check_line_free(waypoints[i], waypoints[j], occ_grid):
                        del waypoints[j + 1 : i]
                        i = j
                        shortened = True
                        break
                if not shortened:
                    i -= 1

        return np.array(waypoints)

    # ── line-of-sight check ──────────────────────────────────────────────

    def _check_line_free(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        occ_grid,
    ) -> bool:
        """Return ``True`` if the straight line between *p1* and *p2* is free.

        Samples points at ``resolution`` spacing along the segment and queries
        the occupancy grid for each.
        """
        diff = p2 - p1
        dist = np.linalg.norm(diff)
        n_samples = max(int(dist / self.cfg.resolution), 1)
        for i in range(n_samples + 1):
            pt = p1 + diff * (i / n_samples)
            if not occ_grid.is_free_inflate(pt):
                return False
        return True
