"""
SFC Visualization — convert CIRI halfplane polytopes to drawable wireframes.

Provides :func:`polytope_to_edges` which takes an (K, 4) H-representation
polytope and returns a list of edge segments suitable for Isaac Sim's
``_debug_draw.draw_lines()`` interface.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

logger = logging.getLogger(__name__)


def _find_interior_point(planes: np.ndarray) -> Optional[np.ndarray]:
    """Find a strict interior point of the polytope via Chebyshev centre LP.

    Each row of *planes* (K, 4) is ``[a, b, c, d]`` with ``a*x+b*y+c*z+d <= 0``.

    Returns:
        (3,) interior point, or *None* if infeasible.
    """
    normals = planes[:, :3]
    offsets = planes[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-30)

    # LP: max t  s.t.  n_i^T p + ||n_i|| * t <= -d_i
    A_ub = np.hstack([normals, norms_safe])
    b_ub = -offsets
    c = np.array([0.0, 0.0, 0.0, -1.0])  # minimise -t

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 4,
                  method="highs")
    if res.success and res.x[3] > 1e-8:
        return res.x[:3]
    return None


def polytope_to_edges(
    planes: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert an H-representation polytope to a list of wireframe edges.

    Args:
        planes: (K, 4) halfplane constraints.  Each row ``[a, b, c, d]``
            defines ``a*x + b*y + c*z + d <= 0``.

    Returns:
        List of ``(start, end)`` pairs — each a (3,) world-frame point.
        Empty list on degenerate / infeasible polytopes.
    """
    if planes is None or planes.shape[0] < 4:
        return []

    interior = _find_interior_point(planes)
    if interior is None:
        return []

    try:
        hs = HalfspaceIntersection(planes, interior)
    except Exception:
        return []

    vertices = hs.intersections
    if len(vertices) < 4:
        return []

    try:
        hull = ConvexHull(vertices)
    except Exception:
        return []

    # Extract unique edges from triangular facets
    edges_set: set = set()
    for simplex in hull.simplices:
        n = len(simplex)
        for i in range(n):
            e = tuple(sorted((simplex[i], simplex[(i + 1) % n])))
            edges_set.add(e)

    result = []
    for i, j in edges_set:
        result.append((vertices[i].copy(), vertices[j].copy()))
    return result


def polytope_to_draw_data(
    planes: np.ndarray,
    color: Tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.6),
    line_width: float = 2.0,
) -> Tuple[list, list, list, list]:
    """Convert polytope to draw_lines()-compatible lists.

    Returns:
        ``(starts, ends, colors, sizes)`` ready for
        ``_debug_draw.draw_lines(starts, ends, colors, sizes)``.
    """
    edges = polytope_to_edges(planes)
    starts = [e[0].tolist() for e in edges]
    ends = [e[1].tolist() for e in edges]
    colors = [color] * len(edges)
    sizes = [line_width] * len(edges)
    return starts, ends, colors, sizes
