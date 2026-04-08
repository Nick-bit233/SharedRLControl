"""
CIRI (Convex Inflated Radii Iterative) safe flight corridor generator.

Python port of the C++ implementation from slope_inspection/IPC/include/ciri.cpp
Original copyright: MIT License, Zhepei Wang (wangzhepei@live.com), 2021.

Generates convex polytopes (safe flight corridors) around seed line segments,
ensuring the polytope is free of obstacles. The output is an H-representation
polytope — a matrix of halfplane coefficients (K, 4) where each row
[a, b, c, d] defines a*x + b*y + c*z + d <= 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CIRIConfig:
    """Parameters for the CIRI corridor generator."""
    robot_radius: float = 0.35
    iterations: int = 3
    epsilon: float = 1e-10


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------

class Ellipsoid:
    """Ellipsoid E = {x : ||C^{-1}(x - d)||_2 <= 1}.

    Parameterised either by shape matrix *C* (3x3) and centre *d* (3,), or by
    rotation *R* (3x3), semi-axis radii *r* (3,) and centre *d*, so that
    C = R @ diag(r) @ R^T.
    """

    def __init__(
        self,
        C: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        *,
        R: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
    ):
        if R is not None and r is not None:
            self._R = np.asarray(R, dtype=np.float64).copy()
            self._r = np.asarray(r, dtype=np.float64).copy()
            self._d = np.asarray(d, dtype=np.float64).copy()
            self._C = self._R @ np.diag(self._r) @ self._R.T
            # Analytical inverse: C_inv = R @ diag(1/r) @ R.T
            r_inv = 1.0 / np.maximum(self._r, 1e-30)
            self._C_inv = self._R @ np.diag(r_inv) @ self._R.T
        elif C is not None:
            self._C = np.asarray(C, dtype=np.float64).copy()
            self._d = np.asarray(d, dtype=np.float64).copy()
            U, S, _ = np.linalg.svd(self._C, full_matrices=True)
            if np.linalg.det(U) < 0.0:
                self._R = U[:, [1, 0, 2]]
                self._r = S[[1, 0, 2]]
            else:
                self._R = U
                self._r = S
            r_inv = 1.0 / np.maximum(self._r, 1e-30)
            self._C_inv = self._R @ np.diag(r_inv) @ self._R.T
        else:
            raise ValueError("Provide either (C, d) or (R, r, d).")

    # -- properties ----------------------------------------------------------

    @property
    def C_mat(self) -> np.ndarray:
        return self._C

    @property
    def C_inv(self) -> np.ndarray:
        return self._C_inv

    @property
    def d_vec(self) -> np.ndarray:
        return self._d

    @property
    def R_mat(self) -> np.ndarray:
        return self._R

    @property
    def r_vec(self) -> np.ndarray:
        return self._r

    # -- coordinate transforms -----------------------------------------------

    def point_to_ellipsoid(self, pt_w: np.ndarray) -> np.ndarray:
        """Transform world-frame point(s) to ellipsoid frame.

        Args:
            pt_w: (3,) single point **or** (3, N) point cloud.

        Returns:
            Same shape as input, in ellipsoid coordinates.
        """
        if pt_w.ndim == 1:
            return self._C_inv @ (pt_w - self._d)
        # (3, N)
        return self._C_inv @ (pt_w - self._d[:, None])

    def plane_to_ellipsoid(self, plane_w: np.ndarray) -> np.ndarray:
        """Transform world-frame halfplane(s) to ellipsoid frame.

        A single plane is (4,).  A batch of planes is (M, 4).
        plane = [n0, n1, n2, d]  with n·x + d <= 0.

        Transformation: n_e = C^T @ n,  d_e = n^T @ d_vec + d_plane.
        (No normalisation — matches C++ ``Ellipsoid::toEllipsoidFrame``.)
        """
        if plane_w.ndim == 1:
            n_e = self._C.T @ plane_w[:3]
            d_e = plane_w[:3] @ self._d + plane_w[3]
            return np.array([n_e[0], n_e[1], n_e[2], d_e])
        # (M, 4)
        n_e = plane_w[:, :3] @ self._C          # (M, 3)
        d_e = plane_w[:, :3] @ self._d + plane_w[:, 3]  # (M,)
        out = np.empty_like(plane_w)
        out[:, :3] = n_e
        out[:, 3] = d_e
        return out

    def plane_to_world(self, plane_e: np.ndarray) -> np.ndarray:
        """Transform ellipsoid-frame halfplane(s) to world frame.

        Inverse of :meth:`plane_to_ellipsoid`.
        n_w = C^{-T} @ n_e,  d_w = d_e - n_w^T @ d_vec.
        """
        if plane_e.ndim == 1:
            n_w = self._C_inv.T @ plane_e[:3]
            d_w = plane_e[3] - n_w @ self._d
            return np.array([n_w[0], n_w[1], n_w[2], d_w])
        n_w = plane_e[:, :3] @ self._C_inv      # (M, 3)
        d_w = plane_e[:, 3] - (n_w * self._d[None, :]).sum(axis=1)
        out = np.empty_like(plane_e)
        out[:, :3] = n_w
        out[:, 3] = d_w
        return out

    def point_to_world(self, pt_e: np.ndarray) -> np.ndarray:
        """Transform ellipsoid-frame point(s) to world frame."""
        if pt_e.ndim == 1:
            return self._C @ pt_e + self._d
        return self._C @ pt_e + self._d[:, None]

    def dist(self, pt_w: np.ndarray) -> np.ndarray:
        """Ellipsoidal distance for (3,N) world-frame points."""
        return np.linalg.norm(self.point_to_ellipsoid(pt_w), axis=0)

    def inside(self, pt_w: np.ndarray) -> bool:
        """Check if a single world-frame point is inside the ellipsoid."""
        return float(np.linalg.norm(self._C_inv @ (pt_w - self._d))) <= 1.0

    def points_inside(
        self, pc_w: np.ndarray
    ) -> Tuple[bool, np.ndarray, int]:
        """Return (has_inside, inside_pts (3,K), min_dist_id_in_inside)."""
        dists = np.linalg.norm(self._C_inv @ (pc_w - self._d[:, None]), axis=0)
        mask = dists <= 1.0
        if not np.any(mask):
            return False, np.empty((3, 0)), -1
        inside_pts = pc_w[:, mask]
        inside_dists = dists[mask]
        min_id = int(np.argmin(inside_dists))
        return True, inside_pts, min_id


# ---------------------------------------------------------------------------
# MVIE helpers (simplified)
# ---------------------------------------------------------------------------

def _find_chebyshev_center(hPoly: np.ndarray) -> Optional[np.ndarray]:
    """Find the Chebyshev centre of a polytope {x : A x + b <= 0}.

    Each row of *hPoly* (K, 4) is [a, b, c, d] with a*x+b*y+c*z+d <= 0.
    The Chebyshev centre maximises the distance to the nearest halfplane, i.e.
    we solve::

        max  t
        s.t. n_i^T p + d_i + ||n_i|| t <= 0   for all i

    This is a small LP with 4 variables [p (3), t (1)].

    Returns:
        (3,) interior point, or *None* if infeasible / unbounded.
    """
    M = hPoly.shape[0]
    normals = hPoly[:, :3]
    offsets = hPoly[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (M, 1)

    # LP variables: [p0, p1, p2, t]
    # Constraints: n_i^T p + ||n_i|| * t <= -d_i
    A_ub = np.hstack([normals, norms])           # (M, 4)
    b_ub = -offsets                               # (M,)
    c = np.array([0.0, 0.0, 0.0, -1.0])          # maximise t => minimise -t

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 4,
                  method="highs")
    if res.success and res.x[3] > 0.0:
        return res.x[:3]
    return None


def _mvie_simplified(
    hPoly: np.ndarray, prev_ell: Ellipsoid
) -> Optional[Ellipsoid]:
    """Compute a simplified MVIE (Maximum Volume Inscribed Ellipsoid).

    Uses the L-BFGS formulation from the C++ ``MVIE::maxVolInsEllipsoid``.
    The optimisation maximises log(det(L)) (= volume proxy) subject to
    penalty-smoothed polytope containment constraints.

    Falls back to a Chebyshev-centre ellipsoid if the optimisation fails.

    Args:
        hPoly: (M, 4) halfplane polytope.
        prev_ell: Previous ellipsoid (used as initial guess).

    Returns:
        New :class:`Ellipsoid`, or *None* on failure.
    """
    from scipy.optimize import minimize

    M = hPoly.shape[0]
    if M == 0:
        return None

    # -- Step 1: find the deepest interior point via Chebyshev-centre LP -----
    interior = _find_chebyshev_center(hPoly)
    if interior is None:
        return None

    # Normalise half-planes for numerical stability
    normals = hPoly[:, :3]
    offsets = hPoly[:, 3]
    h_norms = np.linalg.norm(normals, axis=1)
    A_norm = normals / h_norms[:, None]                 # (M, 3)
    b_norm = -offsets / h_norms                          # (M,)

    # Transform so that interior is at origin:  A_bar = A_norm / (b_norm - A_norm @ interior)
    margins = b_norm - A_norm @ interior                 # (M,)
    if np.any(margins <= 0):
        return None
    A_bar = A_norm / margins[:, None]                    # (M, 3)

    # -- Step 2: initial guess from previous ellipsoid -----------------------
    R0 = prev_ell.R_mat
    r0 = prev_ell.r_vec
    Q = R0 @ np.diag(r0 ** 2) @ R0.T
    try:
        L0 = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        L0 = np.diag(r0)

    # Pack: x = [p(3), sqrt_diag(3), off_diag(3)]
    p0 = prev_ell.d_vec - interior
    x0 = np.zeros(9)
    x0[:3] = p0
    x0[3] = np.sqrt(max(L0[0, 0], 1e-12))
    x0[4] = np.sqrt(max(L0[1, 1], 1e-12))
    x0[5] = np.sqrt(max(L0[2, 2], 1e-12))
    x0[6] = L0[1, 0]
    x0[7] = L0[2, 1]
    x0[8] = L0[2, 0]

    smooth_eps = 1e-2
    penalty_wt = 1e3

    def _smoothed_l1(mu: float, val: float):
        if val < 0.0:
            return None, None
        if val > mu:
            return val - 0.5 * mu, 1.0
        xdmu = val / mu
        sq = xdmu * xdmu
        return (mu - 0.5 * val) * sq * xdmu, sq * (-0.5 * xdmu + 3.0 * (mu - 0.5 * val) / mu)

    def _cost_and_grad(x):
        p = x[:3]
        rtd = x[3:6]
        cde = x[6:9]

        L = np.zeros((3, 3))
        L[0, 0] = rtd[0] ** 2 + 1e-16
        L[1, 0] = cde[0]
        L[1, 1] = rtd[1] ** 2 + 1e-16
        L[2, 0] = cde[2]
        L[2, 1] = cde[1]
        L[2, 2] = rtd[2] ** 2 + 1e-16

        AL = A_bar @ L             # (M, 3)
        norm_AL = np.linalg.norm(AL, axis=1)   # (M,)
        norm_AL_safe = np.maximum(norm_AL, 1e-30)
        adj_norm_AL = (AL / norm_AL_safe[:, None]).T  # (3, M)
        cons_viola = norm_AL + A_bar @ p - 1.0        # (M,)

        cost = 0.0
        gdp = np.zeros(3)
        gdrtd = np.zeros(3)
        gdcde = np.zeros(3)

        for i in range(M):
            f, df = _smoothed_l1(smooth_eps, cons_viola[i])
            if f is None:
                continue
            cost += f
            vec = df * A_bar[i]
            gdp += vec
            gdrtd += adj_norm_AL[:, i] * vec
            gdcde[0] += adj_norm_AL[0, i] * vec[1]
            gdcde[1] += adj_norm_AL[1, i] * vec[2]
            gdcde[2] += adj_norm_AL[0, i] * vec[2]

        cost *= penalty_wt
        gdp *= penalty_wt
        gdrtd *= penalty_wt
        gdcde *= penalty_wt

        cost -= np.log(L[0, 0]) + np.log(L[1, 1]) + np.log(L[2, 2])
        gdrtd[0] -= 1.0 / L[0, 0]
        gdrtd[1] -= 1.0 / L[1, 1]
        gdrtd[2] -= 1.0 / L[2, 2]

        gdrtd[0] *= 2.0 * rtd[0]
        gdrtd[1] *= 2.0 * rtd[1]
        gdrtd[2] *= 2.0 * rtd[2]

        grad = np.concatenate([gdp, gdrtd, gdcde])
        return cost, grad

    def _cost(x):
        c, _ = _cost_and_grad(x)
        return c

    def _jac(x):
        _, g = _cost_and_grad(x)
        return g

    res = minimize(_cost, x0, jac=_jac, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-8, "gtol": 1e-6})

    x = res.x
    p_opt = x[:3] + interior
    rtd = x[3:6]
    cde = x[6:9]

    L = np.zeros((3, 3))
    L[0, 0] = rtd[0] ** 2
    L[1, 0] = cde[0]
    L[1, 1] = rtd[1] ** 2
    L[2, 0] = cde[2]
    L[2, 1] = cde[1]
    L[2, 2] = rtd[2] ** 2

    U, S, _ = np.linalg.svd(L, full_matrices=True)
    if np.linalg.det(U) < 0.0:
        R_out = U[:, [1, 0, 2]]
        r_out = S[[1, 0, 2]]
    else:
        R_out = U
        r_out = S

    r_out = np.maximum(r_out, 1e-12)
    return Ellipsoid(R=R_out, r=r_out, d=p_opt)


# ---------------------------------------------------------------------------
# CIRI Corridor Generator
# ---------------------------------------------------------------------------

class CIRICorridor:
    """CIRI safe flight corridor generator.

    Port of ``ciri::CIRI`` from the C++ codebase.
    """

    def __init__(self, cfg: Optional[CIRIConfig] = None):
        self.cfg = cfg or CIRIConfig()

    # -- public API ----------------------------------------------------------

    def generate(
        self,
        boundary: np.ndarray,
        obstacles: np.ndarray,
        seed_a: np.ndarray,
        seed_b: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Generate a safe flight corridor around a seed line segment.

        Args:
            boundary: (M, 4) halfplane constraints defining the bounding
                domain.  Each row ``[a, b, c, d]``: ``a*x+b*y+c*z+d <= 0``.
            obstacles: Obstacle points — accepted as (3, N) **or** (N, 3).
                Transposed internally to (3, N) if needed.
            seed_a: (3,) start of seed line segment.
            seed_b: (3,) end of seed line segment.

        Returns:
            ``(K, 4)`` halfplane array defining the corridor, or ``None`` on
            failure.  Each row ``[a, b, c, d]``: ``a*x+b*y+c*z+d <= 0``.
        """
        eps = self.cfg.epsilon
        boundary = np.asarray(boundary, dtype=np.float64)
        seed_a = np.asarray(seed_a, dtype=np.float64).ravel()
        seed_b = np.asarray(seed_b, dtype=np.float64).ravel()

        # Normalise obstacle shape to (3, N)
        obstacles = np.asarray(obstacles, dtype=np.float64)
        if obstacles.size == 0:
            obstacles = np.empty((3, 0), dtype=np.float64)
        elif obstacles.ndim == 1:
            obstacles = obstacles.reshape(3, 1)
        elif obstacles.shape[0] != 3:
            obstacles = obstacles.T

        # Check seed inside boundary
        ah = np.append(seed_a, 1.0)
        bh = np.append(seed_b, 1.0)
        if boundary.size > 0:
            if (boundary @ ah).max() > eps or (boundary @ bh).max() > eps:
                logger.warning("Seed points not inside boundary — aborting.")
                return None

        M = boundary.shape[0]
        N = obstacles.shape[1]

        # Initial ellipsoid centred on seed midpoint
        E = Ellipsoid(C=np.eye(3), d=(seed_a + seed_b) / 2.0)
        if np.linalg.norm(seed_a - seed_b) > 0.1:
            E = self._find_ellipsoid(obstacles, seed_a, seed_b)

        hPoly: Optional[np.ndarray] = None

        for loop in range(self.cfg.iterations):
            fwd_a = E.point_to_ellipsoid(seed_a)
            fwd_b = E.point_to_ellipsoid(seed_b)

            bd_e = E.plane_to_ellipsoid(boundary) if M > 0 else np.empty((0, 4))

            # Boundary distances in ellipsoid frame
            if M > 0:
                bd_norms = np.linalg.norm(bd_e[:, :3], axis=1)
                bd_norms_safe = np.maximum(bd_norms, 1e-30)
                distDs = np.abs(bd_e[:, 3]) / bd_norms_safe
            else:
                distDs = np.empty(0)

            # Obstacle distances in ellipsoid frame
            if N > 0:
                pc_e = E.point_to_ellipsoid(obstacles)
                distRs = np.linalg.norm(pc_e, axis=0).copy()
            else:
                pc_e = np.empty((3, 0))
                distRs = np.empty(0)

            bd_flags = np.ones(M, dtype=bool)
            pc_flags = np.ones(N, dtype=bool)

            planes: list[np.ndarray] = []

            # Find initial closest
            bdMinId = int(distDs.argmin()) if M > 0 else -1
            minSqrD = distDs[bdMinId] if M > 0 else np.inf
            pcMinId = int(distRs.argmin()) if N > 0 else -1
            minSqrR = distRs[pcMinId] if N > 0 else np.inf

            # Pre-compute for vectorised culling
            obs_normal_thresh = self.cfg.robot_radius - eps

            completed = False

            for _ in range(M + N):
                if completed:
                    break

                if minSqrD < minSqrR:
                    p_e = bd_e[bdMinId]
                    temp_plane_w = E.plane_to_world(p_e)
                    bd_flags[bdMinId] = False
                else:
                    if self.cfg.robot_radius < eps:
                        temp_plane_w = self._tangent_plane_no_radius(
                            E, pc_e, distRs, pcMinId, fwd_a, fwd_b, eps,
                        )
                    else:
                        temp_plane_w = self._tangent_plane_with_radius(
                            E, obstacles, pc_e, pcMinId,
                            seed_a, seed_b, eps,
                        )
                    pc_flags[pcMinId] = False

                # Vectorised boundary update
                completed = True
                if M > 0:
                    active_bd = np.where(bd_flags)[0]
                    if active_bd.size > 0:
                        completed = False
                        bdMinId = active_bd[int(distDs[active_bd].argmin())]
                        minSqrD = distDs[bdMinId]
                    else:
                        minSqrD = np.inf

                # Vectorised obstacle culling and min-distance update
                if N > 0:
                    active_pc = np.where(pc_flags)[0]
                    if active_pc.size > 0:
                        vals = temp_plane_w[:3] @ obstacles[:, active_pc] + temp_plane_w[3]
                        cull_mask = vals > obs_normal_thresh
                        pc_flags[active_pc[cull_mask]] = False
                        remaining = active_pc[~cull_mask]
                        if remaining.size > 0:
                            completed = False
                            best = int(distRs[remaining].argmin())
                            pcMinId = remaining[best]
                            minSqrR = distRs[pcMinId]
                        else:
                            minSqrR = np.inf
                    else:
                        minSqrR = np.inf

                planes.append(temp_plane_w.copy())

            hPoly = np.array(planes) if planes else np.empty((0, 4))

            # Skip MVIE on last iteration
            if loop == self.cfg.iterations - 1:
                break

            new_E = _mvie_simplified(hPoly, E)
            if new_E is None:
                logger.warning("MVIE failed at iteration %d.", loop)
                break
            E = new_E

        if hPoly is None or hPoly.size == 0:
            return None

        # Sanity check
        if np.any(np.isnan(hPoly)):
            logger.warning("NaN in generated planes — aborting.")
            return None

        # Verify the polytope has an interior
        interior = _find_chebyshev_center(hPoly)
        if interior is None:
            logger.warning("Generated polytope is empty — aborting.")
            return None

        return hPoly

    # -- private helpers -----------------------------------------------------

    def _find_ellipsoid(
        self,
        pc: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> Ellipsoid:
        """Find an initial ellipsoid aligned with the seed line segment.

        Matches ``CIRI::findEllipsoid`` in the C++ source.  The ellipsoid is
        oriented along (b - a), then shrunk iteratively until no obstacle
        points lie inside.
        """
        robot_r = self.cfg.robot_radius
        eps = self.cfg.epsilon

        f = np.linalg.norm(a - b) / 2.0
        r = np.array([f, f, f])
        centre = (a + b) / 2.0

        r[0] += robot_r
        if r[0] > 0:
            ratio = r[1] / r[0]
            r *= ratio

        # Rotation aligning X-axis to (b - a)
        direction = b - a
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12:
            Ri = np.eye(3)
        else:
            Ri = Rotation.align_vectors(
                [direction / direction_norm], [[1.0, 0.0, 0.0]]
            )[0].as_matrix()

        E = Ellipsoid(R=Ri, r=r.copy(), d=centre)
        Rf = Ri.copy()

        # Phase 1: shrink r[1] (y-radius)
        has_inside, obs_inside, min_id = E.points_inside(pc)
        if not has_inside:
            return E

        pw = obs_inside[:, min_id]
        for _ in range(100):
            p_e = Ri.T @ (pw - E.d_vec)
            roll = np.arctan2(p_e[2], p_e[1])
            cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
            q_roll = Rotation.from_quat([sr, 0.0, 0.0, cr])
            Rf = Ri @ q_roll.as_matrix()

            p_e = Rf.T @ (pw - E.d_vec)
            if abs(p_e[0]) < r[0]:
                denom = 1.0 - (p_e[0] / r[0]) ** 2
                if denom > eps:
                    r[1] = abs(p_e[1]) / np.sqrt(denom)

            E = Ellipsoid(R=Rf, r=r.copy(), d=centre)
            has_inside, obs_inside, min_id = E.points_inside(obs_inside)
            if not has_inside:
                break
            pw = obs_inside[:, min_id]

        # Phase 2: shrink r[2] (z-radius) — restart from full cloud
        has_inside, obs_inside, min_id = E.points_inside(pc)
        if not has_inside:
            return E

        pw = obs_inside[:, min_id]
        for _ in range(100):
            p = Rf.T @ (pw - E.d_vec)
            denom = 1.0 - (p[0] / r[0]) ** 2 - (p[1] / max(r[1], 1e-30)) ** 2
            if denom > eps:
                r[2] = abs(p[2]) / np.sqrt(denom)

            E = Ellipsoid(R=Rf, r=r.copy(), d=centre)
            has_inside, obs_inside, min_id = E.points_inside(obs_inside)
            if not has_inside:
                break
            pw = obs_inside[:, min_id]

        return Ellipsoid(R=Rf, r=r.copy(), d=centre)

    @staticmethod
    def _tangent_plane_no_radius(
        E: Ellipsoid,
        pc_e: np.ndarray,
        distRs: np.ndarray,
        idx: int,
        fwd_a: np.ndarray,
        fwd_b: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Tangent plane at an obstacle point (zero robot radius).

        Mirrors the ``robot_r_ < epsilon_`` branch in the C++ source.
        """
        pt_e = pc_e[:, idx]
        d_norm = distRs[idx]
        d_norm_safe = max(d_norm, 1e-30)

        tangent = np.empty(4)
        tangent[:3] = pt_e / d_norm_safe
        tangent[3] = -d_norm

        # Adjust if seed point *a* is on the wrong side
        if tangent[:3] @ fwd_a + tangent[3] > eps:
            delta = pt_e - fwd_a
            dsq = delta @ delta
            if dsq > 1e-30:
                tangent[:3] = fwd_a - (delta @ fwd_a / dsq) * delta
            d_norm = np.linalg.norm(tangent[:3])
            d_norm_safe = max(d_norm, 1e-30)
            tangent[3] = -d_norm
            tangent[:3] /= d_norm_safe
            distRs[idx] = d_norm

        # Adjust for seed point *b* (applied twice in the original code)
        for _ in range(2):
            if tangent[:3] @ fwd_b + tangent[3] > eps:
                delta = pt_e - fwd_b
                dsq = delta @ delta
                if dsq > 1e-30:
                    tangent[:3] = fwd_b - (delta @ fwd_b / dsq) * delta
                d_norm = np.linalg.norm(tangent[:3])
                d_norm_safe = max(d_norm, 1e-30)
                tangent[3] = -d_norm
                tangent[:3] /= d_norm_safe
                distRs[idx] = d_norm

        return E.plane_to_world(tangent)

    def _tangent_plane_with_radius(
        self,
        E: Ellipsoid,
        pc_w: np.ndarray,
        pc_e: np.ndarray,
        idx: int,
        seed_a: np.ndarray,
        seed_b: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Tangent plane at an obstacle with non-zero robot radius.

        Mirrors the ``robot_r_ >= epsilon_`` branch in the C++ source.
        Uses a sphere-inflated obstacle representation.
        """
        pt_e = pc_e[:, idx]
        pt_w = pc_w[:, idx]
        robot_r = self.cfg.robot_radius
        C_inv = E.C_inv

        # Sphere template scaled by robot_radius, transformed to ellipsoid frame
        sphere_C = robot_r * np.eye(3)
        E_pe = Ellipsoid(C=C_inv @ sphere_C, d=pt_e)

        # Closest point on the inflated-obstacle ellipsoid to origin
        close_pt_e = self._closest_point_on_ellipsoid(E_pe, np.zeros(3))
        c_pt_w = E.point_to_world(close_pt_e)

        diff = pt_w - c_pt_w
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-30:
            n_w = np.array([0.0, 0.0, 1.0])
        else:
            n_w = diff / diff_norm

        temp_plane_w = np.empty(4)
        temp_plane_w[:3] = n_w
        temp_plane_w[3] = -n_w @ c_pt_w

        # Ensure seed points stay inside
        if n_w @ seed_a + temp_plane_w[3] > -eps:
            self._find_tangent_plane_of_sphere(
                pt_w, robot_r, seed_a, E.d_vec, temp_plane_w, eps,
            )
        elif n_w @ seed_b + temp_plane_w[3] > -eps:
            self._find_tangent_plane_of_sphere(
                pt_w, robot_r, seed_b, E.d_vec, temp_plane_w, eps,
            )

        return temp_plane_w

    @staticmethod
    def _closest_point_on_ellipsoid(
        ell: Ellipsoid, query: np.ndarray
    ) -> np.ndarray:
        """Approximate closest point on *ell*'s surface to *query*.

        Uses an iterative projection approach.
        """
        R = ell.R_mat
        r = np.maximum(ell.r_vec, 1e-12)
        d = ell.d_vec

        q_local = R.T @ (query - d)

        # Newton iterations for distance to axis-aligned ellipsoid
        # Start from the normalised projection
        p = q_local.copy()
        for _ in range(50):
            on_ell = p / r
            val = np.sum(on_ell ** 2) - 1.0
            if abs(val) < 1e-12:
                break
            grad = 2.0 * p / (r ** 2)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-30:
                break
            n = grad / grad_norm
            # Project onto surface along the gradient direction
            step = val / (2.0 * np.sum(p ** 2 / r ** 4) / grad_norm)
            p = p - step * n

        # Final normalisation to the surface
        pn = p / r
        pn_norm = np.linalg.norm(pn)
        if pn_norm > 1e-30:
            p = r * (pn / pn_norm)

        return R @ p + d

    @staticmethod
    def _find_tangent_plane_of_sphere(
        centre: np.ndarray,
        r: float,
        pass_point: np.ndarray,
        seed_p: np.ndarray,
        out_plane: np.ndarray,
        eps: float,
    ) -> None:
        """Compute tangent plane of a sphere that passes through *pass_point*
        and keeps *seed_p* on the correct side.

        Mirrors ``CIRI::findTangentPlaneOfSphere`` in the C++ source.
        Modifies *out_plane* in place.
        """
        seed = seed_p.copy()
        dif = pass_point - pass_point  # NOTE: zero vector (matches C++ bug)
        if np.linalg.norm(dif) < 1e-3:
            diff_pc = pass_point - centre
            if np.linalg.norm(diff_pc[:2]) > 1e-3:
                v1 = diff_pc.copy()
                v1[2] = 0.0
                v1 /= max(np.linalg.norm(v1), 1e-30)
                cross = np.cross(v1, np.array([0.0, 0.0, 1.0]))
                cn = np.linalg.norm(cross)
                if cn > 1e-30:
                    seed = seed_p + 0.01 * cross / cn
            else:
                cross = np.cross(diff_pc, np.array([1.0, 0.0, 0.0]))
                cn = np.linalg.norm(cross)
                if cn > 1e-30:
                    seed = seed_p + 0.01 * cross / cn

        P = pass_point - centre
        norm_vec = np.cross(P, seed - centre)
        nn = np.linalg.norm(norm_vec)
        if nn < 1e-30:
            return
        norm_vec /= nn

        # Rotation aligning norm_vec -> Z-axis
        R = Rotation.align_vectors(
            [[0.0, 0.0, 1.0]], [norm_vec]
        )[0].as_matrix()

        P = R @ P
        C = R @ (seed - centre)

        r2 = r * r
        p1p2n = P[0] ** 2 + P[1] ** 2
        if p1p2n <= r2:
            return
        d_val = np.sqrt(p1p2n - r2)
        rp1p2n = r / p1p2n

        q11 = rp1p2n * (P[0] * r - P[1] * d_val)
        q21 = rp1p2n * (P[1] * r + P[0] * d_val)
        q12 = rp1p2n * (P[0] * r + P[1] * d_val)
        q22 = rp1p2n * (P[1] * r - P[0] * d_val)

        Q = np.zeros(3)
        if q11 * C[0] + q21 * C[1] < 0:
            Q[0] = q12
            Q[1] = q22
        else:
            Q[0] = q11
            Q[1] = q21

        out_plane[:3] = R.T @ Q
        Q_w = out_plane[:3] + centre
        out_plane[3] = -Q_w @ out_plane[:3]

        if out_plane[:3] @ seed + out_plane[3] > eps:
            out_plane *= -1.0
