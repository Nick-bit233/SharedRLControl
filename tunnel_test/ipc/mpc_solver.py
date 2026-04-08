"""
MPC Trajectory Optimizer — ported from slope_inspection/IPC/include/mpc.cpp

QP formulation:
    min 1/2 * u^T H u + f^T u
    subject to  Alow <= A_sys @ u <= Aupp   (system: input/accel/vel bounds)
                T @ A_p @ u <= -D_T - T @ B_p   (SFC: safe flight corridor)

State vector (9D per step): [px, py, pz, vx, vy, vz, ax, ay, az]
Input vector (3D per step): [ux, uy, uz]  (jerk)
"""

import numpy as np
import osqp
from scipy import sparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MPCConfig:
    horizon: int = 15
    step: float = 0.1  # seconds per MPC step → 1.5s total horizon

    # Cost weights
    R_p: float = 200.0   # position tracking
    R_v: float = 0.0     # velocity tracking
    R_a: float = 0.0     # acceleration tracking
    R_u: float = 0.1     # input smoothness
    R_u_con: float = 0.2 # input continuity
    R_pN: float = 200.0  # terminal position
    R_vN: float = 100.0  # terminal velocity
    R_aN: float = 100.0  # terminal acceleration

    # Drag coefficients
    D_x: float = 0.0
    D_y: float = 0.0
    D_z: float = 0.0

    # Velocity bounds
    v_min: np.ndarray = field(default_factory=lambda: np.array([-10.0, -10.0, -10.0]))
    v_max: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0]))

    # Acceleration bounds
    a_min: np.ndarray = field(default_factory=lambda: np.array([-20.0, -20.0, -20.0]))
    a_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0]))

    # Jerk (input) bounds
    u_min: np.ndarray = field(default_factory=lambda: np.array([-50.0, -50.0, -50.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 50.0]))


class MPCSolver:
    """
    Model Predictive Control solver for 3D trajectory optimization.

    Ported from slope_inspection/IPC/include/mpc.cpp (MPCClass).
    Uses OSQP to solve the resulting QP at each control step.
    """

    def __init__(self, cfg: Optional[MPCConfig] = None):
        self.cfg = cfg or MPCConfig()
        N = self.cfg.horizon
        dt = self.cfg.step

        # Drag matrix
        self.Drag = np.diag([self.cfg.D_x, self.cfg.D_y, self.cfg.D_z])

        # Build time-invariant matrices
        self.Ax, self.Bx = self._system_model(dt)
        self.M, self.C = self._mpc_model(self.Ax, self.Bx, N)

        # Cost matrices (quadratic term — constant)
        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = np.eye(3) * self.cfg.R_p
        Q[3:6, 3:6] = np.eye(3) * self.cfg.R_v
        Q[6:9, 6:9] = np.eye(3) * self.cfg.R_a

        F = np.zeros((9, 9))
        F[0:3, 0:3] = np.eye(3) * self.cfg.R_pN
        F[3:6, 3:6] = np.eye(3) * self.cfg.R_vN
        F[6:9, 6:9] = np.eye(3) * self.cfg.R_aN

        R = np.eye(3) * self.cfg.R_u
        R_con = np.eye(3) * self.cfg.R_u_con

        self.H = self._quadratic_term(self.M, self.C, Q, R, R_con, F, N)

        # Extract p/v/a sub-matrices from M and C (for constraints and SFC)
        self.A_p = np.zeros((3 * N, self.C.shape[1]))
        self.A_v = np.zeros((3 * N, self.C.shape[1]))
        self.A_a = np.zeros((3 * N, self.C.shape[1]))
        self.M_p = np.zeros((3 * N, self.M.shape[1]))
        self.M_v = np.zeros((3 * N, self.M.shape[1]))
        self.M_a = np.zeros((3 * N, self.M.shape[1]))

        for i in range(N):
            self.A_p[3*i:3*i+3, :] = self.C[9*i+0:9*i+3, :]
            self.A_v[3*i:3*i+3, :] = self.C[9*i+3:9*i+6, :]
            self.A_a[3*i:3*i+3, :] = self.C[9*i+6:9*i+9, :]
            self.M_p[3*i:3*i+3, :] = self.M[9*i+0:9*i+3, :]
            self.M_v[3*i:3*i+3, :] = self.M[9*i+3:9*i+6, :]
            self.M_a[3*i:3*i+3, :] = self.M[9*i+6:9*i+9, :]

        # Build the time-invariant system constraint matrix A_sys (input + accel + vel)
        n_u = 3 * N
        A_u = np.eye(n_u)
        self.A_sys = np.vstack([A_u, self.A_a, self.A_v])  # (9*N, 3*N)

        # Replicate per-step bounds
        self._u_low = np.tile(self.cfg.u_min, N)
        self._u_upp = np.tile(self.cfg.u_max, N)
        self._a_low = np.tile(self.cfg.a_min, N)
        self._a_upp = np.tile(self.cfg.a_max, N)
        self._v_low = np.tile(self.cfg.v_min, N)
        self._v_upp = np.tile(self.cfg.v_max, N)

        # Pre-compute Q_bar @ C for linear term
        Q_bar = np.zeros((9 * N, 9 * N))
        for i in range(N):
            Q_bar[9*i:9*i+9, 9*i:9*i+9] = Q
        Q_bar[9*(N-1):9*N, 9*(N-1):9*N] = F
        self._QC = Q_bar @ self.C  # (9N, 3N)

        # Working state
        self.X_0 = np.zeros(9)
        self.X_r = np.zeros(9 * N)
        self.u_optimal = np.zeros(3 * N)
        self._sfc_planes = {}  # step_index → (K, 4) planes

        # Cached OSQP solver for warm-starting
        self._cached_solver = None
        self._cached_n_constraints = 0

    # ---- Public API ----

    def set_status(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
        """Set current drone state [p, v, a] (each 3D)."""
        vel = np.clip(vel, self.cfg.v_min, self.cfg.v_max)
        acc = np.clip(acc, self.cfg.a_min, self.cfg.a_max)
        self.X_0[:3] = pos
        self.X_0[3:6] = vel
        self.X_0[6:9] = acc

    def set_goal(self, pos_r: np.ndarray, vel_r: np.ndarray, acc_r: np.ndarray, step: int):
        """Set reference state for a given MPC horizon step."""
        if step < 0 or step >= self.cfg.horizon:
            return
        idx = step * 9
        self.X_r[idx:idx+3] = pos_r
        self.X_r[idx+3:idx+6] = vel_r
        self.X_r[idx+6:idx+9] = acc_r

    def set_sfc(self, planes: np.ndarray, step: int):
        """
        Set Safe Flight Corridor constraints for a given MPC step.

        Args:
            planes: (K, 4) array, each row [a, b, c, d] defines a half-space a*x+b*y+c*z+d <= 0.
            step: MPC horizon step index.
        """
        if planes.shape[0] == 0 or step < 0 or step >= self.cfg.horizon:
            return
        self._sfc_planes[step] = planes.copy()

    def clear_sfc(self):
        """Clear all SFC constraints."""
        self._sfc_planes.clear()

    def solve(self) -> bool:
        """Solve the MPC QP. Returns True on success."""
        N = self.cfg.horizon

        # Linear term: f = ((x0^T M^T - x_r^T) Q_bar C)^T
        f = (self.X_0 @ self.M.T - self.X_r) @ self._QC  # (3N,)

        # System constraints bounds (depend on current state)
        B_p = self.M_p @ self.X_0
        B_v = self.M_v @ self.X_0
        B_a = self.M_a @ self.X_0

        A_sys_low = np.concatenate([self._u_low, self._a_low - B_a, self._v_low - B_v])
        A_sys_upp = np.concatenate([self._u_upp, self._a_upp - B_a, self._v_upp - B_v])

        # Build SFC constraints if present
        if self._sfc_planes:
            T_rows = []
            D_rows = []
            for step_idx in sorted(self._sfc_planes.keys()):
                planes = self._sfc_planes[step_idx]
                for row in planes:
                    t_row = np.zeros(3 * N)
                    t_row[step_idx*3:step_idx*3+3] = row[:3]
                    T_rows.append(t_row)
                    D_rows.append(row[3])

            T = np.array(T_rows)   # (K_total, 3N)
            D_T = np.array(D_rows) # (K_total,)
            A_sfc = T @ self.A_p   # (K_total, 3N)
            A_sfc_upp = -D_T - T @ B_p

            A_all = np.vstack([self.A_sys, A_sfc])
            Alow_all = np.concatenate([A_sys_low, np.full(len(D_T), -1e30)])
            Aupp_all = np.concatenate([A_sys_upp, A_sfc_upp])
        else:
            A_all = self.A_sys
            Alow_all = A_sys_low
            Aupp_all = A_sys_upp

        # Solve with OSQP — reuse cached solver when constraint shape matches
        n_constraints = A_all.shape[0]

        if self._cached_solver is not None and self._cached_n_constraints == n_constraints:
            try:
                self._cached_solver.update(
                    q=f,
                    l=Alow_all,
                    u=Aupp_all,
                    Ax=sparse.csc_matrix(A_all).data,
                )
                result = self._cached_solver.solve()
                if result.info.status in ('solved', 'solved_inaccurate'):
                    self.u_optimal = result.x
                    return True
            except Exception:
                self._cached_solver = None

        # Cold start — create new solver
        H_sparse = sparse.csc_matrix(self.H)
        A_sparse = sparse.csc_matrix(A_all)

        solver = osqp.OSQP()
        solver.setup(
            P=H_sparse, q=f, A=A_sparse, l=Alow_all, u=Aupp_all,
            verbose=False, warm_start=True,
            max_iter=4000, eps_abs=1e-5, eps_rel=1e-5
        )
        result = solver.solve()

        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            self.u_optimal = result.x
            self._cached_solver = solver
            self._cached_n_constraints = n_constraints
            return True
        else:
            self._cached_solver = None
            return False

    def get_optimal_cmd(self, step: int) -> np.ndarray:
        """Get optimal jerk input [ux, uy, uz] at given MPC step."""
        step = min(step, self.cfg.horizon - 1)
        return self.u_optimal[step*3:step*3+3].copy()

    def get_optimal_trajectory(self) -> np.ndarray:
        """
        Roll out the full optimal trajectory.

        Returns:
            (N, 9) array — predicted state [p, v, a] at each horizon step.
        """
        N = self.cfg.horizon
        trajectory = np.zeros((N, 9))
        x = self.X_0.copy()
        for i in range(N):
            u = self.get_optimal_cmd(i)
            x = self.Ax @ x + self.Bx @ u
            trajectory[i] = x
        return trajectory

    def get_first_velocity(self) -> np.ndarray:
        """Extract velocity from the first step of optimal trajectory."""
        traj = self.get_optimal_trajectory()
        return traj[0, 3:6].copy()

    # ---- Internal methods (ported from mpc.cpp) ----

    def _system_model(self, dt: float):
        """Build discrete-time linear system matrices A (9x9), B (9x3)."""
        I3 = np.eye(3)
        A = np.zeros((9, 9))
        A[0:3, 0:3] = I3
        A[0:3, 3:6] = I3 * dt
        A[0:3, 6:9] = I3 * dt**2 * 0.5
        A[3:6, 3:6] = I3 - self.Drag * dt
        A[3:6, 6:9] = I3 * dt
        A[6:9, 6:9] = I3

        B = np.zeros((9, 3))
        B[0:3, :] = I3 * (dt**3 / 6.0)
        B[3:6, :] = I3 * (dt**2 / 2.0)
        B[6:9, :] = I3 * dt
        return A, B

    def _mpc_model(self, A, B, N):
        """Build prediction matrices M (9N x 9) and C (9N x 3N)."""
        n_x, n_u = A.shape[0], B.shape[1]
        M = np.zeros((N * n_x, n_x))
        C = np.zeros((N * n_x, N * n_u))

        A_pow = np.eye(n_x)
        for i in range(N):
            A_pow = A_pow @ A
            M[n_x*i:n_x*(i+1), :] = A_pow

            # First column block
            C[n_x*i:n_x*(i+1), 0:n_u] = A_pow @ np.linalg.solve(A, B)

            # Shift previous row
            if i > 0:
                C[n_x*i:n_x*(i+1), n_u:n_u*(i+1)] = C[n_x*(i-1):n_x*i, 0:n_u*i]

        return M, C

    def _quadratic_term(self, M, C, Q, R, R_con, F, N):
        """Build H matrix: H = C^T Q_bar C + R_bar + R_con_bar."""
        n_u = 3
        # Q_bar with terminal cost
        Q_bar = np.zeros((9 * N, 9 * N))
        for i in range(N):
            Q_bar[9*i:9*(i+1), 9*i:9*(i+1)] = Q
        Q_bar[9*(N-1):9*N, 9*(N-1):9*N] = F

        # R_bar
        R_bar = np.zeros((n_u * N, n_u * N))
        for i in range(N):
            R_bar[n_u*i:n_u*(i+1), n_u*i:n_u*(i+1)] = R

        # R_con_bar (input continuity penalty)
        R_con_bar = np.zeros((n_u * N, n_u * N))
        for i in range(N):
            if i == 0:
                R_con_bar[0:n_u, 0:n_u] = R_con
            elif i == N - 1:
                R_con_bar[n_u*i:n_u*(i+1), n_u*i:n_u*(i+1)] = R_con
                R_con_bar[n_u*i:n_u*(i+1), n_u*(i-1):n_u*i] = -2 * R_con
            else:
                R_con_bar[n_u*i:n_u*(i+1), n_u*i:n_u*(i+1)] = 2 * R_con
                R_con_bar[n_u*i:n_u*(i+1), n_u*(i-1):n_u*i] = -2 * R_con

        H = C.T @ Q_bar @ C + R_bar + R_con_bar
        # Ensure symmetry for OSQP
        H = 0.5 * (H + H.T)
        return H
