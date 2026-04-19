"""
Standalone quaternion rotation utilities.
Extracted from omni_drones.utils.torch — no Isaac Sim dependency.
Convention: quaternion = [w, x, y, z] (scalar-first).
"""
import torch
import functools


def _manual_batch(func):
    """Flatten arbitrary batch dims → (N, D), run, then unflatten."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(
            arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor)
        )
        if len(batch_shapes) != 1:
            raise ValueError(
                f"All tensor args must share the same batch shape, got {batch_shapes}"
            )
        batch_shape = batch_shapes.pop()
        flat_args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
        flat_kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*flat_args, **flat_kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped


@_manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.  q: (N,4) [w,x,y,z], v: (N,3)."""
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a + b + c


@_manual_batch
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vector v by quaternion q.  q: (N,4) [w,x,y,z], v: (N,3)."""
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c


def extract_yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from full rotation.  q: (..., 4) [w,x,y,z]."""
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    half_yaw = yaw / 2.0
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=-1)


def vec_to_world(vec: torch.Tensor, drone_state: torch.Tensor,
                 yaw_only: bool = True) -> torch.Tensor:
    """
    Transform body-frame vector to world frame using orientation from drone_state.
    vec:         (N, 3)
    drone_state: (N, 10) = [vel_b(3), ang_vel_b(3), quat(4)]
    """
    if drone_state.dim() == 3:
        q = drone_state[..., 0, 6:10]
    else:
        q = drone_state[..., 6:10]

    if yaw_only:
        q = extract_yaw_quat(q)

    return quat_rotate(q, vec)
