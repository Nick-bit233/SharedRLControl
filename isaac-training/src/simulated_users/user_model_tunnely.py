import logging
from typing import Any, Literal, Optional, TYPE_CHECKING

import torch

from src.core.profiler import get_profiler

if TYPE_CHECKING:
    from src.datasets.trajectory_dataset import TrajectoryDataset


class UserModelTunnel:
    """Offline dataset-backed user command player for tunnel tasks."""

    def __init__(
        self,
        num_envs,
        cfg,
        use_lib_noise=False,
        logger=None,
        offline_mode: bool = False,
        dataset: Optional["TrajectoryDataset"] = None,
        sampling_mode: Literal["scaled", "raw"] = "scaled",
    ):
        del use_lib_noise
        if not offline_mode:
            raise ValueError(
                "UserModelTunnel is offline-only on this branch. "
                "Set user_model.offline_mode=true and provide user_model.dataset_path."
            )
        if dataset is None:
            raise ValueError("dataset must be provided when UserModelTunnel is offline-only")
        if sampling_mode not in {"scaled", "raw"}:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

        self.num_envs = int(num_envs)
        self.cfg = cfg
        self.dataset = dataset
        self.sampling_mode = sampling_mode
        self.device = torch.device(cfg.device)
        self.dt = float(cfg.sim.dt)
        self.logger = logger or logging.getLogger("user_model_tunnel")
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self._dataset_eval_generator: Optional[torch.Generator] = None
        self._dataset_eval_seed: Optional[int] = None
        self._use_eval_dataset_generator = False

        # cfg.env.map_range is stored as [Isaac-y, Isaac-x, Isaac-z] half-extents
        # in tunnel configs; sample_scaled expects IsaacSim x/y/z order.
        raw_map_range = list(cfg.env.map_range)
        if len(raw_map_range) >= 2:
            raw_map_range[0], raw_map_range[1] = raw_map_range[1], raw_map_range[0]
        self.env_map_range = torch.tensor(raw_map_range, dtype=torch.float32, device=self.device)

        self.sampling_lower_bounds = None
        self.sampling_upper_bounds = None
        sampling_bounds_cfg = cfg.user_model.get("sampling_bounds", None)
        if sampling_bounds_cfg is not None:
            lower, upper = self._build_sampling_bounds(
                sampling_bounds_cfg,
                cfg.user_model.get("sampling_bounds_expansion", 0.0),
            )
            self.sampling_lower_bounds = torch.tensor(lower, dtype=torch.float32, device=self.device)
            self.sampling_upper_bounds = torch.tensor(upper, dtype=torch.float32, device=self.device)
            print(f"[UserModel] Using explicit sampling_bounds lower={lower} upper={upper}")

        self.buffer_size = int(cfg.algo.training_frame_num)
        if self.buffer_size > self.dataset.metadata.trajectory_length:
            raise ValueError(
                f"training_frame_num ({self.buffer_size}) must be <= dataset "
                f"trajectory_length ({self.dataset.metadata.trajectory_length})"
            )
        self.refill_blend_steps = max(0, int(cfg.user_model.get("refill_blend_steps", 12)))

        self.action_buffer = torch.zeros(self.num_envs, self.buffer_size, 3, device=self.device)
        self.buffer_read_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.next_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.scale_factors = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.has_last_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print(f"[UserModel] Offline dataset playback enabled with sampling_mode='{sampling_mode}'")

    def set_eval_seed(self, seed: int) -> None:
        """Reset the offline dataset sampler for deterministic evaluation."""
        self._dataset_eval_seed = int(seed)
        self._dataset_eval_generator = torch.Generator(device=self.device)
        self._dataset_eval_generator.manual_seed(self._dataset_eval_seed + 1_000_003)
        self._use_eval_dataset_generator = True

    def _ensure_eval_dataset_generator(self, seed: int) -> None:
        if self._dataset_eval_generator is None or self._dataset_eval_seed != int(seed):
            self.set_eval_seed(seed)
        else:
            self._use_eval_dataset_generator = True

    @staticmethod
    def _axis_bounds(bounds_cfg: Any, axis: str) -> list[float]:
        values = bounds_cfg.get(axis, None)
        if values is None:
            raise ValueError(f"user_model.sampling_bounds.{axis} must be set")
        values = list(values)
        if len(values) != 2:
            raise ValueError(f"user_model.sampling_bounds.{axis} must have two values")
        lower, upper = float(values[0]), float(values[1])
        if upper <= lower:
            raise ValueError(
                f"user_model.sampling_bounds.{axis} upper must be > lower, got {values}"
            )
        return [lower, upper]

    @staticmethod
    def _parse_bounds_expansion(expansion_cfg: Any) -> list[float]:
        if expansion_cfg is None:
            return [0.0, 0.0, 0.0]
        if isinstance(expansion_cfg, (int, float)):
            value = float(expansion_cfg)
            return [value, value, value]
        if hasattr(expansion_cfg, "get"):
            return [float(expansion_cfg.get(axis, 0.0)) for axis in ("x", "y", "z")]
        values = list(expansion_cfg)
        if len(values) != 3:
            raise ValueError(
                "user_model.sampling_bounds_expansion must be a scalar, "
                "a 3-value list, or a mapping with x/y/z"
            )
        return [float(v) for v in values]

    @classmethod
    def _build_sampling_bounds(
        cls,
        bounds_cfg: Any,
        expansion_cfg: Any,
    ) -> tuple[list[float], list[float]]:
        lower = []
        upper = []
        expansion = cls._parse_bounds_expansion(expansion_cfg)
        for axis, pad in zip(("x", "y", "z"), expansion):
            axis_lower, axis_upper = cls._axis_bounds(bounds_cfg, axis)
            if pad < 0:
                raise ValueError("sampling_bounds_expansion values must be non-negative")
            lower.append(axis_lower - pad)
            upper.append(axis_upper + pad)
        return lower, upper

    def _active_generator(self) -> Optional[torch.Generator]:
        return self._dataset_eval_generator if self._use_eval_dataset_generator else None

    def _sample_traj_indices(self, count: int) -> torch.Tensor:
        return torch.randint(
            0,
            self.dataset.metadata.num_trajectories,
            (count,),
            device=self.device,
            generator=self._active_generator(),
        )

    def _reset_streams(self, env_ids: torch.Tensor, start_pos: torch.Tensor) -> None:
        count = len(env_ids)
        self.traj_indices[env_ids] = self._sample_traj_indices(count)
        self.next_offsets[env_ids] = 0
        self.has_last_action[env_ids] = False
        self.last_action[env_ids] = 0.0
        self._refresh_scale_factors(env_ids, start_pos)

    def _refresh_scale_factors(self, env_ids: torch.Tensor, start_pos: torch.Tensor) -> None:
        if self.sampling_mode == "raw":
            self.scale_factors[env_ids] = 1.0
            return

        full_offsets = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        _, positions = self.dataset.get_windows(
            self.traj_indices[env_ids],
            full_offsets,
            self.dataset.metadata.trajectory_length,
        )
        self.scale_factors[env_ids] = self.dataset.compute_scale_factors(
            positions,
            start_pos,
            map_bounds=self.env_map_range,
            lower_bounds=self.sampling_lower_bounds,
            upper_bounds=self.sampling_upper_bounds,
        )

    def reset(self, pos, quat, env_ids, seed=None):
        """
        Reset offline command streams for env_ids.

        pos and quat are accepted for env API compatibility. Commands are stored
        in the dataset body frame, so quat is intentionally unused.
        """
        del quat
        if pos.ndim == 3:
            pos = pos.squeeze(1)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if seed is not None:
            self._ensure_eval_dataset_generator(int(seed))
        else:
            self._use_eval_dataset_generator = False

        self.buffer_read_idx[env_ids] = 0
        self._reset_streams(env_ids, pos)
        self._refill_from_dataset(env_ids, pos)

    def step(self, drone_state, drone_pos_w):
        del drone_state
        profiler = get_profiler()
        profiler.start("user_model/step")

        pos = drone_pos_w.squeeze(1) if drone_pos_w.ndim == 3 else drone_pos_w
        needs_refill = self.buffer_read_idx >= self.buffer_size
        if needs_refill.any():
            with profiler.timer("user_model/refill_buffer"):
                idxs = needs_refill.nonzero(as_tuple=False).squeeze(-1)
                self._refill_from_dataset(idxs, pos[idxs])
                self.buffer_read_idx[idxs] = 0

        read_indices = self.buffer_read_idx.view(-1, 1, 1).expand(-1, 1, 3)
        action = torch.gather(self.action_buffer, 1, read_indices).squeeze(1)
        self.last_action = action.clone()
        self.has_last_action[:] = True
        self.buffer_read_idx += 1

        profiler.stop("user_model/step")
        return action, needs_refill

    def _refill_from_dataset(self, env_ids, start_pos):
        """Refill action buffer using contiguous windows from offline trajectories."""
        profiler = get_profiler()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = len(env_ids)
        if count == 0:
            return

        with profiler.timer("user_model/dataset_sample"):
            self._resample_exhausted_streams(env_ids, start_pos)
            offsets = self.next_offsets[env_ids].clone()
            velocities, _ = self.dataset.get_windows(
                self.traj_indices[env_ids],
                offsets,
                self.buffer_size,
            )
            velocities = velocities[..., :3].clone()
            velocities *= self.scale_factors[env_ids].view(-1, 1, 1)
            self.next_offsets[env_ids] += self.buffer_size
            self._blend_new_stream_prefix(env_ids, velocities)

        self.action_buffer[env_ids] = velocities

    def _resample_exhausted_streams(self, env_ids: torch.Tensor, start_pos: torch.Tensor) -> None:
        remaining = self.dataset.metadata.trajectory_length - self.next_offsets[env_ids]
        exhausted = remaining < self.buffer_size
        if not exhausted.any():
            return

        exhausted_envs = env_ids[exhausted]
        exhausted_pos = start_pos[exhausted]
        self.traj_indices[exhausted_envs] = self._sample_traj_indices(len(exhausted_envs))
        self.next_offsets[exhausted_envs] = 0
        self._refresh_scale_factors(exhausted_envs, exhausted_pos)

    def _blend_new_stream_prefix(self, env_ids: torch.Tensor, velocities: torch.Tensor) -> None:
        blend_steps = min(self.refill_blend_steps, velocities.shape[1])
        if blend_steps <= 0:
            return

        starts_new_stream = self.next_offsets[env_ids] == self.buffer_size
        do_blend = starts_new_stream & self.has_last_action[env_ids]
        if not do_blend.any():
            return

        rows = do_blend.nonzero(as_tuple=False).squeeze(-1)
        alpha = torch.linspace(
            0.0,
            1.0,
            blend_steps + 1,
            device=self.device,
            dtype=velocities.dtype,
        )[1:].view(1, blend_steps, 1)
        prev = self.last_action[env_ids[rows]].view(-1, 1, 3)
        velocities[rows, :blend_steps] = (
            (1.0 - alpha) * prev + alpha * velocities[rows, :blend_steps]
        )
