from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")

from src.core.profiler import get_profiler, reset_profiler
from src.datasets.trajectory_dataset import (
    TrajectoryDataset,
    TrajectoryMetadata,
    create_trajectory_dataset,
)
from src.simulated_users.user_model_tunnely import UserModelTunnel


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _write_dataset(path: Path, velocities: np.ndarray) -> TrajectoryDataset:
    velocities = velocities.astype(np.float32)
    positions = np.cumsum(velocities[..., :3], axis=1).astype(np.float32)
    bboxes = np.zeros((velocities.shape[0], 6), dtype=np.float32)
    styles = {
        "noise_freq": np.ones(velocities.shape[0], dtype=np.float32) * 0.1,
        "smoothness": np.ones(velocities.shape[0], dtype=np.float32) * 0.5,
        "laziness": np.zeros(velocities.shape[0], dtype=np.float32),
    }
    metadata = TrajectoryMetadata(
        num_trajectories=velocities.shape[0],
        trajectory_length=velocities.shape[1],
        action_dim=velocities.shape[2],
        dt=0.1,
        max_speed=10.0,
        max_speed_z=10.0,
        max_speed_yaw=0.0,
        reference_map_bounds=(10.0, 10.0, 10.0),
    )
    create_trajectory_dataset(
        str(path),
        velocities,
        positions,
        bboxes,
        styles,
        metadata,
    )
    return TrajectoryDataset(str(path), device=torch.device("cpu"))


def _cfg(buffer_size: int = 4, blend_steps: int = 2) -> DotDict:
    return DotDict(
        device="cpu",
        sim=DotDict(dt=0.1),
        env=DotDict(map_range=[10.0, 10.0, 10.0]),
        algo=DotDict(training_frame_num=buffer_size),
        user_model=DotDict(refill_blend_steps=blend_steps),
    )


def test_get_windows_returns_indexed_contiguous_slices(tmp_path: Path):
    velocities = np.stack(
        [
            np.arange(18, dtype=np.float32).reshape(6, 3),
            100.0 + np.arange(18, dtype=np.float32).reshape(6, 3),
        ]
    )
    dataset = _write_dataset(tmp_path / "windows.h5", velocities)

    windows, _ = dataset.get_windows(
        torch.tensor([1, 0]),
        torch.tensor([2, 1]),
        window_size=3,
    )

    expected = np.stack([velocities[1, 2:5], velocities[0, 1:4]])
    assert torch.allclose(windows, torch.from_numpy(expected))

    with pytest.raises(ValueError, match="start_offsets"):
        dataset.get_windows(torch.tensor([0]), torch.tensor([4]), window_size=3)


def test_user_model_replays_one_trajectory_contiguously_across_buffer_refills(tmp_path: Path):
    reset_profiler()
    get_profiler(enabled=False)

    velocities = np.zeros((1, 8, 3), dtype=np.float32)
    velocities[0, :, 0] = np.arange(8, dtype=np.float32)
    dataset = _write_dataset(tmp_path / "playback.h5", velocities)
    model = UserModelTunnel(
        num_envs=1,
        cfg=_cfg(buffer_size=4, blend_steps=2),
        offline_mode=True,
        dataset=dataset,
        sampling_mode="raw",
    )

    env_ids = torch.tensor([0])
    model.reset(
        pos=torch.zeros(1, 1, 3),
        quat=torch.zeros(1, 4),
        env_ids=env_ids,
    )

    actions = []
    refills = []
    for _ in range(8):
        action, needs_refill = model.step(
            drone_state=torch.zeros(1, 1),
            drone_pos_w=torch.zeros(1, 3),
        )
        actions.append(action[0].clone())
        refills.append(bool(needs_refill[0]))

    assert torch.allclose(torch.stack(actions)[:, 0], torch.arange(8, dtype=torch.float32))
    assert refills == [False, False, False, False, True, False, False, False]


def test_user_model_blends_fallback_stream_after_dataset_exhaustion(tmp_path: Path):
    reset_profiler()
    get_profiler(enabled=False)

    velocities = np.zeros((1, 8, 3), dtype=np.float32)
    velocities[0, :, 0] = np.arange(8, dtype=np.float32)
    dataset = _write_dataset(tmp_path / "fallback.h5", velocities)
    model = UserModelTunnel(
        num_envs=1,
        cfg=_cfg(buffer_size=4, blend_steps=2),
        offline_mode=True,
        dataset=dataset,
        sampling_mode="raw",
    )
    model.reset(
        pos=torch.zeros(1, 3),
        quat=torch.zeros(1, 4),
        env_ids=torch.tensor([0]),
    )

    for _ in range(8):
        model.step(drone_state=torch.zeros(1, 1), drone_pos_w=torch.zeros(1, 3))

    action, needs_refill = model.step(
        drone_state=torch.zeros(1, 1),
        drone_pos_w=torch.zeros(1, 3),
    )

    assert bool(needs_refill[0])
    assert torch.allclose(action[0], torch.tensor([3.5, 0.0, 0.0]))


def test_seeded_reset_reenables_eval_dataset_generator(tmp_path: Path):
    velocities = np.zeros((1, 4, 3), dtype=np.float32)
    dataset = _write_dataset(tmp_path / "eval_seed.h5", velocities)
    model = UserModelTunnel(
        num_envs=1,
        cfg=_cfg(buffer_size=4),
        offline_mode=True,
        dataset=dataset,
        sampling_mode="raw",
    )

    model.set_eval_seed(7)
    model.reset(
        pos=torch.zeros(1, 3),
        quat=torch.zeros(1, 4),
        env_ids=torch.tensor([0]),
    )
    assert not model._use_eval_dataset_generator

    model.reset(
        pos=torch.zeros(1, 3),
        quat=torch.zeros(1, 4),
        env_ids=torch.tensor([0]),
        seed=7,
    )
    assert model._use_eval_dataset_generator


def test_user_model_tunnel_is_offline_only(tmp_path: Path):
    velocities = np.zeros((1, 4, 3), dtype=np.float32)
    dataset = _write_dataset(tmp_path / "offline_only.h5", velocities)

    with pytest.raises(ValueError, match="offline-only"):
        UserModelTunnel(
            num_envs=1,
            cfg=_cfg(buffer_size=4),
            offline_mode=False,
            dataset=dataset,
            sampling_mode="raw",
        )
