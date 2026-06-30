#!/usr/bin/env python3
"""Offline smoke checks for SRLC fake-MAVROS dry-run assets."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.pcd_io import read_pcd_xyz  # noqa: E402
from tunnel_deployment.pcd_raycast import PcdRaycaster  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd-file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--launch-file", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--position", nargs=3, type=float, default=[0.0, 0.0, 2.0])
    parser.add_argument("--map-origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--lidar-range", type=float, default=4.0)
    parser.add_argument("--lidar-vfov", nargs=2, type=float, default=[-10.0, 20.0])
    parser.add_argument("--lidar-vbeams", type=int, default=4)
    parser.add_argument("--lidar-hres", type=float, default=10.0)
    parser.add_argument("--skip-policy", action="store_true")
    return parser.parse_args()


def local_to_map(pos, origin, yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    c, s = np.cos(yaw), np.sin(yaw)
    pos = np.asarray(pos, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    return np.array([c * pos[0] - s * pos[1], s * pos[0] + c * pos[1], pos[2]], dtype=np.float32) + origin


def check_launch(path):
    if not path:
        return
    ET.parse(path)
    print(f"[ok] launch XML parses: {path}")


def check_pcd(path):
    points = read_pcd_xyz(path)
    if len(points) == 0:
        raise RuntimeError(f"PCD has no finite xyz points: {path}")
    print(f"[ok] PCD points={len(points)} bounds_min={points.min(axis=0)} bounds_max={points.max(axis=0)}")


def check_lidar(args):
    raycaster = PcdRaycaster(args.pcd_file, resolution=0.1, inflate=(0.15, 0.15, 0.05))
    pos_map = local_to_map(args.position, args.map_origin, args.map_yaw_deg)
    points = raycaster.raycast(
        pos_map,
        np.deg2rad(args.map_yaw_deg),
        args.lidar_range,
        args.lidar_vfov[0],
        args.lidar_vfov[1],
        args.lidar_vbeams,
        args.lidar_hres,
    )
    expected = int(360.0 / args.lidar_hres) * args.lidar_vbeams
    if points.shape != (expected, 3):
        raise RuntimeError(f"Raycast shape mismatch: {points.shape} != {(expected, 3)}")
    dists = np.linalg.norm(points - pos_map.reshape(1, 3), axis=-1)
    ranges = np.clip((args.lidar_range - dists) / args.lidar_range, 0.0, 1.0)
    if ranges.shape[0] != expected or not np.isfinite(ranges).all():
        raise RuntimeError("LiDAR range image contains invalid values")
    nearest = raycaster.nearest_distance(pos_map)
    front_bins = [0, 1, int(360.0 / args.lidar_hres) - 2, int(360.0 / args.lidar_hres) - 1]
    front = args.lidar_range * (1.0 - float(np.max(ranges.reshape(-1, args.lidar_vbeams)[front_bins, :])))
    print(f"[ok] LiDAR shape=(1,{int(360.0 / args.lidar_hres)},{args.lidar_vbeams}) min_dist={nearest:.3f} front_dist={front:.3f}")


def check_policy(args):
    if args.skip_policy:
        print("[skip] policy checkpoint load")
        return
    if not args.checkpoint:
        raise RuntimeError("--checkpoint is required unless --skip-policy is set")
    from tunnel_deployment.policy_net import TunnelPolicyNet
    import torch

    policy = TunnelPolicyNet.from_checkpoint(args.checkpoint, action_limit=2.0, min_concentration=2.0, device=args.device)
    policy.debug = False
    state = torch.zeros(1, 10, device=args.device)
    state[:, 6] = 1.0
    human_action = torch.tensor([[0.8, 0.05, 0.0]], dtype=torch.float32, device=args.device)
    lidar = torch.zeros(1, 1, 36, 4, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        action = policy(state, human_action, lidar, deterministic=True)
    if tuple(action.shape) != (1, 3) or not torch.isfinite(action).all():
        raise RuntimeError(f"Policy output is invalid: shape={tuple(action.shape)}")
    print(f"[ok] policy checkpoint loads and outputs {action.detach().cpu().numpy()[0].tolist()}")


def main():
    args = parse_args()
    check_launch(args.launch_file)
    check_pcd(args.pcd_file)
    check_lidar(args)
    check_policy(args)
    print("[ok] SRLC dry-run offline smoke check passed")


if __name__ == "__main__":
    main()
