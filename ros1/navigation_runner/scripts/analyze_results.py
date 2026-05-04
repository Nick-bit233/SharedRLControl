#!/usr/bin/env python3
"""Analyze ROS1 tunnel experiment results from flat or batch outputs."""

import argparse
import csv
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ExperimentAnalyzer:
    COLORS = {'rl': '#2196F3', 'ipc': '#FF5722'}
    LABELS = {'rl': 'RL Policy', 'ipc': 'IPC Algorithm'}

    def __init__(self, data_dir, output_dir=None, pcd_file=None, safety_min_dist=None):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(
            output_dir or os.path.join(self.data_dir, 'analysis')
        )
        self.override_pcd_file = pcd_file
        self.safety_min_dist = 0.2 if safety_min_dist is None else float(safety_min_dist)
        self.tree_cache = {}
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _load_json(path):
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    @staticmethod
    def _load_pcd(filepath):
        points = []
        header_done = False
        with open(filepath, 'r') as handle:
            for line in handle:
                if not header_done:
                    if line.startswith('DATA'):
                        header_done = True
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(points, dtype=np.float32) if points else None

    def _get_tree(self, pcd_file):
        if not pcd_file or not os.path.exists(pcd_file) or not HAS_SCIPY:
            return None
        pcd_file = os.path.abspath(pcd_file)
        if pcd_file not in self.tree_cache:
            pts = self._load_pcd(pcd_file)
            self.tree_cache[pcd_file] = cKDTree(pts) if pts is not None else None
        return self.tree_cache[pcd_file]

    def _collect_map_feasibility(self):
        rows = []
        for obstacle_json in glob.glob(os.path.join(self.data_dir, 'b*_obstacles.json')):
            try:
                payload = self._load_json(obstacle_json)
            except (OSError, json.JSONDecodeError):
                continue
            sampling = payload.get('sampling') or {}
            feasibility = sampling.get('feasibility') or {}
            connectivity = feasibility.get('connectivity') or {}
            rows.append({
                'file': os.path.basename(obstacle_json),
                'seed': payload.get('seed'),
                'sampling_mode': sampling.get('sampling_mode', 'uniform'),
                'attempts': sampling.get('attempts'),
                'min_footprint_gap': feasibility.get('min_footprint_gap'),
                'mean_footprint_gap': feasibility.get('mean_footprint_gap'),
                'max_obstacles_per_local_window': feasibility.get(
                    'max_obstacles_per_local_window'
                ),
                'max_local_area_fraction': feasibility.get('max_local_area_fraction'),
                'connected': connectivity.get('connected'),
                'free_fraction': connectivity.get('free_fraction'),
                'path_length_m': connectivity.get('path_length_m'),
            })

        if not rows:
            return {}

        numeric_keys = [
            'attempts',
            'min_footprint_gap',
            'mean_footprint_gap',
            'max_obstacles_per_local_window',
            'max_local_area_fraction',
            'free_fraction',
            'path_length_m',
        ]
        aggregate = {'count': len(rows)}
        for key in numeric_keys:
            vals = [row[key] for row in rows if row.get(key) is not None]
            if vals:
                aggregate[f'{key}_mean'] = float(np.mean(vals))
                aggregate[f'{key}_min'] = float(np.min(vals))
                aggregate[f'{key}_max'] = float(np.max(vals))
        connected_vals = [row.get('connected') for row in rows if row.get('connected') is not None]
        if connected_vals:
            aggregate['connected_rate'] = float(np.mean(connected_vals))
        modes = defaultdict(int)
        for row in rows:
            modes[row.get('sampling_mode', 'unknown')] += 1
        aggregate['sampling_modes'] = dict(modes)
        return {'aggregate': aggregate, 'maps': rows}

    def _resolve_run_dir(self, run_dir):
        if not run_dir:
            return ''
        run_dir = os.path.abspath(run_dir)
        if os.path.isdir(run_dir):
            return run_dir

        batch_root = os.path.basename(self.data_dir.rstrip(os.sep))
        parts = run_dir.split(os.sep)
        if batch_root in parts:
            suffix = parts[parts.index(batch_root) + 1:]
            local_dir = os.path.join(self.data_dir, *suffix)
            if os.path.isdir(local_dir):
                return os.path.abspath(local_dir)
        for idx, part in enumerate(parts):
            if part.startswith('batch_') and len(part) == len('batch_000') and part[6:].isdigit():
                local_dir = os.path.join(self.data_dir, *parts[idx:])
                if os.path.isdir(local_dir):
                    return os.path.abspath(local_dir)
        return run_dir

    def _discover_batch_runs(self):
        manifest_path = os.path.join(self.data_dir, 'batch_manifest.json')
        batch_manifest = self._load_json(manifest_path) if os.path.exists(manifest_path) else None
        batch_config = None
        if batch_manifest is not None:
            batch_config = batch_manifest.get('batch_config', {})
            runs = batch_manifest.get('runs', [])
        else:
            batch_config_path = os.path.join(self.data_dir, 'batch_config.json')
            batch_config = self._load_json(batch_config_path) if os.path.exists(batch_config_path) else {}
            runs = []
            for summary_path in glob.glob(
                os.path.join(self.data_dir, '**', 'run_summary.json'),
                recursive=True,
            ):
                runs.append(self._load_json(summary_path))

        normalized = []
        for run in runs:
            method = str(run.get('method', '')).lower()
            if method not in ('rl', 'ipc'):
                continue
            run_dir = run.get('run_dir')
            if not run_dir:
                run_dir = os.path.dirname(
                    os.path.join(self.data_dir, run.get('data_file', '')) or self.data_dir
                )
            run_dir = self._resolve_run_dir(run_dir)
            data_file = run.get('data_file', '')
            npz_path = os.path.join(run_dir, data_file) if data_file else ''
            if not npz_path or not os.path.exists(npz_path):
                matches = sorted(glob.glob(os.path.join(run_dir, '*.npz')))
                npz_path = matches[-1] if matches else ''

            pcd_file = self.override_pcd_file or run.get('pcd_file', '')
            if pcd_file and not os.path.isabs(pcd_file):
                pcd_file = os.path.abspath(os.path.join(run_dir, pcd_file))

            normalized.append({
                'method': method,
                'trial_id': int(run.get('trial_id', -1)),
                'run_id': run.get('run_id', ''),
                'batch_idx': int(run.get('batch_idx', -1)),
                'run_idx': int(run.get('run_idx', -1)),
                'map_seed': int(run.get('map_seed', -1)),
                'user_model_seed': int(run.get('user_model_seed', -1)),
                'termination_reason': run.get('termination_reason', ''),
                'goal_reached': bool(run.get('goal_reached', False)),
                'collision': bool(run.get('collision', False)),
                'pcd_file': pcd_file,
                'tunnel_world': run.get('tunnel_world', ''),
                'run_dir': run_dir,
                'npz_path': npz_path,
                'summary': run,
            })

        normalized.sort(key=lambda item: (
            item['batch_idx'],
            item['run_idx'],
            item['trial_id'],
            item['method'],
        ))
        return normalized, batch_config or {}

    def _discover_flat_runs(self):
        runs = []
        files = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
        for path in files:
            base = os.path.basename(path)
            if 'summary' in base:
                continue
            data = dict(np.load(path, allow_pickle=True))
            method = str(data.get('method', '')).lower()
            if method not in ('rl', 'ipc'):
                continue
            pcd_file = self.override_pcd_file or str(data.get('pcd_file', ''))
            if pcd_file and not os.path.isabs(pcd_file):
                pcd_file = os.path.abspath(os.path.join(self.data_dir, pcd_file))
            runs.append({
                'method': method,
                'trial_id': int(data.get('trial_id', len(runs))),
                'run_id': str(data.get('run_id', '')),
                'batch_idx': int(data.get('batch_idx', -1)),
                'run_idx': int(data.get('run_idx', -1)),
                'map_seed': int(data.get('map_seed', -1)),
                'user_model_seed': int(data.get('user_model_seed', -1)),
                'termination_reason': str(data.get('termination_reason', '')),
                'goal_reached': bool(data.get('goal_reached', False)),
                'collision': bool(data.get('collision', False)),
                'pcd_file': pcd_file,
                'tunnel_world': str(data.get('tunnel_world', '')),
                'run_dir': self.data_dir,
                'npz_path': path,
                'summary': {},
            })
        runs.sort(key=lambda item: (item['trial_id'], item['method']))
        return runs, {
            'num_trials': max(len([r for r in runs if r['method'] == 'rl']),
                              len([r for r in runs if r['method'] == 'ipc'])),
            'completed_trials': len(runs),
        }

    def discover_runs(self):
        if os.path.exists(os.path.join(self.data_dir, 'batch_manifest.json')) or \
           os.path.exists(os.path.join(self.data_dir, 'batch_config.json')) or \
           glob.glob(os.path.join(self.data_dir, '**', 'run_summary.json'), recursive=True):
            return self._discover_batch_runs()
        return self._discover_flat_runs()

    @staticmethod
    def _first_present(data, keys):
        for key in keys:
            if key in data:
                return np.array(data[key])
        return None

    @staticmethod
    def _finite_min(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else float('nan')

    @staticmethod
    def _finite_mean(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float('nan')

    @staticmethod
    def _resample_by_arclength(traj, spacing):
        traj = np.asarray(traj, dtype=np.float32)
        if len(traj) == 0:
            return traj
        if len(traj) == 1:
            return traj.copy()

        seg_lens = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = float(arc[-1])
        if total_len <= 1e-6:
            return traj[:1].copy()

        sample_d = np.arange(0.0, total_len + spacing * 0.5, spacing)
        sample_d[-1] = min(sample_d[-1], total_len)
        sampled = np.empty((len(sample_d), traj.shape[1]), dtype=np.float32)
        for dim in range(traj.shape[1]):
            sampled[:, dim] = np.interp(sample_d, arc, traj[:, dim])
        return sampled

    @staticmethod
    def _point_to_polyline_dist(points, polyline):
        points = np.asarray(points, dtype=np.float32)
        polyline = np.asarray(polyline, dtype=np.float32)
        if len(points) == 0 or len(polyline) == 0:
            return np.full(len(points), np.nan, dtype=np.float32)
        if len(polyline) == 1:
            return np.linalg.norm(points - polyline[0], axis=1)

        seg_starts = polyline[:-1]
        seg_ends = polyline[1:]
        seg_vecs = seg_ends - seg_starts
        seg_len_sq = np.sum(seg_vecs * seg_vecs, axis=1)
        seg_len_sq[seg_len_sq == 0.0] = 1e-12

        min_dists = np.full(len(points), np.inf, dtype=np.float32)
        for start, vec, length_sq in zip(seg_starts, seg_vecs, seg_len_sq):
            rel = points - start
            t = np.clip(np.sum(rel * vec, axis=1) / length_sq, 0.0, 1.0)
            closest = start + t[:, None] * vec
            min_dists = np.minimum(min_dists, np.linalg.norm(points - closest, axis=1))
        return min_dists

    def _compute_tcr_metrics(self, pos, ts, human_cmd_world, spacing=0.5):
        if human_cmd_world is None or len(human_cmd_world) != len(pos):
            return {1: float('nan'), 2: float('nan'), 5: float('nan')}
        if len(pos) < 2 or len(ts) != len(pos):
            return {1: float('nan'), 2: float('nan'), 5: float('nan')}

        dt = np.diff(ts, prepend=ts[0])
        if len(dt) > 1:
            fallback_dt = float(np.nanmedian(dt[1:]))
        else:
            fallback_dt = 0.02
        if not np.isfinite(fallback_dt) or fallback_dt <= 0.0:
            fallback_dt = 0.02
        dt[dt <= 0.0] = fallback_dt

        reference = np.empty_like(pos, dtype=np.float32)
        reference[0] = pos[0]
        for idx in range(1, len(pos)):
            reference[idx] = reference[idx - 1] + human_cmd_world[idx - 1] * dt[idx]

        reference_sampled = self._resample_by_arclength(reference, spacing)
        dists = self._point_to_polyline_dist(reference_sampled, pos)
        return {
            threshold: float(np.nanmean(dists < threshold))
            for threshold in (1, 2, 5)
        }

    def compute_metrics(self, run):
        npz_path = run['npz_path']
        if not npz_path or not os.path.exists(npz_path):
            return None, None

        data = dict(np.load(npz_path, allow_pickle=True))
        pos = self._first_present(data, ['position', 'positions'])
        if pos is None or len(pos) < 2:
            return None, None

        ts = self._first_present(data, ['timestamps'])
        if ts is None or len(ts) != len(pos):
            ts = np.arange(len(pos), dtype=np.float32) * 0.02
        vel = self._first_present(data, ['velocity'])
        if vel is None or len(vel) != len(pos):
            dt = np.diff(ts, prepend=ts[0] if len(ts) else 0.0)
            dt[dt == 0] = 0.02
            vel = np.vstack([np.zeros(3, dtype=np.float32), np.diff(pos, axis=0)]) / dt[:, None]
        cmd_world = self._first_present(data, ['cmd_vel_world', 'ctrl_vels_w', 'cmd_vel'])
        if cmd_world is None or len(cmd_world) != len(pos):
            cmd_world = np.zeros_like(pos)
        human_cmd_world = self._first_present(data, ['human_cmd_world', 'human_vels_w'])
        collision_flags = self._first_present(data, ['collision_flags', 'collisions'])
        if collision_flags is None or len(collision_flags) != len(pos):
            collision_flags = np.zeros(len(pos), dtype=bool)

        metric = {
            'method': run['method'],
            'trial_id': run['trial_id'],
            'run_id': run['run_id'],
            'batch_idx': run['batch_idx'],
            'run_idx': run['run_idx'],
            'map_seed': run['map_seed'],
            'user_model_seed': run['user_model_seed'],
            'goal_reached': bool(data.get('goal_reached', run['goal_reached'])),
            'collision': bool(data.get('collision', run['collision'])) or bool(np.any(collision_flags)),
            'termination_reason': str(data.get('termination_reason', run['termination_reason'])),
            'total_time': float(data.get('total_time', ts[-1] if len(ts) else 0.0)),
            'max_x': float(np.max(pos[:, 0])),
            'samples': int(len(pos)),
            'data_file': os.path.relpath(npz_path, self.data_dir),
            'pcd_file': run['pcd_file'],
            'tunnel_world': run['tunnel_world'],
        }

        metric['total_distance'] = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        speeds = np.linalg.norm(vel, axis=1)
        metric['avg_speed'] = float(np.mean(speeds))
        metric['max_speed'] = float(np.max(speeds))
        metric['forward_speed'] = float(np.mean(vel[:, 0]))
        metric['lateral_std'] = float(np.std(pos[:, 1]))
        metric['vertical_std'] = float(np.std(pos[:, 2]))
        if len(vel) > 2:
            dt = np.diff(ts)
            dt[dt == 0] = 0.02
            accel = np.diff(vel, axis=0) / dt[:, None]
            metric['accel_variance'] = float(np.mean(np.var(accel, axis=0)))
        else:
            metric['accel_variance'] = 0.0
        if len(cmd_world) > 1:
            metric['cmd_smoothness'] = float(
                np.mean(np.linalg.norm(np.diff(cmd_world, axis=0), axis=1))
            )
        else:
            metric['cmd_smoothness'] = 0.0
        tcr_metrics = self._compute_tcr_metrics(pos, ts, human_cmd_world)
        metric['tcr_at_1'] = tcr_metrics[1]
        metric['tcr_at_2'] = tcr_metrics[2]
        metric['tcr_at_5'] = tcr_metrics[5]

        min_dist_series = self._first_present(data, ['min_obstacle_dist'])
        if min_dist_series is None or len(min_dist_series) != len(pos):
            tree = self._get_tree(run['pcd_file'])
            if tree is not None:
                min_dist_series = tree.query(pos)[0]
            else:
                min_dist_series = np.full(len(pos), np.nan, dtype=np.float32)
        monitored_dist_series = self._first_present(data, ['min_obstacle_dist_monitored'])
        if monitored_dist_series is None or len(monitored_dist_series) != len(pos):
            monitored_dist_series = min_dist_series

        metric['min_obstacle_dist'] = self._finite_min(min_dist_series)
        metric['avg_obstacle_dist'] = self._finite_mean(min_dist_series)
        metric['monitored_min_obstacle_dist'] = self._finite_min(monitored_dist_series)
        metric['monitored_avg_obstacle_dist'] = self._finite_mean(monitored_dist_series)
        metric['pct_close_05m'] = float(np.nanmean(min_dist_series < 0.5) * 100.0)
        metric['pct_close_1m'] = float(np.nanmean(min_dist_series < 1.0) * 100.0)
        metric['pct_close_02m'] = float(np.nanmean(monitored_dist_series < 0.2) * 100.0)
        metric['pct_close_safety_min'] = float(
            np.nanmean(monitored_dist_series < self.safety_min_dist) * 100.0
        )

        last_window = ts >= max(ts[-1] - 10.0, ts[0])
        if np.any(last_window):
            metric['last_dx_10s'] = float(np.max(pos[last_window, 0]) - np.min(pos[last_window, 0]))
            metric['last_speed_10s'] = float(np.mean(speeds[last_window]))
            metric['last_forward_speed_10s'] = float(np.mean(vel[last_window, 0]))
        else:
            metric['last_dx_10s'] = 0.0
            metric['last_speed_10s'] = 0.0
            metric['last_forward_speed_10s'] = 0.0

        likely_trap = (
            (not metric['goal_reached'])
            and (not metric['collision'])
            and np.isfinite(metric['monitored_min_obstacle_dist'])
            and metric['monitored_min_obstacle_dist'] < self.safety_min_dist
            and metric['pct_close_safety_min'] > 20.0
            and metric['last_dx_10s'] < 0.25
            and metric['last_speed_10s'] < 0.12
        )
        metric['likely_safety_hold_trap'] = bool(likely_trap)

        trajectory = {
            'position': pos,
            'velocity': vel,
            'timestamps': ts,
            'cmd_world': cmd_world,
            'min_dist_series': min_dist_series,
            'monitored_dist_series': monitored_dist_series,
        }
        return metric, trajectory

    def analyze_all(self):
        runs, batch_config = self.discover_runs()
        self.safety_min_dist = float(batch_config.get('safety_min_dist', self.safety_min_dist))
        metrics_by_method = defaultdict(list)
        trajectories_by_method = defaultdict(list)
        all_metrics = []

        for run in runs:
            metric, trajectory = self.compute_metrics(run)
            if metric is None:
                continue
            metrics_by_method[run['method']].append(metric)
            trajectories_by_method[run['method']].append(trajectory)
            all_metrics.append(metric)

        for method in metrics_by_method:
            metrics_by_method[method].sort(key=lambda item: (
                item['batch_idx'],
                item['run_idx'],
                item['trial_id'],
            ))

        return runs, batch_config, metrics_by_method, trajectories_by_method, all_metrics

    def print_comparison(self, results):
        metrics_spec = [
            ('goal_reached', 'Success Rate (%)', lambda v: np.mean(v) * 100),
            ('collision', 'Collision Rate (%)', lambda v: np.mean(v) * 100),
            ('max_x', 'Max Forward (m)', np.mean),
            ('total_time', 'Completion Time (s)', np.mean),
            ('avg_speed', 'Avg Speed (m/s)', np.mean),
            ('forward_speed', 'Forward Speed (m/s)', np.mean),
            ('tcr_at_1', 'TCR@1', np.nanmean),
            ('tcr_at_2', 'TCR@2', np.nanmean),
            ('tcr_at_5', 'TCR@5', np.nanmean),
            ('accel_variance', 'Accel Variance', np.mean),
            ('cmd_smoothness', 'Cmd Smoothness', np.mean),
            ('lateral_std', 'Lateral StdDev (m)', np.mean),
            ('min_obstacle_dist', 'Min Obstacle Dist (m)', np.mean),
            ('monitored_min_obstacle_dist', 'Monitored Min Dist (m)', np.mean),
            ('pct_close_safety_min', 'Time < Safety Min (%)', np.mean),
            ('last_dx_10s', 'Last 10s X Progress (m)', np.mean),
            ('likely_safety_hold_trap', 'Likely Safety Trap (%)', lambda v: np.mean(v) * 100),
        ]

        print("\n" + "=" * 76)
        print(f"{'Metric':<30} {'RL':>20} {'IPC':>20}")
        print("-" * 76)
        for key, name, agg in metrics_spec:
            row = {}
            for method in ('rl', 'ipc'):
                values = [item[key] for item in results.get(method, []) if key in item]
                if values:
                    row[method] = f"{agg(values):.2f} +/- {np.std(values):.2f}"
                else:
                    row[method] = "N/A"
            print(f"{name:<30} {row['rl']:>20} {row['ipc']:>20}")

    def plot_comparison(self, trajectories, results):
        if not HAS_MPL:
            print("Skipping plots (matplotlib not installed)")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
        methods_present = [method for method in ('rl', 'ipc') if results.get(method)]

        def draw_metric_boxplot(ax, metric_key, title, ylabel):
            series = []
            labels = []
            methods = []
            for method in methods_present:
                values = [
                    item[metric_key]
                    for item in results[method]
                    if metric_key in item and np.isfinite(item[metric_key])
                ]
                if values:
                    series.append(values)
                    labels.append(self.LABELS[method])
                    methods.append(method)

            if series:
                box = ax.boxplot(series, labels=labels, patch_artist=True)
                for patch, method in zip(box['boxes'], methods):
                    patch.set_facecolor(self.COLORS[method])
                    patch.set_alpha(0.7)
            else:
                ax.text(
                    0.5,
                    0.5,
                    'No valid data',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                )
                ax.set_xticks([])

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y')

        def draw_summary_table(ax):
            row_defs = [
                (
                    'Success Rate',
                    lambda rows: np.mean([item['goal_reached'] for item in rows]) * 100.0,
                    '{:.1f}%',
                ),
                (
                    'Collision Rate',
                    lambda rows: np.mean([item['collision'] for item in rows]) * 100.0,
                    '{:.1f}%',
                ),
                (
                    'TCR@1',
                    lambda rows: np.nanmean([item.get('tcr_at_1', np.nan) for item in rows]),
                    '{:.3f}',
                ),
                (
                    'TCR@2',
                    lambda rows: np.nanmean([item.get('tcr_at_2', np.nan) for item in rows]),
                    '{:.3f}',
                ),
                (
                    'TCR@5',
                    lambda rows: np.nanmean([item.get('tcr_at_5', np.nan) for item in rows]),
                    '{:.3f}',
                ),
                (
                    'Completion Time',
                    lambda rows: np.mean([
                        item['total_time'] for item in rows if item['goal_reached']
                    ]),
                    '{:.2f}s',
                ),
            ]
            table_rows = []
            for label, getter, fmt in row_defs:
                row = [label]
                for method in methods_present:
                    rows = results.get(method, [])
                    try:
                        value = getter(rows)
                    except (FloatingPointError, ZeroDivisionError, ValueError):
                        value = np.nan
                    row.append(fmt.format(value) if np.isfinite(value) else 'N/A')
                table_rows.append(row)

            ax.axis('off')
            table = ax.table(
                cellText=table_rows,
                colLabels=['Metric'] + [self.LABELS[m] for m in methods_present],
                cellLoc='center',
                colLoc='center',
                loc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.7)
            for (row_idx, col_idx), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#EAEAEA')
                elif col_idx == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#F7F7F7')
            ax.set_title('Key Metrics (TCR@k = trajectory coverage rate within k meters)')

        ax = fig.add_subplot(gs[0, :2])
        for method, tlist in trajectories.items():
            for idx, traj in enumerate(tlist):
                pos = np.array(traj['position'])
                ax.plot(
                    pos[:, 0], pos[:, 1],
                    color=self.COLORS[method],
                    alpha=0.2 + 0.5 * (idx == 0),
                    linewidth=1.0,
                    label=self.LABELS[method] if idx == 0 else None,
                )
        ax.set_title('Trajectories (Top-Down)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        if ax.lines:
            ax.legend()

        ax = fig.add_subplot(gs[0, 2])
        speed_series = []
        speed_labels = []
        speed_colors = []
        for method in methods_present:
            speeds = [
                np.linalg.norm(np.array(traj['velocity']), axis=1)
                for traj in trajectories.get(method, [])
                if len(traj.get('velocity', []))
            ]
            if speeds:
                speed_series.append(np.concatenate(speeds))
                speed_labels.append(self.LABELS[method])
                speed_colors.append(self.COLORS[method])
        if speed_series:
            ax.hist(
                speed_series,
                bins=30,
                density=True,
                alpha=0.55,
                label=speed_labels,
                color=speed_colors,
            )
        else:
            ax.text(0.5, 0.5, 'No speed data', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_title('Speed Distribution')
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        if speed_series:
            ax.legend()

        ax = fig.add_subplot(gs[1, 0])
        draw_summary_table(ax)

        ax = fig.add_subplot(gs[1, 1])
        draw_metric_boxplot(ax, 'max_x', 'Forward Distance', 'm')

        ax = fig.add_subplot(gs[1, 2])
        draw_metric_boxplot(ax, 'min_obstacle_dist', 'Minimum Obstacle Distance', 'm')

        ax = fig.add_subplot(gs[2, :2])
        for method, tlist in trajectories.items():
            for traj in tlist:
                ax.plot(
                    np.array(traj['timestamps']),
                    np.array(traj['min_dist_series']),
                    color=self.COLORS[method],
                    alpha=0.25,
                    linewidth=0.8,
                )
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5m')
        ax.axhline(y=self.safety_min_dist, color='orange', linestyle='-.', alpha=0.6,
                   label=f'safety_min={self.safety_min_dist:.2f}m')
        ax.axhline(y=0.05, color='black', linestyle=':', alpha=0.5, label='0.05m')
        ax.set_title('Nearest Obstacle Distance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

        ax = fig.add_subplot(gs[2, 2])
        draw_metric_boxplot(ax, 'pct_close_safety_min', 'Safety-Min Exposure', '% time')

        plt.suptitle('ROS1 Tunnel Navigation Results', fontsize=14, fontweight='bold')
        plot_path = os.path.join(self.output_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved: {plot_path}")

    def export_csv(self, all_metrics):
        csv_path = os.path.join(self.output_dir, 'metrics.csv')
        all_keys = sorted({key for row in all_metrics for key in row.keys()})
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(all_keys)
            for row in all_metrics:
                writer.writerow([row.get(key, '') for key in all_keys])
        print(f"CSV saved: {csv_path}")

    def export_summary(self, batch_config, results):
        summary = {
            'batch_config': batch_config,
            'methods': {},
            'map_feasibility': self._collect_map_feasibility(),
        }
        for method, rows in results.items():
            if not rows:
                continue
            summary['methods'][method] = {
                'count': len(rows),
                'success_rate': float(np.mean([row['goal_reached'] for row in rows])),
                'collision_rate': float(np.mean([row['collision'] for row in rows])),
                'non_collision_failure_rate': float(
                    np.mean([
                        (not row['goal_reached']) and (not row['collision'])
                        for row in rows
                    ])
                ),
                'likely_safety_hold_trap_rate': float(
                    np.mean([row.get('likely_safety_hold_trap', False) for row in rows])
                ),
                'max_x_mean': float(np.mean([row['max_x'] for row in rows])),
                'tcr_at_1_mean': float(np.nanmean([row.get('tcr_at_1', np.nan) for row in rows])),
                'tcr_at_2_mean': float(np.nanmean([row.get('tcr_at_2', np.nan) for row in rows])),
                'tcr_at_5_mean': float(np.nanmean([row.get('tcr_at_5', np.nan) for row in rows])),
                'min_obstacle_dist_mean': float(np.mean([row['min_obstacle_dist'] for row in rows])),
                'monitored_min_obstacle_dist_mean': float(
                    np.nanmean([row.get('monitored_min_obstacle_dist', np.nan) for row in rows])
                ),
                'pct_close_safety_min_mean': float(
                    np.mean([row.get('pct_close_safety_min', 0.0) for row in rows])
                ),
                'last_dx_10s_mean': float(np.mean([row.get('last_dx_10s', 0.0) for row in rows])),
                'last_speed_10s_mean': float(np.mean([row.get('last_speed_10s', 0.0) for row in rows])),
                'avg_speed_mean': float(np.mean([row['avg_speed'] for row in rows])),
            }
        path = os.path.join(self.output_dir, 'summary.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary saved: {path}")

    def export_render_results(self, batch_config, results):
        render_data_dir = os.path.join(self.output_dir, 'render_data')
        os.makedirs(render_data_dir, exist_ok=True)

        for obstacle_json in glob.glob(os.path.join(self.data_dir, 'b*_obstacles.json')):
            shutil.copy2(
                obstacle_json,
                os.path.join(render_data_dir, os.path.basename(obstacle_json)),
            )
        flat_obstacles = os.path.join(self.data_dir, 'obstacles.json')
        if os.path.exists(flat_obstacles):
            shutil.copy2(flat_obstacles, os.path.join(render_data_dir, 'obstacles.json'))

        payload = {
            'batch_config': batch_config,
            'data_dir': 'render_data',
            'per_trial': {'IPC': [], 'RL': []},
        }
        for method, label in (('ipc', 'IPC'), ('rl', 'RL')):
            for row in results.get(method, []):
                source_npz = os.path.join(self.data_dir, row['data_file'])
                target_npz = os.path.join(render_data_dir, row['data_file'])
                os.makedirs(os.path.dirname(target_npz), exist_ok=True)
                shutil.copy2(source_npz, target_npz)
                payload['per_trial'][label].append({
                    'trial_id': int(row['trial_id']),
                    'trial_seed': int(row['user_model_seed']),
                    'batch_idx': int(row['batch_idx']),
                    'run_idx': int(row['run_idx']),
                    'success': bool(row['goal_reached']),
                    'goal_reached': bool(row['goal_reached']),
                    'collision': bool(row['collision']),
                    'crash_reason': '' if row['goal_reached'] else row['termination_reason'],
                    'max_x_reached': float(row['max_x']),
                    'path_length_m': float(row['total_distance']),
                    'avg_speed': float(row['avg_speed']),
                    'data_file': row['data_file'],
                })
        path = os.path.join(self.output_dir, 'compare_results_ros1.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        print(f"Render-compatible JSON saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ROS1 tunnel experiment data')
    parser.add_argument('--data-dir', required=True,
                        help='Flat run directory or batch output root')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--pcd-file', default=None,
                        help='Override PCD map for flat analysis')
    parser.add_argument('--safety-min-dist', type=float, default=None,
                        help='Override safety_min_dist for trap metrics')
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(
        args.data_dir,
        args.output_dir,
        args.pcd_file,
        safety_min_dist=args.safety_min_dist,
    )
    runs, batch_config, results, trajectories, all_metrics = analyzer.analyze_all()
    print(f"Loaded runs: {len(runs)}")
    analyzer.print_comparison(results)
    analyzer.plot_comparison(trajectories, results)
    analyzer.export_csv(all_metrics)
    analyzer.export_summary(batch_config, results)
    analyzer.export_render_results(batch_config, results)


if __name__ == '__main__':
    main()
