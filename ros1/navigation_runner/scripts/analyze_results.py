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

    def __init__(self, data_dir, output_dir=None, pcd_file=None):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(
            output_dir or os.path.join(self.data_dir, 'analysis')
        )
        self.override_pcd_file = pcd_file
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
            run_dir = os.path.abspath(run_dir)
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

        min_dist_series = self._first_present(data, ['min_obstacle_dist'])
        if min_dist_series is None or len(min_dist_series) != len(pos):
            tree = self._get_tree(run['pcd_file'])
            if tree is not None:
                min_dist_series = tree.query(pos)[0]
            else:
                min_dist_series = np.full(len(pos), np.nan, dtype=np.float32)
        metric['min_obstacle_dist'] = float(np.nanmin(min_dist_series))
        metric['avg_obstacle_dist'] = float(np.nanmean(min_dist_series))
        metric['pct_close_05m'] = float(np.nanmean(min_dist_series < 0.5) * 100.0)
        metric['pct_close_1m'] = float(np.nanmean(min_dist_series < 1.0) * 100.0)

        trajectory = {
            'position': pos,
            'velocity': vel,
            'timestamps': ts,
            'cmd_world': cmd_world,
            'min_dist_series': min_dist_series,
        }
        return metric, trajectory

    def analyze_all(self):
        runs, batch_config = self.discover_runs()
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
            ('accel_variance', 'Accel Variance', np.mean),
            ('cmd_smoothness', 'Cmd Smoothness', np.mean),
            ('lateral_std', 'Lateral StdDev (m)', np.mean),
            ('min_obstacle_dist', 'Min Obstacle Dist (m)', np.mean),
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
        ax.legend()

        ax = fig.add_subplot(gs[0, 2])
        for method, tlist in trajectories.items():
            for traj in tlist:
                speed = np.linalg.norm(np.array(traj['velocity']), axis=1)
                ax.plot(
                    np.array(traj['timestamps']),
                    speed,
                    color=self.COLORS[method],
                    alpha=0.25,
                    linewidth=0.8,
                )
        ax.set_title('Speed Profiles')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.grid(True, alpha=0.3)

        methods_present = [method for method in ('rl', 'ipc') if results.get(method)]

        ax = fig.add_subplot(gs[1, 0])
        rates = [np.mean([item['goal_reached'] for item in results[m]]) * 100 for m in methods_present]
        bars = ax.bar([self.LABELS[m] for m in methods_present], rates,
                      color=[self.COLORS[m] for m in methods_present])
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0,
                    f"{rate:.0f}%", ha='center')
        ax.set_title('Success Rate')
        ax.set_ylabel('%')
        ax.set_ylim(0, 110)

        ax = fig.add_subplot(gs[1, 1])
        data = [[item['max_x'] for item in results[m]] for m in methods_present]
        box = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                         patch_artist=True)
        for patch, method in zip(box['boxes'], methods_present):
            patch.set_facecolor(self.COLORS[method])
            patch.set_alpha(0.7)
        ax.set_title('Forward Distance')
        ax.set_ylabel('m')
        ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[1, 2])
        data = [[item['min_obstacle_dist'] for item in results[m]] for m in methods_present]
        box = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                         patch_artist=True)
        for patch, method in zip(box['boxes'], methods_present):
            patch.set_facecolor(self.COLORS[method])
            patch.set_alpha(0.7)
        ax.set_title('Minimum Obstacle Distance')
        ax.set_ylabel('m')
        ax.grid(True, alpha=0.3, axis='y')

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
        ax.axhline(y=0.05, color='black', linestyle=':', alpha=0.5, label='0.05m')
        ax.set_title('Nearest Obstacle Distance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = fig.add_subplot(gs[2, 2])
        data = [[item['cmd_smoothness'] for item in results[m]] for m in methods_present]
        box = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                         patch_artist=True)
        for patch, method in zip(box['boxes'], methods_present):
            patch.set_facecolor(self.COLORS[method])
            patch.set_alpha(0.7)
        ax.set_title('Command Smoothness')
        ax.set_ylabel('Cmd delta')
        ax.grid(True, alpha=0.3, axis='y')

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
        }
        for method, rows in results.items():
            if not rows:
                continue
            summary['methods'][method] = {
                'count': len(rows),
                'success_rate': float(np.mean([row['goal_reached'] for row in rows])),
                'collision_rate': float(np.mean([row['collision'] for row in rows])),
                'max_x_mean': float(np.mean([row['max_x'] for row in rows])),
                'min_obstacle_dist_mean': float(np.mean([row['min_obstacle_dist'] for row in rows])),
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
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.data_dir, args.output_dir, args.pcd_file)
    runs, batch_config, results, trajectories, all_metrics = analyzer.analyze_all()
    print(f"Loaded runs: {len(runs)}")
    analyzer.print_comparison(results)
    analyzer.plot_comparison(trajectories, results)
    analyzer.export_csv(all_metrics)
    analyzer.export_summary(batch_config, results)
    analyzer.export_render_results(batch_config, results)


if __name__ == '__main__':
    main()
