#!/usr/bin/env python3
"""Analyze Results — process flight data and generate comparison plots.

Loads .npz files from flight_recorder, computes metrics, and generates
comparison plots and CSV for RL vs IPC performance analysis.

Usage:
  python3 analyze_results.py --data-dir /tmp/flight_data --pcd-file tunnel_map.pcd
"""

import os
import glob
import argparse
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


class FlightAnalyzer:
    COLORS = {'rl': '#2196F3', 'ipc': '#FF5722'}
    LABELS = {'rl': 'RL Policy', 'ipc': 'IPC Algorithm'}

    def __init__(self, data_dir, output_dir=None, pcd_file=None):
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        self.obstacle_tree = None
        if pcd_file and os.path.exists(pcd_file) and HAS_SCIPY:
            pts = self._load_pcd(pcd_file)
            if pts is not None:
                self.obstacle_tree = cKDTree(pts)
                print(f"Loaded {len(pts)} obstacle points for distance computation")

    @staticmethod
    def _load_pcd(filepath):
        points, header_done = [], False
        with open(filepath, 'r') as f:
            for line in f:
                if not header_done:
                    if line.startswith('DATA'):
                        header_done = True
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(points) if points else None

    def load_trials(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
        trials = {'rl': [], 'ipc': []}
        for f in files:
            if 'summary' in os.path.basename(f) or 'analysis' in f:
                continue
            data = dict(np.load(f, allow_pickle=True))
            method = str(data.get('method', 'unknown'))
            if method in trials:
                trials[method].append(data)
        print(f"Loaded: RL={len(trials['rl'])} trials, IPC={len(trials['ipc'])} trials")
        return trials

    def compute_metrics(self, trial):
        pos = np.array(trial['position'])
        vel = np.array(trial['velocity'])
        cmd = np.array(trial['cmd_vel'])
        ts = np.array(trial['timestamps'])

        if len(pos) < 2:
            return None

        m = {}
        m['goal_reached'] = bool(trial.get('goal_reached', False))
        m['total_time'] = float(ts[-1]) if len(ts) > 0 else 0
        m['max_x'] = float(np.max(pos[:, 0]))
        m['total_distance'] = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

        speeds = np.linalg.norm(vel, axis=1)
        m['avg_speed'] = float(np.mean(speeds))
        m['max_speed'] = float(np.max(speeds))
        m['forward_speed'] = float(np.mean(vel[:, 0]))

        # Smoothness
        if len(vel) > 2:
            dt = np.diff(ts)
            dt[dt == 0] = 0.02
            accel = np.diff(vel, axis=0) / dt[:, None]
            m['accel_variance'] = float(np.mean(np.var(accel, axis=0)))
        else:
            m['accel_variance'] = 0.0

        if len(cmd) > 1:
            m['cmd_smoothness'] = float(np.mean(np.linalg.norm(np.diff(cmd, axis=0), axis=1)))
        else:
            m['cmd_smoothness'] = 0.0

        # Safety
        if self.obstacle_tree is not None:
            dists, _ = self.obstacle_tree.query(pos)
            m['min_obstacle_dist'] = float(np.min(dists))
            m['avg_obstacle_dist'] = float(np.mean(dists))
            m['pct_close_05m'] = float(np.mean(dists < 0.5) * 100)
            m['pct_close_1m'] = float(np.mean(dists < 1.0) * 100)

        m['lateral_std'] = float(np.std(pos[:, 1]))
        m['vertical_std'] = float(np.std(pos[:, 2]))

        return m

    def analyze_all(self, trials):
        results = {}
        for method, trial_list in trials.items():
            results[method] = [self.compute_metrics(t) for t in trial_list]
            results[method] = [m for m in results[method] if m is not None]
        return results

    def print_comparison(self, results):
        metrics_spec = [
            ('goal_reached', 'Success Rate (%)', lambda v: np.mean(v) * 100),
            ('max_x', 'Max Forward (m)', np.mean),
            ('total_time', 'Completion Time (s)', np.mean),
            ('avg_speed', 'Avg Speed (m/s)', np.mean),
            ('forward_speed', 'Forward Speed (m/s)', np.mean),
            ('accel_variance', 'Accel Variance', np.mean),
            ('cmd_smoothness', 'Cmd Smoothness', np.mean),
            ('lateral_std', 'Lateral StdDev (m)', np.mean),
        ]
        if self.obstacle_tree is not None:
            metrics_spec += [
                ('min_obstacle_dist', 'Min Obstacle Dist (m)', np.mean),
                ('avg_obstacle_dist', 'Avg Obstacle Dist (m)', np.mean),
                ('pct_close_05m', 'Time <0.5m (%)', np.mean),
            ]

        print("\n" + "=" * 70)
        print(f"{'Metric':<30} {'RL':>18} {'IPC':>18}")
        print("-" * 70)

        for key, name, agg in metrics_spec:
            vals = {}
            for method in ['rl', 'ipc']:
                if method in results and results[method]:
                    v = [m[key] for m in results[method] if key in m]
                    if v:
                        vals[method] = f"{agg(v):.2f} +/- {np.std(v):.2f}"
                    else:
                        vals[method] = "N/A"
                else:
                    vals[method] = "N/A"
            print(f"{name:<30} {vals.get('rl','N/A'):>18} {vals.get('ipc','N/A'):>18}")

    def plot_comparison(self, trials, results):
        if not HAS_MPL:
            print("Skipping plots (install matplotlib)")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Trajectories (top-down)
        ax = fig.add_subplot(gs[0, :2])
        for method, tlist in trials.items():
            for i, t in enumerate(tlist):
                pos = np.array(t['position'])
                ax.plot(pos[:, 0], pos[:, 1],
                        color=self.COLORS[method], alpha=0.3 + 0.5*(i == 0),
                        linewidth=1, label=self.LABELS[method] if i == 0 else None)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectories (Top-Down)')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 2. Speed profiles
        ax = fig.add_subplot(gs[0, 2])
        for method, tlist in trials.items():
            for t in tlist:
                spd = np.linalg.norm(np.array(t['velocity']), axis=1)
                ax.plot(np.array(t['timestamps']), spd,
                        color=self.COLORS[method], alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed')
        ax.grid(True, alpha=0.3)

        # 3-5. Bar/box charts
        methods_present = [m for m in ['rl', 'ipc'] if results.get(m)]

        ax = fig.add_subplot(gs[1, 0])
        rates = [np.mean([r['goal_reached'] for r in results[m]]) * 100
                 for m in methods_present]
        bars = ax.bar([self.LABELS[m] for m in methods_present], rates,
                      color=[self.COLORS[m] for m in methods_present])
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()+1,
                    f'{r:.0f}%', ha='center')
        ax.set_ylabel('%')
        ax.set_title('Success Rate')
        ax.set_ylim(0, 110)

        ax = fig.add_subplot(gs[1, 1])
        data = [[r['max_x'] for r in results[m]] for m in methods_present]
        bp = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                        patch_artist=True)
        for p, m in zip(bp['boxes'], methods_present):
            p.set_facecolor(self.COLORS[m])
            p.set_alpha(0.7)
        ax.set_ylabel('m')
        ax.set_title('Forward Distance')
        ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[1, 2])
        data = [[r['avg_speed'] for r in results[m]] for m in methods_present]
        bp = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                        patch_artist=True)
        for p, m in zip(bp['boxes'], methods_present):
            p.set_facecolor(self.COLORS[m])
            p.set_alpha(0.7)
        ax.set_ylabel('m/s')
        ax.set_title('Average Speed')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Obstacle distance (if available)
        if self.obstacle_tree is not None:
            ax = fig.add_subplot(gs[2, :2])
            for method, tlist in trials.items():
                for t in tlist:
                    pos = np.array(t['position'])
                    d, _ = self.obstacle_tree.query(pos)
                    ax.plot(np.array(t['timestamps']), d,
                            color=self.COLORS[method], alpha=0.3, linewidth=0.5)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5m')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (m)')
            ax.set_title('Nearest Obstacle Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 7. Command smoothness
        ax = fig.add_subplot(gs[2, 2])
        data = [[r['cmd_smoothness'] for r in results[m]] for m in methods_present]
        bp = ax.boxplot(data, labels=[self.LABELS[m] for m in methods_present],
                        patch_artist=True)
        for p, m in zip(bp['boxes'], methods_present):
            p.set_facecolor(self.COLORS[m])
            p.set_alpha(0.7)
        ax.set_ylabel('Cmd delta')
        ax.set_title('Command Smoothness')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('RL vs IPC Tunnel Navigation', fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'comparison_plots.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plots saved: {path}")
        plt.close()

    def export_csv(self, results):
        import csv
        path = os.path.join(self.output_dir, 'metrics.csv')
        all_keys = sorted(set(k for mlist in results.values() for m in mlist for k in m))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'trial'] + all_keys)
            for method, mlist in results.items():
                for i, m in enumerate(mlist):
                    w.writerow([method, i] + [m.get(k, '') for k in all_keys])
        print(f"CSV saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze flight comparison data')
    parser.add_argument('--data-dir', default='/tmp/flight_data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--pcd-file', default=None)
    args = parser.parse_args()

    analyzer = FlightAnalyzer(args.data_dir, args.output_dir, args.pcd_file)
    trials = analyzer.load_trials()
    results = analyzer.analyze_all(trials)
    analyzer.print_comparison(results)
    analyzer.plot_comparison(trials, results)
    analyzer.export_csv(results)


if __name__ == '__main__':
    main()
