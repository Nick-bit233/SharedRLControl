#!/usr/bin/env python3
"""Run Comparison — automated multi-trial experiment runner.

Orchestrates RL vs IPC comparison by cycling through methods,
resetting the drone between trials, and collecting flight data.

Usage:
  python3 run_comparison.py --methods rl,ipc --n-trials 5 --timeout 60
"""

import os
import sys
import time
import signal
import subprocess
import argparse

import numpy as np

try:
    import rospy
    from std_msgs.msg import Bool, Empty
    from nav_msgs.msg import Odometry
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class ComparisonRunner:
    def __init__(self, args):
        self.methods = args.methods.split(',')
        self.n_trials = args.n_trials
        self.timeout = args.timeout
        self.output_dir = args.output_dir
        self.goal_x = args.goal_x
        self.start_z = args.start_z
        self.warmup = args.warmup
        self.inter_delay = args.delay
        self.gazebo_launch = args.gazebo_launch

        os.makedirs(self.output_dir, exist_ok=True)

        self.current_pos = None
        self.trial_results = []

    def run(self):
        if not HAS_ROS:
            print("ERROR: ROS not available. Source your workspace first.")
            sys.exit(1)

        rospy.init_node('comparison_runner', anonymous=True)

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.reset_pub = rospy.Publisher(
            '/CERLAB/quadcopter/reset', Empty, queue_size=1)
        self.takeoff_pub = rospy.Publisher(
            '/CERLAB/quadcopter/takeoff', Empty, queue_size=1)
        self.rec_start = rospy.Publisher(
            '/flight_recorder/start', Bool, queue_size=1)
        self.rec_stop = rospy.Publisher(
            '/flight_recorder/stop', Bool, queue_size=1)

        rospy.sleep(1.0)

        print("=" * 60)
        print("RL vs IPC Comparison Experiment")
        print("=" * 60)
        print(f"  Methods:     {self.methods}")
        print(f"  Trials each: {self.n_trials}")
        print(f"  Timeout:     {self.timeout}s")
        print(f"  Goal X:      {self.goal_x}m")
        print(f"  Output:      {self.output_dir}")
        print("=" * 60)

        for method in self.methods:
            print(f"\n--- {method.upper()} ---")
            for trial in range(self.n_trials):
                if rospy.is_shutdown():
                    break
                result = self._run_trial(method, trial)
                self.trial_results.append(result)
                status = 'SUCCESS' if result['success'] else 'FAIL'
                print(f"  Trial {trial+1}/{self.n_trials}: {status} "
                      f"(x={result['max_x']:.1f}m, t={result['duration']:.1f}s)")
                rospy.sleep(self.inter_delay)

        self._print_summary()
        self._save_summary()

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.current_pos = [p.x, p.y, p.z]

    def _reset_drone(self):
        print("    Resetting...")
        self.reset_pub.publish(Empty())
        rospy.sleep(2.0)
        self.takeoff_pub.publish(Empty())
        rospy.sleep(3.0)

        for _ in range(50):
            if self.current_pos and abs(self.current_pos[2] - self.start_z) < 0.5:
                break
            rospy.sleep(0.1)

    def _launch_controller(self, method, trial):
        if method == 'rl':
            cmd = ['roslaunch', 'navigation_runner', 'tunnel_sim.launch',
                   'start_gazebo:=false', f'trial_id:={trial}']
        elif method == 'ipc':
            cmd = ['roslaunch', 'navigation_runner', 'tunnel_ipc_sim.launch',
                   'start_gazebo:=false', f'trial_id:={trial}']
        else:
            raise ValueError(f"Unknown method: {method}")

        return subprocess.Popen(cmd, preexec_fn=os.setsid,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _stop_process(proc):
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _run_trial(self, method, trial_idx):
        result = {
            'method': method, 'trial': trial_idx,
            'success': False, 'max_x': -15.0, 'duration': 0.0, 'collision': False,
        }

        self._reset_drone()

        # Start recorder
        self.rec_start.publish(Bool(data=True))
        rospy.sleep(0.5)

        proc = self._launch_controller(method, trial_idx)
        rospy.sleep(self.warmup)

        start_time = time.time()
        max_x = -15.0

        while not rospy.is_shutdown():
            elapsed = time.time() - start_time

            if self.current_pos:
                max_x = max(max_x, self.current_pos[0])

                if self.current_pos[0] >= self.goal_x:
                    result.update(success=True, duration=elapsed, max_x=max_x)
                    break

                if self.current_pos[2] < 0.1 and elapsed > 3.0:
                    result.update(collision=True, duration=elapsed, max_x=max_x)
                    break

            if elapsed > self.timeout:
                result.update(duration=elapsed, max_x=max_x)
                break

            rospy.sleep(0.1)

        self.rec_stop.publish(Bool(data=True))
        rospy.sleep(0.5)
        self._stop_process(proc)

        return result

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for method in self.methods:
            trials = [r for r in self.trial_results if r['method'] == method]
            if not trials:
                continue
            n = len(trials)
            succ = sum(1 for t in trials if t['success'])
            dists = [t['max_x'] for t in trials]
            times = [t['duration'] for t in trials if t['success']]
            print(f"\n{method.upper()} ({n} trials):")
            print(f"  Success:  {succ}/{n} ({100*succ/n:.0f}%)")
            print(f"  Distance: {np.mean(dists):.1f} +/- {np.std(dists):.1f} m")
            if times:
                print(f"  Time:     {np.mean(times):.1f} +/- {np.std(times):.1f} s")

    def _save_summary(self):
        path = os.path.join(self.output_dir, 'experiment_summary.npz')
        np.savez(path, results=self.trial_results)
        print(f"\nSaved: {path}")


def main():
    parser = argparse.ArgumentParser(description='RL vs IPC Comparison')
    parser.add_argument('--methods', default='rl,ipc')
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--output-dir', default='/tmp/flight_data')
    parser.add_argument('--goal-x', type=float, default=15.0)
    parser.add_argument('--start-z', type=float, default=1.0)
    parser.add_argument('--warmup', type=float, default=8.0)
    parser.add_argument('--delay', type=float, default=3.0)
    parser.add_argument('--gazebo-launch', default='')
    args = parser.parse_args()

    runner = ComparisonRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
