#!/usr/bin/env python3
"""Flight Data Recorder — records drone flight data to .npz files.

Subscribes to odometry and command velocity, records at fixed rate,
and saves per-trial data for post-flight analysis.
"""

import os
import time
import math
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool


class FlightRecorder:
    def __init__(self):
        rospy.init_node('flight_recorder', anonymous=False)

        self.rate = rospy.get_param('~rate', 50.0)
        self.output_dir = rospy.get_param('~output_dir', '/tmp/flight_data')
        self.method = rospy.get_param('~method', 'unknown')
        self.trial_id = rospy.get_param('~trial_id', 0)
        self.auto_start = rospy.get_param('~auto_start', True)
        self.goal_x = rospy.get_param('~goal_x', 15.0)

        self._init_buffers()
        self.recording = False
        self.start_time = None
        self.latest_odom = None
        self.latest_cmd = None
        self.reached_goal = False
        self.data_saved = False

        os.makedirs(self.output_dir, exist_ok=True)

        self.odom_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/odom_raw', Odometry, self._odom_cb, queue_size=1)
        self.cmd_sub = rospy.Subscriber(
            '/CERLAB/quadcopter/cmd_vel', TwistStamped, self._cmd_cb, queue_size=1)

        self.start_sub = rospy.Subscriber(
            '/flight_recorder/start', Bool, self._start_cb, queue_size=1)
        self.stop_sub = rospy.Subscriber(
            '/flight_recorder/stop', Bool, self._stop_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._record_cb)
        rospy.on_shutdown(self._shutdown_cb)

        if self.auto_start:
            rospy.Timer(rospy.Duration(2.0), self._auto_start_cb, oneshot=True)

        rospy.loginfo("[Recorder] Ready. method=%s, trial=%d, rate=%.0fHz",
                      self.method, self.trial_id, self.rate)

    def _init_buffers(self):
        self.buffers = {
            'timestamps': [], 'position': [], 'orientation': [],
            'velocity': [], 'cmd_vel': [], 'angular_vel': [], 'yaw': [],
        }

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _cmd_cb(self, msg):
        self.latest_cmd = msg

    def _start_cb(self, msg):
        if msg.data and not self.recording:
            self._start_recording()

    def _stop_cb(self, msg):
        if msg.data and self.recording:
            self._stop_recording()

    def _auto_start_cb(self, event):
        if not self.recording and self.latest_odom is not None:
            self._start_recording()
        elif not self.recording:
            rospy.Timer(rospy.Duration(1.0), self._auto_start_cb, oneshot=True)

    def _start_recording(self):
        self.recording = True
        self.start_time = rospy.Time.now()
        self._init_buffers()
        self.reached_goal = False
        self.data_saved = False
        rospy.loginfo("[Recorder] STARTED (%s trial %d)", self.method, self.trial_id)

    def _stop_recording(self):
        self.recording = False
        self._save_data()
        rospy.loginfo("[Recorder] STOPPED. %d samples", len(self.buffers['timestamps']))

    def _record_cb(self, event):
        if not self.recording or self.latest_odom is None:
            return

        odom = self.latest_odom
        t = (rospy.Time.now() - self.start_time).to_sec()

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        v = odom.twist.twist.linear
        w = odom.twist.twist.angular

        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        cmd = [0.0, 0.0, 0.0]
        if self.latest_cmd is not None:
            c = self.latest_cmd.twist.linear
            cmd = [c.x, c.y, c.z]

        self.buffers['timestamps'].append(t)
        self.buffers['position'].append([p.x, p.y, p.z])
        self.buffers['orientation'].append([q.x, q.y, q.z, q.w])
        self.buffers['velocity'].append([v.x, v.y, v.z])
        self.buffers['cmd_vel'].append(cmd)
        self.buffers['angular_vel'].append([w.x, w.y, w.z])
        self.buffers['yaw'].append(yaw)

        if p.x >= self.goal_x and not self.reached_goal:
            self.reached_goal = True
            rospy.loginfo("[Recorder] Goal reached at t=%.1fs, x=%.1f", t, p.x)

    def _save_data(self):
        if self.data_saved:
            return

        ts = self.buffers['timestamps']
        filename = "{}_{}_trial{:03d}.npz".format(
            self.method, time.strftime("%Y%m%d_%H%M%S"), self.trial_id)
        filepath = os.path.join(self.output_dir, filename)

        save_dict = {
            'method': self.method,
            'trial_id': self.trial_id,
            'goal_reached': self.reached_goal,
            'total_time': ts[-1] if ts else 0,
        }
        for key, val in self.buffers.items():
            save_dict[key] = np.array(val)

        np.savez_compressed(filepath, **save_dict)
        self.data_saved = True
        rospy.loginfo("[Recorder] Saved: %s (%d samples)", filepath, len(ts))

    def _shutdown_cb(self):
        if self.recording:
            rospy.loginfo("[Recorder] Shutdown while recording — saving buffered data")
            self.recording = False
            self._save_data()


if __name__ == '__main__':
    try:
        FlightRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
