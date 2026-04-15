#!/usr/bin/env python3
"""RC Simulator Node — simulates RC controller for IPC mode switching.

Publishes mavros_msgs/RCIn to drive IPC's FSM through
Manual -> Hover -> Pilot -> AutoPilot, then either:
1. replays the same UserModelTunnel command stream used by the RL tunnel stack, or
2. falls back to fixed stick commands for manual debugging.
"""

import os
import sys

import numpy as np
import rospy
from mavros_msgs.msg import RCIn
from std_msgs.msg import Empty

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tunnel_deployment.user_model import UserModelTunnel


class RCSimulator:
    def __init__(self):
        rospy.init_node('rc_sim_node', anonymous=False)

        # Parameters
        self.rate = rospy.get_param('~rate', 50)  # Hz
        self.init_delay = rospy.get_param('~init_delay', 0.2)
        self.switch_delay = rospy.get_param('~switch_delay', 3.0)
        self.autopilot_latch_delay = rospy.get_param('~autopilot_latch_delay', 0.3)
        self.forward_stick = rospy.get_param('~forward_stick', 0.8)  # [-1, 1]
        self.lateral_stick = rospy.get_param('~lateral_stick', 0.0)
        self.vertical_stick = rospy.get_param('~vertical_stick', 0.0)
        self.yaw_stick = rospy.get_param('~yaw_stick', 0.0)
        self.use_user_model = rospy.get_param('~use_user_model', False)
        self.user_model_simple = rospy.get_param('~user_model_simple', False)
        self.user_model_speed = rospy.get_param('~user_model_speed', 2.0)
        self.user_model_freq_base = rospy.get_param('~user_model_freq_base', 0.1)
        self.user_model_freq_scale = rospy.get_param('~user_model_freq_scale', 0.3)
        self.user_model_seed = int(rospy.get_param('~user_model_seed', 42))
        self.user_model_rate = rospy.get_param('~user_model_rate', 20.0)
        self.user_model_device = rospy.get_param('~device', 'cpu')
        self.auto_takeoff = rospy.get_param('~auto_takeoff', True)
        self.takeoff_wait = rospy.get_param('~takeoff_wait', 4.0)
        self.takeoff_topic = rospy.get_param('~takeoff_topic', '/CERLAB/quadcopter/takeoff')

        self.rc_pub = rospy.Publisher('/mavros/rc/in', RCIn, queue_size=10)
        self.takeoff_pub = rospy.Publisher(self.takeoff_topic, Empty, queue_size=1)

        # 18 channels, all centered at 1500 (neutral)
        self.channels = [1500] * 18
        self.mode = 'init'
        self.mode_time = rospy.Time.now()
        self.current_body_cmd = np.zeros(3, dtype=np.float32)
        self.current_sticks = np.array(
            [self.lateral_stick, self.forward_stick, self.vertical_stick, self.yaw_stick],
            dtype=np.float32,
        )
        self.last_user_model_step = None

        self.user_model = None
        if self.use_user_model:
            self.user_model = self._build_user_model()
            self._refresh_user_command(force=True)

        input_mode = 'usermodel' if self.user_model is not None else 'fixed-stick'
        rospy.loginfo(
            "[RC Sim] Initialized. init_delay=%.2fs, switch_delay=%.2fs, autopilot_latch=%.2fs, input=%s, seed=%d",
            self.init_delay,
            self.switch_delay,
            self.autopilot_latch_delay,
            input_mode,
            self.user_model_seed,
        )

    @staticmethod
    def stick_to_pwm(normalized):
        """Convert normalized [-1, 1] stick value to PWM [1050, 1950]."""
        return int(1500 + max(-1.0, min(1.0, normalized)) * 450)

    def _build_user_model(self):
        dt = 1.0 / max(float(self.user_model_rate), 1e-6)
        try:
            model = UserModelTunnel(
                max_speed=self.user_model_speed,
                dt=dt,
                buffer_size=128,
                simple_mode=self.user_model_simple,
                freq_base=self.user_model_freq_base,
                freq_scale=self.user_model_freq_scale,
                device=self.user_model_device,
            )
        except Exception as exc:
            rospy.logwarn(
                "[RC Sim] UserModel init on %s failed (%s). Falling back to cpu.",
                self.user_model_device,
                exc,
            )
            self.user_model_device = 'cpu'
            model = UserModelTunnel(
                max_speed=self.user_model_speed,
                dt=dt,
                buffer_size=128,
                simple_mode=self.user_model_simple,
                freq_base=self.user_model_freq_base,
                freq_scale=self.user_model_freq_scale,
                device='cpu',
            )
        model.reset(seed=self.user_model_seed)
        return model

    def _body_cmd_to_sticks(self, body_cmd):
        scale = max(float(self.user_model_speed), 1e-6)
        return (
            float(np.clip(body_cmd[1] / scale, -1.0, 1.0)),  # lateral
            float(np.clip(body_cmd[0] / scale, -1.0, 1.0)),  # forward
            float(np.clip(body_cmd[2] / scale, -1.0, 1.0)),  # vertical
            0.0,  # yaw
        )

    def _refresh_user_command(self, force=False):
        if self.user_model is None:
            self.current_body_cmd[:] = 0.0
            self.current_sticks[:] = (
                self.lateral_stick,
                self.forward_stick,
                self.vertical_stick,
                self.yaw_stick,
            )
            return

        now = rospy.Time.now()
        if (
            not force
            and self.last_user_model_step is not None
            and (now - self.last_user_model_step).to_sec() < (1.0 / max(float(self.user_model_rate), 1e-6))
        ):
            return

        cmd = self.user_model.step().squeeze(0).detach().cpu().numpy()
        self.current_body_cmd[:] = cmd.astype(np.float32)
        self.current_sticks[:] = self._body_cmd_to_sticks(self.current_body_cmd)
        self.last_user_model_step = now
        rospy.loginfo_throttle(
            2.0,
            "[RC Sim] UserModel cmd_body=[%.2f, %.2f, %.2f] -> sticks=[%.2f, %.2f, %.2f]",
            self.current_body_cmd[0],
            self.current_body_cmd[1],
            self.current_body_cmd[2],
            self.current_sticks[1],
            self.current_sticks[0],
            self.current_sticks[2],
        )

    def publish_rc(self):
        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.channels = list(self.channels)
        self.rc_pub.publish(msg)

    def _set_neutral_motion_sticks(self):
        self.channels[0] = 1500
        self.channels[1] = 1500
        self.channels[2] = 1500
        self.channels[3] = 1500

    def send_takeoff(self):
        if not self.auto_takeoff:
            return

        wait_start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.takeoff_pub.get_num_connections() > 0:
                break
            if (rospy.Time.now() - wait_start).to_sec() > 5.0:
                break
            rospy.sleep(0.1)

        rospy.loginfo("[RC Sim] Publishing takeoff command on %s", self.takeoff_topic)
        for _ in range(5):
            self.takeoff_pub.publish(Empty())
            rospy.sleep(0.1)
        rospy.loginfo("[RC Sim] Waiting %.1fs for takeoff", self.takeoff_wait)
        rospy.sleep(self.takeoff_wait)

    def update_mode(self):
        elapsed = (rospy.Time.now() - self.mode_time).to_sec()

        if self.mode == 'init' and elapsed > self.init_delay:
            self.mode = 'manual'
            self.mode_time = rospy.Time.now()
            rospy.loginfo("[RC Sim] -> MANUAL")

        elif self.mode == 'manual' and elapsed > self.switch_delay:
            # Manual -> Hover: ch[4]>1800, ch[5]<1500, ch[6]<1500
            self.channels[4] = 1900
            self.channels[5] = 1100
            self.channels[6] = 1100
            self.channels[7] = 1900  # arm switch
            self.mode = 'hover'
            self.mode_time = rospy.Time.now()
            rospy.loginfo("[RC Sim] -> HOVER (ch4=1900)")

        elif self.mode == 'hover' and elapsed > self.switch_delay:
            # Hover -> Pilot: ch[5]>1500
            self.channels[5] = 1600
            self.mode = 'pilot'
            self.mode_time = rospy.Time.now()
            rospy.loginfo("[RC Sim] -> PILOT (ch5=1600)")

        elif self.mode == 'pilot' and elapsed > self.switch_delay:
            # Pilot -> AutoPilot: ch[10]>1500
            self._set_neutral_motion_sticks()
            self.channels[10] = 1600
            self.mode = 'autopilot_latch'
            self.mode_time = rospy.Time.now()
            rospy.loginfo("[RC Sim] -> AUTOPILOT EDGE (ch10=1600, neutral sticks)")

        elif self.mode == 'autopilot_latch' and elapsed > self.autopilot_latch_delay:
            self.mode = 'autopilot'
            self.mode_time = rospy.Time.now()
            self._refresh_user_command(force=True)
            rospy.loginfo("[RC Sim] -> AUTOPILOT ACTIVE (usermodel)")

        # In AutoPilot, replay RL's usermodel command stream unless a fixed-stick
        # fallback was explicitly requested.
        if self.mode == 'autopilot':
            self._refresh_user_command()
            lateral_stick, forward_stick, vertical_stick, yaw_stick = self.current_sticks
            # IPC's RCCallback maps joystick as -(channel - 1500) / 450, so
            # positive desired commands must be encoded as PWM values below 1500.
            self.channels[0] = self.stick_to_pwm(-lateral_stick)
            self.channels[1] = self.stick_to_pwm(-forward_stick)
            self.channels[2] = self.stick_to_pwm(-vertical_stick)
            self.channels[3] = self.stick_to_pwm(-yaw_stick)

    def run(self):
        rate = rospy.Rate(self.rate)

        # Wait for simulation time to advance past 2s.
        # IPC's RCCallback has a bug: static msg_last starts with empty channels
        # and stamp=0.  If the first RC message also has stamp≈0, it skips the
        # safe early-return path and accesses msg_last.channels[9] → segfault.
        # Delaying ensures (msg.stamp − 0) > 1.0 → early return → safe init.
        rospy.loginfo("[RC Sim] Waiting for simulation time to advance...")
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec()
            if t > 2.0:
                break
            rospy.sleep(0.1)
        rospy.loginfo("[RC Sim] Sim time=%.1f, starting RC publishing", rospy.Time.now().to_sec())

        self.send_takeoff()

        self.mode_time = rospy.Time.now()
        while not rospy.is_shutdown():
            self.update_mode()
            self.publish_rc()
            rate.sleep()


if __name__ == '__main__':
    try:
        RCSimulator().run()
    except rospy.ROSInterruptException:
        pass
