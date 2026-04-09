#!/usr/bin/env python3
"""RC Simulator Node — simulates RC controller for IPC mode switching.

Publishes mavros_msgs/RCIn to drive IPC's FSM through
Manual -> Hover -> Pilot -> AutoPilot, then sends continuous
forward joystick commands for autonomous tunnel traversal.
"""

import rospy
from mavros_msgs.msg import RCIn


class RCSimulator:
    def __init__(self):
        rospy.init_node('rc_sim_node', anonymous=False)

        # Parameters
        self.rate = rospy.get_param('~rate', 50)  # Hz
        self.switch_delay = rospy.get_param('~switch_delay', 3.0)
        self.forward_stick = rospy.get_param('~forward_stick', 0.8)  # [-1, 1]
        self.lateral_stick = rospy.get_param('~lateral_stick', 0.0)
        self.vertical_stick = rospy.get_param('~vertical_stick', 0.0)
        self.yaw_stick = rospy.get_param('~yaw_stick', 0.0)

        self.rc_pub = rospy.Publisher('/mavros/rc/in', RCIn, queue_size=10)

        # 18 channels, all centered at 1500 (neutral)
        self.channels = [1500] * 18
        self.mode = 'init'
        self.mode_time = rospy.Time.now()

        rospy.loginfo("[RC Sim] Initialized. switch_delay=%.1fs, forward=%.2f",
                      self.switch_delay, self.forward_stick)

    @staticmethod
    def stick_to_pwm(normalized):
        """Convert normalized [-1, 1] stick value to PWM [1050, 1950]."""
        return int(1500 + max(-1.0, min(1.0, normalized)) * 450)

    def publish_rc(self):
        msg = RCIn()
        msg.header.stamp = rospy.Time.now()
        msg.channels = list(self.channels)
        self.rc_pub.publish(msg)

    def update_mode(self):
        elapsed = (rospy.Time.now() - self.mode_time).to_sec()

        if self.mode == 'init' and elapsed > 1.0:
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
            self.channels[10] = 1600
            self.mode = 'autopilot'
            self.mode_time = rospy.Time.now()
            rospy.loginfo("[RC Sim] -> AUTOPILOT (ch10=1600)")

        # In AutoPilot, set joystick for forward flight
        if self.mode == 'autopilot':
            self.channels[0] = self.stick_to_pwm(self.lateral_stick)
            self.channels[1] = self.stick_to_pwm(self.forward_stick)
            self.channels[2] = self.stick_to_pwm(self.vertical_stick)
            self.channels[3] = self.stick_to_pwm(self.yaw_stick)

    def run(self):
        rate = rospy.Rate(self.rate)
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
