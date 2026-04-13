#!/usr/bin/env python3
"""
Terminal-based keyboard teleop for tunnel navigation keyboard mode.

Publishes body-frame velocity to /tunnel_nav/user_cmd (TwistStamped).
Also publishes control messages:
  /tunnel_nav/takeoff_cmd   (Empty)  — T key
  /tunnel_nav/assist_toggle (Empty)  — R key

Key bindings:
  W/S — forward / backward  (body X)
  A/D — strafe left / right (body Y)
  Q/E — up / down           (body Z)
  T   — trigger takeoff
  R   — toggle RL assist
  ESC / Ctrl-C — quit

Requires a terminal with raw input (runs with curses).
"""
import sys
import curses
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Empty


# Speed settings (m/s)
LINEAR_SPEED = 2.0
VERTICAL_SPEED = 1.0
PUBLISH_RATE = 20  # Hz

HELP_TEXT = """
╔════════════════════════════════════════════╗
║  Tunnel Keyboard Teleop                    ║
╠════════════════════════════════════════════╣
║  W / S  — forward / backward              ║
║  A / D  — strafe left / right             ║
║  Q / E  — up / down                       ║
║  T      — takeoff                         ║
║  R      — toggle RL assist                ║
║  ESC    — quit                            ║
╠════════════════════════════════════════════╣
║  Speed: {linear:.1f} m/s  Vert: {vert:.1f} m/s     ║
╚════════════════════════════════════════════╝
"""


def main(stdscr):
    rospy.init_node("tunnel_keyboard_teleop", anonymous=False)

    cmd_pub = rospy.Publisher("/tunnel_nav/user_cmd", TwistStamped, queue_size=1)
    takeoff_pub = rospy.Publisher("/tunnel_nav/takeoff_cmd", Empty, queue_size=1)
    assist_pub = rospy.Publisher("/tunnel_nav/assist_toggle", Empty, queue_size=1)

    linear_speed = rospy.get_param("~linear_speed", LINEAR_SPEED)
    vertical_speed = rospy.get_param("~vertical_speed", VERTICAL_SPEED)
    rate_hz = rospy.get_param("~rate", PUBLISH_RATE)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(1000.0 / rate_hz))

    vx, vy, vz = 0.0, 0.0, 0.0
    rl_assist = False
    taken_off = False

    def draw_status():
        stdscr.clear()
        lines = HELP_TEXT.format(linear=linear_speed, vert=vertical_speed).strip().split("\n")
        for i, line in enumerate(lines):
            try:
                stdscr.addstr(i, 0, line)
            except curses.error:
                pass
        row = len(lines) + 1
        try:
            stdscr.addstr(row, 0, f"  Velocity: [{vx:+5.2f}, {vy:+5.2f}, {vz:+5.2f}]")
            stdscr.addstr(row + 1, 0, f"  RL Assist: {'ON' if rl_assist else 'OFF'}    Takeoff: {'YES' if taken_off else 'NO'}")
            stdscr.addstr(row + 3, 0, "  Press keys to control drone...")
        except curses.error:
            pass
        stdscr.refresh()

    rate = rospy.Rate(rate_hz)
    rospy.loginfo("[Teleop] Keyboard teleop started. Focus terminal to control.")

    while not rospy.is_shutdown():
        # Reset velocities each tick (keys must be held)
        vx, vy, vz = 0.0, 0.0, 0.0

        key = stdscr.getch()
        while key != -1:
            if key == 27:  # ESC
                rospy.loginfo("[Teleop] ESC pressed, shutting down")
                return
            ch = chr(key).lower() if 0 <= key < 256 else ""

            if ch == "w":
                vx = linear_speed
            elif ch == "s":
                vx = -linear_speed
            elif ch == "a":
                vy = linear_speed
            elif ch == "d":
                vy = -linear_speed
            elif ch == "q":
                vz = vertical_speed
            elif ch == "e":
                vz = -vertical_speed
            elif ch == "t":
                if not taken_off:
                    takeoff_pub.publish(Empty())
                    taken_off = True
                    rospy.loginfo("[Teleop] Takeoff command sent")
            elif ch == "r":
                assist_pub.publish(Empty())
                rl_assist = not rl_assist
                rospy.loginfo(f"[Teleop] RL assist → {'ON' if rl_assist else 'OFF'}")

            key = stdscr.getch()

        # Publish velocity
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "body"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        cmd_pub.publish(msg)

        draw_status()

        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
