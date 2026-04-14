#!/usr/bin/env python3
"""
Terminal-based keyboard teleop for tunnel navigation keyboard mode.

Publishes body-frame velocity to /tunnel_nav/user_cmd (TwistStamped).
Also publishes control messages:
  /tunnel_nav/takeoff_cmd   (Empty)  — T key
  /tunnel_nav/assist_toggle (Empty)  — R key

Velocity is **sticky**: press a key to set that axis, it persists until
you press the opposite key or SPACE (all-stop).  This avoids reliance on
key-repeat and works reliably in any terminal.

Key bindings (sticky):
  W / S   — forward / backward  (body X)  [toggles: press again to stop]
  A / D   — strafe left / right (body Y)
  Q / E   — up / down           (body Z)
  SPACE   — all-stop (zero all axes)
  T       — trigger takeoff
  R       — toggle RL assist
  ESC     — quit

Requires a terminal with raw input (runs with curses).
"""
import curses
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Empty, Bool


PUBLISH_RATE = 20  # Hz


def main(stdscr):
    rospy.init_node("tunnel_keyboard_teleop", anonymous=False)

    cmd_pub = rospy.Publisher("/tunnel_nav/user_cmd", TwistStamped, queue_size=1)
    takeoff_pub = rospy.Publisher("/tunnel_nav/takeoff_cmd", Empty, queue_size=1)
    assist_pub = rospy.Publisher("/tunnel_nav/assist_toggle", Empty, queue_size=1)

    linear_speed = rospy.get_param("~linear_speed", 2.0)
    vertical_speed = rospy.get_param("~vertical_speed", 1.0)
    rate_hz = rospy.get_param("~rate", PUBLISH_RATE)

    # Subscribe to assist status for display
    assist_state = [False]
    def _assist_cb(msg):
        assist_state[0] = msg.data
    rospy.Subscriber("/tunnel_nav/assist_active", Bool, _assist_cb)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(1000.0 / rate_hz))

    # Sticky velocity state
    vx, vy, vz = 0.0, 0.0, 0.0
    taken_off = False
    last_info = ""

    def draw():
        stdscr.erase()
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║  Tunnel Keyboard Teleop  (sticky velocity)  ║",
            "╠══════════════════════════════════════════════╣",
            "║  W/S  — fwd/back    (press again = stop X)  ║",
            "║  A/D  — left/right  (press again = stop Y)  ║",
            "║  Q/E  — up/down     (press again = stop Z)  ║",
            "║  SPACE — all-stop                           ║",
            "║  T     — takeoff    R — toggle RL assist    ║",
            "║  ESC   — quit                               ║",
            f"║  Speed: lin={linear_speed:.1f}  vert={vertical_speed:.1f} m/s         ║",
            "╚══════════════════════════════════════════════╝",
        ]
        for i, line in enumerate(lines):
            try:
                stdscr.addstr(i, 0, line)
            except curses.error:
                pass
        row = len(lines) + 1
        try:
            stdscr.addstr(row, 0,     f"  Vel:  [{vx:+5.2f}, {vy:+5.2f}, {vz:+5.2f}]")
            stdscr.addstr(row + 1, 0, f"  RL Assist: {'ON' if assist_state[0] else 'OFF'}    Takeoff: {'YES' if taken_off else 'NO'}")
            if last_info:
                stdscr.addstr(row + 2, 0, f"  >> {last_info}")
        except curses.error:
            pass
        stdscr.refresh()

    rospy.loginfo("[Teleop] Keyboard teleop started (sticky mode). Focus this window.")

    while not rospy.is_shutdown():
        key = stdscr.getch()
        while key != -1:
            if key == 27:  # ESC
                rospy.loginfo("[Teleop] ESC pressed, shutting down")
                return

            ch = chr(key).lower() if 0 <= key < 256 else ""

            if ch == "w":
                vx = 0.0 if vx > 0 else linear_speed
                last_info = f"fwd {'ON' if vx > 0 else 'OFF'}"
            elif ch == "s":
                vx = 0.0 if vx < 0 else -linear_speed
                last_info = f"back {'ON' if vx < 0 else 'OFF'}"
            elif ch == "a":
                vy = 0.0 if vy > 0 else linear_speed
                last_info = f"left {'ON' if vy > 0 else 'OFF'}"
            elif ch == "d":
                vy = 0.0 if vy < 0 else -linear_speed
                last_info = f"right {'ON' if vy < 0 else 'OFF'}"
            elif ch == "q":
                vz = 0.0 if vz > 0 else vertical_speed
                last_info = f"up {'ON' if vz > 0 else 'OFF'}"
            elif ch == "e":
                vz = 0.0 if vz < 0 else -vertical_speed
                last_info = f"down {'ON' if vz < 0 else 'OFF'}"
            elif ch == " ":
                vx, vy, vz = 0.0, 0.0, 0.0
                last_info = "ALL STOP"
            elif ch == "t":
                if not taken_off:
                    takeoff_pub.publish(Empty())
                    taken_off = True
                    last_info = "TAKEOFF sent"
                    rospy.loginfo("[Teleop] Takeoff command sent")
                else:
                    last_info = "already took off"
            elif ch == "r":
                assist_pub.publish(Empty())
                last_info = "RL assist toggled"
                rospy.loginfo("[Teleop] RL assist toggle sent")

            key = stdscr.getch()

        # Publish velocity at constant rate
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "body"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        cmd_pub.publish(msg)

        draw()

        try:
            rospy.Rate(rate_hz).sleep()
        except rospy.ROSInterruptException:
            break


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
