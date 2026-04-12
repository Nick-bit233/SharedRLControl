#!/bin/bash
# Entrypoint: smart DISPLAY handling for Gazebo
#   - If host X11 is forwarded (DISPLAY already set), use it (GPU rendering)
#   - Otherwise, start Xvfb for headless software rendering

# Source ROS workspaces
source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash 2>/dev/null || true
source /root/catkin_ws/devel/setup.bash 2>/dev/null || true

# Export Gazebo paths
export GAZEBO_PLUGIN_PATH=/root/catkin_ws/src/uav_simulator/plugins:${GAZEBO_PLUGIN_PATH}
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:/root/catkin_ws/src/uav_simulator/models:${GAZEBO_MODEL_PATH}

if [ -n "$DISPLAY" ] && [ -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
    # Host X11 forwarded — use real display (GPU accelerated)
    echo "[entrypoint] Using host display DISPLAY=$DISPLAY (GPU rendering)"
    unset LIBGL_ALWAYS_SOFTWARE
else
    # No host display — start Xvfb for headless mode
    export DISPLAY=:99
    export LIBGL_ALWAYS_SOFTWARE=1
    if ! pgrep -x Xvfb > /dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 1
        echo "[entrypoint] Xvfb started on DISPLAY=:99 (software rendering)"
    fi
fi

exec "$@"
