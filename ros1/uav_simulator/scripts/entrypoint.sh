#!/bin/bash
# Entrypoint: explicit DISPLAY handling for Gazebo
#   - TUNNEL_RENDER_MODE=headless: force Xvfb/software rendering
#   - TUNNEL_RENDER_MODE=x11: require host X11 forwarding
#   - TUNNEL_RENDER_MODE=auto: legacy auto-detection

# Source ROS workspaces
source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash 2>/dev/null || true
source /root/catkin_ws/devel/setup.bash 2>/dev/null || true

# Export Gazebo paths
export GAZEBO_PLUGIN_PATH=/root/catkin_ws/src/uav_simulator/plugins:${GAZEBO_PLUGIN_PATH}
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:/root/catkin_ws/src/uav_simulator/models:${GAZEBO_MODEL_PATH}

start_xvfb() {
    export DISPLAY=:99
    export LIBGL_ALWAYS_SOFTWARE=1
    export QT_X11_NO_MITSHM=1
    if ! pgrep -f "Xvfb :99" > /dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 1
    fi
    echo "[entrypoint] Headless Xvfb active on DISPLAY=:99 (software rendering)"
}

use_host_x11() {
    if [ -z "$DISPLAY" ] || [ ! -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
        echo "[entrypoint] ERROR: TUNNEL_RENDER_MODE=x11 requires DISPLAY and /tmp/.X11-unix" >&2
        exit 64
    fi
    unset LIBGL_ALWAYS_SOFTWARE
    export QT_X11_NO_MITSHM=1
    echo "[entrypoint] Using host display DISPLAY=$DISPLAY (X11 debug rendering)"
}

case "${TUNNEL_RENDER_MODE:-headless}" in
    headless)
        start_xvfb
        ;;
    x11)
        use_host_x11
        ;;
    auto)
        if [ -n "$DISPLAY" ] && [ -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
            use_host_x11
        else
            start_xvfb
        fi
        ;;
    *)
        echo "[entrypoint] ERROR: invalid TUNNEL_RENDER_MODE=${TUNNEL_RENDER_MODE}" >&2
        echo "[entrypoint] Expected one of: headless, x11, auto" >&2
        exit 64
        ;;
esac

exec "$@"
