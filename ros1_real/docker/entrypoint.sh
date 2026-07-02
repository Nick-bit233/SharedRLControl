#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
if [ -f /root/catkin_ws/devel/setup.bash ]; then
  source /root/catkin_ws/devel/setup.bash
fi

if [ -z "${ROS_IP:-}" ]; then
  unset ROS_IP
fi
if [ -z "${ROS_HOSTNAME:-}" ]; then
  unset ROS_HOSTNAME
fi

: "${ROS_MASTER_URI:=http://127.0.0.1:11311}"
export ROS_MASTER_URI

if [ -z "${ROS_IP:-}" ] && [ -z "${ROS_HOSTNAME:-}" ]; then
  export ROS_IP=127.0.0.1
fi

exec "$@"
