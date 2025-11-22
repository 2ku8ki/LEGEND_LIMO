#!/bin/bash
set -e

# ROS 기본 환경 (Melodic)
source /opt/ros/melodic/setup.bash

# Python3용 cv_bridge 워크스페이스
if [ -f "/opt/cvbridge_ws/install/setup.bash" ]; then
  source /opt/cvbridge_ws/install/setup.bash --extend
fi

# 유저 catkin 워크스페이스 (있으면)
if [ -f "/root/LEGEND_WS/drive_ws/devel/setup.bash" ]; then
  source /root/LEGEND_WS/drive_ws/devel/setup.bash
fi

export ROS_PYTHON_VERSION=3

exec "$@"


