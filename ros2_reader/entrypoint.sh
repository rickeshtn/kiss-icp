#!/bin/bash
set -e

# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Source workspace setup if it exists
if [ -f "/home/ros2_ws/install/setup.bash" ]; then
    source /home/ros2_ws/install/setup.bash
fi

exec "$@"