# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    gnupg2 \
    lsb-release \
    locales \
    sudo \
    unzip \
    software-properties-common

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install ROS 1 (Noetic) dependencies
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -sSL http://packages.ros.org/ros.key | sudo -E apt-key add - && \
    apt-get update && \
    apt-get install -y \
    ros-noetic-desktop-full

# Initialize rosdep
RUN rosdep init && \
    rosdep update

# Install ROS 2 (Humble) dependencies
RUN curl -sSL https://raw.githubusercontent.com/ros2/ros2/master/ros2.repos -o ros2.repos && \
    vcs import src < ros2.repos && \
    rosdep install --from-paths src --ignore-src --rosdistro humble -y

# Build ROS 2 workspace
WORKDIR /ros2_ws
RUN colcon build

# Source ROS 1 and ROS 2 setup files
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash"]