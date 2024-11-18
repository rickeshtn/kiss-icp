# Dockerfile

# Use the official ROS 2 Humble image as the base
FROM ros:humble

# Set environment variables
ENV ROS_WS=/home/ros2_ws
ENV SAVE_DIR=/home/ros2_saved_data

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    nano \
    vim \
    lsb-release \
    gnupg2 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PCL dependencies
RUN apt-get update && apt-get install -y \
    libpcl-dev \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages if needed (e.g., Open3D)
# RUN pip3 install open3d

# Create workspace and save directory
RUN mkdir -p $ROS_WS/src
RUN mkdir -p $SAVE_DIR

# Set working directory
WORKDIR $ROS_WS

# Initialize rosdep (only update, assuming initialization is already done)
RUN rosdep update

# Clone or copy your 'topic_saver' package
# Replace the URL with your actual repository or skip if you plan to mount the code
# RUN git clone https://github.com/your_username/topic_saver.git src/topic_saver

# Alternatively, copy from local if you have the package code
# This requires the package code to be in the same directory as the Dockerfile under ./src/topic_saver
# COPY ./src/topic_saver src/topic_saver

# Install ROS 2 dependencies
# If you have a local package, you can install dependencies via rosdep
# RUN rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
# We'll allow building at runtime after mounting the code

# Source ROS 2
ENV ROS2_SETUP /opt/ros/humble/setup.bash

# Create an entrypoint script
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]