cmake_minimum_required(VERSION 3.5)
project(pointcloud_bridge)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

find_package(roscpp REQUIRED)  # ROS1 C++ core package

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(pointcloud_bridge src/pointcloud_subscriber.cpp)

ament_target_dependencies(pointcloud_bridge
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

target_link_libraries(pointcloud_bridge
  ${PCL_LIBRARIES}
  roscpp  # Link with ROS1
)

install(TARGETS
  pointcloud_bridge
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()