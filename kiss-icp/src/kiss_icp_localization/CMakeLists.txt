cmake_minimum_required(VERSION 3.5)
project(kiss_icp_localization)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)  

# Add the installed kiss_icp package path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "/HOST_HOME/Projects/sensmore/kiss_icp_standalone/kiss_icp/install")


# Find the kiss_icp package
find_package(kiss_icp REQUIRED)

# Include directories
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${kiss_icp_INCLUDE_DIRS}
)

# Add executable
add_executable(kiss_icp_localization_node src/kiss_icp_localization.cpp)

# Link libraries
target_link_libraries(kiss_icp_localization_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  Sophus::Sophus
  kiss_icp::kiss_icp_pipeline
  kiss_icp::kiss_icp_core
)

# Link with ROS 2 libraries
ament_target_dependencies(kiss_icp_localization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  PCL
  tf2_ros
  Eigen3
  Sophus
  kiss_icp
  nav_msgs  
)

# Install executable
install(TARGETS kiss_icp_localization_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install other files (e.g., launch files)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Ament package
ament_package()
