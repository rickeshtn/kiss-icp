cmake_minimum_required(VERSION 3.5)
project(graph_optimization)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)
find_package(Eigen3 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(fmt REQUIRED)

# If you installed g2o manually, you might need to specify the path
set(G2O_DIR "/usr/local/include/g2o") # Uncomment and set if necessary
find_package(g2o REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${g2o_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
)

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Declare the executable
add_executable(graph_optimization_node src/graph_optimization_node.cpp)

# Link libraries
target_link_libraries(graph_optimization_node
  ${PCL_LIBRARIES}
  ${g2o_LIBRARIES}
  g2o_core
  g2o_types_slam2d
  g2o_solver_eigen
  g2o_stuff
  ${tf2_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  fmt::fmt
)

# Dependencies
ament_target_dependencies(graph_optimization_node
  rclcpp
  nav_msgs
  sensor_msgs
  PCL
  Eigen3
  g2o
  pcl_conversions
  tf2
  tf2_geometry_msgs
)

# Install target
install(TARGETS graph_optimization_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
