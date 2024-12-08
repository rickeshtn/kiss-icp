cmake_minimum_required(VERSION 3.8)
project(pcl_graph_slam)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable C++14 explicitly
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenCV REQUIRED)

# Include G2O
# G2O does not provide a CMake package, so we need to specify the include directories and libraries manually
# Adjust the paths if G2O is installed in a different location
set(G2O_INCLUDE_DIRS /usr/local/include)
set(G2O_LIBRARY_DIRS /usr/local/lib)

# Verify that G2O directories exist
if(NOT EXISTS "${G2O_INCLUDE_DIRS}")
    message(FATAL_ERROR "G2O include directory not found: ${G2O_INCLUDE_DIRS}")
endif()

if(NOT EXISTS "${G2O_LIBRARY_DIRS}")
    message(FATAL_ERROR "G2O library directory not found: ${G2O_LIBRARY_DIRS}")
endif()

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
)

link_directories(
  ${G2O_LIBRARY_DIRS}
)

# Add the executable
add_executable(graph_slam_node src/graph_slam_node.cpp)

add_executable(graph_slam_nodePC src/graph_slam_node_pc.cpp)

# Add compiler definitions
add_definitions(${PCL_DEFINITIONS})

# Specify dependencies for ament
ament_target_dependencies(graph_slam_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_msgs
  nav_msgs
)

# Link libraries
target_link_libraries(graph_slam_node
  fmt::fmt
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  g2o_core
  g2o_stuff
  g2o_types_slam2d
)

# Install the executable
install(TARGETS graph_slam_node
  DESTINATION lib/${PROJECT_NAME}
)




# Specify dependencies for ament
ament_target_dependencies(graph_slam_nodePC
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_msgs
  nav_msgs
)

# Link libraries
target_link_libraries(graph_slam_nodePC
  fmt::fmt
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  g2o_core
  g2o_stuff
  g2o_types_slam2d
  g2o_types_slam3d   
  g2o_solver_eigen
)

# Install the executable
install(TARGETS graph_slam_nodePC
  DESTINATION lib/${PROJECT_NAME}
)

# Declare ament package
ament_package()
