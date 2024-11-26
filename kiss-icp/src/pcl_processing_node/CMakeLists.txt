cmake_minimum_required(VERSION 3.8)
project(pcl_graph_slam)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(PCL REQUIRED)

# Include g2o headers and link libraries
find_package(Eigen3 REQUIRED)
include_directories(/usr/include/eigen3)

# Add the executable
add_executable(graph_slam_node src/graph_slam_node.cpp)
find_package(nav_msgs REQUIRED)
find_package(fmt REQUIRED)

# Link libraries
ament_target_dependencies(graph_slam_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_msgs
  nav_msgs
)
target_include_directories(graph_slam_node PUBLIC /usr/local/include)

target_include_directories(graph_slam_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(graph_slam_node fmt::fmt)

#target_link_libraries(graph_slam_node ${PCL_LIBRARIES} g2o_core g2o_stuff g2o_types_slam3d)
target_link_libraries(graph_slam_node
  g2o_core
  g2o_stuff
  g2o_types_slam3d
  ${PCL_LIBRARIES}
)


# Install the executable
install(TARGETS graph_slam_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
