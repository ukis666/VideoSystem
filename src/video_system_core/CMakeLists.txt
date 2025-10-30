cmake_minimum_required(VERSION 3.16)
project(video_system_core)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(video_system_interfaces REQUIRED)

add_library(image_event_node_lib
  src/image_event_node.cpp)
target_include_directories(image_event_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(image_event_node_lib
  rclcpp sensor_msgs video_system_interfaces)

add_executable(image_event_node_main
  src/image_event_main.cpp)
target_link_libraries(image_event_node_main
  image_event_node_lib)
ament_target_dependencies(image_event_node_main
  rclcpp sensor_msgs video_system_interfaces)

install(TARGETS
  image_event_node_lib
  image_event_node_main
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include/)

ament_package()
