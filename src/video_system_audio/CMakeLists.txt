cmake_minimum_required(VERSION 3.16)
project(video_system_audio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(video_system_interfaces REQUIRED)
find_package(alsa REQUIRED)

ament_python_install_package(${PROJECT_NAME})

add_executable(audio_capture_node src/audio_capture_node.cpp)
ament_target_dependencies(audio_capture_node rclcpp video_system_interfaces alsa)
install(TARGETS audio_capture_node DESTINATION lib/${PROJECT_NAME})

install(
  PROGRAMS
    video_system_audio/audio_feature_node.py
    video_system_audio/audio_event_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
