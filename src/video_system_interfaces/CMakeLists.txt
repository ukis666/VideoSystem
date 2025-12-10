cmake_minimum_required(VERSION 3.16)
project(video_system_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/Event.msg"
  "msg/AudioPcm16.msg"
  "msg/AudioMel.msg"
  "msg/AudioEvent.msg"
  "msg/AudioEventArray.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
