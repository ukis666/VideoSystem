cmake_minimum_required(VERSION 3.16)
project(video_system_perception)

# ---- C++ tarafı (opsiyonel; derlenir ama çalıştırmak zorunda değilsin) ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)      # cv_bridge açık kalacak
find_package(OpenCV REQUIRED)
find_package(video_system_interfaces REQUIRED)

add_executable(detector_node
        src/main.cpp
        src/detector_node.cpp
)

ament_target_dependencies(detector_node
        rclcpp
        sensor_msgs
        cv_bridge
        video_system_interfaces
)

target_link_libraries(detector_node ${OpenCV_LIBS})

target_include_directories(detector_node PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

install(TARGETS detector_node
        RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ---- Python dedektör ----
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS
        scripts/detector_node_py.py
        DESTINATION lib/${PROJECT_NAME}
)

ament_package()
