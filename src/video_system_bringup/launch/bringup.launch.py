from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='video_system_core',
            executable='image_event_node_main',
            name='image_event_node',
            output='screen',
            parameters=[{'heartbeat_stride': 15}]
        ),
        Node(
            package='video_system_perception',
            executable='detector_node_py',
            name='detector_node_py',
            output='screen',
            parameters=[{
                'model_path': '/home/uki/VideoSystem/src/video_system_perception/models/YoloFineTuned.onnx',
                'visualize': True,
                'conf_threshold': 0.25,
                'nms_threshold': 0.5,
                'frame_rate': 15.0,
                'use_homography': False,
                'homography_path': '/home/uki/VideoSystem/src/video_system_perception/config/homography_room1.npy',
            }]
        ),
    ])
