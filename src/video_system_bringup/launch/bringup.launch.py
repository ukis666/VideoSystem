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
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[{
                'model_path': '/home/uki/VideoSystem/src/video_system_perception/models/YoloFineTuned.onnx',
                'visualize': True,
                'fake_stride': 12
            }]
        ),
    ])
