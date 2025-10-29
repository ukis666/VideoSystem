from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('video_system_bringup'),
            'config', 'params.yaml'),
        description='YAML parameter file'
    )

    image_event = Node(
        package='video_system_core',
        executable='image_event_node_main',
        name='image_event_node',
        parameters=[params_file]
    )

    return LaunchDescription([declare_params, image_event])
