from setuptools import setup

package_name = 'video_system_audio'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'librosa', 'onnxruntime', 'rclpy'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Audio capture, log-mel extraction, and event detection nodes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_feature_node = video_system_audio.audio_feature_node:main',
            'audio_event_node = video_system_audio.audio_event_node:main',
        ],
    },
)
