from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kiss_icp_localization',
            executable='kiss_icp_localization_node',
            name='kiss_icp_localization',
            output='screen',
            parameters=[]
        ),
    ])
