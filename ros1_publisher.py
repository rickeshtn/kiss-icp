#!/usr/bin/env python
import rospy
import pcl
import os
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import re

def load_point_cloud(file_path):
    # Load the point cloud from disk using pcl
    cloud = pcl.load(file_path)
    # Convert PCL PointCloud to NumPy array
    cloud_np = np.array(cloud.to_array())  # Convert to NumPy array
    return cloud_np

def convert_to_ros_msg(cloud_np):
    # Convert the NumPy array to a ROS PointCloud2 message
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"  # Change to appropriate frame id
    
    # Create PointCloud2 message
    ros_cloud = pc2.create_cloud_xyz32(header, cloud_np)
    return ros_cloud

def numerical_sort_key(filename):
    # Extract numbers from the filename using regex
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def publish_point_cloud(directory):
    # Initialize the ROS node
    rospy.init_node('pointcloud_publisher', anonymous=True)
    
    # Create a ROS publisher
    pub = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=10)
    
    # Get all .pcd files from the directory and sort them numerically
    pcd_files = sorted([f for f in os.listdir(directory) if f.endswith('.pcd')], key=numerical_sort_key)

    if not pcd_files:
        rospy.logwarn("No .pcd files found in the directory.")
        return

    rate = rospy.Rate(1)  # Publish at 1 Hz (adjust as needed)
    
    for pcd_file in pcd_files:
        # Full path of the PCD file
        file_path = os.path.join(directory, pcd_file)
        
        rospy.loginfo(f"Publishing point cloud from: {file_path}")
        
        # Load the point cloud from the file
        cloud_np = load_point_cloud(file_path)
        
        # Convert the point cloud to a ROS message
        ros_cloud = convert_to_ros_msg(cloud_np)
        
        # Publish the PointCloud2 message
        pub.publish(ros_cloud)
        
        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        point_cloud_directory = "/home/rickeshtn/Projects/berlin_company_data/pointclouds/"  # Change this to your directory
        publish_point_cloud(point_cloud_directory)
    except rospy.ROSInterruptException:
        pass