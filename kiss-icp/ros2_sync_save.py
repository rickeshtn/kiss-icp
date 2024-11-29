import rclpy
from rclpy.node import Node
import message_filters
import json
import os
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

# Create an output directory for saving synchronized data
output_dir = "synced_data"
os.makedirs(output_dir, exist_ok=True)

# Callback function for synchronized topics
def callback(odom_msg, pointcloud_msg):
    # Extract timestamp and use it for filenames
    timestamp = odom_msg.header.stamp.sec * 1_000_000_000 + odom_msg.header.stamp.nanosec
    file_base_name = f"{timestamp}"

    # Save odometry data to a JSON file
    odom_data = {
        "header": {
            "stamp": {
                "sec": int(odom_msg.header.stamp.sec),
                "nanosec": int(odom_msg.header.stamp.nanosec)
            },
            "frame_id": str(odom_msg.header.frame_id)
        },
        "child_frame_id": str(odom_msg.child_frame_id),
        "pose": {
            "position": {
                "x": float(odom_msg.pose.pose.position.x),
                "y": float(odom_msg.pose.pose.position.y),
                "z": float(odom_msg.pose.pose.position.z)
            },
            "orientation": {
                "x": float(odom_msg.pose.pose.orientation.x),
                "y": float(odom_msg.pose.pose.orientation.y),
                "z": float(odom_msg.pose.pose.orientation.z),
                "w": float(odom_msg.pose.pose.orientation.w)
            }
        },
        "covariance": [float(value) for value in odom_msg.pose.covariance],
        "twist": {
            "linear": {
                "x": float(odom_msg.twist.twist.linear.x),
                "y": float(odom_msg.twist.twist.linear.y),
                "z": float(odom_msg.twist.twist.linear.z)
            },
            "angular": {
                "x": float(odom_msg.twist.twist.angular.x),
                "y": float(odom_msg.twist.twist.angular.y),
                "z": float(odom_msg.twist.twist.angular.z)
            }
        },
        "twist_covariance": [float(value) for value in odom_msg.twist.covariance]
    }
    odom_file_path = os.path.join(output_dir, f"{file_base_name}_odom.json")
    with open(odom_file_path, 'w') as odom_file:
        json.dump(odom_data, odom_file, indent=4)

    # Save point cloud data to a JSON file
    pointcloud_data = []
    for point in pc2.read_points(pointcloud_msg, skip_nans=True):
        point_data = {
            "azimuth_angle": float(point[0]),
            "azimuth_angle_std": float(point[1]),
            "invalid_flags": int(point[2]),
            "elevation_angle": float(point[3]),
            "elevation_angle_std": float(point[4]),
            "range": float(point[5]),
            "range_std": float(point[6]),
            "range_rate": float(point[7]),
            "range_rate_std": float(point[8]),
            "rcs": float(point[9]),
            "measurement_id": int(point[10]),
            "positive_predictive_value": int(point[11]),
            "classification": int(point[12]),
            "multi_target_probability": int(point[13]),
            "object_id": int(point[14]),
            "ambiguity_flag": int(point[15]),
            "sort_index": int(point[16]),
            "x": float(point[17]),
            "y": float(point[18]),
            "z": float(point[19])
        }
        pointcloud_data.append(point_data)
    pointcloud_file_path = os.path.join(output_dir, f"{file_base_name}_points.json")
    with open(pointcloud_file_path, 'w') as pointcloud_file:
        json.dump(pointcloud_data, pointcloud_file, indent=4)

    rclpy.logging.get_logger("sync_and_save_node").info(f"Saved synchronized data: {file_base_name}")

# Main class to initialize node and subscribers
class SyncAndSaveNode(Node):
    def __init__(self):
        super().__init__('sync_and_save_node')

        # Subscribers for the topics
        odom_sub = message_filters.Subscriber(self, Odometry, '/kiss/odometry')
        pointcloud_sub = message_filters.Subscriber(self, PointCloud2, '/sensor/radar/front/points')

        # ApproximateTimeSynchronizer to sync topics based on timestamp
        sync = message_filters.ApproximateTimeSynchronizer([odom_sub, pointcloud_sub], queue_size=10, slop=0.1)
        sync.registerCallback(callback)

        self.get_logger().info("Started syncing /kiss/odometry and /sensor/radar/front/points")

# Main function to run the node
def main(args=None):
    rclpy.init(args=args)
    node = SyncAndSaveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
