import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
import time
import os

class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kiss/frame',
            self.listener_callback,
            10
        )
        self.pointcloud_list = []
        self.start_time = time.time()
        self.output_directory = 'pointclouds'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def listener_callback(self, msg):
        current_time = time.time()
        points = self.convert_pointcloud2_to_array(msg)
        self.pointcloud_list.extend(points)

        if current_time - self.start_time >= 60.0:
            # Accumulate and save every 1 minute
            self.save_pointcloud()
            self.pointcloud_list = []  # Reset the list
            self.start_time = current_time

    def convert_pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data from PointCloud2 message
        fmt = 'fff'  # for x, y, z (3 floats)
        points = []
        for i in range(cloud_msg.width):
            offset = i * cloud_msg.point_step
            x, y, z = struct.unpack_from(fmt, cloud_msg.data, offset)
            points.append([x, y, z])
        return points

    def save_pointcloud(self):
        if not self.pointcloud_list:
            self.get_logger().info('No pointcloud data to save')
            return

        # Save the accumulated point cloud data to a PCD file
        timestamp = int(time.time())
        filename = os.path.join(self.output_directory, f'pointcloud_{timestamp}.pcd')
        self.save_to_pcd(filename, self.pointcloud_list)
        self.get_logger().info(f'Saved pointcloud to {filename}')

    def save_to_pcd(self, filename, points):
        # Custom function to save points to a PCD file
        num_points = len(points)
        header = f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {num_points}\nDATA ascii\n"

        with open(filename, 'w') as f:
            f.write(header)
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")


def main(args=None):
    rclpy.init(args=args)
    pointcloud_accumulator = PointCloudAccumulator()
    rclpy.spin(pointcloud_accumulator)
    pointcloud_accumulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
