import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from tf_transformations import (
    euler_from_quaternion,
    quaternion_from_matrix,
    quaternion_matrix,
    euler_from_matrix,
    unit_vector,
)
from collections import deque
import os
import threading

class AccumulatePointCloudNode(Node):
    def __init__(self):
        super().__init__('accumulate_pointcloud_node')

        # Subscribers for odometry and point cloud
        self.create_subscription(Odometry, '/kiss/odometry', self.odom_callback, 10)
        self.create_subscription(PointCloud2, '/sensor/radar/front/points', self.pointcloud_callback, 10)

        # Stacks to store incoming messages
        self.odom_stack = deque()
        self.pointcloud_stack = deque()

        self.accumulated_points = []

        # Initialize transformation matrices
        self.preodom2 = np.identity(4)  # Previous odometry (initialized to identity)
        self.preodom = None             # Initial odometry (set on first callback)
        self.distance = 0.0             # Cumulative distance

        # Initialize overall transformation (cumulative pose)
        self.overall_transform = np.identity(4)

        self.m_param_nodeInterval = self.declare_parameter('node_interval', 1.0).get_parameter_value().double_value
        self.pointcloud_directory = '/tmp/'

        # Ensure the directory exists
        if not os.path.exists(self.pointcloud_directory):
            os.makedirs(self.pointcloud_directory)

        # Lock for synchronizing access to accumulated points
        self.lock = threading.Lock()

    def odom_callback(self, msg):
        self.get_logger().info("Odometry message received. Stack size: {}".format(len(self.odom_stack)))
        self.odom_stack.append(msg)
        if len(self.odom_stack) > 5 and len(self.pointcloud_stack) > 5:
            self.match_and_accumulate()

    def pointcloud_callback(self, msg):
        self.get_logger().info("PointCloud message received. Stack size: {}".format(len(self.pointcloud_stack)))
        self.pointcloud_stack.append(msg)
        if len(self.odom_stack) > 5 and len(self.pointcloud_stack) > 5:
            self.match_and_accumulate()

    def pose_to_transformation_matrix(self, position, orientation):
        # Normalize quaternion to avoid transformation issues
        quaternion = unit_vector([orientation.x, orientation.y, orientation.z, orientation.w])
        transformation_matrix = quaternion_matrix(quaternion)
        transformation_matrix[0:3, 3] = [position.x, position.y, position.z]
        return transformation_matrix

    def transformation_matrix_to_xyzrpy(self, matrix):
        # Extract translation
        x, y, z = matrix[0:3, 3]
        # Extract rotation matrix
        rotation_matrix = matrix[0:3, 0:3]
        # Convert rotation matrix to roll, pitch, yaw
        roll, pitch, yaw = euler_from_matrix(rotation_matrix)
        return np.array([x, y, z, roll, pitch, yaw])

    def match_and_accumulate(self):
        self.get_logger().info("Matching and accumulating messages.")
        # Pop the oldest messages from both stacks
        odom_msg = self.odom_stack.popleft()
        pointcloud_msg = self.pointcloud_stack.popleft()

        # Process odometry message
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        # Create transformation matrix from odometry
        crtodom = self.pose_to_transformation_matrix(position, orientation)

        # Set the initial odometry if it's not set
        if self.preodom is None:
            self.preodom = crtodom

        # Compute delta_d and diffodom
        delta_d = np.linalg.inv(self.preodom2).dot(crtodom)
        diffodom = np.linalg.inv(self.preodom).dot(crtodom)

        # Update overall transformation
        self.overall_transform = self.overall_transform.dot(delta_d)

        # Convert diffodom to x, y, z, roll, pitch, yaw
        diffodomxyz = self.transformation_matrix_to_xyzrpy(diffodom)

        # Transform point cloud using diffodom
        points = list(pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z")))
        transformed_points = self.apply_transformation(points, diffodom)

        # Compute diff_distance
        diff_distance = np.sqrt(diffodomxyz[0]**2 + diffodomxyz[1]**2)
        self.get_logger().info(f"diff_distance: {diff_distance}")

        # Accumulate transformed points
        with self.lock:
            self.accumulated_points.extend(transformed_points)
            # Limit the number of points to avoid memory overflow
            max_points = 50000  # Limit to 50,000 points
            if len(self.accumulated_points) > max_points:
                self.accumulated_points = self.accumulated_points[-max_points:]

        # Update preodom2
        self.preodom2 = crtodom

        # Compute delta_dxyz and update cumulative distance
        delta_dxyz = self.transformation_matrix_to_xyzrpy(delta_d)
        delta_distance = np.sqrt(delta_dxyz[0]**2 + delta_dxyz[1]**2)
        self.distance += delta_distance

        # Node generation condition
        if np.ceil(diff_distance) < self.m_param_nodeInterval:
            return

        # Process accumulated points
        self.get_logger().info("Distance threshold exceeded. Processing accumulated points.")
        with self.lock:
            self.process_accumulated_points()
            self.accumulated_points = []
            # Reset preodom and preodom2
            self.preodom = crtodom
            self.preodom2 = crtodom
            self.distance = 0.0

    def apply_transformation(self, points, transformation_matrix):
        # Apply the transformation to each point
        transformed_points = []
        for point in points:
            point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
            transformed_point = transformation_matrix.dot(point_homogeneous)
            transformed_points.append([transformed_point[0], transformed_point[1], transformed_point[2]])
        return transformed_points

    def process_accumulated_points(self):
        self.get_logger().info("Processing accumulated points.")
        # Convert the accumulated points to a point cloud
        if len(self.accumulated_points) == 0:
            self.get_logger().info("No points to process.")
            return

        # Save the point cloud to a custom PCD file
        timestamp = self.get_clock().now().to_msg().sec
        pointcloud_filename = os.path.join(self.pointcloud_directory, f'accumulated_pointcloud_{timestamp}.pcd')
        self.save_to_pcd(pointcloud_filename, self.accumulated_points)

        # Extract position and orientation from overall transformation
        position = self.overall_transform[0:3, 3]
        quaternion = quaternion_from_matrix(self.overall_transform)

        # Write the overall pose to the index file
        pose_str = f'{pointcloud_filename} {position[0]} {position[1]} {position[2]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}'
        index_filename = os.path.join(self.pointcloud_directory, 'pointcloud_index.txt')
        with open(index_filename, 'a') as f:
            f.write(pose_str + '\n')

        self.get_logger().info(f"Point cloud saved to {pointcloud_filename} and indexed in {index_filename}.")

    def save_to_pcd(self, filename, points):
        # Custom function to save points to a PCD file
        num_points = len(points)
        header = f"""# .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z
            SIZE 4 4 4
            TYPE F F F
            COUNT 1 1 1
            WIDTH {num_points}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {num_points}
            DATA ascii
            """

        with open(filename, 'w') as f:
            f.write(header)
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main(args=None):
    rclpy.init(args=args)
    node = AccumulatePointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
