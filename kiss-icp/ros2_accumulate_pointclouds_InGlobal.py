import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_matrix, unit_vector
from collections import deque
import os
import threading

class AccumulatePointCloudNode(Node):
    def __init__(self):
        super().__init__('accumulate_pointcloud_node_global')

        # Subscribers for odometry and point cloud
        self.create_subscription(Odometry, '/kiss/odometry', self.odom_callback, 10)
        self.create_subscription(PointCloud2, '/sensor/radar/front/points_filtered', self.pointcloud_callback, 10)

        # Stacks to store incoming messages
        self.odom_stack = deque()
        self.pointcloud_stack = deque()

        self.accumulated_points = []
        self.current_position = None
        self.initial_position = None
        self.overall_offset = np.array([0.0, 0.0, 0.0])
        self.distance_threshold = self.declare_parameter('distance_threshold', 10.0).get_parameter_value().double_value
        self.pointcloud_directory = '/tmp_global/'

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

    def match_and_accumulate(self):
        self.get_logger().info("Matching and accumulating messages.")
        # Pop the oldest messages from both stacks
        odom_msg = self.odom_stack.popleft()
        pointcloud_msg = self.pointcloud_stack.popleft()

        # Process odometry message
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        # Normalize quaternion to avoid transformation issues
        quaternion = unit_vector([orientation.x, orientation.y, orientation.z, orientation.w])

        _, _, yaw = euler_from_quaternion(quaternion)
        
        self.current_position = np.array([position.x, position.y, yaw])

        # Set the initial position if it's not set
        if self.initial_position is None:
            self.initial_position = self.current_position

        # Calculate the Euclidean distance traveled
        distance_traveled = np.linalg.norm(self.current_position[:2] - self.initial_position[:2])

        # Accumulate points from PointCloud2 message
        points = list(pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z")))
        self.get_logger().info("Accumulating points from PointCloud message. Total accumulated points: {}".format(len(self.accumulated_points) + len(points)))

        # Apply the transformation from odometry to the point cloud
        transformed_points = self.apply_transformation(points, position, quaternion)

        # Lock the access to accumulated_points
        with self.lock:
            self.get_logger().info("Number of points to accumulate: {}".format(len(transformed_points)))
            self.accumulated_points.extend(transformed_points)
            # Limit the number of points to avoid memory overflow
            max_points = 50000  # Limit to 50,000 points
            if len(self.accumulated_points) > max_points:
                self.accumulated_points = self.accumulated_points[-max_points:]

        # Accumulate points once distance exceeds threshold
        if distance_traveled >= self.distance_threshold:
            self.get_logger().info("Distance threshold exceeded. Processing accumulated points.")
            with self.lock:
                self.process_accumulated_points()
                self.accumulated_points = []
                # Reset the initial position and add the current offset to the overall offset
                self.overall_offset += self.current_position[:3]
                self.initial_position = np.array([0.0, 0.0, 0.0])
                self.current_position = np.array([0.0, 0.0, 0.0])

    def apply_transformation(self, points, position, quaternion):
        # Create transformation matrix from odometry pose
        translation = [position.x, position.y, position.z]
        transformation_matrix = quaternion_matrix(quaternion)
        transformation_matrix[0:3, 3] = translation

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
        pointcloud_filename = os.path.join(self.pointcloud_directory, f'accumulated_pointcloud_{self.get_clock().now().to_msg().sec}.pcd')
        self.save_to_pcd(pointcloud_filename, self.accumulated_points)
        
        # Save the overall offset as an index entry
        position_offset = self.overall_offset
        orientation = [0.0, 0.0, 0.0, 1.0]  # Reset orientation
        pose_str = f'{pointcloud_filename} {position_offset[0]},{position_offset[1]},{position_offset[2]},{orientation[0]},{orientation[1]},{orientation[2]},{orientation[3]}'
        index_filename = os.path.join(self.pointcloud_directory, 'pointcloud_index.txt')
        with open(index_filename, 'a') as f:
            f.write(pose_str + '\n')
        
        self.get_logger().info(f"Point cloud saved to {pointcloud_filename} and indexed in {index_filename}.")

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
