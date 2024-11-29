import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
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
        self.create_subscription(PointCloud2, '/sensor/radar/front/points', self.pointcloud_callback, 10)

        # Publishers for accumulated point cloud and initial pose
        self.accumulated_pcd_publisher = self.create_publisher(PointCloud2, '/accumulated_pointcloud', 10)
        self.initial_pose_publisher = self.create_publisher(PoseStamped, '/initialPose', 10)

        # Stacks to store incoming messages
        self.odom_stack = deque()
        self.pointcloud_stack = deque()

        self.accumulated_points = []
        self.current_position = None
        self.current_orientation = None  # Store current orientation quaternion
        self.initial_position = None
        self.initial_orientation = None  # Store initial orientation quaternion
        self.overall_offset = np.array([0.0, 0.0, 0.0])
        self.distance_threshold = self.declare_parameter('distance_threshold', 10.0).get_parameter_value().double_value
        self.pointcloud_directory = '/tmp_global/'

        # Ensure the directory exists
        if not os.path.exists(self.pointcloud_directory):
            os.makedirs(self.pointcloud_directory)

        # Lock for synchronizing access to accumulated points and orientation
        self.lock = threading.Lock()

        # Initialize accumulation pose variables
        self.accumulation_position = np.array([0.0, 0.0, 0.0])
        self.accumulation_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Store the latest odometry pose and timestamp
        self.latest_pose = None  # Will hold PoseStamped message

    def odom_callback(self, msg):
        self.get_logger().debug(f"Odometry message received. Stack size: {len(self.odom_stack)}")
        self.odom_stack.append(msg)
        if len(self.odom_stack) > 5 and len(self.pointcloud_stack) > 5:
            self.match_and_accumulate()

    def pointcloud_callback(self, msg):
        self.get_logger().debug(f"PointCloud message received. Stack size: {len(self.pointcloud_stack)}")
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

        # Extract yaw from quaternion
        _, _, yaw = euler_from_quaternion(quaternion)
        
        self.current_position = np.array([position.x, position.y, yaw])
        self.current_orientation = quaternion  # Store current orientation

        # Update the latest_pose with the current PoseStamped message
        self.latest_pose = PoseStamped()
        self.latest_pose.header = odom_msg.header  # Preserve original timestamp and frame_id
        self.latest_pose.pose.position = position
        self.latest_pose.pose.orientation.x = quaternion[0]
        self.latest_pose.pose.orientation.y = quaternion[1]
        self.latest_pose.pose.orientation.z = quaternion[2]
        self.latest_pose.pose.orientation.w = quaternion[3]

        # Set the initial position and orientation if not set
        if self.initial_position is None and self.initial_orientation is None:
            self.initial_position = self.current_position.copy()
            self.initial_orientation = self.current_orientation.copy()

        # Calculate the Euclidean distance traveled
        distance_traveled = np.linalg.norm(self.current_position[:2] - self.initial_position[:2])

        # Accumulate points from PointCloud2 message
        points = list(pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z")))
        self.get_logger().debug(f"Accumulating {len(points)} points from PointCloud message.")

        # Apply the transformation from odometry to the point cloud
        transformed_points = self.apply_transformation(points, position, quaternion)

        # Lock the access to accumulated_points
        with self.lock:
            self.get_logger().debug(f"Number of points to accumulate: {len(transformed_points)}")
            self.accumulated_points.extend(transformed_points)
            # Limit the number of points to avoid memory overflow
            max_points = 50000  # Limit to 50,000 points
            if len(self.accumulated_points) > max_points:
                self.accumulated_points = self.accumulated_points[-max_points:]

        # Accumulate points once distance exceeds threshold
        if distance_traveled >= self.distance_threshold:
            self.get_logger().info("Distance threshold exceeded. Processing accumulated points.")
            with self.lock:
                self.process_accumulated_points(odom_msg)
                self.accumulated_points = []
                # Capture the current pose before resetting
                self.accumulation_position = self.current_position.copy()
                self.accumulation_orientation = self.current_orientation.copy()
                # Reset the initial position and orientation, and add the current offset
                self.overall_offset += self.current_position[:3]
                self.initial_position = np.array([0.0, 0.0, 0.0])
                self.initial_orientation = np.array([0.0, 0.0, 0.0, 1.0])
                self.current_position = np.array([0.0, 0.0, 0.0])
                self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])

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

    def process_accumulated_points(self, odom_msg):
        self.get_logger().info("Processing accumulated points.")
        # Convert the accumulated points to a point cloud
        if len(self.accumulated_points) == 0:
            self.get_logger().info("No points to process.")
            return

        # Save the point cloud to a custom PCD file
        timestamp = odom_msg.header.stamp.sec  # Use the odometry message's timestamp
        pointcloud_filename = os.path.join(self.pointcloud_directory, f'accumulated_pointcloud_{timestamp}.pcd')
        self.save_to_pcd(pointcloud_filename, self.accumulated_points)
        
        # Save the overall offset as an index entry
        position_offset = self.overall_offset
        orientation = [0.0, 0.0, 0.0, 1.0]  # Reset orientation
        pose_str = f'{pointcloud_filename} {position_offset[0]},{position_offset[1]},{position_offset[2]},{orientation[0]},{orientation[1]},{orientation[2]},{orientation[3]}'
        index_filename = os.path.join(self.pointcloud_directory, 'pointcloud_index.txt')
        with open(index_filename, 'a') as f:
            f.write(pose_str + '\n')
        
        self.get_logger().info(f"Point cloud saved to {pointcloud_filename} and indexed in {index_filename}.")

        # Publish the accumulated point cloud
        self.publish_accumulated_pointcloud(pointcloud_filename)

        # Publish the initial pose with the latest odometry pose and its original timestamp
        self.publish_initial_pose(odom_msg)

    def save_to_pcd(self, filename, points):
        # Custom function to save points to a PCD file
        num_points = len(points)
        header = f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {num_points}\nDATA ascii\n"

        with open(filename, 'w') as f:
            f.write(header)
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def publish_accumulated_pointcloud(self, filename):
        # Read the saved PCD file and convert it to PointCloud2 message
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Skip the header
        data_lines = lines[11:]

        points = []
        for line in data_lines:
            x, y, z = map(float, line.strip().split())
            points.append([x, y, z])

        # Create PointCloud2 message
        header = PointCloud2().header
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Assuming the accumulated point cloud is in the 'map' frame

        # Define the PointCloud2 fields
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        ]

        # Convert points to PointCloud2
        accumulated_pcd_msg = pc2.create_cloud(header, fields, points)

        # Publish the accumulated point cloud
        self.accumulated_pcd_publisher.publish(accumulated_pcd_msg)
        self.get_logger().info(f"Published accumulated point cloud to /accumulated_pointcloud with {len(points)} points.")

    def publish_initial_pose(self, odom_msg):
        # Create a PoseStamped message with the latest odometry pose
        initial_pose_msg = PoseStamped()
        initial_pose_msg.header = odom_msg.header  # Preserve original timestamp and frame_id

        # Set position and orientation from the latest odometry message
        initial_pose_msg.pose.position = odom_msg.pose.pose.position
        initial_pose_msg.pose.orientation = odom_msg.pose.pose.orientation

        # Publish the initial pose
        self.initial_pose_publisher.publish(initial_pose_msg)
        self.get_logger().info("Published initial pose to /initialPose with the latest odometry pose and its original timestamp.")

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
