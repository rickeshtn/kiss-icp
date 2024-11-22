import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from tf_transformations import euler_from_quaternion
from collections import deque

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
        self.current_position = None
        self.initial_position = None
        self.distance_threshold = self.declare_parameter('distance_threshold', 10.0).get_parameter_value().double_value

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
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        
        self.current_position = np.array([position.x, position.y, yaw])

        # Set the initial position if it's not set
        if self.initial_position is None:
            self.initial_position = self.current_position

        # Calculate the Euclidean distance traveled
        distance_traveled = np.linalg.norm(self.current_position[:2] - self.initial_position[:2])

        # Accumulate points from PointCloud2 message
        points = list(pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z")))
        self.get_logger().info("Accumulating points from PointCloud message. Total accumulated points: {}".format(len(self.accumulated_points) + len(points)))
        self.accumulated_points.extend(points)

        # Accumulate points once distance exceeds threshold
        if distance_traveled >= self.distance_threshold:
            self.get_logger().info("Distance threshold exceeded. Processing accumulated points.")
            self.process_accumulated_points()
            self.accumulated_points = []
            self.initial_position = self.current_position

    def process_accumulated_points(self):
        self.get_logger().info("Processing accumulated points.")
        # Convert the accumulated points to a point cloud
        if len(self.accumulated_points) == 0:
            self.get_logger().info("No points to process.")
            return

        cloud_np = np.array([[p[0], p[1], p[2]] for p in self.accumulated_points], dtype=np.float32)
        cloud = cloud_np  # Placeholder: Replace with suitable point cloud handling for ROS 2

        # Save the point cloud to a file
        pointcloud_filename = f'/tmp/accumulated_pointcloud_{self.get_clock().now().to_msg().sec}.pcd'
        np.savetxt(pointcloud_filename, cloud, delimiter=' ', header='x y z', comments='')
        
        # Save the first pose as an index entry
        if self.initial_position is not None:
            position = self.initial_position[:2]
            orientation = self.odom_stack[-1].pose.pose.orientation
            pose_str = f'{pointcloud_filename} {position[0]},{position[1]},{self.odom_stack[-1].pose.pose.position.z},{orientation.x},{orientation.y},{orientation.z},{orientation.w}'
            index_filename = '/tmp/pointcloud_index.txt'
            with open(index_filename, 'a') as f:
                f.write(pose_str + '\n')
        
        self.get_logger().info(f"Point cloud saved to {pointcloud_filename} and indexed in {index_filename}.")

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
