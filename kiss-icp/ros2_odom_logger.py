import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import os
import tf_transformations

class OdometrySubscriber(Node):
    def __init__(self):
        super().__init__('odometry_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/kiss/odometry',
            self.odometry_callback,
            10
        )
        self.csv_file_path = 'odom_data.csv'

        # Write CSV header
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp_sec', 'timestamp_nanosec',
                    'position_x', 'position_y', 'position_z',
                    'roll', 'pitch', 'yaw'
                ])

    def odometry_callback(self, msg):
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w

        # Convert quaternion to roll, pitch, yaw
        quaternion = (orientation_x, orientation_y, orientation_z, orientation_w)
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)

        # Write data to CSV
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp_sec, timestamp_nanosec,
                position_x, position_y, position_z,
                roll, pitch, yaw
            ])

def main(args=None):
    rclpy.init(args=args)
    odometry_subscriber = OdometrySubscriber()
    rclpy.spin(odometry_subscriber)
    odometry_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
