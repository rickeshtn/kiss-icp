import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import os

class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')

        # Directory to save pointclouds
        self.save_dir = '/home/rickeshtn/Projects/berlin_company_data/ros1/'  # Update this path
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Initialize counter for sequential file naming
        self.counter = 0

        # Subscribe to the PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/sensor/radar/front/points',  # Update this topic if necessary
            self.listener_callback,
            10
        )
        self.get_logger().info('PointCloudSaver node has been started.')

    def listener_callback(self, msg):
        # Extract points from the PointCloud2 message
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            self.get_logger().warn("Received an empty PointCloud2 message.")
            return

        # Convert points to Open3D PointCloud format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Define the filename with sequential numbering
        filename = os.path.join(self.save_dir, f'pointcloud_{self.counter:05d}.pcd')

        # Save the PointCloud to a PCD file
        try:
            o3d.io.write_point_cloud(filename, pcd)
            self.get_logger().info(f'Saved point cloud to {filename}')
            self.counter += 1
        except Exception as e:
            self.get_logger().error(f'Failed to save point cloud: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down PointCloudSaver node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()