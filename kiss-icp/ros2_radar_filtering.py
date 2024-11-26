import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
from sensor_msgs_py import point_cloud2

class RadarFilterNode(Node):
    def __init__(self):
        super().__init__('radar_filter_node')
        
        # Subscribe to the radar point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/sensor/radar/front/points',  # Updated topic name
            self.point_cloud_callback,
            10)
        
        # Publisher for the filtered point cloud
        self.publisher = self.create_publisher(PointCloud2, '/sensor/radar/front/points_filtered', 10)
        
        self.get_logger().info('Radar Filter Node has been started.')

    def point_cloud_callback(self, msg):
        self.get_logger().debug('Received PointCloud2 message.')

        # Inspect message header
        self.get_logger().debug(f'Message Header: frame_id={msg.header.frame_id}, stamp={msg.header.stamp}')

        # Convert ROS PointCloud2 to NumPy array
        cloud_array = self.pointcloud2_to_numpy(msg)
        
        # Debug: Check the shape and dtype of the resulting NumPy array
        if cloud_array.size == 0:
            self.get_logger().warn('Converted NumPy array is empty.')
            return
        self.get_logger().debug(f'Converted NumPy array shape: {cloud_array.shape}, dtype: {cloud_array.dtype}')

        # Debug: Print first 5 points for inspection
        self.get_logger().debug(f'First 5 points:\n{cloud_array[:5]}')

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_array)
        
        # Log number of points before filtering
        num_points_before = len(pcd.points)
        self.get_logger().debug(f'Number of points before filtering: {num_points_before}')

        # Apply Voxel Grid Downsampling
        voxel_size = 0.1  # Adjust based on your data
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.get_logger().debug(f'Applied voxel downsampling with voxel size: {voxel_size}')
        num_points_voxel = len(pcd_down.points)
        self.get_logger().debug(f'Number of points after voxel downsampling: {num_points_voxel}')

        # Apply Statistical Outlier Removal
        nb_neighbors = 20  # Number of neighbors to analyze for each point
        std_ratio = 2.0    # Standard deviation multiplier
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_filtered = pcd_down.select_by_index(ind)
        self.get_logger().debug(f'Applied statistical outlier removal with nb_neighbors: {nb_neighbors}, std_ratio: {std_ratio}')
        num_points_filtered = len(pcd_filtered.points)
        self.get_logger().debug(f'Number of points after statistical outlier removal: {num_points_filtered}')

        # Apply Ground Plane Removal
        pcd_final = self.ground_plane_removal(pcd_filtered, distance_threshold=0.2)
        num_points_ground_removed = len(pcd_final.points)
        self.get_logger().debug(f'Number of points after ground plane removal: {num_points_ground_removed}')

        # Convert back to ROS PointCloud2
        msg_filtered = self.numpy_to_pointcloud2(np.asarray(pcd_final.points), msg.header)
        
        # Debug: Inspect the size of the filtered message
        self.get_logger().debug(f'Filtered PointCloud2 message size: {len(msg_filtered.data)} bytes')

        # Publish the filtered point cloud
        self.publisher.publish(msg_filtered)
        self.get_logger().debug('Published filtered PointCloud2 message.')

    def pointcloud2_to_numpy(self, cloud_msg):
        """
        Converts a ROS PointCloud2 message to a NumPy array containing x, y, z coordinates.
        """
        try:
            # Extract the XYZ fields from the PointCloud2 message
            points = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points:
                self.get_logger().warn('No valid points found in PointCloud2 message.')
                return np.empty((0, 3), dtype=np.float32)
            
            # Use list comprehension to ensure it's a list of lists, not a list of tuples with named fields
            points_np = np.array([ [x, y, z] for x, y, z in points ], dtype=np.float32)
            return points_np  # Now it's a (N, 3) array of float32
        except Exception as e:
            self.get_logger().error(f'Error converting PointCloud2 to NumPy array: {e}')
            return np.empty((0, 3), dtype=np.float32)

    def numpy_to_pointcloud2(self, points, header):
        """
        Converts a NumPy array containing x, y, z coordinates to a ROS PointCloud2 message.
        """
        try:
            # Define the PointCloud2 fields
            fields = [
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            ]
            # Create the PointCloud2 message
            msg = point_cloud2.create_cloud(header, fields, points)
            return msg
        except Exception as e:
            self.get_logger().error(f'Error converting NumPy array to PointCloud2: {e}')
            return PointCloud2()

    def ground_plane_removal(self, pcd, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
        """
        Removes the ground plane from the point cloud using RANSAC.
        """
        try:
            # Segment the largest planar component
            plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
            [a, b, c, d] = plane_model
            self.get_logger().debug(f'Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')
            
            # Select points that are not part of the plane
            pcd_without_ground = pcd.select_by_index(inliers, invert=True)
            return pcd_without_ground
        except Exception as e:
            self.get_logger().error(f'Error during ground plane removal: {e}')
            return pcd  # Return the original point cloud if an error occurs

def main(args=None):
    rclpy.init(args=args)
    node = RadarFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Radar Filter Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
