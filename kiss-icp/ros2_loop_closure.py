import rclpy
from rclpy.node import Node
import numpy as np
import os
import ceres_python_bindings as ceres
from tf_transformations import unit_vector

class LoopClosureNode(Node):
    def __init__(self):
        super().__init__('loop_closure_node')
        self.pointcloud_directory = '/tmp/'
        self.index_filename = os.path.join(self.pointcloud_directory, 'pointcloud_index.txt')
        self.pointcloud_files = []
        self.poses = []

        # Load the point clouds and poses from the index file
        self.load_pointclouds_and_poses()

        # Perform loop closure optimization
        self.perform_loop_closure()

    def load_pointclouds_and_poses(self):
        if not os.path.exists(self.index_filename):
            self.get_logger().error(f"Index file not found: {self.index_filename}")
            return

        with open(self.index_filename, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                pointcloud_path = parts[0]
                pose_values = list(map(float, parts[1].split(',')))

                # Normalize quaternion to avoid issues
                quaternion = unit_vector([pose_values[3], pose_values[4], pose_values[5], pose_values[6]])

                self.pointcloud_files.append(pointcloud_path)
                position = np.array([pose_values[0], pose_values[1], pose_values[2]])
                orientation = np.array(quaternion)
                self.poses.append((position, orientation))

    def perform_loop_closure(self):
        if len(self.poses) < 2:
            self.get_logger().error("Not enough poses to perform loop closure.")
            return

        # Set up the Ceres problem
        problem = ceres.Problem()

        # Add prior for the first pose
        position, orientation = self.poses[0]
        initial_pose = np.concatenate((position, orientation))
        loss_function = ceres.HuberLoss(1.0)
        problem.add_residual_block(
            ceres.CostFunctionFactory.create_prior_cost_function(initial_pose),
            loss_function,
            initial_pose
        )

        # Add odometry constraints between consecutive poses
        for i in range(1, len(self.poses)):
            prev_position, prev_orientation = self.poses[i - 1]
            current_position, current_orientation = self.poses[i]

            relative_position = current_position - prev_position
            relative_orientation = current_orientation - prev_orientation

            # Create a parameter block for the current pose
            pose_param = np.concatenate((current_position, current_orientation))
            problem.add_parameter_block(pose_param, len(pose_param))

            # Add residual block for odometry
            cost_function = ceres.CostFunctionFactory.create_odometry_cost_function(
                relative_position, relative_orientation
            )
            problem.add_residual_block(cost_function, loss_function, pose_param)

        # Configure the solver
        options = ceres.SolverOptions()
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = True

        # Solve the problem
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        self.get_logger().info(f"Ceres Solver Summary:\n{summary.BriefReport()}")

        # Update poses with optimized values
        optimized_poses = []
        for i in range(len(self.poses)):
            position = problem.parameter_block[i][:3]
            orientation = problem.parameter_block[i][3:]
            optimized_poses.append((position, orientation))
            self.get_logger().info(f"Optimized Pose {i}: Position: {position}, Orientation: {orientation}")

        # Save the optimized poses to a new index file
        optimized_index_filename = os.path.join(self.pointcloud_directory, 'optimized_pointcloud_index.txt')
        with open(optimized_index_filename, 'w') as f:
            for i, (position, orientation) in enumerate(optimized_poses):
                pose_str = f"{self.pointcloud_files[i]} {position[0]},{position[1]},{position[2]},{orientation[0]},{orientation[1]},{orientation[2]},{orientation[3]}"
                f.write(pose_str + '\n')

        self.get_logger().info(f"Optimized poses saved to {optimized_index_filename}")

def main(args=None):
    rclpy.init(args=args)
    node = LoopClosureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
