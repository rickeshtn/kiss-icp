#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam2d/types_slam2d.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <mutex>

class GraphOptimizationNode : public rclcpp::Node
{
public:
    GraphOptimizationNode()
        : Node("graph_optimization_node")
    {
        radar_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/kiss/frame", 10, std::bind(&GraphOptimizationNode::pointCloudCallback, this, std::placeholders::_1));

        // Initialize g2o optimizer
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto linearSolver = std::make_unique<LinearSolverType>();
        auto solver_ptr = std::make_unique<BlockSolverType>(std::move(linearSolver));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

        optimizer_.setAlgorithm(solver);
        optimizer_.setVerbose(false);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received point cloud message with %d points", cloud_msg->width * cloud_msg->height);

        std::lock_guard<std::mutex> lock(mutex_);

        // Process point cloud data (e.g., use for creating nodes in the graph)
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*cloud_msg, cloud);
        radar_points_.push_back(cloud);

        // Manually compute the centroid of the point cloud
        double x_sum = 0.0;
        double y_sum = 0.0;
        int point_count = cloud.points.size();

        for (const auto& point : cloud.points)
        {
            x_sum += point.x;
            y_sum += point.y;
        }

        double x = x_sum / point_count;
        double y = y_sum / point_count;
        double theta = 0.0; // Assuming flat terrain, no orientation information available

        // Add node to the graph
        g2o::VertexSE2* vertex = new g2o::VertexSE2();
        vertex->setId(current_vertex_id_++);
        vertex->setEstimate(g2o::SE2(x, y, theta));
        optimizer_.addVertex(vertex);

        // Add edge to previous node for optimization constraint
        if (previous_vertex_id_ >= 0)
        {
            g2o::EdgeSE2* edge = new g2o::EdgeSE2();
            edge->setVertex(0, optimizer_.vertex(previous_vertex_id_));
            edge->setVertex(1, optimizer_.vertex(current_vertex_id_ - 1));
            edge->setMeasurement(g2o::SE2(x - previous_pose_.x, y - previous_pose_.y, theta - previous_pose_.theta));
            edge->setInformation(Eigen::Matrix3d::Identity());
            optimizer_.addEdge(edge);
        }

        previous_vertex_id_ = current_vertex_id_ - 1;
        previous_pose_ = {x, y, theta};

        // Perform loop closure detection
        detectLoopClosure();
    }

    void detectLoopClosure()
    {
        // Simple loop closure detection: check proximity to previous poses
        if (current_vertex_id_ < 5)
            return;

        auto current_vertex = dynamic_cast<g2o::VertexSE2*>(optimizer_.vertex(current_vertex_id_ - 1));
        g2o::SE2 current_pose = current_vertex->estimate();

        for (int i = 0; i < current_vertex_id_ - 5; ++i)
        {
            auto previous_vertex = dynamic_cast<g2o::VertexSE2*>(optimizer_.vertex(i));
            g2o::SE2 previous_pose = previous_vertex->estimate();

            double distance = (current_pose.translation() - previous_pose.translation()).norm();
            if (distance < loop_closure_threshold_)
            {
                // Add loop closure edge
                g2o::EdgeSE2* edge = new g2o::EdgeSE2();
                edge->setVertex(0, optimizer_.vertex(i));
                edge->setVertex(1, optimizer_.vertex(current_vertex_id_ - 1));
                edge->setMeasurement(current_pose.inverse() * previous_pose);
                edge->setInformation(Eigen::Matrix3d::Identity() * 10.0);
                optimizer_.addEdge(edge);

                RCLCPP_INFO(this->get_logger(), "Loop closure detected between vertex %d and vertex %d", i, current_vertex_id_ - 1);
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_points_sub_;

    g2o::SparseOptimizer optimizer_;
    int current_vertex_id_ = 0;
    int previous_vertex_id_ = -1;
    struct Pose
    {
        double x, y, theta;
    } previous_pose_;

    double loop_closure_threshold_ = 5.0; // meters
    std::vector<pcl::PointCloud<pcl::PointXYZ>> radar_points_;
    std::mutex mutex_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GraphOptimizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
