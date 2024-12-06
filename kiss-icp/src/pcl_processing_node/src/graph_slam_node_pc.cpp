#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <Eigen/Dense>
#include <queue>
#include <cmath>
#include <mutex>

class GraphSlamNode : public rclcpp::Node {
public:
    GraphSlamNode() : Node("graph_slam_node"), vertex_count_(0), loop_closure_triggered_(false) {
        // Subscriptions
        radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sensor/radar/front/points_filtered", 10,
            std::bind(&GraphSlamNode::radarCallback, this, std::placeholders::_1));
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/kiss/odometry", 10,
            std::bind(&GraphSlamNode::odometryCallback, this, std::placeholders::_1));

        // Initialize ICP
        initializeICP();

        // Initialize the optimizer
        initializeGraphOptimizer();

        // Initialize timer to check for inactivity
        timer_ = this->create_wall_timer(
            std::chrono::seconds(10),
            std::bind(&GraphSlamNode::timerCallback, this));

        // Initialize last_data_time_ and loop_closure_triggered_
        last_data_time_ = this->now();
        loop_closure_triggered_ = false;

        // Initialize last odometry variables
        last_odometry_initialized_ = false;
        last_odometry_time_ = this->now();
        last_position_ = Eigen::Vector3d::Zero();
        last_yaw_ = 0.0;

        RCLCPP_INFO(this->get_logger(), "[INIT] Graph SLAM Node Initialized");
    }

private:
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;

    // Graph Optimizer
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
    int vertex_count_;

    // Data storage
    std::vector<Eigen::Isometry3d> pose_history_; // Stores odometry poses
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scan_history_; // Stores point clouds

    // Queues for odometry and point clouds
    std::queue<Eigen::Isometry3d> odometry_queue_;
    std::queue<pcl::PointCloud<pcl::PointXYZ>::Ptr> pointcloud_queue_;

    // Thresholds and parameters
    const size_t MAX_QUEUE_SIZE = 2700; // Adjust as needed
    Eigen::Isometry3d last_odometry_in_queue_ = Eigen::Isometry3d::Identity(); // Last odometry added to the queue
    Eigen::Isometry3d last_added_pose_ = Eigen::Isometry3d::Identity(); // Last pose added to the graph

    // ICP for scan matching
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

    // Timer for inactivity
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_data_time_;
    bool loop_closure_triggered_;

    // Mutex for thread safety (if multi-threaded spinning is used)
    std::mutex mtx_;

    // Variables for odometry processing
    rclcpp::Time last_odometry_time_;
    Eigen::Vector3d last_position_;
    double last_yaw_;
    bool last_odometry_initialized_;

    // Thresholds for speed and yaw rate
    double speed_threshold_ = 0.3; // m/s, adjust as needed
    double yaw_rate_threshold_ = 0.0349066; // rad/s, adjust as needed
    double max_linear_speed_ = 30.0; // m/s, maximum allowable speed
    double max_yaw_rate_ = 0.17; // rad/s, maximum allowable yaw rate

    void initializeICP() {
        // Initialize ICP parameters
        icp_.setMaximumIterations(50);
        icp_.setTransformationEpsilon(1e-6);
        icp_.setEuclideanFitnessEpsilon(1e-6);
        icp_.setMaxCorrespondenceDistance(2.0); // Increased for sparse data
    }

    void initializeGraphOptimizer() {
        auto linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        auto blockSolver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
        auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
        optimizer_ = std::make_unique<g2o::SparseOptimizer>();
        optimizer_->setAlgorithm(solver);

        if (!optimizer_) {
            RCLCPP_ERROR(this->get_logger(), "[ERROR] Graph optimizer initialization failed!");
        } else {
            RCLCPP_INFO(this->get_logger(), "[INIT] Graph optimizer successfully initialized.");
        }
    }

    double getYawFromQuaternion(const Eigen::Quaterniond &q) {
        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
        return euler[2]; // Return only yaw
    }

    void saveGraph(const std::string &filename) {
        if (optimizer_) {
            RCLCPP_INFO(this->get_logger(), "[SAVE] Saving graph to file: %s", filename.c_str());
            optimizer_->save(filename.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "[ERROR] Optimizer is not initialized. Cannot save graph.");
        }
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        RCLCPP_INFO(this->get_logger(), "[ODOM] Received odometry data.");
        RCLCPP_INFO(this->get_logger(), "[ODOM] Header: frame_id = %s, stamp = %u.%u",
                    msg->header.frame_id.c_str(),
                    msg->header.stamp.sec,
                    msg->header.stamp.nanosec);

        // Convert odometry to Eigen::Isometry3d
        Eigen::Vector3d translation(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z);
        Eigen::Quaterniond rotation(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z);

        // Get current time from message
        rclcpp::Time current_time = msg->header.stamp;

        if (!last_odometry_initialized_) {
            last_odometry_time_ = current_time;
            last_position_ = translation;
            last_yaw_ = getYawFromQuaternion(rotation);
            last_odometry_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "[ODOM] Initialized last odometry variables.");
            return;
        }

        double delta_time = (current_time - last_odometry_time_).seconds();

        // If delta_time is zero or negative, skip to avoid division by zero
        if (delta_time <= 0) {
            RCLCPP_WARN(this->get_logger(), "[ODOM] Delta time is zero or negative. Skipping.");
            return;
        }

        // Compute distance moved since last odometry message
        double distance = (translation - last_position_).norm();

        // Compute current yaw
        double current_yaw = getYawFromQuaternion(rotation);

        // Compute yaw difference since last odometry message
        double yaw_difference = current_yaw - last_yaw_;

        // Normalize yaw difference to [-pi, pi]
        if (yaw_difference > M_PI)
            yaw_difference -= 2 * M_PI;
        else if (yaw_difference < -M_PI)
            yaw_difference += 2 * M_PI;

        // Compute linear speed and yaw rate
        double linear_speed = distance / delta_time;
        double yaw_rate = yaw_difference / delta_time;

        RCLCPP_INFO(this->get_logger(), "[ODOM] Linear Speed: %.2f m/s, Yaw Rate: %.2f rad/s", linear_speed, yaw_rate);

        // Ignore sudden spikes in linear speed
        if (linear_speed > max_linear_speed_) {
            RCLCPP_WARN(this->get_logger(), "[ODOM] Linear speed spike detected (%.2f m/s). Ignoring odometry.", linear_speed);
            return;
        }

        // Ignore sudden spikes in yaw rate
        if (std::abs(yaw_rate) > max_yaw_rate_) {
            RCLCPP_WARN(this->get_logger(), "[ODOM] Yaw rate spike detected (%.2f rad/s). Ignoring odometry.", yaw_rate);
            return;
        }

        // Check if speed and yaw rate exceed thresholds
        if (linear_speed >= speed_threshold_ || std::abs(yaw_rate) > yaw_rate_threshold_) {
            // Movement exceeds thresholds, process odometry

            // Create current_pose Isometry3d
            Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();
            current_pose.translate(translation);
            current_pose.rotate(rotation);

            odometry_queue_.push(current_pose);
            last_odometry_in_queue_ = current_pose;
            RCLCPP_INFO(this->get_logger(), "[ODOM] Odometry added to queue. Queue size: %lu", odometry_queue_.size());

            // Update last_data_time_ and reset loop_closure_triggered_
            last_data_time_ = this->now();
            loop_closure_triggered_ = false;
        } else {
            RCLCPP_INFO(this->get_logger(), "[ODOM] Movement below thresholds. Odometry skipped.");
        }

        // Update last odometry time, position, and yaw
        last_odometry_time_ = current_time;
        last_position_ = translation;
        last_yaw_ = current_yaw;
    }

    void radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        RCLCPP_INFO(this->get_logger(), "[RADAR] Received radar point cloud.");
        RCLCPP_INFO(this->get_logger(), "[RADAR] Header: frame_id = %s, stamp = %u.%u",
                    msg->header.frame_id.c_str(),
                    msg->header.stamp.sec,
                    msg->header.stamp.nanosec);

        pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *current_scan);

        if (current_scan->empty()) {
            RCLCPP_WARN(this->get_logger(), "[RADAR] Received an empty point cloud. Skipping.");
            return;
        }

        pointcloud_queue_.push(current_scan);
        RCLCPP_INFO(this->get_logger(), "[RADAR] Point cloud added to queue. Queue size: %lu", pointcloud_queue_.size());

        // Update last_data_time_ and reset loop_closure_triggered_
        last_data_time_ = this->now();
        loop_closure_triggered_ = false;

        processSensorData();
    }

    void processSensorData() {
        while (!odometry_queue_.empty() && !pointcloud_queue_.empty()) {
            Eigen::Isometry3d current_pose = odometry_queue_.front();
            odometry_queue_.pop();
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan = pointcloud_queue_.front();
            pointcloud_queue_.pop();

            // Add the odometry as a vertex and associate the point cloud
            addVertex(current_pose, pose_history_.empty()); // First pose is fixed
            pose_history_.push_back(current_pose);
            scan_history_.push_back(current_scan);

            // Add edge
            if (vertex_count_ > 1) {
                Eigen::Isometry3d relative_pose = last_added_pose_.inverse() * current_pose;
                addEdge(vertex_count_ - 2, vertex_count_ - 1, relative_pose);
            }

            last_added_pose_ = current_pose;
            RCLCPP_INFO(this->get_logger(), "[PROCESS] Pose and point cloud added to graph. Total poses: %lu, Total scans: %lu",
                        pose_history_.size(), scan_history_.size());

            // Start loop closure detection if needed
            if (pose_history_.size() >= MAX_QUEUE_SIZE) {
                detectLoopClosure();
            }
        }
    }

    void detectLoopClosure() {
        RCLCPP_INFO(this->get_logger(), "[LOOP] Starting loop closure detection...");
        Eigen::Isometry3d current_pose = pose_history_.back();
        bool loop_found = false;

        // Parameters for loop closure detection
        const double DISTANCE_Y_THRESHOLD = 25.0; // X meters
        const double YAW_THRESHOLD = 0.0872665 / 2; // Y degrees (5 degrees = 0.0872665 radians)
        const double DISTANCE_Z_THRESHOLD = 5.0;

        for (size_t i = 2; i < pose_history_.size()-2; ++i) { // Skip recent poses to avoid false positives
            double delta_y = std::abs(current_pose.translation().y() - pose_history_[i].translation().y());
            double delta_z = std::abs(current_pose.translation().z() - pose_history_[i].translation().z());
            double current_yaw = getYawFromQuaternion(Eigen::Quaterniond(current_pose.rotation()));
            double previous_yaw = getYawFromQuaternion(Eigen::Quaterniond(pose_history_[i].rotation()));
            double yaw_difference = std::abs(current_yaw - previous_yaw);

            // Normalize yaw difference to [0, pi]
            if (yaw_difference > M_PI) {
                yaw_difference = 2 * M_PI - yaw_difference;
            }

            RCLCPP_INFO(this->get_logger(), "[LOOP] Checking pose %lu: Distance Y: %.2f, Distance Z: %.2f, Yaw Diff: %.2f radians",
                        i, delta_y, delta_z, yaw_difference);

            // Check if both conditions are satisfied
            if (delta_y < DISTANCE_Y_THRESHOLD && yaw_difference < YAW_THRESHOLD && delta_z < DISTANCE_Z_THRESHOLD) {
                RCLCPP_INFO(this->get_logger(), "[LOOP] Potential loop closure between pose %lu and %lu", i, pose_history_.size() - 1);

                pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan = scan_history_.back();
                pcl::PointCloud<pcl::PointXYZ>::Ptr previous_scan = scan_history_[i];

                // Preprocess scans
                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_current_scan(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_previous_scan(new pcl::PointCloud<pcl::PointXYZ>());

                preprocessPointCloud(current_scan, filtered_current_scan);
                preprocessPointCloud(previous_scan, filtered_previous_scan);

                // Provide initial guess using odometry
                Eigen::Matrix4f initial_guess = (pose_history_[i].inverse() * current_pose).matrix().cast<float>();

                // Perform ICP for confirmation
                icp_.setInputSource(filtered_current_scan);
                icp_.setInputTarget(filtered_previous_scan);
                pcl::PointCloud<pcl::PointXYZ> aligned_scan;
                icp_.align(aligned_scan, initial_guess);
                std::cout<<"icp_.hasConverged() && icp_.getFitnessScore() "<<icp_.hasConverged()<<","<<icp_.getFinalTransformation()<<std::endl;
                if (icp_.hasConverged() && icp_.getFitnessScore() < 2.5) { // Adjust fitness score threshold as needed
                    Eigen::Matrix4f correction = icp_.getFinalTransformation();
                    Eigen::Isometry3d loop_transform(correction.cast<double>());

                    addEdge(i, vertex_count_ - 1, loop_transform);

                    // Optimize the graph
                    optimizer_->initializeOptimization();
                    optimizer_->optimize(10);

                    generateAndSaveCorrectedPointCloud();
                    RCLCPP_INFO(this->get_logger(), "[LOOP] Loop closure applied and graph optimized.");

                    loop_found = true;
                    break;
                } else {
                    RCLCPP_WARN(this->get_logger(), "[LOOP] ICP failed or fitness score too high. Skipping loop closure.");
                }
            }
        }

        if (!loop_found) {
            RCLCPP_INFO(this->get_logger(), "[LOOP] No loop closure detected.");
        }
    }

    void preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &output) {
        // Voxel Grid Downsampling
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f); // Adjust leaf size as needed
        voxel_filter.setInputCloud(input);
        voxel_filter.filter(*output);

        // Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(output);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*output);
    }

    void generateAndSaveCorrectedPointCloud() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>());

        for (int i = 0; i < vertex_count_; ++i) {
            auto vertex = dynamic_cast<g2o::VertexSE3*>(optimizer_->vertex(i));
            Eigen::Isometry3d optimized_pose = vertex->estimate();

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*scan_history_[i], *transformed_scan, optimized_pose.matrix().cast<float>());
            *global_map += *transformed_scan;
        }

        pcl::io::savePCDFileBinary("corrected_pointcloud.pcd", *global_map);
        RCLCPP_INFO(this->get_logger(), "[OUTPUT] Corrected point cloud saved to corrected_pointcloud.pcd");
    }

    void addVertex(const Eigen::Isometry3d &pose, bool fixed = false) {
        auto vertex = new g2o::VertexSE3();
        vertex->setId(vertex_count_++);
        vertex->setEstimate(pose);
        vertex->setFixed(fixed);
        optimizer_->addVertex(vertex);
        RCLCPP_INFO(this->get_logger(), "[GRAPH] Vertex added. ID: %d, Fixed: %d", vertex->id(), fixed);

        saveGraph("graph_incremental.g2o");
    }

    void addEdge(int from, int to, const Eigen::Isometry3d &relative_pose) {
        if (!optimizer_ || !optimizer_->vertex(from) || !optimizer_->vertex(to)) {
            RCLCPP_ERROR(this->get_logger(), "[GRAPH] Invalid vertices: %d, %d. Cannot add edge.", from, to);
            return;
        }

        auto edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer_->vertex(from));
        edge->setVertex(1, optimizer_->vertex(to));
        edge->setMeasurement(relative_pose);
        edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        optimizer_->addEdge(edge);

        RCLCPP_INFO(this->get_logger(), "[GRAPH] Edge added between vertices %d and %d", from, to);

        saveGraph("graph_incremental.g2o");
    }

    void timerCallback() {
        std::lock_guard<std::mutex> lock(mtx_);
        auto current_time = this->now();
        auto duration = current_time - last_data_time_;
        double duration_sec = duration.seconds();

        if (duration_sec >= 60.0 && !loop_closure_triggered_ && !pose_history_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[TIMER] No data received for 60 seconds. Triggering loop closure detection.");
            detectLoopClosure();
            loop_closure_triggered_ = true;
        }
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GraphSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
