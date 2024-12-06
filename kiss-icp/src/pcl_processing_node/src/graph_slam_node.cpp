#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <queue>
#include <mutex>
#include <memory> // Include this header for std::make_unique

// Include G2O headers for 2D SLAM
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/types_slam2d.h>

// Include OpenCV for image processing and SIFT features
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class GraphSlam2DNode : public rclcpp::Node {
public:
    GraphSlam2DNode() : Node("graph_slam_2d_node"), vertex_count_(0), loop_closure_triggered_(false) {
        // Subscription to /kiss/local_map
        local_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/kiss/local_map", 10,
            std::bind(&GraphSlam2DNode::localMapCallback, this, std::placeholders::_1));

        // Initialize the optimizer
        initializeGraphOptimizer();

        // Initialize timer to check for inactivity
        timer_ = this->create_wall_timer(
            std::chrono::seconds(10),
            std::bind(&GraphSlam2DNode::timerCallback, this));

        // Initialize last_data_time_ and loop_closure_triggered_
        last_data_time_ = this->now();
        loop_closure_triggered_ = false;

        RCLCPP_INFO(this->get_logger(), "[INIT] Graph SLAM 2D Node Initialized");
    }

private:
    // Subscription
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_sub_;

    // Graph Optimizer
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
    int vertex_count_;

    // Data storage
    std::vector<Eigen::Isometry2d> pose_history_; // Stores estimated poses
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scan_history_; // Stores point clouds

    // Thresholds and parameters
    const size_t MAX_QUEUE_SIZE = 2700; // Adjust as needed

    // Timer for inactivity
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_data_time_;
    bool loop_closure_triggered_;

    // Mutex for thread safety
    std::mutex mtx_;

    // Parameters for bird's eye view image
    double resolution_ = 0.05; // 5 cm per pixel
    int grid_width_ = 2000;    // Adjust based on expected area (100m x 100m area)
    int grid_height_ = 2000;

    // Storage for images and features
    std::vector<cv::Mat> birdseye_images_;
    std::vector<std::vector<cv::KeyPoint>> keypoints_history_;
    std::vector<cv::Mat> descriptors_history_;

    void initializeGraphOptimizer() {
        // Initialize optimizer for 2D SLAM

        // Define the block solver traits (pose dimension 3, landmark dimension 1)
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;

        // Define the linear solver using Eigen
        typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

        // Create the linear solver
        auto linearSolver = std::make_unique<LinearSolverType>();

        // Create the block solver
        auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));

        // Set up the optimization algorithm (Levenberg-Marquardt)
        // Use a raw pointer as setAlgorithm expects a raw pointer
        auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

        // Create the optimizer and set the algorithm
        optimizer_ = std::make_unique<g2o::SparseOptimizer>();
        optimizer_->setAlgorithm(solver); // Pass raw pointer

        if (!optimizer_->solver()) {
            RCLCPP_ERROR(this->get_logger(), "[ERROR] Graph optimizer initialization failed!");
        } else {
            RCLCPP_INFO(this->get_logger(), "[INIT] Graph optimizer successfully initialized.");
        }

        // Add a dummy landmark with a unique ID
        int landmark_id = 1000; // Starting ID for landmarks to avoid collision with pose IDs
        auto landmark = new g2o::VertexPointXY();
        landmark->setId(landmark_id);
        landmark->setEstimate(Eigen::Vector2d(0, 0));
        landmark->setFixed(true); // Landmark is fixed and won't be optimized
        optimizer_->addVertex(landmark);

        RCLCPP_INFO(this->get_logger(), "[INIT] Dummy landmark added. ID: %d", landmark->id());
    }



    void saveGraph(const std::string &filename) {
        if (optimizer_) {
            RCLCPP_INFO(this->get_logger(), "[SAVE] Saving graph to file: %s", filename.c_str());
            optimizer_->save(filename.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "[ERROR] Optimizer is not initialized. Cannot save graph.");
        }
    }

    void localMapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        RCLCPP_INFO(this->get_logger(), "[LOCAL_MAP] Received local map point cloud.");
        RCLCPP_INFO(this->get_logger(), "[LOCAL_MAP] Header: frame_id = %s, stamp = %u.%u",
                    msg->header.frame_id.c_str(),
                    msg->header.stamp.sec,
                    msg->header.stamp.nanosec);

        // Convert ROS message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *current_scan);

        if (current_scan->empty()) {
            RCLCPP_WARN(this->get_logger(), "[LOCAL_MAP] Received an empty point cloud. Skipping.");
            return;
        }

        // Update last_data_time_
        last_data_time_ = this->now();
        loop_closure_triggered_ = false;

        // Process the local map
        processLocalMap(current_scan);
    }

    void processLocalMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan) {
        // Convert point cloud to bird's eye view image
        cv::Mat current_image;
        pointCloudToBirdseyeImage(current_scan, current_image);

        // Extract SIFT features from current image
        std::vector<cv::KeyPoint> current_keypoints;
        cv::Mat current_descriptors;
        extractSIFTFeatures(current_image, current_keypoints, current_descriptors);

        // Add the current scan and features to history
        scan_history_.push_back(current_scan);
        birdseye_images_.push_back(current_image);
        keypoints_history_.push_back(current_keypoints);
        descriptors_history_.push_back(current_descriptors);

        // Estimate relative pose to the previous local map
        Eigen::Isometry2d current_pose = Eigen::Isometry2d::Identity();

        if (!pose_history_.empty()) {
            // Get previous data
            size_t prev_idx = pose_history_.size() - 1;
            cv::Mat prev_descriptors = descriptors_history_[prev_idx];
            std::vector<cv::KeyPoint> prev_keypoints = keypoints_history_[prev_idx];

            // Match features
            std::vector<cv::DMatch> matches;
            matchFeatures(prev_descriptors, current_descriptors, matches);

            // Estimate relative pose
            Eigen::Isometry2d relative_pose;
            if (estimateRelativePose(prev_keypoints, current_keypoints, matches, relative_pose)) {
                // Update current pose based on the previous pose and the relative transformation
                current_pose = pose_history_.back() * relative_pose;

                // Add edge to the graph
                addEdge(vertex_count_ - 1, vertex_count_, relative_pose);
            } else {
                RCLCPP_WARN(this->get_logger(), "[PROCESS] Pose estimation between local maps failed. Skipping edge addition.");
                // Use the previous pose as the current pose (or handle differently)
                current_pose = pose_history_.back();
            }
        }

        // Add current pose to history
        pose_history_.push_back(current_pose);

        // Add vertex to the graph
        addVertex(current_pose, pose_history_.size() == 1); // First pose is fixed

        RCLCPP_INFO(this->get_logger(), "[PROCESS] Local map and pose added to graph. Total poses: %lu", pose_history_.size());

        // Loop closure detection if necessary
        if (pose_history_.size() >= MAX_QUEUE_SIZE) {
            detectLoopClosureWithSIFT();
        }
    }

    void pointCloudToBirdseyeImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan, cv::Mat& birdseye_image) {
        // Initialize the color image (3 channels)
        birdseye_image = cv::Mat::zeros(grid_height_, grid_width_, CV_8UC3);

        // Define vertical boundaries for splitting into zones
        double z_min = std::numeric_limits<double>::max();
        double z_max = std::numeric_limits<double>::lowest();

        // First, find the minimum and maximum z-values
        for (const auto& point : scan->points) {
            if (point.z < z_min) z_min = point.z;
            if (point.z > z_max) z_max = point.z;
        }

        // Handle the case where z_min == z_max (flat terrain)
        if (z_max - z_min < 0.01) {
            z_max = z_min + 0.01; // Add a small value to avoid division by zero
        }

        // Split the range into three equal zones
        double z_range = z_max - z_min;
        double zone_height = z_range / 3.0;

        double z_boundary1 = z_min + zone_height;
        double z_boundary2 = z_min + 2 * zone_height;

        for (const auto& point : scan->points) {
            // Project point onto the XY-plane
            double x = point.x;
            double y = point.y;

            // Convert to pixel coordinates
            int pixel_x = static_cast<int>((x / resolution_) + grid_width_ / 2);
            int pixel_y = static_cast<int>((y / resolution_) + grid_height_ / 2);

            // Check bounds
            if (pixel_x >= 0 && pixel_x < grid_width_ && pixel_y >= 0 && pixel_y < grid_height_) {
                // Determine which zone the point belongs to
                cv::Vec3b& pixel = birdseye_image.at<cv::Vec3b>(pixel_y, pixel_x);
                if (point.z <= z_boundary1) {
                    // Lower zone - set Red channel
                    pixel[2] = 255;
                } else if (point.z <= z_boundary2) {
                    // Middle zone - set Green channel
                    pixel[1] = 255;
                } else {
                    // Upper zone - set Blue channel
                    pixel[0] = 255;
                }
            }
        }

        // Optionally, apply Gaussian blur to each channel separately
        std::vector<cv::Mat> channels(3);
        cv::split(birdseye_image, channels);
        for (auto& channel : channels) {
            cv::GaussianBlur(channel, channel, cv::Size(3, 3), 0);
        }
        cv::merge(channels, birdseye_image);
    }

    // Function to extract SIFT features
    void extractSIFTFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
        // Convert the color image to grayscale
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // Use SIFT to detect and compute features
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(gray_image, cv::Mat(), keypoints, descriptors);
    }

    // Function to match features between two images using SIFT descriptors
    void matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& good_matches) {
        // Convert descriptors to CV_32F if necessary
        cv::Mat desc1 = descriptors1;
        cv::Mat desc2 = descriptors2;
        if (descriptors1.type() != CV_32F) {
            descriptors1.convertTo(desc1, CV_32F);
        }
        if (descriptors2.type() != CV_32F) {
            descriptors2.convertTo(desc2, CV_32F);
        }

        // Use FLANN-based matcher for floating-point descriptors
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);

        // Filter matches based on distance
        double max_dist = 0;
        double min_dist = 100;
        for (const auto& match : matches) {
            double dist = match.distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (const auto& match : matches) {
            if (match.distance <= std::max(2 * min_dist, 0.02)) {
                good_matches.push_back(match);
            }
        }
    }

    // Function to estimate pose between two scans
    bool estimateRelativePose(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                              const std::vector<cv::DMatch>& matches, Eigen::Isometry2d& relative_pose) {
        if (matches.size() < 10) {
            // Not enough matches to estimate pose
            return false;
        }

        // Extract matched points
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // Estimate affine transformation
        cv::Mat inliers;
        cv::Mat affine = cv::estimateAffine2D(points1, points2, inliers);

        if (affine.empty()) {
            // Estimation failed
            return false;
        }

        // Convert affine transformation to Eigen::Isometry2d
        Eigen::Matrix3d transformation = Eigen::Matrix3d::Identity();
        transformation(0, 0) = affine.at<double>(0, 0);
        transformation(0, 1) = affine.at<double>(0, 1);
        transformation(0, 2) = affine.at<double>(0, 2);
        transformation(1, 0) = affine.at<double>(1, 0);
        transformation(1, 1) = affine.at<double>(1, 1);
        transformation(1, 2) = affine.at<double>(1, 2);

        relative_pose = Eigen::Isometry2d(transformation);

        return true;
    }

    void detectLoopClosureWithSIFT() {
        RCLCPP_INFO(this->get_logger(), "[LOOP] Starting loop closure detection using SIFT features...");

        size_t current_idx = pose_history_.size() - 1;
        cv::Mat current_descriptors = descriptors_history_[current_idx];
        std::vector<cv::KeyPoint> current_keypoints = keypoints_history_[current_idx];

        // Parameters for loop closure detection
        const double LOOP_CLOSURE_DISTANCE_THRESHOLD = 50.0; // Meters
        const size_t MIN_MATCH_COUNT = 30;

        for (size_t i = 0; i < current_idx - 10; ++i) { // Skip recent poses to avoid false positives
            // Check if the pose is within a certain distance
            double distance = (pose_history_[current_idx].translation() - pose_history_[i].translation()).norm();
            if (distance > LOOP_CLOSURE_DISTANCE_THRESHOLD) {
                continue;
            }

            // Match features
            std::vector<cv::DMatch> matches;
            matchFeatures(descriptors_history_[i], current_descriptors, matches);

            // Check if enough matches are found
            if (matches.size() >= MIN_MATCH_COUNT) {
                // Estimate relative pose
                Eigen::Isometry2d relative_pose;
                if (estimateRelativePose(keypoints_history_[i], current_keypoints, matches, relative_pose)) {
                    // Add loop closure edge
                    addEdge(i, current_idx, relative_pose);

                    // Optimize the graph
                    optimizer_->initializeOptimization();
                    optimizer_->optimize(10);

                    RCLCPP_INFO(this->get_logger(), "[LOOP] Loop closure detected and applied between poses %lu and %lu", i, current_idx);

                    // Generate and save the corrected bird's eye view map
                    generateAndSaveCorrectedBirdseyeImage();

                    // Optionally, break after finding one loop closure
                    break;
                } else {
                    RCLCPP_WARN(this->get_logger(), "[LOOP] Pose estimation for loop closure failed between poses %lu and %lu", i, current_idx);
                }
            }
        }
    }

    void generateAndSaveCorrectedBirdseyeImage() {
        // Initialize the corrected bird's eye view image as a color image
        cv::Mat corrected_birdseye_image = cv::Mat::zeros(grid_height_, grid_width_, CV_8UC3);

        for (int i = 0; i < vertex_count_; ++i) {
            auto vertex_ptr = optimizer_->vertex(i);
            if (!vertex_ptr) {
                RCLCPP_WARN(this->get_logger(), "[GRAPH] Vertex %d not found in optimizer. Skipping.", i);
                continue;
            }

            auto vertex = dynamic_cast<g2o::VertexSE2*>(vertex_ptr);
            if (!vertex) {
                RCLCPP_WARN(this->get_logger(), "[GRAPH] Vertex %d is not of type VertexSE2. Skipping.", i);
                continue;
            }

            double x = vertex->estimate().translation()[0];
            double y = vertex->estimate().translation()[1];
            double theta = vertex->estimate().rotation().angle();
            Eigen::Isometry2d optimized_pose = Eigen::Isometry2d::Identity();
            optimized_pose.translate(Eigen::Vector2d(x, y));
            optimized_pose.rotate(theta);

            // Retrieve the color image
            if (i >= birdseye_images_.size()) {
                RCLCPP_WARN(this->get_logger(), "[GRAPH] No birdseye image for vertex %d. Skipping.", i);
                continue;
            }
            cv::Mat image = birdseye_images_[i];

            // Transform image according to optimized pose
            cv::Mat transformed_image;
            cv::warpAffine(image, transformed_image, getAffineTransform(optimized_pose), cv::Size(grid_width_, grid_height_), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);

            // Combine with the corrected bird's eye view image
            cv::bitwise_or(corrected_birdseye_image, transformed_image, corrected_birdseye_image);
        }

        cv::imwrite("corrected_birdseye_map.png", corrected_birdseye_image);
        RCLCPP_INFO(this->get_logger(), "[OUTPUT] Corrected bird's eye view map saved to corrected_birdseye_map.png");
    }

    // Helper function to get affine transform matrix from Eigen::Isometry2d
    cv::Mat getAffineTransform(const Eigen::Isometry2d& pose) {
        cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);

        // Rotation matrix components
        double cos_theta = pose.rotation()(0, 0);
        double sin_theta = pose.rotation()(1, 0);

        // Set rotation components
        affine.at<double>(0, 0) = cos_theta;
        affine.at<double>(0, 1) = -sin_theta;
        affine.at<double>(1, 0) = sin_theta;
        affine.at<double>(1, 1) = cos_theta;

        // Set translation components (convert from meters to pixels)
        affine.at<double>(0, 2) = (pose.translation().x() / resolution_) + grid_width_ / 2;
        affine.at<double>(1, 2) = (pose.translation().y() / resolution_) + grid_height_ / 2;

        return affine;
    }

    void addVertex(const Eigen::Isometry2d& pose, bool fixed = false) {
        auto vertex = new g2o::VertexSE2();
        vertex->setId(vertex_count_++);
        vertex->setEstimate(g2o::SE2(pose.translation().x(), pose.translation().y(), std::atan2(pose.rotation()(1,0), pose.rotation()(0,0))));
        vertex->setFixed(fixed);
        optimizer_->addVertex(vertex);
        RCLCPP_INFO(this->get_logger(), "[GRAPH] Vertex added. ID: %d, Fixed: %d", vertex->id(), fixed);

        saveGraph("graph_incremental.g2o");
    }

    void addEdge(int from, int to, const Eigen::Isometry2d& relative_pose) {
        if (!optimizer_ || !optimizer_->vertex(from) || !optimizer_->vertex(to)) {
            RCLCPP_ERROR(this->get_logger(), "[GRAPH] Invalid vertices: %d, %d. Cannot add edge.", from, to);
            return;
        }

        auto edge = new g2o::EdgeSE2();
        edge->setVertex(0, optimizer_->vertex(from));
        edge->setVertex(1, optimizer_->vertex(to));
        edge->setMeasurement(g2o::SE2(relative_pose.translation().x(), relative_pose.translation().y(), std::atan2(relative_pose.rotation()(1,0), relative_pose.rotation()(0,0))));
        edge->setInformation(Eigen::Matrix3d::Identity()); // Adjust the information matrix as needed
        optimizer_->addEdge(edge);

        RCLCPP_INFO(this->get_logger(), "[GRAPH] Edge added between vertices %d and %d", from, to);

        saveGraph("graph_incremental.g2o");
    }

    void timerCallback() {
        std::lock_guard<std::mutex> lock(mtx_);
        auto current_time = this->now();
        auto duration = current_time - last_data_time_;
        double duration_sec = duration.seconds();

        if (duration_sec >= 30.0 && !loop_closure_triggered_ && !pose_history_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[TIMER] No data received for 30 seconds. Triggering loop closure detection.");
            detectLoopClosureWithSIFT();
            loop_closure_triggered_ = true;
        }
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GraphSlam2DNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
