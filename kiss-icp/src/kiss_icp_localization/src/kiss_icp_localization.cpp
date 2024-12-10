#include <memory>
#include <vector>
#include <mutex>
#include <fstream>      // Added for file I/O
#include <iomanip>      // Added for setting precision in timestamp
#include <chrono>       // Added for getting current time
#include <ctime>        // Added for time formatting

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <kiss_icp/pipeline/KissICP.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>  // Added for Euler angle conversions

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

class KISSICPLocalization : public rclcpp::Node
{
public:
    KISSICPLocalization()
        : Node("kiss_icp_localization"), first_scan_(true), initial_pose_received_(false)
    {
        // Load the pre-built map from a PCD file
        map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        std::string map_file = "/HOST_HOME/Projects/berlin_company_data/ros1_accumulated_cloud/map_Manual.pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_cloud_) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read map file: %s", map_file.c_str());
            rclcpp::shutdown();
            return;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Loaded map with %zu points", map_cloud_->points.size());
        }

        // Convert the map cloud to a vector of Eigen::Vector3d
        map_points_.reserve(map_cloud_->size());
        for (const auto& point : map_cloud_->points)
        {
            map_points_.emplace_back(point.x, point.y, point.z);
        }

        // Initialize KISS-ICP with the desired configuration
        kiss_icp::pipeline::KISSConfig config;
        config.max_range = 100.0;
        config.min_range = 0.0;
        config.voxel_size = 1.0;
        config.max_points_per_voxel = 20;
        config.initial_threshold = 2.0;
        config.min_motion_th = 0.1;
        config.max_num_iterations = 500;
        config.convergence_criterion = 0.0001;
        config.max_num_threads = 0;

        kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

        // Add the map points to KISS-ICP
        kiss_icp_->AddPoints(map_points_);

        // Initialize the pose as identity
        current_pose_ = Sophus::SE3d();

        // Initialize TF broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Subscriber to the accumulated point clouds
        accumulated_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/kiss/local_map", 10,
            std::bind(&KISSICPLocalization::accumulated_callback, this, std::placeholders::_1));

        // Subscriber to the odometry (acts as initial pose)
        odometry_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/kiss/odometry", 10,
            std::bind(&KISSICPLocalization::odometry_callback, this, std::placeholders::_1));

        // Publisher for the estimated pose (Odometry)
        pose_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/localized_pose", 10);

        // Publisher for the map
        map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/map", 1);

        // Timer to publish the map at a fixed rate
        map_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&KISSICPLocalization::publish_map, this));

        // Open CSV file for logging
        open_log_file();
    }

    ~KISSICPLocalization()
    {
        // Close the CSV file
        if (csv_file_.is_open())
        {
            csv_file_.close();
        }
    }

private:
    void open_log_file()
    {
        // Get current time for filename
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        // Format time as YYYYMMDD_HHMMSS
        char time_str[20];
        std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", std::localtime(&now_c));

        // Create filename
        std::string filename = "localization_log_" + std::string(time_str) + ".csv";

        // Open file
        csv_file_.open(filename, std::ios::out);

        if (csv_file_.is_open())
        {
            // Write CSV header
            csv_file_ << "timestamp,x,y,z,roll,pitch,yaw\n";
            RCLCPP_INFO(this->get_logger(), "Opened log file: %s", filename.c_str());
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s", filename.c_str());
        }
    }

    void accumulated_callback(const sensor_msgs::msg::PointCloud2::SharedPtr accumulated_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        if (!initial_pose_received_)
        {
            RCLCPP_WARN(this->get_logger(), "Odometry pose not received yet. Skipping localization.");
            return;
        }

        // Convert incoming accumulated cloud from ROS PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*accumulated_msg, *accumulated_cloud);

        if (accumulated_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty accumulated cloud.");
            return;
        }

        // Convert accumulated_cloud to a vector of Eigen::Vector3d
        std::vector<Eigen::Vector3d> accumulated_points;
        accumulated_points.reserve(accumulated_cloud->size());
        for (const auto& point : accumulated_cloud->points)
        {
            accumulated_points.emplace_back(point.x, point.y, point.z);
        }

        // Perform KISS-ICP registration
        const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(accumulated_points);

        // Get the current estimated pose from KISS-ICP
        const Sophus::SE3d pose = kiss_icp_->pose();

        // Update the current pose by applying the initial_guess_
        // This assumes that the KISS-ICP pose is relative to the initial pose
        current_pose_ = initial_guess_ * pose;

        
        // Broadcast the transform from 'map' to 'base_link'
        broadcast_transform(current_pose_, accumulated_msg->header.stamp);

        // Convert the pose to an Odometry message
        nav_msgs::msg::Odometry localized_odometry;
        localized_odometry.header.stamp = accumulated_msg->header.stamp;
        localized_odometry.header.frame_id = "map";
        localized_odometry.child_frame_id = "base_link";  // Set to your robot's frame

        // Set the pose
        localized_odometry.pose.pose.position.x = current_pose_.translation().x();
        localized_odometry.pose.pose.position.y = current_pose_.translation().y();
        localized_odometry.pose.pose.position.z = current_pose_.translation().z();

        Eigen::Quaterniond q(current_pose_.so3().unit_quaternion());
        localized_odometry.pose.pose.orientation.x = q.x();
        localized_odometry.pose.pose.orientation.y = q.y();
        localized_odometry.pose.pose.orientation.z = q.z();
        localized_odometry.pose.pose.orientation.w = q.w();

        // Optionally, set the pose covariance (initialize with zeros or actual data)
        for (int i = 0; i < 36; ++i) {
            localized_odometry.pose.covariance[i] = 0.0;
        }

        // Set the twist (velocity) - if not available, set to zero
        localized_odometry.twist.twist.linear.x = 0.0;
        localized_odometry.twist.twist.linear.y = 0.0;
        localized_odometry.twist.twist.linear.z = 0.0;
        localized_odometry.twist.twist.angular.x = 0.0;
        localized_odometry.twist.twist.angular.y = 0.0;
        localized_odometry.twist.twist.angular.z = 0.0;

        // Optionally, set the twist covariance (initialize with zeros or actual data)
        for (int i = 0; i < 36; ++i) {
            localized_odometry.twist.covariance[i] = 0.0;
        }

        // Publish the localized odometry
        pose_publisher_->publish(localized_odometry);

        RCLCPP_INFO(this->get_logger(), "Published localized odometry: [x: %.2f, y: %.2f, z: %.2f]",
                    localized_odometry.pose.pose.position.x,
                    localized_odometry.pose.pose.position.y,
                    localized_odometry.pose.pose.position.z);

        // Log data to CSV file
        if (csv_file_.is_open())
        {
            // Extract timestamp
            auto timestamp = accumulated_msg->header.stamp.sec + accumulated_msg->header.stamp.nanosec * 1e-9;

            // Convert quaternion to roll, pitch, yaw
            Eigen::Quaterniond quat(q.w(), q.x(), q.y(), q.z());
            Eigen::Vector3d euler_angles = quat.toRotationMatrix().eulerAngles(0, 1, 2);  // Roll, Pitch, Yaw

            double roll = euler_angles[0];
            double pitch = euler_angles[1];
            double yaw = euler_angles[2];

            // Write to CSV
            csv_file_ << std::fixed << std::setprecision(9)
                      << timestamp << ","
                      << localized_odometry.pose.pose.position.x << ","
                      << localized_odometry.pose.pose.position.y << ","
                      << localized_odometry.pose.pose.position.z << ","
                      << roll << ","
                      << pitch << ","
                      << yaw << "\n";
        }
    }

    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr odometry_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // Extract position
        double x = odometry_msg->pose.pose.position.x;
        double y = odometry_msg->pose.pose.position.y;
        double z = odometry_msg->pose.pose.position.z;

        // Extract orientation
        double qx = odometry_msg->pose.pose.orientation.x;
        double qy = odometry_msg->pose.pose.orientation.y;
        double qz = odometry_msg->pose.pose.orientation.z;
        double qw = odometry_msg->pose.pose.orientation.w;

        // Convert to Sophus::SE3d
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Sophus::SE3d initial_pose(q, Eigen::Vector3d(x, y, z));

        // Set as initial guess for ICP (only once)
        if (!initial_pose_received_)
        {
            initial_guess_ = initial_pose;
            initial_pose_received_ = true;

            RCLCPP_INFO(this->get_logger(), "Received initial pose from /kiss/odometry: [x: %.2f, y: %.2f, z: %.2f]",
                        x, y, z);
        }
    }

    void publish_map()
    {
        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(*map_cloud_, map_msg);
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = "map";
        map_publisher_->publish(map_msg);
        RCLCPP_INFO(this->get_logger(), "Published map to /map");
    }

    void broadcast_transform(const Sophus::SE3d& pose, const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped transform_stamped;

        transform_stamped.header.stamp = stamp;
        transform_stamped.header.frame_id = "map";
        transform_stamped.child_frame_id = "base_link";

        transform_stamped.transform.translation.x = pose.translation().x();
        transform_stamped.transform.translation.y = pose.translation().y();
        transform_stamped.transform.translation.z = pose.translation().z();

        Eigen::Quaterniond q(pose.unit_quaternion());

        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(transform_stamped);
    }

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr accumulated_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
    std::vector<Eigen::Vector3d> map_points_;
    std::unique_ptr<kiss_icp::pipeline::KissICP> kiss_icp_;
    Sophus::SE3d current_pose_;
    Sophus::SE3d initial_guess_;
    bool first_scan_;
    bool initial_pose_received_;
    std::mutex data_mutex_;

    // Timer for publishing the map
    rclcpp::TimerBase::SharedPtr map_timer_;

    // CSV file for logging
    std::ofstream csv_file_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KISSICPLocalization>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
