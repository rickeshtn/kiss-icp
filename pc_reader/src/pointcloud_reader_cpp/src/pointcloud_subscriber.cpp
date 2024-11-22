#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>
#include <filesystem>

class PointCloudSubscriber : public rclcpp::Node
{
public:
    PointCloudSubscriber()
    : Node("pointcloud_subscriber"), count_(0)
    {
        // Ensure that the directory exists for saving point clouds and the index file
        std::string save_dir = "/tmp/pointclouds";
        if (!std::filesystem::exists(save_dir)) {
            if (!std::filesystem::create_directory(save_dir)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create directory for saving point clouds.");
                return;
            }
        }

        // Initialize index file
        index_file_.open(save_dir + "/pointcloud_index.txt", std::ios::out | std::ios::app); // Open for append mode
        if (!index_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open index file for writing.");
            return;
        }

        // Subscriber to the PointCloud2 topic
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sensor/radar/front/points",
            10,
            std::bind(&PointCloudSubscriber::topic_callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "PointCloudSubscriber node has been started.");
    }

    ~PointCloudSubscriber() {
        if (index_file_.is_open()) {
            index_file_.close(); // Close the index file when the node is shut down
        }
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Ensure the timestamp is valid
        if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
            RCLCPP_WARN(this->get_logger(), "Received PointCloud with missing timestamp. Skipping.");
            return;  // Skip processing if timestamp is missing
        }

        // Convert ROS PointCloud2 message to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        RCLCPP_INFO(this->get_logger(), "Received PointCloud with %zu points.", pcl_cloud.points.size());

        // Skip processing if the point cloud is empty
        if (pcl_cloud.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty PointCloud. Skipping.");
            return;  // Skip processing if the point cloud is empty
        }

        // Generate the filename for the point cloud
        std::string filename = "/tmp/pointclouds/pointcloud_" + std::to_string(count_) + ".pcd";

        // Save the PointCloud to disk as PCD (ASCII format)
        if (pcl::io::savePCDFileASCII(filename, pcl_cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save PointCloud to %s", filename.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Saved PointCloud to %s", filename.c_str());

        // Record timestamp and the filename in the index file
        std::stringstream ss;
        ss << msg->header.stamp.sec << "." << msg->header.stamp.nanosec; // Full timestamp (sec.nanosec)
        std::string timestamp = ss.str();

        // Write the timestamp and the filename to the index file
        index_file_ << timestamp << ", " << filename.substr(filename.find_last_of("/\\") + 1) << "\n"; // Extract file name
        RCLCPP_INFO(this->get_logger(), "Recorded timestamp: %s with ID: %zu", timestamp.c_str(), count_);

        count_++;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    size_t count_; // Used as the point cloud ID
    std::ofstream index_file_; // File stream for index file
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudSubscriber>());
    rclcpp::shutdown();
    return 0;
}
