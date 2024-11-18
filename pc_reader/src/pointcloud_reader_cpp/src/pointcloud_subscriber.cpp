// pointcloud_subscriber.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/PointCloud2.h>  // ROS1 header
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>  // ROS1 core
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

class PointCloudSubscriber : public rclcpp::Node
{
public:
    PointCloudSubscriber()
    : Node("pointcloud_subscriber"), count_(0)
    {
        // ROS 1 initialization
        ros::init(argc_, argv_, "ros1_pointcloud_publisher");
        ros::NodeHandle nh;

        // ROS 1 Publisher (to publish PointCloud2)
        ros1_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/ros1/pointcloud", 10);

        // Directory to save PCD files
        save_dir_ = "/HOST_HOME/Projects/berlin_company_data/pointclouds/"; // Change this to match your mounted directory
        if (!fs::exists(save_dir_))
        {
            fs::create_directories(save_dir_);
            RCLCPP_INFO(this->get_logger(), "Created directory: %s", save_dir_.c_str());
        }

        // Subscriber to the PointCloud2 topic
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sensor/radar/front/points",
            10,
            std::bind(&PointCloudSubscriber::topic_callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "PointCloudSubscriber node has been started.");
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS 2 PointCloud2 message to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        RCLCPP_INFO(this->get_logger(), "Received PointCloud with %zu points.", pcl_cloud.points.size());

        // Save PointCloud to disk as PCD file
        if (!pcl_cloud.points.empty())
        {
            // Create a filename using the count variable
            std::string filename = save_dir_ + "pointcloud_" + std::to_string(count_) + ".pcd";
            pcl::io::savePCDFileASCII(filename, pcl_cloud);  // Save as PCD (ASCII format)

            RCLCPP_INFO(this->get_logger(), "Saved PointCloud to: %s", filename.c_str());
            count_++;
        }

        // Convert the PointCloud2 from ROS 2 to ROS 1 message
        sensor_msgs::PointCloud2 ros1_msg;
        pcl_conversions::moveFromPCL(pcl_cloud, ros1_msg);  // Conversion from PCL to ROS1 PointCloud2

        // Publish to ROS 1
        ros1_pub_.publish(ros1_msg);

        // Spin ROS 1 callbacks to ensure message gets published
        ros::spinOnce();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    ros::Publisher ros1_pub_;  // ROS 1 publisher
    size_t count_;
    std::string save_dir_; // Directory to save PCD files
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    // Initialize ROS 1
    ros::init(argc, argv, "pointcloud_bridge_node");

    // Create and run the ROS 2 node
    rclcpp::spin(std::make_shared<PointCloudSubscriber>());
    rclcpp::shutdown();

    return 0;
}