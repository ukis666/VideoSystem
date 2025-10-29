#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <video_system_interfaces/msg/event.hpp>

class ImageEventNode : public rclcpp::Node {
public:
  ImageEventNode();

private:
  void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<video_system_interfaces::msg::Event>::SharedPtr pub_;
  int32_t heartbeat_stride_;
  uint64_t counter_;
};
