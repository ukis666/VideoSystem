#include "video_system_core/image_event_node.hpp"
#include <functional>

ImageEventNode::ImageEventNode()
: Node("image_event_node"), counter_(0)
{
  heartbeat_stride_ = this->declare_parameter<int>("heartbeat_stride", 15);

  rclcpp::QoS img_qos{rclcpp::KeepLast(5)};
  img_qos.best_effort(); // webcam için genelde uygun

  // Kameradan ham görüntü bekliyoruz: /camera/image_raw
  sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", img_qos,
      std::bind(&ImageEventNode::onImage, this, std::placeholders::_1));

  pub_ = this->create_publisher<video_system_interfaces::msg::Event>(
      "/events", rclcpp::QoS(10).reliable());

  RCLCPP_INFO(this->get_logger(), "image_event_node started. heartbeat_stride=%d", heartbeat_stride_);
}

void ImageEventNode::onImage(const sensor_msgs::msg::Image::SharedPtr /*msg*/) {
  if (++counter_ % static_cast<uint64_t>(heartbeat_stride_) == 0) {
    video_system_interfaces::msg::Event ev;
    ev.stamp = this->now();
    ev.label = "heartbeat";
    ev.score = 1.0f;
    ev.x = 0; ev.y = 0; ev.w = 64; ev.h = 64;
    pub_->publish(ev);
  }
}
