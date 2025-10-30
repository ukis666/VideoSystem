#include <rclcpp/rclcpp.hpp>
#include "video_system_core/image_event_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageEventNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
