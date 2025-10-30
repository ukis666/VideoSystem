#include "rclcpp/rclcpp.hpp"
#include "video_system_perception/detector_node.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // Sınıf isim alanında: video_system_perception::DetectorNode
  auto node = std::make_shared<video_system_perception::DetectorNode>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
