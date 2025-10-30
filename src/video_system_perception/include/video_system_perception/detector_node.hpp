#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "video_system_interfaces/msg/event.hpp"

namespace video_system_perception {

    class DetectorNode : public rclcpp::Node {
    public:
        DetectorNode();
        ~DetectorNode() override;

    private:
        void onImage(const sensor_msgs::msg::Image::SharedPtr msg);

        // Parametreler
        std::string model_path_;
        bool visualize_{true};
        int fake_stride_{12}; // geriye dönük uyumluluk için tutuluyor (kullanılmıyor)

        // DNN ağı
        cv::dnn::Net net_;

        // ROS 2 I/O
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
        rclcpp::Publisher<video_system_interfaces::msg::Event>::SharedPtr pub_;

        // Durum
        uint64_t frame_count_{0};
    };

} // namespace video_system_perception
