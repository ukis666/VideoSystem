#include "video_system_perception/detector_node.hpp"

using video_system_interfaces::msg::Event;

namespace vsp = video_system_perception;

namespace {

// (left, top, width, height) değerlerini görüntü sınırları içine sabitle
    static inline void clampRect(cv::Rect &r, int W, int H) {
        r.x = std::max(0, std::min(r.x, W - 1));
        r.y = std::max(0, std::min(r.y, H - 1));
        r.width  = std::max(0, std::min(r.width,  W - r.x));
        r.height = std::max(0, std::min(r.height, H - r.y));
    }

} // namespace

vsp::DetectorNode::DetectorNode() : rclcpp::Node("detector_node") {
    // Parametreleri ilan et ve oku
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<bool>("visualize", true);
    this->declare_parameter<int>("fake_stride", 12); // uyumluluk

    this->get_parameter("model_path", model_path_);
    this->get_parameter("visualize", visualize_);
    this->get_parameter("fake_stride", fake_stride_);

    // Modeli yükle
    if (!model_path_.empty()) {
        try {
            net_ = cv::dnn::readNetFromONNX(model_path_);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(this->get_logger(), "Loaded ONNX model: %s", model_path_.c_str());
        } catch (const std::exception &e) {
            RCLCPP_WARN(this->get_logger(), "Failed to load ONNX model: %s", e.what());
        }
    } else {
        RCLCPP_WARN(this->get_logger(),
                    "Parameter 'model_path' is empty. Inference will be skipped.");
    }

    // Subscriber ve Publisher
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&vsp::DetectorNode::onImage, this, std::placeholders::_1));

    pub_ = this->create_publisher<Event>("/events", 10);

    if (visualize_) {
        try { cv::namedWindow("YOLOv8 Detection (preview)", cv::WINDOW_NORMAL); }
        catch (...) {}
    }

    RCLCPP_INFO(this->get_logger(), "detector_node started (fake_stride=%d)", fake_stride_);
}

vsp::DetectorNode::~DetectorNode() {
    if (visualize_) {
        try { cv::destroyWindow("YOLOv8 Detection (preview)"); }
        catch (...) {}
    }
}

// Ana görüntü geri çağrısı
void vsp::DetectorNode::onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    frame_count_++;

    // ROS -> OpenCV
    cv::Mat frame;
    try {
        // Paylaşımlı dönüşüm kopyadan kaçınır; renk BGR8 bekliyoruz
        frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    } catch (const std::exception &e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "cv_bridge error: %s", e.what());
        return;
    }

    // ----------- DEDESYON -----------
    if (!model_path_.empty() && !net_.empty()) {
        try {
            // 1) Giriş blob’u
            constexpr int IN_W = 640;
            constexpr int IN_H = 640;
            cv::Mat blob = cv::dnn::blobFromImage(
                    frame, 1.0 / 255.0, cv::Size(IN_W, IN_H),
                    cv::Scalar(), /*swapRB=*/true, /*crop=*/false);

            net_.setInput(blob);

            // 2) İleri geçiş
            cv::Mat out = net_.forward();

            // 3) Çıktıyı N x 84 2B matrise indirgeme
            cv::Mat det; // rows=N(=8400), cols=84
            if (out.dims == 3) {
                // Tipik YOLOv8: [1, 84, 8400]
                if (out.size[0] == 1 && out.size[1] > 4) {
                    const int C = out.size[1]; // 84
                    const int N = out.size[2]; // 8400
                    cv::Mat c_by_n(C, N, CV_32F, out.ptr<float>()); // 84 x 8400 görünüm
                    det = c_by_n.t(); // 8400 x 84
                } else if (out.size[0] == 1 && out.size[2] > 4) {
                    // Güvenli tarafta kalmak adına genel durum
                    const int A = out.size[1];
                    const int B = out.size[2];
                    cv::Mat a_by_b(A, B, CV_32F, out.ptr<float>());
                    det = a_by_b.t(); // B x A
                } else {
                    RCLCPP_WARN(this->get_logger(),
                                "Unexpected 3D output shape: [%d, %d, %d]",
                                out.size[0], out.size[1], out.size[2]);
                    return;
                }
            } else if (out.dims == 2) {
                // Bazı kurulumlarda zaten 8400 x 84 veya 84 x 8400 gelir
                if (out.cols == 84) {
                    det = out; // 8400 x 84
                } else if (out.rows == 84) {
                    det = out.t(); // 8400 x 84
                } else {
                    // Son çare: en uzun kenarı satır sayısı varsay
                    det = (out.cols > out.rows) ? out.t() : out;
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "Unexpected DNN output dims=%d", out.dims);
                return;
            }

            // 4) Ölçekleme katsayıları
            const float x_factor = static_cast<float>(frame.cols) / static_cast<float>(IN_W);
            const float y_factor = static_cast<float>(frame.rows) / static_cast<float>(IN_H);

            // 5) Taramalar
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;

            const float CONF_THR = 0.45f;
            const float NMS_THR  = 0.50f;

            for (int i = 0; i < det.rows; ++i) {
                const float* p = det.ptr<float>(i);
                float cx = p[0], cy = p[1], w = p[2], h = p[3];

                // sınıf skorları [4..cols-1]
                cv::Mat scores(1, det.cols - 4, CV_32F, const_cast<float*>(p + 4));
                cv::Point cid; double max_score;
                cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &cid);

                if (max_score < CONF_THR) continue;

                int left   = static_cast<int>((cx - 0.5f * w) * x_factor);
                int top    = static_cast<int>((cy - 0.5f * h) * y_factor);
                int width  = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                cv::Rect r(left, top, width, height);
                clampRect(r, frame.cols, frame.rows);

                boxes.emplace_back(r);
                confidences.emplace_back(static_cast<float>(max_score));
                class_ids.emplace_back(cid.x);
            }

            // 6) NMS
            std::vector<int> keep;
            cv::dnn::NMSBoxes(boxes, confidences, CONF_THR, NMS_THR, keep);

            // 7) Yayınla + çiz
            for (int k : keep) {
                const cv::Rect &b = boxes[k];
                const int cls     = class_ids[k];
                const float sc    = confidences[k];

                Event ev;
                ev.stamp = this->now();
                ev.label = "ID:" + std::to_string(cls);
                ev.score = sc;
                ev.x = b.x; ev.y = b.y; ev.w = b.width; ev.h = b.height;
                pub_->publish(ev);

                if (visualize_) {
                    cv::rectangle(frame, b, cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame,
                                ev.label + ":" + cv::format("%.2f", sc),
                                {b.x, std::max(0, b.y - 6)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
                }
            }

        } catch (const std::exception &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                 "DNN forward error: %s", e.what());
        }
    }

    // ----------- GÖRSELLEŞTİRME -----------
    if (visualize_) {
        try {
            cv::imshow("YOLOv8 Detection (preview)", frame);
            cv::waitKey(1);
        } catch (...) {
            // Headless ortamda hata fırlarsa sessiz geç
        }
    }
}
