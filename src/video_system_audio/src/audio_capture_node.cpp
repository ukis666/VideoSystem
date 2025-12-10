#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "video_system_interfaces/msg/audio_pcm16.hpp"

#include <alsa/asoundlib.h>

using namespace std::chrono_literals;
using video_system_interfaces::msg::AudioPcm16;

class AudioCaptureNode : public rclcpp::Node
{
public:
  AudioCaptureNode()
  : Node("audio_capture_node")
  {
    declare_parameter<std::string>("device", "hw:1,0");
    declare_parameter<int>("sample_rate", 16000);
    declare_parameter<int>("channels", 1);
    declare_parameter<int>("chunk_ms", 500);

    device_ = get_parameter("device").as_string();
    sample_rate_ = get_parameter("sample_rate").as_int();
    channels_ = get_parameter("channels").as_int();
    chunk_ms_ = get_parameter("chunk_ms").as_int();

    frame_size_ = static_cast<size_t>(sample_rate_ * chunk_ms_ / 1000);

    pub_ = create_publisher<AudioPcm16>("audio_pcm", 10);

    open_alsa();
    timer_ = create_wall_timer(
      std::chrono::milliseconds(chunk_ms_),
      std::bind(&AudioCaptureNode::capture_callback, this));
  }

  ~AudioCaptureNode() override
  {
    if (pcm_handle_ != nullptr) {
      snd_pcm_close(pcm_handle_);
    }
  }

private:
  void open_alsa()
  {
    int err;

    if ((err = snd_pcm_open(&pcm_handle_, device_.c_str(), SND_PCM_STREAM_CAPTURE, 0)) < 0) {
      RCLCPP_FATAL(get_logger(), "Cannot open audio device %s: %s", device_.c_str(), snd_strerror(err));
      throw std::runtime_error("ALSA open failed");
    }

    snd_pcm_hw_params_t * params;
    snd_pcm_hw_params_malloc(&params);
    snd_pcm_hw_params_any(pcm_handle_, params);
    snd_pcm_hw_params_set_access(pcm_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle_, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm_handle_, params, channels_);
    unsigned int rate = static_cast<unsigned int>(sample_rate_);
    snd_pcm_hw_params_set_rate_near(pcm_handle_, params, &rate, nullptr);

    if ((err = snd_pcm_hw_params(pcm_handle_, params)) < 0) {
      RCLCPP_FATAL(get_logger(), "Cannot set hw params: %s", snd_strerror(err));
      snd_pcm_hw_params_free(params);
      throw std::runtime_error("ALSA hw_params failed");
    }
    snd_pcm_hw_params_free(params);

    RCLCPP_INFO(get_logger(), "ALSA capture opened: device=%s, rate=%u, channels=%d", device_.c_str(), rate, channels_);

    snd_pcm_prepare(pcm_handle_);
  }

  void capture_callback()
  {
    std::vector<int16_t> buffer(frame_size_ * static_cast<size_t>(channels_));
    const size_t frames_to_read = frame_size_;

    int err = snd_pcm_readi(pcm_handle_, buffer.data(), frames_to_read);
    if (err == -EPIPE) {
      snd_pcm_prepare(pcm_handle_);
      RCLCPP_WARN(get_logger(), "ALSA overrun, stream prepared again.");
      return;
    }
    if (err < 0) {
      RCLCPP_ERROR(get_logger(), "ALSA read error: %s", snd_strerror(err));
      return;
    }

    auto msg = AudioPcm16();
    msg.stamp = now();
    msg.sample_rate = static_cast<uint32_t>(sample_rate_);
    msg.num_channels = static_cast<uint16_t>(channels_);
    msg.data = buffer;

    pub_->publish(msg);
  }

  std::string device_;
  int sample_rate_ {};
  int channels_ {};
  int chunk_ms_ {};
  size_t frame_size_ {};

  snd_pcm_t * pcm_handle_ = nullptr;

  rclcpp::Publisher<AudioPcm16>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<AudioCaptureNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::cerr << "AudioCaptureNode fatal error: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
