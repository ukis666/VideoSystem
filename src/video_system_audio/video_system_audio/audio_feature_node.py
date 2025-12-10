#!/usr/bin/env python3
from collections import deque

import librosa
import numpy as np
import rclpy
from rclpy.node import Node

from video_system_interfaces.msg import AudioMel, AudioPcm16


class AudioFeatureNode(Node):
    def __init__(self):
        super().__init__('audio_feature_node')

        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('n_mels', 64)
        self.declare_parameter('win_length_ms', 25.0)
        self.declare_parameter('hop_length_ms', 10.0)
        self.declare_parameter('window_duration_s', 0.96)

        self.sample_rate = self.get_parameter('sample_rate').value
        self.n_mels = self.get_parameter('n_mels').value
        self.win_length_ms = self.get_parameter('win_length_ms').value
        self.hop_length_ms = self.get_parameter('hop_length_ms').value
        self.window_duration_s = self.get_parameter('window_duration_s').value

        self.win_length = int(self.sample_rate * self.win_length_ms / 1000.0)
        self.hop_length = int(self.sample_rate * self.hop_length_ms / 1000.0)
        self.window_size = int(self.sample_rate * self.window_duration_s)

        self.buffer = deque()

        self.sub_pcm = self.create_subscription(
            AudioPcm16,
            'audio_pcm',
            self.pcm_callback,
            10,
        )

        self.pub_mel = self.create_publisher(AudioMel, 'audio_mel', 10)

        self.get_logger().info(
            f"AudioFeatureNode: sr={self.sample_rate}, n_mels={self.n_mels}, "
            f"win={self.win_length} samples, hop={self.hop_length} samples, "
            f"window_size={self.window_size} samples",
        )

    def pcm_callback(self, msg: AudioPcm16):
        self.buffer.extend(msg.data)

        while len(self.buffer) >= self.window_size:
            window_samples = [self.buffer.popleft() for _ in range(self.window_size)]
            samples = np.asarray(window_samples, dtype=np.float32)
            samples = samples / 32768.0

            mel_spec = librosa.feature.melspectrogram(
                y=samples,
                sr=self.sample_rate,
                n_fft=512,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=0.0,
                fmax=self.sample_rate / 2.0,
                center=True,
                power=2.0,
            )

            log_mel = librosa.power_to_db(mel_spec + 1e-10, ref=1.0)
            n_mels, n_frames = log_mel.shape

            msg_out = AudioMel()
            msg_out.stamp = msg.stamp
            msg_out.sample_rate = self.sample_rate
            msg_out.n_mels = n_mels
            msg_out.n_frames = n_frames
            msg_out.hop_length_s = self.hop_length_ms / 1000.0
            msg_out.win_length_s = self.win_length_ms / 1000.0
            msg_out.data = log_mel.T.astype(np.float32).flatten().tolist()

            self.pub_mel.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = AudioFeatureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
