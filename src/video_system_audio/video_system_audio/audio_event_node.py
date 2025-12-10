#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Time
from video_system_interfaces.msg import AudioEvent, AudioEventArray, AudioMel


@dataclass
class EventState:
    active: bool = False
    start_time: Optional[Time] = None
    frames_above: int = 0


class AudioEventNode(Node):
    def __init__(self):
        super().__init__('audio_event_node')

        self.declare_parameter('model_path', 'models/audio_cry_scream.onnx')
        self.declare_parameter('cry_threshold', 0.6)
        self.declare_parameter('scream_threshold', 0.7)
        self.declare_parameter('min_event_frames', 3)
        self.declare_parameter('idx_cry', 10)
        self.declare_parameter('idx_scream', 20)

        model_path = self.get_parameter('model_path').value
        self.cry_threshold = self.get_parameter('cry_threshold').value
        self.scream_threshold = self.get_parameter('scream_threshold').value
        self.min_event_frames = self.get_parameter('min_event_frames').value
        self.idx_cry = self.get_parameter('idx_cry').value
        self.idx_scream = self.get_parameter('idx_scream').value

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        self.state_cry = EventState()
        self.state_scream = EventState()

        self.sub_mel = self.create_subscription(AudioMel, 'audio_mel', self.mel_callback, 10)
        self.pub_events = self.create_publisher(AudioEventArray, 'audio_events', 10)

        self.get_logger().info(f"AudioEventNode loaded model: {model_path}")

    def mel_callback(self, msg: AudioMel):
        data = np.array(msg.data, dtype=np.float32)
        if data.size != msg.n_frames * msg.n_mels:
            self.get_logger().warn("AudioMel data size mismatch")
            return
        features = data.reshape((msg.n_frames, msg.n_mels))

        x = features[np.newaxis, :, :]

        outputs = self.session.run(None, {self.input_name: x})
        logits = outputs[0]
        probs = self.softmax(logits[0])

        p_cry = float(probs[self.idx_cry])
        p_scream = float(probs[self.idx_scream])

        events_msg = AudioEventArray()
        frame_dt = msg.hop_length_s

        cry_event = self.update_event_state(
            label="cry_high",
            prob=p_cry,
            threshold=self.cry_threshold,
            state=self.state_cry,
            msg=msg,
            frame_dt=frame_dt,
        )
        if cry_event is not None:
            events_msg.events.append(cry_event)

        scream_event = self.update_event_state(
            label="scream",
            prob=p_scream,
            threshold=self.scream_threshold,
            state=self.state_scream,
            msg=msg,
            frame_dt=frame_dt,
        )
        if scream_event is not None:
            events_msg.events.append(scream_event)

        if events_msg.events:
            self.pub_events.publish(events_msg)

    @staticmethod
    def softmax(x):
        x = np.array(x, dtype=np.float32)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def update_event_state(self, label, prob, threshold, state: EventState, msg: AudioMel, frame_dt: float):
        if prob >= threshold:
            state.frames_above += 1
            if not state.active and state.frames_above >= self.min_event_frames:
                state.active = True
                state.start_time = msg.stamp
        else:
            if state.active:
                event_msg = AudioEvent()
                event_msg.label = label
                event_msg.start = state.start_time
                event_msg.end = msg.stamp
                event_msg.probability = float(prob)

                state.active = False
                state.frames_above = 0
                state.start_time = None

                return event_msg
            state.frames_above = 0

        return None


def main(args=None):
    rclpy.init(args=args)
    node = AudioEventNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
