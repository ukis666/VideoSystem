#!/usr/bin/env python3
import rclpy
from rclpy.node import Node as RclpyNode
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from video_system_interfaces.msg import Event
import onnxruntime as ort

# (İsteğe bağlı launch API burada tutulabilir; isim çakışmasın diye alias kullanma notu)
from launch import LaunchDescription
from launch_ros.actions import Node as LaunchNode
def generate_launch_description():
    return LaunchDescription([
        LaunchNode(package='image_tools', executable='cam2image',
                   parameters=[{'video_device': '/dev/video0'}],
                   remappings=[('/image', '/camera/image_raw')]),
        LaunchNode(package='video_system_perception', executable='detector_node_py',
                   parameters=[{
                       'model_path': '/home/uki/VideoSystem/src/video_system_perception/models/yolov8n.onnx',
                       'visualize': True, 'conf_threshold': 0.25, 'nms_threshold': 0.5
                   }])
    ])

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

class DetectorNodePy(RclpyNode):
    def __init__(self):
        super().__init__('detector_node_py')
        self.declare_parameter('model_path', '')
        self.declare_parameter('visualize', True)
        self.declare_parameter('conf_threshold', 0.45)
        self.declare_parameter('nms_threshold', 0.50)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.visualize   = self.get_parameter('visualize').get_parameter_value().bool_value
        self.conf_thr    = float(self.get_parameter('conf_threshold').get_parameter_value().double_value)
        self.nms_thr     = float(self.get_parameter('nms_threshold').get_parameter_value().double_value)

        if not self.model_path:
            raise RuntimeError("model_path parametresi boş")

        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.in_w, self.in_h = 640, 640

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Event, '/events', 10)
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.on_image, 10)

        if self.visualize:
            try: cv2.namedWindow('YOLOv8 ORT (preview)', cv2.WINDOW_NORMAL)
            except: pass

        self.get_logger().info(f'Loaded ORT model: {self.model_path}')

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = self.session.run(None, {self.input_name: blob})[0]  # (1,84,8400)
        det = out[0].T  # (8400,84)

        x_factor = frame.shape[1] / self.in_w
        y_factor = frame.shape[0] / self.in_h

        boxes, scores, class_ids = [], [], []
        for i in range(det.shape[0]):
            cx, cy, w, h = det[i, :4]
            cls = int(np.argmax(det[i, 4:]))
            sc  = float(det[i, 4 + cls])
            if sc < self.conf_thr:
                continue

            left   = int((cx - 0.5 * w) * x_factor)
            top    = int((cy - 0.5 * h) * y_factor)
            width  = int(w * x_factor)
            height = int(h * y_factor)

            left   = max(0, min(left, frame.shape[1]-1))
            top    = max(0, min(top,  frame.shape[0]-1))
            width  = max(0, min(width,  frame.shape[1]-left))
            height = max(0, min(height, frame.shape[0]-top))

            boxes.append([left, top, width, height])
            scores.append(sc)
            class_ids.append(cls)

        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thr, self.nms_thr)
        if len(idxs) > 0:
            for idx in np.array(idxs).flatten():
                x, y, w, h = boxes[idx]
                cls_id = class_ids[idx]
                score = float(scores[idx])

                label_text = "UNKNOWN"
                if 0 <= cls_id < len(COCO_CLASSES):
                    label_text = COCO_CLASSES[cls_id]

                ev = Event()
                ev.stamp = self.get_clock().now().to_msg()
                ev.label = label_text
                ev.score = score
                ev.x, ev.y, ev.w, ev.h = int(x), int(y), int(w), int(h)
                self.pub.publish(ev)

                if self.visualize:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label_text}:{score:.2f}', (x, max(0, y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.visualize:
            try:
                cv2.imshow('YOLOv8 ORT (preview)', frame)
                cv2.waitKey(1)
            except: pass

def main():
    rclpy.init()
    node = DetectorNodePy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
