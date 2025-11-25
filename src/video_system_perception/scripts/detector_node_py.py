#!/usr/bin/env python3
import rclpy
from rclpy.node import Node as RclpyNode
import cv2
import numpy as np

from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from video_system_interfaces.msg import Event
import onnxruntime as ort

from launch import LaunchDescription
from launch_ros.actions import Node as LaunchNode


def generate_launch_description():
    """Embedded launch description for quick local testing."""
    return LaunchDescription([
        LaunchNode(
            package="image_tools",
            executable="cam2image",
            parameters=[{"video_device": "/dev/video0"}],
            remappings=[("/image", "/camera/image_raw")],
        ),
        LaunchNode(
            package="video_system_perception",
            executable="detector_node_py",
            parameters=[{
                "model_path": "/home/uki/VideoSystem/src/video_system_perception/models/yolov8n.onnx",
                "visualize": True,
                "conf_threshold": 0.25,
                "nms_threshold": 0.5,
                "frame_rate": 15.0,
                # Homography example (update path and enable when ready)
                "use_homography": False,
                "homography_path": "/home/uki/VideoSystem/src/video_system_perception/config/homography_room1.npy",
            }],
        ),
    ])


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class DetectorNodePy(RclpyNode):
    """ONNX Runtime YOLOv8 detector with ByteTrack MOT and optional homography mapping."""

    def __init__(self):
        super().__init__("detector_node_py")

        # Detection parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("visualize", True)
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("nms_threshold", 0.50)
        self.declare_parameter("frame_rate", 15.0)

        # Homography parameters
        self.declare_parameter("use_homography", False)
        self.declare_parameter("homography_path", "")

        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )
        self.conf_thr = float(
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.nms_thr = float(
            self.get_parameter("nms_threshold").get_parameter_value().double_value
        )
        self.frame_rate = float(
            self.get_parameter("frame_rate").get_parameter_value().double_value
        )

        self.use_homography = (
            self.get_parameter("use_homography").get_parameter_value().bool_value
        )
        self.homography_path = (
            self.get_parameter("homography_path").get_parameter_value().string_value
        )
        self.H = None  # 3x3 homography matrix in float32

        if not self.model_path:
            raise RuntimeError("model_path parametresi boş")

        # ONNX Runtime session
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.in_w, self.in_h = 640, 640

        # ByteTrack tracker
        self.tracker = BYTETracker(
            track_thresh=self.conf_thr,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=self.frame_rate,
        )

        # Homography loading
        if self.use_homography:
            if not self.homography_path:
                self.get_logger().error(
                    "use_homography true ancak homography_path boş; homografi devre dışı bırakılıyor."
                )
                self.use_homography = False
            else:
                try:
                    mat = np.load(self.homography_path)
                    mat = np.asarray(mat, dtype=np.float32)
                    if mat.shape == (3, 3):
                        self.H = mat
                        self.get_logger().info(
                            f"Homography matrix yüklendi: {self.homography_path}"
                        )
                    else:
                        self.get_logger().error(
                            f"Geçersiz homografi boyutu {mat.shape}; (3,3) bekleniyor. Homografi devre dışı."
                        )
                        self.use_homography = False
                except Exception as e:
                    self.get_logger().error(
                        f"Homografi matrisi yüklenemedi ({self.homography_path}): {e}"
                    )
                    self.use_homography = False

        # ROS I/O
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Event, "/events", 10)
        self.sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.on_image,
            10,
        )

        if self.visualize:
            try:
                cv2.namedWindow("YOLOv8 ORT (preview)", cv2.WINDOW_NORMAL)
            except Exception:
                pass

        self.get_logger().info(f"Loaded ORT model: {self.model_path}")
        self.get_logger().info(
            f"ByteTrack initialized (conf={self.conf_thr}, fps={self.frame_rate})"
        )
        if self.use_homography and self.H is not None:
            self.get_logger().info("Homography mapping etkin.")

    def _apply_homography(self, x: float, y: float, w: float, h: float):
        """Apply homography to the bounding-box center and return (wx, wy) in target plane.

        Returns None if homography is not enabled or numerically invalid.
        """
        if not (self.use_homography and self.H is not None):
            return None

        cx = x + 0.5 * w
        cy = y + 0.5 * h

        p = np.array([cx, cy, 1.0], dtype=np.float32)
        pw = self.H @ p
        if abs(pw[2]) < 1e-6:
            return None

        wx = float(pw[0] / pw[2])
        wy = float(pw[1] / pw[2])
        return wx, wy

    def on_image(self, msg: Image) -> None:
        """Image callback: run detection + tracking and publish events."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        img_h, img_w = frame.shape[:2]

        # Preprocess for YOLOv8 ONNX
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.in_w, self.in_h),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = (
            resized.astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[None, ...]

        # Inference
        outputs = self.session.run(None, {self.input_name: blob})[0]  # (1,84,8400)
        det = outputs[0].T  # (8400,84)

        x_factor = img_w / self.in_w
        y_factor = img_h / self.in_h

        # Collect raw detections in (x, y, w, h) format
        boxes_xywh = []
        scores = []
        class_ids = []

        for i in range(det.shape[0]):
            cx, cy, w, h = det[i, :4]
            cls = int(np.argmax(det[i, 4:]))
            sc = float(det[i, 4 + cls])
            if sc < self.conf_thr:
                continue

            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            left = max(0, min(left, img_w - 1))
            top = max(0, min(top, img_h - 1))
            width = max(0, min(width, img_w - left))
            height = max(0, min(height, img_h - top))

            boxes_xywh.append([left, top, width, height])
            scores.append(sc)
            class_ids.append(cls)

        # No detections before NMS
        if not boxes_xywh:
            self.tracker.update(
                output_results=np.empty((0, 5), dtype=np.float32),
                img_info=(img_w, img_h),
                img_size=(img_w, img_h),
            )
            if self.visualize:
                try:
                    cv2.imshow("YOLOv8 ORT (preview)", frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        # Apply NMS on (x, y, w, h) boxes
        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            self.conf_thr,
            self.nms_thr,
        )

        if len(idxs) == 0:
            self.tracker.update(
                output_results=np.empty((0, 5), dtype=np.float32),
                img_info=(img_w, img_h),
                img_size=(img_w, img_h),
            )
            if self.visualize:
                try:
                    cv2.imshow("YOLOv8 ORT (preview)", frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        idxs = np.array(idxs).flatten()

        # NMS-filtered detections
        nms_boxes_xywh = []
        nms_scores = []
        nms_class_ids = []

        for idx in idxs:
            x, y, w, h = boxes_xywh[idx]
            nms_boxes_xywh.append([x, y, w, h])
            nms_scores.append(float(scores[idx]))
            nms_class_ids.append(int(class_ids[idx]))

        nms_boxes_xywh = np.asarray(nms_boxes_xywh, dtype=np.float32)
        nms_scores = np.asarray(nms_scores, dtype=np.float32)
        nms_class_ids = np.asarray(nms_class_ids, dtype=np.int32)

        # Convert to tlbr (x1, y1, x2, y2) for ByteTrack
        tlbr_boxes = nms_boxes_xywh.copy()
        tlbr_boxes[:, 2] += tlbr_boxes[:, 0]
        tlbr_boxes[:, 3] += tlbr_boxes[:, 1]

        # Prepare input for ByteTrack: [x1, y1, x2, y2, score]
        detections_for_tracker = np.concatenate(
            [tlbr_boxes, nms_scores[:, np.newaxis]],
            axis=1,
        )

        # Update tracker
        tracks = self.tracker.update(
            output_results=detections_for_tracker,
            img_info=(img_w, img_h),
            img_size=(img_w, img_h),
        )

        # Match detections to tracks
        track_ids = [-1] * len(tlbr_boxes)
        track_ids = match_detections_with_tracks(
            tlbr_boxes=tlbr_boxes,
            track_ids=track_ids,
            tracks=tracks,
        )

        # Keep only detections that are associated with an active track
        tracked_boxes_xywh = []
        tracked_scores = []
        tracked_class_ids = []
        tracked_ids = []

        for box_xywh, score, cls_id, tid in zip(
            nms_boxes_xywh,
            nms_scores,
            nms_class_ids,
            track_ids,
        ):
            if tid == -1:
                continue
            tracked_boxes_xywh.append(box_xywh)
            tracked_scores.append(score)
            tracked_class_ids.append(cls_id)
            tracked_ids.append(int(tid))

        if not tracked_boxes_xywh:
            if self.visualize:
                try:
                    cv2.imshow("YOLOv8 ORT (preview)", frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        tracked_boxes_xywh = np.asarray(tracked_boxes_xywh, dtype=np.float32)
        tracked_scores = np.asarray(tracked_scores, dtype=np.float32)
        tracked_class_ids = np.asarray(tracked_class_ids, dtype=np.int32)
        tracked_ids = np.asarray(tracked_ids, dtype=np.int32)

        # Log active track IDs for debugging
        self.get_logger().info(f"Active track IDs: {tracked_ids.tolist()}")

        # Publish Event messages for tracked detections
        now = self.get_clock().now().to_msg()

        for (x, y, w, h), score, cls_id, tid in zip(
            tracked_boxes_xywh,
            tracked_scores,
            tracked_class_ids,
            tracked_ids,
        ):
            cls_id_int = int(cls_id)
            label_text = "UNKNOWN"
            if 0 <= cls_id_int < len(COCO_CLASSES):
                label_text = COCO_CLASSES[cls_id_int]

            ev = Event()
            ev.stamp = now
            ev.label = label_text
            ev.score = float(score)
            ev.x = int(x)
            ev.y = int(y)
            ev.w = int(w)
            ev.h = int(h)

            # Optional fields depending on message definition
            if hasattr(ev, "track_id"):
                ev.track_id = int(tid)

            world_xy = self._apply_homography(x, y, w, h)
            if world_xy is not None:
                wx, wy = world_xy
                if hasattr(ev, "world_x"):
                    ev.world_x = float(wx)
                if hasattr(ev, "world_y"):
                    ev.world_y = float(wy)
                self.get_logger().info(
                    f"Track {tid} mapped to world coords: ({wx:.3f}, {wy:.3f})"
                )

            self.pub.publish(ev)

            if self.visualize:
                try:
                    x_i, y_i, w_i, h_i = int(x), int(y), int(w), int(h)
                    cv2.rectangle(
                        frame,
                        (x_i, y_i),
                        (x_i + w_i, y_i + h_i),
                        (0, 255, 0),
                        2,
                    )
                    label_draw = f"ID {tid} | {label_text}:{score:.2f}"
                    cv2.putText(
                        frame,
                        label_draw,
                        (x_i, max(0, y_i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                except Exception:
                    pass

        if self.visualize:
            try:
                cv2.imshow("YOLOv8 ORT (preview)", frame)
                cv2.waitKey(1)
            except Exception:
                pass


def main() -> None:
    rclpy.init()
    node = DetectorNodePy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

