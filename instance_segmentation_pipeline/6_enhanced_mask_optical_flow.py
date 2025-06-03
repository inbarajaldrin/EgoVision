#!/usr/bin/env python3
"""
ROS2 node for real-time Detectron2 inference with optical flow tracking.
Reduces detection flickering by tracking objects between frames.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import cv2
import numpy as np
import json
import torch
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import your existing Detectron2 components
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import os


@dataclass
class TrackedObject:
    """Represents a tracked object with detection and tracking history."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    last_detection_frame: int
    tracking_points: np.ndarray  # Points for optical flow tracking
    age: int = 0
    missed_detections: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # dx, dy per frame


class Detectron2TrackingROS(Node):
    """ROS2 node for real-time Detectron2 inference with tracking."""
    
    def __init__(self):
        super().__init__('detectron2_tracking_inference')
        
        # Declare parameters
        self.declare_parameter('model_path', 'output/model_final.pth')
        self.declare_parameter('config_path', 'output/config.yaml')
        self.declare_parameter('annotations_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('publish_detections', True)
        self.declare_parameter('queue_size', 1)
        
        # Tracking parameters
        self.declare_parameter('max_missed_detections', 10)
        self.declare_parameter('tracking_confidence_boost', 0.1)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('detection_interval', 3)  # Run detection every N frames
        self.declare_parameter('optical_flow_points', 100)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.annotations_path = self.get_parameter('annotations_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        self.publish_det = self.get_parameter('publish_detections').get_parameter_value().bool_value
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        
        # Tracking parameters
        self.max_missed_detections = self.get_parameter('max_missed_detections').get_parameter_value().integer_value
        self.tracking_confidence_boost = self.get_parameter('tracking_confidence_boost').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.detection_interval = self.get_parameter('detection_interval').get_parameter_value().integer_value
        self.optical_flow_points = self.get_parameter('optical_flow_points').get_parameter_value().integer_value
        
        # Tracking state
        self.tracked_objects: List[TrackedObject] = []
        self.next_track_id = 0
        self.frame_count = 0
        self.prev_gray = None
        self.current_gray = None
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize Detectron2 model
        self.get_logger().info("Loading Detectron2 model...")
        self.setup_detectron2()
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            queue_size
        )
        
        if self.publish_viz:
            self.vis_pub = self.create_publisher(
                Image,
                '/detectron2/visualization',
                queue_size
            )
        
        if self.publish_det:
            self.det_pub = self.create_publisher(
                Detection2DArray,
                '/detectron2/detections',
                queue_size
            )
            
            self.json_pub = self.create_publisher(
                String,
                '/detectron2/detections_json',
                queue_size
            )
        
        # Performance tracking
        self.total_inference_time = 0.0
        self.total_tracking_time = 0.0
        
        self.get_logger().info(f"Detectron2 Tracking ROS node initialized")
        self.get_logger().info(f"Detection interval: every {self.detection_interval} frames")
        self.get_logger().info(f"Max missed detections: {self.max_missed_detections}")
        self.get_logger().info(f"IoU threshold: {self.iou_threshold}")
    
    def setup_detectron2(self):
        """Initialize Detectron2 predictor and load class names."""
        # Setup configuration
        self.cfg = get_cfg()
        
        # Load config
        if os.path.exists(self.config_path):
            self.cfg.merge_from_file(self.config_path)
            self.get_logger().info(f"Loaded config from: {self.config_path}")
        else:
            self.get_logger().warn("Config file not found, using default Mask R-CNN config")
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Set model parameters
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        # Load class names
        self.class_names = self.get_class_names()
        
        # Setup metadata for visualization
        self.metadata = MetadataCatalog.get("__detectron2_tracking_ros__")
        self.metadata.thing_classes = self.class_names
        
        self.get_logger().info(f"Using device: {self.cfg.MODEL.DEVICE}")
    
    def get_class_names(self):
        """Extract class names from annotations file."""
        # Try direct path first
        if self.annotations_path and os.path.exists(self.annotations_path):
            annotation_files = [self.annotations_path]
        else:
            # Try to find the original annotations file in common locations
            annotation_files = [
                os.path.join(os.path.dirname(self.model_path), "..", "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "..", "..", "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "coco_annotations.json"),
                "/home/aaugus11/Downloads/FBM_Assembly3_raw2/coco_annotations.json",
                os.path.expanduser("~/Downloads/FBM_Assembly3_raw2/coco_annotations.json"),
            ]
        
        for ann_path in annotation_files:
            if os.path.exists(ann_path):
                try:
                    self.get_logger().info(f"Loading class names from: {ann_path}")
                    with open(ann_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    if "categories" in coco_data and len(coco_data["categories"]) > 0:
                        categories = sorted(coco_data["categories"], key=lambda x: x.get("id", 0))
                        class_names = [cat["name"] for cat in categories]
                        self.get_logger().info(f"Found class names: {class_names}")
                        return class_names
                    
                except Exception as e:
                    self.get_logger().error(f"Could not load class names from {ann_path}: {e}")
        
        # Fallback to generic names
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        class_names = [f"class_{i}" for i in range(num_classes)]
        self.get_logger().warn(f"Using generic class names: {class_names}")
        return class_names
    
    def ros_image_to_cv2(self, msg):
        """Convert ROS Image message to OpenCV image."""
        if msg.encoding == "rgb8":
            channels = 3
            dtype = np.uint8
        elif msg.encoding == "bgr8":
            channels = 3
            dtype = np.uint8
        elif msg.encoding == "mono8":
            channels = 1
            dtype = np.uint8
        elif msg.encoding == "mono16":
            channels = 1
            dtype = np.uint16
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        
        img_array = np.frombuffer(msg.data, dtype=dtype)
        
        if channels == 1:
            img = img_array.reshape((msg.height, msg.width))
        else:
            img = img_array.reshape((msg.height, msg.width, channels))
        
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def cv2_to_ros_image(self, cv_image, encoding="bgr8"):
        """Convert OpenCV image to ROS Image message."""
        msg = Image()
        msg.height, msg.width = cv_image.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
        msg.data = cv_image.tobytes()
        return msg
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_tracking_points(self, bbox, mask=None):
        """Extract corner points and edge points for optical flow tracking."""
        x1, y1, x2, y2 = bbox
        
        # Corner points
        corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],  # Corners
            [(x1+x2)/2, y1], [(x1+x2)/2, y2],        # Top/bottom center
            [x1, (y1+y2)/2], [x2, (y1+y2)/2],        # Left/right center
            [(x1+x2)/2, (y1+y2)/2]                   # Center
        ], dtype=np.float32)
        
        # Add some points within the bounding box
        width, height = x2 - x1, y2 - y1
        num_internal_points = min(20, self.optical_flow_points - len(corners))
        
        if num_internal_points > 0:
            internal_points = []
            for _ in range(num_internal_points):
                px = x1 + np.random.uniform(0.2, 0.8) * width
                py = y1 + np.random.uniform(0.2, 0.8) * height
                internal_points.append([px, py])
            
            corners = np.vstack([corners, np.array(internal_points, dtype=np.float32)])
        
        return corners.reshape(-1, 1, 2)
    
    def track_objects_optical_flow(self):
        """Track existing objects using optical flow."""
        if self.prev_gray is None or len(self.tracked_objects) == 0:
            return
        
        track_start_time = time.time()
        
        for obj in self.tracked_objects:
            if obj.tracking_points is None or len(obj.tracking_points) == 0:
                continue
            
            # Track points using Lucas-Kanade optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, self.current_gray,
                obj.tracking_points, None, **self.lk_params
            )
            
            # Keep only good points
            good_new = new_points[status == 1]
            good_old = obj.tracking_points[status == 1]
            
            if len(good_new) < 3:  # Need at least 3 points for reasonable tracking
                obj.missed_detections += 1
                continue
            
            # Calculate movement
            movement = np.mean(good_new - good_old, axis=0)
            obj.velocity = (float(movement[0]), float(movement[1]))
            
            # Update bounding box based on point movement
            x1, y1, x2, y2 = obj.bbox
            x1 += movement[0]
            y1 += movement[1]
            x2 += movement[0]
            y2 += movement[1]
            
            # Ensure bbox stays within image bounds
            img_h, img_w = self.current_gray.shape
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))
            
            obj.bbox = (x1, y1, x2, y2)
            obj.tracking_points = good_new.reshape(-1, 1, 2)
            obj.age += 1
        
        tracking_time = time.time() - track_start_time
        self.total_tracking_time += tracking_time
    
    def run_detection(self, cv_image):
        """Run Detectron2 detection on the current frame."""
        start_time = time.time()
        outputs = self.predictor(cv_image)
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        instances = outputs["instances"].to("cpu")
        detections = []
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for i in range(len(instances)):
                x1, y1, x2, y2 = boxes[i]
                class_id = classes[i]
                confidence = scores[i]
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(confidence)
                })
        
        return detections, inference_time
    
    def associate_detections_to_tracks(self, detections):
        """Associate new detections with existing tracks using IoU matching."""
        if len(self.tracked_objects) == 0:
            # No existing tracks, create new ones
            for det in detections:
                self.create_new_track(det)
            return
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracked_objects), len(detections)))
        for i, obj in enumerate(self.tracked_objects):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.calculate_iou(obj.bbox, det['bbox'])
        
        # Find best matches using greedy assignment
        matched_tracks = set()
        matched_detections = set()
        
        # Sort by IoU score descending
        matches = []
        for i in range(len(self.tracked_objects)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Assign matches
        for track_idx, det_idx, iou_score in matches:
            if track_idx not in matched_tracks and det_idx not in matched_detections:
                self.update_track(track_idx, detections[det_idx])
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # Mark unmatched tracks as missed
        for i, obj in enumerate(self.tracked_objects):
            if i not in matched_tracks:
                obj.missed_detections += 1
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                self.create_new_track(det)
    
    def create_new_track(self, detection):
        """Create a new tracked object."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        tracking_points = self.extract_tracking_points(detection['bbox'])
        
        tracked_obj = TrackedObject(
            track_id=track_id,
            class_id=detection['class_id'],
            class_name=detection['class_name'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            last_detection_frame=self.frame_count,
            tracking_points=tracking_points,
            age=0,
            missed_detections=0
        )
        
        self.tracked_objects.append(tracked_obj)
    
    def update_track(self, track_idx, detection):
        """Update an existing track with new detection."""
        obj = self.tracked_objects[track_idx]
        
        # Update bbox and confidence
        obj.bbox = detection['bbox']
        obj.confidence = detection['confidence'] + self.tracking_confidence_boost
        obj.last_detection_frame = self.frame_count
        obj.missed_detections = 0
        obj.age += 1
        
        # Update tracking points
        obj.tracking_points = self.extract_tracking_points(detection['bbox'])
    
    def cleanup_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        self.tracked_objects = [
            obj for obj in self.tracked_objects
            if obj.missed_detections < self.max_missed_detections
        ]
    
    def image_callback(self, msg):
        """Process incoming camera images with tracking."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.ros_image_to_cv2(msg)
            
            # Convert to grayscale for optical flow
            self.current_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Run detection periodically or on first frame
            if self.frame_count % self.detection_interval == 0 or self.prev_gray is None:
                detections, inference_time = self.run_detection(cv_image)
                self.associate_detections_to_tracks(detections)
            else:
                inference_time = 0.0
            
            # Track existing objects using optical flow
            if self.prev_gray is not None:
                self.track_objects_optical_flow()
            
            # Clean up lost tracks
            self.cleanup_lost_tracks()
            
            # Publish results
            if self.publish_det:
                self.publish_tracked_detections(msg.header)
            
            if self.publish_viz:
                self.publish_tracking_visualization(cv_image, msg.header)
            
            # Update frame state
            self.prev_gray = self.current_gray.copy()
            self.frame_count += 1
            
            # Log performance every 30 frames
            if self.frame_count % 30 == 0:
                avg_fps = self.frame_count / (self.total_inference_time + self.total_tracking_time + 1e-6)
                detection_fps = (self.frame_count // self.detection_interval) / (self.total_inference_time + 1e-6)
                
                self.get_logger().info(
                    f"Frame {self.frame_count}: Total FPS: {avg_fps:.1f}, "
                    f"Detection FPS: {detection_fps:.1f}, "
                    f"Active tracks: {len(self.tracked_objects)}"
                )
                
                if len(self.tracked_objects) > 0:
                    track_summary = []
                    class_counts = {}
                    for obj in self.tracked_objects:
                        class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
                    
                    for class_name, count in class_counts.items():
                        track_summary.append(f"{class_name}:{count}")
                    
                    self.get_logger().info(f"Tracking: {', '.join(track_summary)}")
        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def publish_tracked_detections(self, header):
        """Publish tracked detections."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        json_data = {
            "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
            "frame_id": header.frame_id,
            "total_tracks": len(self.tracked_objects),
            "detections": []
        }
        
        for obj in self.tracked_objects:
            # Create Detection2D message
            detection = Detection2D()
            detection.header = header
            
            # Set bounding box
            x1, y1, x2, y2 = obj.bbox
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Set detection result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(obj.class_id)
            hypothesis.hypothesis.score = float(min(1.0, obj.confidence))  # Cap at 1.0
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
            
            # Add to JSON data
            json_data["detections"].append({
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "confidence": float(min(1.0, obj.confidence)),
                "age": obj.age,
                "missed_detections": obj.missed_detections,
                "velocity": obj.velocity,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                }
            })
        
        self.det_pub.publish(detection_array)
        
        json_msg = String()
        json_msg.data = json.dumps(json_data, indent=2)
        self.json_pub.publish(json_msg)
    
    def publish_tracking_visualization(self, cv_image, header):
        """Publish visualization with tracking information."""
        try:
            vis_img = cv_image.copy()
            
            # Colors for different tracks
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]
            
            for obj in self.tracked_objects:
                x1, y1, x2, y2 = [int(coord) for coord in obj.bbox]
                color = colors[obj.track_id % len(colors)]
                
                # Draw bounding box
                thickness = 3 if obj.missed_detections == 0 else 1
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                
                # Draw tracking points
                if obj.tracking_points is not None and len(obj.tracking_points) > 0:
                    for point in obj.tracking_points:
                        pt = tuple(map(int, point.ravel()))
                        cv2.circle(vis_img, pt, 2, color, -1)
                
                # Draw label with track info
                label = f"ID:{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
                if obj.missed_detections > 0:
                    label += f" (miss:{obj.missed_detections})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw velocity vector
                if obj.velocity != (0.0, 0.0):
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    end_x = int(center_x + obj.velocity[0] * 10)
                    end_y = int(center_y + obj.velocity[1] * 10)
                    cv2.arrowedLine(vis_img, (center_x, center_y), (end_x, end_y), color, 2)
            
            # Add summary info
            summary_text = f"Tracks: {len(self.tracked_objects)} | Frame: {self.frame_count}"
            cv2.putText(vis_img, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to ROS image and publish
            vis_msg = self.cv2_to_ros_image(vis_img, "bgr8")
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error creating tracking visualization: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = Detectron2TrackingROS()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()