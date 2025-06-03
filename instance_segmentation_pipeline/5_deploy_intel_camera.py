#!/usr/bin/env python3
"""
ROS2 node for real-time Detectron2 inference on camera stream.
Subscribes to camera topic and publishes detection results with visualization.
Run ros2 launch realsense2_camera rs_launch.py before running this code
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

# Import your existing Detectron2Predictor class
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import os


class Detectron2ROS(Node):
    """ROS2 node for real-time Detectron2 inference."""
    
    def __init__(self):
        super().__init__('detectron2_inference')
        
        # Declare parameters
        self.declare_parameter('model_path', 'output/model_final.pth')
        self.declare_parameter('config_path', 'output/config.yaml')
        self.declare_parameter('annotations_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('publish_detections', True)
        self.declare_parameter('queue_size', 1)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.annotations_path = self.get_parameter('annotations_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        self.publish_det = self.get_parameter('publish_detections').get_parameter_value().bool_value
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        
        # Initialize image conversion (no cv_bridge needed)
        self.get_logger().info("Initializing image processing...")
        
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
            
            # Also publish a JSON string for easier debugging
            self.json_pub = self.create_publisher(
                String,
                '/detectron2/detections_json',
                queue_size
            )
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info(f"Detectron2 ROS node initialized")
        self.get_logger().info(f"Subscribed to: {self.camera_topic}")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Classes: {self.class_names}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
    
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
        self.metadata = MetadataCatalog.get("__detectron2_ros__")
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
                # Add some common paths
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
            else:
                self.get_logger().debug(f"File not found: {ann_path}")
        
        # Fallback to generic names
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        class_names = [f"class_{i}" for i in range(num_classes)]
        self.get_logger().warn(f"Using generic class names: {class_names}")
        self.get_logger().warn(f"To fix this, specify the correct annotations_path parameter")
        return class_names
    
    def ros_image_to_cv2(self, msg):
        """Convert ROS Image message to OpenCV image without cv_bridge."""
        # Get image data
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
        
        # Convert image data to numpy array
        img_array = np.frombuffer(msg.data, dtype=dtype)
        
        if channels == 1:
            img = img_array.reshape((msg.height, msg.width))
        else:
            img = img_array.reshape((msg.height, msg.width, channels))
        
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def cv2_to_ros_image(self, cv_image, encoding="bgr8"):
        """Convert OpenCV image to ROS Image message without cv_bridge."""
        msg = Image()
        msg.height, msg.width = cv_image.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
        msg.data = cv_image.tobytes()
        return msg
    
    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert ROS image to OpenCV (without cv_bridge)
            cv_image = self.ros_image_to_cv2(msg)
            
            # Record inference start time
            import time
            start_time = time.time()
            
            # Run inference
            outputs = self.predictor(cv_image)
            
            # Record inference end time
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            # Extract results
            instances = outputs["instances"].to("cpu")
            
            # Publish detections if requested
            if self.publish_det and len(instances) > 0:
                self.publish_detections(instances, msg.header)
            
            # Publish visualization if requested
            if self.publish_viz:
                self.publish_visualization(cv_image, outputs, msg.header)
            
            # Log performance every 30 frames
            if self.frame_count % 30 == 0:
                avg_fps = self.frame_count / self.total_inference_time
                self.get_logger().info(
                    f"Processed {self.frame_count} frames, "
                    f"Average FPS: {avg_fps:.1f}, "
                    f"Last inference: {inference_time*1000:.1f}ms"
                )
                
                # Log current detections
                if len(instances) > 0:
                    classes = instances.pred_classes.numpy()
                    scores = instances.scores.numpy()
                    unique_classes = np.unique(classes)
                    
                    detection_summary = []
                    for class_id in unique_classes:
                        class_mask = classes == class_id
                        count = np.sum(class_mask)
                        max_conf = np.max(scores[class_mask])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                        detection_summary.append(f"{class_name}:{count}({max_conf:.2f})")
                    
                    self.get_logger().info(f"Current detections: {', '.join(detection_summary)}")
        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def publish_detections(self, instances, header):
        """Publish detection results as Detection2DArray and JSON."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        # Prepare JSON data
        json_data = {
            "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
            "frame_id": header.frame_id,
            "total_detections": len(instances),
            "detections": []
        }
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for i in range(len(instances)):
                # Create Detection2D message
                detection = Detection2D()
                detection.header = header
                
                # Set bounding box
                x1, y1, x2, y2 = boxes[i]
                detection.bbox.center.position.x = float((x1 + x2) / 2)
                detection.bbox.center.position.y = float((y1 + y2) / 2)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)
                
                # Set detection result
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(classes[i])
                hypothesis.hypothesis.score = float(scores[i])
                detection.results.append(hypothesis)
                
                detection_array.detections.append(detection)
                
                # Add to JSON data
                class_name = self.class_names[classes[i]] if classes[i] < len(self.class_names) else f"unknown_{classes[i]}"
                json_data["detections"].append({
                    "class_id": int(classes[i]),
                    "class_name": class_name,
                    "confidence": float(scores[i]),
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
        
        # Publish both formats
        self.det_pub.publish(detection_array)
        
        json_msg = String()
        json_msg.data = json.dumps(json_data, indent=2)
        self.json_pub.publish(json_msg)
    
    def publish_visualization(self, cv_image, outputs, header):
        """Publish visualization image."""
        try:
            # Create visualizer
            v = Visualizer(
                cv_image[:, :, ::-1],  # BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            
            # Draw predictions
            instances = outputs["instances"].to("cpu")
            vis_output = v.draw_instance_predictions(instances)
            vis_img = vis_output.get_image()[:, :, ::-1]  # RGB to BGR
            
            # Ensure the image is in the correct format for OpenCV
            vis_img = np.ascontiguousarray(vis_img, dtype=np.uint8)
            
            # Add summary text overlay
            if len(instances) > 0:
                classes = instances.pred_classes.numpy()
                unique_classes, counts = np.unique(classes, return_counts=True)
                
                y_offset = 30
                # Total detections
                total_text = f"Total: {len(instances)}"
                cv2.putText(vis_img, total_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
                
                # Class counts
                for class_id, count in zip(unique_classes, counts):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                    count_text = f"{class_name}: {count}"
                    cv2.putText(vis_img, count_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 20
            
            # Convert to ROS image and publish (without cv_bridge)
            vis_msg = self.cv2_to_ros_image(vis_img, "bgr8")
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error creating visualization: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = Detectron2ROS()
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