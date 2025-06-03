#!/usr/bin/env python3
"""
ROS2 node for Detectron2 mask segmentation with optical flow contour tracking.
1. Uses Detectron2 to segment objects and extract masks
2. Converts masks to contours
3. Tracks contours using optical flow without running detection again
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Bool
import cv2
import numpy as np
import json
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

# Import Detectron2 components
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
import os


@dataclass
class TrackedContour:
    """Represents a tracked object contour with rigid shape preservation."""
    object_id: int
    class_id: int
    class_name: str
    
    # Original shape (never changes)
    original_contour: np.ndarray  # Original contour shape from detection
    reference_centroid: np.ndarray  # Original centroid
    
    # Tracking points (subset for optical flow)
    tracking_points: np.ndarray  # Small set of points for optical flow
    
    # Current transformation
    current_centroid: np.ndarray  # Current estimated centroid
    confidence: float
    
    # Fields with default values MUST come last
    current_rotation: float = 0.0  # Current rotation angle
    current_scale: float = 1.0     # Current scale factor
    age: int = 0
    lost_tracking: bool = False


class DetectronContourTracker(Node):
    """ROS2 node for mask segmentation and contour tracking."""
    
    def __init__(self):
        super().__init__('detectron2_contour_tracker')
        
        # Declare parameters
        self.declare_parameter('model_path', 'output/model_final.pth')
        self.declare_parameter('config_path', 'output/config.yaml')
        self.declare_parameter('annotations_path', '')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('trigger_topic', '/detectron2/trigger_detection')
        self.declare_parameter('reset_topic', '/detectron2/reset_tracking')
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('publish_masks', True)
        self.declare_parameter('queue_size', 1)
        
        # Contour and tracking parameters
        self.declare_parameter('contour_approximation_epsilon', 0.002)  # More detailed contour
        self.declare_parameter('min_contour_area', 500)  # Lower threshold
        self.declare_parameter('max_tracking_error', 30.0)
        self.declare_parameter('contour_point_spacing', 2)  # More dense sampling
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.annotations_path = self.get_parameter('annotations_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.trigger_topic = self.get_parameter('trigger_topic').get_parameter_value().string_value
        self.reset_topic = self.get_parameter('reset_topic').get_parameter_value().string_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        self.publish_masks = self.get_parameter('publish_masks').get_parameter_value().bool_value
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        
        # Contour parameters
        self.contour_epsilon = self.get_parameter('contour_approximation_epsilon').get_parameter_value().double_value
        self.min_contour_area = self.get_parameter('min_contour_area').get_parameter_value().double_value
        self.max_tracking_error = self.get_parameter('max_tracking_error').get_parameter_value().double_value
        self.contour_point_spacing = self.get_parameter('contour_point_spacing').get_parameter_value().integer_value
        
        # State variables
        self.tracked_contours: List[TrackedContour] = []
        self.next_object_id = 0
        self.frame_count = 0
        self.prev_gray = None
        self.current_gray = None
        self.detection_mode = True  # Start in detection mode
        
        # Optical flow parameters for contour tracking
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
        
        # Trigger detection subscriber
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_detection_callback,
            1
        )
        
        # Reset tracking subscriber
        self.reset_sub = self.create_subscription(
            Bool,
            self.reset_topic,
            self.reset_tracking_callback,
            1
        )
        
        if self.publish_viz:
            self.vis_pub = self.create_publisher(
                Image,
                '/detectron2/contour_visualization',
                queue_size
            )
        
        if self.publish_masks:
            self.mask_pub = self.create_publisher(
                Image,
                '/detectron2/segmentation_masks',
                queue_size
            )
        
        # Status and results publishers
        self.status_pub = self.create_publisher(
            String,
            '/detectron2/tracking_status',
            1
        )
        
        self.results_pub = self.create_publisher(
            String,
            '/detectron2/contour_results',
            queue_size
        )
        
        self.get_logger().info(f"Detectron2 Contour Tracker initialized")
        self.get_logger().info(f"Send Bool(True) to '{self.trigger_topic}' to trigger detection")
        self.get_logger().info(f"Send Bool(True) to '{self.reset_topic}' to reset tracking")
        self.get_logger().info(f"Mode: Detection (waiting for trigger)")
    
    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.ros_image_to_cv2(msg)
            
            # Convert to grayscale for optical flow
            self.current_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            if self.detection_mode:
                # Run segmentation to create new contours
                combined_mask, inference_time = self.run_segmentation(cv_image)
                
                # Publish segmentation mask
                if self.publish_masks:
                    mask_msg = self.cv2_to_ros_image(combined_mask, "mono8")
                    mask_msg.header = msg.header
                    self.mask_pub.publish(mask_msg)
                
                self.detection_mode = False  # Switch to tracking mode
                self.get_logger().info("Switched to tracking mode - tracking contours with optical flow")
            
            else:
                # Track existing contours using optical flow
                if self.prev_gray is not None:
                    self.track_contours_optical_flow()
            
            # Publish visualization and results
            if self.publish_viz:
                self.publish_contour_visualization(cv_image, msg.header)
            
            self.publish_tracking_results(msg.header)
            self.publish_status()
            
            # Update frame state
            self.prev_gray = self.current_gray.copy()
            self.frame_count += 1
            
            # Log status every 30 frames
            if self.frame_count % 30 == 0:
                active_tracks = [obj for obj in self.tracked_contours if not obj.lost_tracking]
                self.get_logger().info(
                    f"Frame {self.frame_count}: Active contours: {len(active_tracks)}/{len(self.tracked_contours)}"
                )
        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def setup_detectron2(self):
        """Initialize Detectron2 predictor and load class names."""
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
        
        self.get_logger().info(f"Using device: {self.cfg.MODEL.DEVICE}")
        self.get_logger().info(f"Classes: {self.class_names}")
    
    def get_class_names(self):
        """Extract class names from annotations file."""
        if self.annotations_path and os.path.exists(self.annotations_path):
            annotation_files = [self.annotations_path]
        else:
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
    
    def trigger_detection_callback(self, msg):
        """Callback to trigger new detection."""
        if msg.data:
            self.detection_mode = True
            self.get_logger().info("Detection triggered - will segment objects in next frame")
    
    def reset_tracking_callback(self, msg):
        """Callback to reset all tracking."""
        if msg.data:
            self.tracked_contours.clear()
            self.next_object_id = 0
            self.detection_mode = True
            self.get_logger().info("Tracking reset - cleared all tracked contours")
    

    def run_segmentation(self, cv_image):
        """Run Detectron2 segmentation and extract contours."""
        self.get_logger().info("Running Detectron2 segmentation...")
        
        start_time = time.time()
        outputs = self.predictor(cv_image)
        inference_time = time.time() - start_time
        
        instances = outputs["instances"].to("cpu")
        combined_mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)
        
        if len(instances) > 0:
            masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            self.get_logger().info(f"Found {len(instances)} objects in {inference_time*1000:.1f}ms")
            
            # Clear existing tracked contours
            self.tracked_contours.clear()
            self.next_object_id = 0
            
            for i in range(len(instances)):
                mask = masks[i].astype(np.uint8) * 255
                class_id = classes[i]
                confidence = scores[i]
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for j, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    
                    if area < self.min_contour_area:
                        continue
                    
                    # Create rigid tracked contour
                    tracked_contour = self.create_rigid_tracked_contour(
                        contour, class_id, class_name, confidence
                    )
                    
                    if tracked_contour is not None:
                        self.tracked_contours.append(tracked_contour)
                        self.next_object_id += 1
                        
                        self.get_logger().info(f"✓ Created rigid contour track {tracked_contour.object_id}: {class_name}")
                
                # Add to combined mask for visualization
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        else:
            self.get_logger().info("No objects detected")
        
        self.get_logger().info(f"Total tracked contours created: {len(self.tracked_contours)}")
        return combined_mask, inference_time
    
    def create_rigid_tracked_contour(self, contour, class_id, class_name, confidence):
        """Create a tracked contour that preserves its original shape."""
        # Smooth and simplify the original contour
        epsilon = self.contour_epsilon * cv2.arcLength(contour, True)
        original_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate reference centroid
        M = cv2.moments(original_contour)
        if M["m00"] == 0:
            return None
        
        reference_centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
        
        # Select strategic tracking points (not the whole contour)
        tracking_points = self.select_tracking_points(original_contour, reference_centroid)
        
        if len(tracking_points) < 5:
            return None
        
        tracked_contour = TrackedContour(
            object_id=self.next_object_id,
            class_id=int(class_id),
            class_name=class_name,
            original_contour=original_contour,
            reference_centroid=reference_centroid,
            tracking_points=tracking_points,
            current_centroid=reference_centroid.copy(),
            confidence=float(confidence)
        )
        
        return tracked_contour
    
    def select_tracking_points(self, contour, centroid):
        """Select a strategic subset of points for optical flow tracking."""
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Method 1: Select points at regular intervals around the contour
        num_tracking_points = min(12, len(points))  # Use 8-12 tracking points
        if len(points) >= num_tracking_points:
            indices = np.linspace(0, len(points)-1, num_tracking_points, dtype=int)
            selected_points = points[indices]
        else:
            selected_points = points
        
        # Method 2: Add some points that are far from centroid (corners/extremes)
        distances = np.linalg.norm(points - centroid, axis=1)
        far_indices = np.argsort(distances)[-4:]  # 4 farthest points
        far_points = points[far_indices]
        
        # Combine and remove duplicates
        all_points = np.vstack([selected_points, far_points])
        unique_points = []
        for point in all_points:
            if not unique_points:
                unique_points.append(point)
            else:
                distances_to_existing = np.linalg.norm(np.array(unique_points) - point, axis=1)
                if np.min(distances_to_existing) > 10:  # At least 10 pixels apart
                    unique_points.append(point)
        
        tracking_points = np.array(unique_points, dtype=np.float32).reshape(-1, 1, 2)
        return tracking_points
    
    def track_contours_optical_flow(self):
        """Track contours using optical flow while preserving shape."""
        if self.prev_gray is None or len(self.tracked_contours) == 0:
            return
        
        for tracked_obj in self.tracked_contours:
            if tracked_obj.lost_tracking or tracked_obj.tracking_points is None:
                continue
            
            # Track the subset of tracking points using optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, self.current_gray,
                tracked_obj.tracking_points, None, **self.lk_params
            )
            
            # Filter out points with high tracking error
            good_points_mask = (status.ravel() == 1) & (error.ravel() < self.max_tracking_error)
            good_new_points = new_points[good_points_mask]
            good_old_points = tracked_obj.tracking_points[good_points_mask]
            
            # Check if we have enough good tracking points
            if len(good_new_points) < 4:
                tracked_obj.lost_tracking = True
                self.get_logger().info(f"Lost tracking for object {tracked_obj.object_id}: {tracked_obj.class_name}")
                continue
            
            # Estimate transformation from good tracking points
            transformation = self.estimate_rigid_transformation(
                good_old_points.reshape(-1, 2), 
                good_new_points.reshape(-1, 2)
            )
            
            if transformation is not None:
                # Update current transformation
                tracked_obj.current_centroid = transformation['translation'] + tracked_obj.reference_centroid
                tracked_obj.current_rotation += transformation['rotation']
                tracked_obj.current_scale *= transformation['scale']
                
                # Update tracking points for next frame
                tracked_obj.tracking_points = good_new_points.reshape(-1, 1, 2)
                tracked_obj.age += 1
            else:
                # If transformation estimation fails, just update centroid
                new_centroid = np.mean(good_new_points.reshape(-1, 2), axis=0)
                tracked_obj.current_centroid = new_centroid
                tracked_obj.tracking_points = good_new_points.reshape(-1, 1, 2)
                tracked_obj.age += 1
    
    def estimate_rigid_transformation(self, old_points, new_points):
        """Estimate translation, rotation, and scale from point correspondences."""
        if len(old_points) < 2 or len(new_points) < 2:
            return None
        
        try:
            # Calculate centroids
            old_centroid = np.mean(old_points, axis=0)
            new_centroid = np.mean(new_points, axis=0)
            
            # Center the points
            old_centered = old_points - old_centroid
            new_centered = new_points - new_centroid
            
            # Estimate scale
            old_scale = np.mean(np.linalg.norm(old_centered, axis=1))
            new_scale = np.mean(np.linalg.norm(new_centered, axis=1))
            
            if old_scale == 0:
                scale = 1.0
            else:
                scale = new_scale / old_scale
                # Limit scale changes to prevent runaway scaling
                scale = np.clip(scale, 0.8, 1.2)
            
            # Estimate rotation using cross-correlation
            rotation = 0.0
            if len(old_points) >= 2:
                # Use the first two points to estimate rotation
                old_vec = old_centered[1] - old_centered[0] if len(old_centered) > 1 else old_centered[0]
                new_vec = new_centered[1] - new_centered[0] if len(new_centered) > 1 else new_centered[0]
                
                old_angle = np.arctan2(old_vec[1], old_vec[0])
                new_angle = np.arctan2(new_vec[1], new_vec[0])
                rotation = new_angle - old_angle
                
                # Limit rotation changes
                rotation = np.clip(rotation, -0.1, 0.1)  # Max 0.1 radians per frame
            
            # Translation is the difference in centroids
            translation = new_centroid - old_centroid
            
            return {
                'translation': translation,
                'rotation': rotation,
                'scale': scale
            }
        
        except Exception as e:
            self.get_logger().warn(f"Failed to estimate transformation: {e}")
            return None
    
    def get_current_contour(self, tracked_obj):
        """Get the current contour by applying transformation to original contour."""
        # Start with original contour
        contour_points = tracked_obj.original_contour.reshape(-1, 2).astype(np.float32)
        
        # Center around reference centroid
        centered_points = contour_points - tracked_obj.reference_centroid
        
        # Apply scale
        scaled_points = centered_points * tracked_obj.current_scale
        
        # Apply rotation
        if abs(tracked_obj.current_rotation) > 0.01:  # Only rotate if significant
            cos_theta = np.cos(tracked_obj.current_rotation)
            sin_theta = np.sin(tracked_obj.current_rotation)
            rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                       [sin_theta, cos_theta]])
            rotated_points = np.dot(scaled_points, rotation_matrix.T)
        else:
            rotated_points = scaled_points
        
        # Apply translation to current centroid
        current_contour = rotated_points + tracked_obj.current_centroid
        
        return current_contour.astype(np.int32)
    
    def publish_contour_visualization(self, cv_image, header):
        """Publish visualization with tracked contours (using rigid shapes)."""
        try:
            vis_img = cv_image.copy()
            
            # Colors for different objects
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]
            
            active_count = 0
            for tracked_obj in self.tracked_contours:
                color = colors[tracked_obj.object_id % len(colors)]
                
                if tracked_obj.lost_tracking:
                    continue
                
                active_count += 1
                
                # Get current transformed contour (preserves original shape)
                current_contour = self.get_current_contour(tracked_obj)
                
                # Draw the rigid contour
                cv2.polylines(vis_img, [current_contour], True, color, 2)
                
                # Draw tracking points (for debugging)
                for point in tracked_obj.tracking_points:
                    pt = tuple(map(int, point.ravel()))
                    cv2.circle(vis_img, pt, 4, color, -1)
                
                # Draw centroid
                centroid_pt = tuple(map(int, tracked_obj.current_centroid))
                cv2.circle(vis_img, centroid_pt, 6, color, 2)
                
                # Draw label
                label = f"ID:{tracked_obj.object_id} {tracked_obj.class_name} (Age:{tracked_obj.age})"
                cv2.putText(vis_img, label, centroid_pt, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add status info
            mode_text = "DETECTION MODE" if self.detection_mode else "TRACKING MODE"
            status_text = f"{mode_text} | Active: {active_count} | Frame: {self.frame_count}"
            cv2.putText(vis_img, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to ROS image and publish
            vis_msg = self.cv2_to_ros_image(vis_img, "bgr8")
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error creating contour visualization: {e}")
    
    def publish_tracking_results(self, header):
        """Publish contour tracking results as JSON (with rigid contours)."""
        results = {
            "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
            "frame_id": header.frame_id,
            "frame_count": self.frame_count,
            "mode": "detection" if self.detection_mode else "tracking",
            "total_objects": len(self.tracked_contours),
            "active_objects": len([obj for obj in self.tracked_contours if not obj.lost_tracking]),
            "contours": []
        }
        
        for tracked_obj in self.tracked_contours:
            if tracked_obj.lost_tracking:
                continue
            
            # Get current rigid contour
            current_contour = self.get_current_contour(tracked_obj)
            contour_list = current_contour.tolist()
            
            results["contours"].append({
                "object_id": tracked_obj.object_id,
                "class_id": tracked_obj.class_id,
                "class_name": tracked_obj.class_name,
                "confidence": tracked_obj.confidence,
                "age": tracked_obj.age,
                "centroid": tracked_obj.current_centroid.tolist(),
                "rotation": float(tracked_obj.current_rotation),
                "scale": float(tracked_obj.current_scale),
                "contour_points": contour_list,
                "num_points": len(contour_list),
                "tracking_points": len(tracked_obj.tracking_points)
            })
        
        results_msg = String()
        results_msg.data = json.dumps(results, indent=2)
        self.results_pub.publish(results_msg)

    def publish_status(self):
        """Publish current tracking status."""
        active_objects = [obj for obj in self.tracked_contours if not obj.lost_tracking]
        
        status = {
            "mode": "detection" if self.detection_mode else "tracking",
            "frame_count": self.frame_count,
            "total_tracked_objects": len(self.tracked_contours),
            "active_tracked_objects": len(active_objects),
            "ready_for_trigger": not self.detection_mode
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = DetectronContourTracker()
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