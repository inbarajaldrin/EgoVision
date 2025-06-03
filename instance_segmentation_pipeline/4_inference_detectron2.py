#!/usr/bin/env python3
"""
Detectron2 inference script for trained mask models.
Generic script that works with any trained model and properly handles multiple object classes.
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo


class Detectron2Predictor:
    """Generic predictor class for any trained Detectron2 model."""
    
    def __init__(self, model_path: str, config_path: str = None, confidence_threshold: float = 0.5, 
                 annotations_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model (.pth file)
            config_path: Path to config.yaml (optional, will auto-detect)
            confidence_threshold: Minimum confidence for predictions
            annotations_path: Path to instances_train.json file with class names (optional)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect config path if not provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        
        self.config_path = config_path
        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = self._get_class_names(annotations_path)
        
        print(f"Loaded model: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def _setup_config(self):
        """Setup configuration for inference."""
        cfg = get_cfg()
        
        # Try to load from saved config first
        if os.path.exists(self.config_path):
            cfg.merge_from_file(self.config_path)
            print(f"Loaded config from: {self.config_path}")
        else:
            # Fallback to default Mask R-CNN config
            print("Config file not found, using default Mask R-CNN config")
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Set model weights and confidence threshold
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        return cfg
    
    def _get_class_names(self, annotations_path: str = None):
        """Extract class names from the original dataset annotations."""
        if annotations_path and os.path.exists(annotations_path):
            # Use provided annotations path
            possible_paths = [annotations_path]
        else:
            # Try to find the original annotations file
            possible_paths = [
                os.path.join(os.path.dirname(self.model_path), "..", "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "..", "..", "annotations", "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "..", "..", "..", "annotations", "instances_train.json"),
                # Also check in the same directory as the model
                os.path.join(os.path.dirname(self.model_path), "instances_train.json"),
                os.path.join(os.path.dirname(self.model_path), "coco_annotations.json"),
            ]
        
        for ann_path in possible_paths:
            if os.path.exists(ann_path):
                try:
                    print(f"Trying to load class names from: {ann_path}")
                    with open(ann_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    if "categories" in coco_data and len(coco_data["categories"]) > 0:
                        # Sort by id to ensure correct order (COCO format uses 1-based indexing)
                        categories = sorted(coco_data["categories"], key=lambda x: x.get("id", 0))
                        class_names = [cat["name"] for cat in categories]
                        print(f"Found class names from annotations: {class_names}")
                        print(f"Category details: {[(cat['id'], cat['name']) for cat in categories]}")
                        return class_names
                    else:
                        print(f"No categories found in {ann_path}")
                except Exception as e:
                    print(f"Could not load class names from {ann_path}: {e}")
            else:
                print(f"File not found: {ann_path}")
        
        # Fallback to generic names based on number of classes
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"Could not find annotations file. Using generic class names: {class_names}")
        print(f"Tip: Use --annotations-path to specify the location of your instances_train.json file")
        return class_names
    
    def predict_image(self, image_path: str):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            tuple: (detectron2_outputs, parsed_results, original_image)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and predict
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        outputs = self.predictor(img)
        
        # Extract results
        instances = outputs["instances"].to("cpu")
        
        # Parse results by class
        results = {
            "image_path": image_path,
            "image_shape": img.shape,
            "total_detections": len(instances),
            "detections_by_class": {},
            "all_detections": []
        }
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
            
            # Group by class
            for i in range(len(instances)):
                class_id = int(classes[i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                score = float(scores[i])
                box = boxes[i].tolist()
                
                # Add to class grouping
                if class_name not in results["detections_by_class"]:
                    results["detections_by_class"][class_name] = []
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": score,
                    "bbox": box,  # [x1, y1, x2, y2]
                    "mask": masks[i] if masks is not None else None
                }
                
                results["detections_by_class"][class_name].append(detection)
                results["all_detections"].append(detection)
        
        return outputs, results, img
    
    def visualize_predictions(self, image_path: str, output_path: str = None, 
                            show_class_names: bool = True, show_scores: bool = True):
        """
        Visualize predictions on an image with proper class labels.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            show_class_names: Whether to show class names in labels
            show_scores: Whether to show confidence scores in labels
            
        Returns:
            numpy.ndarray: Visualization image
        """
        outputs, results, img = self.predict_image(image_path)
        
        # Create visualizer with class names
        metadata = MetadataCatalog.get("__temp__")
        metadata.thing_classes = self.class_names
        
        v = Visualizer(
            img[:, :, ::-1],  # Convert BGR to RGB
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE if results["total_detections"] > 0 else ColorMode.IMAGE
        )
        
        # Draw predictions
        instances = outputs["instances"].to("cpu")
        vis_output = v.draw_instance_predictions(instances)
        vis_img = vis_output.get_image()
        
        # Add summary text
        y_offset = 30
        total_text = f"Total detections: {results['total_detections']}"
        cv2.putText(vis_img, total_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Add class-wise summary
        for class_name, detections in results["detections_by_class"].items():
            count_text = f"{class_name}: {len(detections)}"
            cv2.putText(vis_img, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, vis_img[:, :, ::-1])  # Convert RGB back to BGR for saving
            print(f"Saved visualization: {output_path}")
        
        return vis_img
    
    def print_detection_summary(self, results: dict):
        """Print a detailed summary of detections."""
        print(f"\nDetection Summary:")
        print(f"Total detections: {results['total_detections']}")
        
        if results['total_detections'] == 0:
            print("No objects detected.")
            return
        
        print("\nBy class:")
        for class_name, detections in results["detections_by_class"].items():
            print(f"  {class_name}: {len(detections)} detections")
            
            # Show top 3 detections for this class
            sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            for i, det in enumerate(sorted_detections[:3]):
                x1, y1, x2, y2 = det["bbox"]
                print(f"    #{i+1}: confidence={det['confidence']:.3f}, "
                      f"bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    def batch_predict(self, input_folder: str, output_folder: str, 
                     extensions: list = ['.jpg', '.jpeg', '.png']):
        """
        Run inference on all images in a folder.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save results
            extensions: List of image file extensions to process
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process...")
        
        batch_summary = {
            "total_images": len(image_files),
            "total_detections": 0,
            "class_counts": {},
            "image_results": []
        }
        
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
            
            try:
                # Run inference
                outputs, results, img = self.predict_image(str(img_file))
                
                # Save visualization
                vis_output_path = output_path / f"{img_file.stem}_result{img_file.suffix}"
                self.visualize_predictions(str(img_file), str(vis_output_path))
                
                # Update batch summary
                batch_summary["total_detections"] += results["total_detections"]
                
                image_result = {
                    "image": img_file.name,
                    "detections": results["total_detections"],
                    "classes_found": list(results["detections_by_class"].keys())
                }
                
                # Update class counts
                for class_name, detections in results["detections_by_class"].items():
                    if class_name not in batch_summary["class_counts"]:
                        batch_summary["class_counts"][class_name] = 0
                    batch_summary["class_counts"][class_name] += len(detections)
                    
                batch_summary["image_results"].append(image_result)
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                batch_summary["image_results"].append({
                    "image": img_file.name,
                    "error": str(e)
                })
        
        # Save detailed summary
        summary_path = output_path / "batch_inference_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        # Print summary
        print(f"\nBatch inference complete!")
        print(f"Processed {len(image_files)} images")
        print(f"Total detections: {batch_summary['total_detections']}")
        print(f"Classes found: {list(batch_summary['class_counts'].keys())}")
        for class_name, count in batch_summary['class_counts'].items():
            print(f"   {class_name}: {count} detections")
        print(f"Results saved to: {output_folder}")
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Detectron2 model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--input", required=True, help="Input image or folder path")
    parser.add_argument("--output", help="Output path for results")
    parser.add_argument("--config", help="Path to config.yaml (auto-detected if not provided)")
    parser.add_argument("--annotations", help="Path to instances_train.json file with class names")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--batch", action="store_true", help="Process folder of images")
    parser.add_argument("--detailed", action="store_true", help="Show detailed detection info")
    parser.add_argument("--no-class-names", action="store_true", help="Don't show class names in visualization")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = Detectron2Predictor(
        model_path=args.model,
        config_path=args.config,
        confidence_threshold=args.confidence,
        annotations_path=args.annotations
    )
    
    if args.batch:
        # Batch processing
        if not args.output:
            args.output = os.path.join(os.path.dirname(args.input), "inference_results")
        
        predictor.batch_predict(args.input, args.output)
    
    else:
        # Single image processing
        if not args.output:
            input_name = Path(args.input).stem
            args.output = f"{input_name}_result.jpg"
        
        print(f"Processing: {args.input}")
        
        # Run inference
        outputs, results, img = predictor.predict_image(args.input)
        
        # Create visualization
        vis_img = predictor.visualize_predictions(
            args.input, 
            args.output, 
            show_class_names=not args.no_class_names
        )
        
        # Print results
        predictor.print_detection_summary(results)
        print(f"\nResult saved: {args.output}")


if __name__ == "__main__":
    main()

