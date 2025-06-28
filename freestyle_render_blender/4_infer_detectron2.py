#!/usr/bin/env python3
"""
Detectron2 inference script for trained IKEA mask detection model.
Runs inference on test images and saves results with visualizations.
"""

import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
import time

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
import detectron2.data.transforms as T


def load_trained_model(model_path: str, config_path: str = None, model_type: str = "mask_rcnn_r101", 
                      num_classes: int = None, score_threshold: float = 0.5):
    """Load the trained Detectron2 model for inference."""
    
    cfg = get_cfg()
    
    # Load model configuration
    if model_type == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    elif model_type == "mask_rcnn_r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    elif model_type == "mask_rcnn_x101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    
    # Load custom config if available
    if config_path and os.path.exists(config_path):
        cfg.merge_from_file(config_path)
        print(f"Loaded training config from: {config_path}")
    
    # Set model weights
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.MASK_ON = True
    
    # Set number of classes if provided
    if num_classes:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    print(f"Loaded trained model from: {model_path}")
    print(f"Detection threshold: {score_threshold}")
    print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    
    return predictor, cfg


def get_class_names_from_dataset(dataset_path: str):
    """Extract class names from the original COCO dataset."""
    annotations_path = os.path.join(dataset_path, "annotations", "instances_train2025.json")
    
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        categories = coco_data.get("categories", [])
        class_names = [cat["name"] for cat in sorted(categories, key=lambda x: x["id"])]
        
        print(f"Found {len(class_names)} object classes:")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
        
        return class_names
    else:
        print(f"Could not find annotations at: {annotations_path}")
        return None


def run_inference_on_image(predictor, image_path: str, class_names: list = None):
    """Run inference on a single image."""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Run inference
    start_time = time.time()
    outputs = predictor(img)
    inference_time = time.time() - start_time
    
    # Extract predictions
    instances = outputs["instances"].to("cpu")
    
    results = {
        "image_path": image_path,
        "image_shape": img.shape,
        "inference_time": inference_time,
        "num_detections": len(instances),
        "detections": []
    }
    
    # Process each detection
    for i in range(len(instances)):
        detection = {
            "class_id": instances.pred_classes[i].item(),
            "class_name": class_names[instances.pred_classes[i].item()] if class_names else f"class_{instances.pred_classes[i].item()}",
            "score": instances.scores[i].item(),
            "bbox": instances.pred_boxes[i].tensor.numpy().tolist()[0],  # [x1, y1, x2, y2]
            "mask_area": instances.pred_masks[i].sum().item()
        }
        results["detections"].append(detection)
    
    return results, outputs


def visualize_results(img, outputs, class_names: list = None, save_path: str = None):
    """Create visualization of detection results."""
    
    # Create proper metadata for visualization
    if class_names:
        # Create a proper metadata object with required methods
        class CustomMetadata:
            def __init__(self, thing_classes):
                self.thing_classes = thing_classes
                self.thing_colors = None
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        metadata = CustomMetadata(class_names)
    else:
        metadata = None
    
    # Create visualizer
    visualizer = Visualizer(
        img[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW  # Makes background B&W, objects colored
    )
    
    # Draw predictions
    vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_image = vis.get_image()[:, :, ::-1]  # RGB to BGR for OpenCV
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        print(f"Saved visualization: {save_path}")
    
    return vis_image


def save_masks_separately(outputs, image_name: str, output_dir: str):
    """Save individual masks as separate images."""
    instances = outputs["instances"].to("cpu")
    masks_dir = os.path.join(output_dir, "individual_masks", image_name.replace('.jpg', '').replace('.png', ''))
    os.makedirs(masks_dir, exist_ok=True)
    
    saved_masks = []
    
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255
        mask_path = os.path.join(masks_dir, f"mask_{i}_class_{instances.pred_classes[i].item()}.png")
        cv2.imwrite(mask_path, mask)
        saved_masks.append(mask_path)
    
    return saved_masks


def process_image_folder(predictor, images_folder: str, output_dir: str, 
                        class_names: list = None, save_visualizations: bool = True,
                        save_individual_masks: bool = False):
    """Process all images in a folder."""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f"*{ext}"))
        image_files.extend(Path(images_folder).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No image files found in: {images_folder}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    all_results = []
    total_detections = 0
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Run inference
        results, outputs = run_inference_on_image(predictor, str(image_path), class_names)
        
        if results is None:
            continue
        
        all_results.append(results)
        total_detections += results["num_detections"]
        
        # Print results for this image
        if results["num_detections"] > 0:
            print(f"   Found {results['num_detections']} objects (inference: {results['inference_time']:.3f}s)")
            for det in results["detections"]:
                print(f"      - {det['class_name']}: {det['score']:.3f}")
        else:
            print(f"   No objects detected (inference: {results['inference_time']:.3f}s)")
        
        # Save visualization
        if save_visualizations and results["num_detections"] > 0:
            img = cv2.imread(str(image_path))
            vis_path = os.path.join(vis_dir, f"detected_{image_path.name}")
            visualize_results(img, outputs, class_names, vis_path)
        
        # Save individual masks
        if save_individual_masks and results["num_detections"] > 0:
            save_masks_separately(outputs, image_path.name, output_dir)
    
    # Save summary results
    summary = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "average_detections_per_image": total_detections / len(image_files) if image_files else 0,
        "results": all_results
    }
    
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Processed: {len(image_files)} images")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {summary['average_detections_per_image']:.2f}")
    print(f"Results saved to: {results_path}")
    if save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Detectron2 model")
    parser.add_argument("--model-path", 
        default="./trained_model/model_final.pth",
        help="Path to trained model file")
    parser.add_argument("--config-path", 
        default="./trained_model/training_config.yaml",
        help="Path to training config file")
    parser.add_argument("--images-folder", 
        default="./test_images",
        help="Path to folder containing test images")

    parser.add_argument("--output-dir", 
        default="./inference_results",
        help="Output directory for results")

    parser.add_argument("--dataset-path",
        default="./rendered_line_drawing",
        help="Path to original dataset (to get class names)")
    parser.add_argument("--model-type", default="mask_rcnn_r101",
                       choices=["mask_rcnn", "mask_rcnn_r101", "mask_rcnn_x101"],
                       help="Model architecture used for training")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--save-individual-masks", action="store_true",
                       help="Save individual object masks as separate images")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip saving visualization images")
    
    args = parser.parse_args()
    
    print("="*60)
    print("DETECTRON2 INFERENCE")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Images: {args.images_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.score_threshold}")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Make sure you've completed training first!")
        return
    
    # Check if images folder exists
    if not os.path.exists(args.images_folder):
        print(f"Images folder not found: {args.images_folder}")
        return
    
    # Get class names from dataset
    class_names = get_class_names_from_dataset(args.dataset_path)
    num_classes = len(class_names) if class_names else None
    
    # Load trained model
    try:
        predictor, cfg = load_trained_model(
            args.model_path, 
            args.config_path, 
            args.model_type,
            num_classes,
            args.score_threshold
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Run inference on all images
    process_image_folder(
        predictor=predictor,
        images_folder=args.images_folder,
        output_dir=args.output_dir,
        class_names=class_names,
        save_visualizations=not args.no_visualizations,
        save_individual_masks=args.save_individual_masks
    )
    
    print(f"\nInference completed! Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()
    
#python3 4_infer_detectron2.py --save-individual-masks
