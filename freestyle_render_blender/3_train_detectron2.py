#!/usr/bin/env python3
"""
Modified Detectron2 training script for IKEA mask detection dataset.
Trains a Mask R-CNN model on your synthetic dataset with train/val split.
"""

import os
import json
import cv2
import random
import argparse
from pathlib import Path

import torch
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo
import detectron2.utils.comm as comm


class CustomTrainer(DefaultTrainer):
    """Custom trainer with evaluation during training."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def register_dataset(dataset_path: str, dataset_name: str = "custom_dataset"):
    """Register the COCO dataset with Detectron2 - adapted for your structure."""
    
    # Use your exact file structure
    train_annotations_path = os.path.join(dataset_path, "annotations", "instances_train2025.json")
    val_annotations_path = os.path.join(dataset_path, "annotations", "instances_val2025.json")
    images_path = os.path.join(dataset_path, "images")
    
    # Verify paths exist
    if not os.path.exists(train_annotations_path):
        raise FileNotFoundError(f"Training annotations not found: {train_annotations_path}")
    if not os.path.exists(val_annotations_path):
        raise FileNotFoundError(f"Validation annotations not found: {val_annotations_path}")
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    # Register both training and validation datasets
    train_dataset_name = f"{dataset_name}_train"
    val_dataset_name = f"{dataset_name}_val"
    
    register_coco_instances(train_dataset_name, {}, train_annotations_path, images_path)
    register_coco_instances(val_dataset_name, {}, val_annotations_path, images_path)
    
    # Get dataset info
    with open(train_annotations_path, 'r') as f:
        train_data = json.load(f)
    with open(val_annotations_path, 'r') as f:
        val_data = json.load(f)
    
    num_classes = len(train_data.get("categories", []))
    num_train_images = len(train_data.get("images", []))
    num_val_images = len(val_data.get("images", []))
    num_train_annotations = len(train_data.get("annotations", []))
    num_val_annotations = len(val_data.get("annotations", []))
    
    print(f"Registered datasets:")
    print(f"Training: {train_dataset_name} - {num_train_images} images, {num_train_annotations} annotations")
    print(f"Validation: {val_dataset_name} - {num_val_images} images, {num_val_annotations} annotations")
    print(f"Number of classes: {num_classes}")
    
    # Print categories
    categories = train_data.get("categories", [])
    print(f"Object categories:")
    for cat in categories:
        print(f"      - {cat['name']} (ID: {cat['id']})")
    
    return train_dataset_name, val_dataset_name, num_classes


def setup_config(train_dataset_name: str, val_dataset_name: str, num_classes: int, 
                output_dir: str, learning_rate: float = 0.00025, batch_size: int = 2, 
                max_iter: int = 3000, model_type: str = "mask_rcnn"):
    """Setup Detectron2 configuration - optimized for your use case."""
    cfg = get_cfg()
    
    # Choose model based on type - added more options for better accuracy
    if model_type == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif model_type == "mask_rcnn_r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    elif model_type == "mask_rcnn_x101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Dataset configuration - now includes validation set
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)  # Added validation set
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Training configuration - improved for better results
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))  # Learning rate decay
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    cfg.SOLVER.WARMUP_ITERS = 300  # Warmup iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save every 1000 iterations
    
    # Model configuration - optimized for accuracy
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increased for better training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3    # NMS threshold
    
    # Enable mask prediction
    cfg.MODEL.MASK_ON = True
    
    # Input configuration for your 640x480 images
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 800
    
    # Output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Evaluation configuration
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate every 1000 iterations
    
    return cfg


def visualize_dataset(dataset_name: str, num_samples: int = 3, output_dir: str = None):
    """Visualize some samples from the dataset."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"Visualizing {num_samples} samples from {dataset_name}...")
    
    for i, d in enumerate(random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))):
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Could not load image: {d['file_name']}")
            continue
            
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = visualizer.draw_dataset_dict(d)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"sample_{i}_{os.path.basename(d['file_name'])}")
            cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
            print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Detectron2 mask model on IKEA synthetic dataset")
    parser.add_argument("--dataset-path", 
                    default=str(Path.cwd()),  # auto-uses current directory
                    help="Path to dataset folder (default: current working directory)")
    parser.add_argument("--output-dir", 
                    default=str(Path.cwd() / "trained_model"), 
                    help="Output directory for trained model (default: ./trained_model)")
    parser.add_argument("--dataset-name", default="custom_dataset", help="Dataset name")
    parser.add_argument("--model-type", default="mask_rcnn_r101", 
                       choices=["mask_rcnn", "mask_rcnn_r101", "mask_rcnn_x101"], 
                       help="Model architecture (r101 recommended for accuracy)")
    parser.add_argument("--learning-rate", type=float, default=0.0001, 
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Batch size")
    parser.add_argument("--max-iter", type=int, default=5000, 
                       help="Maximum training iterations")
    parser.add_argument("--visualize", action="store_true", 
                       help="Visualize dataset samples")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MASK_RCNN Detectron2 TRAINING")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max iterations: {args.max_iter}")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available - training will be very slow!")
    
    # Register dataset
    try:
        train_dataset_name, val_dataset_name, num_classes = register_dataset(
            args.dataset_path, args.dataset_name
        )
    except FileNotFoundError as e:
        print(f"Dataset registration failed: {e}")
        return
    
    # Visualize dataset if requested
    if args.visualize:
        vis_output_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_output_dir, exist_ok=True)
        visualize_dataset(train_dataset_name, num_samples=3, output_dir=vis_output_dir)
        visualize_dataset(val_dataset_name, num_samples=2, output_dir=vis_output_dir)
    
    # Setup configuration
    print("\nSetting up training configuration...")
    cfg = setup_config(
        train_dataset_name=train_dataset_name,
        val_dataset_name=val_dataset_name,
        num_classes=num_classes,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        model_type=args.model_type
    )
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        f.write(cfg.dump())
    print(f"Saved training config to: {config_path}")
    
    # Setup trainer
    print("\n🏋Initializing trainer...")
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Model and logs saved to: {args.output_dir}")
    print(f"Final trained model: {os.path.join(args.output_dir, 'model_final.pth')}")
    print(f"Training metrics: {os.path.join(args.output_dir, 'metrics.json')}")
    print(f"Training config: {config_path}")
    
    # Print instructions for using the model
    print("\n" + "="*60)
    print("HOW TO USE YOUR TRAINED MODEL:")
    print("="*60)
    print("1. Load the model:")
    print(f"   from detectron2.config import get_cfg")
    print(f"   from detectron2.engine import DefaultPredictor")
    print(f"   ")
    print(f"   cfg = get_cfg()")
    print(f"   cfg.merge_from_file('{config_path}')")
    print(f"   cfg.MODEL.WEIGHTS = '{os.path.join(args.output_dir, 'model_final.pth')}'")
    print(f"   predictor = DefaultPredictor(cfg)")
    print(f"   ")
    print("2. Run inference on new images:")
    print(f"   import cv2")
    print(f"   img = cv2.imread('path/to/your/image.jpg')")
    print(f"   outputs = predictor(img)")
    print(f"   ")
    print("3. Extract detected masks and bounding boxes:")
    print(f"   instances = outputs['instances']")
    print(f"   masks = instances.pred_masks")
    print(f"   boxes = instances.pred_boxes")
    print(f"   classes = instances.pred_classes")
    print("="*60)


if __name__ == "__main__":
    main()


#python3 3_train.py --dataset-path ./rendered_line_drawing --model-type mask_rcnn_r101 --batch-size 2 --max-iter 5000 --visualize

