#!/usr/bin/env python3
"""
Detectron2 training script for mask models using synthetic dataset.
Trains a Mask R-CNN model on your allen_key dataset.
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


def register_dataset(dataset_path: str, dataset_name: str = "allen_key"):
    """Register the COCO dataset with Detectron2."""
    annotations_path = os.path.join(dataset_path, "annotations", "instances_train.json")
    images_path = os.path.join(dataset_path, "images")
    
    # Verify paths exist
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    # Register the dataset
    register_coco_instances(f"{dataset_name}_train", {}, annotations_path, images_path)
    
    # Get dataset info
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    num_classes = len(coco_data.get("categories", []))
    num_images = len(coco_data.get("images", []))
    
    print(f"✅ Registered dataset: {dataset_name}_train")
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Number of images: {num_images}")
    
    return f"{dataset_name}_train", num_classes


def setup_config(dataset_name: str, num_classes: int, output_dir: str, 
                learning_rate: float = 0.00025, batch_size: int = 2, 
                max_iter: int = 1000, model_type: str = "mask_rcnn"):
    """Setup Detectron2 configuration."""
    cfg = get_cfg()
    
    # Choose model based on type
    if model_type == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif model_type == "mask_rcnn_x101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()  # No test set for now
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Training configuration
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # No learning rate decay
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save every 500 iterations
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Evaluation configuration
    cfg.TEST.EVAL_PERIOD = 500  # Evaluate every 500 iterations
    
    return cfg


def visualize_dataset(dataset_name: str, num_samples: int = 3, output_dir: str = None):
    """Visualize some samples from the dataset."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"🖼️  Visualizing {num_samples} samples from dataset...")
    
    for i, d in enumerate(random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"sample_{i}.jpg")
            cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
            print(f"💾 Saved visualization: {output_path}")
        else:
            # Display using matplotlib if available
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                plt.imshow(vis.get_image())
                plt.axis('off')
                plt.title(f"Sample {i+1}: {os.path.basename(d['file_name'])}")
                plt.show()
            except ImportError:
                print("matplotlib not available for display. Consider providing output_dir.")


def main():
    parser = argparse.ArgumentParser(description="Train Detectron2 mask model on synthetic dataset")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset folder")
    parser.add_argument("--output-dir", default="./output", help="Output directory for model and logs")
    parser.add_argument("--dataset-name", default="dataset", help="Dataset name")
    parser.add_argument("--model-type", default="mask_rcnn", 
                       choices=["mask_rcnn", "mask_rcnn_x101"], 
                       help="Model architecture to use")
    parser.add_argument("--learning-rate", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--visualize", action="store_true", help="Visualize dataset samples")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    print("Starting Detectron2 training...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max iterations: {args.max_iter}")
    
    # Register dataset
    dataset_name, num_classes = register_dataset(args.dataset_path, args.dataset_name)
    
    # Visualize dataset if requested
    if args.visualize:
        vis_output_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_output_dir, exist_ok=True)
        visualize_dataset(dataset_name, num_samples=3, output_dir=vis_output_dir)
    
    # Setup configuration
    cfg = setup_config(
        dataset_name=dataset_name,
        num_classes=num_classes,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        model_type=args.model_type
    )
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(cfg.dump())
    print(f"Saved config to: {config_path}")
    
    # Setup trainer
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    print(f"Model and logs saved to: {args.output_dir}")
    print(f"Final model: {os.path.join(args.output_dir, 'model_final.pth')}")


if __name__ == "__main__":
    main()