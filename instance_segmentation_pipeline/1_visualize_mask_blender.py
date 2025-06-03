#!/usr/bin/env python3
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import os

def quick_visualize(data_dir="~/Downloads/FBM_Assembly3_raw2", image_id=0):
    """Quick visualization of a COCO image with original, annotated, and mask views"""
    
    # Expand path
    data_dir = os.path.expanduser(data_dir)
    json_path = os.path.join(data_dir, "coco_annotations.json")
    
    # Load COCO data
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Find the image
    image_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            image_info = img
            break
    
    if image_info is None:
        print(f"Image ID {image_id} not found!")
        available_ids = [img['id'] for img in coco_data['images']]
        print(f"Available IDs: {available_ids}")
        return
    
    # Load image and mask
    image_path = os.path.join(data_dir, image_info['file_name'])
    mask_path = os.path.join(data_dir, image_info['file_name'].replace('.png', '_mask.png'))
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = None
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Get annotations
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Create visualization with 3 panels (or 2 if no mask)
    num_panels = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(20, 8))
    
    # Ensure axes is always a list
    if num_panels == 1:
        axes = [axes]
    elif num_panels == 2 and not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    # Panel 1: Original image (no annotations)
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image: {image_info['file_name']}")
    axes[0].axis('off')
    
    # Panel 2: Annotated image with bounding boxes and segmentations
    axes[1].imshow(image)
    axes[1].set_title(f"Annotated Image ({len(annotations)} objects)")
    axes[1].axis('off')
    
    # Colors for different objects
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        
        # Draw bounding box
        x, y, width, height = ann['bbox']
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        axes[1].add_patch(rect)
        
        # Add label
        category_name = "unknown"
        for cat in coco_data['categories']:
            if cat['id'] == ann['category_id']:
                category_name = cat['name']
                break
        
        axes[1].text(x, y-5, f"{category_name} #{ann['id']}", 
                    fontsize=10, color=color, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Draw segmentation if available
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) >= 6:  # Need at least 3 points (6 coordinates)
                    points = np.array(seg).reshape(-1, 2)
                    polygon = Polygon(points, closed=True, fill=True, 
                                    facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
                    axes[1].add_patch(polygon)
    
    # Panel 3: Mask if available
    if mask is not None:
        axes[2].imshow(mask)
        axes[2].set_title(f"Segmentation Mask")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print details
    print(f"\nImage Details:")
    print(f"File: {image_info['file_name']}")
    print(f"Size: {image_info['width']}x{image_info['height']}")
    print(f"Objects: {len(annotations)}")
    
    for ann in annotations:
        category_name = "unknown"
        for cat in coco_data['categories']:
            if cat['id'] == ann['category_id']:
                category_name = cat['name']
                break
        
        print(f"  - {category_name} #{ann['id']}: bbox={ann['bbox']}, area={ann['area']}")

# Quick usage examples:
if __name__ == "__main__":
    # Visualize image with ID 0
    print("Visualizing image ID 0:")
    quick_visualize(image_id=999)
    
    # You can also call with different parameters:
    # quick_visualize(data_dir="~/Downloads/newnew_test", image_id=1)
    # quick_visualize(image_id=2)
    # quick_visualize(image_id=3)