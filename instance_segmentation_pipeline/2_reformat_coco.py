#!/usr/bin/env python3
"""
Script to reformat Blender-generated synthetic data for Detectron2 training.
Converts data structure from:
  input_folder/
  ├── coco_annotations.json
  ├── scene_0000.png
  ├── scene_0000_mask.png
  └── ...

To Detectron2 format:
  output_folder/
  ├── annotations/
  │   └── instances_train.json
  ├── images/
  │   ├── scene_0000.png
  │   └── scene_0001.png
  └── masks/
      ├── scene_0000_mask.png
      └── scene_0001_mask.png
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any


def validate_input_folder(input_folder: Path) -> bool:
    """Validate that the input folder has the expected structure."""
    if not input_folder.exists():
        print(f"Error: Input folder {input_folder} does not exist")
        return False
    
    # Check for coco_annotations.json
    coco_file = input_folder / "coco_annotations.json"
    if not coco_file.exists():
        print(f"Error: coco_annotations.json not found in {input_folder}")
        return False
    
    # Check for at least one image-mask pair
    image_files = list(input_folder.glob("scene_*.png"))
    mask_files = list(input_folder.glob("scene_*_mask.png"))
    
    # Filter out mask files from image files
    image_files = [f for f in image_files if not f.name.endswith("_mask.png")]
    
    if not image_files:
        print(f"Error: No scene images found in {input_folder}")
        return False
    
    if not mask_files:
        print(f"Error: No mask images found in {input_folder}")
        return False
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    return True


def create_output_structure(output_folder: Path) -> None:
    """Create the Detectron2 dataset folder structure."""
    output_folder.mkdir(exist_ok=True)
    (output_folder / "annotations").mkdir(exist_ok=True)
    (output_folder / "images").mkdir(exist_ok=True)
    (output_folder / "masks").mkdir(exist_ok=True)


def copy_images_and_masks(input_folder: Path, output_folder: Path) -> tuple[List[str], List[str]]:
    """Copy images and masks to their respective folders."""
    images_folder = output_folder / "images"
    masks_folder = output_folder / "masks"
    
    # Get all image files (excluding masks)
    image_files = sorted([f for f in input_folder.glob("scene_*.png") 
                         if not f.name.endswith("_mask.png")])
    mask_files = sorted(input_folder.glob("scene_*_mask.png"))
    
    copied_images = []
    copied_masks = []
    
    # Copy images
    for img_file in image_files:
        dest_path = images_folder / img_file.name
        shutil.copy2(img_file, dest_path)
        copied_images.append(img_file.name)
        print(f"Copied image: {img_file.name}")
    
    # Copy masks
    for mask_file in mask_files:
        dest_path = masks_folder / mask_file.name
        shutil.copy2(mask_file, dest_path)
        copied_masks.append(mask_file.name)
        print(f"Copied mask: {mask_file.name}")
    
    return copied_images, copied_masks


def process_coco_annotations(input_folder: Path, output_folder: Path, 
                           copied_images: List[str]) -> None:
    """Process and copy COCO annotations, updating image paths."""
    input_coco = input_folder / "coco_annotations.json"
    output_coco = output_folder / "annotations" / "instances_train.json"
    
    with open(input_coco, 'r') as f:
        coco_data = json.load(f)
    
    # Update image file names to match the new structure
    if "images" in coco_data:
        for img_info in coco_data["images"]:
            if "file_name" in img_info:
                # Ensure the file_name matches what we actually copied
                original_name = Path(img_info["file_name"]).name
                if original_name in copied_images:
                    img_info["file_name"] = original_name
                else:
                    print(f"Warning: Image {original_name} in annotations but not found in copied images")
    
    # Save the updated annotations
    with open(output_coco, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Processed annotations saved to: {output_coco}")

def main():
    parser = argparse.ArgumentParser(
        description="Reformat Blender synthetic data for Detectron2 training"
    )
    parser.add_argument(
        "input_folder", 
        type=str, 
        help="Path to input folder containing Blender-generated data"
    )
    parser.add_argument(
        "output_folder", 
        type=str, 
        help="Path to output folder for Detectron2-formatted dataset"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite output folder if it exists"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    # Validate input
    if not validate_input_folder(input_folder):
        return 1
    
    # Check if output folder exists
    if output_folder.exists() and not args.overwrite:
        print(f"Error: Output folder {output_folder} already exists. Use --overwrite to overwrite.")
        return 1
    
    # Create output structure
    print(f"Creating Detectron2 dataset structure in: {output_folder}")
    create_output_structure(output_folder)
    
    # Copy images and masks
    print("Copying images and masks...")
    copied_images, copied_masks = copy_images_and_masks(input_folder, output_folder)
    
    # Process annotations
    print("Processing COCO annotations...")
    process_coco_annotations(input_folder, output_folder, copied_images)
    
    print(f"\nDataset reformatting complete!")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Images: {len(copied_images)}")
    print(f"Masks: {len(copied_masks)}")
    print(f"\nNext steps:")
    print(f"1. Use the dataset in Detectron2 with:")
    print(f"   register_coco_instances('my_dataset', {{}}, ")
    print(f"                          '{output_folder}/annotations/instances_train.json',")
    print(f"                          '{output_folder}/images')")


if __name__ == "__main__":
    exit(main())