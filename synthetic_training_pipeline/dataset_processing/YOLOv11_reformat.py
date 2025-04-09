import os
import shutil
import random
import math
import cv2
import numpy as np
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Source and destination
src_dir = Path("~/Desktop/Blender_OBB_Dataset").expanduser()
yolo_root = src_dir / "YOLOv11_Dataset"

# Create new structure matching Roboflow format
for split in ["train", "val", "test"]:
    os.makedirs(yolo_root / split / "images", exist_ok=True)
    os.makedirs(yolo_root / split / "labels", exist_ok=True)

# Get image-label pairs
images = sorted([f for f in src_dir.glob("*_rgb0001.png")])
labels = sorted([f for f in src_dir.glob("*_obb0001.txt")])
assert len(images) == len(labels), "Image and label count mismatch!"

# Use all images for training
train_idx = list(range(len(images)))

# Randomly sample 20% for validation and 10% for testing (copying from training set)
val_idx = random.sample(train_idx, int(0.2 * len(images)))
remaining = list(set(train_idx) - set(val_idx))
test_idx = random.sample(remaining, int(0.1 * len(images)))

# Class mapping
class_map = {
    "Jenga_Block": 0,
}

def parse_obb_line(line):
    name, coords_str = line.strip().split(": ")
    coords_str = coords_str.strip("[]")
    point_strs = coords_str.split("), (")
    coords = []
    for p in point_strs:
        p = p.replace("(", "").replace(")", "")
        x, y = map(int, p.split(","))
        coords.append((x, y))
    return name, coords

def normalize_point(x, y, w_img, h_img):
    return x / w_img, y / h_img

def convert_obb_to_yolo11(txt_path, img_w, img_h):
    lines = open(txt_path).readlines()
    yolo_lines = []

    for line in lines:
        class_name, points = parse_obb_line(line)
        if class_name not in class_map:
            continue

        norm_points = [normalize_point(x, y, img_w, img_h) for x, y in points]
        flat_coords = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
        yolo_lines.append(f"{class_map[class_name]} " + " ".join(flat_coords))

    return "\n".join(yolo_lines) + "\n" if yolo_lines else ""

def process(idx_list, split_name):
    for idx in idx_list:
        img_src = images[idx]
        lbl_src = labels[idx]

        base_name = img_src.stem.split("_")[0]  # '000'
        img_dst = yolo_root / split_name / "images" / f"{base_name}.png"
        lbl_dst = yolo_root / split_name / "labels" / f"{base_name}.txt"

        # Copy image
        shutil.copy2(img_src, img_dst)

        # Read image size
        img = cv2.imread(str(img_src))
        h_img, w_img = img.shape[:2]

        # Generate YOLOv11 label
        yolo_label = convert_obb_to_yolo11(lbl_src, w_img, h_img)
        with open(lbl_dst, 'w') as f:
            f.write(yolo_label)

# Process all sets
process(train_idx, "train")
process(val_idx, "val")
process(test_idx, "test")

# Create dataset.yaml
with open(yolo_root / "dataset.yaml", "w") as f:
    f.write(f"""
train: {yolo_root/'train/images'}
val: {yolo_root/'val/images'}
test: {yolo_root/'test/images'}
nc: 1
names: ["Jenga_Block"]
""")

print(" YOLOv11 Dataset prepared in Roboflow-compatible format with train/val/test splits.")
