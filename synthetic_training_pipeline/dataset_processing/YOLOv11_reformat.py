#!/usr/bin/env python3
import os
import random
import shutil
import cv2
import numpy as np
import ast
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# CONFIGURE YOUR SPLIT RATIOS
# ──────────────────────────────────────────────────────────────────────
VAL_FRACTION  = 0.20   # e.g. 20% validation
TEST_FRACTION = 0.10   # e.g. 10% test
assert VAL_FRACTION + TEST_FRACTION < 1.0, "Splits must sum to <1"

# ──────────────────────────────────────────────────────────────────────
# PATHS & RAW CLASS LOADING
# ──────────────────────────────────────────────────────────────────────
random.seed(42)
src_dir     = Path("/home/aaugus11/Downloads/FBM_Assembly3/rendered_obb").expanduser()
yolo_root   = src_dir / "YOLOv11_Dataset"
classes_txt = src_dir / "classes.txt"

# Read full list of class names (including duplicates)
full_names = []
with open(classes_txt, "r") as cf:
    for line in cf:
        entry = line.strip()
        if not entry or entry == "__background__":
            continue
        _, name = entry.split("_", 1)
        full_names.append(name)

# Derive the unique “base” class names, in order of first appearance
base_names = []
for name in full_names:
    base = name.split(".", 1)[0]
    if base not in base_names:
        base_names.append(base)

# Build mapping from full duplicate name → base class ID
class_map = {
    full: base_names.index(full.split(".", 1)[0])
    for full in full_names
}

# ──────────────────────────────────────────────────────────────────────
# MAKE TRAIN/VAL/TEST FOLDERS
# ──────────────────────────────────────────────────────────────────────
for split in ["train", "val", "test"]:
    (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)
    (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# COLLECT SOURCE FILES
# ──────────────────────────────────────────────────────────────────────
images = sorted(src_dir.glob("*_rgb0001.png"))
labels = sorted(src_dir.glob("*_obb0001.txt"))
assert len(images) == len(labels), "Image/label count mismatch"

# ──────────────────────────────────────────────────────────────────────
# SPLIT INDICES
# ──────────────────────────────────────────────────────────────────────
N      = len(images)
n_val  = int(N * VAL_FRACTION)
n_test = int(N * TEST_FRACTION)

all_idx   = list(range(N))
val_idx   = random.sample(all_idx, n_val)
remaining = list(set(all_idx) - set(val_idx))
test_idx  = random.sample(remaining, n_test)
train_idx = list(set(remaining) - set(test_idx))

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ──────────────────────────────────────────────────────────────────────
# PARSING ORIGINAL OBB LINES
# ──────────────────────────────────────────────────────────────────────
def parse_obb_line(line):
    raw_name, raw_pts = line.strip().split(":", 1)
    name = raw_name.lower().replace(" ", "_")
    pts = ast.literal_eval(raw_pts.strip())
    # axis-aligned?
    if (
        isinstance(pts, (list, tuple))
        and len(pts) == 4
        and all(isinstance(v, (int, float)) for v in pts)
    ):
        x0, y0, x1, y1 = pts
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        corners = [(float(p[0]), float(p[1])) for p in pts]
    return name, corners

# ──────────────────────────────────────────────────────────────────────
# CONVERT ONE LABEL FILE TO YOLOv11 FORMAT
# ──────────────────────────────────────────────────────────────────────
def convert_obb_to_yolo11(txt_path, img_w, img_h):
    lines = open(txt_path).readlines()
    out_lines = []
    for line in lines:
        cls_name, corners = parse_obb_line(line)
        # map duplicate → base
        if cls_name not in class_map:
            continue
        cls_id = class_map[cls_name]
        # normalize
        norm = [(x / img_w, y / img_h) for x, y in corners]
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        out_lines.append(f"{cls_id} {flat}")
    return "\n".join(out_lines)

# ──────────────────────────────────────────────────────────────────────
# COPY & CONVERT FUNCTION
# ──────────────────────────────────────────────────────────────────────
def process(idx_list, split):
    for i in idx_list:
        img_src = images[i]
        lbl_src = labels[i]
        base    = img_src.stem.split("_")[0]

        img_dst = yolo_root / split / "images" / f"{base}.png"
        lbl_dst = yolo_root / split / "labels" / f"{base}.txt"

        shutil.copy2(img_src, img_dst)
        im = cv2.imread(str(img_src))
        h, w = im.shape[:2]

        yolo_txt = convert_obb_to_yolo11(lbl_src, w, h)
        with open(lbl_dst, "w") as f:
            if yolo_txt:
                f.write(yolo_txt + "\n")

# ──────────────────────────────────────────────────────────────────────
# RUN PROCESSING
# ──────────────────────────────────────────────────────────────────────
process(train_idx, "train")
process(val_idx,   "val")
process(test_idx,  "test")

# ──────────────────────────────────────────────────────────────────────
# WRITE dataset.yaml WITH ONLY BASE CLASSES
# ──────────────────────────────────────────────────────────────────────
with open(yolo_root / "dataset.yaml", "w") as f:
    f.write(f"train: {yolo_root/'train/images'}\n")
    f.write(f"val:   {yolo_root/'val/images'}\n")
    f.write(f"test:  {yolo_root/'test/images'}\n")
    f.write(f"nc:    {len(base_names)}\n")
    names_list = ", ".join(f"'{n}'" for n in base_names)
    f.write(f"names: [{names_list}]\n")

print("YOLOv11 Dataset prepared with de-duplicated classes and splits.")
