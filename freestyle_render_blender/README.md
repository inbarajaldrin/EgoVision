---

# Freestyle Line Drawing Instance Segmentation

This module enables instance segmentation training using synthetic line drawings rendered from Blender. It is part of the **EgoVision** project and supports the full pipeline from CAD `.obj` files to training a Detectron2 model using rendered contours.

---

## Pipeline Steps

### 1. Convert OBJ Files to PLY

Place your `.obj` files in the following directory (availble at `assets/fmb/`):

```
EgoVision/freestyle_render_blender/
```

Then run the following script to convert and scale them:

```bash
cd freestyle_render_blender/
python3 1_convert_scale_center_obj_ply.py
```

This will generate `.ply` versions of the same objects with appropriate scaling and centering.

---

### 2. Render Line Drawings in Blender

Run this code `2_render_line_drawings_blender.py` inside Blender
Ensure camera pose `cam_poses_level2.npy` path is set in the blender code

This script:

* Captures line drawings from those views
* Saves them under `rendered_line_drawing/` with matching annotations

---

### 3. Train a Mask R-CNN on Rendered Line Drawings

To train a Detectron2-based model on the rendered dataset:

```bash
python3 3_train_detectron2.py \
    --dataset-path ./rendered_line_drawing \
    --model-type mask_rcnn_r101 \
    --batch-size 2 \
    --max-iter 5000 \
    --visualize
```

This will:

* Train on the rendered dataset in COCO format
* Output weights and logs under `trained_model/`

---

### 4. Run Inference on Test Images

To test your trained model on unseen drawings or scans:

```bash
python3 4_infer_detectron2.py --save-individual-masks
```

This will:

* Load the model from `trained_model/model_final.pth`
* Run inference on all images in `test_images/`
* Save results with mask overlays and optional individual masks in `inference_results/`

---

## Folder Summary

```
freestyle_render_blender/
├── 1_convert_scale_center_obj_ply.py      # Convert OBJ to centered, scaled PLY
├── 2_render_line_drawings_blender.py      # Render synthetic line drawings
├── 3_train_detectron2.py                  # Train Mask R-CNN on line drawings
├── 4_infer_detectron2.py                  # Run inference on test drawings
assets/
└── fmb/
    ├── base.obj / .ply
    ├── fork.obj / .ply
    ├── line.obj / .ply
    └── cam_poses_level2.npy
```
