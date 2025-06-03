# EgoVision: Egocentric RGB-D Mesh Reconstruction & Synthetic Data-Driven Object Detection for Self-Supervised Robotic Manipulation

**EgoVision** is a self-supervised, simulation-driven pipeline for egocentric object detection in robotic workspaces. Starting from real-world RGB-D scans of the workspace and individual objects, the system reconstructs textured 3D mesh models, which are then imported into Blender to generate synthetic scenes.

These scenes are rendered with randomized views and corresponding oriented bounding box (OBB) or mask annotations, producing a fully annotated dataset without manual labeling. This dataset is used to train either a YOLO OBB detector or a Mask R-CNN segmenter, which can then be deployed on the same egocentric camera for real-time object detection in the original environment.

In addition, a work‑in‑progress module aligns 2D schematic instructions (e.g., IKEA manuals) with 3D scene data to enable instruction‑conditioned manipulation.

---

## Three Main Pipelines

### 1. RGBD Mesh Pipeline (`rgbd_mesh_pipeline/`)

Converts Polycam `.glb` scans into clean, aligned `.ply` and `.obj` meshes.

1. **GLB→PLY Export**: Blender script (`blender_scripts/0_conv_glb_ply.txt`) batches Polycam `.glb` → colored `.ply`.
2. **Floor Removal**: `1_mesh_plane_removal.py` / `1_mesh_plane_removal2.py` remove planar ground.
3. **Centering**: `2_centre_ply.py` recenters mesh at origin.
4. **Alignment & Fusion**: `3_align_merge_mesh.py` (TEASER++ + FPFH + ICP) merges multi‑view scans. Optional manual tweak with `3_manual_rotate.py`.
5. **PLY→OBJ Export**: `4_conv_ply_obj.py` bakes vertex colors → `.obj` + `.mtl` for Blender.

---

### 2. Synthetic Training Pipeline (`synthetic_training_pipeline/`)

Uses the fused meshes to generate a self‑supervised detection model based on oriented bounding boxes.

1. **Blend & Render**: Blender scripts produce RGB + OBB annotation pairs.
2. **Dataset Formatting**: `dataset_processing/YOLOv11_reformat.py` → YOLOv11n folder structure.
3. **Training**: `train_yolo11n/train_yolo11n.py` trains oriented bounding‑box detector.
4. **Inference**: `inference/infer_and_draw.py` runs model on test images and visualizes results.
5. **Evaluation**: `evaluation/baseline.py` computes IoU, centroid & angle error; summary in `baseline.pdf`.
6. **Feature Tracking & Annotation (WIP)**: `feature_tracking_annotation/` matches sketchified OBJ renders to manual‑crop images for cross‑modal annotation.

---

### 3. Instance Segmentation Pipeline (`instance_segmentation_pipeline/`)

Trains a Mask R-CNN instance segmentation model using synthetic RGB + mask pairs from Blender.

1. **Mask Rendering**: `0_blender_mask_gen_single_obj.py` or `0_blender_mask_gen_multi_obj.py` produce RGB + mask image pairs.
2. **Visualize Masks**: `1_visualize_mask_blender.py` displays overlays for validation.
3. **COCO Formatting**: `2_reformat_coco.py` converts data into Detectron2-compatible structure.
4. **Training**: `3_train_detectron2.py` launches training on COCO dataset.
5. **Inference**: `4_inference_detectron2.py` runs on test images.
6. **Real-Time Deployment**: `5_deploy_intel_camera.py` performs segmentation from RealSense RGB stream.

---

## Prerequisites & Installation

* **Blender** 4.3 for GLB→PLY export & data rendering
* **Python 3.8+** virtual environment
* Required packages:

  ```bash
  pip install open3d numpy scikit-learn ultralytics detectron2 opencv-python pycocotools
  ```
* TEASER++ build instructions: [https://github.com/MIT-SPARK/TEASER-plusplus.git](https://github.com/MIT-SPARK/TEASER-plusplus.git)

---

## Repository Structure

```bash
EgoVision/
├── assets/                           # Raw `.blend` and `.glb` scans
├── rgbd_mesh_pipeline/               # Scan‑to‑mesh processing
│   ├── blender_scripts/
│   ├── 1_mesh_plane_removal.py
│   ├── 2_centre_ply.py
│   ├── 3_align_merge_mesh.py
│   ├── 3_manual_rotate.py
│   └── 4_conv_ply_obj.py

├── synthetic_training_pipeline/      # Synthetic data → YOLO model
│   ├── blender_scripts/
│   ├── dataset_processing/
│   ├── train_yolo11n/
│   ├── inference/
│   ├── evaluation/
│   └── feature_tracking_annotation/

├── instance_segmentation_pipeline/   # Synthetic data → Mask R-CNN
│   ├── 0_blender_mask_gen_single_obj.py
│   ├── 0_blender_mask_gen_multi_obj.py
│   ├── 1_visualize_mask_blender.py
│   ├── 2_reformat_coco.py
│   ├── 3_train_detectron2.py
│   ├── 4_inference_detectron2.py
│   ├── 5_deploy_intel_camera.py
│   ├── assets/
│   ├── COCO_Dataset/
│   └── runs/
```

*EgoGrasp: closing the loop from real-world perception to simulation-based learning to enhance real-time robotic vision.*
