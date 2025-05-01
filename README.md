# EgoVision: Egocentric RGB-D Mesh Reconstruction & Synthetic Data-Driven Object Detection for Self-Supervised Robotic Manipulation

**EgoVision** is a self-supervised, simulation-driven pipeline for egocentric object detection in robotic workspaces. Starting from real-world RGB-D scans of the workspace and individual objects, the system reconstructs textured 3D mesh models, which are then imported into Blender to generate synthetic scenes.

These scenes are rendered with randomized views and corresponding oriented bounding box (OBB) annotations, producing a fully annotated dataset without manual labeling. This dataset is used to train a YOLO OBB detector, which can then be deployed on the same egocentric camera for real-time object detection in the original environment.

In addition, a work‑in‑progress module aligns 2D schematic instructions (e.g., IKEA manuals) with 3D scene data to enable instruction‑conditioned manipulation.

---

## Two Main Pipelines

### 1. RGBD Mesh Pipeline (`rgbd_mesh_pipeline/`)
Converts Polycam `.glb` scans into clean, aligned `.ply` and `.obj` meshes.

1. **GLB→PLY Export**: Blender script (`blender_scripts/0_conv_glb_ply.txt`) batches Polycam `.glb` → colored `.ply`.
2. **Floor Removal**: `1_mesh_plane_removal.py` / `1_mesh_plane_removal2.py` remove planar ground.
3. **Centering**: `2_centre_ply.py` recenters mesh at origin.
4. **Alignment & Fusion**: `3_align_merge_mesh.py` (TEASER++ + FPFH + ICP) merges multi‑view scans. Optional manual tweak with `3_manual_rotate.py`.
5. **PLY→OBJ Export**: `4_conv_ply_obj.py` bakes vertex colors → `.obj` + `.mtl` for Blender.

### 2. Synthetic Training Pipeline (`synthetic_training_pipeline/`)
Uses the fused meshes to generate a self‑supervised detection model.

1. **Blend & Render**: Blender scripts produce RGB + OBB annotation pairs.
2. **Dataset Formatting**: `dataset_processing/YOLOv11_reformat.py` → YOLOv11n folder structure.
3. **Training**: `train_yolo11n/train_yolo11n.py` trains oriented bounding‑box detector.
4. **Inference**: `inference/infer_and_draw.py` runs model on test images and visualizes results.
5. **Evaluation**: `evaluation/baseline.py` computes IoU, centroid & angle error; summary in `baseline.pdf`.
6. **Feature Tracking & Annotation (WIP)**: `feature_tracking_annotation/` matches sketchified OBJ renders to manual‑crop images for cross‑modal annotation.

---

## Prerequisites & Installation

- **Blender** 4.3 for GLB→PLY export & data rendering
- **Python 3.8+** virtual environment
- Required packages:
  ```bash
  pip install open3d numpy scikit-learn ultralytics
  ```
- TEASER++ build instructions: https://github.com/MIT-SPARK/TEASER-plusplus.git

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
│   └── feature_tracking_annotation/  # (Planned) 2D sketch → 3D manipulation
```


_EgoGrasp: closing the loop from real-world perception to simulation-based learning to enhance real-time robotic vision._

