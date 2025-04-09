# EgoGrasp : A Self-Supervised Perception-to-Action Pipeline for Robotic Manipulation

EgoGrasp is a self-supervised, simulation-driven object detection pipeline designed to enable robotic arms to manipulate previously unseen objects and scenes using only egocentric RGBD input.

This system starts with real-world depth and color data captured from a robot-mounted camera, processes it into CAD-like models, and then leverages synthetic rendering to train a YOLOv11n OBB (Oriented Bounding Box) detector—without any manual labeling. The trained model is then deployed back on the robot for accurate object detection and grasp planning in real scenes.

## Project Overview

This project consists of two interconnected pipelines:

---

### 1. Self-Supervised Object Learning and Synthetic Training Loop

**Goal:** Learn to detect and manipulate unknown objects using only real RGBD scans and synthetic data.

#### Pipeline Stages:

1. **Data Acquisition:**
   - Capture real RGBD point clouds from an egocentric robotic arm.

2. **Point Cloud Processing:**
   - Clean and segment individual objects from the point cloud.

3. **CAD Model Generation:**
   - Convert object point clouds into textured CAD models (with RGB color).

4. **Synthetic Dataset Creation:**
   - Use Blender to render synthetic views of the object from multiple angles.
   - Save both RGB images and OBB annotations.

5. **Dataset Reformatting:**
   - Convert the Blender-generated dataset to the YOLOv11n OBB format.

6. **Model Training:**
   - Train a YOLOv11n model on the synthetic dataset for OBB detection.

7. **Deployment:**
   - Use the trained model on the robotic arm for real-time pick-and-place based on live egocentric camera feed.

---

### 2. Instruction-Conditioned Manipulation (In Progress)

**Goal:** Interpret 2D schematic instructions (e.g., IKEA manuals) and execute corresponding manipulation.

#### Key Ideas:

1. **Instruction Parsing:**
   - Input: 2D sketchified image of a part from an instruction manual.
   - Output: Detected part features and geometry.

2. **2D-to-3D Matching:**
   - Match the 2D schematic part with 3D scene data captured from the RGBD sensor.

3. **Object Recognition:**
   - Use the trained YOLOv11n model to assign a class to the detected object.

4. **Manipulation:**
   - Execute a pick-and-place task based on the interpreted instruction.

---

## Repository Structure

```bash
EgoGrasp/
├── synthetic_training_pipeline/      # Fully implemented part
│   ├── assets/                       # Contains .blend files for objects
│   ├── blender_scripts/             # Blender OBB + rendering scripts
│   ├── visualize/                   # Scripts to visualize Blender or YOLOv11n outputs
│   ├── dataset_processing/          # Converts raw Blender output to YOLOv11n format
│   ├── train_yolo11n/               # Training scripts and model checkpoints
│   ├── inference/                   # Uses trained model to predict OBBs
│   ├── evaluation/                  # Baseline evaluation scripts and results (e.g., IoU, angle diff)
│   ├── YOLOv11_Dataset/             # Reformatted dataset in YOLOv11n format
│   └── README.md                    # README for synthetic training pipeline only
│
├── instruction_pipeline/            # (Planned) IKEA-style schematic to 3D recognition + manipulation
│   └── README.md                    # Will describe instruction-conditioned manipulation steps
│
├── baseline.pdf                     # Quantitative evaluation results (visual + metric)
└── README.md                        # This file (top-level overview)
