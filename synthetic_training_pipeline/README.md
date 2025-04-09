 
# Synthetic Training Pipeline for EgoGrasp

This module enables self-supervised object detection training using synthetic data generated from Blender. It is part of the larger EgoGrasp project and covers the pipeline from 3D model to YOLOv11n training and evaluation.

---

## Pipeline Steps

### 1. Prepare Object CAD Model

- Place your `.blend` file (e.g., `jenga.blend`) inside the `assets/` folder.

```
synthetic_training_pipeline/assets/jenga.blend
```

### 2. Generate Synthetic Data in Blender

- Open `jenga.blend` in Blender.
- Open the script in `blender_scripts/blender_yolo11n.txt`, copy and paste it into Blender's scripting environment, and run it.

This will generate RGB images and OBB annotations like the following:

```
000_rgb0001.png   000_obb0001.txt
001_rgb0001.png   001_obb0001.txt
...
```

Files will be created for different viewing angles of the object.

---

### 3. Reformat Data to YOLOv11n Dataset

Run the following script to convert Blender's RGB/OBB pairs into YOLOv11n dataset format:

```bash
python dataset_processing/YOLOv11_reformat.py
```

This creates the structure:

```
YOLOv11_Dataset/
├── dataset.yaml
├── train/
├── val/
└── test/
    ├── images/
    └── labels/
```

---

### 4. Train YOLOv11n OBB Detector

To train the detector on the synthetic dataset:

```bash
python train_yolo11n/train_yolo11n.py
```

The resulting model (e.g., `best.pt`, `last.pt`) will be stored under a subfolder of:

```
train_yolo11n/runs/obb/jenga_obb_aug/weights/
```

---

### 5. Visualize Blender Annotations

To verify OBB overlays generated directly from Blender output:

```bash
python visualize/visualize_obb_blender.py
```

To verify YOLOv11n-formatted labels:

```bash
python visualize/visualize_obb_yolo11n.py
```

---

### 6. Inference Using Trained Model

Use the trained YOLOv11n model to predict on test set images and draw predicted OBBs:

```bash
python inference/infer_and_draw.py
```

This randomly selects 10 test images and saves visualized output.

---

### 7. Evaluate Predictions with Baseline Metrics

The script `baseline.py` computes and visualizes the following metrics between predicted and ground truth oriented bounding boxes:

- IoU (Intersection over Union)
- Centroid Distance (in pixels)
- Orientation Angle Difference (in degrees)

Run:

```bash
python evaluation/baseline.py
```

Output visualizations and a 4-column comparison figure are saved.


## 🔹 Folder Summary

```
synthetic_training_pipeline/
├── assets/                      # Blender object files (e.g. jenga.blend)
├── blender_scripts/            # Blender data generation script
├── dataset_processing/         # Reformat Blender data to YOLOv11n dataset format
├── YOLOv11_Dataset/            # Reformatted dataset (train/val/test)
├── train_yolo11n/              # Training script and model weights
├── inference/                  # Inference script for test images
├── visualize/                  # Helper visualization scripts
├── evaluation/                 # Metrics and summary PDF
│   └── baseline.pdf            # Visual + quantitative results
└── README.md                   # This file
```

---

