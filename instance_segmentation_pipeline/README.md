# Instance Segmentation Pipeline for EgoGrasp

This module enables instance segmentation training using synthetic masks generated from Blender. It is part of the larger EgoGrasp project and covers the pipeline from 3D scenes to Detectron2 Mask R-CNN training and real-time inference.

---

## Pipeline Steps

### 1. Prepare Object Scene in Blender

* Place your `.blend` file (e.g., `Allen_key.blend`) inside the `assets/` folder.

```
instance_segmentation_pipeline/assets/Allen_key.blend
```

---

### 2. Generate RGB + Mask Renders in Blender

* Run the script inside blender

| Script                             | Purpose                                                        |
| ---------------------------------- | -------------------------------------------------------------- |
| `0_blender_mask_gen_single_obj.py` | Renders RGB and binary masks for single-object scenes.         |
| `0_blender_mask_gen_multi_obj.py`  | Renders RGB and per-instance ID masks for multi-object scenes. |

```bash
blender -b assets/Allen_key.blend \
        --python 0_blender_mask_gen_single_obj.py
```

Each frame outputs:

```
scene_0000.png      scene_0000_mask.png
scene_0001.png      scene_0001_mask.png
...
```

---

### 3. Visualize Generated Masks

Use this script to preview mask overlays on RGB images:

```bash
python 1_visualize_mask_blender.py --data_dir renders/ --image_id 0
```

It opens a composite window showing raw RGB, mask overlay, and segmentation map.

---

### 4. Reformat Data to COCO Format

Convert the rendered RGB and mask images into Detectron2-compatible COCO format:

```bash
python 2_reformat_coco.py renders/ COCO_Dataset/
```

Resulting folder structure:

```
COCO_Dataset/
├── annotations/
│   └── instances_train.json
├── images/
├── masks/
```

---

### 5. Train Detectron2 Mask R-CNN

To start training:

```bash
python 3_train_detectron2.py \
    --dataset_path COCO_Dataset/ \
    --output_dir runs/mask_rcnn/
```

Weights and training logs are saved under:

```
runs/mask_rcnn/
├── model_final.pth
├── metrics.json
└── events.out.tfevents... (TensorBoard logs)
```

---

### 6. Run Inference on Test Images

Use a trained model to predict on test images:

```bash
python 4_inference_detectron2.py \
    --model_path runs/mask_rcnn/model_final.pth \
    --input_dir test_images/ \
    --output_dir predictions/
```

Each prediction will be saved with mask overlays and confidence scores.

---

### 7. Deploy on RealSense Camera

Run the model on a live RGB stream from an Intel RealSense camera:

```bash
python 5_deploy_intel_camera.py \
    --model_path runs/mask_rcnn/model_final.pth \
    --camera_topic /camera/color/image_raw
```

The window displays real-time segmentation results from the live feed.

---

## Folder Summary

```
instance_segmentation_pipeline/
├── 0_blender_mask_gen_single_obj.py
├── 0_blender_mask_gen_multi_obj.py
├── 1_visualize_mask_blender.py
├── 2_reformat_coco.py
├── 3_train_detectron2.py
├── 4_inference_detectron2.py
├── 5_deploy_intel_camera.py
```

---

