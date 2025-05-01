# RGBD Mesh Pipeline

A step‑by‑step workflow for converting Polycam `.glb` scans into clean, aligned `.ply` meshes and exporting them as `.obj` for downstream synthetic data generation.

---

## Prerequisites

- **Blender** (tested on 4.3)  
- **Python 3.8+** with:
  - `open3d`
  - `numpy`
  - `scikit-learn`
  - `teaserpp_python` (install from https://github.com/MIT-SPARK/TEASER-plusplus.git)

---

## Folder Structure

```
rgbd_mesh_pipeline/
├── blender_scripts/
│   └── 0_conv_glb_ply.txt
├── 1_mesh_plane_removal.py
├── 1_mesh_plane_removal2.py
├── 2_centre_ply.py
├── 3_align_merge_mesh.py
├── 3_manual_rotate.py
├── 4_conv_ply_obj.py
└── helper_scripts/
    ├── align.py
    ├── ply_plane_segment_area.py
    └── ply_plane_segmenter.py
```

---

## Pipeline Steps

### 1. Convert GLB → PLY  
Use Blender to batch‑export your Polycam captures:

```bash
# In Blender's Text Editor, open:
rgbd_mesh_pipeline/blender_scripts/0_conv_glb_ply.txt
# Copy‑paste and run.
```

---

### 2. Remove Floor / Planar Segmentation  
Strip away the ground plane so only the object remains.

```bash
# Option A (default):
python3 1_mesh_plane_removal.py input.ply output_no_floor.ply

# Option B (alternate parameters):
python3 1_mesh_plane_removal2.py input.ply output_no_floor.ply
```

_Tune parameters inside the scripts for best results._

---

### 3. Center the Mesh  
Re‑center your object around the origin:

```bash
python3 2_centre_ply.py output_no_floor.ply centred.ply
```

---

### 4. Align & Merge Multiple Views  
Register two or more centred `.ply` files via TEASER++ + FPFH + ICP:

```bash
python3 3_align_merge_mesh.py view1.ply view2.ply view3.ply -o merged.ply 
```

- **Install TEASER++** in your venv first:  
  `pip install teaserpp-python` or follow https://github.com/MIT-SPARK/TEASER-plusplus.git
- If alignment isn’t perfect, re-run or…

---

### 5. Manual Rotation (Optional)  
Fine-tune two meshes by hand if automatic alignment is off:

1. Open `3_manual_rotate.py` in a Python IDE.  
2. Run and interactively rotate; enter values to save corrected transform.  
3. It outputs a rotated `.ply`.

---

### 6. Convert PLY → OBJ  
Bake vertex colors onto triangles and export for Blender import:

```bash
python3 4_conv_ply_obj.py merged.ply merged.obj
```

Now you have a full‑color `.obj` + `.mtl` ready for synthetic‑data rendering.

---

## Blender Scene Setup for Synthetic Data

Inside Blender’s Scripting workspace, run in order:

1. **Import your merged OBJ**  
   `blender_scripts/1_import_objs.txt`  
2. **Add camera & lights**  
   `blender_scripts/2_add_camera.txt`  
3. **Add rigid‑body physics (optional)**  
   `blender_scripts/3_add_rigid_body.txt`  
4. **Start YOLO data render**  
   `blender_scripts/4_yolo.txt`  

> **Tip:** In `4_yolo.txt`, you can adjust:
> - Number of object duplicates (to use less images but have more annotated data) 
> - Drop heights / cluster counts  

This generates RGB images + OBB annotation `.txt` pairs for detection training.

---

## Next: Synthetic Training Pipeline

Once your images & OBBs are in `synthetic_training_pipeline/blender_outputs/`, reformat for YOLOv11n:

```bash
cd synthetic_training_pipeline
python3 dataset_processing/YOLOv11_reformat.py
```

Then follow the steps in that module’s README to train, infer, and evaluate.

---
