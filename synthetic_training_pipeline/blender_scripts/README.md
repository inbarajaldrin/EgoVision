## Script Overview (`blender_yolo11n.py`)

1. **Configuration**
   - **Camera name**: identifies which scene camera to use.
   - **Output directory**: where rendered images and annotation files are saved.
   - **Dataset parameters**: number of images (`num_images`), clusters per image (`num_clusters`), duplicate count, resolution, physics frames, etc.
   - **Spatial constraints**: floor dimensions (`base_width`, `base_length`), object drop height, and minimum Z threshold.

2. **Scene Setup**
   - Switches render engine to Cycles and configures resolution.
   - Enables Blender’s Object Index pass via the compositor for mask‑based visibility checks.
   - Validates that `TopCam` exists and sets it as the active camera.

3. **Object Preparation**  
   - Starts by collecting each mesh object in the scene (excluding the floor).  
   - You can control the `duplicate` parameter to create extra copies of each original mesh up front.  Adjust `duplicate = N` in the script to automatically add N clones per original.  
   - Blender names each duplicate uniquely (e.g. `MyObject.001`, `MyObject.002`, etc.), and we assign each a distinct `pass_index` so that every instance—original and duplicate—gets its own class entry.  
   - During rendering, each object (original or duplicate) will also be randomly flipped 180° around the X/Y axes, enabling the pipeline to capture both sides of non‑symmetric meshes.  

4. **Main Rendering Loop**
   For each output index `i`:
   1. **Placement**
      - Randomly clusters objects into `num_clusters` zones on the floor.
      - Randomly rotates each object (including flips) and positions it above its cluster center.
   2. **Physics Simulation**
      - Adds Rigid Body physics to all objects (active) and the floor (passive).
      - Bakes a short simulation so objects settle naturally under gravity.
   3. **Render**
      - Renders an RGB image (`%03d_rgb0001.png`).
   4. **Annotation**
      - Reads the Object Index pass to obtain a per‑pixel object ID mask.
      - For each object:
        - Skips if below the floor plane or not visible (no mask pixels).
        - Skips if less than 20% of its mask‑area is visible.
        - Projects the 8 corner points of its 3D bounding box into the image.
        - Skips if the projected box is entirely outside the frame.
        - Writes a text file (`%03d_obb0001.txt`) listing `(x,y)` pixel coordinates of each OBB corner.
   5. **Reset**
      - Restores original object transforms.

5. **Cleanup**
   - After all renders, deletes any duplicates created at the start.

---

## Outputs

- **`classes.txt`**: maps each `pass_index` to a sanitized object name.
- **`[i]_rgb0001.png`**: the rendered RGB image for each scene index `i`.
- **`[i]_obb0001.txt`**: the 8 `(x,y)` coordinates of each visible object’s projected 3D bounding box.

Use these files to train a YOLOv11‐style 2D OBB object detector. Subsequent scripts in `dataset_processing/` convert these annotations to normalized YOLOv11 format for training.

