import bpy
import numpy as np
from mathutils import Matrix, Vector
import os
import json
import cv2
from datetime import datetime

# --------------- CONFIG ------------------
ply_folder = '/home/aaugus11/Projects/cnos/fmb_dataset/'
poses_file = '/home/aaugus11/Projects/cnos/src/poses/predefined_poses/cam_poses_level2.npy'
output_base_dir = '/home/aaugus11/Projects/cnos/fmb_dataset/rendered_line_drawing/'

# Create COCO dataset structure
images_dir = os.path.join(output_base_dir, 'images')
annotations_dir = os.path.join(output_base_dir, 'annotations')
masks_dir = os.path.join(output_base_dir, 'masks')  # For debugging

os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# COCO annotation structure
coco_dataset = {
    "info": {
        "description": "Synthetic Instruction Manual Objects Dataset",
        "url": "",
        "version": "1.0",
        "year": 2025,
        "contributor": "Blender Synthetic Generation",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "licenses": [
        {
            "id": 1,
            "name": "Custom License",
            "url": ""
        }
    ],
    "images": [],
    "annotations": [],
    "categories": []
}

# Global counters
image_id = 1
annotation_id = 1
category_id = 1
category_map = {}

# --------------- CLEAN SCENE ------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
    bpy.data.objects.remove(cam, do_unlink=True)

# --------------- SETUP RENDERING ------------------
scene = bpy.context.scene
view_layer = scene.view_layers["ViewLayer"]

# Enable Freestyle rendering
scene.render.use_freestyle = True
view_layer.use_freestyle = True
view_layer.freestyle_settings.as_render_pass = True
bpy.context.scene.view_settings.view_transform = 'Standard'

# Enable compositor nodes
scene.use_nodes = True

# Set render resolution
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.image_settings.file_format = 'PNG'

# Remove the IndexOB pass setup since it's causing issues
# view_layer.use_pass_object_index = True

def setup_freestyle_compositor():
    """Setup compositor nodes for freestyle rendering"""
    tree = scene.node_tree
    tree.nodes.clear()
    
    rl = tree.nodes.new(type='CompositorNodeRLayers')
    rl.location = (-300, 0)
    
    mix = tree.nodes.new(type='CompositorNodeMixRGB')
    mix.location = (0, 0)
    mix.blend_type = 'MIX'
    mix.inputs[0].default_value = 1.0
    mix.use_alpha = True
    
    comp = tree.nodes.new(type='CompositorNodeComposite')
    comp.location = (300, 0)
    
    tree.links.new(rl.outputs['Freestyle'], mix.inputs[2])
    tree.links.new(mix.outputs['Image'], comp.inputs['Image'])

def setup_mask_compositor():
    """Setup compositor nodes for mask rendering using material override"""
    # We'll use material override method instead of IndexOB pass
    # This will be handled in the create_material_mask function
    pass

def create_material_mask(output_path, image_name, obj):
    """Create object mask by rendering with material override - disabling freestyle"""
    scene = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]
    
    # Store original materials and visibility for ALL objects
    original_materials = {}
    original_visibility = {}
    original_world = scene.world
    original_film_transparent = scene.render.film_transparent
    
    # CRITICAL: Store and disable freestyle settings
    original_use_freestyle = scene.render.use_freestyle
    original_freestyle_enabled = view_layer.use_freestyle
    
    print(f"DEBUG: Creating mask for {obj.name}")
    
    try:
        # DISABLE FREESTYLE for mask rendering
        scene.render.use_freestyle = False
        view_layer.use_freestyle = False
        print("DEBUG: Disabled freestyle for mask rendering")
        
        # Get all mesh objects in scene
        all_scene_objects = [o for o in scene.objects if o.type == 'MESH']
        
        # Store original visibility for ALL mesh objects
        for scene_obj in all_scene_objects:
            original_visibility[scene_obj] = scene_obj.hide_render
            scene_obj.hide_render = True  # Hide everything initially
        
        print(f"DEBUG: Hidden {len(all_scene_objects)} total mesh objects")
        
        # Store original materials for target object
        if obj.data.materials:
            original_materials[obj] = [slot.material for slot in obj.material_slots]
        
        # Create black world for background
        if "TempBlackWorld" in bpy.data.worlds:
            bpy.data.worlds.remove(bpy.data.worlds["TempBlackWorld"])
            
        black_world = bpy.data.worlds.new("TempBlackWorld")
        black_world.use_nodes = True
        black_world.node_tree.nodes.clear()
        
        # Create background shader and world output
        background = black_world.node_tree.nodes.new('ShaderNodeBackground')
        world_output = black_world.node_tree.nodes.new('ShaderNodeOutputWorld')
        
        # Set background to pure black
        background.inputs['Color'].default_value = (0, 0, 0, 1)  # Pure black
        background.inputs['Strength'].default_value = 1.0
        
        # Connect background to world output
        black_world.node_tree.links.new(background.outputs['Background'], world_output.inputs['Surface'])
        
        # Apply black world and disable film transparency
        scene.world = black_world
        scene.render.film_transparent = False
        
        # Create white emission material for target object
        if "TempWhiteMaterial" in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials["TempWhiteMaterial"])
            
        white_material = bpy.data.materials.new("TempWhiteMaterial")
        white_material.use_nodes = True
        white_material.node_tree.nodes.clear()
        
        # Create emission shader with pure white
        emission = white_material.node_tree.nodes.new('ShaderNodeEmission')
        output_node = white_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        
        emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
        emission.inputs['Strength'].default_value = 2.0  # Increase strength for pure white
        
        white_material.node_tree.links.new(emission.outputs['Emission'], output_node.inputs['Surface'])
        
        # Apply white material to target object and make it visible
        obj.data.materials.clear()
        obj.data.materials.append(white_material)
        obj.hide_render = False
        
        print(f"DEBUG: Applied white material to {obj.name}, hide_render = {obj.hide_render}")
        
        # Temporarily disable compositor to avoid any interference
        original_use_nodes = scene.use_nodes
        scene.use_nodes = False
        
        # Render mask
        mask_filepath = os.path.join(output_path, f"{image_name}_mask")
        scene.render.filepath = mask_filepath
        bpy.ops.render.render(write_still=True)
        
        # Restore compositor setting
        scene.use_nodes = original_use_nodes
        
        # Restore freestyle settings FIRST
        scene.render.use_freestyle = original_use_freestyle
        view_layer.use_freestyle = original_freestyle_enabled
        
        # Restore original materials for target object
        if obj in original_materials:
            obj.data.materials.clear()
            for mat in original_materials[obj]:
                obj.data.materials.append(mat)
        
        # Restore original visibility for ALL objects
        for scene_obj, visibility in original_visibility.items():
            scene_obj.hide_render = visibility
        
        # Restore original world and render settings
        scene.world = original_world
        scene.render.film_transparent = original_film_transparent
        
        # Clean up temporary materials and world
        bpy.data.materials.remove(white_material)
        bpy.data.worlds.remove(black_world)
        
        mask_path = f"{mask_filepath}.png"
        if os.path.exists(mask_path):
            print(f"DEBUG: Mask created successfully at {mask_path}")
            return mask_path
        else:
            print(f"DEBUG: Mask file not found at {mask_path}")
            return None
        
    except Exception as e:
        print(f"Error creating mask: {e}")
        
        # Emergency cleanup
        try:
            # Restore freestyle settings
            scene.render.use_freestyle = original_use_freestyle
            view_layer.use_freestyle = original_freestyle_enabled
            
            # Restore materials
            if obj in original_materials:
                obj.data.materials.clear()
                for mat in original_materials[obj]:
                    obj.data.materials.append(mat)
            
            # Restore visibility
            for scene_obj, visibility in original_visibility.items():
                scene_obj.hide_render = visibility
            
            # Restore world
            scene.world = original_world
            scene.render.film_transparent = original_film_transparent
            
            # Restore compositor
            scene.use_nodes = original_use_nodes
            
            # Cleanup temp materials
            if "TempWhiteMaterial" in bpy.data.materials:
                bpy.data.materials.remove(bpy.data.materials["TempWhiteMaterial"])
            if "TempBlackWorld" in bpy.data.worlds:
                bpy.data.worlds.remove(bpy.data.worlds["TempBlackWorld"])
        except:
            pass
            
        return None

def mask_to_coco_segmentation(mask_path):
    """Convert binary mask to COCO segmentation format"""
    try:
        # Read mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read mask: {mask_path}")
            return None, None, None
        
        # Threshold to ensure binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in mask")
            return None, None, None
        
        # Get the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        area = cv2.contourArea(largest_contour)
        if area < 100:  # Minimum area threshold
            print(f"Contour too small: {area}")
            return None, None, None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = [float(x), float(y), float(w), float(h)]  # COCO format: [x, y, width, height]
        
        # Convert contour to segmentation format
        segmentation = []
        if len(largest_contour) > 2:
            # Simplify contour to reduce points
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Flatten the contour points
            contour_flat = simplified_contour.flatten()
            if len(contour_flat) >= 6:  # Need at least 3 points (6 coordinates)
                segmentation = [contour_flat.astype(float).tolist()]
        
        if not segmentation:
            print("Could not create valid segmentation")
            return None, None, None
        
        return segmentation, bbox, float(area)
        
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None, None, None

# --------------- LOAD CAMERA POSES ------------------
poses = np.load(poses_file)

def look_at(cam_obj, target):
    direction = (target - cam_obj.location).normalized()
    up = Vector((0, 0, 1))
    right = direction.cross(up).normalized()
    up_corrected = right.cross(direction).normalized()
    rot = Matrix((right, up_corrected, -direction)).transposed()
    cam_obj.matrix_world = Matrix.Translation(cam_obj.location) @ rot.to_4x4()

# Create cameras
cameras = []
for idx, pose in enumerate(poses):
    t = pose[:3, 3] / 1000  # mm to meters
    t = t * 0.2  # Move 50% closer
    location = Vector(t)
    
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = f"Cam_{idx:03d}"
    look_at(cam, Vector((0, 0, 0)))
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100
    cameras.append(cam)

# --------------- PROCESS PLY FILES ------------------
ply_files = [f for f in os.listdir(ply_folder) if f.lower().endswith('.ply')]

for filename in ply_files:
    object_name = os.path.splitext(filename)[0]
    print(f"\nProcessing object: {object_name}")
    
    # Add category if not exists
    if object_name not in category_map:
        category_info = {
            "id": category_id,
            "name": object_name,
            "supercategory": "instruction_manual_objects"
        }
        coco_dataset["categories"].append(category_info)
        category_map[object_name] = category_id
        category_id += 1
    
    current_category_id = category_map[object_name]
    
    # Import PLY
    filepath = os.path.join(ply_folder, filename)
    bpy.ops.wm.ply_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    obj.location = (0, 0, 0)
    obj.pass_index = 1
    
    # Skip first and last camera
    for cam in cameras[1:-1]:
        bpy.context.scene.camera = cam
        
        # Generate unique filenames
        image_filename = f"{object_name}_{cam.name}.png"
        mask_filename = f"{object_name}_{cam.name}_mask.png"
        
        image_path = os.path.join(images_dir, image_filename)
        temp_mask_path = os.path.join(masks_dir, mask_filename)
        
        # Render freestyle image
        setup_freestyle_compositor()
        scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        
        # Create mask using material override method
        temp_mask_path = create_material_mask(masks_dir, f"{object_name}_{cam.name}", obj)
        
        if temp_mask_path and os.path.exists(temp_mask_path):
            # Process mask to get COCO annotations
            segmentation, bbox, area = mask_to_coco_segmentation(temp_mask_path)
            
            if segmentation and bbox and area:
                # Add image info
                image_info = {
                    "id": image_id,
                    "width": 640,
                    "height": 480,
                    "file_name": image_filename,
                    "license": 1,
                    "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                coco_dataset["images"].append(image_info)
                
                # Add annotation
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": current_category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(annotation_info)
                
                print(f"✓ Processed {cam.name} - Image ID: {image_id}")
                
                image_id += 1
                annotation_id += 1
            else:
                print(f"✗ Failed to process mask for {cam.name}")
                # Remove the failed image file
                if os.path.exists(image_path):
                    os.remove(image_path)
        else:
            print(f"✗ Failed to create mask for {cam.name}")
            if os.path.exists(image_path):
                os.remove(image_path)
    
    # Delete imported object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()

# --------------- SAVE COCO ANNOTATIONS ------------------
# Save complete dataset
annotations_file = os.path.join(annotations_dir, 'instances_train.json')
with open(annotations_file, 'w') as f:
    json.dump(coco_dataset, f, indent=2)

# Create train/val split (80/20)
total_images = len(coco_dataset["images"])
train_count = int(total_images * 0.8)

# Training set
train_dataset = {
    "info": coco_dataset["info"],
    "licenses": coco_dataset["licenses"],
    "categories": coco_dataset["categories"],
    "images": coco_dataset["images"][:train_count],
    "annotations": [ann for ann in coco_dataset["annotations"] 
                   if ann["image_id"] <= train_count]
}

# Validation set
val_images = coco_dataset["images"][train_count:]
val_image_ids = [img["id"] for img in val_images]
val_dataset = {
    "info": coco_dataset["info"],
    "licenses": coco_dataset["licenses"], 
    "categories": coco_dataset["categories"],
    "images": val_images,
    "annotations": [ann for ann in coco_dataset["annotations"] 
                   if ann["image_id"] in val_image_ids]
}

# Save split datasets
train_file = os.path.join(annotations_dir, 'instances_train2025.json')
val_file = os.path.join(annotations_dir, 'instances_val2025.json')

with open(train_file, 'w') as f:
    json.dump(train_dataset, f, indent=2)

with open(val_file, 'w') as f:
    json.dump(val_dataset, f, indent=2)

# --------------- SUMMARY ------------------
print(f"\n{'='*50}")
print("DATASET GENERATION COMPLETE!")
print(f"{'='*50}")
print(f"Output directory: {output_base_dir}")
print(f"Total images: {len(coco_dataset['images'])}")
print(f"Total annotations: {len(coco_dataset['annotations'])}")
print(f"Categories: {len(coco_dataset['categories'])}")
print(f"Train images: {len(train_dataset['images'])}")
print(f"Validation images: {len(val_dataset['images'])}")
print(f"\nFiles created:")
print(f"  - {train_file}")
print(f"  - {val_file}")
print(f"  - Images in: {images_dir}")
print(f"  - Debug masks in: {masks_dir}")

# Print categories
print(f"\nDetected categories:")
for cat in coco_dataset["categories"]:
    count = len([ann for ann in coco_dataset["annotations"] if ann["category_id"] == cat["id"]])
    print(f"  - {cat['name']} (ID: {cat['id']}): {count} instances")

print(f"\nReady for Detectron2 training!") 
