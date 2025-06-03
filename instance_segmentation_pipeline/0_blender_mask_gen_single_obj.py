#!/usr/bin/env python3
import bpy
import bmesh
import numpy as np
import json
import os
import random
import math
from datetime import datetime
from mathutils import Vector
import cv2
import bpy_extras.object_utils

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
camera_name = "TopCam"
output_directory = os.path.expanduser("~/Downloads/newnew_test")
num_images = 5
num_clusters = 1
duplicate = 0
image_resolution = (512, 512)
physics_sim_frames = 40
max_drop_height = 0.1
base_width, base_length = 1.2, 2.0
margin = 0.1
base_name = "Cube"  # Floor/table object name
table_z_threshold = 0.0

def setup_render_settings(output_path, image_name):
    """Configure render settings for image and mask generation"""
    scene = bpy.context.scene
    
    # Set render engine and basic settings
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x, scene.render.resolution_y = image_resolution
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    
    # Set output path
    scene.render.filepath = os.path.join(output_path, image_name)
    
    # Enable render passes for segmentation
    view_layer = scene.view_layers["ViewLayer"]
    
    # Try to enable object index pass
    try:
        view_layer.use_pass_object_index = True
        view_layer.use_pass_material_index = True
        print("Object index pass enabled")
    except AttributeError:
        print("Object index pass not available in current render engine")
    
    return scene

def assign_object_indices(objects):
    """Assign unique indices to mesh objects"""
    for i, obj in enumerate(objects):
        obj.pass_index = i + 1
    return objects

def get_object_bounds_2d(obj, camera, scene):
    """Get 2D bounding box of object in camera view"""
    # Get object vertices in world space
    mesh = obj.data
    world_coords = [obj.matrix_world @ v.co for v in mesh.vertices]
    
    # Project to camera space
    render = scene.render
    camera_coords = []
    
    for coord in world_coords:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, coord)
        
        # Convert to pixel coordinates
        x = co_2d.x * render.resolution_x
        y = (1 - co_2d.y) * render.resolution_y  # Flip Y axis
        
        # Only include points in front of camera
        if co_2d.z > 0:
            camera_coords.append((x, y))
    
    if not camera_coords:
        return None
    
    # Calculate bounding box
    xs, ys = zip(*camera_coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Clamp to image bounds
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(render.resolution_x, max_x)
    max_y = min(render.resolution_y, max_y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return None
    
    return [min_x, min_y, width, height]

def get_object_segmentation_mask(obj, mask_image, obj_index=None):
    """Extract segmentation polygon from binary mask"""
    # Convert mask to numpy array
    pixels = np.array(mask_image.pixels[:])
    
    # Reshape to image dimensions (RGBA)
    height = mask_image.size[1]
    width = mask_image.size[0]
    pixels = pixels.reshape((height, width, 4))
    
    # ADD THIS LINE:
    pixels = np.flipud(pixels)  # Flip vertically to match image coordinates

    # Extract white pixels (objects) from the mask
    # Check red channel for white pixels (value close to 1.0)
    object_mask = (pixels[:, :, 0] > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], 0
    
    # Get all contours (since we can't distinguish individual objects in a binary mask)
    # For now, we'll return the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Flatten contour points
    segmentation = largest_contour.flatten().tolist()
    area = cv2.contourArea(largest_contour)
    
    return [segmentation], area

def create_manual_mask(output_path, image_name, scene, objects):
    """Create object index mask by rendering objects with materials"""
    # Store original materials and visibility
    original_materials = {}
    original_visibility = {}
    
    if not objects:
        print("No visible mesh objects found")
        return None
    
    # Store original world and render settings
    original_world = scene.world
    original_film_transparent = scene.render.film_transparent
    
    # Create black world for background
    black_world = bpy.data.worlds.new("BlackWorld")
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
    
    # Create white emission material for target objects
    white_material = bpy.data.materials.new(name="WhiteMask")
    white_material.use_nodes = True
    white_material.node_tree.nodes.clear()
    
    # Create emission shader with pure white
    emission = white_material.node_tree.nodes.new('ShaderNodeEmission')
    output = white_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    
    emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    emission.inputs['Strength'].default_value = 1.0
    
    white_material.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Hide base object and apply white material to target objects
    base_obj = bpy.data.objects.get(base_name)
    if base_obj:
        original_visibility[base_obj] = base_obj.hide_render
        base_obj.hide_render = True  # Hide base object in render
    
    # Apply white material to target objects only
    for obj in objects:
        if obj.name != base_name:  # Skip base object
            original_materials[obj] = [slot.material for slot in obj.material_slots]
            obj.data.materials.clear()
            obj.data.materials.append(white_material)
    
    # Render mask with proper filename
    mask_filepath = os.path.join(output_path, f"{image_name}_mask")
    scene.render.filepath = mask_filepath
    bpy.ops.render.render(write_still=True)
    
    # Restore original materials
    for obj, materials in original_materials.items():
        obj.data.materials.clear()
        for mat in materials:
            obj.data.materials.append(mat)
    
    # Restore original visibility
    for obj, visibility in original_visibility.items():
        obj.hide_render = visibility
    
    # Restore original world and render settings
    scene.world = original_world
    scene.render.film_transparent = original_film_transparent
    
    # Clean up temporary materials and world
    bpy.data.materials.remove(white_material)
    bpy.data.worlds.remove(black_world)
    
    # Reset render filepath
    scene.render.filepath = os.path.join(output_path, image_name)
    
    # Load mask image
    mask_path = f"{mask_filepath}.png"
    if os.path.exists(mask_path):
        mask_image = bpy.data.images.load(mask_path)
        return mask_image
    else:
        print(f"Mask file not found: {mask_path}")
        return None

def render_image_and_mask(output_path, image_name, objects, render_mask=True):
    """Render the main image and optionally the object index mask"""
    scene = bpy.context.scene
    
    # Render main image
    bpy.ops.render.render(write_still=True)
    
    # If we don't need the mask, return None
    if not render_mask:
        return None
    
    # Disable compositor to avoid frame number suffixes
    scene.use_nodes = False
    
    # Include ALL scene objects for mask rendering, the function will handle base exclusion
    all_objects = [o for o in scene.objects if o.type == 'MESH']
    mask_image = create_manual_mask(output_path, image_name, scene, all_objects)
    
    return mask_image

def create_coco_annotation(objects, camera, scene, mask_image, image_info):
    """Create COCO format annotations"""
    annotations = []
    categories = []
    category_map = {}
    
    annotation_id = 1
    category_id = 1
    
    for obj in objects:
        # Skip base object
        if obj.name == base_name:
            continue
            
        # Skip objects below table threshold
        zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
        if max(zs) < table_z_threshold:
            continue
            
        obj_index = obj.pass_index
        
        # Create category if not exists
        category_name = obj.name.split('_')[0].lower()  # Use underscore split like code2
        if category_name not in category_map:
            category_map[category_name] = category_id
            categories.append({
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            })
            category_id += 1
        
        # Get 2D bounding box
        bbox = get_object_bounds_2d(obj, camera, scene)
        if bbox is None:
            continue
        
        # Get segmentation mask
        segmentation, area = get_object_segmentation_mask(obj, mask_image, obj_index)
        if not segmentation or area == 0:
            continue
        
        # Create annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_info["id"],
            "category_id": category_map[category_name],
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": 0
        }
        
        annotations.append(annotation)
        annotation_id += 1
    
    return annotations, categories

def generate_cluster_centers(n):
    """Generate cluster centers for object placement"""
    return [(random.uniform(-base_width/2+margin, base_width/2-margin),
             random.uniform(-base_length/2+margin, base_length/2-margin))
            for _ in range(n)]

def half_extent_xy(obj):
    """Calculate half extents of object in XY plane"""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [p.x for p in corners]
    ys = [p.y for p in corners]
    return (max(xs)-min(xs))/2, (max(ys)-min(ys))/2

def randomize_objects(objects, centers):
    """Randomize object positions and rotations"""
    for obj in objects:
        # Skip base object
        if obj.name == base_name:
            continue
            
        # Random rotation
        obj.rotation_euler = (
            math.pi * (random.random() < 0.5),
            math.pi * (random.random() < 0.5),
            random.uniform(0, 2 * math.pi)
        )
        
        # Random position within cluster
        cx, cy = random.choice(centers)
        hx, hy = half_extent_xy(obj)
        
        x = random.uniform(max(-base_width/2 + hx + margin, cx - 0.05),
                          min(base_width/2 - hx - margin, cx + 0.05))
        y = random.uniform(max(-base_length/2 + hy + margin, cy - 0.05),
                          min(base_length/2 - hy - margin, cy + 0.05))
        
        obj.location = (x, y, random.uniform(0, max_drop_height))

def setup_physics(objects):
    """Setup rigid body physics for objects"""
    for obj in objects:
        # Skip base object for active physics
        if obj.name == base_name:
            continue
            
        if not obj.rigid_body:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        obj.rigid_body.mass = 1.0
    
    # Setup floor/table
    floor = bpy.data.objects.get(base_name)
    if floor and not floor.rigid_body:
        bpy.context.view_layer.objects.active = floor
        bpy.ops.rigidbody.object_add()
        floor.rigid_body.type = 'PASSIVE'
        floor.rigid_body.collision_shape = 'BOX'

def run_physics_simulation():
    """Run physics simulation"""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = physics_sim_frames
    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(physics_sim_frames)

def delete_file_if_exists(filepath):
    """Delete file if it exists"""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Deleted: {filepath}")
        except OSError as e:
            print(f"Failed to delete {filepath}: {e}")

def is_object_in_camera_view(obj, camera, scene):
    """Check if object is visible within camera frustum"""
    # Get all bounding box corners in world space
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    
    visible_corners = 0
    for corner in corners:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, corner)
        
        # Check if point is in front of camera (z > 0) and within frame bounds
        if (co_2d.z > 0 and 
            0 <= co_2d.x <= 1 and 
            0 <= co_2d.y <= 1):
            visible_corners += 1
    
    # Object is considered visible if at least 2 corners are in view
    return visible_corners >= 2

def has_sufficient_projected_area(obj, camera, scene, min_area_ratio=0.001):
    """Check if object has sufficient projected area in camera view"""
    bbox = get_object_bounds_2d(obj, camera, scene)
    if bbox is None:
        return False
    
    # Calculate projected area as ratio of total image area
    projected_area = bbox[2] * bbox[3]
    total_area = scene.render.resolution_x * scene.render.resolution_y
    area_ratio = projected_area / total_area
    
    return area_ratio > min_area_ratio

def is_object_properly_visible(obj, camera, scene):
    """Combined check for object visibility"""
    # Z-height test (your existing logic)
    zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
    if max(zs) < table_z_threshold:
        return False
    
    # Camera view test
    if not is_object_in_camera_view(obj, camera, scene):
        return False
    
    # Projected area test
    if not has_sufficient_projected_area(obj, camera, scene):
        return False
    
    return True

def export_coco_dataset_with_randomization():
    """Main function to export COCO dataset with randomization"""
    # Setup directories
    os.makedirs(output_directory, exist_ok=True)
    
    # Setup scene and camera
    scene = bpy.context.scene
    if camera_name not in bpy.data.objects:
        raise ValueError(f"Camera '{camera_name}' not found.")
    scene.camera = bpy.data.objects[camera_name]
    
    # Get original objects (excluding base/floor)
    original_objects = [o for o in scene.objects 
                       if o.type == 'MESH' and o.name != base_name]
    
    # Create duplicates if needed
    dups_global = []
    for o in list(original_objects):
        for _ in range(duplicate):
            c = o.copy()
            c.data = o.data.copy()
            bpy.context.collection.objects.link(c)
            dups_global.append(c)
    original_objects.extend(dups_global)
    
    # Store original transforms
    original_transforms = {
        o.name: (o.location.copy(), o.rotation_euler.copy()) 
        for o in original_objects
    }
    
    # Create category mapping (excluding base object)
    cat_map = {}
    for obj in original_objects:
        if obj.name != base_name:
            cls = obj.name.split("_")[0].lower()
            if cls not in cat_map:
                cat_map[cls] = len(cat_map) + 1
    
    # Initialize COCO dataset structure
    coco_dataset = {
        "info": {
            "description": "Blender Generated Dataset with Randomization",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Blender Script",
            "date_created": datetime.now().isoformat()
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
        "categories": [{"id": v, "name": k, "supercategory": "object"} 
                      for k, v in cat_map.items()]
    }
    
    all_annotations = []
    annotation_id_counter = 1
    
    # Generate images (skip first iteration but still process physics)
    for img_id in range(num_images):
        print(f"[Image {img_id+1}/{num_images}]")
        
        # Generate cluster centers and randomize objects
        centers = generate_cluster_centers(num_clusters)
        randomize_objects(original_objects, centers)
        
        # Setup physics and run simulation
        setup_physics(original_objects)
        run_physics_simulation()
        
        # Skip rendering and annotation creation for the first image
        if img_id == 0:
            print("Skipping first image (physics warmup)")
            # Reset object transforms
            for obj in original_objects:
                if obj.name in original_transforms:
                    loc, rot = original_transforms[obj.name]
                    obj.location, obj.rotation_euler = loc, rot
            continue
        
        # Adjust image numbering to start from 1 instead of 0
        actual_img_id = img_id - 1  # This makes the first saved image have ID 0
        image_name = f"scene_{actual_img_id:04d}"
        
        # Setup render settings
        scene = setup_render_settings(output_directory, image_name)
        camera = scene.camera
                
        # Get visible objects (excluding base) for annotation
        visible_objects = [obj for obj in original_objects if obj.visible_get() and obj.name != base_name]

        # Check if objects are properly visible
        properly_visible = [obj for obj in visible_objects 
                        if is_object_properly_visible(obj, camera, scene)]

        if not properly_visible:
            print(f"No properly visible objects found for image {actual_img_id}, skipping...")
            # Reset and continue
            for obj in original_objects:
                if obj.name in original_transforms:
                    loc, rot = original_transforms[obj.name]
                    obj.location, obj.rotation_euler = loc, rot
            continue

        # Use the properly visible objects for annotation
        visible_objects = properly_visible
        assign_object_indices(visible_objects)

        # Render image and mask (function handles base exclusion internally)
        mask_image = render_image_and_mask(output_directory, image_name, visible_objects, render_mask=True)
        
        if mask_image is None:
            print(f"Failed to create mask for image {actual_img_id}")
            continue
        
        # Create image info
        image_info = {
            "id": actual_img_id,
            "file_name": f"{image_name}.png",
            "width": image_resolution[0],
            "height": image_resolution[1],
            "date_captured": datetime.now().isoformat()
        }
        
        coco_dataset["images"].append(image_info)
        
        # Create annotations (function handles base exclusion internally)
        annotations, categories = create_coco_annotation(
            visible_objects, camera, scene, mask_image, image_info
        )
        
        # Update annotation IDs
        for ann in annotations:
            ann["id"] = annotation_id_counter
            annotation_id_counter += 1
        
        all_annotations.extend(annotations)
        
        print(f"Generated {len(annotations)} annotations for image {actual_img_id}")
        
        # Cleanup mask image from Blender
        bpy.data.images.remove(mask_image)
        
        # Reset object transforms
        for obj in original_objects:
            if obj.name in original_transforms:
                loc, rot = original_transforms[obj.name]
                obj.location, obj.rotation_euler = loc, rot
    
    # Add all annotations to dataset
    coco_dataset["annotations"] = all_annotations
    
    # Save COCO JSON
    json_path = os.path.join(output_directory, "coco_annotations.json")
    with open(json_path, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"Exported {len(all_annotations)} total annotations across {len(coco_dataset['images'])} images")
    print(f"Dataset saved to: {json_path}")
    
    # Cleanup duplicates
    for dup in dups_global:
        bpy.data.objects.remove(dup, do_unlink=True)
    
    return coco_dataset

# Example usage
if __name__ == "__main__":
    export_coco_dataset_with_randomization()
    print("✓ Done!")