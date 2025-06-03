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
output_directory = os.path.expanduser("~/Downloads/FBM_Assembly3_raw2")
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

def extract_category_name(obj_name):
    """Use the full object name as category name (only removing Blender numeric suffixes)"""
    import re
    
    # Only remove Blender's automatic numeric suffixes like .001, .002, etc.
    # This preserves your intentional naming like "fork_yellow", "fork_green", "line_orange"
    clean_name = re.sub(r'\.\d{3,}$', '', obj_name)  # Removes .001, .002, etc.
    
    # Convert to lowercase for consistency
    return clean_name.lower()

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

# IMPROVED: Better segmentation for multiple objects
def get_object_mask_per_category(mask_image, objects, camera, scene):
    """Extract individual object masks by analyzing connected components"""
    # Convert mask to numpy array
    pixels = np.array(mask_image.pixels[:])
    
    # Reshape to image dimensions (RGBA)
    height = mask_image.size[1]
    width = mask_image.size[0]
    pixels = pixels.reshape((height, width, 4))
    
    # Flip vertically to match image coordinates
    pixels = np.flipud(pixels)

    # Extract white pixels (objects) from the mask
    object_mask = (pixels[:, :, 0] > 0.5).astype(np.uint8)
    
    # Find all contours
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {}
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Create a dictionary to store segmentation for each object
    object_segmentations = {}
    
    # For each visible object, try to match it with a contour
    visible_objects = [obj for obj in objects if obj.name != base_name]
    
    # Sort visible objects by their 2D bounding box area (largest first)
    def get_2d_area(obj):
        bbox = get_object_bounds_2d(obj, camera, scene)
        return bbox[2] * bbox[3] if bbox else 0
    
    visible_objects.sort(key=get_2d_area, reverse=True)
    
    # Match objects to contours based on size and position
    used_contours = set()
    
    for obj in visible_objects:
        bbox = get_object_bounds_2d(obj, camera, scene)
        if bbox is None:
            continue
            
        obj_center_x = bbox[0] + bbox[2] / 2
        obj_center_y = bbox[1] + bbox[3] / 2
        
        best_contour = None
        best_score = float('inf')
        
        for i, contour in enumerate(contours):
            if i in used_contours:
                continue
                
            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            contour_center_x = M["m10"] / M["m00"]
            contour_center_y = M["m01"] / M["m00"]
            
            # Calculate distance between centers
            distance = math.sqrt((obj_center_x - contour_center_x)**2 + 
                               (obj_center_y - contour_center_y)**2)
            
            # Prefer contours that are close to the object's projected center
            if distance < best_score:
                best_score = distance
                best_contour = (i, contour)
        
        if best_contour is not None:
            contour_idx, contour = best_contour
            used_contours.add(contour_idx)
            
            # Store segmentation for this object
            segmentation = contour.flatten().tolist()
            area = cv2.contourArea(contour)
            
            object_segmentations[obj.name] = {
                'segmentation': [segmentation],
                'area': area
            }
    
    return object_segmentations

def create_manual_mask(output_path, image_name, scene, objects):
    """Create object index mask by rendering objects with materials"""
    # Store original materials and visibility
    original_materials = {}
    original_visibility = {}
    
    # DEBUG: Print which objects we're processing
    print(f"DEBUG: create_manual_mask received {len(objects)} objects:")
    for obj in objects:
        print(f"  - {obj.name} (type: {obj.type})")
    
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
    
    # ENHANCED: Hide ALL objects first, then selectively show only target objects
    all_scene_objects = [obj for obj in scene.objects if obj.type == 'MESH']
    
    # Store original visibility for ALL mesh objects
    for obj in all_scene_objects:
        original_visibility[obj] = obj.hide_render
        obj.hide_render = True  # Hide everything initially
    
    print(f"DEBUG: Hidden {len(all_scene_objects)} total mesh objects")
    
    # Only show and apply white material to target objects
    target_object_names = [obj.name for obj in objects if obj.name != base_name]
    print(f"DEBUG: Target objects for white material: {target_object_names}")
    
    for obj in objects:
        if obj.name != base_name:  # Skip base object
            # Store original materials
            original_materials[obj] = [slot.material for slot in obj.material_slots]
            
            # Clear materials and apply white
            obj.data.materials.clear()
            obj.data.materials.append(white_material)
            
            # Make sure object is visible in render
            obj.hide_render = False
            print(f"DEBUG: Applied white material to {obj.name}, hide_render = {obj.hide_render}")
    
    # Render mask with proper filename
    mask_filepath = os.path.join(output_path, f"{image_name}_mask")
    scene.render.filepath = mask_filepath
    bpy.ops.render.render(write_still=True)
    
    # Restore original materials
    for obj, materials in original_materials.items():
        obj.data.materials.clear()
        for mat in materials:
            obj.data.materials.append(mat)
    
    # Restore original visibility for ALL objects
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

def render_image_and_mask(output_path, image_name, visible_objects, render_mask=True):
    """Render the main image and optionally the object index mask"""
    scene = bpy.context.scene
    
    # Render main image
    bpy.ops.render.render(write_still=True)
    
    # If we don't need the mask, return None
    if not render_mask:
        return None
    
    # Disable compositor to avoid frame number suffixes
    scene.use_nodes = False
    
    # FIXED: Only pass the visible objects for mask rendering
    # This ensures only properly visible objects appear in the mask
    mask_image = create_manual_mask(output_path, image_name, scene, visible_objects)
    
    return mask_image

# IMPROVED: Better multi-category annotation creation
def create_coco_annotation(objects, camera, scene, mask_image, image_info, category_map):
    """Create COCO format annotations with improved multi-category support"""
    annotations = []

    annotation_id = 1
    
    # Get segmentation data for all objects
    object_segmentations = get_object_mask_per_category(mask_image, objects, camera, scene)
    
    print(f"Using category mapping: {category_map}")
    
    # Create annotations for each object
    for obj in objects:
        # Skip base object
        if obj.name == base_name:
            continue
            
        # Skip objects below table threshold
        zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
        if max(zs) < table_z_threshold:
            continue
        
        category_name = extract_category_name(obj.name)
        
        # Get 2D bounding box
        bbox = get_object_bounds_2d(obj, camera, scene)
        if bbox is None:
            print(f"No 2D bbox for object: {obj.name}")
            continue
        
        # Get segmentation mask for this specific object
        if obj.name not in object_segmentations:
            print(f"No segmentation found for object: {obj.name}")
            continue
            
        seg_data = object_segmentations[obj.name]
        segmentation = seg_data['segmentation']
        area = seg_data['area']
        
        if not segmentation or area == 0:
            print(f"Invalid segmentation for object: {obj.name}")
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
        print(f"Created annotation for {obj.name} (category: {category_name})")
        annotation_id += 1
    
    return annotations

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

# IMPROVED: Better object placement for multiple categories
def randomize_objects(objects, centers):
    """Randomize object positions and rotations with category-aware placement"""
    # Group objects by category
    category_objects = {}
    for obj in objects:
        if obj.name == base_name:
            continue
            
        category = extract_category_name(obj.name)
        if category not in category_objects:
            category_objects[category] = []
        category_objects[category].append(obj)
    
    print(f"Randomizing objects by category: {list(category_objects.keys())}")
    
    # Distribute objects across clusters, mixing categories
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
        
        # Add some randomness to avoid perfect clustering
        cluster_spread = 0.15  # Increase spread for better separation
        
        x = random.uniform(max(-base_width/2 + hx + margin, cx - cluster_spread),
                          min(base_width/2 - hx - margin, cx + cluster_spread))
        y = random.uniform(max(-base_length/2 + hy + margin, cy - cluster_spread),
                          min(base_length/2 - hy - margin, cy + cluster_spread))
        
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
    # Z-height test
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
    
    print(f"Found {len(original_objects)} objects to process:")
    for obj in original_objects:
        category = extract_category_name(obj.name)
        print(f"  - {obj.name} (category: {category})")
    
    # Create duplicates if needed
    dups_global = []
    for o in list(original_objects):
        for i in range(duplicate):
            c = o.copy()
            c.data = o.data.copy()
            # Ensure unique naming for duplicates
            c.name = f"{o.name}_dup_{i+1}"
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
            cls = extract_category_name(obj.name)
            if cls not in cat_map:
                cat_map[cls] = len(cat_map) + 1
    
    print(f"Category mapping: {cat_map}")
    
    # Initialize COCO dataset structure
    coco_dataset = {
        "info": {
            "description": "Blender Generated Multi-Category Dataset",
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
    
    # Generate images
    for img_id in range(num_images):
        print(f"\n[Image {img_id+1}/{num_images}]")
        
        # Generate cluster centers and randomize objects
        centers = generate_cluster_centers(num_clusters)
        randomize_objects(original_objects, centers)
        
        # Setup physics and run simulation
        setup_physics(original_objects)
        run_physics_simulation()
        
        # Skip rendering for the first image (physics warmup)
        if img_id == 0:
            print("Skipping first image (physics warmup)")
            for obj in original_objects:
                if obj.name in original_transforms:
                    loc, rot = original_transforms[obj.name]
                    obj.location, obj.rotation_euler = loc, rot
            continue
        
        # Adjust image numbering
        actual_img_id = img_id - 1
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
            for obj in original_objects:
                if obj.name in original_transforms:
                    loc, rot = original_transforms[obj.name]
                    obj.location, obj.rotation_euler = loc, rot
            continue

        # Use the properly visible objects for annotation
        visible_objects = properly_visible
        assign_object_indices(visible_objects)
        
        print(f"Processing {len(visible_objects)} visible objects:")
        for obj in visible_objects:
            category = extract_category_name(obj.name)
            print(f"  - {obj.name} (category: {category})")

        # Render image and mask
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
        
        # Create annotations
        annotations = create_coco_annotation(
            visible_objects, camera, scene, mask_image, image_info, cat_map
        )
        
        # Update annotation IDs
        for ann in annotations:
            ann["id"] = annotation_id_counter
            annotation_id_counter += 1
        
        all_annotations.extend(annotations)
        
        print(f"Generated {len(annotations)} annotations for image {actual_img_id}")
        
        # Show category breakdown
        category_counts = {}
        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = next(cat["name"] for cat in coco_dataset["categories"] if cat["id"] == cat_id)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        if category_counts:
            print(f"Category breakdown: {category_counts}")
        
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
    
    # Final statistics
    final_category_counts = {}
    for ann in all_annotations:
        cat_id = ann["category_id"]
        cat_name = next(cat["name"] for cat in coco_dataset["categories"] if cat["id"] == cat_id)
        final_category_counts[cat_name] = final_category_counts.get(cat_name, 0) + 1
    
    print(f"\n✓ Dataset Generation Complete!")
    print(f"Total annotations: {len(all_annotations)} across {len(coco_dataset['images'])} images")
    print(f"Final category distribution: {final_category_counts}")
    print(f"Dataset saved to: {json_path}")
    
    # Cleanup duplicates
    for dup in dups_global:
        bpy.data.objects.remove(dup, do_unlink=True)
    
    return coco_dataset

# Example usage
if __name__ == "__main__":
    export_coco_dataset_with_randomization()
    print("✓ Done!")