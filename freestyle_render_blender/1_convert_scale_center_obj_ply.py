#!/usr/bin/env python3
"""
Clean OBJ → PLY converter and centering tool
- Processes all OBJ files in current folder
- Outputs: only final centered+scaled PLY files with same base name
"""

import numpy as np
import trimesh
from pathlib import Path
import sys

def create_high_quality_mesh(obj_file, target_vertices=32000):
    """Load and densify OBJ mesh into high-quality mesh"""
    print(f"Converting {obj_file} to high-quality mesh...")

    mesh = trimesh.load(obj_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    subdivided_mesh = mesh.copy()
    subdivision_count = 0
    while len(subdivided_mesh.vertices) < target_vertices and subdivision_count < 8:
        try:
            subdivided_mesh = subdivided_mesh.subdivide()
            subdivision_count += 1
        except Exception:
            break

    smooth_mesh = mesh.copy()
    try:
        smooth_mesh = smooth_mesh.smoothed()
        while len(smooth_mesh.vertices) < target_vertices//2 and len(smooth_mesh.vertices) < 50000:
            smooth_mesh = smooth_mesh.subdivide()
            if len(smooth_mesh.vertices) > target_vertices:
                break
    except Exception:
        smooth_mesh = subdivided_mesh.copy()

    final_mesh = subdivided_mesh if len(subdivided_mesh.vertices) > len(smooth_mesh.vertices) else smooth_mesh

    target_size = 70  # mm
    current_max = max(final_mesh.extents)
    if current_max > 0:
        scale_factor = target_size / current_max
        final_mesh.vertices *= scale_factor

    final_mesh.vertices -= final_mesh.centroid
    final_mesh.update_faces(final_mesh.unique_faces())
    final_mesh.update_faces(final_mesh.nondegenerate_faces())
    final_mesh.fix_normals()

    # Add colors
    base_color = 120
    color_variation = np.random.randint(-20, 20, len(final_mesh.vertices))
    vertex_colors = np.zeros((len(final_mesh.vertices), 4), dtype=np.uint8)
    vertex_colors[:, 0] = np.clip(base_color + color_variation, 102, 255)
    vertex_colors[:, 1] = np.clip(base_color + color_variation, 102, 255)
    vertex_colors[:, 2] = np.clip(base_color + color_variation, 102, 255)
    vertex_colors[:, 3] = 255
    final_mesh.visual.vertex_colors = vertex_colors
    final_mesh.visual.face_colors = None

    return final_mesh

def center_and_scale_and_export(obj_file):
    """Full pipeline for one OBJ file"""
    high_quality_mesh = create_high_quality_mesh(obj_file, target_vertices=25000)

    # Final global scaling to 0.1m (10cm)
    scale = 0.1 / high_quality_mesh.extents.max()
    high_quality_mesh.apply_scale(scale)

    output_file = Path(obj_file).with_suffix('.ply')
    high_quality_mesh.export(output_file)
    print(f"Exported final PLY: {output_file}")

if __name__ == "__main__":
    obj_files = list(Path('.').glob('*.obj'))
    if not obj_files:
        print("No OBJ files found in current directory.")
        sys.exit(1)

    for obj_path in obj_files:
        print(f"\nProcessing {obj_path}")
        center_and_scale_and_export(str(obj_path))

