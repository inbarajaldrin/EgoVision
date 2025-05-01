#!/usr/bin/env python3
"""
detect_planes_debug.py
──────────────────────
Standalone script to test and visualize RANSAC plane detection on a PLY file.

Usage:
    python3 detect_planes_debug.py your_file.ply

Example:
    python3 detect_planes_debug.py processed_ply/1.ply --dist_thresh 0.0005 --min_points 25
"""

import argparse
import open3d as o3d
import numpy as np

def detect_planes(pcd, distance_thresh=0.001, min_points=30, max_planes=5):
    planes = []
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
    remaining = pcd.voxel_down_sample(0.001)  # Adjust voxel size here
    print(f"[Info] Downsampled to {len(remaining.points)} points")

    for i in range(max_planes):
        if len(remaining.points) < 3:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=3,
                num_iterations=1000)
        except RuntimeError:
            print("[Warning] RANSAC failed at iteration", i)
            break

        if len(inliers) < min_points:
            print(f"[Info] Skipping plane {i} — too few inliers: {len(inliers)}")
            break

        plane = remaining.select_by_index(inliers)
        plane.paint_uniform_color(colors[i % len(colors)])
        planes.append(plane)
        print(f"[✓] Plane {i}: {len(inliers)} inliers")

        remaining = remaining.select_by_index(inliers, invert=True)

    return planes, remaining

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to PLY file")
    parser.add_argument("--dist_thresh", type=float, default=0.001, help="RANSAC distance threshold")
    parser.add_argument("--min_points", type=int, default=30, help="Min inliers to accept plane")
    parser.add_argument("--max_planes", type=int, default=5, help="Max number of planes to find")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"[Info] Loaded point cloud with {len(pcd.points)} points")

    planes, remaining = detect_planes(
        pcd,
        distance_thresh=args.dist_thresh,
        min_points=args.min_points,
        max_planes=args.max_planes
    )

    print(f"[Done] Detected {len(planes)} planes")
    remaining.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries(planes + [remaining])

if __name__ == "__main__":
    main()
