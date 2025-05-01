#!/usr/bin/env python3
"""
compare_planes_debug.py
────────────────────────
Compare planes from two PLY files by normal alignment, centroid distance, and area.

Usage:
    python3 compare_planes_debug.py file1.ply file2.ply --dist_thresh 0.0005 --min_points 25
"""

import argparse
import open3d as o3d
import numpy as np

def estimate_area(pcd):
    try:
        hull, _ = pcd.compute_convex_hull()
        return hull.get_surface_area()
    except:
        return 0.0

def detect_planes(pcd, distance_thresh=0.001, min_points=30, max_planes=5, label=""):
    planes = []
    remaining = pcd.voxel_down_sample(0.001)
    for i in range(max_planes):
        if len(remaining.points) < 3:
            break
        try:
            model, inliers = remaining.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=3,
                num_iterations=1000)
        except:
            break
        if len(inliers) < min_points:
            break
        plane = remaining.select_by_index(inliers)
        normal = model[:3] / np.linalg.norm(model[:3])
        centroid = np.mean(np.asarray(plane.points), axis=0)
        area = estimate_area(plane)
        print(f"[{label}] Plane {i}: Area = {area:.6f} m²")
        planes.append({'normal': normal, 'centroid': centroid, 'area': area, 'cloud': plane})
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

def compare_planes(planes1, planes2, angle_thresh=15, dist_thresh=0.01, area_thresh=0.001):
    matches = []
    for i, p1 in enumerate(planes1):
        for j, p2 in enumerate(planes2):
            angle = angle_between(p1['normal'], p2['normal'])
            dist = np.linalg.norm(p1['centroid'] - p2['centroid'])
            area_diff = abs(p1['area'] - p2['area'])
            if angle < angle_thresh and dist < dist_thresh and area_diff < area_thresh:
                matches.append({
                    'plane1': i, 'plane2': j,
                    'angle': angle, 'distance': dist,
                    'area1': p1['area'], 'area2': p2['area'],
                    'area_diff': area_diff
                })
    return sorted(matches, key=lambda x: (x['angle'], x['distance'], x['area_diff']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply1", help="First PLY file")
    parser.add_argument("ply2", help="Second PLY file")
    parser.add_argument("--dist_thresh", type=float, default=0.001)
    parser.add_argument("--min_points", type=int, default=30)
    parser.add_argument("--max_planes", type=int, default=5)
    parser.add_argument("--angle_thresh", type=float, default=15)
    parser.add_argument("--centroid_thresh", type=float, default=0.01)
    parser.add_argument("--area_thresh", type=float, default=0.001)
    args = parser.parse_args()

    print(f"🔍 Loading {args.ply1} and {args.ply2}")
    pcd1 = o3d.io.read_point_cloud(args.ply1)
    pcd2 = o3d.io.read_point_cloud(args.ply2)

    planes1 = detect_planes(pcd1, args.dist_thresh, args.min_points, args.max_planes, label="File 1")
    planes2 = detect_planes(pcd2, args.dist_thresh, args.min_points, args.max_planes, label="File 2")

    print(f"[✓] Detected {len(planes1)} planes in file 1")
    print(f"[✓] Detected {len(planes2)} planes in file 2")

    matches = compare_planes(planes1, planes2, args.angle_thresh, args.centroid_thresh, args.area_thresh)
    print(f"\n🔗 Found {len(matches)} plane match candidates:")
    for m in matches:
        print(f"  Plane {m['plane1']} ↔ Plane {m['plane2']}: "
              f"Angle={m['angle']:.2f}°, "
              f"CentroidDist={m['distance']:.4f} m, "
              f"Area1={m['area1']:.6f}, Area2={m['area2']:.6f}, "
              f"ΔArea={m['area_diff']:.6f}")

if __name__ == "__main__":
    main()
