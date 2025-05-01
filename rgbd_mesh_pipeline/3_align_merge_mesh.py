#!/usr/bin/env python3
"""
align5.py – TEASER++ + ICP alignment of multiple PLY meshes **and** full-resolution mesh fusion.

Usage
-----
python3 align5.py scan1.ply scan2.ply [scan3.ply …] -o merged_mesh.ply

Produces:
  * merged_mesh.ply  – full-resolution union of the input meshes,
                       each transformed by the TEASER++/ICP result.
  * merged_mesh.txt  – newline-delimited 4×4 matrices for each scan.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import teaserpp_python
from sklearn.neighbors import NearestNeighbors
import copy

def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"No triangles found in mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh

def mesh_to_pcd(mesh: o3d.geometry.TriangleMesh, n: int):
    return mesh.sample_points_uniformly(number_of_points=n)

def preprocess_pcd(pcd, voxel: float):
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100),
    )
    return pcd, np.asarray(fpfh.data)

def mutual_correspondences(f_src: np.ndarray, f_dst: np.ndarray):
    knn_dst = NearestNeighbors(n_neighbors=1).fit(f_dst.T)
    knn_src = NearestNeighbors(n_neighbors=1).fit(f_src.T)
    idx_dst = knn_dst.kneighbors(f_src.T, return_distance=False).flatten()
    idx_src = knn_src.kneighbors(f_dst.T, return_distance=False).flatten()
    keep = np.arange(len(idx_dst)) == idx_src[idx_dst]
    return np.column_stack((np.arange(len(idx_dst))[keep], idx_dst[keep]))

def teaser_pose(src_pts: np.ndarray, dst_pts: np.ndarray, corr: np.ndarray):
    if len(corr) < 3:
        raise RuntimeError("Need ≥3 correspondences for TEASER++")
    solver = teaserpp_python.RobustRegistrationSolver(
        noise_bound=0.01,
        cbar2=1.0,
        estimate_scaling=False,
        rotation_estimation_algorithm=teaserpp_python.RotationEstimationAlgorithm.GNC_TLS,
        rotation_gnc_factor=1.4,
        rotation_max_iterations=1000,
        rotation_cost_threshold=1e-12,
    )
    solver.solve(src_pts[corr[:, 0]].T, dst_pts[corr[:, 1]].T)
    sol = solver.getSolution()
    T = np.eye(4)
    T[:3, :3] = sol.rotation
    T[:3, 3] = sol.translation
    return T

def align_pair(src_mesh, dst_mesh, voxel, n_sample):
    # Sample each mesh to a point cloud for alignment
    pcd_src = mesh_to_pcd(src_mesh, n_sample)
    pcd_dst = mesh_to_pcd(dst_mesh, n_sample)

    p_src, f_src = preprocess_pcd(pcd_src, voxel)
    p_dst, f_dst = preprocess_pcd(pcd_dst, voxel)

    corr = mutual_correspondences(f_src, f_dst)
    print(f"  • mutual correspondences: {len(corr)}")
    T0 = teaser_pose(np.asarray(p_src.points), np.asarray(p_dst.points), corr)

    # ICP refinement
    p_src_t = p_src.transform(T0.copy())
    reg = o3d.pipelines.registration.registration_icp(
        p_src_t,
        p_dst,
        max_correspondence_distance=voxel * 2,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    print(f"  • ICP RMSE: {reg.inlier_rmse:.6f}")
    return reg.transformation @ T0

def main():
    parser = argparse.ArgumentParser(description="Align & fuse multiple PLY meshes")
    parser.add_argument("inputs", nargs='+', help="Input PLY mesh files (≥2)")
    parser.add_argument("-o", "--output", required=True, help="Output merged mesh (.ply)")
    parser.add_argument("--voxel", type=float, default=0.003,
                        help="Voxel size for down-sampling during alignment (m)")
    parser.add_argument("--sample", type=int, default=50_000,
                        help="Number of points to sample per mesh")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Require at least two input meshes to fuse.")

    # Load all meshes
    meshes = [load_mesh(str(p)) for p in args.inputs]
    transforms = [np.eye(4)]

    print("📐 Computing pairwise transforms…")
    for i in range(1, len(meshes)):
        print(f"→ aligning scan {i} → scan {i-1}")
        T = align_pair(meshes[i], meshes[i-1], args.voxel, args.sample)
        transforms.append(T)

    # Write out all transforms
    tf_path = Path(args.output).with_suffix(".txt")
    with open(tf_path, 'w') as f:
        for T in transforms:
            for row in T:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("\n")
    print(f"✔ Wrote transforms → {tf_path}")

    # Fuse the **original** high-res meshes
    print("🔗 Fusing original meshes with computed transforms…")
    final_mesh = o3d.geometry.TriangleMesh()              # ← Initialize here!
    for mesh, T in zip(meshes, transforms):
        m = copy.deepcopy(mesh)                           # clone the mesh
        m.transform(T)                                    # in-place transform
        final_mesh += m                                   # accumulate into union

    final_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output), final_mesh)
    print(f"✅ Merged mesh saved → {args.output}")

if __name__ == "__main__":
    main()
