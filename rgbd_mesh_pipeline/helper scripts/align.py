#!/usr/bin/env python3
"""
robust_multi_align.py
─────────────────────
Accurate, sequential alignment + merging of **N** PLY / mesh files
(tuned for symmetric / flipped Lego-type parts).

Pipeline
1.  Down-sample, orient normals, compute FPFH.
2.  For every *new* cloud:
      • try candidate 180 ° flips (no-flip, X, Y, Z)  
      • TEASER++ global registration on *mutual* FPFH matches  
      • 2-stage point-to-plane ICP refinement (coarse → fine)  
      • pick the transform with best ICP fitness
3.  Transform the new cloud, add it to the merged model, voxel-merge.
4.  Save intermediate PLYs + 4×4 transforms for debugging.

Requirements
- open3d >= 0.18
- teaserpp_python ( pip install teaserpp-python )
- numpy, scipy, scikit-learn
"""

import argparse, copy, itertools, os
from pathlib import Path

import numpy as np
import open3d as o3d
import teaserpp_python
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors


# ────────────────────────── defaults ──────────────────────────
VOXEL               = 0.005     # m
NORMAL_RADIUS_MULT  = 2
FPFH_RADIUS_MULT    = 5
TEASER_NOISE        = 0.005
ICP_COARSE_MULT     = 5         # * voxel size
ICP_FINE_MULT       = 1
FLIP_AXES           = [(0, 0, 0),
                       (np.pi, 0, 0),
                       (0, np.pi, 0),
                       (0, 0, np.pi)]
MIN_CORR            = 20
RANSAC_ITERS        = 1000
SAVE_STEPS          = True      # dump merged_up_to_i.ply


# ────────────────────────── helpers ──────────────────────────
def load_points(path: Path, n_sample=None) -> o3d.geometry.PointCloud:
    """Read PLY / mesh.  If triangle mesh, sample uniformly."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles):                       # it *is* a mesh
        if n_sample is None:
            n_sample = 100000
        return mesh.sample_points_uniformly(n_sample)
    return o3d.io.read_point_cloud(str(path))     # already point cloud


def preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
    """Voxel-downsample, orient normals, compute FPFH features."""
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * NORMAL_RADIUS_MULT, max_nn=30))
    down.orient_normals_towards_camera_location()  # consistent sign
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * FPFH_RADIUS_MULT, max_nn=100))
    return down, fpfh


def mutual_correspondences(src_fpfh, tgt_fpfh, dist_thresh=0.9):
    """Return Nx2 array of symmetric-NN indices that pass distance check."""
    NN_tgt = NearestNeighbors(n_neighbors=1).fit(tgt_fpfh.data.T)
    d_st, idx_st = NN_tgt.kneighbors(src_fpfh.data.T)
    NN_src = NearestNeighbors(n_neighbors=1).fit(src_fpfh.data.T)
    d_ts, idx_ts = NN_src.kneighbors(tgt_fpfh.data.T)

    keep = (np.arange(len(idx_st)) ==
            idx_ts[idx_st.flatten()].flatten()) & (d_st.flatten() < dist_thresh)
    return np.vstack([np.arange(len(idx_st))[keep],
                      idx_st.flatten()[keep]]).T


def teaser_transform(src_down, tgt_down, corr_idx, noise):
    """Solve rigid transform with TEASER++ given correspondence indices."""
    src = np.asarray(src_down.points)[corr_idx[:, 0]].T  # 3×N
    tgt = np.asarray(tgt_down.points)[corr_idx[:, 1]].T
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = noise
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = \
        teaserpp_python.RotationEstimationAlgorithm.GNC_TLS
    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src, tgt)
    sol = solver.getSolution()

    T = np.eye(4)
    T[:3, :3] = sol.rotation
    T[:3, 3] = sol.translation
    return T


def two_stage_icp(src_full, tgt_full, init, voxel):
    """Coarse → fine point-to-plane ICP."""
    # Coarse
    icp1 = o3d.pipelines.registration.registration_icp(
        src_full, tgt_full, max_correspondence_distance=voxel * ICP_COARSE_MULT,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    # Fine
    icp2 = o3d.pipelines.registration.registration_icp(
        src_full, tgt_full, max_correspondence_distance=voxel * ICP_FINE_MULT,
        init=icp1.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    return icp2.transformation, icp2.fitness, icp2.inlier_rmse


# ────────────────────── pairwise alignment ───────────────────
def align_pair(src_full, src_down, src_feat,
               tgt_full, tgt_down, tgt_feat, voxel):
    """Try flips, return best rigid transform & fitness."""
    best_T = np.eye(4)
    best_fit = -1
    best_rmse = np.inf

    for flip_idx, (rx, ry, rz) in enumerate(FLIP_AXES):
        src_flip = copy.deepcopy(src_down)
        R_flip = R.from_euler('xyz', (rx, ry, rz)).as_matrix()
        src_flip.rotate(R_flip, center=(0.0, 0.0, 0.0))

        # recompute FPFH for flipped (cheaper than full recompute → use existing normals)
        fl_feat = o3d.pipelines.registration.compute_fpfh_feature(
            src_flip, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * FPFH_RADIUS_MULT, max_nn=100))
        corr = mutual_correspondences(fl_feat, tgt_feat)
        if corr.shape[0] < MIN_CORR:
            continue
        try:
            T_teaser = teaser_transform(src_flip, tgt_down, corr, TEASER_NOISE)
        except RuntimeError:
            # TEASER may fail for ill-posed sets
            continue

        # Need full-resolution copy with the same flip for ICP
        src_flip_full = copy.deepcopy(src_full)
        src_flip_full.rotate(R_flip, center=(0.0, 0.0, 0.0))
        T_icp, fit, rmse = two_stage_icp(src_flip_full, tgt_full, T_teaser, voxel)

        if fit > best_fit or (fit == best_fit and rmse < best_rmse):
            best_fit, best_rmse, best_T = fit, rmse, T_icp
            # Compose with flip rotation wrt original source
            T_flip = np.eye(4)
            T_flip[:3, :3] = R_flip
            best_T = T_icp @ T_flip

    return best_T, best_fit, best_rmse


# ──────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Accurate multi-PLY alignment / merging")
    ap.add_argument("plys", nargs="+", help="input PLY / mesh files (≥2)")
    ap.add_argument("-o", "--output", required=True, help="merged output PLY")
    ap.add_argument("--voxel", type=float, default=VOXEL, help="voxel size [m]")
    ap.add_argument("--nosave_steps", action="store_true", help="skip intermediate PLY dumps")
    args = ap.parse_args()

    voxel = args.voxel
    save_steps = not args.nosave_steps
    out_path = Path(args.output)

    # Load & pre-process all clouds once
    full_clouds, downs, feats = [], [], []
    for p in args.plys:
        pcd_full = load_points(Path(p))
        pcd_down, fpfh = preprocess(pcd_full, voxel)
        full_clouds.append(pcd_full)
        downs.append(pcd_down)
        feats.append(fpfh)

    # Start merged model with first cloud
    merged_full = copy.deepcopy(full_clouds[0])
    merged_full = merged_full.voxel_down_sample(voxel)  # keep res uniform
    merged_down, merged_feat = preprocess(merged_full, voxel)

    transforms = [np.eye(4)]
    print(f"▶ Using {args.plys[0]} as global reference")

    # Sequentially align each remaining cloud
    for idx in range(1, len(full_clouds)):
        print(f"\n=== Align cloud {idx+1}/{len(full_clouds)}: {args.plys[idx]} ===")

        T, fit, rmse = align_pair(full_clouds[idx], downs[idx], feats[idx],
                                  merged_full, merged_down, merged_feat,
                                  voxel)
        print(f"  ✓ best fitness={fit:.3f}, rmse={rmse:.4f}")
        transforms.append(T)

        # Apply transform & merge
        new_cloud = copy.deepcopy(full_clouds[idx])
        new_cloud.transform(T)
        merged_full += new_cloud
        merged_full = merged_full.voxel_down_sample(voxel)  # fuse & control size

        # Update features for next iteration
        merged_down, merged_feat = preprocess(merged_full, voxel)

        if save_steps:
            step_name = out_path.parent / f"merged_up_to_{idx}.ply"
            o3d.io.write_point_cloud(str(step_name), merged_full)
            print(f"  ⤵ saved {step_name}")

        # print transformation matrix
        print("  Transform:\n", np.array_str(T, precision=4))

    # ── final save ──
    o3d.io.write_point_cloud(str(out_path), merged_full)
    print(f"\n✅ Merged cloud written → {out_path}")

    # save transformation matrices
    for i, T in enumerate(transforms):
        np.savetxt(out_path.parent / f"transform_{i}.txt", T, fmt="%.6f")


if __name__ == "__main__":
    main()
