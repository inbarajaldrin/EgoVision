#!/usr/bin/env python3
"""
centre_origin.py
────────────────
Translate every supplied PLY (triangle-mesh *or* point cloud) so that its
centroid lies at (0,0,0).  Meshes stay meshes; clouds stay clouds.

Usage
-----
python3 centre_origin.py scan.ply                      # → scan_centered.ply
python3 centre_origin.py scan.ply -o centred_scan.ply  # explicit output name
python3 centre_origin.py scan1.ply scan2.ply           # multiple inputs
"""

import argparse, pathlib as pl, sys
import numpy as np
import open3d as o3d


def centre(path: pl.Path, out_path: pl.Path | None):
    """Load ⇒ compute centroid ⇒ translate ⇒ write back (same type)."""
    geom = o3d.io.read_triangle_mesh(str(path))
    is_mesh = len(geom.triangles) > 0

    if not is_mesh:                                   # plain point cloud PLY
        geom = o3d.io.read_point_cloud(str(path))
        if len(geom.points) == 0:
            print(f"⚠️  {path.name}: no points – skipped.")
            return
        centroid = np.asarray(geom.points).mean(axis=0)
        geom.translate(-centroid)
        out = out_path or path.with_name(path.stem + "_centered.ply")
        o3d.io.write_point_cloud(str(out), geom)
    else:                                             # triangle mesh
        if len(geom.vertices) == 0:
            print(f"⚠️  {path.name}: empty mesh – skipped.")
            return
        centroid = np.asarray(geom.vertices).mean(axis=0)
        geom.translate(-centroid)
        out = out_path or path.with_name(path.stem + "_centered.ply")
        o3d.io.write_triangle_mesh(str(out), geom, write_ascii=True)

    print(f"✓ {path.name} centred → {out.name}")


def main():
    ap = argparse.ArgumentParser(description="Re-centre PLY point clouds or meshes.")
    ap.add_argument("plys", nargs="+", help="input PLY files")
    ap.add_argument("-o", "--output",
                    help="output path (allowed only if one input file)")
    args = ap.parse_args()

    if args.output and len(args.plys) != 1:
        sys.exit("Error: --output may be used only with a single input file.")

    for p in args.plys:
        centre(pl.Path(p), pl.Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
