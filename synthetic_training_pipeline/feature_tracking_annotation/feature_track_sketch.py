#!/usr/bin/env python3
"""
Robust 2-D feature matching between sketch renders and manual images.
Rank by RANSAC-verified inliers (SIFT ➜ FLANN ➜ ratio test ➜ homography).
"""

import cv2
import numpy as np
import os, argparse
from pathlib import Path

# ────────────── PARAMETERS ────────────────────────────────────────────────
EDGE_LOW, EDGE_HIGH =  50, 150      # Canny thresholds
RATIO_TEST           = 0.75         # Lowe ratio
MIN_INLIERS          = 4            # Ignore candidates below this
MAX_FEATURES         = 4000         # per image

# ────────────── FEATURE DETECTOR CHOICE ───────────────────────────────────
try:
    detector = cv2.SIFT_create(MAX_FEATURES)          # needs opencv-contrib
    norm      = cv2.NORM_L2
except AttributeError:
    detector = cv2.AKAZE_create()                     # fallback (binary)
    norm      = cv2.NORM_HAMMING

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                              dict(checks=50)) if norm == cv2.NORM_L2 \
       else cv2.BFMatcher(norm)

# ────────────── HELPERS ───────────────────────────────────────────────────
def edge_image(img):
    """Convert to edge map for more stable matching on line drawings."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, EDGE_LOW, EDGE_HIGH)
    return edges

def load_folder(path):
    imgs = []
    for p in sorted(Path(path).iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            imgs.append((p.name, im))
    return imgs

def sift_features(img):
    kp, des = detector.detectAndCompute(img, None)
    return kp or [], des

def match(desA, desB):
    if desA is None or desB is None: return []
    if isinstance(flann, cv2.FlannBasedMatcher) and desA.dtype != np.float32:
        desA, desB = desA.astype(np.float32), desB.astype(np.float32)
    pairs = flann.knnMatch(desA, desB, k=2)
    good  = [m for m,n in pairs if m.distance < RATIO_TEST * n.distance]
    return good

def geom_verify(kpA, kpB, matches):
    if len(matches) < 4: return 0  # RANSAC needs ≥4
    src = np.float32([kpA[m.queryIdx].pt for m in matches])
    dst = np.float32([kpB[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers

# ────────────── MAIN LOGIC ────────────────────────────────────────────────
def main(sketch_dir, manual_dir):
    sketches = load_folder(sketch_dir)
    manuals  = load_folder(manual_dir)
    if not sketches or not manuals:
        print("No images found. Check paths."); return

    # Pre-extract manual features once
    manual_feats = []
    for name, img in manuals:
        edges = edge_image(img)
        kp, des = sift_features(edges)
        manual_feats.append((name, kp, des))

    best = dict(score=0)

    for s_name, s_img in sketches:
        s_edges      = edge_image(s_img)
        s_kp, s_des  = sift_features(s_edges)

        for m_name, m_kp, m_des in manual_feats:
            good = match(s_des, m_des)
            inliers = geom_verify(s_kp, m_kp, good)
            if inliers > best.get("score", 0):
                best = dict(score=inliers,
                            sketch=s_name, manual=m_name,
                            matches=good, kpA=s_kp, kpB=m_kp)

    if best["score"] >= MIN_INLIERS:
        print("Best sketch     :", best["sketch"])
        print("Matched manual  :", best["manual"])
        print("Inliers (score) :", best["score"])
        # Optional: draw & save visualisation
        vis = cv2.drawMatches(
            cv2.imread(os.path.join(sketch_dir, best["sketch"])), best["kpA"],
            cv2.imread(os.path.join(manual_dir,  best["manual"])),  best["kpB"],
            best["matches"], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        out = "best_match_vis.png"
        cv2.imwrite(out, vis)
        print(f"Visualised matches ➜ {out}")
    else:
        print("No reliable match (inlier count too low).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch_dir")
    ap.add_argument("manual_dir")
    main(**vars(ap.parse_args()))
